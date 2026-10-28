from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os
import jieba
import numpy as np
from rank_bm25 import BM25Okapi


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        self.load_document()

        self.bm25: BM25Okapi | None = None
        self.bm25_docs: list[Document] = []
        self._hybrid_conf = chroma_conf.get("hybrid", {})
        if self._hybrid_conf.get("enabled", False):
            self._init_bm25()

        self.reranker = None
        self._reranker_conf = chroma_conf.get("reranker", {})
        if self._reranker_conf.get("enabled", False):
            self._init_reranker()

    def _tokenize(self, text: str) -> list[str]:
        """用 jieba 做中文分词 → BM25 需要分词后的语料"""
        return [t.strip() for t in jieba.lcut(text) if t.strip()]

    def _init_bm25(self):
        """
        从已加载的向量库中重建 BM25 索引。
        为什么需要这一步：BM25 是基于分词的关键词匹配，必须有 corpus（语料库）才能初始化。
        什么时候调用：__init__ 时（加载完知识库） + load_document 后（增量加载新文档后要重建索引）。
        """
        # 从 Chroma 取所有已存入的文档（Chroma 是权威数据源）
        all_data = self.vector_store.get()
        if not all_data or not all_data.get("documents"):
            logger.warning("[BM25] 向量库为空，暂不初始化 BM25")
            return
        # 把 Chroma 的原始文档转回 Document 对象（带 metadata，比如 source 文件名）
        self.bm25_docs = [
            Document(page_content=doc, metadata=all_data["metadatas"][i] if all_data.get("metadatas") else {})
            for i, doc in enumerate(all_data["documents"])
        ]
        # 对每个文档做 jieba 分词 → BM25Okapi 的入参是 [[分词1, 分词2, ...], [分词1, ...]]
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"[BM25] 初始化完成，共索引 {len(self.bm25_docs)} 条文档")

    def _init_reranker(self):
        """
        初始化 Reranker（百炼云端 API）。
        验证 dashscope.TextReRank 可用，和 Embedding/LLM 共用 DASHSCOPE_API_KEY。
        """
        try:
            import dashscope
            from http import HTTPStatus
            model_name = self._reranker_conf.get("model_name", "qwen3-rerank")
            # 轻量调用验证连通性
            resp = dashscope.TextReRank.call(
                model=model_name,
                query="ping",
                documents=["pong"],
                top_n=1
            )
            if resp.status_code == HTTPStatus.OK:
                self.reranker = model_name   # 存模型名表示已启用
                logger.info(f"[Reranker] 百炼 qwen3-rerank 连通验证通过")
            else:
                logger.warning(f"[Reranker] 百炼 API 返回异常: {resp.status_code} {resp.message}")
                self.reranker = None
        except Exception as e:
            logger.warning(f"[Reranker] 百炼 API 不可用: {e}，Reranker 已禁用")
            self.reranker = None

    def similarity_search_with_score(self, query: str) -> list[tuple]:
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=chroma_conf["k"] * 2,
        )

    def hybrid_search(self, query: str) -> list[tuple[Document, float, str]]:
        """
        Hybrid 检索核心方法：向量语义召回 + BM25 关键词召回 → RRF 融合排名
        返回: [(doc, rrf_score, source), ...]  source = 'hybrid'(两边都召回到) | 'vector'(仅向量) | 'bm25'(仅BM25)
        
        为什么用 Hybrid：
          纯向量靠 embedding 的语义相似度，可能漏掉只含关键词但语义不那么像的文档（召回率低）。
          纯 BM25 靠关键词精确匹配，可能漏掉语义相关但用词不同的文档（精准率低）。
          Hybrid 把两边的结果用 RRF 融合，两边都召回到的文档分数最高 → 既补召回又保精准。
        
        RRF 公式（Reciprocal Rank Fusion）：
          rrf_score = Σ 1/(rrf_k + rank + 1)
          rrf_k 是平滑参数（经验值 60），作用是让排名靠前的文档分高，但差距不会太悬殊。
          为什么用 RRF 而不是加权平均分数：向量的 cosine distance 和 BM25 的分数量纲完全不同，不能直接加，
          RRF 只看排名，不看原始分数，天然解决量纲问题。
        """
        hc = self._hybrid_conf
        vector_top_k = hc.get("vector_top_k", 10)   # 配置文件: chroma.yml → hybrid.vector_top_k  向量召回多少条做粗候选
        bm25_top_k = hc.get("bm25_top_k", 10)        # 配置文件: chroma.yml → hybrid.bm25_top_k    BM25 召回多少条做粗候选
        rrf_k = hc.get("rrf_k", 60)                   # 配置文件: chroma.yml → hybrid.rrf_k         RRF 平滑参数
        final_k = chroma_conf["k"]                    # 配置文件: chroma.yml → k                    最终返回给 LLM 多少条

        # ═══════════════════════════════════════════════════════════════
        # 阶段1: 向量语义召回（Chroma + 通义 embedding）
        # 作用：把 query 转成向量，和向量库做 cosine 距离，取最接近的 vector_top_k 条
        # ═══════════════════════════════════════════════════════════════
        vector_results = self.vector_store.similarity_search_with_score(query=query, k=vector_top_k)
        # 存每个文档的分数和排名 → 后面 RRF 融合要用到排名
        # doc_id 取前 80 字做唯一标识（完整内容当 key 可能太长）
        vector_scores = {}
        vector_rank = {}
        for rank, (doc, score) in enumerate(vector_results):
            doc_id = doc.page_content[:80]
            vector_scores[doc_id] = score
            vector_rank[doc_id] = rank

        # ═══════════════════════════════════════════════════════════════
        # 阶段2: BM25 关键词召回（jieba 分词 + BM25Okapi）
        # 作用：把 query 分词后做 TF-IDF 加权的关键词匹配，取分数最高的 bm25_top_k 条
        # ═══════════════════════════════════════════════════════════════
        bm25_results: list[tuple[Document, float]] = []
        if self.bm25 is not None and self.bm25_docs:
            # 对用户 query 做 jieba 分词 → ["扫地", "机器人", "充电"]
            tokenized_query = self._tokenize(query)
            # BM25 对每个文档算一个分数 → bm25_scores_arr[i] 就是第 i 个文档和 query 的 BM25 分数
            bm25_scores_arr = self.bm25.get_scores(tokenized_query)
            # 按 BM25 分数从高到低排，取前 bm25_top_k 个文档的索引
            top_indices = np.argsort(bm25_scores_arr)[::-1][:bm25_top_k]
            # 把 BM25 分数归一化（除以最大值，让分数范围在 0~1 之间，方便后面调试和日志展示）
            bm25_max = max(bm25_scores_arr) if bm25_scores_arr.max() > 0 else 1.0
            for idx in top_indices:
                if bm25_scores_arr[idx] == 0:    # BM25 分数为 0 说明完全没匹配到关键词，跳过
                    continue
                doc = self.bm25_docs[idx]
                normalized_score = bm25_scores_arr[idx] / bm25_max
                bm25_results.append((doc, normalized_score))

        # 同样存 BM25 的 doc_id → 排名映射
        bm25_scores = {}
        bm25_rank = {}
        for rank, (doc, score) in enumerate(bm25_results):
            doc_id = doc.page_content[:80]
            bm25_scores[doc_id] = score
            bm25_rank[doc_id] = rank

        # ═══════════════════════════════════════════════════════════════
        # 阶段3: RRF 融合排名
        # 作用：把向量和 BM25 两边的排名合并成一个最终分数
        # 直觉：同时在两边排名靠前的 → RRF 分数最高 → 最相关
        # ═══════════════════════════════════════════════════════════════
        # 合并两边所有 doc_id（去重）
        all_ids = set(vector_rank.keys()) | set(bm25_rank.keys())
        rrf_scores: dict[str, float] = {}
        for doc_id in all_ids:
            rrf = 0.0
            # 在向量列表里的排名 → 加一份分
            if doc_id in vector_rank:
                rrf += 1.0 / (rrf_k + vector_rank[doc_id] + 1)
            # 在 BM25 列表里的排名 → 再加一份分（两边都有就加两次！这就是 hybrid 的优势）
            if doc_id in bm25_rank:
                rrf += 1.0 / (rrf_k + bm25_rank[doc_id] + 1)
            rrf_scores[doc_id] = rrf

        # 按 RRF 分数从高到低排
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        # ═══════════════════════════════════════════════════════════════
        # 截断策略：Reranker 开了 → 取 top_n=15 条粗候选（多拉一些给精排选）
        #           Reranker 没开 → 直接取 final_k=3 条（和原来一样）
        # ═══════════════════════════════════════════════════════════════
        rerank_top_n = self._reranker_conf.get("top_n", 15) if self.reranker is not None else final_k
        seen_contents = set()
        fused_results: list[tuple[Document, float, str]] = []
        for doc_id in sorted_ids:
            if doc_id in seen_contents:
                continue
            seen_contents.add(doc_id)
            # 标记来源：hybrid=两边都召回到（最可靠），vector=仅向量，bm25=仅BM25
            source = "hybrid" if doc_id in vector_rank and doc_id in bm25_rank else (
                "vector" if doc_id in vector_rank else "bm25"
            )
            rrf = rrf_scores[doc_id]
            # 从对应的原始列表里取回 Document 对象（带完整 page_content 和 metadata）
            if doc_id in vector_rank:
                for d, s in vector_results:
                    if d.page_content[:80] == doc_id:
                        fused_results.append((d, rrf, source))
                        break
            else:
                for d, s in bm25_results:
                    if d.page_content[:80] == doc_id:
                        fused_results.append((d, rrf, source))
                        break
            # Reranker 开了 → 取 rerank_top_n 条粗候选；没开 → 取 final_k 条直接返回
            if len(fused_results) >= rerank_top_n:
                break

        # ═══════════════════════════════════════════════════════════════
        # 阶段4: Reranker 精排（百炼 qwen3-rerank 云端 API）
        # 把 query + 粗候选文档列表 发给百炼 API → API 返回按相关性排序的结果
        # 为什么比 RRF 准：RRF 只看排名，Reranker 真正理解语义，精准率从 ~75% → ~90%
        # ═══════════════════════════════════════════════════════════════
        if self.reranker is not None and fused_results:
            import dashscope
            from http import HTTPStatus
            # 粗候选文档内容列表（顺序和 fused_results 一致）
            doc_texts = [doc.page_content for doc, rrf, src in fused_results]
            model_name = self.reranker   # 存的是模型名（如 "qwen3-rerank"）
            resp = dashscope.TextReRank.call(
                model=model_name,
                query=query,
                documents=doc_texts,
                top_n=len(fused_results)   # 让 API 返回所有粗候选的分数（后面自己取 Top final_k）
            )
            if resp.status_code == HTTPStatus.OK:
                # API 返回的 results 按分数从高到低排，每个 result 有 index 和 relevance_score
                # index 是粗候选列表里的原始位置（0-based）
                reranked = []
                for r in resp.output.results:
                    orig_doc, orig_rrf, orig_src = fused_results[r.index]
                    score = r.relevance_score
                    logger.info(f"[Reranker粗候选#{r.index}] rrf={orig_rrf:.6f} → rerank={score:.4f} [{orig_src}] {orig_doc.page_content[:50]}")
                    reranked.append((orig_doc, float(score), orig_src))
                # 取精排后的 Top final_k
                fused_results = reranked[:final_k]
                logger.info(f"[Reranker] 精排完成: 粗候选 {len(doc_texts)} 条 → 最终 Top {len(fused_results)} 条")
            else:
                # API 调用失败 → 降级，直接用 RRF 的结果（不精排）
                logger.warning(f"[Reranker] 百炼 API 失败，降级用 RRF 结果: {resp.status_code}")
                fused_results = fused_results[:final_k]

        logger.info(f"[Hybrid] query='{query}' → 向量召回 {len(vector_results)} 条, BM25 召回 {len(bm25_results)} 条, 融合后 {len(fused_results)} 条")
        return fused_results

    def _rebuild_bm25_after_load(self):
        self._init_bm25()

    def load_document(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        loaded_new = False
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue
                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue
                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功")
                loaded_new = True
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败: {str(e)}", exc_info=True)
                continue

        if loaded_new and self._hybrid_conf.get("enabled", False):
            self._rebuild_bm25_after_load()


if __name__ == '__main__':
    vs = VectorStoreService()
    res = vs.hybrid_search("机器人无法充电")
    for doc, score, src in res:
        print(f"[{src}] rrf={score:.4f} | {doc.page_content[:80]}")
        print("-" * 30)