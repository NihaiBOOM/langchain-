from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.documents import Document
from utils.logger_handler import logger
from utils.config_handler import chroma_conf


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.__init__chain()

    def __init__chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def rag_summarize(self, query: str) -> str:
        """
        RAG 检索 + LLM 总结的完整流程。
        
        两条路径（由 chroma.yml 的 hybrid.enabled 控制）：
          路径A (Hybrid): hybrid_search → RRF 融合后的 3 条文档 → 组装 context → LLM 总结
          路径B (纯向量): similarity_search_with_score → score_threshold 过滤 → 取前 k 条 → 组装 context → LLM 总结
        
        共同点：最终都是把检索到的参考资料组装成 context，连同用户 query 一起丢给 LLM，
        让 LLM 基于参考资料总结出专业回答（而不是让 LLM 瞎编）。
        """
        hybrid_conf = chroma_conf.get("hybrid", {})          # 配置文件: chroma.yml → hybrid
        threshold = chroma_conf.get("score_threshold", 0.5)  # 配置文件: chroma.yml → score_threshold  纯向量模式用的相似度阈值
        final_k = chroma_conf["k"]                           # 配置文件: chroma.yml → k  最终取几条给 LLM

        if hybrid_conf.get("enabled", False) and self.vector_store.bm25 is not None:
            # ── 路径A: Hybrid 模式（向量 + BM25 → RRF 融合 → 可选 Reranker 精排）──────
            # hybrid_search 内部流程：
            #   没开 Reranker: 向量召回10条 + BM25召回10条 → RRF融合 → 直接取 Top 3
            #   开了 Reranker: 向量召回10条 + BM25召回10条 → RRF融合取 Top 15 粗候选 → CrossEncoder 精排 → 取真正最精准的 Top 3
            fused_results = self.vector_store.hybrid_search(query)
            # 只取 Document 对象，丢掉分数和 src 标记（LLM 不需要这些）
            context_docs = [doc for doc, score, src in fused_results]
            # 判断分数标签：Reranker 开了 → 分数是 0~1 的相关性分数；没开 → 分数是 RRF 排名融合分数
            score_label = "Reranker" if self.vector_store.reranker is not None else "RRF"
            logger.info(f"[RAG] Hybrid+{score_label} 模式，检索到 {len(context_docs)} 条, query='{query}'")
            # 日志里打印每条的分数和来源，方便调试看检索效果
            for i, (doc, score, src) in enumerate(fused_results):
                preview = doc.page_content[:80].replace("\n", " ")
                source = doc.metadata.get("source", "未知")
                logger.info(f"[RAG] 结果{i+1}: [{source}] {score_label}={score:.4f} src={src} | {preview}...")
        else:
            # ── 路径B: 纯向量模式（Chroma similarity + score_threshold 过滤）──────
            # similarity_search_with_score 取 k*2 条（多拉一些，过滤后可能不够 k 条）
            raw_docs = self.vector_store.similarity_search_with_score(query)
            # score_threshold 过滤：Chroma 返回的是 cosine distance（0~2，越小越相似），大于阈值的扔掉
            scored_docs = [(doc, s) for doc, s in raw_docs if s <= threshold]
            # 截断到 final_k 条（配置文件里的 k=3）
            context_docs = [doc for doc, _ in scored_docs][:final_k]
            logger.info(f"[RAG] 纯向量模式，检索到 {len(context_docs)} 条, query='{query}', 阈值={threshold}")
            for i, (doc, score) in enumerate(scored_docs[:final_k]):
                preview = doc.page_content[:80].replace("\n", " ")
                source = doc.metadata.get("source", "未知")
                logger.info(f"[RAG] 结果{i+1}: [{source}] 距离={score:.4f} | {preview}...")

        # ── 空检索防御 ──────────────────────────────────────────────────────────────
        # 如果过滤后一条都没剩，直接返回友好提示，不调 LLM（LLM 拿到空 context 会瞎编）
        if not context_docs:
            logger.warning("=" * 50)
            logger.warning(f"[RAG 空检索防御] 触发！query='{query}'")
            logger.warning("=" * 50)
            return "【知识库未命中】未在知识库中检索到相关资料，请换一个关键词试试，比如「充电故障」「迷路问题」「清扫效率」等。"

        # ── 组装 context ────────────────────────────────────────────────────────────
        # 把检索到的 3 条文档拼成一个长字符串，每条约 300 字，带编号和元数据（文件名）
        # 为什么要加元数据：让 LLM 知道这条资料来自哪个文件，回答时可以引用出处
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料: {doc.page_content} | 参考元数据: {doc.metadata}\n"

        # ── 调 LLM 总结 ─────────────────────────────────────────────────────────────
        # chain = rag_summarize.txt 的 prompt 模板 → 通义 qwen3-max → 字符串输出
        # 把用户 query 和组装好的 context 一起喂进去，LLM 按 prompt 约束总结出专业回答
        return self.chain.invoke({"input": query, "context": context})


if __name__ == "__main__":
    rag = RagSummarizeService()
    result = rag.rag_summarize("机器人无法充电")
    print(result)