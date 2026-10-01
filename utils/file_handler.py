import hashlib
import os
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 获取文件的md5值
def get_file_md5_hex(file_path: str) -> str:

    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]文件{file_path}不是文件")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096
    try:
        with open(file_path, 'rb') as f:# "rb":以二进制模式打开文件
            while chunk := f.read(chunk_size):#分批次每一次读取4096字节
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算]计算文件{file_path}的md5值失败: {e}")
        return None


    # 列出目录下的所有文件(根据允许的文件类型)
def listdir_with_allowed_type(path: str, allowed_types: list[str]) -> list[str]:
    files = []
    if not os.path.exists(path):
        logger.error(f"[listdir_with_allowed_type] 目录{path}不存在")
        return allowed_types

    for f in os.scandir(path):
        if f.name.endswith(allowed_types):#找到符合的结尾的文件
            files.append(os.path.join(path, f.name))#组合成完整路径

    return tuple(files)


def pdf_loader(file_path: str,passwd:str = None) -> Document:
    return PyPDFLoader(file_path,password=passwd).load()

def txt_loader(file_path: str) -> list[Document]:
    return TextLoader(file_path,encoding="utf-8").load()
