# indexing/ingest.py
"""
PDF文档加载与切分模块
"""
import logging
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import requests
from langchain_community.document_loaders import PyMuPDFLoader, pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.setting import QaConfig

logger = logging.getLogger(__name__)


class PdfIngestor:
    """PDF文档加载与切分"""
    # 文本分隔符
    SEPARATORS = [
        r"\n\n+",  # 段落分隔（PDF 中常见的大段空白）
        r"\n",  # 行分隔
        r"。|！|？|；",  # 中文句末/分句标点（优先按句子切）
        r"\.\s|\!\s|\?\s",  # 英文句末标点（后跟空格，避免切分小数或缩写）
        r"；|;\s",  # 中英文分号
        r"，|,\s",  # 中英文逗号（最后防线，尽量保持语义完整）
        r"\s",  # 空格
        r""  # 空字符（兜底，强制按字符切）
    ]

    def __init__(self, config: QaConfig) -> None:
        """
        初始化PDF加载器
        :param config: 配置信息
        """
        self._config = config

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            is_separator_regex=True,
            separators=self.SEPARATORS
        )

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """
        切分文档为文本块
        :param documents: 原始 Document 列表
        :return: 切分后的 Document 列表
        """

        if not documents:
            logger.warning("PDF文档为空！无法切分")

        try:
            chunks = self._splitter.split_documents(documents)
            logger.debug(f"文档切分完成:{len(documents)}页-->{len(chunks)}文本块")
            return chunks
        except Exception as e:
            raise RuntimeError(f"文档切分失败：{e}")



    def ingest(self, pdf_source: str) -> list[Document]:
        """
        从 PDF 文件路径或 URL 加载并切分文档
        :param pdf_source: PDF文件路径或网络 URL
        :return: 切分后的 Document 列表
        """
        if not pdf_source:
            raise ValueError("pdf_source 不能为空")

        # 按字符串形态判断来源，不做网络探测
        if self._is_url(pdf_source):
            logger.info(f"检测到URL，从网络加载PDF: {pdf_source}")
            return self._ingest_from_url(pdf_source)

        logger.info(f"检测到本地文件，加载PDF: {pdf_source}")
        return self._ingest_from_file(pdf_source)


    def _is_url(self, s: str) -> bool:
        if not isinstance(s, str) or not s.strip():
            return False
        try:
            r = urlparse(s)
            valid = r.scheme in ("http", "https") and bool(r.netloc)
        except ValueError as e:
            return False
        return valid


    def _ingest_from_file(self,file_path:str)->list[Document]:
        """
        从本地文件加载pdf,切分
        :param file_path: 文件路径
        :return: 返回切分后的document列表
        """
        path=Path(file_path)

        #验证文件是否存在
        if not path.exists():
            raise FileNotFoundError(f"pdf文件不存在（也不是合法URL）：{path.resolve()}")
        if not path.is_file():
            raise ValueError(f"{file_path}路径不是文件")

        try:
            loader=PyMuPDFLoader(str(path))
            documents=loader.load()
            return self._split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"PDF加载失败({file_path}):{e}")

    def _ingest_from_url(self,file_url:str)->list[Document]:
        """
        先带超时下载pdf到临时文件，再本地加载、切分
        :param file_url:文件url
        :return: 返回切分后的documents列表
        """
        temp_path: Path | None = None
        try:
            # 下载到配置中指定的临时目录
            temp_dir = Path(self._config.temp_dir)
            suffix = Path(urlparse(file_url).path).suffix or ".pdf"
            temp_path = temp_dir / f"{uuid4().hex}{suffix}"

            logger.info(f"开始下载PDF: {file_url} -> {temp_path}")
            # timeout=(连接超时, 读超时)，避免 URL 不可用时无限等待
            with requests.get(file_url, timeout=(5, 60), stream=True) as resp:
                resp.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        f.write(chunk)

            logger.info(f"下载完成: {file_url} -> {temp_path}")
            return self._ingest_from_file(str(temp_path))
        except requests.RequestException as e:
            logger.error(f"从URL下载pdf文件失败: {file_url} -> {e}")
            raise RuntimeError(f"从URL下载pdf文件失败({file_url}):{e}")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pdf=r""
    url=""
    ingestor=PdfIngestor(QaConfig())
    data=ingestor.ingest(url)
    print(data)
