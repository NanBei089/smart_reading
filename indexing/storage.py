# indexing/storage.py
"""
PDF文件存储与哈希计算模块
"""
import hashlib


class PdfBytesStore:
    """根据文件内容计算 hash"""

    def compute_hash(self, pdf_bytes) -> str:
        """从字节数据计算MD5哈希值"""
        return hashlib.md5(pdf_bytes).hexdigest()

    def compute_hash_from_file(self, pdf_file) -> str:
        """从文件路径计算MD5哈希值（流式读取文件内容）"""
        hash_md5 = hashlib.md5()
        with open(pdf_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

