"""
配置管理模块
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class QaConfig:
    """QA系统配置"""

    #文本拆分
    chunk_size:int=200
    chunk_overlap:int=20

    #向量数据库
    chroma_root_dir:str="./chroma"
    collection_prefix:str="pdf_qa"

    #embedding配置
    embedding_model:str="qwen3.7-text-embedding-flash"

    #临时文件目录
    temp_dir:str="./temp"

    def __post_init__(self):
        """初始化后创建目录"""
        for dir_path in [self.chroma_root_dir,self.temp_dir]:
            Path(dir_path).mkdir(parents=True,exist_ok=True)