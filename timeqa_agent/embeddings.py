"""
本地Embedding模型实现

使用本地部署的BGE-M3模型进行文本嵌入
"""

from typing import List, Callable, Optional
import numpy as np
from pathlib import Path


def create_local_embed_fn(
    model_path: Optional[str] = None,
    normalize_embeddings: bool = True
) -> Optional[Callable[[List[str]], List[List[float]]]]:
    """
    创建本地BGE-M3嵌入函数
    
    Args:
        model_path: BGE-M3模型路径，默认为'./models/bge-m3/bge-m3'
        normalize_embeddings: 是否归一化嵌入向量
    
    Returns:
        嵌入函数，接受文本列表，返回向量列表；如果无法加载模型则返回None
    """
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        print("警告: 未安装 FlagEmbedding，本地embedding不可用。请运行: pip install -u FlagEmbedding")
        return None
    
    # 默认模型路径 - 使用相对于当前工作目录的路径
    # if model_path is None:
    model_path = "./models/bge-m3/bge-m3"
    
    # 确保路径是绝对路径且格式正确
    model_path_obj = Path(model_path)
    if not model_path_obj.is_absolute():
        model_path_obj = Path.cwd() / model_path
    model_path = str(model_path_obj.resolve()).replace('\\', '/')
    
    try:
        print(f"正在加载本地BGE-M3模型: {model_path}")
        # 使用本地路径时，确保路径存在
        if not Path(model_path).exists():
            print(f"错误: 模型路径不存在: {model_path}")
            return None
        
        # 使用FlagEmbedding库加载模型
        model = BGEM3FlagModel(
            model_path,
            use_fp16=False  # 如果GPU支持，可以设为True以提高速度
        )
        print("模型加载成功")
    except Exception as e:
        print(f"错误: 无法加载本地BGE-M3模型 '{model_path}': {e}")
        return None
    
    def embed_fn(texts: List[str]) -> List[List[float]]:
        """本地模型嵌入函数"""
        if not texts:
            return []
        
        # 使用FlagEmbedding模型生成嵌入向量
        results = model.encode(
            texts,
            batch_size=32,
            max_length=8192,
            return_dense=True,
            return_sparse=False,  # 如果不需要稀疏向量，设为False
            return_colbert_vecs=False  # 如果不需要colbert向量，设为False
        )
        
        # 获取密集向量
        embeddings = results['dense_vecs']
        
        # 如果需要归一化
        if normalize_embeddings:
            # L2归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # 避免除零
            embeddings = embeddings / norms
        
        # 转换为Python列表
        return embeddings.tolist()
    
    return embed_fn