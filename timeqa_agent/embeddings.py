"""
嵌入模型实现

支持多种嵌入模型:
- Contriever (facebook/contriever, facebook/contriever-msmarco)
- DPR (Dense Passage Retrieval)
- BGE-M3 (本地模型)
"""

from typing import List, Callable, Optional, Tuple
import numpy as np
from pathlib import Path
import warnings


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
        # 添加 local_files_only=True 确保只从本地加载，不会尝试从 HuggingFace Hub 下载
        model = BGEM3FlagModel(
            model_path,
            use_fp16=False,  # 如果GPU支持，可以设为True以提高速度
            local_files_only=True  # 强制仅使用本地文件，不访问 HuggingFace Hub
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


def create_contriever_embed_fn(
    model_name: str = "facebook/contriever-msmarco",
    device: str = "cpu",
    normalize: bool = True,
    batch_size: int = 32
) -> Optional[Callable[[List[str]], List[List[float]]]]:
    """
    创建 Contriever 嵌入函数

    Args:
        model_name: HuggingFace 模型名称或本地路径
        device: 设备 ("cpu", "cuda", "cuda:0" 等)
        normalize: 是否归一化嵌入向量
        batch_size: 批处理大小

    Returns:
        嵌入函数，接受文本列表，返回向量列表；如果无法加载模型则返回None
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("警告: 未安装 transformers 或 torch。请运行: pip install transformers torch")
        return None

    try:
        print(f"正在加载 Contriever 模型: {model_name}")

        # 处理本地路径：确保路径是绝对路径且存在
        model_path = Path(model_name)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_name

        # 检查是否为本地路径
        is_local = model_path.exists()

        if is_local:
            # 本地路径：使用绝对路径并转换为正确的格式
            model_path_str = str(model_path.resolve())
            print(f"  检测到本地路径: {model_path_str}")
            tokenizer = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True)
            model = AutoModel.from_pretrained(model_path_str, local_files_only=True)
        else:
            # HuggingFace Hub 路径
            print(f"  从 HuggingFace Hub 加载...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

        # 移动模型到指定设备
        model = model.to(device)
        model.eval()

        print(f"Contriever 模型加载成功 (device: {device})")
    except Exception as e:
        print(f"错误: 无法加载 Contriever 模型 '{model_name}': {e}")
        return None

    def mean_pooling(token_embeddings, attention_mask):
        """均值池化"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_fn(texts: List[str]) -> List[List[float]]:
        """Contriever 嵌入函数"""
        if not texts:
            return []

        all_embeddings = []

        # 批处理
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                # 获取模型输出
                outputs = model(**inputs)

                # 均值池化
                embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])

                # 归一化
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        # 合并所有批次
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # 转换为 Python 列表
        return all_embeddings.numpy().tolist()

    return embed_fn


def create_dpr_embed_fn(
    ctx_encoder_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
    question_encoder_name: str = "facebook/dpr-question_encoder-single-nq-base",
    device: str = "cpu",
    batch_size: int = 32
) -> Optional[Tuple[Callable[[List[str]], List[List[float]]], Callable[[List[str]], List[List[float]]]]]:
    """
    创建 DPR 嵌入函数

    DPR 使用两个独立的编码器：一个用于上下文（文档），一个用于问题（查询）

    Args:
        ctx_encoder_name: 上下文编码器名称或路径
        question_encoder_name: 问题编码器名称或路径
        device: 设备 ("cpu", "cuda", "cuda:0" 等)
        batch_size: 批处理大小

    Returns:
        (ctx_embed_fn, question_embed_fn) 元组；如果无法加载模型则返回None
    """
    try:
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
        import torch
    except ImportError:
        print("警告: 未安装 transformers 或 torch。请运行: pip install transformers torch")
        return None

    try:
        print(f"正在加载 DPR 模型...")
        print(f"  - 上下文编码器: {ctx_encoder_name}")
        print(f"  - 问题编码器: {question_encoder_name}")

        # 处理上下文编码器路径
        ctx_path = Path(ctx_encoder_name)
        if not ctx_path.is_absolute():
            ctx_path = Path.cwd() / ctx_encoder_name
        is_ctx_local = ctx_path.exists()

        # 处理问题编码器路径
        q_path = Path(question_encoder_name)
        if not q_path.is_absolute():
            q_path = Path.cwd() / question_encoder_name
        is_q_local = q_path.exists()

        # 加载上下文编码器
        if is_ctx_local:
            ctx_path_str = str(ctx_path.resolve())
            print(f"  检测到本地上下文编码器: {ctx_path_str}")
            ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_path_str, local_files_only=True)
            ctx_encoder = DPRContextEncoder.from_pretrained(ctx_path_str, local_files_only=True)
        else:
            print(f"  从 HuggingFace Hub 加载上下文编码器...")
            ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder_name)
            ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder_name)

        ctx_encoder = ctx_encoder.to(device)
        ctx_encoder.eval()

        # 加载问题编码器
        if is_q_local:
            q_path_str = str(q_path.resolve())
            print(f"  检测到本地问题编码器: {q_path_str}")
            q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_path_str, local_files_only=True)
            q_encoder = DPRQuestionEncoder.from_pretrained(q_path_str, local_files_only=True)
        else:
            print(f"  从 HuggingFace Hub 加载问题编码器...")
            q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_name)
            q_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name)

        q_encoder = q_encoder.to(device)
        q_encoder.eval()

        print(f"DPR 模型加载成功 (device: {device})")
    except Exception as e:
        print(f"错误: 无法加载 DPR 模型: {e}")
        return None

    def ctx_embed_fn(texts: List[str]) -> List[List[float]]:
        """DPR 上下文嵌入函数（用于索引文档）"""
        if not texts:
            return []

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = ctx_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                embeddings = ctx_encoder(**inputs).pooler_output
                all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings.numpy().tolist()

    def question_embed_fn(texts: List[str]) -> List[List[float]]:
        """DPR 问题嵌入函数（用于查询）"""
        if not texts:
            return []

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = q_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                embeddings = q_encoder(**inputs).pooler_output
                all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings.numpy().tolist()

    return ctx_embed_fn, question_embed_fn


def create_embed_fn(
    model_type: str = "contriever",
    model_name: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> Optional[Callable[[List[str]], List[List[float]]]]:
    """
    根据配置创建嵌入函数的工厂方法

    Args:
        model_type: 模型类型 ("contriever", "dpr", "bge-m3")
        model_name: 模型名称或路径（如果为 None，使用默认值）
        device: 设备
        **kwargs: 传递给具体创建函数的额外参数

    Returns:
        嵌入函数；如果无法创建则返回None

    Examples:
        # 使用 Contriever
        embed_fn = create_embed_fn("contriever", "facebook/contriever-msmarco")

        # 使用 DPR（注意：DPR 返回两个函数）
        result = create_embed_fn("dpr")
        if result:
            ctx_embed_fn, question_embed_fn = result

        # 使用 BGE-M3
        embed_fn = create_embed_fn("bge-m3", "./models/bge-m3/bge-m3")
    """
    model_type = model_type.lower()

    if model_type == "contriever":
        if model_name is None:
            model_name = "facebook/contriever-msmarco"
        return create_contriever_embed_fn(
            model_name=model_name,
            device=device,
            **kwargs
        )

    elif model_type == "dpr":
        warnings.warn(
            "DPR 返回两个嵌入函数（上下文编码器和问题编码器）。"
            "请使用 create_dpr_embed_fn() 直接调用。",
            UserWarning
        )
        ctx_encoder = kwargs.get("ctx_encoder_name", "facebook/dpr-ctx_encoder-single-nq-base")
        question_encoder = kwargs.get("question_encoder_name", "facebook/dpr-question_encoder-single-nq-base")

        result = create_dpr_embed_fn(
            ctx_encoder_name=ctx_encoder,
            question_encoder_name=question_encoder,
            device=device,
            batch_size=kwargs.get("batch_size", 32)
        )

        if result:
            # 对于统一接口，返回上下文编码器（用于索引）
            ctx_embed_fn, _ = result
            return ctx_embed_fn
        return None

    elif model_type == "bge-m3":
        if model_name is None:
            model_name = "./models/bge-m3/bge-m3"
        return create_local_embed_fn(
            model_path=model_name,
            normalize_embeddings=kwargs.get("normalize_embeddings", True)
        )

    else:
        raise ValueError(
            f"不支持的模型类型: {model_type}。"
            f"支持的类型: 'contriever', 'dpr', 'bge-m3'"
        )