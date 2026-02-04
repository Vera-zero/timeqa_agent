"""
TimeQA 模块配置

所有可配置参数集中管理
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, asdict
from pathlib import Path

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


class ChunkStrategy(str, Enum):
    """分块策略"""
    FIXED_SIZE = "fixed_size"  # 固定大小分块
    SENTENCE = "sentence"      # 按句子分块


@dataclass
class ChunkConfig:
    """分块配置"""
    strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    
    # 固定大小分块参数
    chunk_size: int = 1500         # 分块大小（字符数）
    chunk_overlap: int = 100       # 重叠大小（字符数）
    
    # 句子分块参数
    max_sentences: int = 10        # 每个分块最大句子数
    min_chunk_size: int = 500      # 最小分块大小（字符数）
    max_chunk_size: int = 2000     # 最大分块大小（字符数）
    
    # 通用参数
    preserve_sentences: bool = True  # 固定大小分块时是否尽量保持句子完整
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = asdict(self)
        d["strategy"] = self.strategy.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkConfig":
        """从字典创建"""
        if "strategy" in data:
            data["strategy"] = ChunkStrategy(data["strategy"])
        return cls(**data)


@dataclass
class ExtractorConfig:
    """抽取器配置"""
    # API 配置
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/chat/completions"
    temperature: float = 0.1
    max_retries: int = 3
    timeout: int = 180
    
    # 抽取配置
    batch_size: int = 1            # 每次处理的分块数（目前只支持1）
    include_implicit_time: bool = True  # 是否包含隐式时间引用
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractorConfig":
        """从字典创建"""
        return cls(**data)


@dataclass
class DisambiguatorConfig:
    """实体消歧配置"""
    # Embedding API 配置
    embed_model: str = "deepseek-embed"
    embed_base_url: str = "https://api.deepseek.com/v1"
    embed_batch_size: int = 100  # 每批次处理的实体数

    # 本地模型配置
    local_embed_model_path: Optional[str] = "d:/Verause/science/codes/models/bge_m3/bge_m3"  # 本地模型路径
    
    # 相似度阈值
    similarity_threshold: float = 0.85  # 高于此阈值认为是同一实体
    
    # 文本拼接权重（用于生成 embedding 文本）
    canonical_name_weight: float = 2.0  # canonical_name 重复次数（增加权重）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisambiguatorConfig":
        """从字典创建"""
        return cls(**data)


@dataclass
class TimelineConfig:
    """Timeline 抽取配置"""
    # API 配置
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/chat/completions"
    temperature: float = 0.1
    max_retries: int = 3
    timeout: int = 180
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineConfig":
        """从字典创建"""
        return cls(**data)


@dataclass
class GraphStoreConfig:
    """图存储配置"""
    store_original_sentence: bool = True   # 是否存储原始句子
    store_chunk_metadata: bool = True      # 是否存储分块元数据
    store_entity_aliases: bool = True      # 是否存储实体别名
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphStoreConfig":
        """从字典创建"""
        return cls(**data)


class FusionMode(str, Enum):
    """融合模式"""
    RRF = "rrf"                    # 倒数排名融合 (Reciprocal Rank Fusion)
    WEIGHTED_SUM = "weighted_sum"  # 加权求和
    MAX_SCORE = "max_score"        # 取最大分数
    INTERLEAVE = "interleave"      # 交错合并


class VotingFusionMode(str, Enum):
    """投票融合模式"""
    RRF = "rrf"                    # 倒数排名融合
    WEIGHTED = "weighted"          # 加权投票
    VOTE_COUNT = "vote_count"      # 投票计数 + 平均分数
    BORDA = "borda"                # Borda Count


@dataclass
class VotingConfig:
    """投票检索器配置"""
    # 各层权重
    entity_layer_weight: float = 0.25      # 实体层权重
    timeline_layer_weight: float = 0.30    # 时间线层权重
    event_layer_weight: float = 0.45       # 事件层权重
    
    # 衰减因子（从高层传播到事件）
    entity_decay_factor: float = 0.7       # 实体到事件的衰减
    timeline_decay_factor: float = 0.9     # 时间线到事件的衰减
    
    # 融合算法
    fusion_mode: VotingFusionMode = VotingFusionMode.RRF
    rrf_k: int = 60                        # RRF 参数
    
    # 投票阈值
    min_votes: int = 1                     # 最少投票数
    min_score: float = 0.0                 # 最低聚合分数
    
    # 各层检索数量倍数（相对于 top_k）
    entity_retrieve_multiplier: float = 2.0
    timeline_retrieve_multiplier: float = 2.0
    event_retrieve_multiplier: float = 2.0
    
    # 是否启用各层
    enable_entity_layer: bool = True
    enable_timeline_layer: bool = True
    enable_event_layer: bool = True
    
    # 投票计数模式的参数
    vote_count_alpha: float = 0.3          # 投票数权重
    vote_count_beta: float = 0.7           # 平均分数权重
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = asdict(self)
        d["fusion_mode"] = self.fusion_mode.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VotingConfig":
        """从字典创建"""
        if "fusion_mode" in data:
            data = data.copy()
            data["fusion_mode"] = VotingFusionMode(data["fusion_mode"])
        return cls(**data)


class VectorIndexType(str, Enum):
    """向量索引类型"""
    FLAT = "flat"      # 暴力搜索 (FAISS IndexFlat)
    IVF = "ivf"        # 倒排索引 (FAISS IndexIVFFlat)
    HNSW = "hnsw"      # 近似最近邻 (FAISS IndexHNSWFlat)


class VectorMetric(str, Enum):
    """向量距离度量"""
    COSINE = "cosine"  # 余弦相似度
    L2 = "l2"          # 欧氏距离
    IP = "ip"          # 内积


# 默认停用词
DEFAULT_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all",
    "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "and", "but", "if", "or", "because",
    "until", "while", "although", "though", "since",
})


@dataclass
class RetrieverConfig:
    """检索器通用配置"""
    # 通用参数
    top_k: int = 10                          # 默认返回数量
    score_threshold: float = 0.0             # 分数阈值
    include_metadata: bool = True            # 是否包含元数据
    fuzzy_match: bool = True                 # 是否模糊匹配
    case_sensitive: bool = False             # 是否大小写敏感
    
    # 关键词检索参数
    use_tfidf: bool = True                   # 是否使用 TF-IDF 排序
    min_keyword_length: int = 2              # 最小关键词长度
    
    # 语义检索参数
    embedding_dim: int = 768                 # 嵌入维度
    embed_batch_size: int = 32               # 批量嵌入大小
    similarity_threshold: float = 0.5        # 语义相似度阈值
    cache_embeddings: bool = True            # 是否缓存嵌入
    
    # 向量索引参数
    vector_index_type: str = "flat"          # 索引类型: flat, hnsw
    vector_metric: str = "cosine"            # 距离度量: cosine, l2, ip
    hnsw_m: int = 16                         # HNSW M 参数
    hnsw_ef_construction: int = 200          # HNSW 构建时的 ef
    hnsw_ef_search: int = 50                 # HNSW 搜索时的 ef
    
    # 混合检索参数
    keyword_weight: float = 0.3              # 关键词检索权重
    semantic_weight: float = 0.7             # 语义检索权重
    fusion_mode: str = "rrf"                 # 融合模式: rrf, weighted_sum, max_score, interleave
    rrf_k: float = 60.0                      # RRF 参数
    enable_keyword: bool = True              # 启用关键词检索
    enable_semantic: bool = True             # 启用语义检索
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrieverConfig":
        """从字典创建"""
        return cls(**data)
    
    def get_fusion_mode(self) -> FusionMode:
        """获取融合模式枚举"""
        return FusionMode(self.fusion_mode)
    
    def get_vector_index_type(self) -> VectorIndexType:
        """获取向量索引类型枚举"""
        return VectorIndexType(self.vector_index_type)
    
    def get_vector_metric(self) -> VectorMetric:
        """获取向量距离度量枚举"""
        return VectorMetric(self.vector_metric)
    
    def get_stopwords(self) -> frozenset:
        """获取停用词集合"""
        return DEFAULT_STOPWORDS


@dataclass
class TimeQAConfig:
    """TimeQA 完整配置"""
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    disambiguator: DisambiguatorConfig = field(default_factory=DisambiguatorConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    graph_store: GraphStoreConfig = field(default_factory=GraphStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    voting: VotingConfig = field(default_factory=VotingConfig)
    
    # 路径配置
    data_dir: str = "data/timeqa"
    corpus_dir: str = "data/timeqa/corpus"
    output_dir: str = "data/timeqa/processed"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chunk": self.chunk.to_dict(),
            "extractor": self.extractor.to_dict(),
            "disambiguator": self.disambiguator.to_dict(),
            "timeline": self.timeline.to_dict(),
            "graph_store": self.graph_store.to_dict(),
            "retriever": self.retriever.to_dict(),
            "voting": self.voting.to_dict(),
            "data_dir": self.data_dir,
            "corpus_dir": self.corpus_dir,
            "output_dir": self.output_dir,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeQAConfig":
        """从字典创建"""
        return cls(
            chunk=ChunkConfig.from_dict(data.get("chunk", {})),
            extractor=ExtractorConfig.from_dict(data.get("extractor", {})),
            disambiguator=DisambiguatorConfig.from_dict(data.get("disambiguator", {})),
            timeline=TimelineConfig.from_dict(data.get("timeline", {})),
            graph_store=GraphStoreConfig.from_dict(data.get("graph_store", {})),
            retriever=RetrieverConfig.from_dict(data.get("retriever", {})),
            voting=VotingConfig.from_dict(data.get("voting", {})),
            data_dir=data.get("data_dir", "data/timeqa"),
            corpus_dir=data.get("corpus_dir", "data/timeqa/corpus"),
            output_dir=data.get("output_dir", "data/timeqa/processed"),
        )
    
    def save(self, path: str) -> None:
        """保存配置到文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "TimeQAConfig":
        """从文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def load_or_default(cls, path: str) -> "TimeQAConfig":
        """从文件加载配置，如果不存在则返回默认配置"""
        if os.path.exists(path):
            return cls.load(path)
        return cls()


# 默认配置文件路径
DEFAULT_CONFIG_PATH = "configs/timeqa_config.json"


def get_default_config() -> TimeQAConfig:
    """获取默认配置（优先从默认配置文件加载）"""
    return TimeQAConfig.load_or_default(DEFAULT_CONFIG_PATH)


def load_config(path: Optional[str] = None) -> TimeQAConfig:
    """加载配置
    
    优先级：
    1. 如果指定了 path，从该路径加载
    2. 否则尝试从默认配置文件加载
    3. 如果配置文件不存在，使用默认配置
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    return TimeQAConfig.load_or_default(path)