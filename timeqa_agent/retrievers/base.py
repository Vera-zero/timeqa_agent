"""
检索器基类和通用数据结构
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from ..config import RetrieverConfig


class RetrievalMode(str, Enum):
    """检索模式"""
    KEYWORD = "keyword"      # 关键词检索
    SEMANTIC = "semantic"    # 语义检索
    HYBRID = "hybrid"        # 混合检索


@dataclass
class RetrievalResult:
    """检索结果基类"""
    node_id: str                             # 节点ID
    node_type: str                           # 节点类型
    score: float = 0.0                       # 相关性分数
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class EntityResult(RetrievalResult):
    """实体检索结果"""
    canonical_name: str = ""
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    source_event_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = "entity"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "canonical_name": self.canonical_name,
            "description": self.description,
            "aliases": self.aliases,
            "source_event_ids": self.source_event_ids,
        })
        return base
    
    def get_searchable_text(self) -> str:
        """获取可搜索的拼接文本"""
        parts = [self.canonical_name, self.description]
        parts.extend(self.aliases)
        return " ".join(filter(None, parts))


@dataclass
class EventResult(RetrievalResult):
    """事件检索结果"""
    event_id: str = ""
    event_description: str = ""
    time_type: str = ""
    time_start: str = ""
    time_end: str = ""
    time_expression: str = ""
    entity_names: List[str] = field(default_factory=list)
    original_sentence: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    doc_title: str = ""
    
    def __post_init__(self):
        self.node_type = "event"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "event_id": self.event_id,
            "event_description": self.event_description,
            "time_type": self.time_type,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "time_expression": self.time_expression,
            "entity_names": self.entity_names,
            "original_sentence": self.original_sentence,
        })
        return base
    
    def get_searchable_text(self) -> str:
        """获取可搜索的拼接文本"""
        parts = [
            self.event_description,
            self.time_expression,
            self.original_sentence,
        ]
        parts.extend(self.entity_names)
        return " ".join(filter(None, parts))


@dataclass
class TimelineResult(RetrievalResult):
    """时间线检索结果"""
    timeline_id: str = ""
    timeline_name: str = ""
    description: str = ""
    entity_canonical_name: str = ""
    time_span_start: str = ""
    time_span_end: str = ""
    event_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = "timeline"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "timeline_id": self.timeline_id,
            "timeline_name": self.timeline_name,
            "description": self.description,
            "entity_canonical_name": self.entity_canonical_name,
            "time_span_start": self.time_span_start,
            "time_span_end": self.time_span_end,
            "event_ids": self.event_ids,
        })
        return base
    
    def get_searchable_text(self) -> str:
        """获取可搜索的拼接文本"""
        parts = [
            self.timeline_name,
            self.description,
            self.entity_canonical_name,
        ]
        return " ".join(filter(None, parts))


class BaseRetriever(ABC):
    """检索器基类"""
    
    def __init__(self, graph_store, config: Optional[RetrieverConfig] = None):
        """
        初始化检索器
        
        Args:
            graph_store: TimelineGraphStore 实例
            config: 检索器配置
        """
        self.graph_store = graph_store
        self.config = config or RetrieverConfig()
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行检索
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            **kwargs: 其他参数
            
        Returns:
            检索结果列表
        """
        pass
    
    def _get_top_k(self, top_k: Optional[int]) -> int:
        """获取实际的 top_k 值"""
        return top_k if top_k is not None else self.config.top_k
    
    def _filter_by_threshold(
        self,
        results: List[RetrievalResult],
        threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """按分数阈值过滤结果"""
        threshold = threshold if threshold is not None else self.config.score_threshold
        return [r for r in results if r.score >= threshold]
    
    def _sort_by_score(
        self,
        results: List[RetrievalResult],
        descending: bool = True
    ) -> List[RetrievalResult]:
        """按分数排序"""
        return sorted(results, key=lambda x: x.score, reverse=descending)
