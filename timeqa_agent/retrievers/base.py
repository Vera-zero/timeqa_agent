"""
检索器基类和通用数据结构
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

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

    # ========== 事件溯源到Chunk的方法 ==========

    def get_chunk_info_by_event(
        self,
        event: Union[str, EventResult]
    ) -> Optional[Dict[str, str]]:
        """
        根据事件获取对应chunk的元信息

        Args:
            event: 事件ID字符串或EventResult对象

        Returns:
            chunk元信息字典，包含:
                - chunk_id: chunk唯一标识
                - doc_id: 所属文档ID
                - doc_title: 文档标题
            如果无法获取则返回None
        """
        chunk_id = None
        doc_id = None
        doc_title = None

        # 如果是EventResult对象，直接提取属性
        if isinstance(event, EventResult):
            chunk_id = event.chunk_id
            doc_id = event.doc_id
            doc_title = event.doc_title
        # 如果是字符串（event_id），从graph_store查询
        elif isinstance(event, str):
            event_data = self.graph_store.get_event(event)
            if event_data:
                chunk_id = event_data.get("chunk_id", "")
                doc_id = event_data.get("doc_id", "")
                doc_title = event_data.get("doc_title", "")

        # 检查是否成功获取了chunk_id
        if not chunk_id:
            return None

        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_title": doc_title,
        }

    def get_chunk_by_event(
        self,
        event: Union[str, EventResult],
        chunks_store: Union[Dict[str, Any], str, Path]
    ) -> Optional[Dict[str, Any]]:
        """
        根据事件获取完整的chunk数据（包括内容）

        Args:
            event: 事件ID字符串或EventResult对象
            chunks_store: chunk数据存储，支持以下格式:
                - Dict[str, Any]: chunk_id到chunk数据的映射字典
                - str/Path: chunk数据文件路径（JSON格式）
                    * 可以是chunk列表: [{"chunk_id": ..., "content": ...}, ...]
                    * 可以是chunk字典: {"chunk_id_1": {...}, "chunk_id_2": {...}}

        Returns:
            完整的chunk数据字典（包括content字段）
            如果无法获取则返回None

        Examples:
            >>> # 使用字典存储
            >>> chunks = {"chunk-001": {"chunk_id": "chunk-001", "content": "..."}}
            >>> chunk = retriever.get_chunk_by_event(event, chunks)

            >>> # 使用文件路径
            >>> chunk = retriever.get_chunk_by_event(event, "data/chunks.json")
        """
        # 1. 获取chunk_id
        chunk_info = self.get_chunk_info_by_event(event)
        if not chunk_info:
            return None

        chunk_id = chunk_info["chunk_id"]

        # 2. 加载chunks_store（如果是文件路径）
        chunks_dict = None
        if isinstance(chunks_store, (str, Path)):
            chunks_path = Path(chunks_store)
            if not chunks_path.exists():
                return None

            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            # 处理两种JSON格式
            if isinstance(chunks_data, list):
                # 列表格式：转换为字典
                chunks_dict = {chunk["chunk_id"]: chunk for chunk in chunks_data}
            elif isinstance(chunks_data, dict):
                # 字典格式：直接使用
                chunks_dict = chunks_data
            else:
                return None
        elif isinstance(chunks_store, dict):
            chunks_dict = chunks_store
        else:
            return None

        # 3. 查找chunk
        return chunks_dict.get(chunk_id)

    def get_chunks_for_events(
        self,
        events: List[EventResult],
        chunks_store: Union[Dict[str, Any], str, Path],
        deduplicate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量获取多个事件对应的chunks

        Args:
            events: 事件列表（EventResult对象）
            chunks_store: chunk数据存储（格式同get_chunk_by_event）
            deduplicate: 是否去重（默认True）
                - True: 同一个chunk只返回一次
                - False: 保留所有chunk（可能有重复）

        Returns:
            chunk数据列表

        Examples:
            >>> events = retriever.retrieve("查询")
            >>> chunks = retriever.get_chunks_for_events(events, "data/chunks.json")
            >>> print(f"检索到 {len(events)} 个事件，来自 {len(chunks)} 个chunks")
        """
        chunks = []
        seen_chunk_ids = set()

        for event in events:
            chunk = self.get_chunk_by_event(event, chunks_store)
            if chunk:
                chunk_id = chunk.get("chunk_id")

                # 去重逻辑
                if deduplicate:
                    if chunk_id not in seen_chunk_ids:
                        chunks.append(chunk)
                        seen_chunk_ids.add(chunk_id)
                else:
                    chunks.append(chunk)

        return chunks

    def get_surrounding_chunks(
        self,
        chunk: Union[str, Dict[str, Any]],
        chunks_store: Union[Dict[str, Any], str, Path],
        before: int = 1,
        after: int = 1
    ) -> Dict[str, Any]:
        """
        获取指定chunk的前后上下文chunks

        Args:
            chunk: chunk_id字符串或chunk数据字典
            chunks_store: chunk数据存储（格式同get_chunk_by_event）
            before: 获取前面N个chunk（默认1）
            after: 获取后面N个chunk（默认1）

        Returns:
            包含前后chunk的字典:
            {
                "before": [前面的chunks列表，按顺序],
                "current": 当前chunk,
                "after": [后面的chunks列表，按顺序],
                "doc_id": 文档ID,
                "chunk_index": 当前chunk索引
            }
            如果无法获取则返回None

        Examples:
            >>> # 从chunk_id获取
            >>> result = retriever.get_surrounding_chunks("doc-001-chunk-0005", chunks_file, before=2, after=2)
            >>> print(f"前面: {len(result['before'])} 个, 后面: {len(result['after'])} 个")

            >>> # 从chunk字典获取
            >>> chunk = {"chunk_id": "doc-001-chunk-0005", "content": "..."}
            >>> result = retriever.get_surrounding_chunks(chunk, chunks_file, before=1, after=1)
        """
        # 1. 获取当前chunk信息
        chunk_id = None
        current_chunk = None

        if isinstance(chunk, str):
            chunk_id = chunk
        elif isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id")
            current_chunk = chunk

        if not chunk_id:
            return None

        # 2. 解析chunk_id，获取doc_id和chunk_index
        # chunk_id格式: "{doc_id}-chunk-{chunk_index:04d}"
        try:
            parts = chunk_id.rsplit("-chunk-", 1)
            if len(parts) != 2:
                return None

            doc_id = parts[0]
            chunk_index = int(parts[1])
        except (ValueError, IndexError):
            return None

        # 3. 加载chunks_store
        chunks_dict = None
        if isinstance(chunks_store, (str, Path)):
            chunks_path = Path(chunks_store)
            if not chunks_path.exists():
                return None

            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            # 处理两种JSON格式
            if isinstance(chunks_data, list):
                chunks_dict = {c["chunk_id"]: c for c in chunks_data}
            elif isinstance(chunks_data, dict):
                chunks_dict = chunks_data
            else:
                return None
        elif isinstance(chunks_store, dict):
            chunks_dict = chunks_store
        else:
            return None

        # 4. 获取当前chunk（如果还没有）
        if not current_chunk:
            current_chunk = chunks_dict.get(chunk_id)
            if not current_chunk:
                return None

        # 5. 构造前后chunk的ID并查找
        before_chunks = []
        after_chunks = []

        # 前面的chunks（从近到远）
        for i in range(1, before + 1):
            prev_index = chunk_index - i
            if prev_index < 0:
                break
            prev_chunk_id = f"{doc_id}-chunk-{prev_index:04d}"
            prev_chunk = chunks_dict.get(prev_chunk_id)
            if prev_chunk:
                before_chunks.insert(0, prev_chunk)  # 插入到开头，保持顺序
            else:
                break  # 如果中间缺失，停止查找

        # 后面的chunks（从近到远）
        for i in range(1, after + 1):
            next_index = chunk_index + i
            next_chunk_id = f"{doc_id}-chunk-{next_index:04d}"
            next_chunk = chunks_dict.get(next_chunk_id)
            if next_chunk:
                after_chunks.append(next_chunk)
            else:
                break  # 如果中间缺失，停止查找

        return {
            "before": before_chunks,
            "current": current_chunk,
            "after": after_chunks,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "total_chunks": len(before_chunks) + 1 + len(after_chunks)
        }

    def get_surrounding_chunks_by_event(
        self,
        event: Union[str, EventResult],
        chunks_store: Union[Dict[str, Any], str, Path],
        before: int = 1,
        after: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        根据事件获取对应chunk及其前后上下文

        Args:
            event: 事件ID字符串或EventResult对象
            chunks_store: chunk数据存储（格式同get_chunk_by_event）
            before: 获取前面N个chunk（默认1）
            after: 获取后面N个chunk（默认1）

        Returns:
            包含前后chunk的字典（格式同get_surrounding_chunks）
            如果无法获取则返回None

        Examples:
            >>> event = retriever.retrieve("John career", target_type="event")[0]
            >>> result = retriever.get_surrounding_chunks_by_event(event, chunks_file, before=2, after=2)
            >>> print(f"事件所在chunk的前后文:")
            >>> for chunk in result['before']:
            >>>     print(f"  前: {chunk['content'][:100]}")
            >>> print(f"  当前: {result['current']['content'][:100]}")
            >>> for chunk in result['after']:
            >>>     print(f"  后: {chunk['content'][:100]}")
        """
        # 1. 获取事件对应的chunk
        chunk = self.get_chunk_by_event(event, chunks_store)
        if not chunk:
            return None

        # 2. 获取前后chunk
        return self.get_surrounding_chunks(chunk, chunks_store, before, after)


# ========== 三层递进检索结果类 ==========


@dataclass
class HierarchicalEventResult(EventResult):
    """三层递进检索的事件结果（包含溯源信息）"""

    source_entity_names: List[str] = field(default_factory=list)
    source_entity_scores: List[float] = field(default_factory=list)
    source_timeline_id: Optional[str] = None
    hierarchical_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "source_entity_names": self.source_entity_names,
            "source_entity_scores": self.source_entity_scores,
            "source_timeline_id": self.source_timeline_id,
            "hierarchical_score": self.hierarchical_score,
        })
        return base


@dataclass
class HierarchicalTimelineResult(TimelineResult):
    """三层递进检索的时间线结果（包含溯源信息）"""

    source_entity_names: List[str] = field(default_factory=list)
    source_entity_scores: List[float] = field(default_factory=list)
    hierarchical_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "source_entity_names": self.source_entity_names,
            "source_entity_scores": self.source_entity_scores,
            "hierarchical_score": self.hierarchical_score,
        })
        return base


@dataclass
class HierarchicalRetrievalResults:
    """三层递进检索完整结果"""

    events: List[HierarchicalEventResult] = field(default_factory=list)
    timelines: List[HierarchicalTimelineResult] = field(default_factory=list)

    # 中间层结果（可选，用于调试）
    layer1_entities: Optional[List[EntityResult]] = None
    layer2_all_timelines: Optional[List[TimelineResult]] = None
    layer2_all_events: Optional[List[EventResult]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "events": [e.to_dict() for e in self.events],
            "timelines": [t.to_dict() for t in self.timelines],
        }
        if self.layer1_entities is not None:
            result["layer1_entities"] = [e.to_dict() for e in self.layer1_entities]
        if self.layer2_all_timelines is not None:
            result["layer2_all_timelines"] = [t.to_dict() for t in self.layer2_all_timelines]
        if self.layer2_all_events is not None:
            result["layer2_all_events"] = [e.to_dict() for e in self.layer2_all_events]
        return result
