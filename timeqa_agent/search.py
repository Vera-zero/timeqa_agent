"""
Search Query Generator Module

生成检索语句模块。输入可以是单一实体或句子，输出检索语句。

功能:
1. 判断输入类型（单一实体 vs 句子）
2. 对于单一实体：调用 LLM 生成实体描述，timeline 和 event 为原始实体名
3. 对于句子：调用 LLM 生成针对实体、时间线、事件的检索语句
"""

from __future__ import annotations

import os
import json
import re
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

from .config import QueryParserConfig


@dataclass
class StructuredRelation:
    """结构化时间关系

    表示从事件中提取的结构化时间关系，格式为: relation_type(subject, object, start_time, end_time)
    """
    relation_id: str                          # 关系唯一标识符
    relation_type: str                        # 关系类型（如 works_for, studies_at, married_to 等）
    subject: str                              # 主体实体（人物、组织等）
    object_entity: str                        # 客体实体（工作单位、学习机构等）
    time_start: Optional[str]                 # 开始时间（YYYY, YYYY-MM, 或 YYYY-MM-DD）
    time_end: Optional[str]                   # 结束时间（用于时间段）
    confidence: float                         # 置信度分数（0.0-1.0）
    source_event_ids: List[str]              # 来源事件 ID 列表
    source_description: str                   # 原始事件描述

    def __str__(self) -> str:
        """格式化为: relation_type(subject, object, start_time, end_time)"""
        time_part = ""
        if self.time_start and self.time_end:
            time_part = f", {self.time_start}, {self.time_end}"
        elif self.time_start:
            time_part = f", {self.time_start}"
        return f"{self.relation_type}({self.subject}, {self.object_entity}{time_part})"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "subject": self.subject,
            "object_entity": self.object_entity,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "confidence": self.confidence,
            "source_event_ids": self.source_event_ids,
            "source_description": self.source_description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredRelation":
        """从字典创建"""
        return cls(
            relation_id=data.get("relation_id", ""),
            relation_type=data.get("relation_type", ""),
            subject=data.get("subject", ""),
            object_entity=data.get("object_entity", ""),
            time_start=data.get("time_start"),
            time_end=data.get("time_end"),
            confidence=data.get("confidence", 1.0),
            source_event_ids=data.get("source_event_ids", []),
            source_description=data.get("source_description", ""),
        )


@dataclass
class RetrievalQueries:
    """检索语句集合"""
    entity_query: str              # 实体检索语句：标准化名称+简短描述
    timeline_query: str            # 时间线检索语句：时间线名称+描述+相关实体
    event_queries: List[str]       # 事件检索语句列表：将主干问句转为多个陈述句

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_query": self.entity_query,
            "timeline_query": self.timeline_query,
            "event_queries": self.event_queries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalQueries":
        return cls(
            entity_query=data.get("entity_query", ""),
            timeline_query=data.get("timeline_query", ""),
            event_queries=data.get("event_queries", []),
        )


@dataclass
class RetrievalResults:
    """检索结果集合

    存储使用生成的检索语句进行检索后的结果
    """
    # 原始检索语句
    queries: RetrievalQueries

    # 检索结果（来自 retrievers 模块）
    entities: List[Any] = field(default_factory=list)      # EntityResult 列表
    timelines: List[Any] = field(default_factory=list)     # TimelineResult 列表
    events: List[Any] = field(default_factory=list)        # EventResult 列表

    # 检索元信息
    retrieval_mode: str = ""       # 使用的检索模式（hybrid/keyword/semantic/voting/hierarchical）
    entity_top_k: int = 0          # 实体检索数量
    timeline_top_k: int = 0        # 时间线检索数量
    event_top_k: int = 0           # 事件检索数量

    # 合并后的事件（包含从时间线提取的事件）
    merged_events: List[Any] = field(default_factory=list)  # 合并去重后的事件列表

    # 结构化事件关系（从合并事件中提取的结构化关系）
    structured_events: List[StructuredRelation] = field(default_factory=list)  # 结构化时间关系列表

    # 问题解析和过滤后的结构化事件
    question_analysis: Optional[Any] = None  # QueryParseResult
    filtered_structured_events: List[StructuredRelation] = field(default_factory=list)  # 过滤后的结构化关系列表

    def extract_and_merge_events(self, graph_store: Optional[Any] = None) -> List[Any]:
        """
        从时间线中提取事件，与已有事件结果合并去重

        Args:
            graph_store: 图存储实例（用于查询时间线的事件）

        Returns:
            合并去重后的事件列表
        """
        if graph_store is None:
            # 如果没有提供 graph_store，尝试从时间线结果的 event_ids 字段提取
            # 这种情况下只能返回事件ID，无法获取完整事件信息
            all_event_ids = set()

            # 收集已有事件的ID
            for event in self.events:
                if hasattr(event, 'event_id'):
                    all_event_ids.add(event.event_id)
                elif hasattr(event, 'node_id'):
                    all_event_ids.add(event.node_id)

            # 从时间线中提取事件ID
            for timeline in self.timelines:
                if hasattr(timeline, 'event_ids'):
                    all_event_ids.update(timeline.event_ids)

            # 只能返回已有的事件对象（无法从ID构造完整事件）
            self.merged_events = list(self.events)
            return self.merged_events

        # 使用 graph_store 提取完整事件信息
        all_events_dict = {}  # 使用字典去重：event_id -> event_object

        # 1. 先添加直接检索到的事件
        for event in self.events:
            event_id = event.event_id if hasattr(event, 'event_id') else event.node_id
            all_events_dict[event_id] = event

        # 2. 从时间线中提取事件
        for timeline in self.timelines:
            timeline_id = timeline.timeline_id if hasattr(timeline, 'timeline_id') else timeline.node_id

            # 获取时间线包含的所有事件
            timeline_events = graph_store.get_timeline_events(timeline_id)

            for event_data in timeline_events:
                event_id = event_data.get('event_id') or event_data.get('node_id')

                # 如果该事件还未被添加，则添加它
                if event_id and event_id not in all_events_dict:
                    # 将字典数据转换为 EventResult 对象
                    from .retrievers.base import EventResult

                    event_obj = EventResult(
                        node_id=event_id,
                        node_type="event",
                        event_id=event_id,
                        event_description=event_data.get('event_description', ''),
                        time_type=event_data.get('time_type', ''),
                        time_start=event_data.get('time_start', ''),
                        time_end=event_data.get('time_end', ''),
                        time_expression=event_data.get('time_expression', ''),
                        entity_names=event_data.get('entity_names', []),
                        original_sentence=event_data.get('original_sentence', ''),
                        chunk_id=event_data.get('chunk_id', ''),
                        doc_id=event_data.get('doc_id', ''),
                        doc_title=event_data.get('doc_title', ''),
                        score=0.0,  # 从时间线提取的事件，分数设为0
                    )
                    all_events_dict[event_id] = event_obj

        # 转换为列表
        self.merged_events = list(all_events_dict.values())

        return self.merged_events

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "queries": self.queries.to_dict(),
            "entities": [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.entities],
            "timelines": [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.timelines],
            "events": [ev.to_dict() if hasattr(ev, 'to_dict') else ev for ev in self.events],
            "merged_events": [ev.to_dict() if hasattr(ev, 'to_dict') else ev for ev in self.merged_events],
            "structured_events": [se.to_dict() if hasattr(se, 'to_dict') else se for se in self.structured_events],
            "filtered_structured_events": [se.to_dict() if hasattr(se, 'to_dict') else se for se in self.filtered_structured_events],
            "question_analysis": self.question_analysis.to_dict() if self.question_analysis and hasattr(self.question_analysis, 'to_dict') else None,
            "retrieval_mode": self.retrieval_mode,
            "entity_top_k": self.entity_top_k,
            "timeline_top_k": self.timeline_top_k,
            "event_top_k": self.event_top_k,
            "summary": {
                "num_entities": len(self.entities),
                "num_timelines": len(self.timelines),
                "num_events": len(self.events),
                "num_merged_events": len(self.merged_events),
                "num_structured_relations": len(self.structured_events),
                "num_filtered_structured_relations": len(self.filtered_structured_events),
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResults":
        """从字典创建"""
        # 解析 structured_events
        structured_events = []
        for se_data in data.get("structured_events", []):
            if isinstance(se_data, dict):
                structured_events.append(StructuredRelation.from_dict(se_data))
            else:
                structured_events.append(se_data)

        return cls(
            queries=RetrievalQueries.from_dict(data.get("queries", {})),
            entities=data.get("entities", []),
            timelines=data.get("timelines", []),
            events=data.get("events", []),
            merged_events=data.get("merged_events", []),
            structured_events=structured_events,
            retrieval_mode=data.get("retrieval_mode", ""),
            entity_top_k=data.get("entity_top_k", 0),
            timeline_top_k=data.get("timeline_top_k", 0),
            event_top_k=data.get("event_top_k", 0),
        )


# ============================================================
# 事件结构化器
# ============================================================

class EventStructurizer:
    """事件结构化器

    使用 LLM 将非结构化的事件描述转换为结构化的时间关系
    """

    def __init__(
        self,
        config: Optional[QueryParserConfig] = None,
        token: Optional[str] = None,
    ):
        """初始化事件结构化器

        Args:
            config: 查询解析配置（使用 structuring_* 设置）
            token: LLM API 令牌
        """
        self.config = config or QueryParserConfig()

        # 获取 token
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("请设置 VENUS_API_TOKEN 或 OPENAI_API_KEY 环境变量")

    def _call_api(self, messages: List[dict]) -> str:
        """调用 LLM API

        Args:
            messages: 消息列表

        Returns:
            API 响应内容
        """
        # 使用 structuring_model（如果指定），否则使用默认 model
        model = self.config.structuring_model or self.config.model
        base_url = self.config.structuring_base_url or self.config.base_url
        temperature = self.config.structuring_temperature
        timeout = self.config.structuring_timeout
        max_retries = self.config.structuring_max_retries

        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'response_format': {'type': 'json_object'},
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout,
                )

                if response.status_code != 200:
                    raise Exception(f"API 调用失败: {response.status_code} - {response.text}")

                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"API 调用失败，重试 {attempt + 1}/{max_retries}: {e}")

        return ""

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 JSON 响应

        Args:
            content: API 响应内容

        Returns:
            解析后的 JSON 对象
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取 JSON
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                return json.loads(match.group(1))
            else:
                raise ValueError(f"无法解析 JSON 响应: {content[:200]}")

    def structurize_event(
        self,
        event_description: str,
        entity_names: List[str],
        time_start: Optional[str],
        time_end: Optional[str],
        event_id: str = "",
    ) -> List[StructuredRelation]:
        """结构化单个事件

        Args:
            event_description: 事件描述
            entity_names: 事件中提及的实体名称列表
            time_start: 事件开始时间
            time_end: 事件结束时间
            event_id: 事件 ID（用于追溯）

        Returns:
            StructuredRelation 对象列表
        """
        if not event_description or not entity_names:
            return []

        user_prompt = EVENT_STRUCTURIZATION_USER_PROMPT.format(
            event_description=event_description,
            entity_names=", ".join(entity_names),
            time_start=time_start or "N/A",
            time_end=time_end or "N/A",
        )

        messages = [
            {"role": "system", "content": EVENT_STRUCTURIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = self._call_api(messages)
            data = self._parse_json_response(content)

            relations = []
            for i, rel_data in enumerate(data.get("relations", []), 1):
                relation_id = rel_data.get("relation_id", f"{event_id}-rel-{i}" if event_id else f"rel-{id(rel_data)}")
                relation = StructuredRelation(
                    relation_id=relation_id,
                    relation_type=rel_data.get("relation_type", "unknown"),
                    subject=rel_data.get("subject", ""),
                    object_entity=rel_data.get("object_entity", ""),
                    time_start=rel_data.get("time_start"),
                    time_end=rel_data.get("time_end"),
                    confidence=rel_data.get("confidence", 1.0),
                    source_event_ids=[event_id] if event_id else [],
                    source_description=event_description,
                )
                relations.append(relation)

            return relations
        except Exception as e:
            print(f"事件结构化失败: {event_description[:80]}... 错误: {e}")
            return []

    def structurize_events(
        self,
        events: List[Any],
        batch_size: Optional[int] = None,
    ) -> List[StructuredRelation]:
        """结构化多个事件

        Args:
            events: EventResult 对象列表
            batch_size: 批处理大小（如果未指定则使用配置默认值）

        Returns:
            所有 StructuredRelation 对象列表
        """
        if not events:
            return []

        batch_size = batch_size or self.config.structuring_batch_size
        all_relations = []

        for i, event in enumerate(events, 1):
            # 获取事件 ID
            event_id = event.event_id if hasattr(event, 'event_id') else (event.node_id if hasattr(event, 'node_id') else "")

            # 打印进度
            event_desc_preview = event.event_description[:50] if hasattr(event, 'event_description') else str(event)[:50]
            print(f"  结构化事件 {i}/{len(events)}: {event_desc_preview}...")

            # 提取关系
            relations = self.structurize_event(
                event_description=event.event_description if hasattr(event, 'event_description') else "",
                entity_names=event.entity_names if hasattr(event, 'entity_names') else [],
                time_start=event.time_start if hasattr(event, 'time_start') else None,
                time_end=event.time_end if hasattr(event, 'time_end') else None,
                event_id=event_id,
            )

            all_relations.extend(relations)

            # 批处理进度打印
            if i % batch_size == 0:
                print(f"  已处理 {i}/{len(events)} 个事件，提取了 {len(all_relations)} 条关系")

        return all_relations


# ============================================================
# 事件过滤器
# ============================================================

class EventFilter:
    """事件过滤器

    使用 LLM 基于问题解析结果过滤结构化事件，保留与问题相关的事件
    """

    def __init__(
        self,
        config: Optional[QueryParserConfig] = None,
        token: Optional[str] = None,
    ):
        """初始化事件过滤器

        Args:
            config: 查询解析配置（使用 filtering_* 设置）
            token: LLM API 令牌
        """
        self.config = config or QueryParserConfig()

        # 获取 token
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("请设置 VENUS_API_TOKEN 或 OPENAI_API_KEY 环境变量")

    def _call_api(self, messages: List[dict]) -> str:
        """调用 LLM API

        Args:
            messages: 消息列表

        Returns:
            API 响应内容
        """
        # 使用 filtering_model（如果指定），否则使用默认 model
        model = self.config.filtering_model or self.config.model
        base_url = self.config.filtering_base_url or self.config.base_url
        temperature = self.config.filtering_temperature
        timeout = self.config.filtering_timeout
        max_retries = self.config.filtering_max_retries

        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'response_format': {'type': 'json_object'},
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout,
                )

                if response.status_code != 200:
                    raise Exception(f"API 调用失败: {response.status_code} - {response.text}")

                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"API 调用失败，重试 {attempt + 1}/{max_retries}: {e}")

        return ""

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 JSON 响应

        Args:
            content: API 响应内容

        Returns:
            解析后的 JSON 对象
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取 JSON
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                return json.loads(match.group(1))
            else:
                raise ValueError(f"无法解析 JSON 响应: {content[:200]}")

    def filter_structured_events(
        self,
        structured_events: List[StructuredRelation],
        question_analysis: Any,  # QueryParseResult
    ) -> List[StructuredRelation]:
        """基于问题解析过滤结构化事件

        Args:
            structured_events: StructuredRelation 对象列表
            question_analysis: QueryParseResult 对象

        Returns:
            过滤后的 StructuredRelation 对象列表
        """
        if not structured_events:
            return []

        # 准备关系数据用于 prompt
        relations_data = []
        for i, rel in enumerate(structured_events):
            relations_data.append({
                "index": i,
                "relation": str(rel),  # 使用 __str__ 格式: relation_type(subject, object, time_start, time_end)
                "relation_type": rel.relation_type,
                "subject": rel.subject,
                "object_entity": rel.object_entity,
                "time_start": rel.time_start,
                "time_end": rel.time_end,
                "confidence": rel.confidence,
                "source_description": rel.source_description[:100] if rel.source_description else "",
            })

        # 构建 user prompt
        user_prompt = EVENT_FILTER_USER_PROMPT.format(
            original_question=question_analysis.original_question,
            question_stem=question_analysis.question_stem,
            time_constraint_type=question_analysis.time_constraint.constraint_type.value,
            time_constraint_description=question_analysis.time_constraint.description,
            normalized_time=question_analysis.time_constraint.normalized_time or "N/A",
            event_type=question_analysis.event_type.value,
            answer_type=question_analysis.answer_type.value,
            relations_json=json.dumps(relations_data, ensure_ascii=False, indent=2),
            total_relations=len(structured_events),
        )

        messages = [
            {"role": "system", "content": EVENT_FILTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = self._call_api(messages)
            data = self._parse_json_response(content)

            # 提取相关事件的索引
            relevant_indices = set(data.get("relevant_indices", []))
            reasoning = data.get("reasoning", "")

            # 过滤事件
            filtered = [structured_events[i] for i in relevant_indices if i < len(structured_events)]

            # 打印过滤理由（如果有）
            if reasoning:
                print(f"  过滤理由: {reasoning}")

            return filtered
        except Exception as e:
            print(f"事件过滤失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果过滤失败，返回所有事件
            return structured_events


# ============================================================
# 检索语句生成 Prompt - Few-shot
# ============================================================

RETRIEVAL_SYSTEM_PROMPT = """You are an expert in generating retrieval queries for a temporal knowledge base. Given an input (either a single entity or a question stem), generate retrieval queries for three layers:

## Input Types

1. **Single Entity**: A standalone entity name (e.g., "Thierry Audel", "Barack Obama", "Google Inc.")
   - For single entities, return the entity name directly in all three queries

2. **Question Stem**: A question without time constraints (e.g., "Which team did Thierry Audel play for?")
   - For question stems, generate detailed retrieval queries as described below

## Layer Definitions

1. **Entity Query**: Generate a query to retrieve relevant entities
   - **For single entity input**: Format as "[Entity Name] [brief description from common knowledge]"
   - **For question stem**: Format as "[Entity Name] [brief description from common knowledge]"
   - Use the standardized/canonical name of the entity
   - Include a brief description based on commonly known facts

2. **Timeline Query**: Generate a query to retrieve relevant timelines
   - **For single entity input**: Return the entity name as-is
   - **For question stem**: Format as "[Entity Name]'s [aspect/career/life phase]"
   - Focus on the aspect of the entity's life/career that is relevant to the question

3. **Event Queries**: Generate event query to retrieve relevant events
   - **For single entity input**: Return the entity name in a list with one element
   - **For question stem**: Use the question stem directly as the event query
   - Do NOT generate additional variations or common knowledge based statements
   - Only output the question stem itself (or entity name for single entity input)

## Output Format
```json
{
  "entity_query": "Entity name + brief description (always include description for both single entity and question stem)",
  "timeline_query": "Timeline name + description + related entities (or just entity name for single entity)",
  "event_queries": [
    "Question stem (or entity name for single entity)"
  ]
}
```

## Few-shot Examples

### Example 1: Question Stem Input
Input: "Which team did Thierry Audel play for?"
Input Type: Question Stem

Output:
```json
{
  "entity_query": "Thierry Audel, a French professional footballer who plays as a centre back",
  "timeline_query": "Thierry Audel's football career",
  "event_queries": [
    "Which team did Thierry Audel play for?"
  ]
}
```

### Example 2: Question Stem Input
Input: "Who was Anna Karina married to?"
Input Type: Question Stem

Output:
```json
{
  "entity_query": "Anna Karina, a Danish-French film actress, director, writer, and singer",
  "timeline_query": "Anna Karina's personal life and marriages",
  "event_queries": [
    "Who was Anna Karina married to?"
  ]
}
```

### Example 3: Single Entity Input
Input: "Thierry Audel"
Input Type: Single Entity

Output:
```json
{
  "entity_query": "Thierry Audel, a French professional footballer who plays as a centre back",
  "timeline_query": "Thierry Audel",
  "event_queries": [
    "Thierry Audel"
  ]
}
```

### Example 4: Single Entity Input
Input: "Barack Obama"
Input Type: Single Entity

Output:
```json
{
  "entity_query": "Barack Obama, the 44th President of the United States",
  "timeline_query": "Barack Obama",
  "event_queries": [
    "Barack Obama"
  ]
}
```

### Example 5: Question Stem Input
Input: "What position did Carl Eric Almgren hold?"
Input Type: Question Stem

Output:
```json
{
  "entity_query": "Carl Eric Almgren, a Swedish Army officer and general",
  "timeline_query": "Carl Eric Almgren's military career",
  "event_queries": [
    "What position did Carl Eric Almgren hold?"
  ]
}
```

### Example 6: Question Stem Input
Input: "Who did Knox Cunningham serve as Parliamentary Private Secretary to?"
Input Type: Question Stem

Output:
```json
{
  "entity_query": "Knox Cunningham, a Northern Irish barrister, businessman and politician",
  "timeline_query": "Knox Cunningham's political career",
  "event_queries": [
    "Who did Knox Cunningham serve as Parliamentary Private Secretary to?"
  ]
}
```

## Important Notes
1. **Identify input type first**: Determine if the input is a single entity or a question stem
2. **For single entities**:
   - entity_query: Include a brief description from common knowledge (e.g., "Barack Obama, the 44th President of the United States")
   - timeline_query: Use the entity name as-is
   - event_queries: Use the entity name as-is
3. **For question stems**: The event query MUST be the question stem itself - do NOT add any variations
4. Use the entity's commonly known canonical name in all queries
5. The timeline query should focus on the relevant aspect (career, education, achievements, etc.)
6. Do NOT generate additional queries based on common knowledge
"""

RETRIEVAL_USER_PROMPT = """Generate retrieval queries for the following input:

Input: {input_text}

First, determine if this is a single entity name or a question stem, then generate appropriate retrieval queries.

Output in JSON format:"""


# ============================================================
# 事件结构化 Prompt
# ============================================================

EVENT_STRUCTURIZATION_SYSTEM_PROMPT = """You are an expert in temporal relation extraction. Your task is to extract structured temporal relations from event descriptions.

## Relation Types (Non-exhaustive List)

Common temporal relation types include:

- **Professional Relations**:
  - works_for(person, organization, start_time, end_time) - Employment relation
  - studies_at(person, institution, start_time, end_time) - Education relation
  - serves_as(person, position, start_time, end_time) - Position holding
  - manages(person, organization, start_time, end_time) - Management relation

- **Family Relations**:
  - married_to(person1, person2, start_time, end_time) - Marriage relation
  - child_of(person1, person2) - Parent-child relation
  - parent_of(person1, person2) - Parent relation
  - member_of_family(person, family_name) - Family membership

- **Organizational Relations**:
  - leads(person, organization, start_time, end_time) - Leadership relation
  - founded(person, organization, time) - Foundation relation
  - acquired(organization1, organization2, time) - Acquisition relation
  - merged_with(organization1, organization2, time) - Merger relation

- **Sports Relations**:
  - plays_for(athlete, team, start_time, end_time) - Team membership
  - coached_by(team, coach, start_time, end_time) - Coaching relation
  - won(person/team, award, time) - Award winning

- **Other Relations**:
  - located_in(entity, location, start_time, end_time) - Location relation
  - associated_with(entity1, entity2, start_time, end_time) - Association relation
  - recognized_as(person, achievement, time) - Recognition relation

## Extraction Rules

1. **Accuracy**: Only extract relations that are explicitly stated or strongly implied in the event description
2. **Entity Normalization**: Use canonical/standard names for entities. If pronouns are used, resolve them to actual entity names
3. **Temporal Information**:
   - Extract both time_start and time_end (if both are available)
   - For point-in-time events, use only time_start
   - For interval events, use both time_start and time_end
4. **Confidence**: Assign confidence based on the clarity of the relation:
   - 1.0: Explicit, unambiguous relation
   - 0.8: Implicit but strongly implied relation
   - 0.6: Possible but uncertain relation
5. **Multiple Relations**: If an event describes multiple relations, extract all of them
6. **Minimal Relations**: Do not create relations without sufficient evidence
7. **Relation Type Selection**: Choose the most specific and appropriate relation type for the event
8. **Time Format**: Maintain the same time format as the original event (YYYY, YYYY-MM, or YYYY-MM-DD)

## Output Format

Output a JSON object containing a "relations" array:

```json
{
  "relations": [
    {
      "relation_type": "works_for",
      "subject": "Person Name",
      "object_entity": "Organization Name",
      "time_start": "1995",
      "time_end": "2005",
      "confidence": 0.95
    },
    {
      "relation_type": "serves_as",
      "subject": "Person Name",
      "object_entity": "Position Title",
      "time_start": "2005",
      "time_end": null,
      "confidence": 0.85
    }
  ]
}
```

## Important Notes
- If temporal information is unavailable, use null for time_start and time_end
- Confidence should be a float between 0.0 and 1.0
- Use null (not empty strings) for missing optional values
- If no relations can be extracted, return an empty relations array: {"relations": []}
- Relation types should be in English (e.g., works_for), but entity names should maintain their original language
"""

EVENT_STRUCTURIZATION_USER_PROMPT = """Extract structured temporal relations from the following event:

**Event Description**: {event_description}

**Mentioned Entities**: {entity_names}

**Temporal Information**:
- Start Time: {time_start}
- End Time: {time_end}

Please identify and extract all temporal relations in this event. For each relation, specify:
1. Relation type (choose from common types or create a descriptive type)
2. Subject entity
3. Object entity
4. Time period (start and/or end)
5. Your confidence in the extraction

Output in JSON format:"""


# ============================================================
# 事件过滤 Prompt - 基于问题解析过滤结构化事件
# ============================================================

EVENT_FILTER_SYSTEM_PROMPT = """You are an expert at analyzing event relevance for question answering.

Your task: Given a question analysis and a list of structured time relations, determine which relations are relevant for answering the question.

## Relevance Criteria

Consider these factors when judging relevance:

1. **Question Stem Match**: Does the relation relate to what the question is asking about?
   - E.g., "Which team did X play for?" → relevant: plays_for(X, team, ...)
   - Irrelevant: married_to(X, person, ...), born_in(X, location, ...)

2. **Time Constraint Match**: If there's a time constraint, does the relation overlap with it?
   - Explicit time (e.g., "in 2013"): Check if relation's time range includes/overlaps 2013
   - Implicit time (e.g., "during his presidency"): Use context to judge temporal overlap
   - No time constraint: Time matching is not required

3. **Event Type Match**: Does the relation's temporal nature match the expected event type?
   - Point events: Birth, death, award, appointment → typically have only time_start
   - Interval events: Employment, position, membership → have time_start and time_end
   - Be flexible: Some relations may not perfectly match but still be relevant

4. **Answer Type Match**: Could this relation help determine the answer?
   - Entity answer: Relation should involve relevant entities as subject or object
   - Time answer: Relation should provide time information
   - Other types: Judge based on question semantics

5. **Entity Relevance**: Does the relation involve entities mentioned in the question?
   - Direct mention: Relation's subject/object matches question entity
   - Indirect connection: Relation involves related entities

## Output Format

Return a JSON object:
```json
{
  "relevant_indices": [0, 3, 5, 8],
  "reasoning": "Brief explanation of filtering logic"
}
```

- `relevant_indices`: Array of indices (0-based) of relevant relations
- `reasoning`: Brief explanation of why these relations were selected

## Important Notes
- Be inclusive rather than exclusive: When in doubt, keep the relation
- Consider that questions may be ambiguous - include potentially relevant relations
- Time matching should be flexible (allow near matches, consider incomplete time info)
- If no relations are relevant, return empty array: {"relevant_indices": [], "reasoning": "..."}
"""

EVENT_FILTER_USER_PROMPT = """## Question Analysis

**Original Question**: {original_question}
**Question Stem**: {question_stem}
**Time Constraint**: {time_constraint_type} - {time_constraint_description}
**Normalized Time**: {normalized_time}
**Event Type**: {event_type}
**Answer Type**: {answer_type}

## Structured Relations to Filter

Total relations: {total_relations}

```json
{relations_json}
```

## Task

Analyze each relation and determine which ones are relevant for answering the question based on the criteria above.
Return the indices of relevant relations in JSON format.
"""


class SearchQueryGenerator:
    """检索语句生成器

    支持两种输入类型：
    1. 单一实体：调用 LLM 生成实体描述，timeline 和 event 使用原始实体名
    2. 句子/问题：调用 LLM 生成多层检索语句
    """

    def __init__(
        self,
        config: Optional[QueryParserConfig] = None,
        token: Optional[str] = None,
        graph_store: Optional[Any] = None,
        retriever: Optional[Any] = None,
    ):
        """初始化检索语句生成器

        Args:
            config: 查询解析器配置
            token: API token
            graph_store: 图存储实例（用于检索阶段）
            retriever: 检索器实例（用于检索阶段）
        """
        self.config = config or QueryParserConfig()

        # 获取 token
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("请设置 VENUS_API_TOKEN 或 OPENAI_API_KEY 环境变量")

        # 检索相关组件（可选）
        self.graph_store = graph_store
        self.retriever = retriever

    def _is_single_entity(self, text: str) -> bool:
        """
        判断输入是否为单一实体

        规则：
        1. 不包含问号
        2. 不包含常见疑问词 (who, what, where, when, why, how, which, did, does, is, are, was, were)
        3. 不包含明显的动词短语模式
        4. 较短（通常少于10个词）

        Args:
            text: 输入文本

        Returns:
            bool: True 表示单一实体，False 表示句子/问题
        """
        text_lower = text.lower().strip()

        # 规则1: 包含问号肯定是问题
        if '?' in text:
            return False

        # 规则2: 包含疑问词
        question_words = [
            'who', 'what', 'where', 'when', 'why', 'how', 'which',
            'did', 'does', 'is', 'are', 'was', 'were', 'has', 'have',
            'can', 'could', 'would', 'should', 'will', 'shall'
        ]
        words = text_lower.split()
        if any(word in question_words for word in words):
            return False

        # 规则3: 检查是否包含明显的动词短语模式（简单规则）
        # 如果包含 "play for", "work at", "serve as" 等，通常是句子
        verb_patterns = [
            r'\bplay(?:ed)?\s+for\b',
            r'\bwork(?:ed)?\s+at\b',
            r'\bserve(?:d)?\s+as\b',
            r'\bjoined?\b',
            r'\bmarried\s+to\b',
            r'\bheld?\b',
            r'\bbecame\b',
            r'\bwon\b',
            r'\bscore(?:d)?\b',
        ]
        if any(re.search(pattern, text_lower) for pattern in verb_patterns):
            return False

        # 规则4: 词数检查（单一实体通常较短）
        if len(words) > 10:
            return False

        # 如果都不满足，认为是单一实体
        return True

    def _call_api(self, messages: List[dict]) -> str:
        """调用 LLM API"""
        payload = {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature,
            'response_format': {'type': 'json_object'},
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.config.timeout,
                )

                if response.status_code != 200:
                    raise Exception(f"API 调用失败: {response.status_code} - {response.text}")

                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                print(f"API 调用失败，重试 {attempt + 1}/{self.config.max_retries}: {e}")

        return ""

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 JSON 响应"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取 JSON
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                return json.loads(match.group(1))
            else:
                raise ValueError(f"无法解析 JSON 响应: {content[:200]}")

    def generate_retrieval_queries(self, input_text: str) -> RetrievalQueries:
        """
        生成检索语句

        Args:
            input_text: 输入文本（可以是单一实体或问题句子）

        Returns:
            RetrievalQueries: 检索语句集合
        """
        # 判断输入类型（用于日志输出）
        is_entity = self._is_single_entity(input_text)

        if is_entity:
            print(f"检测到单一实体: {input_text}，调用 LLM 生成检索语句...")
        else:
            print(f"检测到句子/问题，调用 LLM 生成检索语句...")

        # 所有情况都调用 LLM 生成
        user_prompt = RETRIEVAL_USER_PROMPT.format(input_text=input_text)

        messages = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        content = self._call_api(messages)
        data = self._parse_json_response(content)

        return RetrievalQueries(
            entity_query=data.get("entity_query", ""),
            timeline_query=data.get("timeline_query", ""),
            event_queries=data.get("event_queries", []),
        )

    def structurize_events(
        self,
        events: List[Any],
        enable_structuring: bool = True,
    ) -> List[StructuredRelation]:
        """结构化事件为时间关系

        Args:
            events: EventResult 对象列表
            enable_structuring: 是否执行结构化

        Returns:
            StructuredRelation 对象列表
        """
        if not enable_structuring or not events:
            return []

        # 懒加载：只在需要时创建 EventStructurizer
        if not hasattr(self, '_structurizer'):
            self._structurizer = EventStructurizer(
                config=self.config,
                token=self.token,
            )

        return self._structurizer.structurize_events(events)

    def filter_events(
        self,
        structured_events: List[StructuredRelation],
        question_analysis: Any,  # QueryParseResult
        enable_filtering: bool = True,
    ) -> List[StructuredRelation]:
        """基于问题解析过滤结构化事件

        Args:
            structured_events: StructuredRelation 对象列表
            question_analysis: QueryParseResult 对象
            enable_filtering: 是否执行过滤

        Returns:
            过滤后的 StructuredRelation 对象列表
        """
        if not enable_filtering or not structured_events or not question_analysis:
            return structured_events

        # 懒加载：只在需要时创建 EventFilter
        if not hasattr(self, '_filter'):
            self._filter = EventFilter(
                config=self.config,
                token=self.token,
            )

        return self._filter.filter_structured_events(structured_events, question_analysis)

    def retrieve_with_queries(
        self,
        input_text: str,
        retrieval_mode: str = "hybrid",
        entity_top_k: int = 5,
        timeline_top_k: int = 10,
        event_top_k: int = 20,
        question_analysis: Optional[Any] = None,  # QueryParseResult
        **retriever_kwargs
    ) -> RetrievalResults:
        """
        生成检索语句并执行检索

        Args:
            input_text: 输入文本（可以是单一实体或问题句子）
            retrieval_mode: 检索模式（hybrid/keyword/semantic）
            entity_top_k: 实体检索数量
            timeline_top_k: 时间线检索数量
            event_top_k: 事件检索数量
            question_analysis: 问题解析结果（用于过滤结构化事件）
            **retriever_kwargs: 传递给检索器的额外参数

        Returns:
            RetrievalResults: 包含检索语句和检索结果
        """
        # 检查是否初始化了 retriever
        if self.retriever is None:
            raise ValueError("未初始化检索器，请在创建 SearchQueryGenerator 时传入 retriever 参数")

        # 第一步：生成检索语句
        print(f"\n{'='*60}")
        print("阶段 1: 生成检索语句")
        print(f"{'='*60}")
        queries = self.generate_retrieval_queries(input_text)

        print(f"\n生成的检索语句:")
        print(f"  实体查询: {queries.entity_query}")
        print(f"  时间线查询: {queries.timeline_query}")
        print(f"  事件查询: {queries.event_queries}")

        # 第二步：执行检索
        print(f"\n{'='*60}")
        print(f"阶段 2: 执行检索（模式: {retrieval_mode}）")
        print(f"{'='*60}")

        entities = []
        timelines = []
        events = []

        # 根据检索模式调用不同的检索方法
        try:
            # 检索实体
            print(f"\n检索实体 (query: '{queries.entity_query}', top_k: {entity_top_k})...")
            if retrieval_mode == "keyword":
                entity_results = self.retriever._keyword_retriever.retrieve(
                    queries.entity_query,
                    top_k=entity_top_k,
                    target_type="entity",
                    **retriever_kwargs
                )
            elif retrieval_mode == "semantic":
                entity_results = self.retriever._semantic_retriever.retrieve(
                    queries.entity_query,
                    top_k=entity_top_k,
                    target_type="entity",
                    **retriever_kwargs
                )
            else:  # hybrid
                entity_results = self.retriever.retrieve(
                    queries.entity_query,
                    top_k=entity_top_k,
                    target_type="entity",
                    **retriever_kwargs
                )
            entities = [r for r in entity_results]
            print(f"  检索到 {len(entities)} 个实体")

            # 检索时间线
            print(f"\n检索时间线 (query: '{queries.timeline_query}', top_k: {timeline_top_k})...")
            if retrieval_mode == "keyword":
                timeline_results = self.retriever._keyword_retriever.retrieve(
                    queries.timeline_query,
                    top_k=timeline_top_k,
                    target_type="timeline",
                    **retriever_kwargs
                )
            elif retrieval_mode == "semantic":
                timeline_results = self.retriever._semantic_retriever.retrieve(
                    queries.timeline_query,
                    top_k=timeline_top_k,
                    target_type="timeline",
                    **retriever_kwargs
                )
            else:  # hybrid
                timeline_results = self.retriever.retrieve(
                    queries.timeline_query,
                    top_k=timeline_top_k,
                    target_type="timeline",
                    **retriever_kwargs
                )
            timelines = [r for r in timeline_results]
            print(f"  检索到 {len(timelines)} 条时间线")

            # 检索事件（对每个事件查询都进行检索，然后合并去重）
            print(f"\n检索事件 (queries: {len(queries.event_queries)} 条, top_k: {event_top_k})...")
            all_event_results = []
            seen_event_ids = set()

            for i, event_query in enumerate(queries.event_queries, 1):
                print(f"  事件查询 {i}/{len(queries.event_queries)}: '{event_query}'")
                if retrieval_mode == "keyword":
                    event_results = self.retriever._keyword_retriever.retrieve(
                        event_query,
                        top_k=event_top_k,
                        target_type="event",
                        **retriever_kwargs
                    )
                elif retrieval_mode == "semantic":
                    event_results = self.retriever._semantic_retriever.retrieve(
                        event_query,
                        top_k=event_top_k,
                        target_type="event",
                        **retriever_kwargs
                    )
                else:  # hybrid
                    event_results = self.retriever.retrieve(
                        event_query,
                        top_k=event_top_k,
                        target_type="event",
                        **retriever_kwargs
                    )

                # 去重合并
                for result in event_results:
                    event_id = result.event_id if hasattr(result, 'event_id') else result.node_id
                    if event_id not in seen_event_ids:
                        all_event_results.append(result)
                        seen_event_ids.add(event_id)

            events = all_event_results[:event_top_k]  # 限制总数
            print(f"  检索到 {len(events)} 个事件（去重后）")

        except Exception as e:
            print(f"检索过程出错: {e}")
            import traceback
            traceback.print_exc()

        # 第三步：构造结果
        results = RetrievalResults(
            queries=queries,
            entities=entities,
            timelines=timelines,
            events=events,
            retrieval_mode=retrieval_mode,
            entity_top_k=entity_top_k,
            timeline_top_k=timeline_top_k,
            event_top_k=event_top_k,
        )

        # 第四步：提取时间线中的事件并合并去重
        print(f"\n{'='*60}")
        print("阶段 3: 合并时间线中的事件")
        print(f"{'='*60}")
        merged_events = results.extract_and_merge_events(self.graph_store)
        print(f"  原始事件: {len(events)} 个")
        print(f"  时间线包含事件: {len(merged_events) - len(events)} 个（新增）")
        print(f"  合并后事件: {len(merged_events)} 个（去重）")

        # 第四步：事件结构化（可选）
        if self.config.enable_event_structuring:
            print(f"\n{'='*60}")
            print("阶段 4: 事件结构化")
            print(f"{'='*60}")
            structured_relations = self.structurize_events(
                merged_events,
                enable_structuring=True,
            )
            results.structured_events = structured_relations
            print(f"  原始事件: {len(merged_events)} 个")
            print(f"  抽取关系: {len(structured_relations)} 条")
        else:
            results.structured_events = []

        # 第五步：事件过滤（可选，基于问题解析）
        if question_analysis and self.config.enable_event_filtering:
            print(f"\n{'='*60}")
            print("阶段 5: 基于问题解析过滤事件")
            print(f"{'='*60}")
            filtered_structured = self.filter_events(
                results.structured_events,
                question_analysis,
                enable_filtering=True,
            )
            results.question_analysis = question_analysis
            results.filtered_structured_events = filtered_structured
            print(f"  结构化关系: {len(results.structured_events)} 条")
            print(f"  过滤后保留: {len(filtered_structured)} 条")
            print(f"  过滤掉: {len(results.structured_events) - len(filtered_structured)} 条")
        else:
            results.filtered_structured_events = []

        print(f"\n{'='*60}")
        print("检索完成")
        print(f"{'='*60}")
        print(f"  实体: {len(entities)} 个")
        print(f"  时间线: {len(timelines)} 条")
        print(f"  直接检索事件: {len(events)} 个")
        print(f"  合并后事件: {len(merged_events)} 个")
        if self.config.enable_event_structuring:
            print(f"  结构化关系: {len(results.structured_events)} 条")
        if question_analysis and self.config.enable_event_filtering:
            print(f"  过滤后关系: {len(results.filtered_structured_events)} 条")

        return results
