"""
Time Event Extraction Module

Extract temporal events from text chunks using LLM
"""

from __future__ import annotations

import os
import json
import requests
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .chunker import Chunk
from .config import ExtractorConfig, PriorEventsContextMode


class TimeType(str, Enum):
    """Time type"""
    POINT = "point"      # Point-in-time event (e.g., received an award in 1959)
    RANGE = "range"      # Time range event (e.g., served as Chief from 1961 to 1967)


@dataclass
class Entity:
    """Entity with basic information"""
    name: str              # 文本中出现的实体名
    canonical_name: str    # 标准化/完整名称（用于消歧）
    description: str       # 基于上下文的实体描述
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create from dictionary"""
        return cls(
            name=data.get("name", ""),
            canonical_name=data.get("canonical_name", data.get("name", "")),
            description=data.get("description", ""),
        )


@dataclass
class TimeEvent:
    """Time Event"""
    event_id: str                  # Unique event identifier
    
    # Event content
    original_sentence: str         # Original sentence from text
    event_description: str         # Event description
    
    # Entity information (with detailed metadata)
    entities: List[Entity]         # Entities involved with full metadata
    
    # Time information
    time_type: TimeType            # Time type
    time_start: Optional[str]      # Start time (point events only have this)
    time_end: Optional[str]        # End time (for range events)
    time_expression: str           # Original time expression
    
    # Source information
    chunk_id: str                  # Source chunk ID
    doc_id: str                    # Source document ID
    doc_title: str                 # Document title
    
    # Optional metadata
    confidence: float = 1.0        # Confidence score
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "original_sentence": self.original_sentence,
            "event_description": self.event_description,
            "entities": [e.to_dict() if isinstance(e, Entity) else e for e in self.entities],
            "time_type": self.time_type.value,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "time_expression": self.time_expression,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeEvent":
        """Create from dictionary"""
        entities_data = data.get("entities", [])
        entities = [
            Entity.from_dict(e) if isinstance(e, dict) else e 
            for e in entities_data
        ]
        
        return cls(
            event_id=data["event_id"],
            original_sentence=data["original_sentence"],
            event_description=data["event_description"],
            entities=entities,
            time_type=TimeType(data["time_type"]),
            time_start=data.get("time_start"),
            time_end=data.get("time_end"),
            time_expression=data["time_expression"],
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            doc_title=data["doc_title"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


# Time Event Extraction Prompt
EXTRACTION_SYSTEM_PROMPT = """
You are an expert in temporal event extraction. Your task is to comprehensively extract all events containing temporal information from the given text.

## Extraction Rules
1. **Completeness**: You must extract ALL events containing temporal information from the text. Do not miss any.
2. **Accuracy**: Preserve the original sentence and accurately identify time expressions.
3. **Entity Recognition**: Identify all entities involved in the event with their canonical names and descriptions.
   - **For singular entities**: Extract as usual.
   - **For plural/compound entities** (e.g., "they", "their", "the couple", "both", etc.):
     - First, identify the compound entity as a whole (e.g., "They").
     - Then, identify **each individual entity** that constitutes the compound entity.
     - Use the naming pattern: For the compound entity, use its original name (e.g., "They"). For individual parts, use the pattern "[compound_entity_name]_part" (e.g., "They_part").
     - Provide canonical names and descriptions for **each** individual entity based on context.
4. **Time Classification**:
   - point: Events occurring at a specific point in time, e.g., "In 1959 he was made a Queen's Counsel"
   - range: Events spanning a period of time, e.g., "served as Chief of the Defence Staff from 1961 to 1967"

## Output Format
Output a JSON object containing an events array:
```json
{
  "events": [
    {
      "original_sentence": "The exact sentence copied from the text",
      "event_description": "A concise description of the event",
      "entities": [
        {
          "name": "The entity name as it appears in the text (e.g., 'They')",
          "canonical_name": "The standardized/full name of the entity (e.g., 'Arnulf Øverland and Bartholine Eufemia Leganger')",
          "description": "A brief description of the entity based on context (e.g., 'Married couple who divorced')"
        },
        {
          "name": "They_part",
          "canonical_name": "Arnulf Øverland",
          "description": "One of the married couple who divorced"
        },
        {
          "name": "They_part",
          "canonical_name": "Bartholine Eufemia Leganger",
          "description": "One of the married couple who divorced"
        }
      ],
      "time_type": "point or range",
      "time_start": "Start time in standardized format (e.g., 1961, 1961-03, 1961-03-15)",
      "time_end": "End time (only for range type, null for point type)",
      "time_expression": "Original time expression from text (e.g., from 1961 to 1967)"
    }
  ]
}
```

## Important Notes
1. If there are no temporal events in the text, return an empty array: {"events": []}
2. A single sentence may contain multiple temporal events; extract them separately
3. Preserve the original form of time expressions
4. Always provide clear canonical names and descriptions to help identify entities
5. **For plural/compound entities**: You MUST identify each constituent individual entity. The `entities` array should contain the compound entity entry PLUS entries for each individual part.
6. If the context does not provide enough information to identify individual entities within a compound reference, make reasonable inferences based on available information and note any uncertainties in the description.
```
"""


EXTRACTION_USER_PROMPT = """Extract all temporal events from the following text:

## Document Title
{doc_title}

## Text Content
{content}

Please comprehensively extract all events containing temporal information and output in JSON format:
"""


# User prompt with prior extracted events context
EXTRACTION_USER_PROMPT_WITH_CONTEXT = """Extract all temporal events from the following text:

## Document Title
{doc_title}

## Prior Extracted Events (from previous text chunks)
The following events have been extracted from earlier parts of this document. Use them as context to help resolve relative time expressions (e.g., "8 years old", "two years later", "at that time") by referencing these known temporal anchors.

{prior_events}

## Text Content (current chunk to extract from)
{content}

Please comprehensively extract all events containing temporal information and output in JSON format:
"""


# Review Prompt for Multi-Round Extraction
REVIEW_SYSTEM_PROMPT = """
You are an expert reviewer for temporal event extraction. Your task is to find any temporal events that were missed in the initial extraction.

## Review Rules
1. **Completeness Check**: Carefully examine the text for ANY temporal information that wasn't captured
2. **Focus Areas**:
   - Implicit time references (e.g., "later", "afterwards", "previously")
   - Relative temporal expressions (e.g., "two years later", "the following month")
   - Events with indirect temporal context
   - Complex temporal relationships between events
3. **Avoid Duplicates**: Do NOT extract events that are already captured in the existing events list

## Extraction Rules
1. **Completeness**: You must extract ALL events containing temporal information from the text. Do not miss any.
2. **Accuracy**: Preserve the original sentence and accurately identify time expressions.
3. **Entity Recognition**: Identify all entities involved in the event with their canonical names and descriptions.
   - **For singular entities**: Extract as usual.
   - **For plural/compound entities** (e.g., "they", "their", "the couple", "both", etc.):
     - First, identify the compound entity as a whole (e.g., "They").
     - Then, identify **each individual entity** that constitutes the compound entity.
     - Use the naming pattern: For the compound entity, use its original name (e.g., "They"). For individual parts, use the pattern "[compound_entity_name]_part" (e.g., "They_part").
     - Provide canonical names and descriptions for **each** individual entity based on context.
4. **Time Classification**:
   - point: Events occurring at a specific point in time, e.g., "In 1959 he was made a Queen's Counsel"
   - range: Events spanning a period of time, e.g., "served as Chief of the Defence Staff from 1961 to 1967"

## Output Format
Output only NEW events in the same JSON format as bellow. If no additional events are found, return {"events": []}.

{
  "events": [
    {
      "original_sentence": "The exact sentence copied from the text",
      "event_description": "A concise description of the event",
      "entities": [
        {
          "name": "The entity name as it appears in the text (e.g., 'They')",
          "canonical_name": "The standardized/full name of the entity (e.g., 'Arnulf Øverland and Bartholine Eufemia Leganger')",
          "description": "A brief description of the entity based on context (e.g., 'Married couple who divorced')"
        },
        {
          "name": "They_part",
          "canonical_name": "Arnulf Øverland",
          "description": "One of the married couple who divorced"
        },
        {
          "name": "They_part",
          "canonical_name": "Bartholine Eufemia Leganger",
          "description": "One of the married couple who divorced"
        }
      ],
      "time_type": "point or range",
      "time_start": "Start time in standardized format (e.g., 1961, 1961-03, 1961-03-15)",
      "time_end": "End time (only for range type, null for point type)",
      "time_expression": "Original time expression from text (e.g., from 1961 to 1967)"
    }
  ]
}

## Important Notes
1. If there are no temporal events in the text, return an empty array: {"events": []}
2. A single sentence may contain multiple temporal events; extract them separately
3. Preserve the original form of time expressions
4. Always provide clear canonical names and descriptions to help identify entities
5. **For plural/compound entities**: You MUST identify each constituent individual entity. The `entities` array should contain the compound entity entry PLUS entries for each individual part.
6. If the context does not provide enough information to identify individual entities within a compound reference, make reasonable inferences based on available information and note any uncertainties in the description.

"""

REVIEW_USER_PROMPT = """Review the following text and the already extracted events. Find any temporal events that were missed.

## Document Title
{doc_title}

## Text Content
{content}

## Already Extracted Events
{existing_events}

Please identify any additional temporal events that were missed:"""


# Review prompt with prior extracted events context
REVIEW_USER_PROMPT_WITH_CONTEXT = """Review the following text and the already extracted events. Find any temporal events that were missed.

## Document Title
{doc_title}

## Prior Extracted Events (from previous text chunks)
The following events have been extracted from earlier parts of this document. Use them as context to help resolve relative time expressions.

{prior_events}

## Text Content (current chunk)
{content}

## Already Extracted Events (from current chunk)
{existing_events}

Please identify any additional temporal events that were missed:"""


class EventExtractor:
    """时间事件抽取器"""
    
    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        token: Optional[str] = None,
    ):
        self.config = config or ExtractorConfig()
        
        # 获取 token
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("请设置 VENUS_API_TOKEN 或 OPENAI_API_KEY 环境变量")
    
    def _call_api(self, messages: List[dict]) -> str:
        """调用 Venus API"""
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
                    timeout=180,
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
    
    def _are_events_similar(self, event1: TimeEvent, event2: TimeEvent) -> bool:
        """检查两个事件是否相似（用于去重）"""
        if (event1.original_sentence == event2.original_sentence and 
            event1.time_expression == event2.time_expression and
            event1.event_description == event2.event_description):
            return True
        return False

    def _deduplicate_events(self, events: List[TimeEvent]) -> List[TimeEvent]:
        """去除重复事件"""
        unique_events = []
        for event in events:
            is_duplicate = any(self._are_events_similar(event, existing)
                              for existing in unique_events)
            if not is_duplicate:
                unique_events.append(event)
        return unique_events

    def _format_existing_events(self, events: List[TimeEvent]) -> str:
        """格式化已提取的事件用于审查轮次"""
        if not events:
            return "None"

        formatted = []
        for event in events:
            formatted.append({
                "original_sentence": event.original_sentence,
                "event_description": event.event_description,
                "time_expression": event.time_expression,
                "time_type": event.time_type.value
            })
        return json.dumps(formatted, indent=2, ensure_ascii=False)

    def _format_prior_events(self, events: List[TimeEvent]) -> str:
        """
        格式化前置已抽取事件（精简格式）

        精简格式只保留对时间推理有用的核心信息：
        - 时间信息 (time/time_range)
        - 事件描述 (event)
        - 主要实体的标准名称 (entities)
        """
        if not events:
            return "None"

        formatted = []
        for event in events:
            # 构建精简的事件表示
            entry = {
                "event": event.event_description,
                "entities": [e.canonical_name for e in event.entities[:3]],  # 最多保留3个主要实体
            }

            # 根据时间类型格式化时间信息
            if event.time_type == TimeType.POINT:
                entry["time"] = event.time_start
            else:  # RANGE
                entry["time_range"] = [event.time_start, event.time_end]

            formatted.append(entry)

        return json.dumps(formatted, indent=2, ensure_ascii=False)

    def _get_prior_events(
        self,
        chunk_index: int,
        all_extracted_events: Dict[int, List[TimeEvent]],
    ) -> List[TimeEvent]:
        """
        根据配置获取前置已抽取事件

        Args:
            chunk_index: 当前分块索引
            all_extracted_events: 所有已抽取事件的字典 {chunk_index: events}

        Returns:
            前置事件列表
        """
        if self.config.prior_events_context_mode == PriorEventsContextMode.NONE:
            return []

        if chunk_index == 0:
            return []  # 第一个分块没有前置事件

        prior_events = []

        if self.config.prior_events_context_mode == PriorEventsContextMode.FULL:
            # 全量模式：所有前置分块的事件
            for i in range(chunk_index):
                if i in all_extracted_events:
                    prior_events.extend(all_extracted_events[i])

        elif self.config.prior_events_context_mode == PriorEventsContextMode.SLIDING_WINDOW:
            # 滑动窗口模式：第一个分块 + 当前分块前N个分块的事件
            window_size = self.config.prior_events_window_size

            # 始终包含第一个分块的事件
            if 0 in all_extracted_events:
                prior_events.extend(all_extracted_events[0])

            # 添加当前分块前 window_size 个分块的事件（不重复添加第一个分块）
            start_idx = max(1, chunk_index - window_size)
            for i in range(start_idx, chunk_index):
                if i in all_extracted_events:
                    prior_events.extend(all_extracted_events[i])

        return prior_events

    def extract_from_chunk(
        self,
        chunk: Chunk,
        prior_events: Optional[List[TimeEvent]] = None,
    ) -> List[TimeEvent]:
        """
        从单个分块中抽取时间事件（支持多轮抽取）

        Args:
            chunk: 文档分块
            prior_events: 前置已抽取事件（用于提供时间上下文）

        Returns:
            时间事件列表
        """
        all_events = []
        prior_events = prior_events or []

        # Round 1: 初始抽取
        initial_events = self._extract_initial_events(chunk, prior_events)
        all_events.extend(initial_events)
        print(f"Round 1: 提取到 {len(initial_events)} 个事件")

        # Round 2+: 审查轮次（如果启用多轮抽取）
        if self.config.enable_multi_round and len(initial_events) > 0:
            for round_num in range(2, self.config.max_rounds + 1):
                review_events = self._extract_review_events(chunk, all_events, round_num, prior_events)
                print(f"Round {round_num}: 审查发现 {len(review_events)} 个新事件")
                if not review_events:  # 如果没有找到新事件，停止
                    break
                all_events.extend(review_events)

        # 去重和重新编号
        unique_events = self._deduplicate_events(all_events)
        removed_count = len(all_events) - len(unique_events)
        if removed_count > 0:
            print(f"去重: 移除了 {removed_count} 个重复事件")

        # 重新分配事件ID
        for i, event in enumerate(unique_events):
            event.event_id = f"{chunk.chunk_id}-event-{i:04d}"

        print(f"最终提取: {len(unique_events)} 个唯一事件")
        return unique_events

    def _extract_initial_events(
        self,
        chunk: Chunk,
        prior_events: Optional[List[TimeEvent]] = None,
    ) -> List[TimeEvent]:
        """执行初始事件抽取"""
        prior_events = prior_events or []

        # 根据是否有前置事件选择提示词模板
        if prior_events:
            user_prompt = EXTRACTION_USER_PROMPT_WITH_CONTEXT.format(
                doc_title=chunk.doc_title,
                content=chunk.content,
                prior_events=self._format_prior_events(prior_events),
            )
        else:
            user_prompt = EXTRACTION_USER_PROMPT.format(
                doc_title=chunk.doc_title,
                content=chunk.content,
            )

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        return self._call_api_and_parse_events(messages, chunk)

    def _extract_review_events(
        self,
        chunk: Chunk,
        existing_events: List[TimeEvent],
        round_num: int,
        prior_events: Optional[List[TimeEvent]] = None,
    ) -> List[TimeEvent]:
        """执行审查轮次的事件抽取"""
        existing_events_str = self._format_existing_events(existing_events)
        prior_events = prior_events or []

        # 根据是否有前置事件选择提示词模板
        if prior_events:
            user_prompt = REVIEW_USER_PROMPT_WITH_CONTEXT.format(
                doc_title=chunk.doc_title,
                content=chunk.content,
                existing_events=existing_events_str,
                prior_events=self._format_prior_events(prior_events),
            )
        else:
            user_prompt = REVIEW_USER_PROMPT.format(
                doc_title=chunk.doc_title,
                content=chunk.content,
                existing_events=existing_events_str,
            )

        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 使用稍高的温度参数进行审查
        original_temp = self.config.temperature
        self.config.temperature = self.config.review_temperature

        try:
            events = self._call_api_and_parse_events(messages, chunk)
            return events
        finally:
            self.config.temperature = original_temp

    def _call_api_and_parse_events(self, messages: List[dict], chunk: Chunk) -> List[TimeEvent]:
        """调用API并解析事件（提取公共逻辑）"""
        content = self._call_api(messages)
        
        # 解析结果
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                data = json.loads(match.group(1))
            else:
                print(f"JSON 解析失败: {content[:200]}")
                return []
        
        # 转换为 TimeEvent 对象
        events = []
        events_data = data.get("events", [])
        
        for i, event_data in enumerate(events_data):
            try:
                entities_data = event_data.get("entities", [])
                entities = [Entity.from_dict(e) for e in entities_data]
                
                event = TimeEvent(
                    event_id=f"temp-{i}",  # 临时ID，后续会重新分配
                    original_sentence=event_data.get("original_sentence", ""),
                    event_description=event_data.get("event_description", ""),
                    entities=entities,
                    time_type=TimeType(event_data.get("time_type", "point")),
                    time_start=event_data.get("time_start"),
                    time_end=event_data.get("time_end"),
                    time_expression=event_data.get("time_expression", ""),
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                )
                events.append(event)
            except Exception as e:
                print(f"事件解析失败: {e}, 数据: {event_data}")
                continue
        
        return events
    
    def extract_from_chunks(
        self,
        chunks: List[Chunk],
        progress_callback: Optional[callable] = None,
    ) -> List[TimeEvent]:
        """
        从多个分块中抽取时间事件

        Args:
            chunks: 分块列表
            progress_callback: 进度回调函数 (current, total)

        Returns:
            所有时间事件列表
        """
        all_events = []
        total = len(chunks)

        # 存储每个分块的已抽取事件 {chunk_index: events}
        extracted_events_by_chunk: Dict[int, List[TimeEvent]] = {}

        for i, chunk in enumerate(chunks):
            # 获取前置事件上下文
            prior_events = self._get_prior_events(i, extracted_events_by_chunk)

            if prior_events:
                print(f"Chunk {i}: 使用 {len(prior_events)} 个前置事件作为上下文")

            # 抽取当前分块的事件
            events = self.extract_from_chunk(chunk, prior_events)

            # 存储当前分块的事件
            extracted_events_by_chunk[i] = events
            all_events.extend(events)

            if progress_callback:
                progress_callback(i + 1, total)

        return all_events
    
    def extract_from_document(
        self,
        chunks: List[Chunk],
        doc_id: str,
    ) -> Dict[str, Any]:
        """
        从文档的所有分块中抽取时间事件，并返回结构化结果

        Args:
            chunks: 该文档的所有分块
            doc_id: 文档ID

        Returns:
            包含文档信息和事件的字典
        """
        # 过滤出属于该文档的分块
        doc_chunks = [c for c in chunks if c.doc_id == doc_id]

        if not doc_chunks:
            return {"doc_id": doc_id, "chunks": [], "events": []}

        # 抽取事件（使用 extract_from_chunks 来支持前置事件上下文）
        all_events = self.extract_from_chunks(doc_chunks)

        return {
            "doc_id": doc_id,
            "doc_title": doc_chunks[0].doc_title,
            "source_idx": doc_chunks[0].source_idx,
            "num_chunks": len(doc_chunks),
            "num_events": len(all_events),
            "chunks": [c.to_dict() for c in doc_chunks],
            "events": [e.to_dict() for e in all_events],
        }
