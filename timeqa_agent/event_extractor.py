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
from .config import ExtractorConfig


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
    
    def extract_from_chunk(self, chunk: Chunk) -> List[TimeEvent]:
        """
        从单个分块中抽取时间事件
        
        Args:
            chunk: 文档分块
            
        Returns:
            时间事件列表
        """
        # 构建 prompt
        user_prompt = EXTRACTION_USER_PROMPT.format(
            doc_title=chunk.doc_title,
            content=chunk.content,
        )
        
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        # 调用 API
        content = self._call_api(messages)
        
        # 解析结果
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取
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
            event_id = f"{chunk.chunk_id}-event-{i:04d}"
            
            try:
                # Parse entities with full metadata
                entities_data = event_data.get("entities", [])
                entities = [Entity.from_dict(e) for e in entities_data]
                
                event = TimeEvent(
                    event_id=event_id,
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
        
        for i, chunk in enumerate(chunks):
            events = self.extract_from_chunk(chunk)
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
        
        # 抽取事件
        all_events = []
        for chunk in doc_chunks:
            events = self.extract_from_chunk(chunk)
            all_events.extend(events)
        
        return {
            "doc_id": doc_id,
            "doc_title": doc_chunks[0].doc_title,
            "source_idx": doc_chunks[0].source_idx,
            "num_chunks": len(doc_chunks),
            "num_events": len(all_events),
            "chunks": [c.to_dict() for c in doc_chunks],
            "events": [e.to_dict() for e in all_events],
        }
