"""
Timeline Extraction Module

Extract timelines from events associated with the same entity
"""

from __future__ import annotations

import os
import json
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .config import TimelineConfig
from .event_extractor import TimeEvent, Entity


@dataclass
class Timeline:
    """A timeline representing a sequence of related events"""
    timeline_id: str                    # Unique timeline identifier
    entity_canonical_name: str          # Associated entity's canonical name
    timeline_name: str                  # Timeline name (e.g., "Legal Career", "Political Career")
    description: str                    # Timeline description
    event_ids: List[str]                # Event IDs in chronological order
    time_span_start: Optional[str]      # Start time of the timeline
    time_span_end: Optional[str]        # End time of the timeline
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeline_id": self.timeline_id,
            "entity_canonical_name": self.entity_canonical_name,
            "timeline_name": self.timeline_name,
            "description": self.description,
            "event_ids": self.event_ids,
            "time_span_start": self.time_span_start,
            "time_span_end": self.time_span_end,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Timeline":
        return cls(
            timeline_id=data["timeline_id"],
            entity_canonical_name=data["entity_canonical_name"],
            timeline_name=data["timeline_name"],
            description=data.get("description", ""),
            event_ids=data.get("event_ids", []),
            time_span_start=data.get("time_span_start"),
            time_span_end=data.get("time_span_end"),
        )


@dataclass
class StandaloneEvent:
    """An event that doesn't belong to any timeline"""
    event_id: str
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "reason": self.reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandaloneEvent":
        return cls(
            event_id=data.get("event_id", ""),
            reason=data.get("reason", ""),
        )


@dataclass
class TimelineExtractionResult:
    """Result of timeline extraction for an entity"""
    entity_canonical_name: str
    timelines: List[Timeline]
    standalone_events: List[StandaloneEvent]
    event_timeline_map: Dict[str, str]  # event_id -> timeline_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_canonical_name": self.entity_canonical_name,
            "num_timelines": len(self.timelines),
            "num_standalone_events": len(self.standalone_events),
            "timelines": [t.to_dict() for t in self.timelines],
            "standalone_events": [e.to_dict() for e in self.standalone_events],
            "event_timeline_map": self.event_timeline_map,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineExtractionResult":
        return cls(
            entity_canonical_name=data.get("entity_canonical_name", ""),
            timelines=[Timeline.from_dict(t) for t in data.get("timelines", [])],
            standalone_events=[StandaloneEvent.from_dict(e) for e in data.get("standalone_events", [])],
            event_timeline_map=data.get("event_timeline_map", {}),
        )


# Timeline Extraction Prompt
TIMELINE_SYSTEM_PROMPT = """You are an expert in temporal event analysis. Your task is to organize events related to a specific entity into meaningful timelines.

## Timeline Types
Common timeline types include (but are not limited to):
- Life Cycle: birth, death, major life milestones
- Education: educational experiences and academic achievements
- Legal Career: legal profession related events
- Political Career: political positions and activities
- Military Career: military service and ranks
- Business Career: business positions and ventures
- Family: marriage, children, family events
- Achievements: awards, honors, recognitions
- Other: any other coherent sequence of events

## Rules
1. Each event can only belong to ONE timeline, or be marked as standalone
2. Events in a timeline should have logical continuity or causal relationship
3. Summary or overview events should be marked as standalone
4. Events within a timeline should be ordered chronologically
5. A timeline should have at least 2 events (single events should be standalone)
6. Timeline names should be concise and descriptive (e.g., "Legal Career", "Political Career")

## Output Format
Output a JSON object with the following structure:
```json
{
  "timelines": [
    {
      "timeline_name": "Legal Career",
      "description": "Career development from bar admission to Queen's Counsel",
      "event_ids": ["event-001", "event-002", "event-003"]
    }
  ],
  "standalone_events": [
    {
      "event_id": "event-000",
      "reason": "Summary event covering entire lifespan"
    }
  ]
}
```
"""

TIMELINE_USER_PROMPT = """Analyze the following events related to entity "{entity_name}" and organize them into timelines.

## Events
{events_json}

Please identify which events belong to the same timeline and which are standalone events. Output in JSON format:"""

TIMELINE_SYSTEM_PROMPT_ITERATIVE = """You are an expert in temporal event analysis. Your task is to organize events related to a specific entity into meaningful timelines.

## Context
You are processing events in batches. Some timelines and standalone events have already been identified from previous batches. Your job is to:
1. Determine if new events belong to existing timelines (use the exact timeline_id provided)
2. Consider if new events can combine with existing standalone events to form new timelines
3. Create new timelines if events don't fit existing ones (use timeline_id="new")
4. Mark events as standalone if they don't belong to any timeline

## Timeline Types
Common timeline types include (but are not limited to):
- Life Cycle: birth, death, major life milestones
- Education: educational experiences and academic achievements
- Legal Career: legal profession related events
- Political Career: political positions and activities
- Military Career: military service and ranks
- Business Career: business positions and ventures
- Family: marriage, children, family events
- Achievements: awards, honors, recognitions
- Other: any other coherent sequence of events

## Rules
1. Each event can only belong to ONE timeline, or be marked as standalone
2. Events in a timeline should have logical continuity or causal relationship
3. Summary or overview events should be marked as standalone
4. Events within a timeline should be ordered chronologically
5. A timeline should have at least 2 events (single events should be standalone)
6. **NEW**: If new events from the current batch can form a meaningful timeline with existing standalone events, create a new timeline and include the event_ids of both the existing standalone events and the new events
7. Timeline names should be concise and descriptive (e.g., "Legal Career", "Political Career")
8. **IMPORTANT**: When assigning to existing timelines, use the exact timeline_id from the existing timelines context
9. **IMPORTANT**: When creating new timelines, use timeline_id="new" and provide timeline_name and description


## Output Format
Output a JSON object with the following structure:
```json
{
  "timelines": [
    {
      "timeline_id": "timeline-0001",  // Use existing ID from context
      "event_ids": ["event-001", "event-002"]
      // Do NOT include timeline_name or description for existing timelines
    },
    {
      "timeline_id": "new",  // For new timeline
      "timeline_name": "Legal Career",  // Required for new timelines
      "description": "Events related to legal profession",  // Required for new timelines
      "event_ids": ["event-003", "event-004"]
    }
  ],
  "standalone_events": [
    {
      "event_id": "event-000",
      "reason": "Summary event covering entire lifespan"
    }
  ]
}
```
"""

TIMELINE_USER_PROMPT_ITERATIVE = """Analyze the following events related to entity "{entity_name}" and organize them into timelines.

## Existing Timelines (from previous batches)
{existing_timelines}

## Existing Standalone Events (from previous batches)
{existing_standalone_events}

## New Events to Classify
{events_json}

Please determine:
1. Which events belong to existing timelines (reference by timeline_id)
2. Whether new events can combine with existing standalone events to form new timelines (include both in event_ids)
3. Which events should form new timelines with only events from current batch (use timeline_id="new" with timeline_name and description)
4. Which are standalone events

Output in JSON format:"""



class TimelineExtractor:
    """Timeline extractor using LLM"""
    
    def __init__(
        self,
        config: Optional[TimelineConfig] = None,
        token: Optional[str] = None,
    ):
        self.config = config or TimelineConfig()
        
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("Please set VENUS_API_TOKEN or OPENAI_API_KEY environment variable")
    
    def _call_api(self, messages: List[dict]) -> str:
        """Call LLM API"""
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
                    raise Exception(f"API call failed: {response.status_code} - {response.text}")
                
                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                print(f"API call failed, retry {attempt + 1}/{self.config.max_retries}: {e}")
        
        return ""
    
    def _group_events_by_entity(
        self,
        events: List[TimeEvent],
    ) -> Dict[str, List[TimeEvent]]:
        """Group events by entity canonical name"""
        entity_events = defaultdict(list)
        
        for event in events:
            # Get all entities in this event
            for entity in event.entities:
                canonical_name = entity.canonical_name
                if canonical_name:
                    entity_events[canonical_name].append(event)
        
        return dict(entity_events)
    
    def _format_events_for_prompt(self, events: List[TimeEvent]) -> str:
        """Format events as JSON for the prompt"""
        events_data = []
        for event in events:
            events_data.append({
                "event_id": event.event_id,
                "event_description": event.event_description,
                "original_sentence": event.original_sentence,
                "time_type": event.time_type.value,
                "time_start": event.time_start,
                "time_end": event.time_end,
                "time_expression": event.time_expression,
            })
        return json.dumps(events_data, indent=2, ensure_ascii=False)

    def _format_existing_timelines(
        self,
        timelines: List[Timeline],
        events: List[TimeEvent],
    ) -> str:
        """
        格式化已有时间线为紧凑的 JSON 字符串供 LLM 使用

        Args:
            timelines: 已有时间线列表
            events: 所有事件(用于查找详情)

        Returns:
            已有时间线的 JSON 字符串
        """
        # 限制时间线数量以避免上下文过大
        timelines_to_include = timelines[:self.config.max_context_timelines]

        event_map = {e.event_id: e for e in events}
        formatted = []

        for tl in timelines_to_include:
            # 获取示例事件用于上下文
            sample_event_ids = tl.event_ids[:3]  # 前3个事件

            formatted.append({
                "timeline_id": tl.timeline_id,
                "timeline_name": tl.timeline_name,
                "description": tl.description,
                "time_span": f"{tl.time_span_start or '?'} - {tl.time_span_end or '?'}",
                "event_count": len(tl.event_ids),
                "sample_event_ids": sample_event_ids,
            })

        return json.dumps(formatted, indent=2, ensure_ascii=False)

    def _format_existing_standalone_events(
        self,
        standalone_events: List[StandaloneEvent],
        events: List[TimeEvent],
    ) -> str:
        """
        格式化已有独立事件为 JSON 字符串供 LLM 使用

        Args:
            standalone_events: 已有独立事件列表
            events: 所有事件(用于查找详情)

        Returns:
            已有独立事件的 JSON 字符串
        """
        if not standalone_events:
            return "[]"

        event_map = {e.event_id: e for e in events}
        formatted = []

        for se in standalone_events:
            event = event_map.get(se.event_id)
            if event:
                formatted.append({
                    "event_id": se.event_id,
                    "event_description": event.event_description,
                    "time_start": event.time_start,
                    "time_end": event.time_end,
                    "reason": se.reason,
                })

        return json.dumps(formatted, indent=2, ensure_ascii=False)

    def _compute_time_span(
        self,
        events: List[TimeEvent],
        event_ids: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Compute the time span of a timeline"""
        event_map = {e.event_id: e for e in events}
        
        starts = []
        ends = []
        
        for event_id in event_ids:
            if event_id not in event_map:
                continue
            event = event_map[event_id]
            
            if event.time_start:
                starts.append(event.time_start)
            if event.time_end:
                ends.append(event.time_end)
            elif event.time_start:
                ends.append(event.time_start)
        
        time_start = min(starts) if starts else None
        time_end = max(ends) if ends else None
        
        return time_start, time_end

    def _split_events_into_batches(
        self,
        events: List[TimeEvent],
    ) -> List[List[TimeEvent]]:
        """
        将事件列表分割成批次用于迭代处理

        Args:
            events: 所有事件

        Returns:
            事件批次列表
        """
        # 可选按时间排序
        if self.config.sort_events_by_time:
            events = sorted(events, key=lambda e: e.time_start or "")

        # 按批次大小分批
        batch_size = self.config.iterative_batch_size
        batches = [events[i:i+batch_size] for i in range(0, len(events), batch_size)]
        return batches

    def _merge_batch_results(
        self,
        existing_timelines: List[Timeline],
        existing_standalone: List[StandaloneEvent],
        batch_result: Dict[str, Any],
        events: List[TimeEvent],
        entity_canonical_name: str,
        timeline_id_offset: int,
    ) -> Tuple[List[Timeline], List[StandaloneEvent], int]:
        """
        合并当前批次结果到已有时间线

        Args:
            existing_timelines: 之前批次的时间线
            existing_standalone: 之前批次的独立事件
            batch_result: 当前批次的 LLM 输出
            events: 所有事件(用于计算时间跨度)
            entity_canonical_name: 实体名称
            timeline_id_offset: 当前时间线 ID 计数器

        Returns:
            (更新后的时间线列表, 更新后的独立事件列表, 新的 timeline_id_offset)
        """
        # 创建时间线查找映射
        timeline_map = {tl.timeline_id: tl for tl in existing_timelines}

        # 收集所有被分配到时间线的事件 ID（用于从独立事件中移除）
        events_in_timelines = set()

        # 处理批次中的时间线
        for tl_data in batch_result.get("timelines", []):
            timeline_id = tl_data.get("timeline_id")
            event_ids = tl_data.get("event_ids", [])

            if not event_ids:
                continue

            # 记录这些事件已被分配到时间线
            events_in_timelines.update(event_ids)

            # 检查是分配到已有时间线还是创建新时间线
            if timeline_id and timeline_id in timeline_map:
                # 分配到已有时间线
                existing_tl = timeline_map[timeline_id]
                existing_tl.event_ids.extend(event_ids)
                # 重新计算时间跨度
                time_start, time_end = self._compute_time_span(events, existing_tl.event_ids)
                existing_tl.time_span_start = time_start
                existing_tl.time_span_end = time_end
            else:
                # 创建新时间线(timeline_id 为 "new" 或不存在的 ID)
                time_start, time_end = self._compute_time_span(events, event_ids)
                new_timeline = Timeline(
                    timeline_id=f"timeline-{timeline_id_offset:04d}",
                    entity_canonical_name=entity_canonical_name,
                    timeline_name=tl_data.get("timeline_name", f"Timeline {timeline_id_offset}"),
                    description=tl_data.get("description", ""),
                    event_ids=event_ids,
                    time_span_start=time_start,
                    time_span_end=time_end,
                )
                existing_timelines.append(new_timeline)
                timeline_map[new_timeline.timeline_id] = new_timeline
                timeline_id_offset += 1

        # 从已有独立事件中移除那些已被纳入时间线的事件
        existing_standalone = [
            se for se in existing_standalone
            if se.event_id not in events_in_timelines
        ]

        # 处理批次中的新独立事件
        for se_data in batch_result.get("standalone_events", []):
            existing_standalone.append(StandaloneEvent(
                event_id=se_data.get("event_id", ""),
                reason=se_data.get("reason", ""),
            ))

        return existing_timelines, existing_standalone, timeline_id_offset

    def extract_timelines_for_entity(
        self,
        entity_canonical_name: str,
        events: List[TimeEvent],
        timeline_id_offset: int = 0,
    ) -> TimelineExtractionResult:
        """
        Extract timelines for a specific entity

        Args:
            entity_canonical_name: The canonical name of the entity
            events: List of events related to this entity
            timeline_id_offset: Starting offset for timeline IDs to ensure global uniqueness

        Returns:
            TimelineExtractionResult
        """
        if not events:
            return TimelineExtractionResult(
                entity_canonical_name=entity_canonical_name,
                timelines=[],
                standalone_events=[],
                event_timeline_map={},
            )
        
        # Skip LLM call for single event - directly mark as standalone
        if len(events) == 1:
            event = events[0]
            return TimelineExtractionResult(
                entity_canonical_name=entity_canonical_name,
                timelines=[],
                standalone_events=[StandaloneEvent(
                    event_id=event.event_id,
                    reason="Single event, no timeline needed",
                )],
                event_timeline_map={},
            )

        # 路由：如果启用迭代模式且事件数超过批次大小，使用迭代抽取
        if self.config.enable_iterative and len(events) > self.config.iterative_batch_size:
            return self._extract_timelines_iterative(
                entity_canonical_name,
                events,
                timeline_id_offset,
            )

        # 单次抽取模式（默认行为）
        # Format events for prompt
        events_json = self._format_events_for_prompt(events)
        
        # Build prompt
        user_prompt = TIMELINE_USER_PROMPT.format(
            entity_name=entity_canonical_name,
            events_json=events_json,
        )
        
        messages = [
            {"role": "system", "content": TIMELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        # Call API
        content = self._call_api(messages)
        
        # Parse result
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                data = json.loads(match.group(1))
            else:
                print(f"JSON parse failed: {content[:200]}")
                return TimelineExtractionResult(
                    entity_canonical_name=entity_canonical_name,
                    timelines=[],
                    standalone_events=[],
                    event_timeline_map={},
                )
        
        # Build timelines
        timelines = []
        event_timeline_map = {}
        
        for i, tl_data in enumerate(data.get("timelines", [])):
            event_ids = tl_data.get("event_ids", [])
            
            # Compute time span
            time_start, time_end = self._compute_time_span(events, event_ids)
            
            timeline = Timeline(
                timeline_id=f"timeline-{timeline_id_offset + i:04d}",
                entity_canonical_name=entity_canonical_name,
                timeline_name=tl_data.get("timeline_name", f"Timeline {i+1}"),
                description=tl_data.get("description", ""),
                event_ids=event_ids,
                time_span_start=time_start,
                time_span_end=time_end,
            )
            timelines.append(timeline)
            
            # Update mapping
            for event_id in event_ids:
                event_timeline_map[event_id] = timeline.timeline_id
        
        # Build standalone events
        standalone_events = []
        for se_data in data.get("standalone_events", []):
            standalone_events.append(StandaloneEvent(
                event_id=se_data.get("event_id", ""),
                reason=se_data.get("reason", ""),
            ))
        
        return TimelineExtractionResult(
            entity_canonical_name=entity_canonical_name,
            timelines=timelines,
            standalone_events=standalone_events,
            event_timeline_map=event_timeline_map,
        )

    def _extract_timelines_iterative(
        self,
        entity_canonical_name: str,
        events: List[TimeEvent],
        timeline_id_offset: int = 0,
    ) -> TimelineExtractionResult:
        """
        迭代式时间线抽取 - 分批处理事件并利用已有时间线作为上下文

        Args:
            entity_canonical_name: 实体的规范名称
            events: 与该实体相关的事件列表
            timeline_id_offset: 时间线 ID 起始偏移量以确保全局唯一性

        Returns:
            TimelineExtractionResult
        """
        # 分批
        batches = self._split_events_into_batches(events)

        print(f"迭代式抽取: 处理 {len(events)} 个事件,分为 {len(batches)} 批")

        # 初始化结果容器
        timelines = []
        standalone_events = []
        event_timeline_map = {}
        current_offset = timeline_id_offset

        # 逐批处理
        for batch_idx, batch in enumerate(batches):
            print(f"处理批次 {batch_idx + 1}/{len(batches)}: {len(batch)} 个事件")

            # 格式化当前批次的事件
            events_json = self._format_events_for_prompt(batch)

            # 根据是否有已有时间线或独立事件选择不同的提示词
            if (timelines or standalone_events) and self.config.include_timeline_context:
                timelines_context = self._format_existing_timelines(timelines, events)
                standalone_context = self._format_existing_standalone_events(standalone_events, events)
                user_prompt = TIMELINE_USER_PROMPT_ITERATIVE.format(
                    entity_name=entity_canonical_name,
                    events_json=events_json,
                    existing_timelines=timelines_context,
                    existing_standalone_events=standalone_context,
                )
                system_prompt = TIMELINE_SYSTEM_PROMPT_ITERATIVE
            else:
                user_prompt = TIMELINE_USER_PROMPT.format(
                    entity_name=entity_canonical_name,
                    events_json=events_json,
                )
                system_prompt = TIMELINE_SYSTEM_PROMPT

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # 调用 API
            content = self._call_api(messages)

            # 解析结果
            try:
                batch_result = json.loads(content)
            except json.JSONDecodeError:
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if match:
                    batch_result = json.loads(match.group(1))
                else:
                    print(f"批次 {batch_idx} JSON 解析失败: {content[:200]}")
                    continue

            # 合并结果
            timelines, standalone_events, current_offset = self._merge_batch_results(
                timelines,
                standalone_events,
                batch_result,
                events,
                entity_canonical_name,
                current_offset,
            )

        # 构建 event_timeline_map
        for tl in timelines:
            for event_id in tl.event_ids:
                event_timeline_map[event_id] = tl.timeline_id

        return TimelineExtractionResult(
            entity_canonical_name=entity_canonical_name,
            timelines=timelines,
            standalone_events=standalone_events,
            event_timeline_map=event_timeline_map,
        )

    def extract_timelines(
        self,
        events: List[TimeEvent],
        target_entities: Optional[List[str]] = None,
    ) -> Dict[str, TimelineExtractionResult]:
        """
        Extract timelines for all entities in the events
        
        Args:
            events: List of all events
            target_entities: Optional list of entity canonical names to process.
                           If None, process all entities.
            
        Returns:
            Dict[entity_canonical_name, TimelineExtractionResult]
        """
        # Group events by entity
        entity_events = self._group_events_by_entity(events)
        
        # Filter entities if specified
        if target_entities:
            entity_events = {
                k: v for k, v in entity_events.items()
                if k in target_entities
            }
        
        # Extract timelines for each entity
        results = {}
        global_timeline_offset = 0  # 全局时间线ID计数器
        for entity_name, entity_event_list in entity_events.items():
            event_count = len(entity_event_list)
            if event_count == 1:
                print(f"Skipping timeline extraction for: {entity_name} (only 1 event)")
            else:
                print(f"Extracting timelines for: {entity_name} ({event_count} events)")
            result = self.extract_timelines_for_entity(
                entity_name,
                entity_event_list,
                timeline_id_offset=global_timeline_offset,
            )
            results[entity_name] = result
            # 更新全局计数器
            global_timeline_offset += len(result.timelines)
        
        return results
