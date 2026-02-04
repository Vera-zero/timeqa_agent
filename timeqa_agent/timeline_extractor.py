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
    
    def extract_timelines_for_entity(
        self,
        entity_canonical_name: str,
        events: List[TimeEvent],
    ) -> TimelineExtractionResult:
        """
        Extract timelines for a specific entity
        
        Args:
            entity_canonical_name: The canonical name of the entity
            events: List of events related to this entity
            
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
                timeline_id=f"timeline-{i:04d}",
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
        for entity_name, entity_event_list in entity_events.items():
            event_count = len(entity_event_list)
            if event_count == 1:
                print(f"Skipping timeline extraction for: {entity_name} (only 1 event)")
            else:
                print(f"Extracting timelines for: {entity_name} ({event_count} events)")
            result = self.extract_timelines_for_entity(entity_name, entity_event_list)
            results[entity_name] = result
        
        return results
