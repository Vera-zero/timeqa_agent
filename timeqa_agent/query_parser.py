"""
Query Parser Module

将用户问题解析为主干部分和时间约束部分。

功能:
1. 将问题分解为主干（核心问题）和时间约束（显式/隐式）
2. 识别事件类型和答案类型
"""

from __future__ import annotations

import os
import json
import re
import requests
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .config import QueryParserConfig


class TimeConstraintType(str, Enum):
    """时间约束类型"""
    EXPLICIT = "explicit"  # 显式时间约束，如 "in 2007", "from 1990 to 2000"
    IMPLICIT = "implicit"  # 隐式时间约束，如 "during the Beijing Olympics", "when he was president"
    NONE = "none"          # 无时间约束


class EventType(str, Enum):
    """事件类型"""
    POINT = "point"        # 时间点事件，如 "出生"、"获奖"、"签约"
    INTERVAL = "interval"  # 时间段事件，如 "任职"、"效力于某球队"、"居住在某地"


class AnswerType(str, Enum):
    """答案类型"""
    ENTITY = "entity"      # 实体类型，如人名、地名、组织名
    TIME = "time"          # 时间类型，如日期、年份、时间段
    NUMBER = "number"      # 数字类型，如数量、金额、排名
    BOOLEAN = "boolean"    # 布尔类型，是/否
    OTHER = "other"        # 其他类型


@dataclass
class TimeConstraint:
    """时间约束信息"""
    constraint_type: TimeConstraintType  # 约束类型
    original_expression: str             # 原始时间表达式
    normalized_time: Optional[str]       # 标准化时间（如果可以解析）
    description: str                     # 时间约束描述

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type.value,
            "original_expression": self.original_expression,
            "normalized_time": self.normalized_time,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeConstraint":
        return cls(
            constraint_type=TimeConstraintType(data.get("constraint_type", "none")),
            original_expression=data.get("original_expression", ""),
            normalized_time=data.get("normalized_time"),
            description=data.get("description", ""),
        )


@dataclass
class QueryParseResult:
    """查询解析结果"""
    original_question: str      # 原始问题
    question_stem: str          # 问题主干（去除时间约束后的核心问题）
    time_constraint: TimeConstraint  # 时间约束信息
    event_type: EventType       # 事件类型（时间点/时间段）
    answer_type: AnswerType     # 答案类型

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_question": self.original_question,
            "question_stem": self.question_stem,
            "time_constraint": self.time_constraint.to_dict(),
            "event_type": self.event_type.value,
            "answer_type": self.answer_type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryParseResult":
        return cls(
            original_question=data.get("original_question", ""),
            question_stem=data.get("question_stem", ""),
            time_constraint=TimeConstraint.from_dict(data.get("time_constraint", {})),
            event_type=EventType(data.get("event_type", "point")),
            answer_type=AnswerType(data.get("answer_type", "entity")),
        )




# ============================================================
# Stage 1: 问题解析 Prompt - 分解主干和时间约束
# ============================================================

PARSE_SYSTEM_PROMPT = """You are an expert in temporal question analysis. Your task is to decompose a question into multiple parts:
1. **Question Stem**: The core question without temporal constraints
2. **Time Constraint**: The temporal constraint (if any)
3. **Event Type**: Whether the question asks about a point-in-time event or an interval event
4. **Answer Type**: The expected type of answer

## Time Constraint Types
- **explicit**: Direct time expressions like "in 2007", "from 1990 to 2000", "before 1980", "during 2015"
- **implicit**: Indirect time references like "during the Beijing Olympics", "when he was president", "after World War II", "during his tenure as CEO"
- **none**: No temporal constraint in the question

## Event Types
- **point**: Point-in-time events that occur at a specific moment, such as "birth", "death", "won an award", "signed a contract", "was appointed"
- **interval**: Interval events that span a period of time, such as "served as", "played for", "worked at", "lived in", "held a position"

## Answer Types
- **entity**: Entity answers like person names, place names, organization names (Who, Which team, Where, What company)
- **time**: Time answers like dates, years, time periods (When, What year, How long)
- **number**: Numeric answers like quantities, amounts, rankings (How many, How much, What rank)
- **boolean**: Yes/No answers (Did, Was, Is, Has)
- **other**: Other types of answers

## Output Format
Output a JSON object with the following structure:
```json
{
  "original_question": "The original question as provided by the user",
  "question_stem": "The core question without time constraint",
  "time_constraint": {
    "constraint_type": "explicit|implicit|none",
    "original_expression": "The original time expression from the question",
    "normalized_time": "Standardized time if explicit (e.g., 2007, 1990-2000), null if implicit or none",
    "description": "Description of the time constraint"
  },
  "event_type": "point|interval",
  "answer_type": "entity|time|number|boolean|other"
}
```

## Examples

Input: "Which team did Thierry Audel play for in 2013?"
Output:
```json
{
  "original_question": "Which team did Thierry Audel play for in 2013?",
  "question_stem": "Which team did Thierry Audel play for?",
  "time_constraint": {
    "constraint_type": "explicit",
    "original_expression": "in 2013",
    "normalized_time": "2013",
    "description": "The year 2013"
  },
  "event_type": "interval",
  "answer_type": "entity"
}
```

Input: "Who was Anna Karina married to during her time at French New Wave?"
Output:
```json
{
  "original_question": "Who was Anna Karina married to during her time at French New Wave?",
  "question_stem": "Who was Anna Karina married to?",
  "time_constraint": {
    "constraint_type": "implicit",
    "original_expression": "during her time at French New Wave",
    "normalized_time": null,
    "description": "During Anna Karina's involvement in the French New Wave cinema movement (1960s)"
  },
  "event_type": "interval",
  "answer_type": "entity"
}
```

Input: "What position did Carl Eric Almgren hold from 1969 to 1976?"
Output:
```json
{
  "original_question": "What position did Carl Eric Almgren hold from 1969 to 1976?",
  "question_stem": "What position did Carl Eric Almgren hold?",
  "time_constraint": {
    "constraint_type": "explicit",
    "original_expression": "from 1969 to 1976",
    "normalized_time": "1969-1976",
    "description": "The period from 1969 to 1976"
  },
  "event_type": "interval",
  "answer_type": "entity"
}
```

Input: "When did Knox Cunningham become a Queen's Counsel?"
Output:
```json
{
  "original_question": "When did Knox Cunningham become a Queen's Counsel?",
  "question_stem": "When did Knox Cunningham become a Queen's Counsel?",
  "time_constraint": {
    "constraint_type": "none",
    "original_expression": "",
    "normalized_time": null,
    "description": "No time constraint"
  },
  "event_type": "point",
  "answer_type": "time"
}
```

Input: "How many goals did Sherif Ashraf score in the 2008-2009 season?"
Output:
```json
{
  "original_question": "How many goals did Sherif Ashraf score in the 2008-2009 season?",
  "question_stem": "How many goals did Sherif Ashraf score?",
  "time_constraint": {
    "constraint_type": "explicit",
    "original_expression": "in the 2008-2009 season",
    "normalized_time": "2008-2009",
    "description": "The 2008-2009 football season"
  },
  "event_type": "interval",
  "answer_type": "number"
}
```

Input: "Did John J. Pettus serve as Governor before the Civil War?"
Output:
```json
{
  "original_question": "Did John J. Pettus serve as Governor before the Civil War?",
  "question_stem": "Did John J. Pettus serve as Governor before the Civil War?",
  "time_constraint": {
    "constraint_type": "implicit",
    "original_expression": "before the Civil War",
    "normalized_time": null,
    "description": "Before the American Civil War (before April 1861)"
  },
  "event_type": "interval",
  "answer_type": "boolean"
}
```

## Important Notes
1. Keep the question stem as close to the original as possible, only removing the time constraint part
2. For implicit time constraints, provide a helpful description that explains the time period
3. If there are multiple time constraints, focus on the primary one
4. For event type: "played for", "worked at", "served as" are typically interval events; "won", "signed", "born", "died" are point events
5. Determine answer type based on the question word: Who/Which/Where -> entity, When -> time, How many -> number, Did/Was/Is -> boolean
"""

PARSE_USER_PROMPT = """Analyze the following question and extract the question stem and time constraint:

Question: {question}

Output in JSON format:"""

STRUCTURE_SYSTEM_PROMPT = """You are an expert at converting temporal events into symbolic temporal relations.

Your task: For each event, produce one or more structured relations that capture:
1) relation (predicate in lower_snake_case)
2) subject (main entity)
3) object (secondary entity, if any)
4) time_start / time_end (use provided values; null if missing)

Rules:
- Use ONLY the information provided in the event list. Do NOT add new facts.
- If a clear subject/object is not identifiable, set subject or object to null and use relation="related_to".
- Prefer a single relation per event unless the event clearly contains multiple distinct relations.
- Keep time fields as-is (do not re-normalize).
- Provide a human-readable symbolic string using underscores for multiword names.

Output JSON format:
{
  "relations": [
    {
      "event_id": "event-0001",
      "relation": "works_for",
      "subject": "Jaroslav Pelikan",
      "object": "Valparaiso University",
      "time_start": "Jan 1946",
      "time_end": "Jan 1949",
      "time_expression": "from Jan, 1946 to Jan, 1949",
      "entities": ["Jaroslav Pelikan", "Valparaiso University"],
      "evidence": "Original sentence or event description",
      "symbolic": "works_for(Jaroslav_Pelikan, Valparaiso_University, Jan_1946, Jan_1949)"
    }
  ]
}
"""

STRUCTURE_USER_PROMPT = """Convert the following events into symbolic temporal relations.

Question (optional):
{question}

Events (JSON):
{events_json}

Output JSON:"""


class QueryParser:
    """查询解析器

    将用户问题解析为主干和时间约束。
    """

    def __init__(
        self,
        config: Optional[QueryParserConfig] = None,
        token: Optional[str] = None,
    ):
        self.config = config or QueryParserConfig()

        # 获取 token
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("请设置 VENUS_API_TOKEN 或 OPENAI_API_KEY 环境变量")

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

    def _call_api_with_config(
        self,
        messages: List[dict],
        model: str,
        base_url: str,
        temperature: float,
        max_retries: int,
        timeout: int,
    ) -> str:
        """Call LLM API with explicit configuration."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
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
                    raise Exception(f"API call failed: {response.status_code} - {response.text}")

                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"API call failed, retry {attempt + 1}/{max_retries}: {e}")

        return ""

    def _format_events_for_structuring(self, events: List[Any]) -> List[Dict[str, Any]]:
        """Prepare event records for structuring prompt."""
        formatted = []
        for idx, event in enumerate(events):
            if isinstance(event, dict):
                data = event
            elif hasattr(event, "to_dict"):
                data = event.to_dict()
            else:
                data = {}

            event_id = data.get("event_id") or data.get("node_id") or f"event-{idx:04d}"

            entity_names = data.get("entity_names")
            if not entity_names:
                entities = data.get("entities") or []
                names = []
                if isinstance(entities, list):
                    for ent in entities:
                        if isinstance(ent, dict):
                            name = ent.get("canonical_name") or ent.get("name")
                        else:
                            name = getattr(ent, "canonical_name", None) or getattr(ent, "name", None)
                        if name:
                            names.append(name)
                entity_names = names

            if not isinstance(entity_names, list):
                entity_names = [entity_names] if entity_names else []

            formatted.append({
                "event_id": event_id,
                "event_description": data.get("event_description", "") or data.get("description", ""),
                "time_start": data.get("time_start"),
                "time_end": data.get("time_end"),
                "time_expression": data.get("time_expression", ""),
                "entity_names": entity_names,
                "original_sentence": data.get("original_sentence", ""),
            })

        return formatted

    def structure_events(
        self,
        events: List[Any],
        question: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Structure merged events into symbolic relations using LLM.

        Args:
            events: event list (EventResult or dict)
            question: optional original question/input text
            batch_size: optional override for batch size

        Returns:
            List of structured relations
        """
        if not events:
            return []

        if not self.config.enable_event_structuring:
            return []

        event_records = self._format_events_for_structuring(events)
        if not event_records:
            return []

        effective_batch_size = batch_size or self.config.structuring_batch_size
        if not effective_batch_size or effective_batch_size <= 0:
            effective_batch_size = len(event_records)

        model = self.config.structuring_model or self.config.model
        base_url = self.config.structuring_base_url or self.config.base_url
        temperature = self.config.structuring_temperature
        max_retries = self.config.structuring_max_retries
        timeout = self.config.structuring_timeout

        all_relations: List[Dict[str, Any]] = []

        for start in range(0, len(event_records), effective_batch_size):
            batch = event_records[start:start + effective_batch_size]
            events_json = json.dumps(batch, ensure_ascii=False, indent=2)
            user_prompt = STRUCTURE_USER_PROMPT.format(
                question=question or "",
                events_json=events_json,
            )
            messages = [
                {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            content = self._call_api_with_config(
                messages=messages,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
            )
            data = self._parse_json_response(content)
            relations = data.get("relations", [])
            if isinstance(relations, list):
                all_relations.extend(relations)

        return all_relations

    def parse_question(self, question: str) -> QueryParseResult:
        """
        解析问题，提取主干、时间约束、事件类型和答案类型

        Args:
            question: 原始问题

        Returns:
            QueryParseResult: 解析结果
        """
        user_prompt = PARSE_USER_PROMPT.format(question=question)

        messages = [
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        content = self._call_api(messages)
        data = self._parse_json_response(content)

        # 构建时间约束对象
        tc_data = data.get("time_constraint", {})
        time_constraint = TimeConstraint(
            constraint_type=TimeConstraintType(tc_data.get("constraint_type", "none")),
            original_expression=tc_data.get("original_expression", ""),
            normalized_time=tc_data.get("normalized_time"),
            description=tc_data.get("description", ""),
        )

        # 解析事件类型
        event_type_str = data.get("event_type", "point")
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.POINT

        # 解析答案类型
        answer_type_str = data.get("answer_type", "entity")
        try:
            answer_type = AnswerType(answer_type_str)
        except ValueError:
            answer_type = AnswerType.ENTITY

        return QueryParseResult(
            original_question=data["original_question"],
            question_stem=data["question_stem"],
            time_constraint=time_constraint,
            event_type=event_type,
            answer_type=answer_type,
        )
