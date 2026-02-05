"""
Query Parser Module

将用户问题解析为主干部分和时间约束部分，并生成针对实体、事件、时间线的检索语句。

功能:
1. 将问题分解为主干（核心问题）和时间约束（显式/隐式）
2. 基于主干生成多层检索语句（实体、时间线、事件）
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
    DURATION = "duration"  # 时间段事件，如 "任职"、"效力于某球队"、"居住在某地"


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
class QueryParserOutput:
    """查询解析器完整输出"""
    parse_result: QueryParseResult    # 解析结果
    retrieval_queries: RetrievalQueries  # 检索语句

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parse_result": self.parse_result.to_dict(),
            "retrieval_queries": self.retrieval_queries.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryParserOutput":
        return cls(
            parse_result=QueryParseResult.from_dict(data.get("parse_result", {})),
            retrieval_queries=RetrievalQueries.from_dict(data.get("retrieval_queries", {})),
        )


# ============================================================
# Stage 1: 问题解析 Prompt - 分解主干和时间约束
# ============================================================

PARSE_SYSTEM_PROMPT = """You are an expert in temporal question analysis. Your task is to decompose a question into multiple parts:
1. **Question Stem**: The core question without temporal constraints
2. **Time Constraint**: The temporal constraint (if any)
3. **Event Type**: Whether the question asks about a point-in-time event or a duration event
4. **Answer Type**: The expected type of answer

## Time Constraint Types
- **explicit**: Direct time expressions like "in 2007", "from 1990 to 2000", "before 1980", "during 2015"
- **implicit**: Indirect time references like "during the Beijing Olympics", "when he was president", "after World War II", "during his tenure as CEO"
- **none**: No temporal constraint in the question

## Event Types
- **point**: Point-in-time events that occur at a specific moment, such as "birth", "death", "won an award", "signed a contract", "was appointed"
- **duration**: Duration events that span a period of time, such as "served as", "played for", "worked at", "lived in", "held a position"

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
  "question_stem": "The core question without time constraint",
  "time_constraint": {
    "constraint_type": "explicit|implicit|none",
    "original_expression": "The original time expression from the question",
    "normalized_time": "Standardized time if explicit (e.g., 2007, 1990-2000), null if implicit or none",
    "description": "Description of the time constraint"
  },
  "event_type": "point|duration",
  "answer_type": "entity|time|number|boolean|other"
}
```

## Examples

Input: "Which team did Thierry Audel play for in 2013?"
Output:
```json
{
  "question_stem": "Which team did Thierry Audel play for?",
  "time_constraint": {
    "constraint_type": "explicit",
    "original_expression": "in 2013",
    "normalized_time": "2013",
    "description": "The year 2013"
  },
  "event_type": "duration",
  "answer_type": "entity"
}
```

Input: "Who was Anna Karina married to during her time at French New Wave?"
Output:
```json
{
  "question_stem": "Who was Anna Karina married to?",
  "time_constraint": {
    "constraint_type": "implicit",
    "original_expression": "during her time at French New Wave",
    "normalized_time": null,
    "description": "During Anna Karina's involvement in the French New Wave cinema movement (1960s)"
  },
  "event_type": "duration",
  "answer_type": "entity"
}
```

Input: "What position did Carl Eric Almgren hold from 1969 to 1976?"
Output:
```json
{
  "question_stem": "What position did Carl Eric Almgren hold?",
  "time_constraint": {
    "constraint_type": "explicit",
    "original_expression": "from 1969 to 1976",
    "normalized_time": "1969-1976",
    "description": "The period from 1969 to 1976"
  },
  "event_type": "duration",
  "answer_type": "entity"
}
```

Input: "When did Knox Cunningham become a Queen's Counsel?"
Output:
```json
{
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
  "question_stem": "How many goals did Sherif Ashraf score?",
  "time_constraint": {
    "constraint_type": "explicit",
    "original_expression": "in the 2008-2009 season",
    "normalized_time": "2008-2009",
    "description": "The 2008-2009 football season"
  },
  "event_type": "duration",
  "answer_type": "number"
}
```

Input: "Did John J. Pettus serve as Governor before the Civil War?"
Output:
```json
{
  "question_stem": "Did John J. Pettus serve as Governor before the Civil War?",
  "time_constraint": {
    "constraint_type": "implicit",
    "original_expression": "before the Civil War",
    "normalized_time": null,
    "description": "Before the American Civil War (before April 1861)"
  },
  "event_type": "duration",
  "answer_type": "boolean"
}
```

## Important Notes
1. Keep the question stem as close to the original as possible, only removing the time constraint part
2. For implicit time constraints, provide a helpful description that explains the time period
3. If there are multiple time constraints, focus on the primary one
4. For event type: "played for", "worked at", "served as" are typically duration events; "won", "signed", "born", "died" are point events
5. Determine answer type based on the question word: Who/Which/Where -> entity, When -> time, How many -> number, Did/Was/Is -> boolean
"""

PARSE_USER_PROMPT = """Analyze the following question and extract the question stem and time constraint:

Question: {question}

Output in JSON format:"""


# ============================================================
# Stage 2: 检索语句生成 Prompt - Few-shot
# ============================================================

RETRIEVAL_SYSTEM_PROMPT = """You are an expert in generating retrieval queries for a temporal knowledge base. Given a question stem (core question without time constraints), generate retrieval queries for three layers:

## Layer Definitions

1. **Entity Query**: Generate a query to retrieve relevant entities
   - Format: "[Entity Name] [brief description from common knowledge]"
   - Use the standardized/canonical name of the entity
   - Include a brief description based on commonly known facts

2. **Timeline Query**: Generate a query to retrieve relevant timelines
   - Format: "[Entity Name]'s [aspect/career/life phase]"
   - Focus on the aspect of the entity's life/career that is relevant to the question

3. **Event Queries**: Generate multiple queries to retrieve relevant events
   - Convert the question stem into declarative statements
   - Generate multiple variations based on common knowledge
   - Each query should be a statement describing a potential event/fact

## Output Format
```json
{
  "entity_query": "Entity name + brief description",
  "timeline_query": "Timeline name + description + related entities",
  "event_queries": [
    "Declarative statement 1",
    "Declarative statement 2",
    ...
  ]
}
```

## Few-shot Examples

### Example 1
Question Stem: "Which team did Thierry Audel play for?"

Output:
```json
{
  "entity_query": "Thierry Audel, a French professional footballer who plays as a centre back",
  "timeline_query": "Thierry Audel's football career",
  "event_queries": [
    "Thierry Audel played for Macclesfield Town",
    "Thierry Audel played for Crewe Alexandra",
    "Thierry Audel played for Lincoln City",
    "Thierry Audel joined a football club",
    "Thierry Audel transferred to a new team"
  ]
}
```

### Example 2
Question Stem: "Who was Anna Karina married to?"

Output:
```json
{
  "entity_query": "Anna Karina, a Danish-French film actress, director, writer, and singer",
  "timeline_query": "Anna Karina's personal life and marriages",
  "event_queries": [
    "Anna Karina married Jean-Luc Godard",
    "Anna Karina married Pierre Fabre",
    "Anna Karina married Daniel Duval",
    "Anna Karina married Dennis Berry",
    "Anna Karina's marriage and divorce"
  ]
}
```

### Example 3
Question Stem: "What position did Carl Eric Almgren hold?"

Output:
```json
{
  "entity_query": "Carl Eric Almgren, a Swedish Army officer and general",
  "timeline_query": "Carl Eric Almgren's military career",
  "event_queries": [
    "Carl Eric Almgren served as Chief of the Army",
    "Carl Eric Almgren served as Chief of the Defence Staff",
    "Carl Eric Almgren served as military commander of the Eastern Military District",
    "Carl Eric Almgren held a position in the Swedish Army",
    "Carl Eric Almgren was promoted to general"
  ]
}
```

### Example 4
Question Stem: "Who did Knox Cunningham serve as Parliamentary Private Secretary to?"

Output:
```json
{
  "entity_query": "Knox Cunningham, a Northern Irish barrister, businessman and politician",
  "timeline_query": "Knox Cunningham's political career",
  "event_queries": [
    "Knox Cunningham served as Parliamentary Private Secretary to Harold Macmillan",
    "Knox Cunningham served as Parliamentary Private Secretary to Jocelyn Simon",
    "Knox Cunningham held political office in the UK Parliament",
    "Knox Cunningham was elected MP for South Antrim"
  ]
}
```

## Important Notes
1. Generate 3-7 event queries, covering different possible answers based on common knowledge
2. Use the entity's commonly known canonical name in all queries
3. The timeline query should focus on the relevant aspect (career, education, achievements, etc.)
4. Event queries should be declarative statements that could match events in a knowledge base
5. Be creative but factual - generate queries for plausible events/facts
"""

RETRIEVAL_USER_PROMPT = """Generate retrieval queries for the following question stem:

Question Stem: {question_stem}

Output in JSON format:"""


class QueryParser:
    """查询解析器

    将用户问题解析为主干和时间约束，并生成多层检索语句。
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
            original_question=question,
            question_stem=data.get("question_stem", question),
            time_constraint=time_constraint,
            event_type=event_type,
            answer_type=answer_type,
        )

    def generate_retrieval_queries(self, question_stem: str) -> RetrievalQueries:
        """
        基于问题主干生成检索语句

        Args:
            question_stem: 问题主干（不含时间约束）

        Returns:
            RetrievalQueries: 检索语句集合
        """
        user_prompt = RETRIEVAL_USER_PROMPT.format(question_stem=question_stem)

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

    def process(self, question: str) -> QueryParserOutput:
        """
        完整处理流程：解析问题并生成检索语句

        Args:
            question: 原始问题

        Returns:
            QueryParserOutput: 完整输出（解析结果 + 检索语句）
        """
        if not self.config.enabled:
            # 如果禁用查询解析器，返回原始问题作为主干，无时间约束
            parse_result = QueryParseResult(
                original_question=question,
                question_stem=question,
                time_constraint=TimeConstraint(
                    constraint_type=TimeConstraintType.NONE,
                    original_expression="",
                    normalized_time=None,
                    description="Query parser disabled",
                ),
                event_type=EventType.POINT,
                answer_type=AnswerType.ENTITY,
            )
            # 生成简单的检索语句
            retrieval_queries = RetrievalQueries(
                entity_query=question,
                timeline_query=question,
                event_queries=[question],
            )
            return QueryParserOutput(
                parse_result=parse_result,
                retrieval_queries=retrieval_queries,
            )

        # Step 1: 解析问题
        print("Step 1: 解析问题，提取主干、时间约束、事件类型和答案类型...")
        parse_result = self.parse_question(question)
        print(f"  - 主干: {parse_result.question_stem}")
        print(f"  - 时间约束类型: {parse_result.time_constraint.constraint_type.value}")
        if parse_result.time_constraint.original_expression:
            print(f"  - 时间表达式: {parse_result.time_constraint.original_expression}")
        print(f"  - 事件类型: {parse_result.event_type.value}")
        print(f"  - 答案类型: {parse_result.answer_type.value}")

        # Step 2: 生成检索语句
        print("\nStep 2: 生成检索语句...")
        retrieval_queries = self.generate_retrieval_queries(parse_result.question_stem)
        print(f"  - 实体查询: {retrieval_queries.entity_query}")
        print(f"  - 时间线查询: {retrieval_queries.timeline_query}")
        print(f"  - 事件查询数: {len(retrieval_queries.event_queries)}")

        return QueryParserOutput(
            parse_result=parse_result,
            retrieval_queries=retrieval_queries,
        )
