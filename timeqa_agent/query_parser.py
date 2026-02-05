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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_question": self.original_question,
            "question_stem": self.question_stem,
            "time_constraint": self.time_constraint.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryParseResult":
        return cls(
            original_question=data.get("original_question", ""),
            question_stem=data.get("question_stem", ""),
            time_constraint=TimeConstraint.from_dict(data.get("time_constraint", {})),
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

PARSE_SYSTEM_PROMPT = """You are an expert in temporal question analysis. Your task is to decompose a question into two parts:
1. **Question Stem**: The core question without temporal constraints
2. **Time Constraint**: The temporal constraint (if any)

## Time Constraint Types
- **explicit**: Direct time expressions like "in 2007", "from 1990 to 2000", "before 1980", "during 2015"
- **implicit**: Indirect time references like "during the Beijing Olympics", "when he was president", "after World War II", "during his tenure as CEO"
- **none**: No temporal constraint in the question

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
  }
}
```

## Examples

Input: "Which team did Attaphol Buspakom play for in 2007?"
Output:
```json
{
  "question_stem": "Which team did Attaphol Buspakom play for?",
  "time_constraint": {
    "constraint_type": "explicit",
    "original_expression": "in 2007",
    "normalized_time": "2007",
    "description": "The year 2007"
  }
}
```

Input: "Where did John Smith work during the Beijing Olympics?"
Output:
```json
{
  "question_stem": "Where did John Smith work?",
  "time_constraint": {
    "constraint_type": "implicit",
    "original_expression": "during the Beijing Olympics",
    "normalized_time": null,
    "description": "During the 2008 Beijing Summer Olympics (August 8-24, 2008)"
  }
}
```

Input: "Who is the CEO of Apple?"
Output:
```json
{
  "question_stem": "Who is the CEO of Apple?",
  "time_constraint": {
    "constraint_type": "none",
    "original_expression": "",
    "normalized_time": null,
    "description": "No time constraint"
  }
}
```

## Important Notes
1. Keep the question stem as close to the original as possible, only removing the time constraint part
2. For implicit time constraints, provide a helpful description that explains the time period
3. If there are multiple time constraints, focus on the primary one
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
   - Format: "[Entity Name]'s [aspect/career/life phase] [brief description] [related entities if applicable]"
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
Question Stem: "Which team did Attaphol Buspakom play for?"

Output:
```json
{
  "entity_query": "Attaphol Buspakom, a Thai professional football player",
  "timeline_query": "Attaphol Buspakom's football career, clubs and teams played for, professional football timeline",
  "event_queries": [
    "Attaphol Buspakom played for Buriram United F.C.",
    "Attaphol Buspakom played for Chonburi F.C.",
    "Attaphol Buspakom played for Thailand national football team",
    "Attaphol Buspakom joined a football club",
    "Attaphol Buspakom transferred to a new team"
  ]
}
```

### Example 2
Question Stem: "What position did John Smith hold at Microsoft?"

Output:
```json
{
  "entity_query": "John Smith, a technology executive at Microsoft",
  "timeline_query": "John Smith's career at Microsoft, positions and roles held, Microsoft employment history",
  "event_queries": [
    "John Smith served as CEO of Microsoft",
    "John Smith served as CTO of Microsoft",
    "John Smith served as Vice President at Microsoft",
    "John Smith held an executive position at Microsoft",
    "John Smith was appointed to a role at Microsoft"
  ]
}
```

### Example 3
Question Stem: "Where did Marie Curie conduct her research?"

Output:
```json
{
  "entity_query": "Marie Curie, a Polish-French physicist and chemist, Nobel Prize winner",
  "timeline_query": "Marie Curie's research career, scientific work locations, academic positions and institutions",
  "event_queries": [
    "Marie Curie conducted research at the University of Paris",
    "Marie Curie worked at the Radium Institute",
    "Marie Curie researched radioactivity in Paris",
    "Marie Curie established a laboratory",
    "Marie Curie performed experiments at a research institution"
  ]
}
```

### Example 4
Question Stem: "Who was the president of the United States?"

Output:
```json
{
  "entity_query": "President of the United States, head of state and government of the USA",
  "timeline_query": "United States presidential history, list of US presidents, American presidential terms",
  "event_queries": [
    "served as President of the United States",
    "was elected President of the United States",
    "held the office of US President",
    "became the President of the United States"
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
        解析问题，提取主干和时间约束

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

        return QueryParseResult(
            original_question=question,
            question_stem=data.get("question_stem", question),
            time_constraint=time_constraint,
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
        print("Step 1: 解析问题，提取主干和时间约束...")
        parse_result = self.parse_question(question)
        print(f"  - 主干: {parse_result.question_stem}")
        print(f"  - 时间约束类型: {parse_result.time_constraint.constraint_type.value}")
        if parse_result.time_constraint.original_expression:
            print(f"  - 时间表达式: {parse_result.time_constraint.original_expression}")

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
