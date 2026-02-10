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
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .config import QueryParserConfig


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
    ):
        self.config = config or QueryParserConfig()

        # 获取 token
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("请设置 VENUS_API_TOKEN 或 OPENAI_API_KEY 环境变量")

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
