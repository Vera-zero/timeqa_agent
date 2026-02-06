"""
Event Validator Module

事件检查器：在事件抽取之后、事件过滤之前对事件进行校验
主要功能：
1. 检查时间格式是否符合 YYYY-MM-DD/YYYY-MM/YYYY/null
2. 对格式不正确的时间调用 LLM 进行重新检查和纠正
"""

from __future__ import annotations

import os
import re
import json
import requests
import argparse
from typing import List, Dict, Any, Optional, Tuple

from .event_extractor import TimeEvent, TimeType
from .config import EventValidatorConfig, load_config


# 合法时间格式的正则表达式
# 匹配: YYYY | YYYY-MM | YYYY-MM-DD | null/None/空
TIME_FORMAT_PATTERN = re.compile(
    r'^(?:'
    r'\d{4}'                          # YYYY
    r'(?:-(?:0[1-9]|1[0-2])'          # -MM (01-12)
    r'(?:-(?:0[1-9]|[12]\d|3[01]))?'  # -DD (01-31) 可选
    r')?'
    r')$'
)


def is_valid_time_format(time_str: Optional[str]) -> bool:
    """
    检查时间字符串是否符合合法格式

    合法格式:
    - YYYY (如 "2008")
    - YYYY-MM (如 "2008-07")
    - YYYY-MM-DD (如 "2008-07-15")
    - null/None/空字符串

    Args:
        time_str: 时间字符串

    Returns:
        是否合法
    """
    if time_str is None or time_str == "" or time_str.lower() == "null":
        return True
    return bool(TIME_FORMAT_PATTERN.match(time_str))


def extract_time_from_text(time_str: str) -> Optional[str]:
    """
    尝试从不规范的时间字符串中提取合法时间

    使用正则表达式尝试提取 YYYY-MM-DD / YYYY-MM / YYYY 格式

    Args:
        time_str: 可能不规范的时间字符串

    Returns:
        提取到的合法时间，如果无法提取则返回 None
    """
    if time_str is None or time_str == "":
        return None

    # 尝试匹配 YYYY-MM-DD
    match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', time_str)
    if match:
        year, month, day = match.groups()
        month = month.zfill(2)
        day = day.zfill(2)
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{year}-{month}-{day}"

    # 尝试匹配 YYYY-MM
    match = re.search(r'(\d{4})-(\d{1,2})(?!\d)', time_str)
    if match:
        year, month = match.groups()
        month = month.zfill(2)
        if 1 <= int(month) <= 12:
            return f"{year}-{month}"

    # 尝试匹配 YYYY
    match = re.search(r'\b(\d{4})\b', time_str)
    if match:
        year = match.group(1)
        if 1000 <= int(year) <= 2100:  # 合理的年份范围
            return year

    return None


# LLM 检查提示词
VALIDATION_SYSTEM_PROMPT = """You are an expert in temporal information extraction and validation.

Your task is to review and correct temporal information (time_start, time_end) in extracted events.

## Time Format Requirements
All time values must follow one of these formats:
- YYYY (e.g., "2008")
- YYYY-MM (e.g., "2008-07")
- YYYY-MM-DD (e.g., "2008-07-15")
- null (when time cannot be determined from context)

## Validation Rules
1. If the time value is in an incorrect format, try to convert it to the correct format based on context
2. If the time is ambiguous or cannot be determined, set it to null
3. For point events (time_type="point"), only time_start should have a value, time_end should be null
4. For range events (time_type="range"), both time_start and time_end may have values
5. If both time_start and time_end are null, it means the time cannot be inferred from context

## Output Format
Output a JSON object with the corrected time values:
```json
{
  "time_start": "corrected value or null",
  "time_end": "corrected value or null",
  "correction_note": "brief explanation of the correction"
}
```
"""


VALIDATION_USER_PROMPT = """Please validate and correct the temporal information for the following event:

## Document Title
{doc_title}

## Original Sentence
{original_sentence}

## Event Description
{event_description}

## Time Expression (from original text)
{time_expression}

## Current Time Values (need validation)
- time_type: {time_type}
- time_start: {time_start}
- time_end: {time_end}

## Issues Detected
{issues}

Please review the context and provide corrected time values in the required format (YYYY, YYYY-MM, YYYY-MM-DD, or null):
"""


class EventValidator:
    """事件检查器"""

    def __init__(
        self,
        config: Optional[EventValidatorConfig] = None,
        token: Optional[str] = None,
    ):
        self.config = config or EventValidatorConfig()

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

    def _check_time_format(self, event: TimeEvent) -> Tuple[bool, List[str]]:
        """
        检查事件的时间格式

        Args:
            event: 事件对象

        Returns:
            (是否通过检查, 问题列表)
        """
        issues = []

        # 检查 time_start
        if not is_valid_time_format(event.time_start):
            issues.append(f"time_start 格式不正确: '{event.time_start}'")

        # 检查 time_end
        if not is_valid_time_format(event.time_end):
            issues.append(f"time_end 格式不正确: '{event.time_end}'")

        # 对于 point 类型，time_end 应该为 null
        if event.time_type == TimeType.POINT and event.time_end is not None:
            # 这不算错误，但可以记录
            pass

        return len(issues) == 0, issues

    def _try_rule_based_fix(self, event: TimeEvent) -> Tuple[bool, TimeEvent]:
        """
        尝试使用基于规则的方式修复时间格式

        Args:
            event: 事件对象

        Returns:
            (是否修复成功, 修复后的事件)
        """
        fixed = False

        # 尝试修复 time_start
        if event.time_start and not is_valid_time_format(event.time_start):
            extracted = extract_time_from_text(event.time_start)
            if extracted:
                event.time_start = extracted
                fixed = True

        # 尝试修复 time_end
        if event.time_end and not is_valid_time_format(event.time_end):
            extracted = extract_time_from_text(event.time_end)
            if extracted:
                event.time_end = extracted
                fixed = True

        return fixed, event

    def _validate_with_llm(self, event: TimeEvent, issues: List[str]) -> TimeEvent:
        """
        使用 LLM 验证并纠正事件的时间格式

        Args:
            event: 事件对象
            issues: 检测到的问题列表

        Returns:
            纠正后的事件
        """
        user_prompt = VALIDATION_USER_PROMPT.format(
            doc_title=event.doc_title,
            original_sentence=event.original_sentence,
            event_description=event.event_description,
            time_expression=event.time_expression,
            time_type=event.time_type.value,
            time_start=event.time_start,
            time_end=event.time_end,
            issues="\n".join(f"- {issue}" for issue in issues),
        )

        messages = [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = self._call_api(messages)
            data = json.loads(content)

            # 更新时间值
            new_start = data.get("time_start")
            new_end = data.get("time_end")

            # 规范化 null 值
            if new_start is not None:
                if isinstance(new_start, str) and new_start.lower() == "null":
                    new_start = None
                elif is_valid_time_format(new_start):
                    event.time_start = new_start if new_start else None
                else:
                    print(f"  LLM 返回的 time_start 仍不合法: {new_start}，保持原值")

            if new_end is not None:
                if isinstance(new_end, str) and new_end.lower() == "null":
                    new_end = None
                    event.time_end = None
                elif is_valid_time_format(new_end):
                    event.time_end = new_end if new_end else None
                else:
                    print(f"  LLM 返回的 time_end 仍不合法: {new_end}，保持原值")

            # 处理 null 字符串
            if new_start == "null" or new_start == "":
                event.time_start = None
            elif new_start is not None and is_valid_time_format(new_start):
                event.time_start = new_start

            if new_end == "null" or new_end == "":
                event.time_end = None
            elif new_end is not None and is_valid_time_format(new_end):
                event.time_end = new_end

            note = data.get("correction_note", "")
            if note:
                print(f"  纠正说明: {note}")

        except Exception as e:
            print(f"  LLM 验证失败: {e}，保持原值")

        return event

    def validate_events(self, events: List[TimeEvent]) -> List[TimeEvent]:
        """
        验证事件列表中的时间格式

        Args:
            events: 事件列表

        Returns:
            验证/纠正后的事件列表
        """
        if not self.config.enabled:
            print("事件检查已禁用，跳过检查")
            return events

        validated_events = []
        total_issues = 0
        rule_fixed = 0
        llm_fixed = 0

        for i, event in enumerate(events):
            # 检查时间格式
            is_valid, issues = self._check_time_format(event)

            if is_valid:
                # 格式正确，直接添加
                validated_events.append(event)
            else:
                total_issues += 1
                print(f"\n事件 {event.event_id} 时间格式问题:")
                for issue in issues:
                    print(f"  - {issue}")

                # 首先尝试基于规则的修复
                fixed_by_rule, event = self._try_rule_based_fix(event)

                # 再次检查
                is_valid_after_rule, remaining_issues = self._check_time_format(event)

                if is_valid_after_rule:
                    print(f"  ✓ 基于规则修复成功")
                    rule_fixed += 1
                    validated_events.append(event)
                else:
                    # 需要 LLM 介入
                    print(f"  → 调用 LLM 进行检查...")
                    event = self._validate_with_llm(event, remaining_issues)

                    # 最终检查
                    is_valid_final, final_issues = self._check_time_format(event)
                    if is_valid_final:
                        print(f"  ✓ LLM 纠正成功")
                        llm_fixed += 1
                    else:
                        print(f"  ⚠ 仍存在格式问题，但保留事件")
                        for issue in final_issues:
                            print(f"    - {issue}")

                    validated_events.append(event)

        # 统计输出
        print(f"\n事件检查完成:")
        print(f"  总事件数: {len(events)}")
        print(f"  格式问题: {total_issues}")
        print(f"  规则修复: {rule_fixed}")
        print(f"  LLM 修复: {llm_fixed}")

        return validated_events


# ========== CLI 入口 ==========


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="事件检查器：验证并纠正事件的时间格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查单文档事件
  python -m timeqa_agent.event_validator -i data/timeqa/event/test_doc0.json -o data/timeqa/event_validate/test_doc0.json

  # 使用配置文件
  python -m timeqa_agent.event_validator -i data/timeqa/event/test.json -o data/timeqa/event_validate/test.json --config configs/timeqa_config.json
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入事件 JSON 文件路径",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出验证后事件 JSON 文件路径",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径",
    )

    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    # 加载配置
    timeqa_config = load_config(args.config) if args.config else None
    validator_config = timeqa_config.event_validator if timeqa_config else EventValidatorConfig()

    # 加载事件
    print(f"加载事件数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        events_data = json.load(f)

    events = [TimeEvent.from_dict(e) for e in events_data]
    print(f"共 {len(events)} 个事件")

    # 验证
    validator = EventValidator(validator_config)
    validated = validator.validate_events(events)

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    validated_data = [e.to_dict() for e in validated]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(validated_data, f, ensure_ascii=False, indent=2)

    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
