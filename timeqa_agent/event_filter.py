"""
Event Filter Module

过滤由 chunk 重叠产生的重复事件：
1. 去除完全相同的事件（event_description + 时间字段完全一致）
2. 对于描述相同但时间粒度不同的事件，仅保留粒度最细的版本
3. 合并后保留所有来源 chunk 的引用（直接写入 chunk_id 字段，逗号分隔）
"""

from __future__ import annotations

import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from .event_extractor import TimeEvent
from .time_utils import TemporalDate, TimeGranularity
from .config import EventFilterConfig, load_config


# 粒度评分：值越大粒度越细
_GRANULARITY_SCORE = {
    TimeGranularity.YEAR: 1,
    TimeGranularity.MONTH: 2,
    TimeGranularity.DAY: 3,
}


class EventFilter:
    """事件过滤器：去除 chunk 重叠导致的重复事件"""

    def __init__(self, config: Optional[EventFilterConfig] = None):
        self.config = config or EventFilterConfig()

    # ========== 公开接口 ==========

    def filter_events(self, events: List[TimeEvent]) -> List[TimeEvent]:
        """
        过滤重复事件

        Args:
            events: 事件列表（来自事件抽取阶段）

        Returns:
            过滤后的事件列表
        """
        if not self.config.enabled:
            return events

        # 按 (doc_id, normalized_description) 分组
        groups: Dict[Tuple[str, str], List[TimeEvent]] = defaultdict(list)
        for event in events:
            key = (event.doc_id, self._normalize(event.event_description))
            groups[key].append(event)

        filtered: List[TimeEvent] = []
        total_removed = 0

        for key, group in groups.items():
            if len(group) == 1:
                # 唯一事件，直接保留
                filtered.append(group[0])
            else:
                # 多个事件，按时间兼容性再分子组
                sub_groups = self._split_by_time_compatibility(group)
                for sub in sub_groups:
                    best = self._select_best_event(sub)
                    filtered.append(best)
                total_removed += len(group) - len(sub_groups)

        # 按原始事件顺序排序（以 event_id 字典序为基准）
        filtered.sort(key=lambda e: e.event_id)

        print(f"  事件过滤: {len(events)} -> {len(filtered)} (去除 {len(events) - len(filtered)} 个重复)")
        return filtered

    # ========== 内部方法 ==========

    @staticmethod
    def _normalize(text: str) -> str:
        """标准化文本用于比较"""
        return text.strip().lower()

    def _split_by_time_compatibility(
        self, events: List[TimeEvent]
    ) -> List[List[TimeEvent]]:
        """
        将同一描述下的事件按时间兼容性再细分

        时间兼容 = 两个时间字段可视为同一事件的不同粒度表示
        不兼容的时间（如 2008 vs 2009）被拆分到不同子组
        """
        sub_groups: List[List[TimeEvent]] = []

        for event in events:
            placed = False
            for sub in sub_groups:
                if self._is_time_compatible(sub[0], event):
                    sub.append(event)
                    placed = True
                    break
            if not placed:
                sub_groups.append([event])

        return sub_groups

    def _is_time_compatible(self, a: TimeEvent, b: TimeEvent) -> bool:
        """
        判断两个事件的时间是否兼容（可视为同一事件的不同粒度）

        兼容条件：
        - time_type 相同
        - time_start 兼容（一个是另一个的粗粒度前缀）
        - time_end 兼容（一个是另一个的粗粒度前缀，或同为 None）
        """
        if a.time_type != b.time_type:
            return False

        if not self._times_compatible(a.time_start, b.time_start):
            return False

        if not self._times_compatible(a.time_end, b.time_end):
            return False

        return True

    @staticmethod
    def _times_compatible(t1: Optional[str], t2: Optional[str]) -> bool:
        """
        判断两个时间字符串是否兼容

        兼容 = 完全相同，或一个是另一个的粗粒度前缀
        例如: "2008" 与 "2008-07" 兼容，"2008" 与 "2009" 不兼容
        """
        if t1 is None and t2 is None:
            return True
        if t1 is None or t2 is None:
            return False

        # 快速路径：完全相同
        if t1 == t2:
            return True

        try:
            d1 = TemporalDate.parse(t1)
            d2 = TemporalDate.parse(t2)
        except ValueError:
            return t1 == t2  # 无法解析时退回精确匹配

        # 年份必须相同
        if d1.year != d2.year:
            return False

        # 如果其中一个只到年份，年份匹配即兼容
        if d1.month is None or d2.month is None:
            return True

        # 月份必须相同
        if d1.month != d2.month:
            return False

        # 如果其中一个只到月份，月份匹配即兼容
        if d1.day is None or d2.day is None:
            return True

        # 日期必须相同
        return d1.day == d2.day

    def _select_best_event(self, events: List[TimeEvent]) -> TimeEvent:
        """
        从一组时间兼容的事件中选出最佳事件（粒度最细）

        1. 计算每个事件的时间粒度得分
        2. 选得分最高者
        3. 合并所有 chunk 引用到 chunk_id 字段（逗号分隔）
        """
        # 收集所有 chunk_id（去重并保持顺序）
        seen_chunk_ids: Dict[str, None] = {}
        for evt in events:
            # chunk_id 可能已经是逗号分隔的多个 id
            for cid in evt.chunk_id.split(","):
                cid = cid.strip()
                if cid and cid not in seen_chunk_ids:
                    seen_chunk_ids[cid] = None

        # 选择粒度最细的事件
        best = max(events, key=lambda e: self._time_granularity_score(e))

        # 将所有来源 chunk_id 合并写入 chunk_id 字段
        best.chunk_id = ",".join(seen_chunk_ids.keys())

        return best

    @staticmethod
    def _time_granularity_score(event: TimeEvent) -> int:
        """
        计算事件的时间粒度得分

        综合 time_start 和 time_end 的粒度，值越大粒度越细
        """
        score = 0

        if event.time_start:
            try:
                d = TemporalDate.parse(event.time_start)
                score += _GRANULARITY_SCORE.get(d.granularity, 0)
            except ValueError:
                pass

        if event.time_end:
            try:
                d = TemporalDate.parse(event.time_end)
                score += _GRANULARITY_SCORE.get(d.granularity, 0)
            except ValueError:
                pass

        return score


# ========== CLI 入口 ==========


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="事件过滤器：去除 chunk 重叠产生的重复事件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 过滤单文档事件
  python -m timeqa_agent.event_filter -i data/timeqa/event/test_doc0.json -o data/timeqa/event_filter/test_doc0.json

  # 使用配置文件
  python -m timeqa_agent.event_filter -i data/timeqa/event/test.json -o data/timeqa/event_filter/test.json --config configs/timeqa_config.json
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
        help="输出过滤后事件 JSON 文件路径",
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
    filter_config = timeqa_config.event_filter if timeqa_config else EventFilterConfig()

    # 加载事件
    print(f"加载事件数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        events_data = json.load(f)

    events = [TimeEvent.from_dict(e) for e in events_data]
    print(f"共 {len(events)} 个事件")

    # 过滤
    event_filter = EventFilter(filter_config)
    filtered = event_filter.filter_events(events)

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    filtered_data = [e.to_dict() for e in filtered]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
