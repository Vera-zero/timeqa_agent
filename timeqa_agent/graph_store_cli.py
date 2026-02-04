"""
图存储命令行工具

支持交互式查询和命令行查询两种模式
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .graph_store import TimelineGraphStore


def print_json(data, indent: int = 2):
    """格式化打印 JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def print_entity(entity: dict):
    """打印实体信息"""
    print(f"\n{'='*60}")
    print(f"实体: {entity.get('canonical_name', 'N/A')}")
    print(f"{'='*60}")
    print(f"  ID: {entity.get('cluster_id', 'N/A')}")
    print(f"  描述: {entity.get('description', 'N/A')}")
    aliases = entity.get('aliases', [])
    if aliases:
        print(f"  别名: {', '.join(aliases)}")
    event_ids = entity.get('source_event_ids', [])
    if event_ids:
        print(f"  关联事件数: {len(event_ids)}")


def print_event(event: dict):
    """打印事件信息"""
    print(f"\n{'-'*60}")
    print(f"事件: {event.get('event_id', 'N/A')}")
    print(f"{'-'*60}")
    print(f"  描述: {event.get('event_description', 'N/A')}")
    print(f"  时间类型: {event.get('time_type', 'N/A')}")
    time_start = event.get('time_start', '')
    time_end = event.get('time_end', '')
    if time_start or time_end:
        print(f"  时间: {time_start} ~ {time_end}" if time_end else f"  时间: {time_start}")
    print(f"  时间表达: {event.get('time_expression', 'N/A')}")
    entities = event.get('entity_names', [])
    if entities:
        print(f"  参与实体: {', '.join(entities)}")
    if event.get('original_sentence'):
        print(f"  原文: {event.get('original_sentence')[:100]}...")


def print_timeline(timeline: dict):
    """打印时间线信息"""
    print(f"\n{'#'*60}")
    print(f"时间线: {timeline.get('timeline_name', 'N/A')}")
    print(f"{'#'*60}")
    print(f"  ID: {timeline.get('timeline_id', 'N/A')}")
    print(f"  描述: {timeline.get('description', 'N/A')}")
    print(f"  所属实体: {timeline.get('entity_canonical_name', 'N/A')}")
    start = timeline.get('time_span_start', '')
    end = timeline.get('time_span_end', '')
    if start or end:
        print(f"  时间跨度: {start} ~ {end}")
    event_ids = timeline.get('event_ids', [])
    print(f"  包含事件数: {len(event_ids)}")


class GraphStoreCLI:
    """图存储命令行接口"""
    
    def __init__(self, graph_path: str):
        self.store = TimelineGraphStore()
        self.graph_path = graph_path
        
        if Path(graph_path).exists():
            self.store.load(graph_path)
        else:
            print(f"警告: 图文件不存在: {graph_path}")
    
    def cmd_stats(self, args=None):
        """显示图统计信息"""
        stats = self.store.get_stats()
        print("\n图统计信息:")
        print(f"  节点总数: {stats['nodes']['total']}")
        print(f"    - 实体: {stats['nodes']['entities']}")
        print(f"    - 事件: {stats['nodes']['events']}")
        print(f"    - 时间线: {stats['nodes']['timelines']}")
        print(f"  边总数: {stats['edges']['total']}")
        print(f"    - 实体参与事件: {stats['edges']['participates_in']}")
        print(f"    - 事件属于时间线: {stats['edges']['belongs_to']}")
        print(f"    - 实体拥有时间线: {stats['edges']['has_timeline']}")
    
    def cmd_list_entities(self, args=None):
        """列出所有实体"""
        entities = self.store.list_all_entities()
        print(f"\n共 {len(entities)} 个实体:")
        for i, name in enumerate(entities, 1):
            print(f"  {i}. {name}")
    
    def cmd_list_timelines(self, args=None):
        """列出所有时间线"""
        timelines = self.store.list_all_timelines()
        print(f"\n共 {len(timelines)} 条时间线:")
        for i, tid in enumerate(timelines, 1):
            tl = self.store.get_timeline(tid)
            if tl:
                print(f"  {i}. [{tid}] {tl.get('timeline_name', 'N/A')} ({tl.get('entity_canonical_name', 'N/A')})")
    
    def cmd_entity(self, name: str):
        """查询实体详情"""
        entity = self.store.get_entity(name)
        if entity:
            print_entity(entity)
        else:
            # 尝试模糊搜索
            results = self.store.get_entities_by_name(name, fuzzy=True)
            if results:
                print(f"\n未找到精确匹配，模糊匹配到 {len(results)} 个实体:")
                for e in results:
                    print_entity(e)
            else:
                print(f"未找到实体: {name}")
    
    def cmd_entity_events(self, name: str):
        """查询实体参与的事件"""
        events = self.store.get_entity_events(name)
        if events:
            print(f"\n实体 '{name}' 参与的 {len(events)} 个事件:")
            for event in events:
                print_event(event)
        else:
            print(f"未找到实体 '{name}' 的事件")
    
    def cmd_entity_timelines(self, name: str):
        """查询实体的时间线"""
        timelines = self.store.get_entity_timelines(name)
        if timelines:
            print(f"\n实体 '{name}' 的 {len(timelines)} 条时间线:")
            for tl in timelines:
                print_timeline(tl)
        else:
            print(f"未找到实体 '{name}' 的时间线")
    
    def cmd_event(self, event_id: str):
        """查询事件详情"""
        event = self.store.get_event(event_id)
        if event:
            print_event(event)
            # 显示所属时间线
            timeline = self.store.get_event_timeline(event_id)
            if timeline:
                print(f"\n  所属时间线: {timeline.get('timeline_name', 'N/A')}")
        else:
            print(f"未找到事件: {event_id}")
    
    def cmd_timeline(self, timeline_id: str):
        """查询时间线详情"""
        timeline = self.store.get_timeline(timeline_id)
        if timeline:
            print_timeline(timeline)
            # 显示包含的事件
            events = self.store.get_timeline_events(timeline_id)
            if events:
                print(f"\n包含的事件 ({len(events)}):")
                for event in events:
                    print_event(event)
        else:
            print(f"未找到时间线: {timeline_id}")
    
    def cmd_timeline_events(self, timeline_id: str):
        """查询时间线包含的事件"""
        events = self.store.get_timeline_events(timeline_id)
        if events:
            print(f"\n时间线 '{timeline_id}' 包含 {len(events)} 个事件:")
            for event in events:
                print_event(event)
        else:
            print(f"未找到时间线 '{timeline_id}' 的事件")
    
    def cmd_search(self, query: str):
        """搜索实体（模糊匹配）"""
        results = self.store.get_entities_by_name(query, fuzzy=True)
        if results:
            print(f"\n搜索 '{query}' 匹配到 {len(results)} 个实体:")
            for entity in results:
                print_entity(entity)
        else:
            print(f"未找到匹配 '{query}' 的实体")
    
    def cmd_time_range(self, start: str, end: str):
        """查询时间范围内的事件"""
        events = self.store.get_events_in_time_range(start, end)
        if events:
            print(f"\n时间范围 {start} ~ {end} 内的 {len(events)} 个事件:")
            for event in events:
                print_event(event)
        else:
            print(f"未找到时间范围 {start} ~ {end} 内的事件")
    
    def interactive(self):
        """交互式模式"""
        print("\n" + "="*60)
        print("TimeQA 图存储交互式查询")
        print("="*60)
        print("命令:")
        print("  stats                    - 显示统计信息")
        print("  entities                 - 列出所有实体")
        print("  timelines                - 列出所有时间线")
        print("  entity <name>            - 查询实体详情")
        print("  entity-events <name>     - 查询实体参与的事件")
        print("  entity-timelines <name>  - 查询实体的时间线")
        print("  event <id>               - 查询事件详情")
        print("  timeline <id>            - 查询时间线详情及其事件")
        print("  search <query>           - 模糊搜索实体")
        print("  time <start> <end>       - 查询时间范围内的事件")
        print("  help                     - 显示帮助")
        print("  quit/exit                - 退出")
        print("="*60)
        
        while True:
            try:
                line = input("\n> ").strip()
                if not line:
                    continue
                
                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if cmd in ("quit", "exit", "q"):
                    print("再见!")
                    break
                elif cmd == "help":
                    self.interactive()
                    break
                elif cmd == "stats":
                    self.cmd_stats()
                elif cmd == "entities":
                    self.cmd_list_entities()
                elif cmd == "timelines":
                    self.cmd_list_timelines()
                elif cmd == "entity" and arg:
                    self.cmd_entity(arg)
                elif cmd == "entity-events" and arg:
                    self.cmd_entity_events(arg)
                elif cmd == "entity-timelines" and arg:
                    self.cmd_entity_timelines(arg)
                elif cmd == "event" and arg:
                    self.cmd_event(arg)
                elif cmd == "timeline" and arg:
                    self.cmd_timeline(arg)
                elif cmd == "search" and arg:
                    self.cmd_search(arg)
                elif cmd == "time":
                    time_parts = arg.split()
                    if len(time_parts) >= 2:
                        self.cmd_time_range(time_parts[0], time_parts[1])
                    else:
                        print("用法: time <start> <end>，例如: time 1990 2000")
                else:
                    print(f"未知命令: {cmd}，输入 help 查看帮助")
                    
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except EOFError:
                print("\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="TimeQA 图存储命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json
  
  # 查看统计信息
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json stats
  
  # 列出所有实体
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json entities
  
  # 查询实体
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json entity "John Smith"
  
  # 查询实体的事件
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json entity-events "John Smith"
  
  # 查询实体的时间线
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json entity-timelines "John Smith"
  
  # 查询事件
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json event "evt_001"
  
  # 查询时间线
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json timeline "tl_001"
  
  # 搜索实体
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json search "Smith"
  
  # 查询时间范围内的事件
  python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json time 1990 2000
"""
    )
    
    parser.add_argument(
        "-g", "--graph",
        required=True,
        help="图文件路径 (.json 或 .graphml)"
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="interactive",
        help="命令: stats, entities, timelines, entity, entity-events, entity-timelines, event, timeline, search, time"
    )
    
    parser.add_argument(
        "args",
        nargs="*",
        help="命令参数"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出"
    )
    
    args = parser.parse_args()
    
    cli = GraphStoreCLI(args.graph)
    
    cmd = args.command.lower()
    cmd_args = args.args
    
    if cmd == "interactive":
        cli.interactive()
    elif cmd == "stats":
        if args.json:
            print_json(cli.store.get_stats())
        else:
            cli.cmd_stats()
    elif cmd == "entities":
        if args.json:
            print_json(cli.store.list_all_entities())
        else:
            cli.cmd_list_entities()
    elif cmd == "timelines":
        if args.json:
            print_json(cli.store.list_all_timelines())
        else:
            cli.cmd_list_timelines()
    elif cmd == "entity" and cmd_args:
        name = " ".join(cmd_args)
        if args.json:
            result = cli.store.get_entity(name)
            if not result:
                result = cli.store.get_entities_by_name(name, fuzzy=True)
            print_json(result)
        else:
            cli.cmd_entity(name)
    elif cmd == "entity-events" and cmd_args:
        name = " ".join(cmd_args)
        if args.json:
            print_json(cli.store.get_entity_events(name))
        else:
            cli.cmd_entity_events(name)
    elif cmd == "entity-timelines" and cmd_args:
        name = " ".join(cmd_args)
        if args.json:
            print_json(cli.store.get_entity_timelines(name))
        else:
            cli.cmd_entity_timelines(name)
    elif cmd == "event" and cmd_args:
        event_id = cmd_args[0]
        if args.json:
            print_json(cli.store.get_event(event_id))
        else:
            cli.cmd_event(event_id)
    elif cmd == "timeline" and cmd_args:
        timeline_id = cmd_args[0]
        if args.json:
            result = cli.store.get_timeline(timeline_id)
            if result:
                result["events"] = cli.store.get_timeline_events(timeline_id)
            print_json(result)
        else:
            cli.cmd_timeline(timeline_id)
    elif cmd == "search" and cmd_args:
        query = " ".join(cmd_args)
        if args.json:
            print_json(cli.store.get_entities_by_name(query, fuzzy=True))
        else:
            cli.cmd_search(query)
    elif cmd == "time" and len(cmd_args) >= 2:
        if args.json:
            print_json(cli.store.get_events_in_time_range(cmd_args[0], cmd_args[1]))
        else:
            cli.cmd_time_range(cmd_args[0], cmd_args[1])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
