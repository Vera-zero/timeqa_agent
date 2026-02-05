"""
查询解析器命令行工具

支持：
- 解析单个问题
- 交互式模式
- 批量处理
- JSON 输出
"""

import argparse
import json
import sys
from typing import Optional

from .config import load_config
from .query_parser import QueryParser, QueryParserOutput


def print_json(data, indent: int = 2):
    """格式化打印 JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def print_parse_result(output: QueryParserOutput, verbose: bool = False):
    """打印解析结果"""
    parse_result = output.parse_result
    retrieval_queries = output.retrieval_queries

    print("\n" + "=" * 60)
    print("查询解析结果")
    print("=" * 60)

    print(f"\n原始问题: {parse_result.original_question}")
    print(f"问题主干: {parse_result.question_stem}")

    print(f"\n时间约束:")
    tc = parse_result.time_constraint
    print(f"  类型: {tc.constraint_type.value}")
    if tc.original_expression:
        print(f"  原始表达式: {tc.original_expression}")
    if tc.normalized_time:
        print(f"  标准化时间: {tc.normalized_time}")
    if tc.description:
        print(f"  描述: {tc.description}")

    print(f"\n" + "-" * 60)
    print("检索语句")
    print("-" * 60)

    print(f"\n实体查询:")
    print(f"  {retrieval_queries.entity_query}")

    print(f"\n时间线查询:")
    print(f"  {retrieval_queries.timeline_query}")

    print(f"\n事件查询 ({len(retrieval_queries.event_queries)} 条):")
    for i, eq in enumerate(retrieval_queries.event_queries, 1):
        print(f"  {i}. {eq}")

    print()


class QueryParserCLI:
    """查询解析器命令行接口"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose

        # 加载配置
        self.config = load_config(config_path) if config_path else load_config()

        # 创建解析器
        self.parser = QueryParser(self.config.query_parser)

        if not self.config.query_parser.enabled:
            print("警告: 查询解析器已禁用，将返回简单结果")

    def cmd_parse(self, question: str) -> QueryParserOutput:
        """解析单个问题"""
        return self.parser.process(question)

    def interactive(self):
        """交互式模式"""
        print("\n" + "=" * 60)
        print("TimeQA 查询解析器 - 交互式模式")
        print("=" * 60)
        print("命令:")
        print("  <问题>           - 解析问题")
        print("  verbose          - 切换详细输出")
        print("  json             - 切换 JSON 输出模式")
        print("  help             - 显示帮助")
        print("  quit/exit        - 退出")
        print("=" * 60)

        json_mode = False

        while True:
            try:
                line = input("\n> ").strip()
                if not line:
                    continue

                cmd = line.lower()

                if cmd in ("quit", "exit", "q"):
                    print("再见!")
                    break
                elif cmd == "help":
                    self.interactive()
                    break
                elif cmd == "verbose":
                    self.verbose = not self.verbose
                    print(f"详细输出: {'开启' if self.verbose else '关闭'}")
                elif cmd == "json":
                    json_mode = not json_mode
                    print(f"JSON 输出: {'开启' if json_mode else '关闭'}")
                else:
                    # 解析问题
                    try:
                        output = self.cmd_parse(line)
                        if json_mode:
                            print_json(output.to_dict())
                        else:
                            print_parse_result(output, self.verbose)
                    except Exception as e:
                        print(f"解析失败: {e}")

            except KeyboardInterrupt:
                print("\n再见!")
                break
            except EOFError:
                print("\n再见!")
                break


def main():
    parser = argparse.ArgumentParser(
        description="TimeQA 查询解析器命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python -m timeqa_agent.query_parser_cli

  # 解析单个问题
  python -m timeqa_agent.query_parser_cli parse "Which team did Attaphol Buspakom play for in 2007?"

  # JSON 格式输出
  python -m timeqa_agent.query_parser_cli parse "Where did John work during the Olympics?" --json

  # 使用指定配置文件
  python -m timeqa_agent.query_parser_cli -c configs/timeqa_config.json parse "When did he graduate?"

  # 仅解析问题（不生成检索语句）
  python -m timeqa_agent.query_parser_cli parse-only "Who was president in 1990?"

  # 仅生成检索语句（直接输入主干）
  python -m timeqa_agent.query_parser_cli retrieval "Which team did Attaphol Buspakom play for?"
"""
    )

    parser.add_argument(
        "-c", "--config",
        help="配置文件路径"
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="interactive",
        help="命令: parse, parse-only, retrieval, interactive"
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="要解析的问题"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出"
    )

    args = parser.parse_args()

    cli = QueryParserCLI(
        config_path=args.config,
        verbose=args.verbose,
    )

    cmd = args.command.lower()

    if cmd == "interactive":
        cli.interactive()
    elif cmd == "parse" and args.question:
        try:
            output = cli.cmd_parse(args.question)
            if args.json:
                print_json(output.to_dict())
            else:
                print_parse_result(output, args.verbose)
        except Exception as e:
            print(f"解析失败: {e}")
            sys.exit(1)
    elif cmd == "parse-only" and args.question:
        try:
            parse_result = cli.parser.parse_question(args.question)
            if args.json:
                print_json(parse_result.to_dict())
            else:
                print(f"\n原始问题: {parse_result.original_question}")
                print(f"问题主干: {parse_result.question_stem}")
                print(f"\n时间约束:")
                tc = parse_result.time_constraint
                print(f"  类型: {tc.constraint_type.value}")
                if tc.original_expression:
                    print(f"  原始表达式: {tc.original_expression}")
                if tc.normalized_time:
                    print(f"  标准化时间: {tc.normalized_time}")
                if tc.description:
                    print(f"  描述: {tc.description}")
        except Exception as e:
            print(f"解析失败: {e}")
            sys.exit(1)
    elif cmd == "retrieval" and args.question:
        try:
            queries = cli.parser.generate_retrieval_queries(args.question)
            if args.json:
                print_json(queries.to_dict())
            else:
                print(f"\n实体查询: {queries.entity_query}")
                print(f"\n时间线查询: {queries.timeline_query}")
                print(f"\n事件查询 ({len(queries.event_queries)} 条):")
                for i, eq in enumerate(queries.event_queries, 1):
                    print(f"  {i}. {eq}")
        except Exception as e:
            print(f"生成检索语句失败: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
