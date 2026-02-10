"""
检索语句生成器命令行工具

支持：
- 生成检索语句（支持单一实体和问题句子输入）
- 交互式模式
- JSON 输出
"""

import argparse
import json
import sys
from typing import Optional

from .config import load_config
from .search import SearchQueryGenerator, RetrievalQueries


def print_json(data, indent: int = 2):
    """格式化打印 JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def print_retrieval_queries(queries: RetrievalQueries, verbose: bool = False):
    """打印检索语句"""
    print("\n" + "=" * 60)
    print("检索语句生成结果")
    print("=" * 60)

    print(f"\n实体查询:")
    print(f"  {queries.entity_query}")

    print(f"\n时间线查询:")
    print(f"  {queries.timeline_query}")

    print(f"\n事件查询 ({len(queries.event_queries)} 条):")
    for i, eq in enumerate(queries.event_queries, 1):
        print(f"  {i}. {eq}")

    print()


class SearchCLI:
    """检索语句生成器命令行接口"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose

        # 加载配置
        self.config = load_config(config_path) if config_path else load_config()

        # 创建生成器
        self.generator = SearchQueryGenerator(self.config.query_parser)

    def cmd_generate(self, input_text: str) -> RetrievalQueries:
        """生成检索语句"""
        return self.generator.generate_retrieval_queries(input_text)

    def interactive(self):
        """交互式模式"""
        print("\n" + "=" * 60)
        print("TimeQA 检索语句生成器 - 交互式模式")
        print("=" * 60)
        print("功能:")
        print("  - 输入单一实体（如 'Barack Obama'）：直接返回实体作为检索语句")
        print("  - 输入问题句子（如 'Which team did he play for?'）：生成多层检索语句")
        print("\n命令:")
        print("  <输入>           - 生成检索语句")
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
                    # 生成检索语句
                    try:
                        queries = self.cmd_generate(line)
                        if json_mode:
                            print_json(queries.to_dict())
                        else:
                            print_retrieval_queries(queries, self.verbose)
                    except Exception as e:
                        print(f"生成失败: {e}")

            except KeyboardInterrupt:
                print("\n再见!")
                break
            except EOFError:
                print("\n再见!")
                break


def main():
    parser = argparse.ArgumentParser(
        description="TimeQA 检索语句生成器命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python -m timeqa_agent.search_cli

  # 生成单一实体的检索语句
  python -m timeqa_agent.search_cli generate "Thierry Audel"

  # 生成问题句子的检索语句
  python -m timeqa_agent.search_cli generate "Which team did Thierry Audel play for?"

  # JSON 格式输出
  python -m timeqa_agent.search_cli generate "Barack Obama" --json

  # 使用指定配置文件
  python -m timeqa_agent.search_cli -c configs/timeqa_config.json generate "Who was Anna Karina married to?"
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
        help="命令: generate, interactive"
    )

    parser.add_argument(
        "input_text",
        nargs="?",
        help="输入文本（实体名或问题句子）"
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

    cli = SearchCLI(
        config_path=args.config,
        verbose=args.verbose,
    )

    cmd = args.command.lower()

    if cmd == "interactive":
        cli.interactive()
    elif cmd == "generate" and args.input_text:
        try:
            queries = cli.cmd_generate(args.input_text)
            if args.json:
                print_json(queries.to_dict())
            else:
                print_retrieval_queries(queries, args.verbose)
        except Exception as e:
            print(f"生成失败: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
