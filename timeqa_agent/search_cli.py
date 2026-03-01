"""
检索语句生成器命令行工具

支持：
- 生成检索语句（支持单一实体和问题句子输入）
- 生成检索语句并执行检索
- 交互式模式
- JSON 输出
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import load_config
from .search import SearchQueryGenerator, RetrievalQueries, RetrievalResults
from .graph_store import TimelineGraphStore
from .retrievers import HybridRetriever
from .query_parser import QueryParseResult


def print_json(data, indent: int = 2):
    """格式化打印 JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def save_results_to_json(
    input_text: str,
    command: str,
    queries: Optional[RetrievalQueries] = None,
    results: Optional[RetrievalResults] = None,
    output_file: str = "search_result.json",
    retrieval_params: Optional[dict] = None,
):
    """将处理结果保存到 JSON 文件

    Args:
        input_text: 输入文本
        command: 执行的命令（generate 或 retrieve）
        queries: 检索语句对象
        results: 检索结果对象
        output_file: 输出文件路径
        retrieval_params: 检索参数（模式、top_k等）
    """
    # 构建保存的数据结构
    save_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": command,
        "input_text": input_text,
    }

    # 添加检索参数（如果有）
    if retrieval_params:
        save_data["retrieval_params"] = retrieval_params

    # 添加检索语句
    if queries:
        save_data["queries"] = queries.to_dict()

    # 添加检索结果
    if results:
        save_data["results"] = results.to_dict()

    # 检查文件是否存在，如果存在则读取现有数据
    output_path = Path(output_file)
    existing_data = []

    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # 如果文件中的数据不是列表，则转换为列表
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except Exception as e:
            print(f"警告: 读取现有结果文件失败: {e}，将创建新文件")
            existing_data = []

    # 追加新数据
    existing_data.append(save_data)

    # 保存到文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 结果已保存到: {output_path}")
    except Exception as e:
        print(f"\n✗ 保存结果失败: {e}")


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


def print_retrieval_results(results: RetrievalResults, verbose: bool = False):
    """打印检索结果"""
    print("\n" + "=" * 60)
    print("检索结果汇总")
    print("=" * 60)
    print(f"检索模式: {results.retrieval_mode}")
    print(f"实体数量: {len(results.entities)}")
    print(f"时间线数量: {len(results.timelines)}")
    print(f"直接检索事件数量: {len(results.events)}")
    print(f"合并后事件数量: {len(results.merged_events)}")

    # 打印实体
    if results.entities:
        print(f"\n{'='*60}")
        print(f"检索到的实体 ({len(results.entities)} 个)")
        print(f"{'='*60}")
        for i, entity in enumerate(results.entities[:10], 1):  # 只显示前10个
            if hasattr(entity, 'canonical_name'):
                print(f"\n{i}. [{entity.canonical_name}] (score: {entity.score:.4f})")
                if verbose and hasattr(entity, 'description'):
                    print(f"   描述: {entity.description}")
            else:
                print(f"\n{i}. {entity}")

    # 打印时间线
    if results.timelines:
        print(f"\n{'='*60}")
        print(f"检索到的时间线 ({len(results.timelines)} 条)")
        print(f"{'='*60}")
        for i, timeline in enumerate(results.timelines[:10], 1):  # 只显示前10条
            if hasattr(timeline, 'timeline_name'):
                print(f"\n{i}. [{timeline.timeline_name}] (score: {timeline.score:.4f})")
                if verbose and hasattr(timeline, 'description'):
                    print(f"   描述: {timeline.description}")
                if hasattr(timeline, 'entity_canonical_name'):
                    print(f"   所属实体: {timeline.entity_canonical_name}")
                if hasattr(timeline, 'event_ids'):
                    print(f"   包含事件: {len(timeline.event_ids)} 个")
            else:
                print(f"\n{i}. {timeline}")

    # 打印直接检索的事件
    if results.events:
        print(f"\n{'='*60}")
        print(f"直接检索到的事件 ({len(results.events)} 个)")
        print(f"{'='*60}")
        for i, event in enumerate(results.events[:10], 1):  # 只显示前10个
            if hasattr(event, 'event_description'):
                print(f"\n{i}. {event.event_description[:80]}... (score: {event.score:.4f})")
                if hasattr(event, 'time_start'):
                    time_str = f"{event.time_start}"
                    if hasattr(event, 'time_end') and event.time_end:
                        time_str += f" ~ {event.time_end}"
                    print(f"   时间: {time_str}")
                if verbose and hasattr(event, 'original_sentence'):
                    print(f"   原文: {event.original_sentence[:100]}...")
            else:
                print(f"\n{i}. {event}")

    # 打印合并后的事件（重点显示）
    if results.merged_events:
        print(f"\n{'='*60}")
        print(f"合并后的所有事件 ({len(results.merged_events)} 个)")
        print(f"包括: 直接检索事件 + 时间线中的事件（已去重）")
        print(f"{'='*60}")

        # 确定显示数量：verbose 模式显示全部，否则只显示前20个
        display_limit = len(results.merged_events) if verbose else min(20, len(results.merged_events))

        for i, event in enumerate(results.merged_events[:display_limit], 1):
            if hasattr(event, 'event_description'):
                desc = event.event_description[:80] if len(event.event_description) > 80 else event.event_description
                score_str = f" (score: {event.score:.4f})" if event.score > 0 else " (来自时间线)"
                print(f"\n{i}. {desc}{score_str}")
                if hasattr(event, 'time_start'):
                    time_str = f"{event.time_start}"
                    if hasattr(event, 'time_end') and event.time_end:
                        time_str += f" ~ {event.time_end}"
                    print(f"   时间: {time_str}")
                if verbose and hasattr(event, 'original_sentence'):
                    print(f"   原文: {event.original_sentence[:100]}...")
            else:
                print(f"\n{i}. {event}")

        # 如果有更多事件未显示，提示用户
        if len(results.merged_events) > display_limit:
            remaining = len(results.merged_events) - display_limit
            print(f"\n... 还有 {remaining} 个事件未显示")
            print(f"提示: 使用 --verbose 或 -v 参数查看所有事件")

    # 打印结构化关系
    if hasattr(results, 'structured_events') and results.structured_events:
        print(f"\n{'='*60}")
        print(f"提取的结构化关系 ({len(results.structured_events)} 条)")
        print(f"{'='*60}")

        # 确定显示数量：verbose 模式显示全部，否则只显示前20条
        display_limit = len(results.structured_events) if verbose else min(20, len(results.structured_events))

        for i, relation in enumerate(results.structured_events[:display_limit], 1):
            # 使用 __str__ 方法格式化输出
            print(f"\n{i}. {relation}")
            print(f"   类型: {relation.relation_type}")
            print(f"   主体: {relation.subject}")
            print(f"   客体: {relation.object_entity}")
            if relation.time_start:
                print(f"   开始: {relation.time_start}")
            if relation.time_end:
                print(f"   结束: {relation.time_end}")
            print(f"   置信度: {relation.confidence:.2f}")
            if verbose and relation.source_description:
                desc_preview = relation.source_description[:80] if len(relation.source_description) > 80 else relation.source_description
                print(f"   来源: {desc_preview}...")

        # 如果有更多关系未显示，提示用户
        if len(results.structured_events) > display_limit:
            remaining = len(results.structured_events) - display_limit
            print(f"\n... 还有 {remaining} 条关系未显示")
            print(f"提示: 使用 --verbose 或 -v 参数查看所有关系")

    # 打印过滤后的结构化关系
    if hasattr(results, 'filtered_structured_events') and results.filtered_structured_events:
        print(f"\n{'='*60}")
        print(f"过滤后的结构化关系 ({len(results.filtered_structured_events)} 条)")
        if results.question_analysis:
            print(f"基于问题: {results.question_analysis.question_stem}")
            if results.question_analysis.time_constraint.constraint_type.value != "none":
                print(f"时间约束: {results.question_analysis.time_constraint.description}")
        print(f"{'='*60}")

        # 确定显示数量：verbose 模式显示全部，否则只显示前20条
        display_limit = len(results.filtered_structured_events) if verbose else min(20, len(results.filtered_structured_events))

        for i, relation in enumerate(results.filtered_structured_events[:display_limit], 1):
            # 使用 __str__ 方法格式化输出
            print(f"\n{i}. {relation}")
            print(f"   类型: {relation.relation_type}")
            print(f"   主体: {relation.subject}")
            print(f"   客体: {relation.object_entity}")
            if relation.time_start:
                print(f"   开始: {relation.time_start}")
            if relation.time_end:
                print(f"   结束: {relation.time_end}")
            print(f"   置信度: {relation.confidence:.2f}")
            if verbose and relation.source_description:
                desc_preview = relation.source_description[:80] if len(relation.source_description) > 80 else relation.source_description
                print(f"   来源: {desc_preview}...")

        # 如果有更多关系未显示，提示用户
        if len(results.filtered_structured_events) > display_limit:
            remaining = len(results.filtered_structured_events) - display_limit
            print(f"\n... 还有 {remaining} 条关系未显示")
            print(f"提示: 使用 --verbose 或 -v 参数查看所有关系")

        # 显示过滤统计
        if hasattr(results, 'structured_events') and results.structured_events:
            original_count = len(results.structured_events)
            filtered_count = len(results.filtered_structured_events)
            removed_count = original_count - filtered_count
            print(f"\n过滤统计: 原始 {original_count} 条 → 保留 {filtered_count} 条 → 移除 {removed_count} 条")

    print()


class SearchCLI:
    """检索语句生成器命令行接口"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        graph_path: Optional[str] = None,
        verbose: bool = False,
        enable_retrieval: bool = False,
        save_results: bool = False,
        output_file: str = "search_result.json",
    ):
        """初始化命令行工具

        Args:
            config_path: 配置文件路径
            graph_path: 图文件路径（用于检索）
            verbose: 是否详细输出
            enable_retrieval: 是否启用检索功能
            save_results: 是否保存结果到JSON文件
            output_file: 结果输出文件路径
        """
        self.verbose = verbose
        self.enable_retrieval = enable_retrieval
        self.save_results = save_results
        self.output_file = output_file

        # 加载配置
        self.config = load_config(config_path) if config_path else load_config()

        # 初始化图存储和检索器（如果需要）
        self.graph_store = None
        self.retriever = None

        if enable_retrieval:
            if not graph_path:
                print("警告: 启用检索功能但未指定图文件路径，检索功能将不可用")
            elif not Path(graph_path).exists():
                print(f"警告: 图文件不存在: {graph_path}，检索功能将不可用")
            else:
                try:
                    # 加载图存储
                    self.graph_store = TimelineGraphStore()
                    self.graph_store.load(graph_path)
                    print(f"已加载图: {graph_path}")

                    # 创建嵌入函数（可选，根据配置）
                    embed_fn = self._create_embed_fn()

                    # 创建检索器
                    self.retriever = HybridRetriever(
                        self.graph_store,
                        embed_fn=embed_fn,
                        config=self.config.retriever,
                    )
                    print(f"检索器已初始化（模式: {self.config.query_parser.retrieval_mode}）")
                except Exception as e:
                    print(f"初始化检索器失败: {e}")
                    self.retriever = None

        # 创建生成器
        self.generator = SearchQueryGenerator(
            config=self.config.query_parser,
            graph_store=self.graph_store,
            retriever=self.retriever,
        )

    def _create_embed_fn(self):
        """创建嵌入函数"""
        try:
            from .embeddings import create_embed_fn

            model_type = self.config.retriever.semantic_model_type
            model_name = self.config.retriever.semantic_model_name
            device = self.config.retriever.semantic_model_device

            print(f"正在加载嵌入模型: {model_type} ({model_name})")

            if model_type.lower() == "contriever":
                embed_fn = create_embed_fn(
                    model_type=model_type,
                    model_name=model_name,
                    device=device,
                    normalize=self.config.retriever.contriever_normalize,
                    batch_size=self.config.retriever.embed_batch_size
                )
            elif model_type.lower() == "dpr":
                embed_fn = create_embed_fn(
                    model_type=model_type,
                    device=device,
                    ctx_encoder_name=self.config.retriever.dpr_ctx_encoder,
                    question_encoder_name=self.config.retriever.dpr_question_encoder,
                    batch_size=self.config.retriever.embed_batch_size
                )
            elif model_type.lower() == "bge-m3":
                if self.config.retriever.bge_m3_model_path:
                    model_name = self.config.retriever.bge_m3_model_path
                embed_fn = create_embed_fn(
                    model_type=model_type,
                    model_name=model_name,
                    normalize_embeddings=True
                )
            else:
                embed_fn = create_embed_fn(
                    model_type=model_type,
                    model_name=model_name,
                    device=device
                )

            if embed_fn:
                print(f"✓ 嵌入模型加载成功")
            return embed_fn

        except Exception as e:
            print(f"警告: 无法加载嵌入模型: {e}")
            print("将使用关键词检索模式")
            return None

    def cmd_generate(self, input_text: str) -> RetrievalQueries:
        """生成检索语句"""
        queries = self.generator.generate_retrieval_queries(input_text)

        # 保存结果到JSON文件（如果启用）
        if self.save_results:
            save_results_to_json(
                input_text=input_text,
                command="generate",
                queries=queries,
                output_file=self.output_file,
            )

        return queries

    def cmd_retrieve(
        self,
        input_text: str,
        retrieval_mode: Optional[str] = None,
        entity_top_k: Optional[int] = None,
        timeline_top_k: Optional[int] = None,
        event_top_k: Optional[int] = None,
        question_analysis_path: Optional[str] = None,
    ) -> RetrievalResults:
        """生成检索语句并执行检索"""
        if self.retriever is None:
            raise ValueError("检索器未初始化，请使用 --graph 参数指定图文件")

        # 加载问题解析（如果提供）
        question_analysis = None
        if question_analysis_path:
            try:
                with open(question_analysis_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    question_analysis = QueryParseResult.from_dict(data)
                    print(f"已加载问题解析: {question_analysis_path}")
            except Exception as e:
                print(f"警告: 加载问题解析失败: {e}")
                question_analysis = None

        # 使用配置中的默认值
        retrieval_mode = retrieval_mode or self.config.query_parser.retrieval_mode
        entity_top_k = entity_top_k or self.config.query_parser.entity_top_k
        timeline_top_k = timeline_top_k or self.config.query_parser.timeline_top_k
        event_top_k = event_top_k or self.config.query_parser.event_top_k

        results = self.generator.retrieve_with_queries(
            input_text=input_text,
            retrieval_mode=retrieval_mode,
            entity_top_k=entity_top_k,
            timeline_top_k=timeline_top_k,
            event_top_k=event_top_k,
            question_analysis=question_analysis,
        )

        # 保存结果到JSON文件（如果启用）
        if self.save_results:
            retrieval_params = {
                "retrieval_mode": retrieval_mode,
                "entity_top_k": entity_top_k,
                "timeline_top_k": timeline_top_k,
                "event_top_k": event_top_k,
            }
            if question_analysis_path:
                retrieval_params["question_analysis_path"] = question_analysis_path

            save_results_to_json(
                input_text=input_text,
                command="retrieve",
                queries=results.queries if hasattr(results, 'queries') else None,
                results=results,
                output_file=self.output_file,
                retrieval_params=retrieval_params,
            )

        return results

    def interactive(self):
        """交互式模式"""
        print("\n" + "=" * 60)
        print("TimeQA 检索语句生成器 - 交互式模式")
        print("=" * 60)
        print("功能:")
        print("  - 输入单一实体（如 'Barack Obama'）：直接返回实体作为检索语句")
        print("  - 输入问题句子（如 'Which team did he play for?'）：生成多层检索语句")
        if self.enable_retrieval:
            print("  - 检索功能已启用")
        if self.save_results:
            print(f"  - 结果保存已启用（输出文件: {self.output_file}）")
        print("\n命令:")
        print("  generate <输入>      - 仅生成检索语句")
        if self.enable_retrieval:
            print("  retrieve <输入>      - 生成检索语句并执行检索")
            print("  mode <hybrid|keyword|semantic>  - 设置检索模式")
            print("  topk <实体数> <时间线数> <事件数>  - 设置各层检索数量")
            print("  question <path>      - 加载问题解析JSON文件（用于过滤）")
        print("  save                 - 切换结果保存功能")
        print("  verbose              - 切换详细输出")
        print("  json                 - 切换 JSON 输出模式")
        print("  help                 - 显示帮助")
        print("  quit/exit            - 退出")
        print("=" * 60)

        json_mode = False
        retrieval_mode = self.config.query_parser.retrieval_mode
        entity_top_k = self.config.query_parser.entity_top_k
        timeline_top_k = self.config.query_parser.timeline_top_k
        event_top_k = self.config.query_parser.event_top_k
        question_analysis_path = None  # 添加问题解析路径变量

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
                elif cmd == "verbose":
                    self.verbose = not self.verbose
                    print(f"详细输出: {'开启' if self.verbose else '关闭'}")
                elif cmd == "save":
                    self.save_results = not self.save_results
                    print(f"结果保存: {'开启' if self.save_results else '关闭'}")
                    if self.save_results:
                        print(f"输出文件: {self.output_file}")
                elif cmd == "json":
                    json_mode = not json_mode
                    print(f"JSON 输出: {'开启' if json_mode else '关闭'}")
                elif cmd == "mode" and self.enable_retrieval:
                    if arg in ("hybrid", "keyword", "semantic"):
                        retrieval_mode = arg
                        print(f"检索模式已设置为: {retrieval_mode}")
                    else:
                        print("用法: mode <hybrid|keyword|semantic>")
                elif cmd == "topk" and self.enable_retrieval:
                    try:
                        topk_parts = arg.split()
                        if len(topk_parts) == 3:
                            entity_top_k = int(topk_parts[0])
                            timeline_top_k = int(topk_parts[1])
                            event_top_k = int(topk_parts[2])
                            print(f"检索数量已设置为: 实体={entity_top_k}, 时间线={timeline_top_k}, 事件={event_top_k}")
                        else:
                            print("用法: topk <实体数> <时间线数> <事件数>")
                    except ValueError:
                        print("用法: topk <实体数> <时间线数> <事件数>")
                elif cmd == "question" and arg and self.enable_retrieval:
                    if Path(arg).exists():
                        question_analysis_path = arg
                        print(f"已设置问题解析文件: {arg}")
                    else:
                        print(f"文件不存在: {arg}")
                elif cmd == "generate" and arg:
                    try:
                        queries = self.cmd_generate(arg)
                        if json_mode:
                            print_json(queries.to_dict())
                        else:
                            print_retrieval_queries(queries, self.verbose)
                    except Exception as e:
                        print(f"生成失败: {e}")
                elif cmd == "retrieve" and arg and self.enable_retrieval:
                    try:
                        results = self.cmd_retrieve(
                            arg,
                            retrieval_mode=retrieval_mode,
                            entity_top_k=entity_top_k,
                            timeline_top_k=timeline_top_k,
                            event_top_k=event_top_k,
                            question_analysis_path=question_analysis_path,
                        )
                        if json_mode:
                            print_json(results.to_dict())
                        else:
                            print_retrieval_results(results, self.verbose)
                    except Exception as e:
                        print(f"检索失败: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # 默认行为：生成检索语句
                    if line:
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

  # 生成检索语句并执行检索
  python -m timeqa_agent.search_cli retrieve "Which team did Thierry Audel play for?" -g data/timeqa/graph/test.json

  # 指定检索模式和数量
  python -m timeqa_agent.search_cli retrieve "Barack Obama" -g data/timeqa/graph/test.json --mode keyword --entity-topk 10 --event-topk 30

  # JSON 格式输出
  python -m timeqa_agent.search_cli generate "Barack Obama" --json

  # 保存测试结果到文件
  python -m timeqa_agent.search_cli retrieve "Who was Anna Karina married to?" -g data/timeqa/graph/test.json --save

  # 指定输出文件路径
  python -m timeqa_agent.search_cli retrieve "Barack Obama" -g data/timeqa/graph/test.json --save --output my_results.json

  # 使用指定配置文件
  python -m timeqa_agent.search_cli -c configs/timeqa_config.json generate "Who was Anna Karina married to?"
"""
    )

    parser.add_argument(
        "-c", "--config",
        help="配置文件路径"
    )

    parser.add_argument(
        "-g", "--graph",
        help="图文件路径（用于检索）"
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="interactive",
        help="命令: generate（仅生成检索语句）, retrieve（生成并执行检索）, interactive"
    )

    parser.add_argument(
        "input_text",
        nargs="?",
        help="输入文本（实体名或问题句子）"
    )

    parser.add_argument(
        "--mode",
        choices=["hybrid", "keyword", "semantic"],
        help="检索模式（用于 retrieve 命令）"
    )

    parser.add_argument(
        "--entity-topk",
        type=int,
        help="实体检索数量（用于 retrieve 命令）"
    )

    parser.add_argument(
        "--timeline-topk",
        type=int,
        help="时间线检索数量（用于 retrieve 命令）"
    )

    parser.add_argument(
        "--event-topk",
        type=int,
        help="事件检索数量（用于 retrieve 命令）"
    )

    parser.add_argument(
        "--question-analysis", "-qa",
        help="问题解析JSON文件路径（用于 retrieve 命令，用于过滤检索结果）"
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

    parser.add_argument(
        "--save",
        action="store_true",
        help="将处理结果保存到JSON文件（用于测试）"
    )

    parser.add_argument(
        "--output",
        default="search_result.json",
        help="结果输出文件路径（默认: search_result.json）"
    )

    args = parser.parse_args()

    # 判断是否需要启用检索功能
    enable_retrieval = args.command == "retrieve" or args.graph is not None

    cli = SearchCLI(
        config_path=args.config,
        graph_path=args.graph,
        verbose=args.verbose,
        enable_retrieval=enable_retrieval,
        save_results=args.save,
        output_file=args.output,
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
    elif cmd == "retrieve" and args.input_text:
        try:
            results = cli.cmd_retrieve(
                args.input_text,
                retrieval_mode=args.mode,
                entity_top_k=args.entity_topk,
                timeline_top_k=args.timeline_topk,
                event_top_k=args.event_topk,
                question_analysis_path=args.question_analysis,
            )
            if args.json:
                print_json(results.to_dict())
            else:
                print_retrieval_results(results, args.verbose)
        except Exception as e:
            print(f"检索失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
