"""
检索器命令行工具

支持关键词、语义、混合检索，以及交互式查询
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Callable

from .graph_store import TimelineGraphStore
from .config import RetrieverConfig, FusionMode, VotingFusionMode, load_config
from .retrievers import (
    HybridRetriever,
    KeywordRetriever,
    SemanticRetriever,
    MultiLayerVotingRetriever,
    RetrievalResult,
    EntityResult,
    EventResult,
    TimelineResult,
    VotingEventResult,
    RetrievalMode,
)


def print_json(data, indent: int = 2):
    """格式化打印 JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def print_result(result: RetrievalResult, verbose: bool = False):
    """打印检索结果"""
    if isinstance(result, VotingEventResult):
        print_voting_event_result(result, verbose)
    elif isinstance(result, EntityResult):
        print_entity_result(result, verbose)
    elif isinstance(result, EventResult):
        print_event_result(result, verbose)
    elif isinstance(result, TimelineResult):
        print_timeline_result(result, verbose)
    else:
        print_generic_result(result, verbose)


def print_entity_result(result: EntityResult, verbose: bool = False):
    """打印实体检索结果"""
    print(f"\n{'='*60}")
    print(f"[实体] {result.canonical_name}  (score: {result.score:.4f})")
    print(f"{'='*60}")
    print(f"  ID: {result.node_id}")
    if result.description:
        print(f"  描述: {result.description}")
    if result.aliases:
        print(f"  别名: {', '.join(result.aliases)}")
    if verbose and result.source_event_ids:
        print(f"  关联事件数: {len(result.source_event_ids)}")


def print_event_result(result: EventResult, verbose: bool = False):
    """打印事件检索结果"""
    print(f"\n{'-'*60}")
    print(f"[事件] {result.event_id}  (score: {result.score:.4f})")
    print(f"{'-'*60}")
    print(f"  描述: {result.event_description}")
    print(f"  时间类型: {result.time_type}")
    if result.time_start or result.time_end:
        time_str = f"{result.time_start} ~ {result.time_end}" if result.time_end else result.time_start
        print(f"  时间: {time_str}")
    if result.time_expression:
        print(f"  时间表达: {result.time_expression}")
    if result.entity_names:
        print(f"  参与实体: {', '.join(result.entity_names)}")
    if verbose and result.original_sentence:
        print(f"  原文: {result.original_sentence[:150]}...")


def print_voting_event_result(result: VotingEventResult, verbose: bool = False):
    """打印投票事件检索结果"""
    print(f"\n{'-'*60}")
    print(f"[事件] {result.event_id}  (agg_score: {result.aggregated_score:.4f}, votes: {result.vote_count})")
    print(f"{'-'*60}")
    print(f"  描述: {result.event_description}")
    print(f"  时间类型: {result.time_type}")
    if result.time_start or result.time_end:
        time_str = f"{result.time_start} ~ {result.time_end}" if result.time_end else result.time_start
        print(f"  时间: {time_str}")
    if result.time_expression:
        print(f"  时间表达: {result.time_expression}")
    if result.entity_names:
        print(f"  参与实体: {', '.join(result.entity_names)}")
    if verbose:
        print(f"  投票来源:")
        for src in result.vote_sources:
            print(f"    - [{src['source_type']}] {src['source_id']} (score: {src['source_score']:.4f}, rank: {src['rank_in_source']})")
        if result.original_sentence:
            print(f"  原文: {result.original_sentence[:150]}...")


def print_timeline_result(result: TimelineResult, verbose: bool = False):
    """打印时间线检索结果"""
    print(f"\n{'#'*60}")
    print(f"[时间线] {result.timeline_name}  (score: {result.score:.4f})")
    print(f"{'#'*60}")
    print(f"  ID: {result.timeline_id}")
    if result.description:
        print(f"  描述: {result.description}")
    print(f"  所属实体: {result.entity_canonical_name}")
    if result.time_span_start or result.time_span_end:
        print(f"  时间跨度: {result.time_span_start} ~ {result.time_span_end}")
    if verbose:
        print(f"  包含事件数: {len(result.event_ids)}")


def print_generic_result(result: RetrievalResult, verbose: bool = False):
    """打印通用检索结果"""
    print(f"\n[{result.node_type}] {result.node_id}  (score: {result.score:.4f})")
    if verbose and result.metadata:
        print(f"  元数据: {result.metadata}")


def create_embed_fn(config) -> Optional[Callable]:
    """创建嵌入函数（优先使用本地BGE-M3模型，回退到API）"""
    import os
    
    # 尝试使用本地BGE-M3模型
    from .embeddings import create_local_embed_fn
    
    # 检查配置中是否有本地模型路径指定
    local_model_path = getattr(config.disambiguator, 'local_embed_model_path', None)
    embed_fn = create_local_embed_fn(local_model_path)
    
    if embed_fn:
        print("✓ 已启用本地BGE-M3模型进行语义检索")
        return embed_fn
    
    print("本地BGE-M3模型不可用，尝试使用API...")
    
    # 获取 API token（作为回退选项）
    token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
    if not token:
        print("警告: 未设置 VENUS_API_TOKEN 或 OPENAI_API_KEY 环境变量，语义检索不可用")
        return None
    
    try:
        import httpx
    except ImportError:
        print("警告: 未安装 httpx，语义检索不可用。请运行: pip install httpx")
        return None
    
    def embed_fn(texts: List[str]) -> List[List[float]]:
        """调用 Embedding API"""
        url = config.disambiguator.embed_base_url
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        with httpx.Client(timeout=60) as client:
            response = client.post(
                url,
                headers=headers,
                json={
                    "model": config.disambiguator.embed_model,
                    "input": texts,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # 按 index 排序
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]
    
    return embed_fn


class RetrieverCLI:
    """检索器命令行接口"""
    
    def __init__(
        self,
        graph_path: str,
        config_path: Optional[str] = None,
        enable_semantic: bool = True,
        verbose: bool = False,
        index_dir: Optional[str] = None,  # 索引缓存目录
        skip_index_load: bool = False,  # 跳过自动加载（用于 rebuild）
    ):
        self.verbose = verbose
        
        # 加载配置
        self.config = load_config(config_path) if config_path else load_config()
        
        # 加载图
        self.store = TimelineGraphStore()
        self.graph_path = graph_path
        
        if Path(graph_path).exists():
            self.store.load(graph_path)
            print(f"已加载图: {graph_path}")
        else:
            print(f"警告: 图文件不存在: {graph_path}")
            return
        
        # 确定索引缓存目录（默认与图文件同目录）
        if index_dir:
            self._index_dir = str(index_dir)
        else:
            graph_file = Path(graph_path)
            self._index_dir = str(graph_file.parent / f"{graph_file.stem}_indexes")
        
        # 检查索引目录是否存在
        index_path = Path(self._index_dir)
        has_cached_index = index_path.exists() and (index_path / "entity_index.json").exists()
        
        # 创建检索器
        embed_fn = None
        if enable_semantic:
            embed_fn = create_embed_fn(self.config)
            if embed_fn:
                print("语义检索已启用")
            else:
                print("语义检索未启用（缺少 embedding 函数）")
        
        # 检索器会自动加载/保存索引缓存
        # 如果指定了 skip_index_load，则临时禁用自动加载（用于重建索引）
        actual_index_dir = None if skip_index_load else self._index_dir
        
        self.retriever = HybridRetriever(
            self.store,
            embed_fn=embed_fn,
            config=self.config.retriever,
            index_dir=actual_index_dir,
        )
        
        self._keyword_retriever = KeywordRetriever(self.store, self.config.retriever)
        self._semantic_retriever = self.retriever._semantic_retriever  # 复用 HybridRetriever 内部的
        
        # 多层投票检索器
        self._voting_retriever = MultiLayerVotingRetriever(
            self.store,
            embed_fn=embed_fn,
            retriever_config=self.config.retriever,
            voting_config=self.config.voting,
            index_dir=actual_index_dir,
        )
        
        # 打印索引加载状态
        if not skip_index_load and has_cached_index:
            if self._semantic_retriever and self._semantic_retriever._indexes_built:
                print(f"✓ 已从缓存加载向量索引: {self._index_dir}")
            else:
                print(f"索引目录存在但未加载（需要 embedding 函数）: {self._index_dir}")
    
    def rebuild_indexes(self) -> None:
        """重新构建并保存向量索引"""
        if not self._semantic_retriever:
            print("错误: 语义检索未启用")
            return
        
        print("正在重新构建向量索引...")
        
        # 清除旧索引
        self._semantic_retriever.invalidate_cache()
        
        # 设置索引目录（如果初始化时跳过了加载）
        if not self._semantic_retriever._index_dir:
            self._semantic_retriever._index_dir = Path(self._index_dir)
            self._semantic_retriever._auto_save = True
        
        # 重新构建
        self._semantic_retriever.build_indexes()
        
        # 保存到磁盘
        self._semantic_retriever.save_indexes(self._index_dir)
        print(f"✓ 向量索引已保存到: {self._index_dir}")
    
    def cmd_search(
        self,
        query: str,
        mode: str = "hybrid",
        target_type: str = "all",
        top_k: int = 10,
        fusion_mode: Optional[str] = None,
        voting_fusion_mode: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """执行检索（检索器会自动处理索引的加载/保存）"""
        results = []
        
        if mode == "keyword":
            results = self._keyword_retriever.retrieve(
                query, top_k=top_k, target_type=target_type
            )
        elif mode == "semantic":
            if self._semantic_retriever:
                results = self._semantic_retriever.retrieve(
                    query, top_k=top_k, target_type=target_type
                )
            else:
                print("错误: 语义检索未启用")
                return []
        elif mode == "voting":
            # 多层投票检索
            if voting_fusion_mode:
                self._voting_retriever.update_voting_config(
                    fusion_mode=VotingFusionMode(voting_fusion_mode)
                )
            results = self._voting_retriever.retrieve(query, top_k=top_k)
        else:  # hybrid
            fm = FusionMode(fusion_mode) if fusion_mode else None
            results = self.retriever.retrieve(
                query, top_k=top_k, target_type=target_type, fusion_mode=fm
            )
        
        return results
    
    def cmd_voting_search_details(
        self,
        query: str,
        top_k: int = 10,
    ) -> dict:
        """执行投票检索并返回详细信息"""
        return self._voting_retriever.retrieve_with_details(query, top_k=top_k)
    
    def cmd_entity_search(
        self,
        query: str,
        entity_name: str,
        top_k: int = 10
    ) -> List[EventResult]:
        """带实体上下文的检索"""
        return self.retriever.search_with_entity_context(
            query, entity_name, top_k=top_k
        )
    
    def cmd_stats(self):
        """显示统计信息"""
        stats = self.store.get_stats()
        print("\n检索器统计信息:")
        print(f"  节点总数: {stats['nodes']['total']}")
        print(f"    - 实体: {stats['nodes']['entities']}")
        print(f"    - 事件: {stats['nodes']['events']}")
        print(f"    - 时间线: {stats['nodes']['timelines']}")
        print(f"\n混合检索配置:")
        print(f"    - 默认 top_k: {self.config.retriever.top_k}")
        print(f"    - 融合模式: {self.config.retriever.fusion_mode}")
        print(f"    - 关键词权重: {self.config.retriever.keyword_weight}")
        print(f"    - 语义权重: {self.config.retriever.semantic_weight}")
        print(f"\n投票检索配置:")
        print(f"    - 融合模式: {self.config.voting.fusion_mode.value}")
        print(f"    - 实体层权重: {self.config.voting.entity_layer_weight}")
        print(f"    - 时间线层权重: {self.config.voting.timeline_layer_weight}")
        print(f"    - 事件层权重: {self.config.voting.event_layer_weight}")
    
    def interactive(self):
        """交互式模式"""
        print("\n" + "="*60)
        print("TimeQA 检索器交互式查询")
        print("="*60)
        print("命令:")
        print("  search <query>                   - 混合检索")
        print("  keyword <query>                  - 关键词检索")
        print("  semantic <query>                 - 语义检索")
        print("  voting <query>                   - 多层投票检索")
        print("  voting-details <query>           - 投票检索（含详细信息）")
        print("  entity <name> <query>            - 实体上下文检索")
        print("  type <entity|event|timeline>    - 设置目标类型过滤")
        print("  topk <n>                         - 设置返回数量")
        print("  fusion <rrf|weighted_sum|max_score|interleave>  - 设置混合融合模式")
        print("  vfusion <rrf|weighted|vote_count|borda>  - 设置投票融合模式")
        print("  stats                            - 显示统计信息")
        print("  verbose                          - 切换详细输出")
        print("  help                             - 显示帮助")
        print("  quit/exit                        - 退出")
        print("="*60)
        
        # 默认设置
        target_type = "all"
        top_k = 10
        fusion_mode = None
        voting_fusion_mode = None
        
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
                elif cmd == "verbose":
                    self.verbose = not self.verbose
                    print(f"详细输出: {'开启' if self.verbose else '关闭'}")
                elif cmd == "type":
                    if arg in ("entity", "event", "timeline", "all"):
                        target_type = arg
                        print(f"目标类型设置为: {target_type}")
                    else:
                        print("用法: type <entity|event|timeline|all>")
                elif cmd == "topk":
                    try:
                        top_k = int(arg)
                        print(f"返回数量设置为: {top_k}")
                    except ValueError:
                        print("用法: topk <数字>")
                elif cmd == "fusion":
                    if arg in ("rrf", "weighted_sum", "max_score", "interleave"):
                        fusion_mode = arg
                        print(f"混合融合模式设置为: {fusion_mode}")
                    else:
                        print("用法: fusion <rrf|weighted_sum|max_score|interleave>")
                elif cmd == "vfusion":
                    if arg in ("rrf", "weighted", "vote_count", "borda"):
                        voting_fusion_mode = arg
                        print(f"投票融合模式设置为: {voting_fusion_mode}")
                    else:
                        print("用法: vfusion <rrf|weighted|vote_count|borda>")
                elif cmd == "voting" and arg:
                    results = self.cmd_search(
                        arg, mode="voting", top_k=top_k,
                        voting_fusion_mode=voting_fusion_mode
                    )
                    self._print_results(results, arg)
                elif cmd == "voting-details" and arg:
                    details = self.cmd_voting_search_details(arg, top_k=top_k)
                    print(f"\n查询 '{arg}' 投票检索详情:")
                    print(f"\n各层检索结果数:")
                    print(f"  - 实体层: {len(details['layer_results']['entity_layer'])}")
                    print(f"  - 时间线层: {len(details['layer_results']['timeline_layer'])}")
                    print(f"  - 事件层: {len(details['layer_results']['event_layer'])}")
                    print(f"\n最终结果: {len(details['final_results'])} 条")
                    for r in details['final_results']:
                        print(f"  [{r['event_id']}] score={r['aggregated_score']:.4f}, votes={r['vote_count']}")
                        print(f"    {r['event_description'][:80]}...")
                elif cmd == "search" and arg:
                    results = self.cmd_search(
                        arg, mode="hybrid", target_type=target_type,
                        top_k=top_k, fusion_mode=fusion_mode
                    )
                    self._print_results(results, arg)
                elif cmd == "keyword" and arg:
                    results = self.cmd_search(
                        arg, mode="keyword", target_type=target_type, top_k=top_k
                    )
                    self._print_results(results, arg)
                elif cmd == "semantic" and arg:
                    results = self.cmd_search(
                        arg, mode="semantic", target_type=target_type, top_k=top_k
                    )
                    self._print_results(results, arg)
                elif cmd == "entity":
                    entity_parts = arg.split(maxsplit=1)
                    if len(entity_parts) >= 2:
                        entity_name, query = entity_parts
                        results = self.cmd_entity_search(query, entity_name, top_k=top_k)
                        self._print_results(results, query)
                    else:
                        print("用法: entity <实体名> <查询>")
                else:
                    # 默认作为混合检索
                    if line:
                        results = self.cmd_search(
                            line, mode="hybrid", target_type=target_type,
                            top_k=top_k, fusion_mode=fusion_mode
                        )
                        self._print_results(results, line)
                    
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except EOFError:
                print("\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")
    
    def _print_results(self, results: List[RetrievalResult], query: str):
        """打印检索结果"""
        if results:
            print(f"\n查询 '{query}' 返回 {len(results)} 条结果:")
            for result in results:
                print_result(result, self.verbose)
        else:
            print(f"未找到匹配 '{query}' 的结果")


def main():
    parser = argparse.ArgumentParser(
        description="TimeQA 检索器命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json
  
  # 混合检索
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "when did John join the company"
  
  # 关键词检索
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json keyword "John Smith"
  
  # 语义检索
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json semantic "career changes"
  
  # 多层投票检索
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json voting "when did John join"
  
  # 投票检索（含详细信息）
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json voting-details "career history"
  
  # 只检索事件
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "2020" -t event
  
  # 只检索实体
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "Smith" -t entity
  
  # 带实体上下文的检索
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json entity-search "John Smith" "when did he graduate"
  
  # 设置返回数量
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "event" -k 20
  
  # 使用特定融合模式
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "career" --fusion weighted_sum
  
  # 投票检索使用特定融合模式
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json voting "career" --vfusion borda
  
  # JSON 输出
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "John" --json
  
  # 禁用语义检索（仅使用关键词）
  python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "John" --no-semantic
"""
    )
    
    parser.add_argument(
        "-g", "--graph",
        required=True,
        help="图文件路径 (.json)"
    )
    
    parser.add_argument(
        "-c", "--config",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="interactive",
        help="命令: search, keyword, semantic, voting, voting-details, entity-search, stats, interactive"
    )
    
    parser.add_argument(
        "args",
        nargs="*",
        help="命令参数"
    )
    
    parser.add_argument(
        "-t", "--type",
        choices=["all", "entity", "event", "timeline"],
        default="all",
        help="目标类型过滤"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="返回结果数量"
    )
    
    parser.add_argument(
        "--fusion",
        choices=["rrf", "weighted_sum", "max_score", "interleave"],
        help="混合检索融合模式"
    )
    
    parser.add_argument(
        "--vfusion",
        choices=["rrf", "weighted", "vote_count", "borda"],
        help="投票检索融合模式"
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
        "--no-semantic",
        action="store_true",
        help="禁用语义检索"
    )
    
    parser.add_argument(
        "--index-dir",
        help="向量索引缓存目录（默认与图文件同目录）"
    )
    
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重新构建向量索引"
    )
    
    args = parser.parse_args()
    
    # 如果需要重建索引，跳过自动加载
    skip_load = args.rebuild_index
    
    cli = RetrieverCLI(
        args.graph,
        config_path=args.config,
        enable_semantic=not args.no_semantic,
        verbose=args.verbose,
        index_dir=args.index_dir,
        skip_index_load=skip_load,
    )
    
    # 处理重建索引命令
    if args.rebuild_index:
        cli.rebuild_indexes()
        if args.command.lower() not in ("search", "semantic", "hybrid", "keyword", "voting", "voting-details", "entity-search"):
            return
    
    cmd = args.command.lower()
    cmd_args = args.args
    
    if cmd == "interactive":
        cli.interactive()
    elif cmd == "stats":
        if args.json:
            print_json(cli.store.get_stats())
        else:
            cli.cmd_stats()
    elif cmd == "search" and cmd_args:
        query = " ".join(cmd_args)
        results = cli.cmd_search(
            query,
            mode="hybrid",
            target_type=args.type,
            top_k=args.top_k,
            fusion_mode=args.fusion
        )
        if args.json:
            print_json([r.to_dict() for r in results])
        else:
            cli._print_results(results, query)
    elif cmd == "keyword" and cmd_args:
        query = " ".join(cmd_args)
        results = cli.cmd_search(
            query,
            mode="keyword",
            target_type=args.type,
            top_k=args.top_k
        )
        if args.json:
            print_json([r.to_dict() for r in results])
        else:
            cli._print_results(results, query)
    elif cmd == "semantic" and cmd_args:
        query = " ".join(cmd_args)
        results = cli.cmd_search(
            query,
            mode="semantic",
            target_type=args.type,
            top_k=args.top_k
        )
        if args.json:
            print_json([r.to_dict() for r in results])
        else:
            cli._print_results(results, query)
    elif cmd == "voting" and cmd_args:
        query = " ".join(cmd_args)
        results = cli.cmd_search(
            query,
            mode="voting",
            top_k=args.top_k,
            voting_fusion_mode=args.vfusion
        )
        if args.json:
            print_json([r.to_dict() for r in results])
        else:
            cli._print_results(results, query)
    elif cmd == "voting-details" and cmd_args:
        query = " ".join(cmd_args)
        details = cli.cmd_voting_search_details(query, top_k=args.top_k)
        if args.json:
            print_json(details)
        else:
            print(f"\n查询 '{query}' 投票检索详情:")
            print(f"\n各层检索结果数:")
            print(f"  - 实体层: {len(details['layer_results']['entity_layer'])}")
            print(f"  - 时间线层: {len(details['layer_results']['timeline_layer'])}")
            print(f"  - 事件层: {len(details['layer_results']['event_layer'])}")
            print(f"\n最终结果: {len(details['final_results'])} 条")
            for r in details['final_results']:
                print(f"  [{r['event_id']}] score={r['aggregated_score']:.4f}, votes={r['vote_count']}")
                desc = r.get('event_description', '')
                print(f"    {desc[:80]}..." if len(desc) > 80 else f"    {desc}")
    elif cmd == "entity-search" and len(cmd_args) >= 2:
        entity_name = cmd_args[0]
        query = " ".join(cmd_args[1:])
        results = cli.cmd_entity_search(query, entity_name, top_k=args.top_k)
        if args.json:
            print_json([r.to_dict() for r in results])
        else:
            cli._print_results(results, query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()