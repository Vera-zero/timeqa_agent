"""
TimeQA Extraction Pipeline

将分块、事件抽取、实体消歧、时间线抽取、图存储串联起来的完整流水线。

支持:
- 命令行执行
- 指定数据集 (test/train/validation)
- 指定起始和结束阶段
- 单文档模式（验证链路）和全量模式
- 中间结果保存和恢复
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import IntEnum
from dataclasses import dataclass

from .config import TimeQAConfig, load_config
from .chunker import DocumentChunker, Chunk
from .event_extractor import EventExtractor, TimeEvent
from .entity_disambiguator import EntityDisambiguator, EntityCluster
from .timeline_extractor import TimelineExtractor, TimelineExtractionResult
from .graph_store import TimelineGraphStore


class Stage(IntEnum):
    """流水线阶段"""
    CHUNK = 1
    EVENT = 2
    DISAMBIGUATE = 3
    TIMELINE = 4
    GRAPH = 5


STAGE_NAMES = {
    Stage.CHUNK: "chunk",
    Stage.EVENT: "event",
    Stage.DISAMBIGUATE: "disambiguate",
    Stage.TIMELINE: "timeline",
    Stage.GRAPH: "graph",
}

STAGE_FROM_NAME = {v: k for k, v in STAGE_NAMES.items()}


@dataclass
class PipelineConfig:
    """流水线配置"""
    split: str = "test"                    # 数据集分割: test/train/validation
    start_stage: Stage = Stage.CHUNK       # 起始阶段
    end_stage: Stage = Stage.GRAPH         # 结束阶段
    mode: str = "single"                   # 模式: single(单文档) / full(全量)
    doc_index: int = 0                     # 单文档模式下的文档索引
    
    # 路径配置
    data_dir: str = "data/timeqa"
    
    @property
    def corpus_path(self) -> str:
        return f"{self.data_dir}/corpus/{self.split}.json"
    
    def stage_output_path(self, stage: Stage) -> str:
        """获取阶段输出路径"""
        stage_name = STAGE_NAMES[stage]
        suffix = f"_doc{self.doc_index}" if self.mode == "single" else ""
        return f"{self.data_dir}/{stage_name}/{self.split}{suffix}.json"
    
    def graph_output_path(self) -> str:
        """获取图输出路径"""
        suffix = f"_doc{self.doc_index}" if self.mode == "single" else ""
        return f"{self.data_dir}/graph/{self.split}{suffix}.json"


class ExtractionPipeline:
    """抽取流水线"""
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        timeqa_config: Optional[TimeQAConfig] = None,
        token: Optional[str] = None,
    ):
        self.pipeline_config = pipeline_config
        self.timeqa_config = timeqa_config or TimeQAConfig()
        self.token = token
        
        # 懒加载各组件
        self._chunker: Optional[DocumentChunker] = None
        self._event_extractor: Optional[EventExtractor] = None
        self._disambiguator: Optional[EntityDisambiguator] = None
        self._timeline_extractor: Optional[TimelineExtractor] = None
        self._graph_store: Optional[TimelineGraphStore] = None
    
    @property
    def chunker(self) -> DocumentChunker:
        if self._chunker is None:
            self._chunker = DocumentChunker(self.timeqa_config.chunk)
        return self._chunker
    
    @property
    def event_extractor(self) -> EventExtractor:
        if self._event_extractor is None:
            self._event_extractor = EventExtractor(
                self.timeqa_config.extractor, 
                token=self.token
            )
        return self._event_extractor
    
    @property
    def disambiguator(self) -> EntityDisambiguator:
        if self._disambiguator is None:
            self._disambiguator = EntityDisambiguator(
                self.timeqa_config.disambiguator,
                token=self.token
            )
        return self._disambiguator
    
    @property
    def timeline_extractor(self) -> TimelineExtractor:
        if self._timeline_extractor is None:
            self._timeline_extractor = TimelineExtractor(
                self.timeqa_config.timeline,
                token=self.token
            )
        return self._timeline_extractor
    
    @property
    def graph_store(self) -> TimelineGraphStore:
        if self._graph_store is None:
            self._graph_store = TimelineGraphStore(self.timeqa_config.graph_store)
        return self._graph_store
    
    def _ensure_dir(self, path: str) -> None:
        """确保目录存在"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def _save_json(self, path: str, data: Any) -> None:
        """保存 JSON 文件"""
        self._ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存: {path}")
    
    def _load_json(self, path: str) -> Any:
        """加载 JSON 文件"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_corpus(self) -> List[Dict[str, Any]]:
        """加载语料库"""
        corpus_path = self.pipeline_config.corpus_path
        print(f"加载语料库: {corpus_path}")
        
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 支持两种格式: 直接列表 或 {"documents": [...]} 对象
        if isinstance(data, dict) and "documents" in data:
            documents = data["documents"]
        else:
            documents = data
        
        # 根据模式过滤
        if self.pipeline_config.mode == "single":
            idx = self.pipeline_config.doc_index
            if idx >= len(documents):
                raise ValueError(f"文档索引 {idx} 超出范围 (共 {len(documents)} 篇)")
            documents = [documents[idx]]
            print(f"单文档模式: 选择第 {idx} 篇文档")
        
        print(f"共 {len(documents)} 篇文档")
        return documents
    
    # ========== 阶段执行 ==========
    
    def run_chunk_stage(self, documents: Optional[List[Dict]] = None) -> List[Dict]:
        """
        阶段1: 文档分块
        
        输入: 语料库文档
        输出: 分块列表
        """
        print("\n" + "="*50)
        print("阶段 1: 文档分块")
        print("="*50)
        
        if documents is None:
            documents = self._load_corpus()
        
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = self.chunker.chunk_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                doc_title=doc["title"],
                source_idx=doc["source_idx"],
            )
            all_chunks.extend(chunks)
            print(f"  [{i+1}/{len(documents)}] {doc['title'][:50]}... -> {len(chunks)} 个分块")
        
        print(f"\n共生成 {len(all_chunks)} 个分块")
        
        # 保存
        output_path = self.pipeline_config.stage_output_path(Stage.CHUNK)
        chunks_data = [c.to_dict() for c in all_chunks]
        self._save_json(output_path, chunks_data)
        
        return chunks_data
    
    def run_event_stage(self, chunks_data: Optional[List[Dict]] = None) -> List[Dict]:
        """
        阶段2: 事件抽取
        
        输入: 分块列表
        输出: 事件列表
        """
        print("\n" + "="*50)
        print("阶段 2: 事件抽取")
        print("="*50)
        
        # 加载或使用传入的分块数据
        if chunks_data is None:
            input_path = self.pipeline_config.stage_output_path(Stage.CHUNK)
            print(f"加载分块数据: {input_path}")
            chunks_data = self._load_json(input_path)
        
        # 转换为 Chunk 对象
        chunks = [Chunk.from_dict(c) for c in chunks_data]
        print(f"共 {len(chunks)} 个分块待处理")
        
        # 抽取事件
        all_events = []
        
        def progress_callback(current: int, total: int):
            if current % 10 == 0 or current == total:
                print(f"  进度: {current}/{total}")
        
        events = self.event_extractor.extract_from_chunks(chunks, progress_callback)
        all_events.extend(events)
        
        print(f"\n共抽取 {len(all_events)} 个事件")
        
        # 保存
        output_path = self.pipeline_config.stage_output_path(Stage.EVENT)
        events_data = [e.to_dict() for e in all_events]
        self._save_json(output_path, events_data)
        
        return events_data
    
    def run_disambiguate_stage(self, events_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        阶段3: 实体消歧
        
        输入: 事件列表
        输出: 实体聚类结果
        """
        print("\n" + "="*50)
        print("阶段 3: 实体消歧")
        print("="*50)
        
        # 加载或使用传入的事件数据
        if events_data is None:
            input_path = self.pipeline_config.stage_output_path(Stage.EVENT)
            print(f"加载事件数据: {input_path}")
            events_data = self._load_json(input_path)
        
        # 转换为 TimeEvent 对象
        events = [TimeEvent.from_dict(e) for e in events_data]
        print(f"共 {len(events)} 个事件待处理")
        
        # 实体消歧
        clusters, entity_to_cluster = self.disambiguator.disambiguate_events(events)
        
        print(f"\n共生成 {len(clusters)} 个实体聚类")
        
        # 保存
        output_path = self.pipeline_config.stage_output_path(Stage.DISAMBIGUATE)
        result = {
            "num_clusters": len(clusters),
            "clusters": [c.to_dict() for c in clusters],
            "entity_to_cluster": entity_to_cluster,
        }
        self._save_json(output_path, result)
        
        return result
    
    def run_timeline_stage(
        self,
        events_data: Optional[List[Dict]] = None,
        disambiguate_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        阶段4: 时间线抽取
        
        输入: 事件列表 + 实体聚类
        输出: 时间线结果
        """
        print("\n" + "="*50)
        print("阶段 4: 时间线抽取")
        print("="*50)
        
        # 加载事件数据
        if events_data is None:
            input_path = self.pipeline_config.stage_output_path(Stage.EVENT)
            print(f"加载事件数据: {input_path}")
            events_data = self._load_json(input_path)
        
        # 加载消歧数据（用于获取实体列表）
        if disambiguate_data is None:
            input_path = self.pipeline_config.stage_output_path(Stage.DISAMBIGUATE)
            print(f"加载消歧数据: {input_path}")
            disambiguate_data = self._load_json(input_path)
        
        # 转换为 TimeEvent 对象
        events = [TimeEvent.from_dict(e) for e in events_data]
        
        # 获取所有实体名称
        clusters = disambiguate_data.get("clusters", [])
        entity_names = [c["canonical_name"] for c in clusters]
        print(f"共 {len(entity_names)} 个实体待处理")
        
        # 抽取时间线
        timeline_results = self.timeline_extractor.extract_timelines(
            events, 
            target_entities=entity_names
        )
        
        total_timelines = sum(len(r.timelines) for r in timeline_results.values())
        print(f"\n共生成 {total_timelines} 条时间线")
        
        # 保存
        output_path = self.pipeline_config.stage_output_path(Stage.TIMELINE)
        result = {
            "num_entities": len(timeline_results),
            "num_timelines": total_timelines,
            "results": {k: v.to_dict() for k, v in timeline_results.items()},
        }
        self._save_json(output_path, result)
        
        return result
    
    def run_graph_stage(
        self,
        events_data: Optional[List[Dict]] = None,
        disambiguate_data: Optional[Dict] = None,
        timeline_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        阶段5: 图存储
        
        输入: 事件 + 实体聚类 + 时间线
        输出: 知识图谱
        """
        print("\n" + "="*50)
        print("阶段 5: 图存储")
        print("="*50)
        
        # 加载数据
        if events_data is None:
            input_path = self.pipeline_config.stage_output_path(Stage.EVENT)
            print(f"加载事件数据: {input_path}")
            events_data = self._load_json(input_path)
        
        if disambiguate_data is None:
            input_path = self.pipeline_config.stage_output_path(Stage.DISAMBIGUATE)
            print(f"加载消歧数据: {input_path}")
            disambiguate_data = self._load_json(input_path)
        
        if timeline_data is None:
            input_path = self.pipeline_config.stage_output_path(Stage.TIMELINE)
            print(f"加载时间线数据: {input_path}")
            timeline_data = self._load_json(input_path)
        
        # 转换对象
        events = [TimeEvent.from_dict(e) for e in events_data]
        clusters = [EntityCluster.from_dict(c) for c in disambiguate_data.get("clusters", [])]
        timeline_results = {
            k: TimelineExtractionResult.from_dict(v) 
            for k, v in timeline_data.get("results", {}).items()
        }
        
        # 导入图
        stats = self.graph_store.import_from_pipeline(events, clusters, timeline_results)
        
        print(f"\n图导入统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 保存图
        output_path = self.pipeline_config.graph_output_path()
        self.graph_store.save(output_path)
        
        # 返回统计
        graph_stats = self.graph_store.get_stats()
        return {
            "import_stats": stats,
            "graph_stats": graph_stats,
        }
    
    # ========== 主执行入口 ==========
    
    def run(self) -> Dict[str, Any]:
        """
        执行流水线
        
        根据配置的起始和结束阶段执行
        """
        config = self.pipeline_config
        print(f"\n{'#'*60}")
        print(f"# TimeQA 抽取流水线")
        print(f"# 数据集: {config.split}")
        print(f"# 模式: {config.mode}" + (f" (文档索引: {config.doc_index})" if config.mode == "single" else ""))
        print(f"# 阶段: {STAGE_NAMES[config.start_stage]} -> {STAGE_NAMES[config.end_stage]}")
        print(f"{'#'*60}")
        
        results = {}
        
        # 中间数据（用于阶段间传递）
        chunks_data = None
        events_data = None
        disambiguate_data = None
        timeline_data = None
        
        # 执行各阶段
        for stage in Stage:
            if stage < config.start_stage:
                continue
            if stage > config.end_stage:
                break
            
            if stage == Stage.CHUNK:
                chunks_data = self.run_chunk_stage()
                results["chunk"] = {"num_chunks": len(chunks_data)}
                
            elif stage == Stage.EVENT:
                events_data = self.run_event_stage(chunks_data)
                results["event"] = {"num_events": len(events_data)}
                
            elif stage == Stage.DISAMBIGUATE:
                disambiguate_data = self.run_disambiguate_stage(events_data)
                results["disambiguate"] = {"num_clusters": disambiguate_data["num_clusters"]}
                
            elif stage == Stage.TIMELINE:
                timeline_data = self.run_timeline_stage(events_data, disambiguate_data)
                results["timeline"] = {
                    "num_entities": timeline_data["num_entities"],
                    "num_timelines": timeline_data["num_timelines"],
                }
                
            elif stage == Stage.GRAPH:
                graph_result = self.run_graph_stage(events_data, disambiguate_data, timeline_data)
                results["graph"] = graph_result
        
        print(f"\n{'='*60}")
        print("流水线执行完成!")
        print(f"{'='*60}")
        
        return results


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TimeQA 抽取流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文档测试（验证链路）
  python -m timeqa_agent.pipeline --split test --mode single --doc-index 0
  
  # 全量处理 test 集
  python -m timeqa_agent.pipeline --split test --mode full
  
  # 从事件抽取阶段开始（使用已有的分块结果）
  python -m timeqa_agent.pipeline --split test --start event
  
  # 只执行分块和事件抽取
  python -m timeqa_agent.pipeline --split test --start chunk --end event
        """
    )
    
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["test", "train", "validation"],
        default="test",
        help="数据集分割 (default: test)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["single", "full"],
        default="single",
        help="处理模式: single(单文档验证) / full(全量处理) (default: single)"
    )
    
    parser.add_argument(
        "--doc-index", "-d",
        type=int,
        default=0,
        help="单文档模式下的文档索引 (default: 0)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        choices=list(STAGE_FROM_NAME.keys()),
        default="chunk",
        help="起始阶段 (default: chunk)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        choices=list(STAGE_FROM_NAME.keys()),
        default="graph",
        help="结束阶段 (default: graph)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/timeqa",
        help="数据目录 (default: data/timeqa)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径"
    )
    
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    
    # 构建流水线配置
    pipeline_config = PipelineConfig(
        split=args.split,
        start_stage=STAGE_FROM_NAME[args.start],
        end_stage=STAGE_FROM_NAME[args.end],
        mode=args.mode,
        doc_index=args.doc_index,
        data_dir=args.data_dir,
    )
    
    # 加载 TimeQA 配置
    timeqa_config = load_config(args.config) if args.config else TimeQAConfig()
    
    # 创建并执行流水线
    pipeline = ExtractionPipeline(pipeline_config, timeqa_config)
    results = pipeline.run()
    
    # 打印结果摘要
    print("\n结果摘要:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
