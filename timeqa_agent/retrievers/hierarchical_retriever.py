"""
三层递进检索器 (Hierarchical Retriever)

检索流程：
1. 第一层：使用混合检索获取 top-k1 实体
2. 第二层：通过图存储收集这些实体关联的所有时间线和事件
3. 第三层：在收集到的候选集中，使用混合检索筛选 top-k2 时间线 + top-k3 事件

与 VotingRetriever 的区别：
- VotingRetriever：三层并行检索 → 投票聚合
- HierarchicalRetriever：三层递进过滤 → 逐步缩小检索范围
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Tuple, Callable, Set

from .base import (
    BaseRetriever,
    RetrievalResult,
    EntityResult,
    EventResult,
    TimelineResult,
    HierarchicalEventResult,
    HierarchicalTimelineResult,
    HierarchicalRetrievalResults,
)
from .hybrid_retriever import HybridRetriever
from ..config import RetrieverConfig, HierarchicalConfig


logger = logging.getLogger(__name__)


class HierarchicalRetriever(BaseRetriever):
    """三层递进检索器

    第一层：检索实体 → 第二层：收集关联数据 → 第三层：候选集内检索
    """

    def __init__(
        self,
        graph_store,
        embed_fn: Optional[Callable] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        hierarchical_config: Optional[HierarchicalConfig] = None,
        index_dir: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        初始化三层递进检索器

        Args:
            graph_store: TimelineGraphStore 实例
            embed_fn: 嵌入函数 (texts: List[str]) -> List[List[float]]
            retriever_config: 检索器通用配置
            hierarchical_config: 三层检索专用配置
            index_dir: 索引保存目录
            auto_save: 是否自动保存索引
        """
        super().__init__(graph_store, retriever_config)

        self.hierarchical_config = hierarchical_config or HierarchicalConfig()

        # 复用混合检索器作为底层检索引擎
        self._hybrid_retriever = HybridRetriever(
            graph_store=graph_store,
            embed_fn=embed_fn,
            config=retriever_config,
            index_dir=index_dir,
            auto_save=auto_save,
        )

    # ========== 公共接口 ==========

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        k1: Optional[int] = None,
        k2: Optional[int] = None,
        k3: Optional[int] = None,
        include_intermediate: Optional[bool] = None,
        **kwargs,
    ) -> HierarchicalRetrievalResults:
        """
        执行三层递进检索

        Args:
            query: 查询字符串
            top_k: 未使用，保留以兼容 BaseRetriever 接口
            k1: 第一层实体数量（覆盖配置值）
            k2: 第三层时间线数量（覆盖配置值）
            k3: 第三层事件数量（覆盖配置值）
            include_intermediate: 是否返回中间层结果（覆盖配置值）

        Returns:
            HierarchicalRetrievalResults
        """
        cfg = self.hierarchical_config
        k1 = k1 if k1 is not None else cfg.k1_entities
        k2 = k2 if k2 is not None else cfg.k2_timelines
        k3 = k3 if k3 is not None else cfg.k3_events
        include_intermediate = (
            include_intermediate if include_intermediate is not None
            else cfg.include_intermediate_results
        )

        logger.info(f"三层递进检索开始: query='{query}', k1={k1}, k2={k2}, k3={k3}")

        # ── 第一层：检索实体 ──
        layer1_entities = self._layer1_search_entities(query, k1)
        logger.info(f"第一层检索到 {len(layer1_entities)} 个实体")

        if not layer1_entities:
            logger.warning("第一层未检索到任何实体，返回空结果")
            return HierarchicalRetrievalResults()

        # ── 第二层：收集关联数据 ──
        layer2_timelines, layer2_events = self._layer2_collect_related(layer1_entities)
        logger.info(
            f"第二层收集到 {len(layer2_timelines)} 条时间线, "
            f"{len(layer2_events)} 个事件"
        )
        if include_intermediate and layer2_timelines:
            timeline_names = [tl.timeline_name for tl in layer2_timelines]
            logger.info(f"第二层收集到的时间线名称: {timeline_names}")

        # ── 第三层：候选集内检索 ──
        final_timelines = self._layer3_search_timelines(
            query, layer2_timelines, k2, layer1_entities
        )
        final_events = self._layer3_search_events(
            query, layer2_events, k3, layer1_entities
        )
        logger.info(
            f"第三层筛选出 {len(final_timelines)} 条时间线, "
            f"{len(final_events)} 个事件"
        )

        # 构建结果
        results = HierarchicalRetrievalResults(
            events=final_events,
            timelines=final_timelines,
        )

        if include_intermediate:
            results.layer1_entities = layer1_entities
            results.layer2_all_timelines = layer2_timelines
            results.layer2_all_events = layer2_events

        return results

    def retrieve_with_details(
        self,
        query: str,
        k1: Optional[int] = None,
        k2: Optional[int] = None,
        k3: Optional[int] = None,
    ) -> HierarchicalRetrievalResults:
        """带中间层详情的检索（调试用）"""
        return self.retrieve(
            query=query,
            k1=k1,
            k2=k2,
            k3=k3,
            include_intermediate=True,
        )

    def invalidate_cache(self) -> None:
        """清除底层检索器缓存"""
        self._hybrid_retriever.invalidate_cache()

    # ========== 第一层：检索实体 ==========

    def _layer1_search_entities(
        self, query: str, k1: int
    ) -> List[EntityResult]:
        """第一层：使用混合检索获取 top-k1 实体"""
        threshold = self.hierarchical_config.entity_score_threshold

        results = self._hybrid_retriever.retrieve(
            query=query,
            top_k=k1,
            target_type="entity",
        )

        entities = []
        for r in results:
            if isinstance(r, EntityResult) and r.score >= threshold:
                entities.append(r)

        return entities[:k1]

    # ========== 第二层：收集关联数据 ==========

    def _layer2_collect_related(
        self, entities: List[EntityResult]
    ) -> Tuple[List[TimelineResult], List[EventResult]]:
        """第二层：通过图存储收集实体关联的所有时间线和事件"""
        all_timelines: List[TimelineResult] = []
        all_events: List[EventResult] = []
        timeline_ids_seen: Set[str] = set()
        event_ids_seen: Set[str] = set()

        for entity in entities:
            canonical_name = entity.canonical_name

            # 收集时间线
            timeline_nodes = self.graph_store.get_entity_timelines(canonical_name)
            for tl_node in timeline_nodes:
                tl_id = tl_node.get("timeline_id", "")
                if tl_id and tl_id not in timeline_ids_seen:
                    all_timelines.append(self._node_to_timeline_result(tl_node))
                    timeline_ids_seen.add(tl_id)

            # 收集事件
            event_nodes = self.graph_store.get_entity_events(canonical_name)
            for ev_node in event_nodes:
                ev_id = ev_node.get("event_id", "")
                if ev_id and ev_id not in event_ids_seen:
                    all_events.append(self._node_to_event_result(ev_node))
                    event_ids_seen.add(ev_id)

        return all_timelines, all_events

    # ========== 第三层：候选集内检索 ==========

    def _layer3_search_timelines(
        self,
        query: str,
        candidate_timelines: List[TimelineResult],
        k2: int,
        source_entities: List[EntityResult],
    ) -> List[HierarchicalTimelineResult]:
        """第三层：在候选时间线中检索 top-k2"""
        if not candidate_timelines:
            return []

        threshold = self.hierarchical_config.timeline_score_threshold
        candidate_ids = {t.timeline_id for t in candidate_timelines}

        # 使用混合检索器检索所有时间线，再过滤到候选集
        results = self._hybrid_retriever.retrieve(
            query=query,
            top_k=len(candidate_timelines),
            target_type="timeline",
        )

        # 只保留候选集中的结果
        filtered: List[TimelineResult] = []
        for r in results:
            if isinstance(r, TimelineResult) and r.timeline_id in candidate_ids:
                if r.score >= threshold:
                    filtered.append(r)

        # 补充：候选集中未被混合检索覆盖的时间线（分数为 0）
        covered_ids = {t.timeline_id for t in filtered}
        for tl in candidate_timelines:
            if tl.timeline_id not in covered_ids:
                filtered.append(tl)

        # 按分数排序
        filtered.sort(key=lambda x: x.score, reverse=True)

        # 构建溯源映射：entity_canonical_name -> (name, score)
        entity_map = {e.canonical_name: e.score for e in source_entities}

        hierarchical_results: List[HierarchicalTimelineResult] = []
        for tl in filtered[:k2]:
            # 找出来源实体
            src_names = []
            src_scores = []
            tl_entity = tl.entity_canonical_name
            if tl_entity and tl_entity in entity_map:
                src_names.append(tl_entity)
                src_scores.append(entity_map[tl_entity])

            hierarchical_results.append(HierarchicalTimelineResult(
                node_id=tl.node_id,
                node_type="timeline",
                score=tl.score,
                metadata=tl.metadata,
                timeline_id=tl.timeline_id,
                timeline_name=tl.timeline_name,
                description=tl.description,
                entity_canonical_name=tl.entity_canonical_name,
                time_span_start=tl.time_span_start,
                time_span_end=tl.time_span_end,
                event_ids=tl.event_ids,
                source_entity_names=src_names,
                source_entity_scores=src_scores,
                hierarchical_score=tl.score,
            ))

        return hierarchical_results

    def _layer3_search_events(
        self,
        query: str,
        candidate_events: List[EventResult],
        k3: int,
        source_entities: List[EntityResult],
    ) -> List[HierarchicalEventResult]:
        """第三层：在候选事件中检索 top-k3"""
        if not candidate_events:
            return []

        threshold = self.hierarchical_config.event_score_threshold
        candidate_ids = {e.event_id for e in candidate_events}

        # 使用混合检索器检索所有事件，再过滤到候选集
        results = self._hybrid_retriever.retrieve(
            query=query,
            top_k=len(candidate_events),
            target_type="event",
        )

        # 只保留候选集中的结果
        filtered: List[EventResult] = []
        for r in results:
            if isinstance(r, EventResult) and r.event_id in candidate_ids:
                if r.score >= threshold:
                    filtered.append(r)

        # 补充：候选集中未被混合检索覆盖的事件（分数为 0）
        covered_ids = {e.event_id for e in filtered}
        for ev in candidate_events:
            if ev.event_id not in covered_ids:
                filtered.append(ev)

        # 按分数排序
        filtered.sort(key=lambda x: x.score, reverse=True)

        # 构建溯源映射
        entity_name_set = {e.canonical_name for e in source_entities}
        entity_map = {e.canonical_name: e.score for e in source_entities}

        hierarchical_results: List[HierarchicalEventResult] = []
        for ev in filtered[:k3]:
            # 找出来源实体
            src_names = []
            src_scores = []
            for en in ev.entity_names:
                if en in entity_name_set:
                    src_names.append(en)
                    src_scores.append(entity_map[en])

            hierarchical_results.append(HierarchicalEventResult(
                node_id=ev.node_id,
                node_type="event",
                score=ev.score,
                metadata=ev.metadata,
                event_id=ev.event_id,
                event_description=ev.event_description,
                time_type=ev.time_type,
                time_start=ev.time_start,
                time_end=ev.time_end,
                time_expression=ev.time_expression,
                entity_names=ev.entity_names,
                original_sentence=ev.original_sentence,
                chunk_id=ev.chunk_id,
                doc_id=ev.doc_id,
                doc_title=ev.doc_title,
                source_entity_names=src_names,
                source_entity_scores=src_scores,
                hierarchical_score=ev.score,
            ))

        return hierarchical_results

    # ========== 内部工具方法 ==========

    @staticmethod
    def _node_to_timeline_result(node: Dict) -> TimelineResult:
        """将图存储节点数据转换为 TimelineResult"""
        tl_id = node.get("timeline_id", "")
        return TimelineResult(
            node_id=f"timeline:{tl_id}",
            node_type="timeline",
            timeline_id=tl_id,
            timeline_name=node.get("timeline_name", ""),
            description=node.get("description", ""),
            entity_canonical_name=node.get("entity_canonical_name", ""),
            time_span_start=node.get("time_span_start", ""),
            time_span_end=node.get("time_span_end", ""),
            event_ids=node.get("event_ids", []),
            score=0.0,
        )

    @staticmethod
    def _node_to_event_result(node: Dict) -> EventResult:
        """将图存储节点数据转换为 EventResult"""
        ev_id = node.get("event_id", "")
        return EventResult(
            node_id=f"event:{ev_id}",
            node_type="event",
            event_id=ev_id,
            event_description=node.get("event_description", ""),
            time_type=node.get("time_type", ""),
            time_start=node.get("time_start", ""),
            time_end=node.get("time_end", ""),
            time_expression=node.get("time_expression", ""),
            entity_names=node.get("entity_names", []),
            original_sentence=node.get("original_sentence", ""),
            chunk_id=node.get("chunk_id", ""),
            doc_id=node.get("doc_id", ""),
            doc_title=node.get("doc_title", ""),
            score=0.0,
        )
