"""
多层投票检索器 (Multi-Layer Voting Retriever)

对实体、时间线、事件三层进行检索，通过投票机制聚合得到最终的事件排名。

设计思想：
- 实体层检索：检索相关实体，实体关联的事件获得投票
- 时间线层检索：检索相关时间线，时间线包含的事件获得投票
- 事件层检索：直接检索事件，事件获得直接投票
- 投票聚合：使用 RRF / 加权求和 / 投票计数等算法聚合所有投票
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple

from .base import (
    BaseRetriever,
    RetrievalResult,
    EntityResult,
    EventResult,
    TimelineResult,
)
from .hybrid_retriever import HybridRetriever
from ..config import RetrieverConfig, VotingConfig, VotingFusionMode
from ..graph_store import TimelineGraphStore


@dataclass
class EventVote:
    """事件投票记录"""
    event_id: str                          # 事件ID
    source_type: str                       # 来源类型: "entity" | "timeline" | "event"
    source_id: str                         # 来源节点ID
    source_score: float                    # 来源检索分数
    vote_weight: float                     # 传播后的投票权重
    rank_in_source: int                    # 在来源检索中的排名
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_score": self.source_score,
            "vote_weight": self.vote_weight,
            "rank_in_source": self.rank_in_source,
        }


@dataclass
class VotingEventResult(EventResult):
    """带投票信息的事件结果"""
    vote_count: int = 0                                    # 投票数
    vote_sources: List[Dict[str, Any]] = field(default_factory=list)  # 投票来源详情
    aggregated_score: float = 0.0                          # 聚合后的分数
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "vote_count": self.vote_count,
            "vote_sources": self.vote_sources,
            "aggregated_score": self.aggregated_score,
        })
        return base


class MultiLayerVotingRetriever(BaseRetriever):
    """
    多层投票检索器
    
    通过实体、时间线、事件三层检索，使用投票机制聚合得到最终事件排名。
    
    工作流程：
    1. 并行执行三层检索
    2. 实体层：检索相关实体 → 获取实体关联的事件 → 投票
    3. 时间线层：检索相关时间线 → 获取时间线包含的事件 → 投票
    4. 事件层：直接检索事件 → 直接投票
    5. 聚合所有投票，计算最终分数
    6. 返回 Top-K 事件
    """
    
    def __init__(
        self,
        graph_store: TimelineGraphStore,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        voting_config: Optional[VotingConfig] = None,
        index_dir: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        初始化多层投票检索器
        
        Args:
            graph_store: 图存储
            embed_fn: 嵌入函数（用于语义检索）
            retriever_config: 底层检索器配置
            voting_config: 投票配置
            index_dir: 索引缓存目录
            auto_save: 是否自动保存索引
        """
        super().__init__(graph_store, retriever_config or RetrieverConfig())
        
        self.voting_config = voting_config or VotingConfig()
        
        # 初始化底层混合检索器
        self._hybrid_retriever = HybridRetriever(
            graph_store=graph_store,
            embed_fn=embed_fn,
            config=self.config,
            index_dir=index_dir,
            auto_save=auto_save,
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[VotingEventResult]:
        """
        执行多层投票检索
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            
        Returns:
            带投票信息的事件结果列表
        """
        top_k = self._get_top_k(top_k)
        
        # 1. 三层并行检索
        entity_results, timeline_results, event_results = self._retrieve_all_layers(query, top_k)
        
        # 2. 收集投票
        event_votes = self._collect_votes(entity_results, timeline_results, event_results)
        
        # 3. 聚合投票
        aggregated_scores = self._aggregate_votes(event_votes)
        
        # 4. 构建结果
        results = self._build_results(event_votes, aggregated_scores)
        
        # 5. 过滤和排序
        results = self._filter_results(results)
        results = sorted(results, key=lambda x: x.aggregated_score, reverse=True)
        
        return results[:top_k]
    
    def retrieve_with_details(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        执行检索并返回详细信息（用于调试和分析）
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            
        Returns:
            包含各层检索结果、投票详情、最终结果的字典
        """
        top_k = self._get_top_k(top_k)
        
        # 1. 三层检索
        entity_results, timeline_results, event_results = self._retrieve_all_layers(query, top_k)
        
        # 2. 收集投票
        event_votes = self._collect_votes(entity_results, timeline_results, event_results)
        
        # 3. 聚合投票
        aggregated_scores = self._aggregate_votes(event_votes)
        
        # 4. 构建结果
        results = self._build_results(event_votes, aggregated_scores)
        results = self._filter_results(results)
        results = sorted(results, key=lambda x: x.aggregated_score, reverse=True)[:top_k]
        
        return {
            "query": query,
            "config": {
                "entity_layer_weight": self.voting_config.entity_layer_weight,
                "timeline_layer_weight": self.voting_config.timeline_layer_weight,
                "event_layer_weight": self.voting_config.event_layer_weight,
                "fusion_mode": self.voting_config.fusion_mode.value,
            },
            "layer_results": {
                "entity_layer": [e.to_dict() for e in entity_results],
                "timeline_layer": [t.to_dict() for t in timeline_results],
                "event_layer": [e.to_dict() for e in event_results],
            },
            "vote_details": {
                event_id: [v.to_dict() for v in votes]
                for event_id, votes in event_votes.items()
            },
            "aggregated_scores": aggregated_scores,
            "final_results": [r.to_dict() for r in results],
        }
    
    # ========== 内部方法：检索 ==========
    
    def _retrieve_all_layers(
        self,
        query: str,
        top_k: int
    ) -> Tuple[List[EntityResult], List[TimelineResult], List[EventResult]]:
        """执行三层检索"""
        entity_results = []
        timeline_results = []
        event_results = []
        
        # 实体层检索
        if self.voting_config.enable_entity_layer:
            entity_top_k = int(top_k * self.voting_config.entity_retrieve_multiplier)
            entity_results = self._hybrid_retriever.retrieve(
                query, entity_top_k, target_type="entity"
            )
            entity_results = [r for r in entity_results if isinstance(r, EntityResult)]
        
        # 时间线层检索
        if self.voting_config.enable_timeline_layer:
            timeline_top_k = int(top_k * self.voting_config.timeline_retrieve_multiplier)
            timeline_results = self._hybrid_retriever.retrieve(
                query, timeline_top_k, target_type="timeline"
            )
            timeline_results = [r for r in timeline_results if isinstance(r, TimelineResult)]
        
        # 事件层检索
        if self.voting_config.enable_event_layer:
            event_top_k = int(top_k * self.voting_config.event_retrieve_multiplier)
            event_results = self._hybrid_retriever.retrieve(
                query, event_top_k, target_type="event"
            )
            event_results = [r for r in event_results if isinstance(r, EventResult)]
        
        return entity_results, timeline_results, event_results
    
    # ========== 内部方法：投票收集 ==========
    
    def _collect_votes(
        self,
        entity_results: List[EntityResult],
        timeline_results: List[TimelineResult],
        event_results: List[EventResult],
    ) -> Dict[str, List[EventVote]]:
        """
        收集所有投票
        
        Returns:
            event_id -> List[EventVote] 的映射
        """
        event_votes: Dict[str, List[EventVote]] = defaultdict(list)
        
        # 实体层投票
        for rank, entity in enumerate(entity_results):
            self._add_entity_votes(event_votes, entity, rank)
        
        # 时间线层投票
        for rank, timeline in enumerate(timeline_results):
            self._add_timeline_votes(event_votes, timeline, rank)
        
        # 事件层直接投票
        for rank, event in enumerate(event_results):
            self._add_event_vote(event_votes, event, rank)
        
        return event_votes
    
    def _add_entity_votes(
        self,
        event_votes: Dict[str, List[EventVote]],
        entity: EntityResult,
        rank: int
    ) -> None:
        """添加实体层投票"""
        # 获取实体关联的所有事件
        related_events = self.graph_store.get_entity_events(entity.canonical_name)
        
        for event_data in related_events:
            event_id = event_data.get("event_id", "")
            if not event_id:
                continue
            
            vote = EventVote(
                event_id=event_id,
                source_type="entity",
                source_id=entity.canonical_name,
                source_score=entity.score,
                vote_weight=entity.score * self.voting_config.entity_decay_factor,
                rank_in_source=rank,
            )
            event_votes[event_id].append(vote)
    
    def _add_timeline_votes(
        self,
        event_votes: Dict[str, List[EventVote]],
        timeline: TimelineResult,
        rank: int
    ) -> None:
        """添加时间线层投票"""
        for event_id in timeline.event_ids:
            if not event_id:
                continue
            
            vote = EventVote(
                event_id=event_id,
                source_type="timeline",
                source_id=timeline.timeline_id,
                source_score=timeline.score,
                vote_weight=timeline.score * self.voting_config.timeline_decay_factor,
                rank_in_source=rank,
            )
            event_votes[event_id].append(vote)
    
    def _add_event_vote(
        self,
        event_votes: Dict[str, List[EventVote]],
        event: EventResult,
        rank: int
    ) -> None:
        """添加事件层直接投票"""
        vote = EventVote(
            event_id=event.event_id,
            source_type="event",
            source_id=event.event_id,
            source_score=event.score,
            vote_weight=event.score,  # 直接使用原始分数
            rank_in_source=rank,
        )
        event_votes[event.event_id].append(vote)
    
    # ========== 内部方法：投票聚合 ==========
    
    def _aggregate_votes(
        self,
        event_votes: Dict[str, List[EventVote]]
    ) -> Dict[str, float]:
        """
        聚合投票计算最终分数
        
        Args:
            event_votes: 事件投票映射
            
        Returns:
            event_id -> aggregated_score 的映射
        """
        fusion_mode = self.voting_config.fusion_mode
        
        if fusion_mode == VotingFusionMode.RRF:
            return self._aggregate_rrf(event_votes)
        elif fusion_mode == VotingFusionMode.WEIGHTED:
            return self._aggregate_weighted(event_votes)
        elif fusion_mode == VotingFusionMode.VOTE_COUNT:
            return self._aggregate_vote_count(event_votes)
        elif fusion_mode == VotingFusionMode.BORDA:
            return self._aggregate_borda(event_votes)
        else:
            return self._aggregate_rrf(event_votes)
    
    def _get_layer_weight(self, source_type: str) -> float:
        """获取层权重"""
        weights = {
            "entity": self.voting_config.entity_layer_weight,
            "timeline": self.voting_config.timeline_layer_weight,
            "event": self.voting_config.event_layer_weight,
        }
        return weights.get(source_type, 1.0)
    
    def _aggregate_rrf(
        self,
        event_votes: Dict[str, List[EventVote]]
    ) -> Dict[str, float]:
        """
        RRF (Reciprocal Rank Fusion) 聚合
        
        公式: score = Σ (layer_weight / (k + rank + 1))
        """
        k = self.voting_config.rrf_k
        scores = {}
        
        for event_id, votes in event_votes.items():
            rrf_score = 0.0
            for vote in votes:
                layer_weight = self._get_layer_weight(vote.source_type)
                rrf_score += layer_weight / (k + vote.rank_in_source + 1)
            scores[event_id] = rrf_score
        
        return scores
    
    def _aggregate_weighted(
        self,
        event_votes: Dict[str, List[EventVote]]
    ) -> Dict[str, float]:
        """
        加权投票聚合
        
        公式: score = Σ (vote_weight * layer_weight)
        """
        scores = {}
        
        for event_id, votes in event_votes.items():
            total_weight = 0.0
            for vote in votes:
                layer_weight = self._get_layer_weight(vote.source_type)
                total_weight += vote.vote_weight * layer_weight
            scores[event_id] = total_weight
        
        return scores
    
    def _aggregate_vote_count(
        self,
        event_votes: Dict[str, List[EventVote]]
    ) -> Dict[str, float]:
        """
        投票计数 + 平均分数聚合
        
        公式: score = vote_count * α + avg_weight * β
        """
        alpha = self.voting_config.vote_count_alpha
        beta = self.voting_config.vote_count_beta
        scores = {}
        
        for event_id, votes in event_votes.items():
            vote_count = len(votes)
            avg_weight = sum(v.vote_weight for v in votes) / vote_count if vote_count > 0 else 0
            
            # 归一化投票数（假设最大投票数为 10）
            normalized_count = min(vote_count / 10.0, 1.0)
            
            scores[event_id] = normalized_count * alpha + avg_weight * beta
        
        return scores
    
    def _aggregate_borda(
        self,
        event_votes: Dict[str, List[EventVote]]
    ) -> Dict[str, float]:
        """
        Borda Count 聚合
        
        公式: score = Σ (layer_weight * (N - rank))
        其中 N 是该层的最大排名
        """
        # 找出每层的最大排名
        max_ranks = {"entity": 0, "timeline": 0, "event": 0}
        for votes in event_votes.values():
            for vote in votes:
                max_ranks[vote.source_type] = max(max_ranks[vote.source_type], vote.rank_in_source)
        
        scores = {}
        for event_id, votes in event_votes.items():
            borda_score = 0.0
            for vote in votes:
                layer_weight = self._get_layer_weight(vote.source_type)
                n = max_ranks[vote.source_type] + 1
                borda_score += layer_weight * (n - vote.rank_in_source)
            scores[event_id] = borda_score
        
        return scores
    
    # ========== 内部方法：结果构建 ==========
    
    def _build_results(
        self,
        event_votes: Dict[str, List[EventVote]],
        aggregated_scores: Dict[str, float],
    ) -> List[VotingEventResult]:
        """构建带投票信息的事件结果"""
        results = []
        
        for event_id, votes in event_votes.items():
            # 获取事件详情
            event_data = self.graph_store.get_event(event_id)
            if not event_data:
                continue
            
            # 构建投票来源信息
            vote_sources = []
            for vote in votes:
                vote_sources.append({
                    "source_type": vote.source_type,
                    "source_id": vote.source_id,
                    "source_score": vote.source_score,
                    "vote_weight": vote.vote_weight,
                    "rank_in_source": vote.rank_in_source,
                })
            
            # 构建结果
            result = VotingEventResult(
                node_id=f"event:{event_id}",
                node_type="event",
                score=aggregated_scores.get(event_id, 0.0),
                event_id=event_id,
                event_description=event_data.get("event_description", ""),
                time_type=event_data.get("time_type", ""),
                time_start=event_data.get("time_start", ""),
                time_end=event_data.get("time_end", ""),
                time_expression=event_data.get("time_expression", ""),
                entity_names=event_data.get("entity_names", []),
                original_sentence=event_data.get("original_sentence", ""),
                chunk_id=event_data.get("chunk_id", ""),
                doc_id=event_data.get("doc_id", ""),
                doc_title=event_data.get("doc_title", ""),
                vote_count=len(votes),
                vote_sources=vote_sources,
                aggregated_score=aggregated_scores.get(event_id, 0.0),
            )
            results.append(result)
        
        return results
    
    def _filter_results(
        self,
        results: List[VotingEventResult]
    ) -> List[VotingEventResult]:
        """过滤结果"""
        filtered = []
        for result in results:
            # 检查最少投票数
            if result.vote_count < self.voting_config.min_votes:
                continue
            # 检查最低分数
            if result.aggregated_score < self.voting_config.min_score:
                continue
            filtered.append(result)
        return filtered
    
    # ========== 工具方法 ==========
    
    def set_embed_fn(
        self,
        embed_fn: Callable[[List[str]], List[List[float]]]
    ) -> None:
        """设置嵌入函数"""
        self._hybrid_retriever.set_embed_fn(embed_fn)
    
    def build_indexes(self) -> None:
        """构建所有索引"""
        self._hybrid_retriever.build_indexes()
    
    def invalidate_cache(self) -> None:
        """清除所有缓存"""
        self._hybrid_retriever.invalidate_cache()
    
    def update_voting_config(self, **kwargs) -> None:
        """更新投票配置"""
        for key, value in kwargs.items():
            if hasattr(self.voting_config, key):
                setattr(self.voting_config, key, value)


# ========== 便捷函数 ==========

def create_voting_retriever(
    graph_store: TimelineGraphStore,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    fusion_mode: str = "rrf",
    entity_weight: float = 0.25,
    timeline_weight: float = 0.30,
    event_weight: float = 0.45,
    **kwargs
) -> MultiLayerVotingRetriever:
    """
    便捷函数：创建多层投票检索器
    
    Args:
        graph_store: 图存储
        embed_fn: 嵌入函数
        fusion_mode: 融合模式 ("rrf", "weighted", "vote_count", "borda")
        entity_weight: 实体层权重
        timeline_weight: 时间线层权重
        event_weight: 事件层权重
        **kwargs: 其他配置参数
        
    Returns:
        配置好的 MultiLayerVotingRetriever 实例
    """
    voting_config = VotingConfig(
        entity_layer_weight=entity_weight,
        timeline_layer_weight=timeline_weight,
        event_layer_weight=event_weight,
        fusion_mode=VotingFusionMode(fusion_mode),
        **{k: v for k, v in kwargs.items() if hasattr(VotingConfig, k)}
    )
    
    retriever_config = RetrieverConfig(
        **{k: v for k, v in kwargs.items() if hasattr(RetrieverConfig, k)}
    )
    
    return MultiLayerVotingRetriever(
        graph_store=graph_store,
        embed_fn=embed_fn,
        retriever_config=retriever_config,
        voting_config=voting_config,
        index_dir=kwargs.get("index_dir"),
        auto_save=kwargs.get("auto_save", True),
    )
