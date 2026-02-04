"""
混合检索器

融合关键词和语义检索策略，支持多种融合模式
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable, Union

from .base import (
    BaseRetriever,
    RetrievalResult,
    EntityResult,
    EventResult,
    TimelineResult,
    RetrievalMode,
)
from .keyword_retriever import KeywordRetriever
from .semantic_retriever import SemanticRetriever
from ..config import RetrieverConfig, FusionMode
from ..graph_store import TimelineGraphStore


class HybridRetriever(BaseRetriever):
    """
    混合检索器
    
    融合关键词和语义检索策略
    """
    
    def __init__(
        self,
        graph_store: TimelineGraphStore,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        config: Optional[RetrieverConfig] = None,
        index_dir: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        初始化混合检索器
        
        Args:
            graph_store: 图存储
            embed_fn: 嵌入函数（用于语义检索）
            config: 检索器配置
            index_dir: 索引缓存目录，设置后会自动加载/保存索引
            auto_save: 是否在首次构建索引后自动保存
        """
        super().__init__(graph_store, config or RetrieverConfig())
        
        # 初始化子检索器
        self._keyword_retriever: Optional[KeywordRetriever] = None
        self._semantic_retriever: Optional[SemanticRetriever] = None
        
        self._embed_fn = embed_fn
        self._index_dir = index_dir
        self._auto_save = auto_save
        
        # 延迟初始化
        self._init_retrievers()
    
    def _init_retrievers(self) -> None:
        """初始化子检索器"""
        if self.config.enable_keyword:
            self._keyword_retriever = KeywordRetriever(
                self.graph_store,
                self.config
            )
        
        if self.config.enable_semantic and self._embed_fn:
            self._semantic_retriever = SemanticRetriever(
                self.graph_store,
                self._embed_fn,
                self.config,
                index_dir=self._index_dir,
                auto_save=self._auto_save,
            )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        target_type: Optional[str] = "all",
        fusion_mode: Optional[FusionMode] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行混合检索
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            target_type: 目标类型
            fusion_mode: 融合模式
            
        Returns:
            检索结果列表
        """
        top_k = self._get_top_k(top_k)
        fusion_mode = fusion_mode or self.config.get_fusion_mode()
        
        # 收集各检索器的结果
        all_results: Dict[str, List[RetrievalResult]] = {}
        
        # 关键词检索
        if self._keyword_retriever:
            keyword_results = self._keyword_retriever.retrieve(
                query, top_k * 2, target_type=target_type
            )
            if keyword_results:
                all_results["keyword"] = keyword_results
        
        # 语义检索
        if self._semantic_retriever:
            semantic_results = self._semantic_retriever.retrieve(
                query, top_k * 2, target_type=target_type
            )
            if semantic_results:
                all_results["semantic"] = semantic_results
        
        # 如果没有任何结果，返回空列表
        if not all_results:
            return []
        
        # 融合结果
        fused_results = self._fuse_results(all_results, fusion_mode)
        
        # 排序并截断
        fused_results = self._sort_by_score(fused_results)
        return fused_results[:top_k]
    
    def search_with_entity_context(
        self,
        query: str,
        entity_name: str,
        top_k: Optional[int] = None
    ) -> List[EventResult]:
        """
        带实体上下文的检索
        
        先锚定实体，再在其事件中进行语义/关键词检索
        
        Args:
            query: 查询字符串
            entity_name: 实体名称
            top_k: 返回数量
            
        Returns:
            事件结果列表
        """
        top_k = self._get_top_k(top_k)
        
        # 获取实体的所有事件
        entity_events = self.graph_store.get_entity_events(entity_name)
        
        if not entity_events:
            return []
        
        # 在事件中进行关键词匹配
        query_lower = query.lower()
        scored_events = []
        
        for event_data in entity_events:
            # 构建事件结果
            event = EventResult(
                node_id=f"event:{event_data.get('event_id', '')}",
                node_type="event",
                score=0.0,
                event_id=event_data.get("event_id", ""),
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
            )
            
            # 计算查询与事件的相关性
            text = event.get_searchable_text().lower()
            
            # 简单的关键词匹配分数
            score = 0.0
            query_terms = query_lower.split()
            for term in query_terms:
                if term in text:
                    score += 1.0
            
            if query_terms:
                score = score / len(query_terms)
            
            event.score = score
            scored_events.append(event)
        
        # 排序
        scored_events.sort(key=lambda x: x.score, reverse=True)
        return scored_events[:top_k]
    
    def search_semantic_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        target_type: Optional[str] = "all"
    ) -> List[RetrievalResult]:
        """
        仅使用语义检索
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            target_type: 目标类型
            
        Returns:
            检索结果列表
        """
        top_k = self._get_top_k(top_k)
        
        if self._semantic_retriever:
            return self._semantic_retriever.retrieve(query, top_k, target_type=target_type)
        
        return []
    
    def search_keyword_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        target_type: Optional[str] = "all"
    ) -> List[RetrievalResult]:
        """
        仅使用关键词检索
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            target_type: 目标类型
            
        Returns:
            检索结果列表
        """
        top_k = self._get_top_k(top_k)
        
        if self._keyword_retriever:
            return self._keyword_retriever.retrieve(query, top_k, target_type=target_type)
        
        return []
    
    # ========== 融合方法 ==========
    
    def _fuse_results(
        self,
        results_dict: Dict[str, List[RetrievalResult]],
        fusion_mode: FusionMode
    ) -> List[RetrievalResult]:
        """
        融合多个检索器的结果
        
        Args:
            results_dict: 各检索器的结果
            fusion_mode: 融合模式
            
        Returns:
            融合后的结果列表
        """
        if fusion_mode == FusionMode.RRF:
            return self._fuse_rrf(results_dict)
        elif fusion_mode == FusionMode.WEIGHTED_SUM:
            return self._fuse_weighted_sum(results_dict)
        elif fusion_mode == FusionMode.MAX_SCORE:
            return self._fuse_max_score(results_dict)
        elif fusion_mode == FusionMode.INTERLEAVE:
            return self._fuse_interleave(results_dict)
        else:
            return self._fuse_rrf(results_dict)
    
    def _fuse_rrf(
        self,
        results_dict: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """
        倒数排名融合 (Reciprocal Rank Fusion)
        
        RRF 分数 = Σ 1 / (k + rank)
        """
        k = self.config.rrf_k
        
        # 收集所有结果的 RRF 分数
        node_scores: Dict[str, float] = {}
        node_results: Dict[str, RetrievalResult] = {}
        
        for retriever_name, results in results_dict.items():
            for rank, result in enumerate(results):
                node_id = result.node_id
                rrf_score = 1.0 / (k + rank + 1)
                
                node_scores[node_id] = node_scores.get(node_id, 0) + rrf_score
                
                # 保留最高分的结果对象
                if node_id not in node_results or result.score > node_results[node_id].score:
                    node_results[node_id] = result
        
        # 更新分数并排序
        fused_results = []
        for node_id, rrf_score in node_scores.items():
            result = node_results[node_id]
            result.score = rrf_score
            fused_results.append(result)
        
        return fused_results
    
    def _fuse_weighted_sum(
        self,
        results_dict: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """
        加权求和融合
        """
        weights = {
            "keyword": self.config.keyword_weight,
            "semantic": self.config.semantic_weight,
        }
        
        node_scores: Dict[str, float] = {}
        node_results: Dict[str, RetrievalResult] = {}
        node_weights: Dict[str, float] = {}
        
        for retriever_name, results in results_dict.items():
            weight = weights.get(retriever_name, 1.0)
            
            for result in results:
                node_id = result.node_id
                weighted_score = result.score * weight
                
                node_scores[node_id] = node_scores.get(node_id, 0) + weighted_score
                node_weights[node_id] = node_weights.get(node_id, 0) + weight
                
                if node_id not in node_results:
                    node_results[node_id] = result
        
        # 归一化并更新分数
        fused_results = []
        for node_id, total_score in node_scores.items():
            result = node_results[node_id]
            # 按参与的权重归一化
            result.score = total_score / node_weights[node_id] if node_weights[node_id] > 0 else 0
            fused_results.append(result)
        
        return fused_results
    
    def _fuse_max_score(
        self,
        results_dict: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """
        取最大分数融合
        """
        node_results: Dict[str, RetrievalResult] = {}
        
        for retriever_name, results in results_dict.items():
            for result in results:
                node_id = result.node_id
                
                if node_id not in node_results or result.score > node_results[node_id].score:
                    node_results[node_id] = result
        
        return list(node_results.values())
    
    def _fuse_interleave(
        self,
        results_dict: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """
        交错合并融合
        """
        seen = set()
        fused_results = []
        
        # 获取最大长度
        max_len = max(len(results) for results in results_dict.values()) if results_dict else 0
        
        for i in range(max_len):
            for retriever_name, results in results_dict.items():
                if i < len(results):
                    result = results[i]
                    if result.node_id not in seen:
                        seen.add(result.node_id)
                        fused_results.append(result)
        
        # 重新计算分数（基于位置）
        for i, result in enumerate(fused_results):
            result.score = 1.0 - (i / len(fused_results)) if fused_results else 0
        
        return fused_results
    
    # ========== 工具方法 ==========
    
    def set_embed_fn(
        self,
        embed_fn: Callable[[List[str]], List[List[float]]]
    ) -> None:
        """设置嵌入函数"""
        self._embed_fn = embed_fn
        
        if self.config.enable_semantic:
            self._semantic_retriever = SemanticRetriever(
                self.graph_store,
                embed_fn,
                self.config
            )
    
    def build_indexes(self) -> None:
        """构建所有索引"""
        if self._keyword_retriever:
            # TF-IDF 索引会在首次查询时自动构建
            pass
        
        if self._semantic_retriever:
            self._semantic_retriever.build_indexes()
    
    def invalidate_cache(self) -> None:
        """清除所有缓存"""
        if self._keyword_retriever:
            self._keyword_retriever.invalidate_cache()
        
        if self._semantic_retriever:
            self._semantic_retriever.invalidate_cache()
