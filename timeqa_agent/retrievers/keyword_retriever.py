"""
关键词检索器

支持精确匹配、模糊匹配和 TF-IDF 排序
将多个字段拼接进行关键词检索
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Any, Literal

from .base import (
    BaseRetriever,
    RetrievalResult,
    EntityResult,
    EventResult,
    TimelineResult,
)
from ..config import RetrieverConfig
from ..graph_store import (
    TimelineGraphStore,
    NODE_TYPE_ENTITY,
    NODE_TYPE_EVENT,
    NODE_TYPE_TIMELINE,
)


class TFIDFIndex:
    """简单的 TF-IDF 索引实现"""
    
    def __init__(self):
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: List[Dict[str, float]] = []
        self._built = False
    
    def add_document(self, doc_id: str, text: str) -> None:
        """添加文档"""
        self.documents.append(text.lower())
        self.doc_ids.append(doc_id)
        self._built = False
    
    def build(self) -> None:
        """构建索引"""
        if not self.documents:
            return
        
        import math
        
        # 计算词频
        doc_term_freqs = []
        doc_count = len(self.documents)
        term_doc_count: Dict[str, int] = {}
        
        for doc in self.documents:
            terms = self._tokenize(doc)
            term_freq: Dict[str, int] = {}
            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
            
            for term in set(terms):
                term_doc_count[term] = term_doc_count.get(term, 0) + 1
            
            doc_term_freqs.append(term_freq)
        
        # 计算 IDF
        for term, count in term_doc_count.items():
            self.idf[term] = math.log((doc_count + 1) / (count + 1)) + 1
        
        # 计算 TF-IDF
        for term_freq in doc_term_freqs:
            tfidf: Dict[str, float] = {}
            max_freq = max(term_freq.values()) if term_freq else 1
            for term, freq in term_freq.items():
                tf = freq / max_freq
                tfidf[term] = tf * self.idf.get(term, 1.0)
            self.tfidf_matrix.append(tfidf)
        
        self._built = True
    
    def query(self, query_str: str, top_k: int = 10) -> List[tuple]:
        """
        查询
        
        Returns:
            List of (doc_id, score) tuples
        """
        if not self._built:
            self.build()
        
        if not self.tfidf_matrix:
            return []
        
        query_terms = self._tokenize(query_str.lower())
        if not query_terms:
            return []
        
        # 计算查询向量
        query_tfidf: Dict[str, float] = {}
        term_freq: Dict[str, int] = {}
        for term in query_terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        
        max_freq = max(term_freq.values())
        for term, freq in term_freq.items():
            if term in self.idf:
                tf = freq / max_freq
                query_tfidf[term] = tf * self.idf[term]
        
        # 计算余弦相似度
        scores = []
        for i, doc_tfidf in enumerate(self.tfidf_matrix):
            score = self._cosine_similarity(query_tfidf, doc_tfidf)
            if score > 0:
                scores.append((self.doc_ids[i], score))
        
        # 排序并返回 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单的空格和标点分词
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) >= 2]
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        # 点积
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))
        
        # 模长
        norm1 = sum(v ** 2 for v in vec1.values()) ** 0.5
        norm2 = sum(v ** 2 for v in vec2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class KeywordRetriever(BaseRetriever):
    """
    关键词检索器
    
    支持对实体、事件、时间线进行关键词匹配检索
    将多个字段拼接后进行匹配
    """
    
    def __init__(
        self,
        graph_store: TimelineGraphStore,
        config: Optional[RetrieverConfig] = None
    ):
        super().__init__(graph_store, config or RetrieverConfig())
        
        # TF-IDF 索引缓存
        self._entity_tfidf: Optional[TFIDFIndex] = None
        self._event_tfidf: Optional[TFIDFIndex] = None
        self._timeline_tfidf: Optional[TFIDFIndex] = None
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        target_type: Optional[Literal["entity", "event", "timeline", "all"]] = "all",
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行关键词检索
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            target_type: 目标类型
            
        Returns:
            检索结果列表
        """
        top_k = self._get_top_k(top_k)
        results = []
        
        if target_type in ("entity", "all"):
            results.extend(self.search_entities(query, top_k))
        
        if target_type in ("event", "all"):
            results.extend(self.search_events(query, top_k))
        
        if target_type in ("timeline", "all"):
            results.extend(self.search_timelines(query, top_k))
        
        # 排序并截断
        results = self._sort_by_score(results)
        return results[:top_k]
    
    def search_entities(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_tfidf: Optional[bool] = None
    ) -> List[EntityResult]:
        """
        搜索实体
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            use_tfidf: 是否使用 TF-IDF
            
        Returns:
            实体结果列表
        """
        top_k = self._get_top_k(top_k)
        use_tfidf = use_tfidf if use_tfidf is not None else self.config.use_tfidf
        
        if use_tfidf:
            return self._search_entities_tfidf(query, top_k)
        else:
            return self._search_entities_match(query, top_k)
    
    def search_events(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_tfidf: Optional[bool] = None
    ) -> List[EventResult]:
        """
        搜索事件
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            use_tfidf: 是否使用 TF-IDF
            
        Returns:
            事件结果列表
        """
        top_k = self._get_top_k(top_k)
        use_tfidf = use_tfidf if use_tfidf is not None else self.config.use_tfidf
        
        if use_tfidf:
            return self._search_events_tfidf(query, top_k)
        else:
            return self._search_events_match(query, top_k)
    
    def search_timelines(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_tfidf: Optional[bool] = None
    ) -> List[TimelineResult]:
        """
        搜索时间线
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            use_tfidf: 是否使用 TF-IDF
            
        Returns:
            时间线结果列表
        """
        top_k = self._get_top_k(top_k)
        use_tfidf = use_tfidf if use_tfidf is not None else self.config.use_tfidf
        
        if use_tfidf:
            return self._search_timelines_tfidf(query, top_k)
        else:
            return self._search_timelines_match(query, top_k)
    
    # ========== 内部方法：TF-IDF 检索 ==========
    
    def _build_entity_tfidf(self) -> TFIDFIndex:
        """构建实体 TF-IDF 索引"""
        if self._entity_tfidf is not None:
            return self._entity_tfidf
        
        index = TFIDFIndex()
        
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_ENTITY:
                continue
            
            # 拼接：实体名 + 描述 + 别名
            canonical_name = data.get("canonical_name", "")
            description = data.get("description", "")
            aliases_str = data.get("aliases", "[]")
            aliases = json.loads(aliases_str) if aliases_str else []
            
            text = " ".join([canonical_name, description] + aliases)
            index.add_document(node_id, text)
        
        index.build()
        self._entity_tfidf = index
        return index
    
    def _build_event_tfidf(self) -> TFIDFIndex:
        """构建事件 TF-IDF 索引"""
        if self._event_tfidf is not None:
            return self._event_tfidf
        
        index = TFIDFIndex()
        
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_EVENT:
                continue
            
            # 拼接：事件描述 + 时间表达式 + 原始句子 + 实体名
            event_description = data.get("event_description", "")
            time_expression = data.get("time_expression", "") or ""
            original_sentence = data.get("original_sentence", "")
            entity_names_str = data.get("entity_names", "[]")
            entity_names = json.loads(entity_names_str) if entity_names_str else []
            
            text = " ".join([event_description, time_expression, original_sentence] + entity_names)
            index.add_document(node_id, text)
        
        index.build()
        self._event_tfidf = index
        return index
    
    def _build_timeline_tfidf(self) -> TFIDFIndex:
        """构建时间线 TF-IDF 索引"""
        if self._timeline_tfidf is not None:
            return self._timeline_tfidf
        
        index = TFIDFIndex()
        
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_TIMELINE:
                continue
            
            # 拼接：时间线名称 + 描述 + 实体名
            timeline_name = data.get("timeline_name", "")
            description = data.get("description", "")
            entity_name = data.get("entity_canonical_name", "")
            
            text = " ".join([timeline_name, description, entity_name])
            index.add_document(node_id, text)
        
        index.build()
        self._timeline_tfidf = index
        return index
    
    def _search_entities_tfidf(self, query: str, top_k: int) -> List[EntityResult]:
        """使用 TF-IDF 搜索实体"""
        index = self._build_entity_tfidf()
        results = index.query(query, top_k)
        
        entity_results = []
        for node_id, score in results:
            data = dict(self.graph_store.graph.nodes[node_id])
            entity_results.append(self._node_to_entity_result(node_id, data, score))
        
        return entity_results
    
    def _search_events_tfidf(self, query: str, top_k: int) -> List[EventResult]:
        """使用 TF-IDF 搜索事件"""
        index = self._build_event_tfidf()
        results = index.query(query, top_k)
        
        event_results = []
        for node_id, score in results:
            data = dict(self.graph_store.graph.nodes[node_id])
            event_results.append(self._node_to_event_result(node_id, data, score))
        
        return event_results
    
    def _search_timelines_tfidf(self, query: str, top_k: int) -> List[TimelineResult]:
        """使用 TF-IDF 搜索时间线"""
        index = self._build_timeline_tfidf()
        results = index.query(query, top_k)
        
        timeline_results = []
        for node_id, score in results:
            data = dict(self.graph_store.graph.nodes[node_id])
            timeline_results.append(self._node_to_timeline_result(node_id, data, score))
        
        return timeline_results
    
    # ========== 内部方法：简单匹配检索 ==========
    
    def _search_entities_match(self, query: str, top_k: int) -> List[EntityResult]:
        """使用简单匹配搜索实体"""
        query_lower = query.lower() if not self.config.case_sensitive else query
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        
        results = []
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_ENTITY:
                continue
            
            # 拼接文本
            canonical_name = data.get("canonical_name", "")
            description = data.get("description", "")
            aliases_str = data.get("aliases", "[]")
            aliases = json.loads(aliases_str) if aliases_str else []
            
            text = " ".join([canonical_name, description] + aliases)
            text_lower = text.lower() if not self.config.case_sensitive else text
            
            # 计算匹配分数
            score = self._calculate_match_score(query_lower, query_terms, text_lower)
            if score > 0:
                results.append(self._node_to_entity_result(node_id, data, score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _search_events_match(self, query: str, top_k: int) -> List[EventResult]:
        """使用简单匹配搜索事件"""
        query_lower = query.lower() if not self.config.case_sensitive else query
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        
        results = []
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_EVENT:
                continue
            
            # 拼接文本
            event_description = data.get("event_description", "")
            time_expression = data.get("time_expression", "") or ""
            original_sentence = data.get("original_sentence", "")
            entity_names_str = data.get("entity_names", "[]")
            entity_names = json.loads(entity_names_str) if entity_names_str else []
            
            text = " ".join([event_description, time_expression, original_sentence] + entity_names)
            text_lower = text.lower() if not self.config.case_sensitive else text
            
            # 计算匹配分数
            score = self._calculate_match_score(query_lower, query_terms, text_lower)
            if score > 0:
                results.append(self._node_to_event_result(node_id, data, score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _search_timelines_match(self, query: str, top_k: int) -> List[TimelineResult]:
        """使用简单匹配搜索时间线"""
        query_lower = query.lower() if not self.config.case_sensitive else query
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        
        results = []
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_TIMELINE:
                continue
            
            # 拼接文本
            timeline_name = data.get("timeline_name", "")
            description = data.get("description", "")
            entity_name = data.get("entity_canonical_name", "")
            
            text = " ".join([timeline_name, description, entity_name])
            text_lower = text.lower() if not self.config.case_sensitive else text
            
            # 计算匹配分数
            score = self._calculate_match_score(query_lower, query_terms, text_lower)
            if score > 0:
                results.append(self._node_to_timeline_result(node_id, data, score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _calculate_match_score(
        self,
        query: str,
        query_terms: Set[str],
        text: str
    ) -> float:
        """
        计算匹配分数
        
        结合精确匹配和词项匹配
        """
        score = 0.0
        
        # 精确匹配（整个查询字符串）
        if query in text:
            score += 1.0
        
        # 词项匹配
        text_terms = set(re.findall(r'\b\w+\b', text))
        matched_terms = query_terms & text_terms
        
        if query_terms:
            term_score = len(matched_terms) / len(query_terms)
            score += term_score * 0.5
        
        # 模糊匹配（子串）
        if self.config.fuzzy_match:
            for term in query_terms:
                if len(term) >= self.config.min_keyword_length and term in text:
                    score += 0.1
        
        return score
    
    # ========== 结果转换方法 ==========
    
    def _node_to_entity_result(
        self,
        node_id: str,
        data: Dict[str, Any],
        score: float
    ) -> EntityResult:
        """将节点数据转换为实体结果"""
        aliases_str = data.get("aliases", "[]")
        aliases = json.loads(aliases_str) if aliases_str else []
        
        event_ids_str = data.get("source_event_ids", "[]")
        event_ids = json.loads(event_ids_str) if event_ids_str else []
        
        return EntityResult(
            node_id=node_id,
            node_type="entity",
            score=score,
            canonical_name=data.get("canonical_name", ""),
            description=data.get("description", ""),
            aliases=aliases,
            source_event_ids=event_ids,
            metadata={
                "cluster_id": data.get("cluster_id", ""),
            }
        )
    
    def _node_to_event_result(
        self,
        node_id: str,
        data: Dict[str, Any],
        score: float
    ) -> EventResult:
        """将节点数据转换为事件结果"""
        entity_names_str = data.get("entity_names", "[]")
        entity_names = json.loads(entity_names_str) if entity_names_str else []
        
        return EventResult(
            node_id=node_id,
            node_type="event",
            score=score,
            event_id=data.get("event_id", ""),
            event_description=data.get("event_description", ""),
            time_type=data.get("time_type", ""),
            time_start=data.get("time_start", ""),
            time_end=data.get("time_end", ""),
            time_expression=data.get("time_expression", ""),
            entity_names=entity_names,
            original_sentence=data.get("original_sentence", ""),
            chunk_id=data.get("chunk_id", ""),
            doc_id=data.get("doc_id", ""),
            doc_title=data.get("doc_title", ""),
        )
    
    def _node_to_timeline_result(
        self,
        node_id: str,
        data: Dict[str, Any],
        score: float
    ) -> TimelineResult:
        """将节点数据转换为时间线结果"""
        event_ids_str = data.get("event_ids", "[]")
        event_ids = json.loads(event_ids_str) if event_ids_str else []
        
        return TimelineResult(
            node_id=node_id,
            node_type="timeline",
            score=score,
            timeline_id=data.get("timeline_id", ""),
            timeline_name=data.get("timeline_name", ""),
            description=data.get("description", ""),
            entity_canonical_name=data.get("entity_canonical_name", ""),
            time_span_start=data.get("time_span_start", ""),
            time_span_end=data.get("time_span_end", ""),
            event_ids=event_ids,
        )
    
    def invalidate_cache(self) -> None:
        """清除 TF-IDF 索引缓存"""
        self._entity_tfidf = None
        self._event_tfidf = None
        self._timeline_tfidf = None
