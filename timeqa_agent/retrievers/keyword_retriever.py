"""
关键词检索器

支持多种关键词检索算法:
- BM25: 基于 rank-bm25 库的 BM25 算法
- TF-IDF: 自定义实现的 TF-IDF 算法

支持精确匹配、模糊匹配和排序
将多个字段拼接进行关键词检索
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Any, Literal, Set

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

# 尝试导入 BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


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


class BM25Index:
    """BM25 索引实现（基于 rank-bm25）"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_stemming: bool = False,
        remove_stopwords: bool = False
    ):
        """
        初始化 BM25 索引

        Args:
            k1: BM25 k1 参数（词频饱和度，通常 1.2-2.0）
            b: BM25 b 参数（文档长度归一化，通常 0.75）
            use_stemming: 是否使用词干提取
            remove_stopwords: 是否移除停用词
        """
        if not BM25_AVAILABLE:
            raise ImportError(
                "BM25 需要 rank-bm25 库。请安装: pip install rank-bm25"
            )

        self.k1 = k1
        self.b = b
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords

        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._built = False

        # 可选: 加载 NLTK 词干提取器和停用词
        self._stemmer = None
        self._stopwords = set()

        if use_stemming or remove_stopwords:
            self._init_nltk()

    def _init_nltk(self):
        """初始化 NLTK 组件（如果需要）"""
        try:
            import nltk
            from nltk.stem import PorterStemmer

            if self.use_stemming:
                self._stemmer = PorterStemmer()

            if self.remove_stopwords:
                try:
                    from nltk.corpus import stopwords
                    self._stopwords = set(stopwords.words('english'))
                except LookupError:
                    # 下载停用词
                    nltk.download('stopwords', quiet=True)
                    from nltk.corpus import stopwords
                    self._stopwords = set(stopwords.words('english'))

        except ImportError:
            if self.use_stemming or self.remove_stopwords:
                print("警告: NLTK 未安装，词干提取和停用词移除功能不可用。")
                print("安装方法: pip install nltk")

    def add_document(self, doc_id: str, text: str) -> None:
        """添加文档"""
        self.documents.append(text.lower())
        self.doc_ids.append(doc_id)
        self._built = False

    def build(self) -> None:
        """构建 BM25 索引"""
        if not self.documents:
            return

        # 对所有文档进行分词
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.documents]

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        self._built = True

    def query(self, query_str: str, top_k: int = 10) -> List[tuple]:
        """
        查询

        Returns:
            List of (doc_id, score) tuples
        """
        if not self._built:
            self.build()

        if not self.bm25:
            return []

        # 对查询进行分词
        tokenized_query = self._tokenize(query_str.lower())

        if not tokenized_query:
            return []

        # 计算 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)

        # 获取 top_k 结果
        doc_score_pairs = [(self.doc_ids[i], scores[i]) for i in range(len(scores))]
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return doc_score_pairs[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """
        分词（并可选地进行词干提取和停用词移除）

        Args:
            text: 输入文本

        Returns:
            分词后的列表
        """
        # 基础分词（使用正则表达式）
        tokens = re.findall(r'\b\w+\b', text.lower())

        # 过滤长度
        tokens = [t for t in tokens if len(t) >= 2]

        # 移除停用词
        if self.remove_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        # 词干提取
        if self.use_stemming and self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens


class KeywordRetriever(BaseRetriever):
    """
    关键词检索器

    支持对实体、事件、时间线进行关键词匹配检索
    将多个字段拼接后进行匹配

    支持的算法:
    - BM25: 基于概率的排序函数（推荐）
    - TF-IDF: 基于词频-逆文档频率
    """

    def __init__(
        self,
        graph_store: TimelineGraphStore,
        config: Optional[RetrieverConfig] = None
    ):
        super().__init__(graph_store, config or RetrieverConfig())

        # 检查算法类型
        self.algorithm = self.config.keyword_algorithm.lower()

        # 索引缓存（通用，可以是 BM25Index 或 TFIDFIndex）
        self._entity_index: Optional[Any] = None
        self._event_index: Optional[Any] = None
        self._timeline_index: Optional[Any] = None

    def _create_index(self) -> Any:
        """根据配置创建索引实例"""
        if self.algorithm == "bm25":
            if not BM25_AVAILABLE:
                print("警告: rank-bm25 未安装，回退到 TF-IDF")
                return TFIDFIndex()
            return BM25Index(
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
                use_stemming=self.config.bm25_use_stemming,
                remove_stopwords=self.config.bm25_remove_stopwords
            )
        elif self.algorithm == "tfidf":
            return TFIDFIndex()
        else:
            # 兼容旧配置
            if self.config.use_tfidf:
                return TFIDFIndex()
            else:
                # 默认使用 BM25
                if BM25_AVAILABLE:
                    return BM25Index(k1=self.config.bm25_k1, b=self.config.bm25_b)
                else:
                    return TFIDFIndex()
    
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
            use_tfidf: 已弃用，改用 config.keyword_algorithm

        Returns:
            实体结果列表
        """
        top_k = self._get_top_k(top_k)

        # 根据配置选择算法（兼容旧参数）
        if use_tfidf is not None:
            # 兼容旧代码
            if use_tfidf:
                return self._search_entities_with_index(query, top_k)
            else:
                return self._search_entities_match(query, top_k)
        else:
            # 使用配置的算法
            if self.algorithm in ("bm25", "tfidf"):
                return self._search_entities_with_index(query, top_k)
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
            use_tfidf: 已弃用，改用 config.keyword_algorithm

        Returns:
            事件结果列表
        """
        top_k = self._get_top_k(top_k)

        # 根据配置选择算法
        if use_tfidf is not None:
            if use_tfidf:
                return self._search_events_with_index(query, top_k)
            else:
                return self._search_events_match(query, top_k)
        else:
            if self.algorithm in ("bm25", "tfidf"):
                return self._search_events_with_index(query, top_k)
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
            use_tfidf: 已弃用，改用 config.keyword_algorithm

        Returns:
            时间线结果列表
        """
        top_k = self._get_top_k(top_k)

        # 根据配置选择算法
        if use_tfidf is not None:
            if use_tfidf:
                return self._search_timelines_with_index(query, top_k)
            else:
                return self._search_timelines_match(query, top_k)
        else:
            if self.algorithm in ("bm25", "tfidf"):
                return self._search_timelines_with_index(query, top_k)
            else:
                return self._search_timelines_match(query, top_k)
    
    # ========== 内部方法：索引构建和检索 ==========

    def _build_entity_index(self) -> Any:
        """构建实体索引"""
        if self._entity_index is not None:
            return self._entity_index

        index = self._create_index()

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
        self._entity_index = index
        return index

    def _build_event_index(self) -> Any:
        """构建事件索引"""
        if self._event_index is not None:
            return self._event_index

        index = self._create_index()

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
        self._event_index = index
        return index

    def _build_timeline_index(self) -> Any:
        """构建时间线索引"""
        if self._timeline_index is not None:
            return self._timeline_index

        index = self._create_index()

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
        self._timeline_index = index
        return index

    def _search_entities_with_index(self, query: str, top_k: int) -> List[EntityResult]:
        """使用索引搜索实体"""
        index = self._build_entity_index()
        results = index.query(query, top_k)

        entity_results = []
        for node_id, score in results:
            data = dict(self.graph_store.graph.nodes[node_id])
            entity_results.append(self._node_to_entity_result(node_id, data, score))

        return entity_results

    def _search_events_with_index(self, query: str, top_k: int) -> List[EventResult]:
        """使用索引搜索事件"""
        index = self._build_event_index()
        results = index.query(query, top_k)

        event_results = []
        for node_id, score in results:
            data = dict(self.graph_store.graph.nodes[node_id])
            event_results.append(self._node_to_event_result(node_id, data, score))

        return event_results

    def _search_timelines_with_index(self, query: str, top_k: int) -> List[TimelineResult]:
        """使用索引搜索时间线"""
        index = self._build_timeline_index()
        results = index.query(query, top_k)

        timeline_results = []
        for node_id, score in results:
            data = dict(self.graph_store.graph.nodes[node_id])
            timeline_results.append(self._node_to_timeline_result(node_id, data, score))

        return timeline_results

    # ========== 内部方法：TF-IDF 检索（已弃用，保留兼容） ==========

    def _build_entity_tfidf(self) -> TFIDFIndex:
        """构建实体 TF-IDF 索引（已弃用，使用 _build_entity_index）"""
        return self._build_entity_index()

    def _build_event_tfidf(self) -> TFIDFIndex:
        """构建事件 TF-IDF 索引（已弃用，使用 _build_event_index）"""
        return self._build_event_index()

    def _build_timeline_tfidf(self) -> TFIDFIndex:
        """构建时间线 TF-IDF 索引（已弃用，使用 _build_timeline_index）"""
        return self._build_timeline_index()

    def _search_entities_tfidf(self, query: str, top_k: int) -> List[EntityResult]:
        """使用 TF-IDF 搜索实体（已弃用，使用 _search_entities_with_index）"""
        return self._search_entities_with_index(query, top_k)

    def _search_events_tfidf(self, query: str, top_k: int) -> List[EventResult]:
        """使用 TF-IDF 搜索事件（已弃用，使用 _search_events_with_index）"""
        return self._search_events_with_index(query, top_k)

    def _search_timelines_tfidf(self, query: str, top_k: int) -> List[TimelineResult]:
        """使用 TF-IDF 搜索时间线（已弃用，使用 _search_timelines_with_index）"""
        return self._search_timelines_with_index(query, top_k)
    
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
        """清除索引缓存"""
        self._entity_index = None
        self._event_index = None
        self._timeline_index = None
