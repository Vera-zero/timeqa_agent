"""
语义检索器

基于向量嵌入的语义相似度检索
支持 FAISS 向量索引（Flat / IVF / HNSW）
"""

from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Optional, Any, Literal, Callable, Union
from pathlib import Path

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

# 尝试导入 FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorIndex:
    """
    向量索引
    
    支持 FAISS 后端:
    - flat: 暴力搜索 (IndexFlatIP / IndexFlatL2)
    - ivf: 倒排索引 (IndexIVFFlat)
    - hnsw: 近似最近邻 (IndexHNSWFlat)
    
    如果 FAISS 不可用，回退到 numpy 实现
    """
    
    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._faiss_index: Optional[Any] = None
        self._vectors_matrix: Optional[np.ndarray] = None
        self._built = False
        self._use_faiss = FAISS_AVAILABLE
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        if self.vectors:
            return len(self.vectors[0])
        return self.config.embedding_dim
    
    def add(
        self,
        id: str,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加向量"""
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        
        self.vectors.append(vector)
        self.ids.append(id)
        self.metadata.append(metadata or {})
        self._built = False
    
    def add_batch(
        self,
        ids: List[str],
        vectors: Union[List[List[float]], np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """批量添加向量"""
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            self.vectors.append(vec)
            self.ids.append(id_)
            self.metadata.append(metadata_list[i] if metadata_list else {})
        
        self._built = False
    
    def build(self) -> None:
        """构建索引"""
        if not self.vectors:
            return
        
        self._vectors_matrix = np.vstack(self.vectors).astype(np.float32)
        dim = self._vectors_matrix.shape[1]
        
        # 对于余弦相似度，先归一化
        if self.config.vector_metric == "cosine":
            norms = np.linalg.norm(self._vectors_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._vectors_matrix = self._vectors_matrix / norms
        
        if self._use_faiss:
            self._build_faiss_index(dim)
        
        self._built = True
    
    def _build_faiss_index(self, dim: int) -> None:
        """构建 FAISS 索引"""
        n_vectors = len(self.vectors)
        index_type = self.config.vector_index_type
        metric = self.config.vector_metric
        
        # 选择距离度量
        # 对于余弦相似度，数据已归一化，使用内积
        if metric in ("cosine", "ip"):
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:  # L2
            faiss_metric = faiss.METRIC_L2
        
        # 根据索引类型创建索引
        if index_type == "hnsw":
            # HNSW 索引
            self._faiss_index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m, faiss_metric)
            self._faiss_index.hnsw.efConstruction = self.config.hnsw_ef_construction
            self._faiss_index.hnsw.efSearch = self.config.hnsw_ef_search
        elif index_type == "ivf" and n_vectors >= 100:
            # IVF 索引（需要足够的向量来训练）
            nlist = min(int(np.sqrt(n_vectors)), 100)  # 聚类数
            quantizer = faiss.IndexFlatIP(dim) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            self._faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss_metric)
            self._faiss_index.train(self._vectors_matrix)
            self._faiss_index.nprobe = min(nlist, 10)  # 搜索时探测的聚类数
        else:
            # Flat 索引（暴力搜索）
            if faiss_metric == faiss.METRIC_INNER_PRODUCT:
                self._faiss_index = faiss.IndexFlatIP(dim)
            else:
                self._faiss_index = faiss.IndexFlatL2(dim)
        
        # 添加向量到索引
        self._faiss_index.add(self._vectors_matrix)
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10
    ) -> List[tuple]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回数量
            
        Returns:
            List of (id, score, metadata) tuples
        """
        if not self._built:
            self.build()
        
        if not self.vectors:
            return []
        
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # 归一化查询向量（余弦相似度）
        if self.config.vector_metric == "cosine":
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        # 确保 top_k 不超过向量数量
        top_k = min(top_k, len(self.vectors))
        
        if self._use_faiss and self._faiss_index is not None:
            return self._search_faiss(query_vector, top_k)
        else:
            return self._search_numpy(query_vector, top_k)
    
    def _search_faiss(
        self,
        query_vector: np.ndarray,
        top_k: int
    ) -> List[tuple]:
        """使用 FAISS 搜索"""
        # FAISS 需要 2D 数组
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # 搜索
        distances, indices = self._faiss_index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS 返回 -1 表示无效结果
                continue
            
            # 转换距离为相似度分数
            if self.config.vector_metric in ("cosine", "ip"):
                # 内积越大越相似
                score = float(dist)
            else:  # L2
                # L2 距离越小越相似，转换为相似度
                score = 1.0 / (1.0 + float(dist))
            
            results.append((
                self.ids[idx],
                score,
                self.metadata[idx]
            ))
        
        return results
    
    def _search_numpy(
        self,
        query_vector: np.ndarray,
        top_k: int
    ) -> List[tuple]:
        """使用 numpy 搜索（回退方案）"""
        # 计算相似度
        if self.config.vector_metric in ("cosine", "ip"):
            scores = np.dot(self._vectors_matrix, query_vector)
        else:  # L2
            distances = np.linalg.norm(self._vectors_matrix - query_vector, axis=1)
            scores = 1 / (1 + distances)
        
        # 获取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.ids[idx],
                float(scores[idx]),
                self.metadata[idx]
            ))
        
        return results
    
    def save(self, path: str) -> None:
        """保存索引到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": {
                "index_type": self.config.vector_index_type,
                "dimension": self.dimension,
                "metric": self.config.vector_metric,
                "use_faiss": self._use_faiss,
            },
            "ids": self.ids,
            "metadata": self.metadata,
        }
        
        # 保存向量为 .npy
        if self.vectors:
            vectors_path = path.with_suffix(".npy")
            np.save(str(vectors_path), np.vstack(self.vectors))
        
        # 保存 FAISS 索引
        if self._use_faiss and self._faiss_index is not None:
            faiss_path = path.with_suffix(".faiss")
            faiss.write_index(self._faiss_index, str(faiss_path))
            data["config"]["faiss_index_saved"] = True
        
        # 保存元数据为 .json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """从文件加载索引"""
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.ids = data["ids"]
        self.metadata = data["metadata"]
        
        # 加载向量
        vectors_path = path.with_suffix(".npy")
        if vectors_path.exists():
            vectors_matrix = np.load(str(vectors_path))
            self.vectors = [vectors_matrix[i] for i in range(len(vectors_matrix))]
            self._vectors_matrix = vectors_matrix.astype(np.float32)
        
        # 尝试加载 FAISS 索引
        faiss_path = path.with_suffix(".faiss")
        if self._use_faiss and faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))
            self._built = True
        else:
            self._built = False
    
    def clear(self) -> None:
        """清空索引"""
        self.vectors = []
        self.ids = []
        self.metadata = []
        self._faiss_index = None
        self._vectors_matrix = None
        self._built = False
    
    def __len__(self) -> int:
        """返回索引中的向量数量"""
        return len(self.ids)
    
    @staticmethod
    def is_faiss_available() -> bool:
        """检查 FAISS 是否可用"""
        return FAISS_AVAILABLE


class SemanticRetriever(BaseRetriever):
    """
    语义检索器

    基于向量嵌入进行语义相似度检索
    支持对实体、事件、时间线进行语义匹配

    支持的模型:
    - Contriever: 无监督密集检索模型（推荐）
    - DPR: Dense Passage Retrieval（双编码器）
    - BGE-M3: 本地多语言模型
    """

    def __init__(
        self,
        graph_store: TimelineGraphStore,
        config: RetrieverConfig,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        index_dir: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        初始化语义检索器

        Args:
            graph_store: 图存储
            config: 检索器配置
            embed_fn: 嵌入函数（可选，如果不提供则根据 config 自动创建）
            index_dir: 索引缓存目录，设置后会自动加载/保存索引
            auto_save: 是否在首次构建索引后自动保存
        """
        super().__init__(graph_store, config)

        # 如果未提供 embed_fn，则根据配置创建
        if embed_fn is None:
            from ..embeddings import create_embed_fn
            print(f"正在根据配置创建嵌入函数...")
            print(f"  模型类型: {config.semantic_model_type}")
            print(f"  模型名称: {config.semantic_model_name}")
            print(f"  设备: {config.semantic_model_device}")

            self.embed_fn = create_embed_fn(
                model_type=config.semantic_model_type,
                model_name=config.semantic_model_name,
                device=config.semantic_model_device,
                normalize=config.contriever_normalize,
                batch_size=config.embed_batch_size
            )

            if self.embed_fn is None:
                raise RuntimeError(
                    f"无法创建嵌入函数。请检查配置或手动提供 embed_fn 参数。"
                )
        else:
            self.embed_fn = embed_fn

        self._index_dir = Path(index_dir) if index_dir else None
        self._auto_save = auto_save

        # 向量索引
        self._entity_index: Optional[VectorIndex] = None
        self._event_index: Optional[VectorIndex] = None
        self._timeline_index: Optional[VectorIndex] = None

        # 索引是否已构建
        self._indexes_built = False

        # 尝试自动加载已缓存的索引
        if self._index_dir:
            self._try_load_indexes()
    
    def _try_load_indexes(self) -> bool:
        """尝试加载已缓存的索引"""
        if not self._index_dir:
            return False
        
        index_path = self._index_dir / "entity_index.json"
        if index_path.exists():
            try:
                self.load_indexes(str(self._index_dir))
                return True
            except Exception:
                pass
        return False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        target_type: Optional[Literal["entity", "event", "timeline", "all"]] = "all",
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行语义检索
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            target_type: 目标类型
            
        Returns:
            检索结果列表
        """
        top_k = self._get_top_k(top_k)
        
        # 确保索引已构建
        if not self._indexes_built:
            self.build_indexes()
            # 首次构建后自动保存
            if self._auto_save and self._index_dir:
                self.save_indexes(str(self._index_dir))
        
        # 获取查询向量
        query_vector = self._embed_text(query)
        
        results = []
        
        if target_type in ("entity", "all"):
            results.extend(self.search_entities(query_vector, top_k))
        
        if target_type in ("event", "all"):
            results.extend(self.search_events(query_vector, top_k))
        
        if target_type in ("timeline", "all"):
            results.extend(self.search_timelines(query_vector, top_k))
        
        # 过滤、排序并截断
        results = self._filter_by_threshold(results, self.config.similarity_threshold)
        results = self._sort_by_score(results)
        return results[:top_k]
    
    def search_entities(
        self,
        query_vector: Union[str, List[float], np.ndarray],
        top_k: Optional[int] = None
    ) -> List[EntityResult]:
        """
        搜索实体
        
        Args:
            query_vector: 查询向量或查询文本
            top_k: 返回数量
            
        Returns:
            实体结果列表
        """
        if isinstance(query_vector, str):
            query_vector = self._embed_text(query_vector)
        
        top_k = self._get_top_k(top_k)
        
        if self._entity_index is None:
            self._build_entity_index()
        
        results = self._entity_index.search(query_vector, top_k)
        
        entity_results = []
        for node_id, score, metadata in results:
            entity_results.append(EntityResult(
                node_id=node_id,
                node_type="entity",
                score=score,
                canonical_name=metadata.get("canonical_name", ""),
                description=metadata.get("description", ""),
                aliases=metadata.get("aliases", []),
                source_event_ids=metadata.get("source_event_ids", []),
                metadata={"cluster_id": metadata.get("cluster_id", "")},
            ))
        
        return entity_results
    
    def search_events(
        self,
        query_vector: Union[str, List[float], np.ndarray],
        top_k: Optional[int] = None
    ) -> List[EventResult]:
        """
        搜索事件
        
        Args:
            query_vector: 查询向量或查询文本
            top_k: 返回数量
            
        Returns:
            事件结果列表
        """
        if isinstance(query_vector, str):
            query_vector = self._embed_text(query_vector)
        
        top_k = self._get_top_k(top_k)
        
        if self._event_index is None:
            self._build_event_index()
        
        results = self._event_index.search(query_vector, top_k)
        
        event_results = []
        for node_id, score, metadata in results:
            event_results.append(EventResult(
                node_id=node_id,
                node_type="event",
                score=score,
                event_id=metadata.get("event_id", ""),
                event_description=metadata.get("event_description", ""),
                time_type=metadata.get("time_type", ""),
                time_start=metadata.get("time_start", ""),
                time_end=metadata.get("time_end", ""),
                time_expression=metadata.get("time_expression", ""),
                entity_names=metadata.get("entity_names", []),
                original_sentence=metadata.get("original_sentence", ""),
                chunk_id=metadata.get("chunk_id", ""),
                doc_id=metadata.get("doc_id", ""),
                doc_title=metadata.get("doc_title", ""),
            ))
        
        return event_results
    
    def search_timelines(
        self,
        query_vector: Union[str, List[float], np.ndarray],
        top_k: Optional[int] = None
    ) -> List[TimelineResult]:
        """
        搜索时间线
        
        Args:
            query_vector: 查询向量或查询文本
            top_k: 返回数量
            
        Returns:
            时间线结果列表
        """
        if isinstance(query_vector, str):
            query_vector = self._embed_text(query_vector)
        
        top_k = self._get_top_k(top_k)
        
        if self._timeline_index is None:
            self._build_timeline_index()
        
        results = self._timeline_index.search(query_vector, top_k)
        
        timeline_results = []
        for node_id, score, metadata in results:
            timeline_results.append(TimelineResult(
                node_id=node_id,
                node_type="timeline",
                score=score,
                timeline_id=metadata.get("timeline_id", ""),
                timeline_name=metadata.get("timeline_name", ""),
                description=metadata.get("description", ""),
                entity_canonical_name=metadata.get("entity_canonical_name", ""),
                time_span_start=metadata.get("time_span_start", ""),
                time_span_end=metadata.get("time_span_end", ""),
                event_ids=metadata.get("event_ids", []),
            ))
        
        return timeline_results
    
    def build_indexes(self) -> None:
        """构建所有索引"""
        self._build_entity_index()
        self._build_event_index()
        self._build_timeline_index()
        self._indexes_built = True
    
    def _build_entity_index(self) -> None:
        """构建实体向量索引"""
        self._entity_index = VectorIndex(self.config)
        
        # 收集所有实体文本
        entities_data = []
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_ENTITY:
                continue
            
            canonical_name = data.get("canonical_name", "")
            description = data.get("description", "")
            aliases_str = data.get("aliases", "[]")
            aliases = json.loads(aliases_str) if aliases_str else []
            
            # 拼接文本用于嵌入
            text = "|".join([canonical_name, description] + aliases)
            
            event_ids_str = data.get("source_event_ids", "[]")
            event_ids = json.loads(event_ids_str) if event_ids_str else []
            
            entities_data.append({
                "node_id": node_id,
                "text": text,
                "metadata": {
                    "cluster_id": data.get("cluster_id", ""),
                    "canonical_name": canonical_name,
                    "description": description,
                    "aliases": aliases,
                    "source_event_ids": event_ids,
                }
            })
        
        if not entities_data:
            return
        
        # 批量嵌入
        texts = [e["text"] for e in entities_data]
        vectors = self._embed_batch(texts)
        
        # 添加到索引
        for entity, vector in zip(entities_data, vectors):
            self._entity_index.add(
                entity["node_id"],
                vector,
                entity["metadata"]
            )
        
        self._entity_index.build()
    
    def _build_event_index(self) -> None:
        """构建事件向量索引"""
        self._event_index = VectorIndex(self.config)
        
        # 收集所有事件文本
        events_data = []
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_EVENT:
                continue
            
            event_description = data.get("event_description", "")
            time_expression = data.get("time_expression", "") or ""
            original_sentence = data.get("original_sentence", "")
            entity_names_str = data.get("entity_names", "[]")
            entity_names = json.loads(entity_names_str) if entity_names_str else []
            
            # 拼接文本用于嵌入
            text = "|".join([event_description, time_expression, original_sentence] + entity_names)
            
            events_data.append({
                "node_id": node_id,
                "text": text,
                "metadata": {
                    "event_id": data.get("event_id", ""),
                    "event_description": event_description,
                    "time_type": data.get("time_type", ""),
                    "time_start": data.get("time_start", ""),
                    "time_end": data.get("time_end", ""),
                    "time_expression": time_expression,
                    "entity_names": entity_names,
                    "original_sentence": original_sentence,
                    "chunk_id": data.get("chunk_id", ""),
                    "doc_id": data.get("doc_id", ""),
                    "doc_title": data.get("doc_title", ""),
                }
            })
        
        if not events_data:
            return
        
        # 批量嵌入
        texts = [e["text"] for e in events_data]
        vectors = self._embed_batch(texts)
        
        # 添加到索引
        for event, vector in zip(events_data, vectors):
            self._event_index.add(
                event["node_id"],
                vector,
                event["metadata"]
            )
        
        self._event_index.build()
    
    def _build_timeline_index(self) -> None:
        """构建时间线向量索引"""
        self._timeline_index = VectorIndex(self.config)
        
        # 收集所有时间线文本
        timelines_data = []
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_TIMELINE:
                continue
            
            timeline_name = data.get("timeline_name", "")
            description = data.get("description", "")
            entity_name = data.get("entity_canonical_name", "")
            
            # 拼接文本用于嵌入
            text = "|".join([entity_name,timeline_name,description])
            event_ids_str = data.get("event_ids", "[]")
            event_ids = json.loads(event_ids_str) if event_ids_str else []
            
            timelines_data.append({
                "node_id": node_id,
                "text": text,
                "metadata": {
                    "timeline_id": data.get("timeline_id", ""),
                    "timeline_name": timeline_name,
                    "description": description,
                    "entity_canonical_name": entity_name,
                    "time_span_start": data.get("time_span_start", ""),
                    "time_span_end": data.get("time_span_end", ""),
                    "event_ids": event_ids,
                }
            })
        
        if not timelines_data:
            return
        
        # 批量嵌入
        texts = [t["text"] for t in timelines_data]
        vectors = self._embed_batch(texts)
        
        # 添加到索引
        for timeline, vector in zip(timelines_data, vectors):
            self._timeline_index.add(
                timeline["node_id"],
                vector,
                timeline["metadata"]
            )
        
        self._timeline_index.build()
    
    def _embed_text(self, text: str) -> List[float]:
        """嵌入单个文本"""
        vectors = self.embed_fn([text])
        return vectors[0]
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        if not texts:
            return []
        
        all_vectors = []
        batch_size = self.config.embed_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectors = self.embed_fn(batch)
            all_vectors.extend(vectors)
        
        return all_vectors
    
    def save_indexes(self, directory: str) -> None:
        """保存索引到目录"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        if self._entity_index:
            self._entity_index.save(str(directory / "entity_index.json"))
        if self._event_index:
            self._event_index.save(str(directory / "event_index.json"))
        if self._timeline_index:
            self._timeline_index.save(str(directory / "timeline_index.json"))
    
    def load_indexes(self, directory: str) -> None:
        """从目录加载索引"""
        directory = Path(directory)
        
        entity_path = directory / "entity_index.json"
        if entity_path.exists():
            self._entity_index = VectorIndex()
            self._entity_index.load(str(entity_path))
        
        event_path = directory / "event_index.json"
        if event_path.exists():
            self._event_index = VectorIndex()
            self._event_index.load(str(event_path))
        
        timeline_path = directory / "timeline_index.json"
        if timeline_path.exists():
            self._timeline_index = VectorIndex()
            self._timeline_index.load(str(timeline_path))
        
        self._indexes_built = True
    
    def invalidate_cache(self) -> None:
        """清除索引缓存"""
        self._entity_index = None
        self._event_index = None
        self._timeline_index = None
        self._indexes_built = False
