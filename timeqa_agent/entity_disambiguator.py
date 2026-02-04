"""
Entity Disambiguation Module

基于 canonical_name + description 的向量相似度实体消歧
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

from .config import DisambiguatorConfig
from .event_extractor import Entity, TimeEvent
from .embeddings import create_local_embed_fn


@dataclass
class EntityCluster:
    """实体聚类结果"""
    cluster_id: str                    # 聚类ID
    canonical_name: str                # 统一的标准名称
    merged_description: str            # 合并后的描述
    member_entities: List[Entity]      # 聚类中的所有实体
    source_event_ids: Set[str]         # 来源事件ID集合
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "canonical_name": self.canonical_name,
            "merged_description": self.merged_description,
            "member_count": len(self.member_entities),
            "member_entities": [e.to_dict() for e in self.member_entities],
            "source_event_ids": list(self.source_event_ids),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityCluster":
        """从字典创建 EntityCluster"""
        return cls(
            cluster_id=data.get("cluster_id", ""),
            canonical_name=data.get("canonical_name", ""),
            merged_description=data.get("merged_description", ""),
            member_entities=[Entity.from_dict(e) for e in data.get("member_entities", [])],
            source_event_ids=set(data.get("source_event_ids", [])),
        )


class EntityDisambiguator:
    """基于向量相似度的实体消歧器"""
    
    def __init__(
        self,
        config: Optional[DisambiguatorConfig] = None,
        token: Optional[str] = None,
    ):
        self.config = config or DisambiguatorConfig()
        
        # 初始化本地嵌入函数
        self.local_embed_fn = create_local_embed_fn(self.config.local_embed_model_path)
        
        # 如果本地模型不可用，则尝试使用API
        if self.local_embed_fn is None:
            print("警告: 本地嵌入模型不可用，尝试使用API...")
            # 获取 token
            if token:
                self.token = token
            else:
                self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
                if not self.token:
                    raise ValueError("请设置 VENUS_API_TOKEN 或 OPENAI_API_KEY 环境变量")
        else:
            print("已成功加载本地BGE-M3模型")
        
        # 缓存
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _get_entity_text(self, entity: Entity) -> str:
        """
        生成用于 embedding 的实体文本
        
        格式: canonical_name (重复以增加权重) + description
        """
        # canonical_name 重复以增加权重
        weight = int(self.config.canonical_name_weight)
        name_part = " ".join([entity.canonical_name] * weight)
        
        # 拼接描述
        text = f"{name_part}. {entity.description}"
        return text.strip()
    
    def _call_embedding_api(self, texts: List[str]) -> List[np.ndarray]:
        """调用本地BGE-M3模型或API"""
        if not texts:
            return []
        
        # 尝试使用本地模型
        if self.local_embed_fn is not None:
            try:
                # 调用本地模型
                embeddings_list = self.local_embed_fn(texts)
                # 转换为numpy数组
                embeddings = [np.array(emb) for emb in embeddings_list]
                return embeddings
            except Exception as e:
                print(f"本地模型调用失败: {e}，尝试使用API...")
        
        # 如果本地模型不可用或失败，使用API
        import json
        import requests
        
        payload = {
            'model': self.config.embed_model,
            'input': texts,
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }
        
        try:
            response = requests.post(
                self.config.embed_base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )
            
            if response.status_code != 200:
                raise Exception(f"Embedding API 调用失败: {response.status_code} - {response.text}")
            
            result = response.json()
            embeddings = [np.array(item['embedding']) for item in result['data']]
            return embeddings
        except Exception as e:
            print(f"Embedding API 调用失败: {e}")
            raise
    
    def _get_embeddings(self, entities: List[Entity]) -> Dict[str, np.ndarray]:
        """
        获取实体的 embeddings，支持缓存
        
        Returns:
            Dict[entity_text, embedding]
        """
        # 生成实体文本
        entity_texts = []
        text_to_entity = {}
        
        for entity in entities:
            text = self._get_entity_text(entity)
            if text not in self._embedding_cache:
                entity_texts.append(text)
            text_to_entity[text] = entity
        
        # 批量获取未缓存的 embeddings
        if entity_texts:
            # 分批处理
            for i in range(0, len(entity_texts), self.config.embed_batch_size):
                batch = entity_texts[i:i + self.config.embed_batch_size]
                embeddings = self._call_embedding_api(batch)
                
                for text, emb in zip(batch, embeddings):
                    self._embedding_cache[text] = emb
        
        # 返回所有 embeddings
        result = {}
        for entity in entities:
            text = self._get_entity_text(entity)
            result[text] = self._embedding_cache[text]
        
        return result
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def compute_similarity_matrix(
        self,
        entities: List[Entity],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        计算实体间的相似度矩阵
        
        Returns:
            (similarity_matrix, entity_texts)
        """
        # 获取 embeddings
        embeddings_dict = self._get_embeddings(entities)
        
        # 构建矩阵
        entity_texts = [self._get_entity_text(e) for e in entities]
        n = len(entities)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self.cosine_similarity(
                        embeddings_dict[entity_texts[i]],
                        embeddings_dict[entity_texts[j]]
                    )
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
        
        return similarity_matrix, entity_texts
    
    def cluster_entities(
        self,
        entities: List[Entity],
        entity_event_map: Optional[Dict[int, str]] = None,
    ) -> List[EntityCluster]:
        """
        对实体进行聚类
        
        Args:
            entities: 实体列表
            entity_event_map: 实体索引 -> 事件ID 的映射
            
        Returns:
            聚类结果列表
        """
        if not entities:
            return []
        
        if entity_event_map is None:
            entity_event_map = {}
        
        # 计算相似度矩阵
        similarity_matrix, entity_texts = self.compute_similarity_matrix(entities)
        
        n = len(entities)
        visited = [False] * n
        clusters = []
        
        # 使用简单的贪心聚类算法
        for i in range(n):
            if visited[i]:
                continue
            
            # 创建新聚类
            cluster_members = [i]
            visited[i] = True
            
            # 找到所有相似的实体
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                
                # 检查与聚类中所有成员的相似度（使用平均链接）
                avg_sim = np.mean([similarity_matrix[m][j] for m in cluster_members])
                
                if avg_sim >= self.config.similarity_threshold:
                    cluster_members.append(j)
                    visited[j] = True
            
            # 构建聚类结果
            member_entities = [entities[idx] for idx in cluster_members]
            source_event_ids = {entity_event_map.get(idx, "") for idx in cluster_members}
            source_event_ids.discard("")
            
            # 选择最长的 canonical_name 作为统一名称
            canonical_name = max(
                [e.canonical_name for e in member_entities],
                key=len
            )
            
            # 合并描述（去重）
            descriptions = list(set(e.description for e in member_entities if e.description))
            merged_description = "; ".join(descriptions) if descriptions else ""
            
            cluster = EntityCluster(
                cluster_id=f"cluster-{len(clusters):04d}",
                canonical_name=canonical_name,
                merged_description=merged_description,
                member_entities=member_entities,
                source_event_ids=source_event_ids,
            )
            clusters.append(cluster)
        
        return clusters
    
    def disambiguate_events(
        self,
        events: List[TimeEvent],
    ) -> Tuple[List[EntityCluster], Dict[str, str]]:
        """
        对事件中的所有实体进行消歧
        
        Args:
            events: 时间事件列表
            
        Returns:
            (聚类结果, 实体名称 -> 聚类ID 的映射)
        """
        # 收集所有实体
        all_entities = []
        entity_event_map = {}  # 实体索引 -> 事件ID
        
        for event in events:
            for entity in event.entities:
                entity_event_map[len(all_entities)] = event.event_id
                all_entities.append(entity)
        
        if not all_entities:
            return [], {}
        
        # 聚类
        clusters = self.cluster_entities(all_entities, entity_event_map)
        
        # 构建映射：(canonical_name, description) -> cluster_id
        entity_to_cluster = {}
        for cluster in clusters:
            for entity in cluster.member_entities:
                key = self._get_entity_text(entity)
                entity_to_cluster[key] = cluster.cluster_id
        
        return clusters, entity_to_cluster
    
    def find_same_entities(
        self,
        entity1: Entity,
        entity2: Entity,
    ) -> Tuple[bool, float]:
        """
        判断两个实体是否是同一个实体
        
        Args:
            entity1: 第一个实体
            entity2: 第二个实体
            
        Returns:
            (是否是同一实体, 相似度分数)
        """
        # 获取 embeddings
        embeddings_dict = self._get_embeddings([entity1, entity2])
        
        text1 = self._get_entity_text(entity1)
        text2 = self._get_entity_text(entity2)
        
        similarity = self.cosine_similarity(
            embeddings_dict[text1],
            embeddings_dict[text2]
        )
        
        is_same = similarity >= self.config.similarity_threshold
        return is_same, similarity
    
    def batch_find_duplicates(
        self,
        entities: List[Entity],
    ) -> List[Tuple[int, int, float]]:
        """
        批量查找重复实体对
        
        Args:
            entities: 实体列表
            
        Returns:
            [(实体1索引, 实体2索引, 相似度), ...]
        """
        if len(entities) < 2:
            return []
        
        # 计算相似度矩阵
        similarity_matrix, _ = self.compute_similarity_matrix(entities)
        
        # 找出所有超过阈值的实体对
        duplicates = []
        n = len(entities)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= self.config.similarity_threshold:
                    duplicates.append((i, j, similarity_matrix[i][j]))
        
        # 按相似度降序排序
        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates