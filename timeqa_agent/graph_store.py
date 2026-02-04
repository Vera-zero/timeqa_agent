"""
Timeline Knowledge Graph Store

使用 NetworkX 存储实体、事件、时间线之间的关系图
支持 GraphML 格式持久化
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

try:
    import networkx as nx
except ImportError:
    raise ImportError("请安装 networkx: pip install networkx")

from .config import GraphStoreConfig
from .event_extractor import TimeEvent, Entity, TimeType
from .entity_disambiguator import EntityCluster
from .timeline_extractor import Timeline, TimelineExtractionResult, StandaloneEvent


# 节点类型常量
NODE_TYPE_ENTITY = "entity"
NODE_TYPE_EVENT = "event"
NODE_TYPE_TIMELINE = "timeline"

# 边类型常量
EDGE_PARTICIPATES_IN = "PARTICIPATES_IN"  # 实体 -> 事件
EDGE_BELONGS_TO = "BELONGS_TO"            # 事件 -> 时间线
EDGE_HAS_TIMELINE = "HAS_TIMELINE"        # 实体 -> 时间线


class TimelineGraphStore:
    """
    时间线知识图谱存储
    
    节点类型:
        - entity: 实体（EntityCluster）
        - event: 时间事件（TimeEvent）
        - timeline: 时间线（Timeline）
    
    边类型:
        - PARTICIPATES_IN: 实体参与事件
        - BELONGS_TO: 事件属于时间线
        - HAS_TIMELINE: 实体拥有时间线
    """
    
    def __init__(self, config: Optional[GraphStoreConfig] = None):
        """初始化图存储"""
        self.config = config or GraphStoreConfig()
        self.graph = nx.MultiDiGraph()  # 支持多重边的有向图
        
        # 索引缓存（加速查询）
        self._entity_name_index: Dict[str, str] = {}  # canonical_name -> node_id
        self._event_index: Dict[str, str] = {}        # event_id -> node_id
        self._timeline_index: Dict[str, str] = {}     # timeline_id -> node_id
    
    # ========== 节点操作 ==========
    
    def add_entity(self, cluster: EntityCluster) -> str:
        """
        添加实体节点
        
        Args:
            cluster: 实体聚类结果
            
        Returns:
            节点ID
        """
        node_id = f"entity:{cluster.cluster_id}"
        
        # 收集所有别名
        aliases = set()
        for member in cluster.member_entities:
            aliases.add(member.name)
            if member.name != member.canonical_name:
                aliases.add(member.canonical_name)
        aliases.discard(cluster.canonical_name)
        
        attrs = {
            "node_type": NODE_TYPE_ENTITY,
            "cluster_id": cluster.cluster_id,
            "canonical_name": cluster.canonical_name,
            "description": cluster.merged_description,
            "source_event_ids": json.dumps(list(cluster.source_event_ids)),
        }
        
        if self.config.store_entity_aliases:
            attrs["aliases"] = json.dumps(list(aliases))
        
        self.graph.add_node(node_id, **attrs)
        self._entity_name_index[cluster.canonical_name] = node_id
        
        return node_id
    
    def add_event(self, event: TimeEvent) -> str:
        """
        添加事件节点
        
        Args:
            event: 时间事件
            
        Returns:
            节点ID
        """
        node_id = f"event:{event.event_id}"
        
        attrs = {
            "node_type": NODE_TYPE_EVENT,
            "event_id": event.event_id,
            "event_description": event.event_description,
            "time_type": event.time_type.value if isinstance(event.time_type, TimeType) else event.time_type,
            "time_start": event.time_start or "",
            "time_end": event.time_end or "",
            "time_expression": event.time_expression,
        }
        
        # 存储原始句子
        if self.config.store_original_sentence:
            attrs["original_sentence"] = event.original_sentence
        
        # 存储分块元数据
        if self.config.store_chunk_metadata:
            attrs["chunk_id"] = event.chunk_id
            attrs["doc_id"] = event.doc_id
            attrs["doc_title"] = event.doc_title
        
        # 存储参与实体的名称列表（便于查询）
        entity_names = [e.canonical_name for e in event.entities]
        attrs["entity_names"] = json.dumps(entity_names)
        
        self.graph.add_node(node_id, **attrs)
        self._event_index[event.event_id] = node_id
        
        return node_id
    
    def add_timeline(self, timeline: Timeline) -> str:
        """
        添加时间线节点
        
        Args:
            timeline: 时间线
            
        Returns:
            节点ID
        """
        node_id = f"timeline:{timeline.timeline_id}"
        
        attrs = {
            "node_type": NODE_TYPE_TIMELINE,
            "timeline_id": timeline.timeline_id,
            "timeline_name": timeline.timeline_name,
            "description": timeline.description,
            "entity_canonical_name": timeline.entity_canonical_name,
            "time_span_start": timeline.time_span_start or "",
            "time_span_end": timeline.time_span_end or "",
            "event_ids": json.dumps(timeline.event_ids),
        }
        
        self.graph.add_node(node_id, **attrs)
        self._timeline_index[timeline.timeline_id] = node_id
        
        return node_id
    
    # ========== 关系操作 ==========
    
    def link_entity_event(self, entity_canonical_name: str, event_id: str, role: str = "participant") -> bool:
        """
        建立实体与事件的关联
        
        Args:
            entity_canonical_name: 实体标准名称
            event_id: 事件ID
            role: 角色（participant, subject, object 等）
            
        Returns:
            是否成功
        """
        entity_node = self._entity_name_index.get(entity_canonical_name)
        event_node = self._event_index.get(event_id)
        
        if not entity_node or not event_node:
            return False
        
        self.graph.add_edge(
            entity_node, event_node,
            edge_type=EDGE_PARTICIPATES_IN,
            role=role,
        )
        return True
    
    def link_event_timeline(self, event_id: str, timeline_id: str, order: int = 0) -> bool:
        """
        建立事件与时间线的关联
        
        Args:
            event_id: 事件ID
            timeline_id: 时间线ID
            order: 事件在时间线中的顺序
            
        Returns:
            是否成功
        """
        event_node = self._event_index.get(event_id)
        timeline_node = self._timeline_index.get(timeline_id)
        
        if not event_node or not timeline_node:
            return False
        
        self.graph.add_edge(
            event_node, timeline_node,
            edge_type=EDGE_BELONGS_TO,
            order=order,
        )
        return True
    
    def link_entity_timeline(self, entity_canonical_name: str, timeline_id: str) -> bool:
        """
        建立实体与时间线的关联
        
        Args:
            entity_canonical_name: 实体标准名称
            timeline_id: 时间线ID
            
        Returns:
            是否成功
        """
        entity_node = self._entity_name_index.get(entity_canonical_name)
        timeline_node = self._timeline_index.get(timeline_id)
        
        if not entity_node or not timeline_node:
            return False
        
        self.graph.add_edge(
            entity_node, timeline_node,
            edge_type=EDGE_HAS_TIMELINE,
        )
        return True
    
    # ========== 查询接口 ==========
    
    def get_entity(self, entity_canonical_name: str) -> Optional[Dict[str, Any]]:
        """获取实体信息"""
        node_id = self._entity_name_index.get(entity_canonical_name)
        if not node_id or node_id not in self.graph:
            return None
        
        data = dict(self.graph.nodes[node_id])
        # 解析 JSON 字段
        if "aliases" in data:
            data["aliases"] = json.loads(data["aliases"])
        if "source_event_ids" in data:
            data["source_event_ids"] = json.loads(data["source_event_ids"])
        return data
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """获取事件信息"""
        node_id = self._event_index.get(event_id)
        if not node_id or node_id not in self.graph:
            return None
        
        data = dict(self.graph.nodes[node_id])
        if "entity_names" in data:
            data["entity_names"] = json.loads(data["entity_names"])
        return data
    
    def get_timeline(self, timeline_id: str) -> Optional[Dict[str, Any]]:
        """获取时间线信息"""
        node_id = self._timeline_index.get(timeline_id)
        if not node_id or node_id not in self.graph:
            return None
        
        data = dict(self.graph.nodes[node_id])
        if "event_ids" in data:
            data["event_ids"] = json.loads(data["event_ids"])
        return data
    
    def get_entity_events(self, entity_canonical_name: str) -> List[Dict[str, Any]]:
        """
        获取实体参与的所有事件
        
        Args:
            entity_canonical_name: 实体标准名称
            
        Returns:
            事件列表
        """
        entity_node = self._entity_name_index.get(entity_canonical_name)
        if not entity_node:
            return []
        
        events = []
        for _, target, data in self.graph.out_edges(entity_node, data=True):
            if data.get("edge_type") == EDGE_PARTICIPATES_IN:
                event_data = self.graph.nodes[target]
                if event_data.get("node_type") == NODE_TYPE_EVENT:
                    event_info = dict(event_data)
                    event_info["role"] = data.get("role", "participant")
                    if "entity_names" in event_info:
                        event_info["entity_names"] = json.loads(event_info["entity_names"])
                    events.append(event_info)
        
        return events
    
    def get_entity_timelines(self, entity_canonical_name: str) -> List[Dict[str, Any]]:
        """
        获取实体关联的所有时间线
        
        Args:
            entity_canonical_name: 实体标准名称
            
        Returns:
            时间线列表
        """
        entity_node = self._entity_name_index.get(entity_canonical_name)
        if not entity_node:
            return []
        
        timelines = []
        for _, target, data in self.graph.out_edges(entity_node, data=True):
            if data.get("edge_type") == EDGE_HAS_TIMELINE:
                timeline_data = self.graph.nodes[target]
                if timeline_data.get("node_type") == NODE_TYPE_TIMELINE:
                    timeline_info = dict(timeline_data)
                    if "event_ids" in timeline_info:
                        timeline_info["event_ids"] = json.loads(timeline_info["event_ids"])
                    timelines.append(timeline_info)
        
        return timelines
    
    def get_event_entities(self, event_id: str) -> List[Dict[str, Any]]:
        """
        获取参与事件的所有实体
        
        Args:
            event_id: 事件ID
            
        Returns:
            实体列表
        """
        event_node = self._event_index.get(event_id)
        if not event_node:
            return []
        
        entities = []
        for source, _, data in self.graph.in_edges(event_node, data=True):
            if data.get("edge_type") == EDGE_PARTICIPATES_IN:
                entity_data = self.graph.nodes[source]
                if entity_data.get("node_type") == NODE_TYPE_ENTITY:
                    entity_info = dict(entity_data)
                    entity_info["role"] = data.get("role", "participant")
                    if "aliases" in entity_info:
                        entity_info["aliases"] = json.loads(entity_info["aliases"])
                    if "source_event_ids" in entity_info:
                        entity_info["source_event_ids"] = json.loads(entity_info["source_event_ids"])
                    entities.append(entity_info)
        
        return entities
    
    def get_event_timeline(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        获取事件所属的时间线
        
        Args:
            event_id: 事件ID
            
        Returns:
            时间线信息（如果存在）
        """
        event_node = self._event_index.get(event_id)
        if not event_node:
            return None
        
        for _, target, data in self.graph.out_edges(event_node, data=True):
            if data.get("edge_type") == EDGE_BELONGS_TO:
                timeline_data = self.graph.nodes[target]
                if timeline_data.get("node_type") == NODE_TYPE_TIMELINE:
                    timeline_info = dict(timeline_data)
                    timeline_info["order"] = data.get("order", 0)
                    if "event_ids" in timeline_info:
                        timeline_info["event_ids"] = json.loads(timeline_info["event_ids"])
                    return timeline_info
        
        return None
    
    def get_timeline_events(self, timeline_id: str) -> List[Dict[str, Any]]:
        """
        获取时间线包含的所有事件（按顺序）
        
        Args:
            timeline_id: 时间线ID
            
        Returns:
            事件列表（按时间顺序排列）
        """
        timeline_node = self._timeline_index.get(timeline_id)
        if not timeline_node:
            return []
        
        events = []
        for source, _, data in self.graph.in_edges(timeline_node, data=True):
            if data.get("edge_type") == EDGE_BELONGS_TO:
                event_data = self.graph.nodes[source]
                if event_data.get("node_type") == NODE_TYPE_EVENT:
                    event_info = dict(event_data)
                    event_info["order"] = data.get("order", 0)
                    if "entity_names" in event_info:
                        event_info["entity_names"] = json.loads(event_info["entity_names"])
                    events.append(event_info)
        
        # 按顺序排序
        events.sort(key=lambda x: x.get("order", 0))
        return events
    
    def get_timeline_entity(self, timeline_id: str) -> Optional[Dict[str, Any]]:
        """
        获取时间线所属的实体
        
        Args:
            timeline_id: 时间线ID
            
        Returns:
            实体信息
        """
        timeline_node = self._timeline_index.get(timeline_id)
        if not timeline_node:
            return None
        
        for source, _, data in self.graph.in_edges(timeline_node, data=True):
            if data.get("edge_type") == EDGE_HAS_TIMELINE:
                entity_data = self.graph.nodes[source]
                if entity_data.get("node_type") == NODE_TYPE_ENTITY:
                    entity_info = dict(entity_data)
                    if "aliases" in entity_info:
                        entity_info["aliases"] = json.loads(entity_info["aliases"])
                    if "source_event_ids" in entity_info:
                        entity_info["source_event_ids"] = json.loads(entity_info["source_event_ids"])
                    return entity_info
        
        return None
    
    def get_entities_by_name(self, name: str, fuzzy: bool = False) -> List[Dict[str, Any]]:
        """
        按名称搜索实体
        
        Args:
            name: 实体名称
            fuzzy: 是否模糊匹配
            
        Returns:
            匹配的实体列表
        """
        results = []
        name_lower = name.lower()
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_ENTITY:
                continue
            
            canonical_name = data.get("canonical_name", "")
            aliases_str = data.get("aliases", "[]")
            aliases = json.loads(aliases_str) if aliases_str else []
            
            matched = False
            if fuzzy:
                if name_lower in canonical_name.lower():
                    matched = True
                else:
                    for alias in aliases:
                        if name_lower in alias.lower():
                            matched = True
                            break
            else:
                if canonical_name == name:
                    matched = True
                elif name in aliases:
                    matched = True
            
            if matched:
                entity_info = dict(data)
                entity_info["aliases"] = aliases
                if "source_event_ids" in entity_info:
                    entity_info["source_event_ids"] = json.loads(entity_info["source_event_ids"])
                results.append(entity_info)
        
        return results
    
    def get_events_in_time_range(self, start: str, end: str) -> List[Dict[str, Any]]:
        """
        获取时间范围内的事件
        
        Args:
            start: 开始时间
            end: 结束时间
            
        Returns:
            事件列表
        """
        # 简单实现：只比较年份
        results = []
        
        try:
            start_year = int(start[:4]) if start else 0
            end_year = int(end[:4]) if end else 9999
        except ValueError:
            return results
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != NODE_TYPE_EVENT:
                continue
            
            event_start = data.get("time_start", "")
            try:
                event_year = int(event_start[:4]) if event_start else None
                if event_year and start_year <= event_year <= end_year:
                    event_info = dict(data)
                    if "entity_names" in event_info:
                        event_info["entity_names"] = json.loads(event_info["entity_names"])
                    results.append(event_info)
            except ValueError:
                continue
        
        return results
    
    # ========== 统计接口 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """获取图统计信息"""
        entity_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == NODE_TYPE_ENTITY)
        event_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == NODE_TYPE_EVENT)
        timeline_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == NODE_TYPE_TIMELINE)
        
        participates_count = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == EDGE_PARTICIPATES_IN)
        belongs_to_count = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == EDGE_BELONGS_TO)
        has_timeline_count = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == EDGE_HAS_TIMELINE)
        
        return {
            "nodes": {
                "total": self.graph.number_of_nodes(),
                "entities": entity_count,
                "events": event_count,
                "timelines": timeline_count,
            },
            "edges": {
                "total": self.graph.number_of_edges(),
                "participates_in": participates_count,
                "belongs_to": belongs_to_count,
                "has_timeline": has_timeline_count,
            },
        }
    
    def list_all_entities(self) -> List[str]:
        """列出所有实体名称"""
        return list(self._entity_name_index.keys())
    
    def list_all_events(self) -> List[str]:
        """列出所有事件ID"""
        return list(self._event_index.keys())
    
    def list_all_timelines(self) -> List[str]:
        """列出所有时间线ID"""
        return list(self._timeline_index.keys())
    
    # ========== 持久化 ==========
    
    def save(self, path: str) -> None:
        """
        保存图到文件
        
        Args:
            path: 文件路径（支持 .graphml 或 .json）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == ".graphml":
            nx.write_graphml(self.graph, str(path))
        elif path.suffix == ".json":
            # 导出为 JSON 格式
            data = {
                "nodes": [],
                "edges": [],
            }
            for node_id, attrs in self.graph.nodes(data=True):
                node_data = {"id": node_id, **attrs}
                data["nodes"].append(node_data)
            
            for source, target, attrs in self.graph.edges(data=True):
                edge_data = {"source": source, "target": target, **attrs}
                data["edges"].append(edge_data)
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}，支持 .graphml 或 .json")
        
        print(f"✓ 图已保存到: {path}")
    
    def load(self, path: str) -> None:
        """
        从文件加载图
        
        Args:
            path: 文件路径
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if path.suffix == ".graphml":
            self.graph = nx.read_graphml(str(path))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.graph = nx.MultiDiGraph()
            for node in data.get("nodes", []):
                node_id = node.pop("id")
                self.graph.add_node(node_id, **node)
            
            for edge in data.get("edges", []):
                source = edge.pop("source")
                target = edge.pop("target")
                self.graph.add_edge(source, target, **edge)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
        
        # 重建索引
        self._rebuild_indexes()
        print(f"✓ 图已从 {path} 加载")
    
    def _rebuild_indexes(self) -> None:
        """重建索引缓存"""
        self._entity_name_index.clear()
        self._event_index.clear()
        self._timeline_index.clear()
        
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("node_type")
            if node_type == NODE_TYPE_ENTITY:
                canonical_name = data.get("canonical_name")
                if canonical_name:
                    self._entity_name_index[canonical_name] = node_id
            elif node_type == NODE_TYPE_EVENT:
                event_id = data.get("event_id")
                if event_id:
                    self._event_index[event_id] = node_id
            elif node_type == NODE_TYPE_TIMELINE:
                timeline_id = data.get("timeline_id")
                if timeline_id:
                    self._timeline_index[timeline_id] = node_id
    
    # ========== 批量导入 ==========
    
    def import_from_pipeline(
        self,
        events: List[TimeEvent],
        entity_clusters: List[EntityCluster],
        timeline_results: Dict[str, TimelineExtractionResult],
    ) -> Dict[str, int]:
        """
        从流水线结果批量导入数据
        
        Args:
            events: 事件列表
            entity_clusters: 实体聚类列表
            timeline_results: 时间线抽取结果（按实体名称索引）
            
        Returns:
            导入统计
        """
        stats = {
            "entities_added": 0,
            "events_added": 0,
            "timelines_added": 0,
            "entity_event_links": 0,
            "event_timeline_links": 0,
            "entity_timeline_links": 0,
        }
        
        # 1. 添加所有实体
        for cluster in entity_clusters:
            self.add_entity(cluster)
            stats["entities_added"] += 1
        
        # 2. 添加所有事件
        event_map = {}  # event_id -> TimeEvent
        for event in events:
            self.add_event(event)
            event_map[event.event_id] = event
            stats["events_added"] += 1
        
        # 3. 建立实体-事件关联
        for event in events:
            for entity in event.entities:
                if self.link_entity_event(entity.canonical_name, event.event_id):
                    stats["entity_event_links"] += 1
        
        # 4. 添加时间线并建立关联
        for entity_name, result in timeline_results.items():
            for timeline in result.timelines:
                self.add_timeline(timeline)
                stats["timelines_added"] += 1
                
                # 实体-时间线关联
                if self.link_entity_timeline(entity_name, timeline.timeline_id):
                    stats["entity_timeline_links"] += 1
                
                # 事件-时间线关联
                for order, event_id in enumerate(timeline.event_ids):
                    if self.link_event_timeline(event_id, timeline.timeline_id, order):
                        stats["event_timeline_links"] += 1
        
        return stats
    
    def clear(self) -> None:
        """清空图"""
        self.graph.clear()
        self._entity_name_index.clear()
        self._event_index.clear()
        self._timeline_index.clear()
