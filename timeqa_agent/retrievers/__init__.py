"""
检索器模块

提供关键词、语义、混合、多层投票和三层递进检索能力
"""

from .base import (
    RetrievalMode,
    RetrievalResult,
    EntityResult,
    EventResult,
    TimelineResult,
    BaseRetriever,
    HierarchicalEventResult,
    HierarchicalTimelineResult,
    HierarchicalRetrievalResults,
)
from .keyword_retriever import KeywordRetriever, TFIDFIndex
from .semantic_retriever import SemanticRetriever, VectorIndex, FAISS_AVAILABLE
from .hybrid_retriever import HybridRetriever
from .voting_retriever import (
    MultiLayerVotingRetriever,
    VotingEventResult,
    EventVote,
    create_voting_retriever,
)
from .hierarchical_retriever import HierarchicalRetriever
from ..config import VotingConfig, VotingFusionMode, HierarchicalConfig

__all__ = [
    # Base
    "RetrievalMode",
    "RetrievalResult",
    "EntityResult",
    "EventResult",
    "TimelineResult",
    "BaseRetriever",
    # Keyword
    "KeywordRetriever",
    "TFIDFIndex",
    # Semantic
    "SemanticRetriever",
    "VectorIndex",
    "FAISS_AVAILABLE",
    # Hybrid
    "HybridRetriever",
    # Voting
    "MultiLayerVotingRetriever",
    "VotingConfig",
    "VotingFusionMode",
    "VotingEventResult",
    "EventVote",
    "create_voting_retriever",
    # Hierarchical (三层递进检索)
    "HierarchicalRetriever",
    "HierarchicalConfig",
    "HierarchicalEventResult",
    "HierarchicalTimelineResult",
    "HierarchicalRetrievalResults",
]
