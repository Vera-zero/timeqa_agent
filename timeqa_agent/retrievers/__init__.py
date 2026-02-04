"""
检索器模块

提供关键词、语义、混合和多层投票检索能力
"""

from .base import (
    RetrievalMode,
    RetrievalResult,
    EntityResult,
    EventResult,
    TimelineResult,
    BaseRetriever,
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
from ..config import VotingConfig, VotingFusionMode

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
]
