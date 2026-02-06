"""
TimeQA Agent - Temporal Event Extraction and Timeline Analysis

This package provides tools for:
- Document chunking
- Temporal event extraction
- Entity disambiguation
- Timeline extraction
- Knowledge graph storage
"""

from .config import (
    ChunkStrategy,
    ChunkConfig,
    ExtractorConfig,
    EventFilterConfig,
    EventValidatorConfig,
    DisambiguatorConfig,
    TimelineConfig,
    GraphStoreConfig,
    RetrieverConfig,
    QueryParserConfig,
    FusionMode,
    VectorIndexType,
    VectorMetric,
    DEFAULT_STOPWORDS,
    TimeQAConfig,
    get_default_config,
    load_config,
)
from .chunker import Chunk, DocumentChunker
from .event_extractor import TimeType, Entity, TimeEvent, EventExtractor
from .event_validator import EventValidator
from .event_filter import EventFilter
from .entity_disambiguator import EntityCluster, EntityDisambiguator
from .timeline_extractor import Timeline, StandaloneEvent, TimelineExtractionResult, TimelineExtractor
from .graph_store import TimelineGraphStore
from .pipeline import ExtractionPipeline, PipelineConfig, Stage, STAGE_NAMES
from .query_parser import (
    TimeConstraintType,
    TimeConstraint,
    QueryParseResult,
    RetrievalQueries,
    QueryParserOutput,
    QueryParser,
)
from .retrievers import (
    RetrievalMode,
    RetrievalResult,
    EntityResult,
    EventResult,
    TimelineResult,
    BaseRetriever,
    KeywordRetriever,
    TFIDFIndex,
    SemanticRetriever,
    VectorIndex,
    HybridRetriever,
)
from .time_utils import (
    TimeGranularity,
    TemporalDate,
    add_years,
    add_months,
    add_days,
    parse_date,
)
from .embeddings import create_local_embed_fn

__version__ = "0.1.0"

__all__ = [
    # Config
    "ChunkStrategy",
    "ChunkConfig",
    "ExtractorConfig",
    "EventFilterConfig",
    "EventValidatorConfig",
    "DisambiguatorConfig",
    "TimelineConfig",
    "GraphStoreConfig",
    "RetrieverConfig",
    "QueryParserConfig",
    "FusionMode",
    "VectorIndexType",
    "VectorMetric",
    "DEFAULT_STOPWORDS",
    "TimeQAConfig",
    "get_default_config",
    "load_config",
    # Chunker
    "Chunk",
    "DocumentChunker",
    # Event Extractor
    "TimeType",
    "Entity",
    "TimeEvent",
    "EventExtractor",
    # Event Validator
    "EventValidator",
    # Event Filter
    "EventFilter",
    # Entity Disambiguator
    "EntityCluster",
    "EntityDisambiguator",
    # Timeline Extractor
    "Timeline",
    "StandaloneEvent",
    "TimelineExtractionResult",
    "TimelineExtractor",
    # Graph Store
    "TimelineGraphStore",
    # Pipeline
    "ExtractionPipeline",
    "PipelineConfig",
    "Stage",
    "STAGE_NAMES",
    # Query Parser
    "TimeConstraintType",
    "TimeConstraint",
    "QueryParseResult",
    "RetrievalQueries",
    "QueryParserOutput",
    "QueryParser",
    # Retrievers
    "RetrievalMode",
    "RetrievalResult",
    "EntityResult",
    "EventResult",
    "TimelineResult",
    "BaseRetriever",
    "KeywordRetriever",
    "TFIDFIndex",
    "SemanticRetriever",
    "VectorIndex",
    "HybridRetriever",
    # Embeddings
    "create_local_embed_fn",
    # Time Utils
    "TimeGranularity",
    "TemporalDate",
    "add_years",
    "add_months",
    "add_days",
    "parse_date",
]