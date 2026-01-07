"""RAG (Retrieval Augmented Generation) system for Tower News.

This module provides the RAG infrastructure for Tower News:
- UnifiedKnowledgeBase: Consolidated knowledge base (recommended)
- QueryProcessor: Query preprocessing and expansion
- Reranker: Two-stage retrieval with Cross-Encoder
- RAGAnalytics: Monitoring and evaluation
- KnowledgeIngester: Reddit content ingestion
- WikiScraper: Fandom wiki scraping
- TowerKnowledgeBase: Legacy Supabase-only implementation
"""

# Core unified system (recommended)
from .unified_knowledge_base import (
    UnifiedKnowledgeBase,
    SearchResult,
    ChunkConfig,
    SearchConfig,
    StorageBackend,
    create_knowledge_base
)

# Query processing
from .query_processor import QueryProcessor, ProcessedQuery, process_query

# Re-ranking (optional, requires sentence-transformers)
try:
    from .reranker import Reranker, CachedReranker, RerankResult, create_reranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    Reranker = None
    CachedReranker = None
    RerankResult = None
    create_reranker = None

# Analytics and monitoring
from .analytics import (
    RAGAnalytics,
    RAGEvaluator,
    QueryLog,
    PerformanceMetrics,
    QualityMetrics,
    SearchTimer,
    create_analytics,
    create_evaluator
)

# Content ingestion
from .ingester import KnowledgeIngester
from .wiki_scraper import WikiScraper

# Legacy implementation (for backwards compatibility)
from .knowledge_base import TowerKnowledgeBase

__all__ = [
    # Core
    "UnifiedKnowledgeBase",
    "SearchResult",
    "ChunkConfig",
    "SearchConfig",
    "StorageBackend",
    "create_knowledge_base",

    # Query processing
    "QueryProcessor",
    "ProcessedQuery",
    "process_query",

    # Re-ranking
    "Reranker",
    "CachedReranker",
    "RerankResult",
    "create_reranker",
    "RERANKER_AVAILABLE",

    # Analytics
    "RAGAnalytics",
    "RAGEvaluator",
    "QueryLog",
    "PerformanceMetrics",
    "QualityMetrics",
    "SearchTimer",
    "create_analytics",
    "create_evaluator",

    # Ingestion
    "KnowledgeIngester",
    "WikiScraper",

    # Legacy
    "TowerKnowledgeBase"
]
