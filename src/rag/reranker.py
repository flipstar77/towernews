"""Re-Ranker - Two-stage retrieval with Cross-Encoder for precision.

Pipeline:
    Query → Retrieval (Top 50, fast) → Re-Ranking (Top 10, precise) → Response
             ↓                          ↓
          Bi-Encoder                 Cross-Encoder
          (embedding similarity)     (query-document pairs)

The Cross-Encoder scores query-document pairs directly, providing
more accurate relevance scores than bi-encoder similarity.
"""

from typing import List, Optional, Callable
from dataclasses import dataclass

# Optional import - graceful fallback if not installed
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None


@dataclass
class RerankResult:
    """Result from re-ranking."""
    id: str
    content: str
    metadata: dict
    original_score: float
    rerank_score: float
    final_score: float


class Reranker:
    """
    Cross-Encoder based re-ranker for improved search precision.

    Uses a two-stage approach:
    1. Initial retrieval with bi-encoder (fast, high recall)
    2. Re-ranking with cross-encoder (slower, high precision)
    """

    # Available models (speed vs accuracy tradeoff)
    MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",      # Fastest
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2", # Good balance
        "accurate": "cross-encoder/ms-marco-TinyBERT-L-2-v2" # Most accurate
    }

    def __init__(
        self,
        model_name: str = None,
        model_type: str = "fast",
        device: str = None
    ):
        """
        Initialize the re-ranker.

        Args:
            model_name: Specific model name (overrides model_type)
            model_type: One of "fast", "balanced", "accurate"
            device: Device to run on ("cpu", "cuda", etc.)
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers required for re-ranking. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name or self.MODELS.get(model_type, self.MODELS["fast"])
        self.model = CrossEncoder(self.model_name, device=device)

        print(f"[Reranker] Loaded model: {self.model_name}")

    def rerank(
        self,
        query: str,
        results: List[dict],
        top_k: int = 10,
        content_key: str = "content",
        score_key: str = "similarity",
        blend_weight: float = 0.8
    ) -> List[RerankResult]:
        """
        Re-rank search results using cross-encoder.

        Args:
            query: Search query
            results: List of search results (dicts with content)
            top_k: Number of results to return
            content_key: Key for content in result dicts
            score_key: Key for original score in result dicts
            blend_weight: Weight for rerank score (1.0 = only rerank, 0.0 = only original)

        Returns:
            Re-ranked results with updated scores
        """
        if not results:
            return []

        # Create query-document pairs
        pairs = [(query, r.get(content_key, "")) for r in results]

        # Get cross-encoder scores
        rerank_scores = self.model.predict(pairs)

        # Normalize scores to 0-1 range
        min_score = min(rerank_scores)
        max_score = max(rerank_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        normalized_scores = [(s - min_score) / score_range for s in rerank_scores]

        # Combine with original scores
        reranked = []
        for i, (result, rerank_score, norm_score) in enumerate(
            zip(results, rerank_scores, normalized_scores)
        ):
            original_score = result.get(score_key, 0.5)

            # Blend scores
            final_score = (blend_weight * norm_score) + ((1 - blend_weight) * original_score)

            reranked.append(RerankResult(
                id=result.get("id", str(i)),
                content=result.get(content_key, ""),
                metadata=result.get("metadata", {}),
                original_score=original_score,
                rerank_score=float(rerank_score),
                final_score=final_score
            ))

        # Sort by final score
        reranked.sort(key=lambda x: x.final_score, reverse=True)

        return reranked[:top_k]

    def rerank_search_results(
        self,
        query: str,
        results: List,  # List[SearchResult]
        top_k: int = 10,
        blend_weight: float = 0.8
    ) -> List:
        """
        Re-rank SearchResult objects from UnifiedKnowledgeBase.

        Args:
            query: Search query
            results: List of SearchResult objects
            top_k: Number of results to return
            blend_weight: Weight for rerank score

        Returns:
            Re-ranked SearchResult objects
        """
        if not results:
            return []

        # Convert SearchResult to dict for processing
        result_dicts = []
        for r in results:
            if hasattr(r, 'to_dict'):
                result_dicts.append(r.to_dict())
            else:
                result_dicts.append({
                    "id": getattr(r, 'id', ''),
                    "content": getattr(r, 'content', ''),
                    "metadata": getattr(r, 'metadata', {}),
                    "similarity": getattr(r, 'similarity', 0.5)
                })

        # Re-rank
        reranked = self.rerank(
            query=query,
            results=result_dicts,
            top_k=top_k,
            content_key="content",
            score_key="similarity",
            blend_weight=blend_weight
        )

        # Convert back to SearchResult-like objects
        from .unified_knowledge_base import SearchResult

        return [
            SearchResult(
                id=r.id,
                content=r.content,
                metadata=r.metadata,
                similarity=r.final_score,
                source=r.metadata.get("source", ""),
                doc_type=r.metadata.get("doc_type", ""),
                score=r.metadata.get("score", 0)
            )
            for r in reranked
        ]


class CachedReranker(Reranker):
    """
    Reranker with LRU caching for repeated queries.

    Useful when the same queries are made frequently.
    """

    def __init__(self, cache_size: int = 100, **kwargs):
        """
        Initialize cached re-ranker.

        Args:
            cache_size: Maximum number of cached query results
            **kwargs: Arguments passed to Reranker
        """
        super().__init__(**kwargs)

        from functools import lru_cache

        self.cache_size = cache_size
        self._cache = {}

    def _cache_key(self, query: str, content_hashes: tuple) -> str:
        """Generate cache key from query and content."""
        import hashlib
        key_str = f"{query}:{content_hashes}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def rerank(
        self,
        query: str,
        results: List[dict],
        top_k: int = 10,
        **kwargs
    ) -> List[RerankResult]:
        """Re-rank with caching."""
        # Create content hash for cache key
        content_hashes = tuple(
            hash(r.get("content", "")[:100])
            for r in results[:20]  # Only hash first 20 for efficiency
        )
        cache_key = self._cache_key(query, content_hashes)

        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached[:top_k]

        # Compute re-ranking
        reranked = super().rerank(query, results, top_k=len(results), **kwargs)

        # Cache result (before top_k truncation)
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = reranked

        return reranked[:top_k]


def create_reranker(
    model_type: str = "fast",
    use_cache: bool = False,
    cache_size: int = 100
) -> Optional[Reranker]:
    """
    Factory function to create a reranker.

    Returns None if sentence-transformers is not available.
    """
    if not CROSS_ENCODER_AVAILABLE:
        print("[Reranker] sentence-transformers not available, skipping re-ranking")
        return None

    if use_cache:
        return CachedReranker(model_type=model_type, cache_size=cache_size)
    else:
        return Reranker(model_type=model_type)


# Convenience function for one-off re-ranking
def rerank_results(
    query: str,
    results: List[dict],
    top_k: int = 10,
    model_type: str = "fast"
) -> List[RerankResult]:
    """
    One-off re-ranking without persistent model.

    Note: This loads the model each time, so it's slower for repeated use.
    Use Reranker class for better performance with multiple queries.
    """
    reranker = create_reranker(model_type=model_type)
    if reranker is None:
        # Return original results if re-ranking unavailable
        return [
            RerankResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                original_score=r.get("similarity", 0.5),
                rerank_score=r.get("similarity", 0.5),
                final_score=r.get("similarity", 0.5)
            )
            for r in results[:top_k]
        ]

    return reranker.rerank(query, results, top_k=top_k)


if __name__ == "__main__":
    # Test the reranker
    print("Reranker Test")
    print("=" * 60)

    if not CROSS_ENCODER_AVAILABLE:
        print("sentence-transformers not installed, skipping test")
        exit(0)

    reranker = Reranker(model_type="fast")

    # Sample results
    query = "best death wave build for tier 15"
    results = [
        {"id": "1", "content": "Death Wave is essential for high tier gameplay. Focus on damage multipliers.", "similarity": 0.85},
        {"id": "2", "content": "I love pizza and playing games on weekends.", "similarity": 0.82},
        {"id": "3", "content": "For T15 DW builds, prioritize critical damage and attack speed.", "similarity": 0.78},
        {"id": "4", "content": "Black Hole is another popular ultimate weapon choice.", "similarity": 0.80},
        {"id": "5", "content": "The best Death Wave setup uses Inner Land Mines with high coin farming.", "similarity": 0.75},
    ]

    print(f"\nQuery: {query}")
    print("\nOriginal ranking:")
    for r in sorted(results, key=lambda x: x["similarity"], reverse=True):
        print(f"  [{r['similarity']:.2f}] {r['content'][:60]}...")

    reranked = reranker.rerank(query, results, top_k=5)

    print("\nRe-ranked:")
    for r in reranked:
        print(f"  [{r.final_score:.2f}] (orig: {r.original_score:.2f}, rerank: {r.rerank_score:.2f})")
        print(f"         {r.content[:60]}...")
