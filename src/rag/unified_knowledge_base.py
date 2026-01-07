"""Unified Knowledge Base - Consolidated RAG system for Tower News.

This module provides a single, unified interface for the knowledge base,
consolidating the previous Supabase-based and local JSON-based implementations.

Features:
- Supabase pgvector as primary backend (production)
- Local JSON/NumPy fallback for offline development
- Improved chunking with overlap
- Hybrid search (semantic + fulltext)
- Metadata boosting and filtering
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from openai import OpenAI

# Optional imports with fallbacks
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False
    np = None


class StorageBackend(Enum):
    """Available storage backends."""
    SUPABASE = "supabase"
    LOCAL = "local"


@dataclass
class SearchResult:
    """A search result from the knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    source: str = ""
    doc_type: str = ""
    score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "similarity": self.similarity,
            "source": self.source,
            "doc_type": self.doc_type,
            "score": self.score
        }


@dataclass
class ChunkConfig:
    """Configuration for chunking documents."""
    max_size: int = 800          # Max characters per chunk
    overlap: float = 0.15        # 15% overlap between chunks
    min_size: int = 100          # Minimum chunk size
    respect_sentences: bool = True  # Don't split mid-sentence


@dataclass
class SearchConfig:
    """Configuration for search behavior."""
    semantic_weight: float = 0.6
    fulltext_weight: float = 0.4
    use_reranking: bool = False
    min_similarity: float = 0.35    # Filter out low-quality results
    boost_guides: float = 1.3
    boost_wiki: float = 1.25
    boost_high_score: float = 1.2   # For Reddit score > 100
    boost_recent: float = 1.15      # For posts < 90 days old
    recency_days: int = 90


class UnifiedKnowledgeBase:
    """
    Unified knowledge base for The Tower game content.

    Supports both Supabase (production) and local storage (development).
    Provides hybrid search combining semantic and fulltext matching.
    """

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536
    LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LOCAL_EMBEDDING_DIMENSIONS = 384

    def __init__(
        self,
        backend: StorageBackend = None,
        use_local_embeddings: bool = False,
        chunk_config: ChunkConfig = None,
        search_config: SearchConfig = None
    ):
        """
        Initialize the unified knowledge base.

        Args:
            backend: Storage backend (auto-detected if None)
            use_local_embeddings: Use sentence-transformers instead of OpenAI
            chunk_config: Chunking configuration
            search_config: Search configuration
        """
        self.chunk_config = chunk_config or ChunkConfig()
        self.search_config = search_config or SearchConfig()

        # Auto-detect backend
        if backend is None:
            if SUPABASE_AVAILABLE and os.getenv("SUPABASE_URL"):
                backend = StorageBackend.SUPABASE
            else:
                backend = StorageBackend.LOCAL

        self.backend = backend
        self.use_local_embeddings = use_local_embeddings and LOCAL_EMBEDDINGS_AVAILABLE

        # Initialize backend
        if self.backend == StorageBackend.SUPABASE:
            self._init_supabase()
        else:
            self._init_local()

        # Initialize embedding model
        self._init_embeddings()

        print(f"[UnifiedKB] Initialized with backend={self.backend.value}, "
              f"local_embeddings={self.use_local_embeddings}")

    def _init_supabase(self):
        """Initialize Supabase client."""
        if not SUPABASE_AVAILABLE:
            raise RuntimeError("Supabase not available. Install with: pip install supabase")

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

        self.supabase: Client = create_client(url, key)

    def _init_local(self):
        """Initialize local JSON storage."""
        self.local_dir = Path("data/knowledge_base")
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_file = self.local_dir / "chunks.json"
        self.embeddings_file = self.local_dir / "embeddings.npy"
        self.index_file = self.local_dir / "index.json"

        # Load existing data
        self.local_chunks = self._load_json(self.chunks_file, {"chunks": []})
        self.local_index = self._load_json(self.index_file, {
            "last_updated": None,
            "total_chunks": 0
        })
        self.local_embeddings = {"vectors": None, "ids": []}
        self._load_local_embeddings()

    def _init_embeddings(self):
        """Initialize embedding model."""
        self.openai_client = None
        self.local_embedding_model = None

        if self.use_local_embeddings:
            try:
                self.local_embedding_model = SentenceTransformer(self.LOCAL_EMBEDDING_MODEL)
                self.embedding_dimensions = self.LOCAL_EMBEDDING_DIMENSIONS
            except Exception as e:
                print(f"[UnifiedKB] Failed to load local embeddings: {e}")
                self.use_local_embeddings = False

        if not self.use_local_embeddings:
            self.openai_client = OpenAI()
            self.embedding_dimensions = self.EMBEDDING_DIMENSIONS

    def _load_json(self, path: Path, default: dict) -> dict:
        """Load JSON file or return default."""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return default

    def _save_json(self, path: Path, data: dict):
        """Save data to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_local_embeddings(self):
        """Load local embeddings from disk."""
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            return

        if self.embeddings_file.exists():
            try:
                data = np.load(self.embeddings_file, allow_pickle=True).item()
                self.local_embeddings['vectors'] = data.get('vectors')
                self.local_embeddings['ids'] = data.get('ids', [])
            except Exception as e:
                print(f"[UnifiedKB] Failed to load embeddings: {e}")

    def _save_local_embeddings(self):
        """Save local embeddings to disk."""
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            return

        if self.local_embeddings.get('vectors') is not None:
            try:
                np.save(self.embeddings_file, {
                    'vectors': self.local_embeddings['vectors'],
                    'ids': self.local_embeddings['ids']
                })
            except Exception as e:
                print(f"[UnifiedKB] Failed to save embeddings: {e}")

    # =========================================================================
    # Chunking
    # =========================================================================

    def chunk_text(
        self,
        text: str,
        doc_type: str = "post",
        config: ChunkConfig = None
    ) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk
            doc_type: Document type (affects chunk size)
            config: Override chunk config

        Returns:
            List of text chunks
        """
        config = config or self.chunk_config

        # Adjust size based on doc type
        size_multipliers = {
            "post": 1.0,
            "comment": 0.5,    # Shorter chunks for comments
            "wiki": 1.25,      # Longer for wiki
            "guide": 1.5       # Longest for guides
        }
        max_size = int(config.max_size * size_multipliers.get(doc_type, 1.0))
        overlap_size = int(max_size * config.overlap)

        if len(text) <= max_size:
            return [text] if len(text) >= config.min_size else []

        chunks = []

        if config.respect_sentences:
            chunks = self._chunk_by_sentences(text, max_size, overlap_size, config.min_size)
        else:
            chunks = self._chunk_by_size(text, max_size, overlap_size, config.min_size)

        return chunks

    def _chunk_by_sentences(
        self,
        text: str,
        max_size: int,
        overlap_size: int,
        min_size: int
    ) -> List[str]:
        """Chunk text respecting sentence boundaries with overlap."""
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size <= max_size:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= min_size:
                        chunks.append(chunk_text)

                    # Calculate overlap - keep last sentences that fit in overlap_size
                    overlap_chunk = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) <= overlap_size:
                            overlap_chunk.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break

                    current_chunk = overlap_chunk + [sentence]
                    current_size = sum(len(s) + 1 for s in current_chunk)
                else:
                    # Sentence is longer than max_size, force split
                    current_chunk = [sentence]
                    current_size = sentence_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_size:
                chunks.append(chunk_text)

        return chunks

    def _chunk_by_size(
        self,
        text: str,
        max_size: int,
        overlap_size: int,
        min_size: int
    ) -> List[str]:
        """Simple size-based chunking with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_size, len(text))
            chunk = text[start:end]

            if len(chunk) >= min_size:
                chunks.append(chunk)

            start = end - overlap_size
            if start >= len(text) - min_size:
                break

        return chunks

    # =========================================================================
    # Embeddings (with LRU cache for query embeddings)
    # =========================================================================

    def __init_cache(self):
        """Initialize embedding cache."""
        if not hasattr(self, '_embedding_cache'):
            self._embedding_cache = {}
            self._cache_max_size = 200

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text with caching."""
        self.__init_cache()

        # Check cache first
        cache_key = text[:500]  # Use first 500 chars as key
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Generate embedding
        if self.use_local_embeddings and self.local_embedding_model:
            embedding = self.local_embedding_model.encode(text).tolist()
        else:
            response = self.openai_client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=text
            )
            embedding = response.data[0].embedding

        # Cache result (evict oldest if full)
        if len(self._embedding_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        self._embedding_cache[cache_key] = embedding

        return embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.use_local_embeddings and self.local_embedding_model:
            embeddings = self.local_embedding_model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        else:
            all_embeddings = []
            batch_size = 2000

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.openai_client.embeddings.create(
                    model=self.EMBEDDING_MODEL,
                    input=batch
                )
                all_embeddings.extend([e.embedding for e in response.data])

            return all_embeddings

    # =========================================================================
    # Document Operations
    # =========================================================================

    def _generate_chunk_id(self, content: str) -> str:
        """Generate unique ID for a chunk based on content hash."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def add_document(
        self,
        content: str,
        source: str,
        doc_type: str = "post",
        metadata: Dict[str, Any] = None,
        score: int = 0,
        chunk: bool = True
    ) -> List[str]:
        """
        Add a document to the knowledge base.

        Args:
            content: Document content
            source: Source identifier (e.g., "reddit_abc123")
            doc_type: Type of document (post, comment, wiki, guide)
            metadata: Additional metadata
            score: Relevance score (e.g., Reddit upvotes)
            chunk: Whether to chunk the content

        Returns:
            List of chunk IDs added
        """
        if not content or len(content.strip()) < 50:
            return []

        metadata = metadata or {}
        metadata["source"] = source
        metadata["doc_type"] = doc_type
        metadata["score"] = score
        metadata["ingested_at"] = datetime.now().isoformat()

        # Chunk content if needed
        if chunk:
            chunks = self.chunk_text(content, doc_type)
        else:
            chunks = [content]

        if not chunks:
            return []

        chunk_ids = []

        for i, chunk_content in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk_content)

            # Add chunk metadata
            chunk_meta = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }

            if self.backend == StorageBackend.SUPABASE:
                result = self._add_document_supabase(
                    chunk_id, chunk_content, chunk_meta, doc_type, score
                )
            else:
                result = self._add_document_local(
                    chunk_id, chunk_content, chunk_meta, doc_type, score
                )

            if result:
                chunk_ids.append(chunk_id)

        return chunk_ids

    def _add_document_supabase(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any],
        doc_type: str,
        score: int
    ) -> bool:
        """Add document to Supabase."""
        try:
            # Check if exists
            existing = self.supabase.table("tower_knowledge")\
                .select("id")\
                .eq("post_id", chunk_id)\
                .limit(1)\
                .execute()

            if existing.data:
                return False  # Already exists

            # Generate embedding
            embedding = self.get_embedding(content)

            # Insert
            result = self.supabase.table("tower_knowledge").insert({
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
                "post_id": chunk_id,
                "post_type": doc_type,
                "score": score
            }).execute()

            return bool(result.data)

        except Exception as e:
            print(f"[UnifiedKB] Error adding to Supabase: {e}")
            return False

    def _add_document_local(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any],
        doc_type: str,
        score: int
    ) -> bool:
        """Add document to local storage."""
        try:
            # Check if exists
            if any(c['id'] == chunk_id for c in self.local_chunks['chunks']):
                return False

            # Add chunk
            self.local_chunks['chunks'].append({
                "id": chunk_id,
                "content": content,
                "metadata": metadata,
                "doc_type": doc_type,
                "score": score
            })

            # Update index
            self.local_index['total_chunks'] = len(self.local_chunks['chunks'])
            self.local_index['last_updated'] = datetime.now().isoformat()

            # Save
            self._save_json(self.chunks_file, self.local_chunks)
            self._save_json(self.index_file, self.local_index)

            return True

        except Exception as e:
            print(f"[UnifiedKB] Error adding locally: {e}")
            return False

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists."""
        if self.backend == StorageBackend.SUPABASE:
            try:
                result = self.supabase.table("tower_knowledge")\
                    .select("id")\
                    .eq("post_id", doc_id)\
                    .limit(1)\
                    .execute()
                return len(result.data) > 0
            except:
                return False
        else:
            return any(c['id'] == doc_id for c in self.local_chunks['chunks'])

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 10,
        doc_type: Optional[str] = None,
        min_score: int = 0,
        use_hybrid: bool = True
    ) -> List[SearchResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            limit: Maximum results
            doc_type: Filter by document type
            min_score: Minimum score filter
            use_hybrid: Use hybrid search (semantic + fulltext)

        Returns:
            List of search results
        """
        if self.backend == StorageBackend.SUPABASE:
            if use_hybrid:
                return self._hybrid_search_supabase(query, limit, doc_type, min_score)
            else:
                return self._semantic_search_supabase(query, limit, doc_type)
        else:
            return self._search_local(query, limit, doc_type)

    def _semantic_search_supabase(
        self,
        query: str,
        limit: int,
        doc_type: Optional[str]
    ) -> List[SearchResult]:
        """Semantic search using embeddings."""
        try:
            query_embedding = self.get_embedding(query)

            # Use match_tower_knowledge_v2 with direct filter_type parameter
            result = self.supabase.rpc(
                "match_tower_knowledge_v2",
                {
                    "query_embedding": query_embedding,
                    "match_count": limit,
                    "filter_type": doc_type,  # Direct column filter (None = no filter)
                    "min_score": 0,
                    "similarity_threshold": 0.3
                }
            ).execute()

            if not result.data:
                return []

            return [
                SearchResult(
                    id=row.get("id", ""),
                    content=row["content"],
                    metadata=row.get("metadata", {}),
                    similarity=row["similarity"],
                    source=row.get("metadata", {}).get("source", ""),
                    doc_type=row.get("post_type", ""),  # v2 returns post_type directly
                    score=row.get("score", 0)  # v2 returns score directly
                )
                for row in result.data
            ]

        except Exception as e:
            print(f"[UnifiedKB] Semantic search error: {e}")
            return []

    def _fulltext_search_supabase(
        self,
        query: str,
        limit: int,
        doc_type: Optional[str]
    ) -> List[SearchResult]:
        """Fulltext search using ILIKE."""
        try:
            keywords = self._extract_keywords(query)
            if not keywords:
                return []

            results = []
            seen_ids = set()

            for keyword in keywords[:3]:
                q = self.supabase.table("tower_knowledge").select("*")

                if doc_type:
                    q = q.eq("post_type", doc_type)

                q = q.ilike("content", f"%{keyword}%")
                response = q.order("score", desc=True).limit(limit).execute()

                for row in (response.data or []):
                    post_id = row.get("post_id", row.get("id", ""))
                    if post_id not in seen_ids:
                        seen_ids.add(post_id)
                        results.append(SearchResult(
                            id=post_id,
                            content=row["content"],
                            metadata=row.get("metadata", {}),
                            similarity=0.8,  # Base score for fulltext match
                            source=row.get("metadata", {}).get("source", ""),
                            doc_type=row.get("post_type", ""),
                            score=row.get("score", 0)
                        ))

                if len(results) >= limit:
                    break

            return results[:limit]

        except Exception as e:
            print(f"[UnifiedKB] Fulltext search error: {e}")
            return []

    def _hybrid_search_supabase(
        self,
        query: str,
        limit: int,
        doc_type: Optional[str],
        min_score: int
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and fulltext results using RRF.
        """
        # Fetch more results than needed for fusion
        fetch_limit = limit * 3

        # Get results from both methods
        semantic_results = self._semantic_search_supabase(query, fetch_limit, doc_type)
        fulltext_results = self._fulltext_search_supabase(query, fetch_limit, doc_type)

        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results,
            fulltext_results,
            self.search_config.semantic_weight,
            self.search_config.fulltext_weight
        )

        # Apply boosting
        boosted_results = self._apply_boosts(fused_results)

        # Filter by min_score (Reddit score)
        if min_score > 0:
            boosted_results = [r for r in boosted_results if r.score >= min_score]

        # Filter by minimum similarity threshold
        min_sim = self.search_config.min_similarity
        if min_sim > 0:
            boosted_results = [r for r in boosted_results if r.similarity >= min_sim]

        # Deduplicate by base ID
        deduplicated = self._deduplicate_results(boosted_results)

        return deduplicated[:limit]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[SearchResult],
        fulltext_results: List[SearchResult],
        semantic_weight: float,
        fulltext_weight: float,
        k: int = 60
    ) -> List[SearchResult]:
        """
        Combine rankings using Reciprocal Rank Fusion.

        RRF Score = sum(1 / (k + rank)) for each ranking list
        """
        scores: Dict[str, float] = {}
        results_map: Dict[str, SearchResult] = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results):
            rrf_score = semantic_weight / (k + rank + 1)
            scores[result.id] = scores.get(result.id, 0) + rrf_score
            results_map[result.id] = result

        # Process fulltext results
        for rank, result in enumerate(fulltext_results):
            rrf_score = fulltext_weight / (k + rank + 1)
            scores[result.id] = scores.get(result.id, 0) + rrf_score
            if result.id not in results_map:
                results_map[result.id] = result

        # Sort by combined RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Update similarity scores with RRF scores
        fused = []
        for doc_id in sorted_ids:
            result = results_map[doc_id]
            result.similarity = scores[doc_id]
            fused.append(result)

        return fused

    def _apply_boosts(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply boosting factors based on metadata."""
        config = self.search_config

        for result in results:
            boost = 1.0

            # Type boost
            if result.doc_type == "guide":
                boost *= config.boost_guides
            elif result.doc_type == "wiki":
                boost *= config.boost_wiki

            # Score boost (high Reddit score)
            if result.score > 100:
                boost *= config.boost_high_score

            # Recency boost
            ingested_at = result.metadata.get("ingested_at")
            if ingested_at:
                try:
                    ingested_date = datetime.fromisoformat(ingested_at.replace('Z', '+00:00'))
                    if datetime.now(ingested_date.tzinfo) - ingested_date < timedelta(days=config.recency_days):
                        boost *= config.boost_recent
                except:
                    pass

            result.similarity *= boost

        # Re-sort after boosting
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on base ID (without chunk suffix)."""
        seen_bases = set()
        deduplicated = []

        for result in results:
            # Extract base ID (remove chunk suffix)
            base_id = result.id
            if "_chunk" in base_id:
                base_id = base_id.rsplit("_chunk", 1)[0]

            if base_id not in seen_bases:
                seen_bases.add(base_id)
                deduplicated.append(result)

        return deduplicated

    def _search_local(
        self,
        query: str,
        limit: int,
        doc_type: Optional[str]
    ) -> List[SearchResult]:
        """Search local storage using embeddings."""
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            print("[UnifiedKB] Local search requires numpy and sentence-transformers")
            return []

        vectors = self.local_embeddings.get('vectors')
        ids = self.local_embeddings.get('ids', [])

        if vectors is None or len(ids) == 0:
            print("[UnifiedKB] No local embeddings. Run build_local_embeddings() first.")
            return []

        # Get query embedding
        query_embedding = np.array(self.get_embedding(query))

        # Compute cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = vectors / norms
        similarities = np.dot(vectors_norm, query_norm)

        # Map IDs to chunks
        id_to_chunk = {c['id']: c for c in self.local_chunks['chunks']}

        # Build results
        results = []
        for chunk_id, sim in zip(ids, similarities):
            chunk = id_to_chunk.get(chunk_id)
            if not chunk:
                continue

            if doc_type and chunk.get('doc_type') != doc_type:
                continue

            results.append(SearchResult(
                id=chunk_id,
                content=chunk['content'],
                metadata=chunk.get('metadata', {}),
                similarity=float(sim),
                source=chunk.get('metadata', {}).get('source', ''),
                doc_type=chunk.get('doc_type', ''),
                score=chunk.get('score', 0)
            ))

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query, removing stop words."""
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'can', 'may', 'might', 'must', 'shall', 'this', 'that',
            'these', 'those', 'what', 'which', 'who', 'whom', 'how', 'when',
            'where', 'why', 'if', 'then', 'so', 'just', 'about', 'into', 'over'
        }

        words = query.lower().split()
        return [w for w in words if w not in stop_words and len(w) >= 2]

    # =========================================================================
    # Top Posts / Recent Posts (for feeds/banners)
    # =========================================================================

    def get_top_posts(
        self,
        limit: int = 10,
        doc_type: str = "post"
    ) -> List[SearchResult]:
        """
        Get top posts sorted by Reddit score.

        Args:
            limit: Maximum number of results
            doc_type: Filter by document type

        Returns:
            List of top-scoring posts
        """
        if self.backend == StorageBackend.SUPABASE:
            try:
                query = self.supabase.table("tower_knowledge")\
                    .select("*")\
                    .eq("post_type", doc_type)\
                    .order("score", desc=True)\
                    .limit(limit)

                result = query.execute()

                return [
                    SearchResult(
                        id=row.get("post_id", row.get("id", "")),
                        content=row["content"],
                        metadata=row.get("metadata", {}),
                        similarity=1.0,
                        source=row.get("metadata", {}).get("source", ""),
                        doc_type=row.get("post_type", ""),
                        score=row.get("score", 0)
                    )
                    for row in (result.data or [])
                ]
            except Exception as e:
                print(f"[UnifiedKB] Error getting top posts: {e}")
                return []
        else:
            # Local: sort by score
            chunks = [c for c in self.local_chunks['chunks'] if c.get('doc_type') == doc_type]
            chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
            return [
                SearchResult(
                    id=c['id'],
                    content=c['content'],
                    metadata=c.get('metadata', {}),
                    similarity=1.0,
                    source=c.get('metadata', {}).get('source', ''),
                    doc_type=c.get('doc_type', ''),
                    score=c.get('score', 0)
                )
                for c in chunks[:limit]
            ]

    def get_recent_posts(
        self,
        limit: int = 20,
        doc_type: str = "post"
    ) -> List[SearchResult]:
        """
        Get recent posts sorted by ingestion date.

        Args:
            limit: Maximum number of results
            doc_type: Filter by document type

        Returns:
            List of recent posts
        """
        if self.backend == StorageBackend.SUPABASE:
            try:
                query = self.supabase.table("tower_knowledge")\
                    .select("*")\
                    .eq("post_type", doc_type)\
                    .order("ingested_at", desc=True)\
                    .limit(limit)

                result = query.execute()

                return [
                    SearchResult(
                        id=row.get("post_id", row.get("id", "")),
                        content=row["content"],
                        metadata=row.get("metadata", {}),
                        similarity=1.0,  # Not a similarity search
                        source=row.get("metadata", {}).get("source", ""),
                        doc_type=row.get("post_type", ""),
                        score=row.get("score", 0)
                    )
                    for row in (result.data or [])
                ]
            except Exception as e:
                print(f"[UnifiedKB] Error getting recent posts: {e}")
                return []
        else:
            # Local: sort by ID (which includes timestamp)
            chunks = [c for c in self.local_chunks['chunks'] if c.get('doc_type') == doc_type]
            chunks.sort(key=lambda x: x.get('id', ''), reverse=True)
            return [
                SearchResult(
                    id=c['id'],
                    content=c['content'],
                    metadata=c.get('metadata', {}),
                    similarity=1.0,
                    source=c.get('metadata', {}).get('source', ''),
                    doc_type=c.get('doc_type', ''),
                    score=c.get('score', 0)
                )
                for c in chunks[:limit]
            ]

    # =========================================================================
    # Context Generation
    # =========================================================================

    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 2000,
        limit: int = 5
    ) -> str:
        """
        Get formatted context for RAG prompt.

        Args:
            query: Search query
            max_tokens: Approximate max tokens (chars / 4)
            limit: Max number of results

        Returns:
            Formatted context string
        """
        results = self.search(query, limit=limit)

        if not results:
            return ""

        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough estimate

        for i, result in enumerate(results, 1):
            source = result.source or "unknown"
            doc_type = result.doc_type or "unknown"
            score = result.score

            header = f"[Source {i} | {doc_type} | score: {score}]"
            content = result.content

            part = f"{header}\n{content}"

            if total_chars + len(part) > max_chars:
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n\n---\n\n".join(context_parts)

    # =========================================================================
    # Maintenance
    # =========================================================================

    def build_local_embeddings(self, force: bool = False) -> int:
        """Build embeddings for all local chunks."""
        if self.backend != StorageBackend.LOCAL:
            print("[UnifiedKB] build_local_embeddings only for local backend")
            return 0

        if not LOCAL_EMBEDDINGS_AVAILABLE:
            print("[UnifiedKB] numpy and sentence-transformers required")
            return 0

        existing_ids = set(self.local_embeddings.get('ids', []))
        chunks_to_embed = [
            c for c in self.local_chunks['chunks']
            if force or c['id'] not in existing_ids
        ]

        if not chunks_to_embed:
            print("[UnifiedKB] All chunks already have embeddings")
            return 0

        print(f"[UnifiedKB] Building embeddings for {len(chunks_to_embed)} chunks...")

        texts = [c['content'] for c in chunks_to_embed]
        ids = [c['id'] for c in chunks_to_embed]

        embeddings = self.get_embeddings_batch(texts)

        # Merge with existing
        if self.local_embeddings.get('vectors') is not None and not force:
            combined_vectors = np.vstack([
                self.local_embeddings['vectors'],
                np.array(embeddings)
            ])
            combined_ids = self.local_embeddings['ids'] + ids
        else:
            combined_vectors = np.array(embeddings)
            combined_ids = ids

        self.local_embeddings['vectors'] = combined_vectors
        self.local_embeddings['ids'] = combined_ids
        self._save_local_embeddings()

        print(f"[UnifiedKB] Built {len(embeddings)} embeddings")
        return len(embeddings)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if self.backend == StorageBackend.SUPABASE:
            try:
                # Count by type with pagination
                all_types = {}
                offset = 0
                batch_size = 1000

                while True:
                    result = self.supabase.table("tower_knowledge")\
                        .select("post_type")\
                        .range(offset, offset + batch_size - 1)\
                        .execute()

                    if not result.data:
                        break

                    for row in result.data:
                        t = row.get("post_type", "unknown")
                        all_types[t] = all_types.get(t, 0) + 1

                    offset += batch_size
                    if len(result.data) < batch_size:
                        break

                return {
                    "backend": "supabase",
                    "total_documents": sum(all_types.values()),
                    "by_type": all_types
                }
            except Exception as e:
                return {"backend": "supabase", "error": str(e)}
        else:
            type_counts = {}
            for chunk in self.local_chunks['chunks']:
                t = chunk.get('doc_type', 'unknown')
                type_counts[t] = type_counts.get(t, 0) + 1

            return {
                "backend": "local",
                "total_documents": len(self.local_chunks['chunks']),
                "total_embeddings": len(self.local_embeddings.get('ids', [])),
                "by_type": type_counts,
                "last_updated": self.local_index.get('last_updated')
            }


# Convenience function for backward compatibility
def create_knowledge_base(**kwargs) -> UnifiedKnowledgeBase:
    """Create a unified knowledge base instance."""
    return UnifiedKnowledgeBase(**kwargs)
