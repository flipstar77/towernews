"""Tower Knowledge Base - Supabase Vector Store for RAG."""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from supabase import create_client, Client


@dataclass
class SearchResult:
    """A search result from the knowledge base."""
    content: str
    metadata: Dict[str, Any]
    similarity: float


class TowerKnowledgeBase:
    """
    Vector-based knowledge base for The Tower game content.
    Uses Supabase pgvector for storage and OpenAI for embeddings.
    """

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536

    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        self.openai = OpenAI()

    def setup_database(self) -> bool:
        """
        Set up the vector store table in Supabase.
        Run this once to initialize the database.
        """
        # SQL to create the table with vector extension
        # This needs to be run in Supabase SQL editor
        sql = """
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Create knowledge base table
        CREATE TABLE IF NOT EXISTS tower_knowledge (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content TEXT NOT NULL,
            embedding vector(1536),
            metadata JSONB DEFAULT '{}',
            post_id TEXT,
            post_type TEXT,  -- 'post', 'comment', 'guide', 'wiki'
            subreddit TEXT DEFAULT 'TheTowerGame',
            score INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            ingested_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Create index for fast similarity search
        CREATE INDEX IF NOT EXISTS tower_knowledge_embedding_idx
        ON tower_knowledge
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

        -- Create index for metadata queries
        CREATE INDEX IF NOT EXISTS tower_knowledge_post_id_idx ON tower_knowledge(post_id);
        CREATE INDEX IF NOT EXISTS tower_knowledge_post_type_idx ON tower_knowledge(post_type);

        -- Create function for similarity search
        CREATE OR REPLACE FUNCTION match_tower_knowledge(
            query_embedding vector(1536),
            match_count int DEFAULT 5,
            filter_metadata jsonb DEFAULT '{}'
        )
        RETURNS TABLE (
            id UUID,
            content TEXT,
            metadata JSONB,
            similarity FLOAT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                tk.id,
                tk.content,
                tk.metadata,
                1 - (tk.embedding <=> query_embedding) AS similarity
            FROM tower_knowledge tk
            WHERE tk.metadata @> filter_metadata
            ORDER BY tk.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        """
        print("[KnowledgeBase] Please run the following SQL in Supabase SQL Editor:")
        print(sql)
        return True

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        response = self.openai.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def add_document(
        self,
        content: str,
        post_id: str,
        post_type: str = "post",
        metadata: Dict[str, Any] = None,
        score: int = 0
    ) -> Optional[str]:
        """Add a document to the knowledge base."""
        try:
            # Generate embedding
            embedding = self.get_embedding(content)

            # Prepare metadata
            meta = metadata or {}
            meta["post_id"] = post_id
            meta["post_type"] = post_type

            # Insert into Supabase
            result = self.supabase.table("tower_knowledge").insert({
                "content": content,
                "embedding": embedding,
                "metadata": meta,
                "post_id": post_id,
                "post_type": post_type,
                "score": score
            }).execute()

            if result.data:
                return result.data[0]["id"]
            return None

        except Exception as e:
            print(f"[KnowledgeBase] Error adding document: {e}")
            return None

    def add_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """Add multiple documents in batch."""
        added = 0
        for doc in documents:
            result = self.add_document(
                content=doc["content"],
                post_id=doc.get("post_id", ""),
                post_type=doc.get("post_type", "post"),
                metadata=doc.get("metadata"),
                score=doc.get("score", 0)
            )
            if result:
                added += 1
        return added

    def _get_base_post_id(self, post_id: str, post_type: str = None) -> str:
        """Extract base post ID without chunk suffix for deduplication.
        For comments, uses comment_id to allow multiple comments per thread."""
        if not post_id:
            return ""
        # Remove _chunkN suffix if present
        if "_chunk" in post_id:
            base = post_id.rsplit("_chunk", 1)[0]
        else:
            base = post_id

        # For comments, the post_id already includes comment_id (e.g., 1mq4wa5_comment_n8oafh9)
        # so each comment is unique. For posts/wiki, deduplicate by base post.
        return base

    def search(
        self,
        query: str,
        limit: int = 5,
        post_type: Optional[str] = None,
        min_score: int = 0
    ) -> List[SearchResult]:
        """
        Search the knowledge base for relevant content.
        Uses hybrid search: semantic similarity + fulltext matching.

        Args:
            query: Search query
            limit: Maximum results to return
            post_type: Filter by type (post, comment, guide, wiki)
            min_score: Minimum Reddit score filter

        Returns:
            List of SearchResult objects
        """
        try:
            results = []
            seen_base_ids = set()

            # 1. First try fulltext search for exact/partial matches
            fulltext_results = self._fulltext_search(query, limit * 2, post_type)
            for r in fulltext_results:
                base_id = self._get_base_post_id(r.metadata.get("post_id", ""))
                if base_id and base_id not in seen_base_ids:
                    seen_base_ids.add(base_id)
                    results.append(r)

            # 2. Then do semantic search
            semantic_results = self._semantic_search(query, limit * 2, post_type)
            for r in semantic_results:
                base_id = self._get_base_post_id(r.metadata.get("post_id", ""))
                if base_id and base_id not in seen_base_ids and len(results) < limit:
                    seen_base_ids.add(base_id)
                    results.append(r)

            # Filter by score if needed
            if min_score > 0:
                results = [r for r in results if r.metadata.get("score", 0) >= min_score]

            return results[:limit]

        except Exception as e:
            print(f"[KnowledgeBase] Search error: {e}")
            return []

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query for fulltext search."""
        # Common stop words to ignore
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'can', 'may', 'might', 'must', 'shall', 'this', 'that',
            'these', 'those', 'what', 'which', 'who', 'whom', 'how', 'when',
            'where', 'why', 'if', 'then', 'so', 'just', 'about', 'into', 'over',
            'after', 'before', 'between', 'under', 'again', 'there', 'here',
            'all', 'each', 'any', 'both', 'more', 'most', 'some', 'such', 'no',
            'not', 'only', 'own', 'same', 'than', 'too', 'very', 'also', 'deal'
        }

        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) >= 3]
        return keywords

    def _fulltext_search(
        self,
        query: str,
        limit: int = 5,
        post_type: Optional[str] = None
    ) -> List[SearchResult]:
        """Fulltext search using ILIKE for keyword matches."""
        if not query.strip():
            return []

        try:
            results = []
            seen_ids = set()

            # Extract keywords from query
            keywords = self._extract_keywords(query)
            if not keywords:
                keywords = [query]  # Fallback to full query

            # Search for each keyword
            for keyword in keywords[:3]:  # Limit to top 3 keywords
                if len(results) >= limit:
                    break

                # Search in content
                q = self.supabase.table("tower_knowledge").select("*")
                if post_type:
                    q = q.eq("post_type", post_type)
                q = q.ilike("content", f"%{keyword}%")
                content_results = q.order("score", desc=True).limit(limit).execute()

                for row in (content_results.data or []):
                    post_id = row.get("post_id", row.get("id", ""))
                    if post_id not in seen_ids:
                        seen_ids.add(post_id)
                        results.append(SearchResult(
                            content=row["content"],
                            metadata=row.get("metadata", {}),
                            similarity=0.95
                        ))

                # Also search in title
                if len(results) < limit:
                    q2 = self.supabase.table("tower_knowledge").select("*")
                    if post_type:
                        q2 = q2.eq("post_type", post_type)
                    q2 = q2.ilike("metadata->>title", f"%{keyword}%")
                    title_results = q2.order("score", desc=True).limit(limit).execute()

                    for row in (title_results.data or []):
                        post_id = row.get("post_id", row.get("id", ""))
                        if post_id not in seen_ids and len(results) < limit:
                            seen_ids.add(post_id)
                            results.append(SearchResult(
                                content=row["content"],
                                metadata=row.get("metadata", {}),
                                similarity=0.92
                            ))

            return results[:limit]

        except Exception as e:
            print(f"[KnowledgeBase] Fulltext search error: {e}")
            return []

    def _semantic_search(
        self,
        query: str,
        limit: int = 5,
        post_type: Optional[str] = None
    ) -> List[SearchResult]:
        """Semantic search using embeddings."""
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)

            # Build filter
            filter_meta = {}
            if post_type:
                filter_meta["post_type"] = post_type

            # Call the match function
            result = self.supabase.rpc(
                "match_tower_knowledge",
                {
                    "query_embedding": query_embedding,
                    "match_count": limit,
                    "filter_metadata": filter_meta
                }
            ).execute()

            if not result.data:
                return []

            return [
                SearchResult(
                    content=row["content"],
                    metadata=row.get("metadata", {}),
                    similarity=row["similarity"]
                )
                for row in result.data
            ]

        except Exception as e:
            print(f"[KnowledgeBase] Semantic search error: {e}")
            return []

    def get_top_posts(
        self,
        limit: int = 10,
        post_type: Optional[str] = None,
        flair: Optional[str] = None
    ) -> List[SearchResult]:
        """Get top posts by score."""
        try:
            q = self.supabase.table("tower_knowledge").select("*")

            if post_type:
                q = q.eq("post_type", post_type)

            result = q.order("score", desc=True).limit(limit).execute()

            if not result.data:
                return []

            results = []
            for row in result.data:
                meta = row.get("metadata", {})
                # Filter by flair if specified
                if flair and flair.lower() not in meta.get("flair", "").lower():
                    continue
                results.append(SearchResult(
                    content=row["content"],
                    metadata=meta,
                    similarity=1.0
                ))

            return results
        except Exception as e:
            print(f"[KnowledgeBase] Get top posts error: {e}")
            return []

    def get_recent_posts(
        self,
        limit: int = 10,
        post_type: Optional[str] = None
    ) -> List[SearchResult]:
        """Get most recent posts by ingestion date."""
        try:
            q = self.supabase.table("tower_knowledge").select("*")

            if post_type:
                q = q.eq("post_type", post_type)

            result = q.order("ingested_at", desc=True).limit(limit).execute()

            if not result.data:
                return []

            return [
                SearchResult(
                    content=row["content"],
                    metadata=row.get("metadata", {}),
                    similarity=1.0
                )
                for row in result.data
            ]
        except Exception as e:
            print(f"[KnowledgeBase] Get recent posts error: {e}")
            return []

    def get_context_for_topic(
        self,
        topic: str,
        post_title: str = "",
        limit: int = 5
    ) -> str:
        """
        Get relevant context for a news topic.
        Combines topic and title for better search.

        Returns formatted context string for LLM.
        """
        # Combine topic and title for search
        search_query = f"{topic} {post_title}".strip()

        results = self.search(search_query, limit=limit)

        if not results:
            return ""

        # Format context
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result.metadata
            source = meta.get("post_type", "unknown")
            score = meta.get("score", 0)

            context_parts.append(
                f"[Source {i} ({source}, score: {score})]:\n{result.content}"
            )

        return "\n\n".join(context_parts)

    def document_exists(self, post_id: str) -> bool:
        """Check if a document already exists."""
        try:
            result = self.supabase.table("tower_knowledge")\
                .select("id")\
                .eq("post_id", post_id)\
                .limit(1)\
                .execute()
            return len(result.data) > 0
        except:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        try:
            # Get all rows with pagination
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

            total = sum(all_types.values())

            return {
                "total_documents": total,
                "by_type": all_types
            }
        except Exception as e:
            print(f"[KnowledgeBase] Stats error: {e}")
            return {"total_documents": 0, "by_type": {}}
