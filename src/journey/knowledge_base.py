"""Knowledge Base - RAG system for Tower game understanding."""

import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from openai import OpenAI

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# ChromaDB has compatibility issues with pydantic 2.x
CHROMADB_AVAILABLE = False


class KnowledgeBase:
    """RAG-based knowledge base for The Tower game.

    Uses sentence-transformers for fast local embeddings (no API calls needed).
    Falls back to OpenAI embeddings if sentence-transformers unavailable.
    """

    def __init__(self, config: dict = None, use_local_embeddings: bool = True):
        self.config = config or {}
        self.openai_client = None  # Lazy initialization
        self.kb_dir = Path("data/journey/knowledge_base")
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        # Embedding model setup
        self.use_local = use_local_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self.embedding_model = None

        if self.use_local:
            self._init_local_embeddings()

        # Initialize JSON storage
        self._init_json_storage()

    def _init_local_embeddings(self):
        """Initialize local sentence-transformer model for embeddings."""
        try:
            # all-MiniLM-L6-v2 is fast and good for semantic search
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[KB] Initialized local embeddings with sentence-transformers")
        except Exception as e:
            print(f"[KB] Failed to load embedding model: {e}")
            self.use_local = False

    def _init_json_storage(self):
        """Initialize JSON file storage."""
        # Knowledge store paths
        self.chunks_file = self.kb_dir / "chunks.json"
        self.embeddings_file = self.kb_dir / "embeddings.npy"  # NumPy for faster loading
        self.embeddings_json = self.kb_dir / "embeddings.json"  # Fallback
        self.index_file = self.kb_dir / "index.json"

        # Load existing data
        self.chunks = self._load_json(self.chunks_file, {"chunks": []})
        self.index = self._load_json(self.index_file, {
            "last_updated": None,
            "total_chunks": 0,
            "sources": []
        })

        # Load embeddings (try NumPy first, then JSON)
        self.embeddings = {"embeddings": {}, "vectors": None, "ids": []}
        self._load_embeddings()

    def _get_openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self.openai_client is None:
            self.openai_client = OpenAI()
        return self.openai_client

    def _load_json(self, path: Path, default: dict) -> dict:
        """Load JSON file or return default."""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return default
        return default

    def _save_json(self, path: Path, data: dict):
        """Save data to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_embeddings(self):
        """Load embeddings from disk (NumPy or JSON)."""
        # Try NumPy format first (faster)
        if self.embeddings_file.exists():
            try:
                data = np.load(self.embeddings_file, allow_pickle=True).item()
                self.embeddings['vectors'] = data.get('vectors')
                self.embeddings['ids'] = data.get('ids', [])
                print(f"[KB] Loaded {len(self.embeddings['ids'])} embeddings from NumPy")
                return
            except Exception as e:
                print(f"[KB] Failed to load NumPy embeddings: {e}")

        # Fallback to JSON
        if self.embeddings_json.exists():
            try:
                json_data = self._load_json(self.embeddings_json, {"embeddings": {}})
                self.embeddings['embeddings'] = json_data.get('embeddings', {})
                print(f"[KB] Loaded {len(self.embeddings['embeddings'])} embeddings from JSON")
            except Exception:
                pass

    def _save_embeddings(self):
        """Save embeddings to disk in NumPy format."""
        if self.embeddings.get('vectors') is not None and self.embeddings.get('ids'):
            try:
                np.save(self.embeddings_file, {
                    'vectors': self.embeddings['vectors'],
                    'ids': self.embeddings['ids']
                })
            except Exception as e:
                print(f"[KB] Failed to save NumPy embeddings: {e}")

    def _chunk_id(self, text: str) -> str:
        """Generate unique ID for a chunk based on content hash."""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.use_local and self.embedding_model:
            # Use local sentence-transformers (fast, no API needed)
            return self.embedding_model.encode(text).tolist()
        else:
            # Fallback to OpenAI
            client = self._get_openai_client()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts at once (much faster)."""
        if self.use_local and self.embedding_model:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        else:
            # OpenAI batch (limited to 2048 per request)
            all_embeddings = []
            for i in range(0, len(texts), 2000):
                batch = texts[i:i+2000]
                client = self._get_openai_client()
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                all_embeddings.extend([e.embedding for e in response.data])
            return all_embeddings

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use numpy for speed
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        else:
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)

    def add_document(
        self,
        content: str,
        source: str,
        doc_type: str = "general",
        metadata: dict = None
    ) -> int:
        """Add a document to the knowledge base.

        Args:
            content: The text content to add
            source: Source identifier (e.g., "video_xyz", "manual")
            doc_type: Type of content (tips, mechanics, strategy, etc.)
            metadata: Additional metadata

        Returns:
            Number of chunks added
        """
        # Split into chunks (simple paragraph-based chunking)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        # If content is short, treat as single chunk
        if len(paragraphs) <= 1:
            paragraphs = [content]

        chunks_added = 0
        for para in paragraphs:
            if len(para) < 50:  # Skip very short chunks
                continue

            chunk_id = self._chunk_id(para)

            # Skip if already exists in JSON store
            if any(c['id'] == chunk_id for c in self.chunks['chunks']):
                continue

            # Create chunk metadata
            chunk = {
                "id": chunk_id,
                "content": para,
                "source": source,
                "type": doc_type,
                "metadata": metadata or {},
                "added_at": datetime.now().isoformat()
            }

            # Add to JSON store
            self.chunks['chunks'].append(chunk)
            chunks_added += 1

        # Update index
        if source not in self.index['sources']:
            self.index['sources'].append(source)
        self.index['total_chunks'] = len(self.chunks['chunks'])
        self.index['last_updated'] = datetime.now().isoformat()

        # Save JSON
        self._save_json(self.chunks_file, self.chunks)
        self._save_json(self.index_file, self.index)

        return chunks_added

    def build_embeddings(self, force: bool = False, batch_size: int = 100) -> int:
        """Build embeddings for all chunks using batch processing.

        Args:
            force: Rebuild all embeddings even if they exist
            batch_size: Number of texts to embed at once (faster)

        Returns:
            Number of embeddings created
        """
        # Determine which chunks need embeddings
        chunks_to_embed = []
        existing_ids = set(self.embeddings.get('ids', []))

        # Also check legacy format
        if self.embeddings.get('embeddings'):
            existing_ids.update(self.embeddings['embeddings'].keys())

        for chunk in self.chunks['chunks']:
            chunk_id = chunk['id']
            if force or chunk_id not in existing_ids:
                chunks_to_embed.append(chunk)

        if not chunks_to_embed:
            print("[KB] All chunks already have embeddings")
            return 0

        print(f"[KB] Building embeddings for {len(chunks_to_embed)} chunks...")

        # Process in batches for efficiency
        all_embeddings = []
        all_ids = []

        for i in range(0, len(chunks_to_embed), batch_size):
            batch = chunks_to_embed[i:i + batch_size]
            texts = [c['content'] for c in batch]
            ids = [c['id'] for c in batch]

            try:
                batch_embeddings = self._get_embeddings_batch(texts)
                all_embeddings.extend(batch_embeddings)
                all_ids.extend(ids)
                print(f"[KB] Processed {i + len(batch)}/{len(chunks_to_embed)} chunks")
            except Exception as e:
                print(f"[KB] Error processing batch {i}: {e}")

        if not all_embeddings:
            return 0

        # Convert to NumPy array for fast similarity search
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Merge with existing embeddings if any
            if self.embeddings.get('vectors') is not None and len(self.embeddings.get('ids', [])) > 0:
                existing_vectors = self.embeddings['vectors']
                existing_ids_list = self.embeddings['ids']

                # Combine old and new
                combined_vectors = np.vstack([existing_vectors, np.array(all_embeddings)])
                combined_ids = existing_ids_list + all_ids
            else:
                combined_vectors = np.array(all_embeddings)
                combined_ids = all_ids

            self.embeddings['vectors'] = combined_vectors
            self.embeddings['ids'] = combined_ids
            self._save_embeddings()
        else:
            # Fallback: store in JSON format
            for chunk_id, embedding in zip(all_ids, all_embeddings):
                self.embeddings['embeddings'][chunk_id] = embedding
            self._save_json(self.embeddings_json, self.embeddings)

        print(f"[KB] Created {len(all_embeddings)} embeddings")
        return len(all_embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str = None
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            doc_type: Filter by document type

        Returns:
            List of relevant chunks with similarity scores
        """
        # Use NumPy vectorized search if available
        if self.use_local and self.embeddings.get('vectors') is not None:
            return self._search_numpy(query, top_k, doc_type)

        # Fallback to legacy JSON embedding search
        return self._search_json(query, top_k, doc_type)

    def _search_numpy(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str = None
    ) -> List[Dict[str, Any]]:
        """Fast vectorized search using NumPy."""
        vectors = self.embeddings.get('vectors')
        ids = self.embeddings.get('ids', [])

        if vectors is None or len(ids) == 0:
            print("[KB] No embeddings available, falling back to JSON search")
            return self._search_json(query, top_k, doc_type)

        # Get query embedding
        query_embedding = np.array(self._get_embedding(query))

        # Compute cosine similarity with all vectors at once (vectorized)
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        # Normalize all vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = vectors / norms
        # Compute similarities
        similarities = np.dot(vectors_norm, query_norm)

        # Create id-to-chunk mapping
        id_to_chunk = {c['id']: c for c in self.chunks['chunks']}

        # Build results with filtering
        results = []
        for i, (chunk_id, sim) in enumerate(zip(ids, similarities)):
            chunk = id_to_chunk.get(chunk_id)
            if not chunk:
                continue

            # Filter by type if specified
            if doc_type and chunk.get('type') != doc_type:
                continue

            results.append({
                **chunk,
                "similarity": float(sim)
            })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def _search_json(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str = None
    ) -> List[Dict[str, Any]]:
        """Search using legacy JSON embeddings."""
        if not self.embeddings.get('embeddings'):
            print("[KB] No embeddings available. Run build_embeddings() first.")
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Calculate similarity for all chunks
        results = []
        for chunk in self.chunks['chunks']:
            chunk_id = chunk['id']

            # Skip if no embedding
            if chunk_id not in self.embeddings['embeddings']:
                continue

            # Filter by type if specified
            if doc_type and chunk.get('type') != doc_type:
                continue

            similarity = self._cosine_similarity(
                query_embedding,
                self.embeddings['embeddings'][chunk_id]
            )

            results.append({
                **chunk,
                "similarity": similarity
            })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def ask(
        self,
        question: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Ask a question and get an AI-generated answer using RAG.

        Args:
            question: The question to answer
            include_sources: Whether to include source references

        Returns:
            Answer with optional sources
        """
        # Search for relevant context
        relevant_chunks = self.search(question, top_k=5)

        if not relevant_chunks:
            return {
                "answer": "I don't have enough information in my knowledge base to answer that question. Try adding more content about The Tower game.",
                "sources": [],
                "confidence": 0.0
            }

        # Build context from relevant chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"[{i+1}] {chunk['content']}")

        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = f"""You are a helpful assistant that answers questions about The Tower mobile game.
Use the following knowledge base context to answer the question. If the context doesn't contain enough information, say so.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, concise answer based on the context above. Reference specific information when possible."""

        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert on The Tower mobile game. Answer questions accurately based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            answer = response.choices[0].message.content

            # Calculate average confidence from similarity scores
            avg_similarity = sum(c['similarity'] for c in relevant_chunks) / len(relevant_chunks)

            result = {
                "answer": answer,
                "confidence": avg_similarity
            }

            if include_sources:
                result["sources"] = [
                    {
                        "source": c['source'],
                        "type": c['type'],
                        "similarity": c['similarity'],
                        "excerpt": c['content'][:200] + "..." if len(c['content']) > 200 else c['content']
                    }
                    for c in relevant_chunks
                ]

            return result

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }

    def import_from_topic_analyses(self) -> int:
        """Import knowledge from saved topic analyses.

        Returns:
            Number of chunks imported
        """
        topic_dir = Path("data/journey/topic_analysis")
        if not topic_dir.exists():
            return 0

        total_chunks = 0

        for file in topic_dir.glob("video_*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)

                video_id = analysis.get("video_id", file.stem)
                video_title = analysis.get("video_title", "Unknown")

                # Add main topic
                main_topic = analysis.get("main_topic", "")
                if main_topic:
                    total_chunks += self.add_document(
                        content=f"Video: {video_title}\n\n{main_topic}",
                        source=f"video_{video_id}",
                        doc_type="overview",
                        metadata={"video_title": video_title}
                    )

                # Add tips
                tips = analysis.get("tips", [])
                if tips:
                    tips_content = "Tips from Tower content creator:\n\n" + "\n\n".join(f"- {tip}" for tip in tips)
                    total_chunks += self.add_document(
                        content=tips_content,
                        source=f"video_{video_id}",
                        doc_type="tips",
                        metadata={"video_title": video_title}
                    )

                # Add questions answered
                questions = analysis.get("questions_answered", [])
                if questions:
                    qa_content = "Questions covered:\n\n" + "\n\n".join(f"Q: {q}" for q in questions)
                    total_chunks += self.add_document(
                        content=qa_content,
                        source=f"video_{video_id}",
                        doc_type="qa",
                        metadata={"video_title": video_title}
                    )

                # Add key takeaways
                takeaways = analysis.get("key_takeaways", [])
                if takeaways:
                    takeaways_content = "Key takeaways:\n\n" + "\n\n".join(f"- {t}" for t in takeaways)
                    total_chunks += self.add_document(
                        content=takeaways_content,
                        source=f"video_{video_id}",
                        doc_type="insights",
                        metadata={"video_title": video_title}
                    )

            except Exception as e:
                print(f"[KB] Error importing {file}: {e}")

        return total_chunks

    def import_from_transcripts(self) -> int:
        """Import knowledge from saved transcripts.

        Returns:
            Number of chunks imported
        """
        transcript_dir = Path("data/journey/transcripts")
        if not transcript_dir.exists():
            return 0

        total_chunks = 0

        for file in transcript_dir.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                transcript = data.get("transcript", "")
                video_id = file.stem

                if transcript:
                    # Split transcript into reasonable chunks
                    # (transcripts can be very long)
                    words = transcript.split()
                    chunk_size = 500  # words per chunk

                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i + chunk_size]
                        chunk_text = " ".join(chunk_words)

                        if len(chunk_text) > 100:  # Skip tiny chunks
                            total_chunks += self.add_document(
                                content=chunk_text,
                                source=f"transcript_{video_id}",
                                doc_type="transcript",
                                metadata={"chunk_index": i // chunk_size}
                            )

            except Exception as e:
                print(f"[KB] Error importing transcript {file}: {e}")

        return total_chunks

    def add_manual_knowledge(self, entries: List[Dict[str, str]]) -> int:
        """Add manually curated knowledge entries.

        Args:
            entries: List of dicts with 'content', 'type', and optional 'title'

        Returns:
            Number of entries added
        """
        total = 0
        for entry in entries:
            content = entry.get("content", "")
            doc_type = entry.get("type", "manual")
            title = entry.get("title", "")

            if title:
                content = f"{title}\n\n{content}"

            if content:
                total += self.add_document(
                    content=content,
                    source="manual",
                    doc_type=doc_type
                )

        return total

    def import_from_reddit(
        self,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 100,
        include_comments: bool = True,
        min_score: int = 5
    ) -> int:
        """Import knowledge from r/TheTowerGame subreddit.

        Args:
            sort: Sort order (hot, new, top, rising)
            time_filter: Time filter for top posts
            limit: Maximum posts to fetch
            include_comments: Whether to also fetch top comments
            min_score: Minimum score for posts/comments

        Returns:
            Number of chunks imported
        """
        import time as time_module

        scraper = RedditScraper()
        total_chunks = 0

        print(f"[KB] Fetching {limit} {sort} posts from r/TheTowerGame...")
        posts = scraper.fetch_posts(sort=sort, time_filter=time_filter, limit=limit)
        print(f"[KB] Found {len(posts)} posts")

        for post in posts:
            if post.get("score", 0) < min_score:
                continue

            # Add post title and content
            post_content = f"Reddit Post: {post.get('title', '')}"
            if post.get('flair'):
                post_content += f" [{post['flair']}]"
            if post.get('selftext'):
                post_content += f"\n\n{post['selftext']}"

            if len(post_content) > 100:
                total_chunks += self.add_document(
                    content=post_content,
                    source=f"reddit_{post.get('id', 'unknown')}",
                    doc_type="reddit_post",
                    metadata={
                        "score": post.get("score"),
                        "author": post.get("author"),
                        "url": post.get("url")
                    }
                )

            # Fetch and add top comments
            if include_comments and post.get("num_comments", 0) > 0:
                time_module.sleep(0.5)  # Rate limiting
                comments = scraper.fetch_post_comments(post.get("id"), limit=10)

                for comment in comments:
                    if comment.get("score", 0) < min_score:
                        continue

                    comment_body = comment.get("body", "")
                    if len(comment_body) > 100 and "[deleted]" not in comment_body:
                        total_chunks += self.add_document(
                            content=f"Reddit Comment on '{post.get('title', '')}':\n\n{comment_body}",
                            source=f"reddit_comment_{comment.get('id', 'unknown')}",
                            doc_type="reddit_comment",
                            metadata={
                                "score": comment.get("score"),
                                "author": comment.get("author"),
                                "post_id": post.get("id")
                            }
                        )

        return total_chunks

    def import_from_notion(self, urls: List[str] = None) -> int:
        """Import knowledge from Tower Notion wiki pages.

        Args:
            urls: List of Notion page URLs (uses defaults if not provided)

        Returns:
            Number of chunks imported
        """
        scraper = NotionScraper()
        total_chunks = 0

        if urls:
            pages = [scraper.fetch_page(url) for url in urls]
        else:
            print("[KB] Fetching known Tower wiki pages...")
            pages = scraper.fetch_all_known_pages()

        for page in pages:
            if not page.get("content"):
                continue

            print(f"[KB] Processing: {page.get('title', page.get('url', 'Unknown'))}")

            # Split long content into chunks
            content = page.get("content", "")
            title = page.get("title", "Tower Wiki")

            # Split by paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            for i, para in enumerate(paragraphs):
                if len(para) > 100:
                    chunk_content = para
                    if i == 0 and title:
                        chunk_content = f"{title}\n\n{para}"

                    total_chunks += self.add_document(
                        content=chunk_content,
                        source=f"notion_{page.get('url', 'unknown')[:50]}",
                        doc_type="wiki",
                        metadata={
                            "page_title": title,
                            "url": page.get("url"),
                            "chunk_index": i
                        }
                    )

        return total_chunks

    def import_from_fandom(self, limit: int = 1000) -> int:
        """Import knowledge from The Tower Fandom wiki.

        Args:
            limit: Maximum number of pages to fetch (default 1000 for full site)

        Returns:
            Number of chunks imported
        """
        scraper = FandomScraper()
        total_chunks = 0

        print(f"[KB] Fetching up to {limit} pages from Fandom wiki...")
        pages = scraper.fetch_all_pages(limit=limit)
        print(f"[KB] Found {len(pages)} pages with content")

        for page in pages:
            if not page.get("content"):
                continue

            # Split long content into chunks
            content = page.get("content", "")
            title = page.get("title", "Wiki Page")

            # Split by paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            for i, para in enumerate(paragraphs):
                if len(para) > 100:
                    chunk_content = para
                    if i == 0 and title:
                        chunk_content = f"{title}\n\n{para}"

                    total_chunks += self.add_document(
                        content=chunk_content,
                        source=f"fandom_{title.replace(' ', '_')[:30]}",
                        doc_type="wiki",
                        metadata={
                            "page_title": title,
                            "url": page.get("url"),
                            "chunk_index": i
                        }
                    )

        return total_chunks

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics.

        Returns:
            Dictionary with stats
        """
        # Count by type
        type_counts = {}
        for chunk in self.chunks['chunks']:
            doc_type = chunk.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        # Count embeddings - check NumPy first, then JSON
        if self.embeddings.get('ids'):
            num_embeddings = len(self.embeddings['ids'])
        else:
            num_embeddings = len(self.embeddings.get('embeddings', {}))

        return {
            "total_chunks": len(self.chunks['chunks']),
            "total_embeddings": num_embeddings,
            "embeddings_complete": num_embeddings >= len(self.chunks['chunks']),
            "chunks_by_type": type_counts,
            "sources": self.index.get('sources', []),
            "last_updated": self.index.get('last_updated'),
            "backend": "numpy" if self.use_local else "json"
        }

    def clear(self):
        """Clear all knowledge base data."""
        self.chunks = {"chunks": []}
        self.embeddings = {"embeddings": {}, "vectors": None, "ids": []}
        self.index = {
            "last_updated": None,
            "total_chunks": 0,
            "sources": []
        }

        # Remove embedding files
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.embeddings_json.exists():
            self.embeddings_json.unlink()

        self._save_json(self.chunks_file, self.chunks)
        self._save_json(self.index_file, self.index)


# Default Tower game knowledge to seed the database
class RedditScraper:
    """Scraper for r/TheTowerGame subreddit."""

    def __init__(self):
        self.base_url = "https://www.reddit.com/r/TheTowerGame"
        self.headers = {
            "User-Agent": "TowerNewsBot/1.0 (Educational Research)"
        }

    def fetch_posts(
        self,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch posts from the subreddit.

        Args:
            sort: Sort order (hot, new, top, rising)
            time_filter: Time filter for top posts (hour, day, week, month, year, all)
            limit: Maximum number of posts to fetch

        Returns:
            List of post dictionaries
        """
        import requests
        import time

        posts = []
        after = None

        while len(posts) < limit:
            url = f"{self.base_url}/{sort}.json"
            params = {
                "limit": min(100, limit - len(posts)),
                "t": time_filter,
                "raw_json": 1
            }
            if after:
                params["after"] = after

            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()

                children = data.get("data", {}).get("children", [])
                if not children:
                    break

                for child in children:
                    post_data = child.get("data", {})
                    posts.append({
                        "id": post_data.get("id"),
                        "title": post_data.get("title", ""),
                        "selftext": post_data.get("selftext", ""),
                        "score": post_data.get("score", 0),
                        "num_comments": post_data.get("num_comments", 0),
                        "created_utc": post_data.get("created_utc"),
                        "author": post_data.get("author", ""),
                        "flair": post_data.get("link_flair_text", ""),
                        "url": f"https://reddit.com{post_data.get('permalink', '')}"
                    })

                after = data.get("data", {}).get("after")
                if not after:
                    break

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"[Reddit] Error fetching posts: {e}")
                break

        return posts[:limit]

    def fetch_post_comments(self, post_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch comments for a specific post.

        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments

        Returns:
            List of comment dictionaries
        """
        import requests

        url = f"https://www.reddit.com/r/TheTowerGame/comments/{post_id}.json"
        params = {"limit": limit, "raw_json": 1}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            comments = []
            if len(data) > 1:
                comment_data = data[1].get("data", {}).get("children", [])
                for child in comment_data:
                    if child.get("kind") == "t1":
                        c = child.get("data", {})
                        comments.append({
                            "id": c.get("id"),
                            "body": c.get("body", ""),
                            "score": c.get("score", 0),
                            "author": c.get("author", "")
                        })

            return comments[:limit]

        except Exception as e:
            print(f"[Reddit] Error fetching comments: {e}")
            return []


class NotionScraper:
    """Scraper for The Tower Notion wiki using Playwright for JavaScript rendering."""

    # Known Tower Notion pages
    KNOWN_PAGES = [
        "https://the-tower.notion.site/",
    ]

    def __init__(self):
        self.playwright_available = False
        try:
            from playwright.sync_api import sync_playwright
            self.playwright_available = True
        except ImportError:
            print("[Notion] Playwright not installed. Run: pip install playwright && playwright install chromium")

    def fetch_page(self, url: str) -> Dict[str, Any]:
        """Fetch and parse a Notion page using Playwright.

        Args:
            url: Notion page URL

        Returns:
            Dictionary with page content
        """
        if not self.playwright_available:
            return {
                "url": url,
                "title": "",
                "content": "",
                "error": "Playwright not installed. Run: pip install playwright && playwright install chromium"
            }

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)

                # Wait for content to load
                page.wait_for_selector(".notion-page-content", timeout=10000)

                # Get page title
                title = page.title()

                # Get all text content
                content = page.evaluate("""
                    () => {
                        const content = document.querySelector('.notion-page-content');
                        if (content) {
                            return content.innerText;
                        }
                        return document.body.innerText;
                    }
                """)

                browser.close()

                return {
                    "url": url,
                    "title": title,
                    "content": content
                }

        except Exception as e:
            print(f"[Notion] Error fetching page: {e}")
            return {"url": url, "title": "", "content": "", "error": str(e)}

    def fetch_all_known_pages(self) -> List[Dict[str, Any]]:
        """Fetch all known Tower wiki pages.

        Returns:
            List of page dictionaries
        """
        import time
        pages = []

        for url in self.KNOWN_PAGES:
            print(f"[Notion] Fetching: {url}")
            page = self.fetch_page(url)
            if page.get("content"):
                pages.append(page)
            elif page.get("error"):
                print(f"[Notion] Error: {page['error']}")
            time.sleep(1)  # Rate limiting

        return pages

    def get_child_pages(self, parent_url: str) -> List[str]:
        """Get all child page links from a Notion page.

        Args:
            parent_url: Parent Notion page URL

        Returns:
            List of child page URLs
        """
        if not self.playwright_available:
            return []

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(parent_url, wait_until="networkidle", timeout=30000)

                # Get all links that look like Notion page links
                links = page.evaluate("""
                    () => {
                        const links = document.querySelectorAll('a[href*="notion.site"]');
                        return Array.from(links).map(a => a.href).filter(href =>
                            href.includes('notion.site') && !href.includes('#')
                        );
                    }
                """)

                browser.close()
                return list(set(links))  # Remove duplicates

        except Exception as e:
            print(f"[Notion] Error getting child pages: {e}")
            return []


class FandomScraper:
    """Scraper for The Tower Fandom wiki using MediaWiki API."""

    BASE_URL = "https://the-tower-idle-tower-defense.fandom.com"
    API_URL = "https://the-tower-idle-tower-defense.fandom.com/api.php"

    def __init__(self):
        self.headers = {
            "User-Agent": "TowerNewsBot/1.0 (Educational Research)"
        }

    def get_all_page_titles(self, limit: int = 500) -> List[str]:
        """Get list of all wiki page titles using MediaWiki API.

        Args:
            limit: Maximum number of pages to fetch

        Returns:
            List of page titles
        """
        import requests

        titles = []
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": min(500, limit),
            "format": "json"
        }

        try:
            while len(titles) < limit:
                response = requests.get(self.API_URL, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()

                pages = data.get("query", {}).get("allpages", [])
                for page in pages:
                    title = page.get("title", "")
                    # Include ALL pages including subpages
                    if title:
                        titles.append(title)

                # Check for continuation
                if "continue" in data:
                    params["apcontinue"] = data["continue"]["apcontinue"]
                else:
                    break

        except Exception as e:
            print(f"[Fandom] Error getting page list: {e}")

        return titles[:limit]

    def fetch_page_content(self, title: str) -> Dict[str, Any]:
        """Fetch page content using MediaWiki API.

        Args:
            title: Page title

        Returns:
            Dictionary with page content
        """
        import requests
        import re

        params = {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json"
        }

        try:
            response = requests.get(self.API_URL, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":  # Page exists
                    revisions = page_data.get("revisions", [])
                    if revisions:
                        raw_content = revisions[0].get("slots", {}).get("main", {}).get("*", "")

                        # Clean up wiki markup
                        content = self._clean_wiki_markup(raw_content)

                        return {
                            "title": page_data.get("title", title),
                            "content": content,
                            "url": f"{self.BASE_URL}/wiki/{title.replace(' ', '_')}"
                        }

            return {"title": title, "content": "", "url": ""}

        except Exception as e:
            print(f"[Fandom] Error fetching page {title}: {e}")
            return {"title": title, "content": "", "error": str(e)}

    def _clean_wiki_markup(self, text: str) -> str:
        """Clean wiki markup to plain text."""
        import re

        # Remove templates like {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)

        # Remove categories like [[Category:...]]
        text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)

        # Convert links like [[Page|Text]] to Text
        text = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', text)

        # Convert simple links [[Page]] to Page
        text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)

        # Remove external links [url text]
        text = re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', text)

        # Remove images [[File:...]]
        text = re.sub(r'\[\[File:[^\]]*\]\]', '', text)

        # Convert headers == Text == to Text
        text = re.sub(r'={2,}\s*([^=]+)\s*={2,}', r'\n\1\n', text)

        # Remove bold/italic markup
        text = re.sub(r"'{2,}", '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def fetch_all_pages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch content from all wiki pages.

        Args:
            limit: Maximum number of pages

        Returns:
            List of page dictionaries with content
        """
        import time

        titles = self.get_all_page_titles(limit=limit)
        print(f"[Fandom] Found {len(titles)} pages to fetch")

        pages = []
        for i, title in enumerate(titles):
            if i % 10 == 0:
                print(f"[Fandom] Fetching {i+1}/{len(titles)}...")

            page = self.fetch_page_content(title)
            if page.get("content") and len(page["content"]) > 100:
                pages.append(page)

            time.sleep(0.2)  # Rate limiting

        print(f"[Fandom] Fetched {len(pages)} pages with content")
        return pages


TOWER_GAME_KNOWLEDGE = [
    {
        "title": "Game Overview",
        "type": "mechanics",
        "content": """The Tower is an idle defense mobile game where you build and upgrade a tower to survive waves of enemies.
The goal is to reach as high a wave as possible while collecting coins, cells, and other resources.
Key resources include coins (primary currency), cells (for upgrades), stones (premium currency), and gems.
Progress is measured in tiers (T1-T17+) and waves (can reach billions with enough upgrades)."""
    },
    {
        "title": "Economy Basics",
        "type": "mechanics",
        "content": """Coins are the primary resource earned from killing enemies.
Coins per hour (CPH) is a key metric for measuring farming efficiency.
Cells are earned from battles and used for permanent upgrades in the lab.
Golden Bots are special enemies that drop extra rewards when killed.
Economy upgrades affect cash generation and coin multipliers."""
    },
    {
        "title": "Tournament System",
        "type": "mechanics",
        "content": """Tournaments run regularly and pit players against each other in brackets.
Tournament performance is based on wave reached within a time limit.
Brackets include Legends (top tier), Champions, and lower divisions.
Tournament rewards include stones, gems, and exclusive items.
Strategy differs between regular runs and tournament runs."""
    },
    {
        "title": "Lab and Research",
        "type": "mechanics",
        "content": """The Lab contains permanent upgrades purchased with cells.
Lab priorities should focus on damage, health, and economy early on.
Ultimate Weapons are powerful upgrades unlocked through lab research.
Lab upgrades persist between runs and provide cumulative bonuses."""
    },
    {
        "title": "Workshop and Cards",
        "type": "mechanics",
        "content": """The Workshop is where you upgrade cards that boost your tower.
Cards provide various bonuses like damage, health regen, and utility.
Card levels increase through spending coins in the workshop.
Card setups vary based on whether you're farming or pushing waves."""
    },
    {
        "title": "Mods and Relics",
        "type": "mechanics",
        "content": """Mods are equipment that provide stat bonuses.
Ancient mods (Anc mods) are the highest tier mods available.
Relics provide special abilities and bonuses.
Health regen relics are particularly valuable for sustain.
Mod pulls are done using gems or special currencies."""
    },
    {
        "title": "Wave and Tier Progression",
        "type": "strategy",
        "content": """Tiers unlock at specific wave milestones.
Higher tiers provide access to new upgrades and content.
LTC (Lifetime Coins) is a measure of total coins earned across all runs.
Progression involves balancing farming runs with push runs.
Wave notation uses B (billion), T (trillion), q (quadrillion)."""
    },
    {
        "title": "Farming Strategies",
        "type": "strategy",
        "content": """Farming runs focus on coin and cell collection efficiency.
4x everything means upgrading all available options to maximize gains.
G comp refers to a specific card composition for farming.
Farming efficiency is measured in coins/hour and cells/hour.
Short runs with quick resets can be more efficient than long pushes."""
    }
]
