"""Knowledge Ingester - Scrapes Reddit and ingests into vector store."""

import requests
import time
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .knowledge_base import TowerKnowledgeBase


class KnowledgeIngester:
    """
    Ingests content from r/TheTowerGame into the knowledge base.
    Handles posts, comments, guides, and wiki content.
    """

    BASE_URL = "https://www.reddit.com"
    USER_AGENT = "TowerNews/1.0 (Knowledge Ingester)"

    def __init__(self):
        self.kb = TowerKnowledgeBase()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        self.last_request = 0
        self.min_delay = 2.0  # Reddit rate limit

    def _rate_limit(self):
        """Ensure we don't hit Reddit rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request = time.time()

    def _clean_text(self, text: str) -> str:
        """Clean text for embedding."""
        if not text:
            return ""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove Reddit formatting
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'[*_~`#>]', '', text)
        return text.strip()

    def _chunk_text(self, text: str, max_length: int = 1500) -> List[str]:
        """Split long text into chunks for embedding."""
        if len(text) <= max_length:
            return [text] if text else []

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def fetch_posts(
        self,
        subreddit: str = "TheTowerGame",
        limit: int = 100,
        timeframe: str = "all",
        sort: str = "top",
        after: str = None
    ) -> List[Dict[str, Any]]:
        """Fetch posts from subreddit."""
        self._rate_limit()

        url = f"{self.BASE_URL}/r/{subreddit}/{sort}.json"
        params = {
            "limit": min(limit, 100),
            "t": timeframe,
            "raw_json": 1
        }
        if after:
            params["after"] = after

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            posts = []
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                posts.append({
                    "id": post.get("id"),
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "author": post.get("author", "[deleted]"),
                    "created_utc": post.get("created_utc", 0),
                    "flair": post.get("link_flair_text", ""),
                    "permalink": post.get("permalink", ""),
                    "url": post.get("url", "")
                })

            # Get next page token
            after_token = data.get("data", {}).get("after")

            return posts, after_token

        except Exception as e:
            print(f"[Ingester] Error fetching posts: {e}")
            return [], None

    def fetch_comments(
        self,
        subreddit: str,
        post_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Fetch comments for a post."""
        self._rate_limit()

        url = f"{self.BASE_URL}/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": limit, "sort": "top", "raw_json": 1}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            comments = []
            if len(data) > 1:
                comment_data = data[1].get("data", {}).get("children", [])
                for child in comment_data:
                    if child.get("kind") != "t1":
                        continue
                    comment = child.get("data", {})
                    if comment.get("body", "").strip():
                        comments.append({
                            "id": comment.get("id"),
                            "body": comment.get("body", ""),
                            "score": comment.get("score", 0),
                            "author": comment.get("author", "[deleted]"),
                            "created_utc": comment.get("created_utc", 0)
                        })

            return comments

        except Exception as e:
            print(f"[Ingester] Error fetching comments: {e}")
            return []

    def ingest_post(
        self,
        post: Dict[str, Any],
        include_comments: bool = True,
        min_comment_score: int = 3,
        max_comments: int = 50
    ) -> int:
        """
        Ingest a single post and its comments.
        Returns number of documents added.
        """
        added = 0
        post_id = post["id"]

        # Skip if already ingested
        if self.kb.document_exists(post_id):
            return 0

        # Prepare post content
        title = post.get("title", "")
        selftext = self._clean_text(post.get("selftext", ""))
        flair = post.get("flair", "")

        # Build Reddit URL
        permalink = post.get("permalink", "")
        reddit_url = f"https://www.reddit.com{permalink}" if permalink else ""

        # Create post document with flair tag
        flair_tag = f"[Flair: {flair}] " if flair else ""
        content = f"{flair_tag}{title}"
        if selftext:
            content += f"\n\n{selftext}"

        # Chunk if too long
        chunks = self._chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{post_id}_chunk{i}" if len(chunks) > 1 else post_id
            result = self.kb.add_document(
                content=chunk,
                post_id=chunk_id,
                post_type="post",
                metadata={
                    "title": title,
                    "flair": flair,
                    "author": post.get("author"),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "permalink": permalink,
                    "reddit_url": reddit_url,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                },
                score=post.get("score", 0)
            )
            if result:
                added += 1

        # Ingest comments
        if include_comments and post.get("num_comments", 0) > 0:
            comments = self.fetch_comments("TheTowerGame", post_id, limit=max_comments)

            for comment in comments:
                if comment.get("score", 0) < min_comment_score:
                    continue

                comment_text = self._clean_text(comment.get("body", ""))
                if not comment_text or len(comment_text) < 20:
                    continue

                # Add context from post title with flair
                flair_info = f" ({flair})" if flair else ""
                comment_content = f"[Re: {title}{flair_info}]\n{comment_text}"

                comment_id = f"{post_id}_comment_{comment['id']}"

                # Skip if exists
                if self.kb.document_exists(comment_id):
                    continue

                result = self.kb.add_document(
                    content=comment_content,
                    post_id=comment_id,
                    post_type="comment",
                    metadata={
                        "parent_post_id": post_id,
                        "parent_title": title,
                        "parent_flair": flair,
                        "author": comment.get("author"),
                        "score": comment.get("score", 0),
                        "reddit_url": reddit_url  # Link to parent post
                    },
                    score=comment.get("score", 0)
                )
                if result:
                    added += 1

        return added

    def ingest_subreddit(
        self,
        subreddit: str = "TheTowerGame",
        max_posts: int = 500,
        timeframe: str = "all",
        sort: str = "top",
        min_score: int = 10,
        include_comments: bool = True,
        progress_callback=None
    ) -> Dict[str, int]:
        """
        Ingest all posts from a subreddit.

        Args:
            subreddit: Subreddit name
            max_posts: Maximum posts to fetch
            timeframe: 'day', 'week', 'month', 'year', 'all'
            sort: 'top', 'hot', 'new'
            min_score: Minimum post score
            include_comments: Whether to include comments
            progress_callback: Optional callback(current, total, message)

        Returns:
            Stats dict with counts
        """
        print(f"[Ingester] Starting ingestion of r/{subreddit}...")

        stats = {
            "posts_fetched": 0,
            "posts_ingested": 0,
            "documents_added": 0,
            "posts_skipped": 0
        }

        after = None
        fetched = 0

        while fetched < max_posts:
            batch_size = min(100, max_posts - fetched)
            posts, after = self.fetch_posts(
                subreddit=subreddit,
                limit=batch_size,
                timeframe=timeframe,
                sort=sort,
                after=after
            )

            if not posts:
                break

            stats["posts_fetched"] += len(posts)
            fetched += len(posts)

            for post in posts:
                if post.get("score", 0) < min_score:
                    stats["posts_skipped"] += 1
                    continue

                if progress_callback:
                    progress_callback(
                        fetched,
                        max_posts,
                        f"Ingesting: {post.get('title', '')[:50]}..."
                    )

                added = self.ingest_post(post, include_comments=include_comments)
                if added > 0:
                    stats["posts_ingested"] += 1
                    stats["documents_added"] += added
                    safe_title = post.get('title', '')[:50].encode('ascii', 'replace').decode('ascii')
                    print(f"[Ingester] Added {added} docs from: {safe_title}...")

            if not after:
                break

        print(f"[Ingester] Ingestion complete:")
        print(f"  Posts fetched: {stats['posts_fetched']}")
        print(f"  Posts ingested: {stats['posts_ingested']}")
        print(f"  Documents added: {stats['documents_added']}")
        print(f"  Posts skipped (low score): {stats['posts_skipped']}")

        return stats

    def ingest_guide(
        self,
        title: str,
        content: str,
        source: str = "manual",
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Manually ingest a guide or wiki content.
        Useful for adding curated game knowledge.
        """
        added = 0
        guide_id = f"guide_{hash(title)}"

        if self.kb.document_exists(guide_id):
            return 0

        clean_content = self._clean_text(content)
        chunks = self._chunk_text(clean_content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{guide_id}_chunk{i}" if len(chunks) > 1 else guide_id

            meta = metadata or {}
            meta.update({
                "title": title,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

            result = self.kb.add_document(
                content=f"[Guide: {title}]\n{chunk}",
                post_id=chunk_id,
                post_type="guide",
                metadata=meta,
                score=1000  # High score for guides
            )
            if result:
                added += 1

        return added
