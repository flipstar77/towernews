"""Bulk Reddit Scraper - Scrapes thousands of posts with pagination."""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict


@dataclass
class ScrapedPost:
    """A scraped Reddit post with all relevant data."""
    id: str
    title: str
    selftext: str
    url: str
    permalink: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    author: str
    flair: Optional[str]
    is_self: bool
    comments: List[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)


class BulkRedditScraper:
    """Scrapes large amounts of Reddit posts using pagination."""

    BASE_URL = "https://www.reddit.com"
    USER_AGENT = "TowerAI/1.0 (Training Data Collector)"

    # Rate limiting settings
    REQUESTS_PER_MINUTE = 30
    MIN_DELAY = 2.0  # Minimum seconds between requests

    def __init__(self, cache_dir: str = "data/ai/reddit_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting to avoid getting blocked."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_DELAY:
            time.sleep(self.MIN_DELAY - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: dict = None) -> Optional[dict]:
        """Make a rate-limited request."""
        self._rate_limit()
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[BulkScraper] Request failed: {e}")
            return None

    def scrape_subreddit(
        self,
        subreddit: str,
        target_count: int = 5000,
        sort: str = "top",
        timeframe: str = "all",
        include_comments: bool = True,
        comments_per_post: int = 10,
        progress_callback: Callable[[str, int, int], None] = None
    ) -> List[ScrapedPost]:
        """
        Scrape posts from a subreddit with pagination.

        Args:
            subreddit: Subreddit name (without r/)
            target_count: Target number of posts to scrape
            sort: Sort method ('top', 'hot', 'new', 'controversial')
            timeframe: Time filter for 'top' ('hour', 'day', 'week', 'month', 'year', 'all')
            include_comments: Whether to fetch comments for each post
            comments_per_post: Number of top comments to fetch per post
            progress_callback: Callback function(status, current, total)

        Returns:
            List of ScrapedPost objects
        """
        posts = []
        after = None
        batch_num = 0

        print(f"[BulkScraper] Starting scrape of r/{subreddit}")
        print(f"[BulkScraper] Target: {target_count} posts, sort: {sort}, timeframe: {timeframe}")

        while len(posts) < target_count:
            batch_num += 1

            # Build URL and params
            url = f"{self.BASE_URL}/r/{subreddit}/{sort}.json"
            params = {"limit": 100, "raw_json": 1}

            if sort == "top":
                params["t"] = timeframe
            if after:
                params["after"] = after

            if progress_callback:
                progress_callback(f"Fetching batch {batch_num}...", len(posts), target_count)

            # Make request
            data = self._make_request(url, params)

            if not data:
                print(f"[BulkScraper] Request failed, stopping at {len(posts)} posts")
                break

            # Extract posts from response
            children = data.get("data", {}).get("children", [])
            if not children:
                print(f"[BulkScraper] No more posts available, got {len(posts)} total")
                break

            # Process each post
            for child in children:
                if len(posts) >= target_count:
                    break

                post_data = child.get("data", {})

                post = ScrapedPost(
                    id=post_data.get("id", ""),
                    title=post_data.get("title", ""),
                    selftext=post_data.get("selftext", ""),
                    url=post_data.get("url", ""),
                    permalink=post_data.get("permalink", ""),
                    score=post_data.get("score", 0),
                    upvote_ratio=post_data.get("upvote_ratio", 0.0),
                    num_comments=post_data.get("num_comments", 0),
                    created_utc=post_data.get("created_utc", 0),
                    author=post_data.get("author", "[deleted]"),
                    flair=post_data.get("link_flair_text"),
                    is_self=post_data.get("is_self", False),
                    comments=[]
                )

                posts.append(post)

            # Get pagination token for next batch
            after = data.get("data", {}).get("after")
            if not after:
                print(f"[BulkScraper] Reached end of subreddit, got {len(posts)} posts")
                break

            print(f"[BulkScraper] Batch {batch_num}: {len(posts)}/{target_count} posts")

        # Fetch comments if requested
        if include_comments and posts:
            print(f"[BulkScraper] Fetching comments for {len(posts)} posts...")
            self._fetch_comments_bulk(
                subreddit, posts, comments_per_post, progress_callback
            )

        return posts

    def _fetch_comments_bulk(
        self,
        subreddit: str,
        posts: List[ScrapedPost],
        comments_per_post: int,
        progress_callback: Callable[[str, int, int], None] = None
    ):
        """Fetch comments for multiple posts."""
        for i, post in enumerate(posts):
            if progress_callback and i % 50 == 0:
                progress_callback(f"Fetching comments...", i, len(posts))

            try:
                comments = self._fetch_post_comments(subreddit, post.id, comments_per_post)
                post.comments = comments
            except Exception as e:
                print(f"[BulkScraper] Failed to fetch comments for {post.id}: {e}")
                post.comments = []

            # Progress update every 100 posts
            if (i + 1) % 100 == 0:
                print(f"[BulkScraper] Comments: {i + 1}/{len(posts)}")

    def _fetch_post_comments(
        self,
        subreddit: str,
        post_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch top comments from a single post."""
        url = f"{self.BASE_URL}/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": limit, "sort": "top", "raw_json": 1}

        data = self._make_request(url, params)

        if not data or len(data) < 2:
            return []

        comments = []
        comment_data = data[1].get("data", {}).get("children", [])

        for child in comment_data[:limit]:
            if child.get("kind") == "t1":
                comment = child.get("data", {})
                comments.append({
                    "id": comment.get("id", ""),
                    "body": comment.get("body", ""),
                    "score": comment.get("score", 0),
                    "author": comment.get("author", "[deleted]"),
                    "created_utc": comment.get("created_utc", 0)
                })

        return comments

    def save_to_cache(self, posts: List[ScrapedPost], filename: str = None) -> str:
        """Save scraped posts to JSON cache."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_scrape_{timestamp}.json"

        filepath = self.cache_dir / filename

        data = {
            "scraped_at": datetime.now().isoformat(),
            "post_count": len(posts),
            "posts": [post.to_dict() for post in posts]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[BulkScraper] Saved {len(posts)} posts to {filepath}")
        return str(filepath)

    def load_from_cache(self, filename: str) -> List[ScrapedPost]:
        """Load posts from JSON cache."""
        filepath = self.cache_dir / filename

        if not filepath.exists():
            print(f"[BulkScraper] Cache file not found: {filepath}")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        posts = []
        for post_data in data.get("posts", []):
            post = ScrapedPost(
                id=post_data.get("id", ""),
                title=post_data.get("title", ""),
                selftext=post_data.get("selftext", ""),
                url=post_data.get("url", ""),
                permalink=post_data.get("permalink", ""),
                score=post_data.get("score", 0),
                upvote_ratio=post_data.get("upvote_ratio", 0.0),
                num_comments=post_data.get("num_comments", 0),
                created_utc=post_data.get("created_utc", 0),
                author=post_data.get("author", "[deleted]"),
                flair=post_data.get("flair"),
                is_self=post_data.get("is_self", False),
                comments=post_data.get("comments", [])
            )
            posts.append(post)

        print(f"[BulkScraper] Loaded {len(posts)} posts from {filepath}")
        return posts

    def list_cache_files(self) -> List[Dict[str, Any]]:
        """List all cached scrape files."""
        files = []
        for filepath in self.cache_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                files.append({
                    "filename": filepath.name,
                    "path": str(filepath),
                    "post_count": data.get("post_count", 0),
                    "scraped_at": data.get("scraped_at", "unknown")
                })
            except:
                pass
        return sorted(files, key=lambda x: x.get("scraped_at", ""), reverse=True)

    def get_stats(self, posts: List[ScrapedPost]) -> Dict[str, Any]:
        """Get statistics about scraped posts."""
        if not posts:
            return {"total_posts": 0}

        total_score = sum(p.score for p in posts)
        total_comments = sum(p.num_comments for p in posts)
        total_comment_texts = sum(len(p.comments or []) for p in posts)

        # Flair distribution
        flair_counts = {}
        for p in posts:
            flair = p.flair or "No Flair"
            flair_counts[flair] = flair_counts.get(flair, 0) + 1

        # Top flairs
        top_flairs = sorted(flair_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Content stats
        text_posts = sum(1 for p in posts if p.is_self and p.selftext)
        link_posts = sum(1 for p in posts if not p.is_self)

        return {
            "total_posts": len(posts),
            "total_score": total_score,
            "avg_score": total_score / len(posts),
            "total_comments": total_comments,
            "avg_comments": total_comments / len(posts),
            "fetched_comments": total_comment_texts,
            "text_posts": text_posts,
            "link_posts": link_posts,
            "top_flairs": top_flairs
        }
