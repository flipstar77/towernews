"""Reddit Scraper Tool - Fetches top posts from subreddits via JSON API."""

import requests
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import datetime


@dataclass
class RedditPost:
    """Represents a Reddit post."""
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
    thumbnail: Optional[str]
    image_url: Optional[str] = None  # Direct image URL (high-res)
    preview_url: Optional[str] = None  # Preview image URL

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score for ranking."""
        return self.score * 0.6 + self.num_comments * 0.4

    @property
    def full_url(self) -> str:
        """Get full Reddit URL."""
        return f"https://www.reddit.com{self.permalink}"


class RedditScraper:
    """Scrapes Reddit subreddits for top posts."""

    BASE_URL = "https://www.reddit.com"
    USER_AGENT = "TowerNews/1.0 (News Aggregator Bot)"

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    def fetch_top_posts(
        self,
        subreddit: str,
        timeframe: str = "day",
        limit: int = 25
    ) -> list[RedditPost]:
        """
        Fetch top posts from a subreddit.

        Args:
            subreddit: Name of the subreddit (without r/)
            timeframe: One of 'hour', 'day', 'week', 'month', 'year', 'all'
            limit: Maximum number of posts to fetch (max 100)

        Returns:
            List of RedditPost objects
        """
        url = f"{self.BASE_URL}/r/{subreddit}/top.json"
        params = {
            "t": timeframe,
            "limit": min(limit, 100)
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        posts = []

        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})

            # Extract image URL from various sources
            image_url = None
            preview_url = None

            # Check if URL is a direct image link
            url = post_data.get("url", "")
            if any(url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                image_url = url
            elif "i.redd.it" in url or "i.imgur.com" in url:
                image_url = url

            # Check preview images (Reddit's own preview system)
            preview = post_data.get("preview", {})
            if preview and "images" in preview:
                images = preview.get("images", [])
                if images:
                    # Get the source (highest quality)
                    source = images[0].get("source", {})
                    if source.get("url"):
                        # Reddit HTML-encodes the URL
                        preview_url = source.get("url", "").replace("&amp;", "&")

                    # If no direct image, use preview as fallback
                    if not image_url and preview_url:
                        image_url = preview_url

            # Check for gallery posts (multiple images)
            if post_data.get("is_gallery"):
                media_metadata = post_data.get("media_metadata", {})
                if media_metadata:
                    # Get first image from gallery
                    for media_id, media_data in media_metadata.items():
                        if media_data.get("status") == "valid" and media_data.get("m", "").startswith("image"):
                            # Get highest resolution
                            if "s" in media_data and "u" in media_data["s"]:
                                image_url = media_data["s"]["u"].replace("&amp;", "&")
                                break

            post = RedditPost(
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
                thumbnail=post_data.get("thumbnail") if post_data.get("thumbnail", "").startswith("http") else None,
                image_url=image_url,
                preview_url=preview_url
            )
            posts.append(post)

        return posts

    def fetch_top_comments(self, subreddit: str, post_id: str, limit: int = 5) -> list[dict]:
        """
        Fetch top comments from a post.

        Args:
            subreddit: Name of the subreddit
            post_id: ID of the post
            limit: Maximum number of comments to fetch

        Returns:
            List of comment dictionaries with 'body', 'score', 'author'
        """
        url = f"{self.BASE_URL}/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": limit, "sort": "top"}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        comments = []

        if len(data) > 1:
            comment_data = data[1].get("data", {}).get("children", [])

            for child in comment_data[:limit]:
                if child.get("kind") == "t1":
                    comment = child.get("data", {})
                    comments.append({
                        "body": comment.get("body", ""),
                        "score": comment.get("score", 0),
                        "author": comment.get("author", "[deleted]")
                    })

        return comments

    def download_image(self, url: str, post_id: str, output_dir: Path = None) -> Optional[str]:
        """
        Download an image from a URL.

        Args:
            url: URL of the image
            post_id: ID of the post (for filename)
            output_dir: Directory to save image (optional)

        Returns:
            Path to saved image, or None if failed
        """
        if not url:
            return None

        try:
            # Create output directory
            if output_dir is None:
                today = datetime.now().strftime("%Y-%m-%d")
                output_dir = Path("output") / today / "images"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine file extension
            ext = ".jpg"  # Default
            if ".png" in url.lower():
                ext = ".png"
            elif ".gif" in url.lower():
                ext = ".gif"
            elif ".webp" in url.lower():
                ext = ".webp"

            output_path = output_dir / f"post_{post_id}{ext}"

            # Download image
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"[RedditScraper] Downloaded image: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"[RedditScraper] Failed to download image {url}: {e}")
            return None

    def download_post_images(self, posts: list[RedditPost]) -> dict[str, str]:
        """
        Download images for multiple posts.

        Args:
            posts: List of RedditPost objects

        Returns:
            Dictionary mapping post_id to local image path
        """
        images = {}
        for post in posts:
            if post.image_url:
                local_path = self.download_image(post.image_url, post.id)
                if local_path:
                    images[post.id] = local_path
        return images

    # Flairs to exclude from news (memes, achievements, help requests)
    EXCLUDED_FLAIRS = [
        "meme", "memes", "humor", "funny",
        "achievement", "achievements", "milestone", "flex",
        "help", "question", "questions", "support", "bug",
        "lfg", "looking for group", "recruitment"
    ]

    # Flairs to prioritize for news content
    PRIORITIZED_FLAIRS = [
        "news", "announcement", "update", "patch",
        "discussion", "info", "information", "guide",
        "strategy", "tip", "tips", "meta"
    ]

    def get_filtered_posts(
        self,
        subreddit: str,
        min_upvotes: int = 10,
        min_comments: int = 5,
        max_posts: int = 5,
        timeframe: str = "day",
        exclude_flairs: list[str] = None,
        prioritize_flairs: list[str] = None
    ) -> list[RedditPost]:
        """
        Get filtered and sorted posts from a subreddit.

        Args:
            subreddit: Name of the subreddit
            min_upvotes: Minimum upvote threshold
            min_comments: Minimum comment threshold
            max_posts: Maximum number of posts to return
            timeframe: Time period for top posts
            exclude_flairs: List of flairs to exclude (default: memes, achievements, help)
            prioritize_flairs: List of flairs to prioritize (default: news, discussion, info)

        Returns:
            List of filtered RedditPost objects, sorted by engagement
        """
        posts = self.fetch_top_posts(subreddit, timeframe, limit=50)

        # Use default exclusions if not specified
        exclude_flairs = exclude_flairs or self.EXCLUDED_FLAIRS
        prioritize_flairs = prioritize_flairs or self.PRIORITIZED_FLAIRS

        # Filter by thresholds and exclude unwanted flairs
        filtered = []
        for p in posts:
            # Check minimum thresholds
            if p.score < min_upvotes or p.num_comments < min_comments:
                continue

            # Check if flair should be excluded
            post_flair = (p.flair or "").lower()
            if any(excluded.lower() in post_flair for excluded in exclude_flairs):
                print(f"[RedditScraper] Excluding post (flair: {p.flair}): {p.title[:50]}...")
                continue

            filtered.append(p)

        # Sort: prioritized flairs first, then by engagement score
        def sort_key(post):
            post_flair = (post.flair or "").lower()
            is_prioritized = any(pf.lower() in post_flair for pf in prioritize_flairs)
            # Prioritized posts get a boost, then sort by engagement
            return (is_prioritized, post.engagement_score)

        filtered.sort(key=sort_key, reverse=True)

        return filtered[:max_posts]

    def run(self, **kwargs) -> dict:
        """
        Tool interface for agents.

        Args:
            subreddit: Name of the subreddit (optional, uses config default)
            min_upvotes: Minimum upvotes (optional)
            min_comments: Minimum comments (optional)
            max_posts: Maximum posts to return (optional)
            download_images: Whether to download images (default True)
            fetch_comments: Whether to fetch top comments for each post (default True)
            comments_limit: Number of comments to fetch per post (default 5)

        Returns:
            Dictionary with 'posts' list and 'images' dict
        """
        reddit_config = self.config.get("sources", {}).get("reddit", {})

        subreddit = kwargs.get("subreddit", reddit_config.get("subreddits", ["TheTowerGame"])[0])
        min_upvotes = kwargs.get("min_upvotes", reddit_config.get("min_upvotes", 10))
        min_comments = kwargs.get("min_comments", reddit_config.get("min_comments", 5))
        max_posts = kwargs.get("max_posts", reddit_config.get("max_posts", 5))
        timeframe = kwargs.get("timeframe", reddit_config.get("timeframe", "day"))
        download_images = kwargs.get("download_images", True)
        fetch_comments = kwargs.get("fetch_comments", True)
        comments_limit = kwargs.get("comments_limit", 5)

        posts = self.get_filtered_posts(
            subreddit=subreddit,
            min_upvotes=min_upvotes,
            min_comments=min_comments,
            max_posts=max_posts,
            timeframe=timeframe
        )

        # Download images if requested
        images = {}
        if download_images:
            images = self.download_post_images(posts)

        # Fetch top comments for each post if requested
        post_comments = {}
        if fetch_comments:
            for p in posts:
                try:
                    comments = self.fetch_top_comments(subreddit, p.id, limit=comments_limit)
                    post_comments[p.id] = comments
                    print(f"[RedditScraper] Fetched {len(comments)} comments for post {p.id}")
                except Exception as e:
                    print(f"[RedditScraper] Failed to fetch comments for {p.id}: {e}")
                    post_comments[p.id] = []

        # Convert to serializable format
        return {
            "posts": [
                {
                    "id": p.id,
                    "title": p.title,
                    "selftext": p.selftext,
                    "url": p.full_url,
                    "score": p.score,
                    "num_comments": p.num_comments,
                    "engagement_score": p.engagement_score,
                    "author": p.author,
                    "flair": p.flair,
                    "is_self": p.is_self,
                    "image_url": p.image_url,
                    "local_image": images.get(p.id),  # Local path if downloaded
                    "top_comments": post_comments.get(p.id, [])  # Top comments
                }
                for p in posts
            ],
            "subreddit": subreddit,
            "count": len(posts),
            "images_downloaded": len(images)
        }
