"""History Tracker - Prevents reporting on the same Reddit posts twice."""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


class HistoryTracker:
    """Tracks which Reddit posts have been reported on to avoid duplicates."""

    def __init__(self, config: dict, history_file: str = "data/post_history.json"):
        """
        Initialize the History Tracker.

        Args:
            config: Configuration dictionary
            history_file: Path to JSON file storing post history
        """
        self.config = config
        self.history_path = Path(history_file)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        # How many days to remember posts (avoid reporting same topic again)
        self.retention_days = config.get("history", {}).get("retention_days", 30)

        # Load existing history
        self.history = self._load_history()

    def _load_history(self) -> dict:
        """Load history from JSON file."""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[HistoryTracker] Could not load history: {e}")
                return {"posts": {}}
        return {"posts": {}}

    def _save_history(self) -> None:
        """Save history to JSON file."""
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[HistoryTracker] Could not save history: {e}")

    def _cleanup_old_entries(self) -> None:
        """Remove entries older than retention_days."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        cutoff_str = cutoff.isoformat()

        posts = self.history.get("posts", {})
        old_count = len(posts)

        # Filter out old entries
        self.history["posts"] = {
            post_id: data
            for post_id, data in posts.items()
            if data.get("reported_at", "") > cutoff_str
        }

        removed = old_count - len(self.history["posts"])
        if removed > 0:
            print(f"[HistoryTracker] Cleaned up {removed} old entries")
            self._save_history()

    def is_reported(self, post_id: str) -> bool:
        """
        Check if a post has already been reported.

        Args:
            post_id: Reddit post ID

        Returns:
            True if already reported, False otherwise
        """
        return post_id in self.history.get("posts", {})

    def mark_reported(self, post_id: str, title: str, url: str = "") -> None:
        """
        Mark a post as reported.

        Args:
            post_id: Reddit post ID
            title: Post title (for reference)
            url: Post URL (optional)
        """
        if "posts" not in self.history:
            self.history["posts"] = {}

        self.history["posts"][post_id] = {
            "title": title,
            "url": url,
            "reported_at": datetime.now().isoformat()
        }
        self._save_history()
        print(f"[HistoryTracker] Marked as reported: {post_id} - {title[:50]}...")

    def filter_unreported(self, posts: list[dict]) -> list[dict]:
        """
        Filter a list of posts to only include unreported ones.

        Args:
            posts: List of post dictionaries with 'id' field

        Returns:
            List of posts that haven't been reported yet
        """
        # First, cleanup old entries
        self._cleanup_old_entries()

        unreported = []
        for post in posts:
            post_id = post.get("id")
            if not post_id:
                continue

            if self.is_reported(post_id):
                print(f"[HistoryTracker] Skipping (already reported): {post.get('title', 'N/A')[:50]}...")
            else:
                unreported.append(post)

        skipped = len(posts) - len(unreported)
        if skipped > 0:
            print(f"[HistoryTracker] Filtered out {skipped} already-reported posts")

        return unreported

    def mark_all_reported(self, posts: list[dict]) -> None:
        """
        Mark multiple posts as reported.

        Args:
            posts: List of post dictionaries with 'id', 'title', 'url' fields
        """
        for post in posts:
            post_id = post.get("id")
            if post_id:
                self.mark_reported(
                    post_id=post_id,
                    title=post.get("title", post.get("post_title", "")),
                    url=post.get("url", "")
                )

    def get_history_stats(self) -> dict:
        """
        Get statistics about the history.

        Returns:
            Dictionary with stats
        """
        posts = self.history.get("posts", {})
        return {
            "total_reported": len(posts),
            "retention_days": self.retention_days,
            "history_file": str(self.history_path)
        }

    def run(self, **kwargs) -> dict:
        """Tool interface for pipeline."""
        action = kwargs.get("action", "filter")

        if action == "filter":
            posts = kwargs.get("posts", [])
            filtered = self.filter_unreported(posts)
            return {"posts": filtered, "filtered_count": len(posts) - len(filtered)}

        elif action == "mark":
            posts = kwargs.get("posts", [])
            self.mark_all_reported(posts)
            return {"marked_count": len(posts)}

        elif action == "stats":
            return self.get_history_stats()

        elif action == "check":
            post_id = kwargs.get("post_id")
            return {"is_reported": self.is_reported(post_id)}

        return {"error": f"Unknown action: {action}"}
