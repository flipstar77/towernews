"""
Daily Sync Script - Incremental updates for the RAG knowledge base.

Features:
- Fetches only new/updated content since last sync
- Prunes low-quality or deleted content
- Updates embeddings for modified documents
- Can be run via cron or GitHub Actions

Usage:
    python scripts/daily_sync.py                 # Full sync
    python scripts/daily_sync.py --reddit-only   # Only Reddit
    python scripts/daily_sync.py --dry-run       # Preview changes
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()


class SyncState:
    """Tracks sync state between runs."""

    def __init__(self, state_file: str = "data/sync_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load sync state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "last_reddit_sync": None,
            "last_wiki_sync": None,
            "last_prune": None,
            "total_syncs": 0,
            "sync_history": []
        }

    def save(self):
        """Save sync state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_last_sync(self, source: str) -> Optional[datetime]:
        """Get last sync time for a source."""
        key = f"last_{source}_sync"
        if key in self.state and self.state[key]:
            return datetime.fromisoformat(self.state[key])
        return None

    def update_sync(self, source: str, stats: dict):
        """Update sync state after successful sync."""
        now = datetime.now().isoformat()
        self.state[f"last_{source}_sync"] = now
        self.state["total_syncs"] += 1

        # Keep last 30 sync records
        self.state["sync_history"].append({
            "source": source,
            "timestamp": now,
            "stats": stats
        })
        self.state["sync_history"] = self.state["sync_history"][-30:]

        self.save()


class IncrementalSyncer:
    """
    Handles incremental updates to the knowledge base.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.state = SyncState()

        # Import after path setup
        from rag import UnifiedKnowledgeBase, KnowledgeIngester

        self.kb = UnifiedKnowledgeBase()
        self.ingester = KnowledgeIngester()

    def sync_reddit(
        self,
        max_posts: int = 100,
        min_score: int = 5
    ) -> Dict[str, Any]:
        """
        Sync new Reddit posts since last sync.

        Args:
            max_posts: Maximum posts to fetch
            min_score: Minimum post score

        Returns:
            Sync statistics
        """
        print("\n" + "=" * 60)
        print("Reddit Incremental Sync")
        print("=" * 60)

        last_sync = self.state.get_last_sync("reddit")
        if last_sync:
            print(f"Last sync: {last_sync.isoformat()}")
            hours_since = (datetime.now() - last_sync).total_seconds() / 3600
            print(f"Hours since last sync: {hours_since:.1f}")
        else:
            print("First sync - fetching recent posts")

        stats = {
            "posts_checked": 0,
            "posts_added": 0,
            "comments_added": 0,
            "skipped_existing": 0,
            "skipped_low_score": 0
        }

        if self.dry_run:
            print("[DRY RUN] Would sync Reddit posts")
            return stats

        # Fetch new and hot posts
        for sort in ["new", "hot"]:
            print(f"\n[Fetching {sort} posts...]")

            posts, _ = self.ingester.fetch_posts(
                subreddit="TheTowerGame",
                limit=max_posts,
                sort=sort,
                timeframe="week" if sort == "hot" else "all"
            )

            stats["posts_checked"] += len(posts)

            for post in posts:
                # Skip low score posts
                if post.get("score", 0) < min_score:
                    stats["skipped_low_score"] += 1
                    continue

                # Check if already exists
                post_id = post["id"]
                if self.kb.document_exists(post_id):
                    stats["skipped_existing"] += 1
                    continue

                # Ingest post
                added = self.ingester.ingest_post(post, include_comments=True)
                if added > 0:
                    stats["posts_added"] += 1
                    stats["comments_added"] += added - 1  # -1 for the post itself

                    title = post.get('title', '')[:40]
                    print(f"  + {title}... ({added} docs)")

        # Update state
        self.state.update_sync("reddit", stats)

        print(f"\nReddit sync complete:")
        print(f"  Posts checked: {stats['posts_checked']}")
        print(f"  New posts: {stats['posts_added']}")
        print(f"  New comments: {stats['comments_added']}")

        return stats

    def sync_wiki(self, max_pages: int = 50) -> Dict[str, Any]:
        """
        Sync updated wiki pages.

        Args:
            max_pages: Maximum pages to check

        Returns:
            Sync statistics
        """
        print("\n" + "=" * 60)
        print("Wiki Incremental Sync")
        print("=" * 60)

        from rag import WikiScraper
        scraper = WikiScraper()

        last_sync = self.state.get_last_sync("wiki")
        if last_sync:
            print(f"Last sync: {last_sync.isoformat()}")
        else:
            print("First wiki sync")

        stats = {
            "pages_checked": 0,
            "pages_added": 0,
            "pages_updated": 0,
            "skipped_existing": 0
        }

        if self.dry_run:
            print("[DRY RUN] Would sync wiki pages")
            return stats

        # Get all pages
        pages = scraper.get_all_pages()[:max_pages]
        stats["pages_checked"] = len(pages)

        for page_path in pages:
            # Create doc ID from path
            doc_id = f"wiki_{page_path.replace('/', '_')}"

            # Check if exists
            if self.kb.document_exists(doc_id):
                stats["skipped_existing"] += 1
                continue

            # Scrape and ingest
            page_data = scraper.scrape_page(page_path)
            if page_data:
                added = scraper.ingest_page(page_data)
                if added > 0:
                    stats["pages_added"] += 1
                    title = page_data.get('title', '')[:40]
                    print(f"  + {title}...")

            time.sleep(0.5)  # Rate limiting

        # Update state
        self.state.update_sync("wiki", stats)

        print(f"\nWiki sync complete:")
        print(f"  Pages checked: {stats['pages_checked']}")
        print(f"  New pages: {stats['pages_added']}")

        return stats

    def prune_low_quality(
        self,
        max_age_days: int = 90,
        min_score: int = 0
    ) -> Dict[str, Any]:
        """
        Remove low-quality documents from the knowledge base.

        Criteria for removal:
        - Score < min_score AND older than max_age_days
        - Content from deleted users with short content

        Args:
            max_age_days: Age threshold in days
            min_score: Minimum score threshold

        Returns:
            Pruning statistics
        """
        print("\n" + "=" * 60)
        print("Pruning Low-Quality Content")
        print("=" * 60)

        stats = {
            "checked": 0,
            "pruned": 0,
            "reasons": {}
        }

        if self.dry_run:
            print("[DRY RUN] Would prune low-quality content")
            return stats

        # Note: Actual pruning requires direct database access
        # This is a placeholder for the pruning logic

        print("Pruning not yet implemented for Supabase backend")
        print("Run SQL directly for now:")
        print(f"""
DELETE FROM tower_knowledge
WHERE
    score < {min_score}
    AND ingested_at < NOW() - INTERVAL '{max_age_days} days'
    AND post_type = 'comment';
        """)

        self.state.state["last_prune"] = datetime.now().isoformat()
        self.state.save()

        return stats

    def rebuild_embeddings(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Rebuild embeddings for documents missing them.

        Returns:
            Rebuild statistics
        """
        print("\n" + "=" * 60)
        print("Rebuilding Missing Embeddings")
        print("=" * 60)

        stats = {
            "checked": 0,
            "rebuilt": 0
        }

        if self.dry_run:
            print("[DRY RUN] Would rebuild embeddings")
            return stats

        # For local backend
        if hasattr(self.kb, 'build_local_embeddings'):
            rebuilt = self.kb.build_local_embeddings()
            stats["rebuilt"] = rebuilt

        print(f"Rebuilt {stats['rebuilt']} embeddings")
        return stats

    def full_sync(self) -> Dict[str, Any]:
        """
        Run full incremental sync.

        Returns:
            Combined statistics
        """
        print("=" * 70)
        print("FULL INCREMENTAL SYNC")
        print(f"Started at: {datetime.now().isoformat()}")
        print("=" * 70)

        all_stats = {}

        # Sync Reddit
        all_stats["reddit"] = self.sync_reddit()

        # Sync Wiki (less frequently)
        last_wiki = self.state.get_last_sync("wiki")
        if not last_wiki or (datetime.now() - last_wiki).days >= 7:
            all_stats["wiki"] = self.sync_wiki()
        else:
            print("\n[Skipping wiki sync - less than 7 days since last sync]")

        # Prune (monthly)
        last_prune = self.state.state.get("last_prune")
        if last_prune:
            last_prune = datetime.fromisoformat(last_prune)
        if not last_prune or (datetime.now() - last_prune).days >= 30:
            all_stats["prune"] = self.prune_low_quality()

        # Summary
        print("\n" + "=" * 70)
        print("SYNC COMPLETE")
        print("=" * 70)

        kb_stats = self.kb.get_stats()
        print(f"Knowledge Base: {kb_stats.get('total_documents', 0)} documents")
        print(f"By type: {kb_stats.get('by_type', {})}")

        return all_stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daily RAG Knowledge Base Sync")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--reddit-only", action="store_true", help="Only sync Reddit")
    parser.add_argument("--wiki-only", action="store_true", help="Only sync Wiki")
    parser.add_argument("--prune", action="store_true", help="Only run pruning")
    parser.add_argument("--rebuild-embeddings", action="store_true", help="Rebuild missing embeddings")

    args = parser.parse_args()

    syncer = IncrementalSyncer(dry_run=args.dry_run)

    if args.reddit_only:
        syncer.sync_reddit()
    elif args.wiki_only:
        syncer.sync_wiki()
    elif args.prune:
        syncer.prune_low_quality()
    elif args.rebuild_embeddings:
        syncer.rebuild_embeddings()
    else:
        syncer.full_sync()


if __name__ == "__main__":
    main()
