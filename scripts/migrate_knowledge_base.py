"""
Migration script: Migrate Journey KB data to Unified Knowledge Base.

This script:
1. Reads all chunks from the old Journey KB (data/journey/knowledge_base/)
2. Transforms them to the new format
3. Imports them into the Unified KB (Supabase or local)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()


def load_journey_chunks() -> list:
    """Load chunks from the old Journey KB."""
    chunks_file = Path("data/journey/knowledge_base/chunks.json")

    if not chunks_file.exists():
        print("[Migration] No Journey KB chunks found")
        return []

    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chunks = data.get("chunks", [])
            print(f"[Migration] Found {len(chunks)} chunks in Journey KB")
            return chunks
    except Exception as e:
        print(f"[Migration] Error loading chunks: {e}")
        return []


def transform_chunk(old_chunk: dict) -> dict:
    """Transform old Journey KB chunk to new format."""
    return {
        "content": old_chunk.get("content", ""),
        "source": old_chunk.get("source", "journey_migration"),
        "doc_type": map_doc_type(old_chunk.get("type", "general")),
        "metadata": {
            **old_chunk.get("metadata", {}),
            "original_id": old_chunk.get("id"),
            "migrated_from": "journey_kb",
            "original_type": old_chunk.get("type")
        },
        "score": calculate_score(old_chunk)
    }


def map_doc_type(old_type: str) -> str:
    """Map old type names to new standardized types."""
    type_mapping = {
        "tips": "guide",
        "mechanics": "wiki",
        "strategy": "guide",
        "overview": "post",
        "qa": "post",
        "insights": "guide",
        "transcript": "post",
        "reddit_post": "post",
        "reddit_comment": "comment",
        "wiki": "wiki",
        "general": "post"
    }
    return type_mapping.get(old_type, "post")


def calculate_score(chunk: dict) -> int:
    """Calculate relevance score for migrated chunk."""
    base_score = 50  # Base score for migrated content

    # Boost based on type
    type_boosts = {
        "tips": 100,
        "mechanics": 80,
        "strategy": 90,
        "wiki": 70,
        "guide": 100
    }

    chunk_type = chunk.get("type", "general")
    base_score += type_boosts.get(chunk_type, 0)

    # Boost if has metadata score
    if "score" in chunk.get("metadata", {}):
        base_score += min(chunk["metadata"]["score"], 200)

    return base_score


def migrate(dry_run: bool = False, force: bool = False):
    """
    Run the migration.

    Args:
        dry_run: If True, only show what would be migrated
        force: If True, re-import even if chunks exist
    """
    from rag.unified_knowledge_base import UnifiedKnowledgeBase

    print("=" * 60)
    print("Journey KB to Unified KB Migration")
    print("=" * 60)

    # Load old chunks
    old_chunks = load_journey_chunks()

    if not old_chunks:
        print("[Migration] No chunks to migrate")
        return

    # Initialize unified KB
    kb = UnifiedKnowledgeBase()

    # Get current stats
    stats = kb.get_stats()
    print(f"\n[Migration] Current Unified KB: {stats.get('total_documents', 0)} documents")

    if dry_run:
        print("\n[DRY RUN] Would migrate the following:")
        for i, chunk in enumerate(old_chunks[:5]):
            transformed = transform_chunk(chunk)
            print(f"\n  Chunk {i+1}:")
            print(f"    Source: {transformed['source']}")
            print(f"    Type: {transformed['doc_type']}")
            print(f"    Score: {transformed['score']}")
            print(f"    Content: {transformed['content'][:100]}...")

        if len(old_chunks) > 5:
            print(f"\n  ... and {len(old_chunks) - 5} more chunks")

        return

    # Migrate chunks
    print(f"\n[Migration] Migrating {len(old_chunks)} chunks...")

    migrated = 0
    skipped = 0
    errors = 0

    for i, old_chunk in enumerate(old_chunks):
        try:
            transformed = transform_chunk(old_chunk)

            # Check if already exists (by original ID)
            original_id = old_chunk.get("id", "")
            if not force and original_id and kb.document_exists(original_id):
                skipped += 1
                continue

            # Add to unified KB (without re-chunking since already chunked)
            result = kb.add_document(
                content=transformed["content"],
                source=transformed["source"],
                doc_type=transformed["doc_type"],
                metadata=transformed["metadata"],
                score=transformed["score"],
                chunk=False  # Don't re-chunk
            )

            if result:
                migrated += 1
            else:
                skipped += 1

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"[Migration] Progress: {i + 1}/{len(old_chunks)} "
                      f"(migrated: {migrated}, skipped: {skipped})")

        except Exception as e:
            print(f"[Migration] Error migrating chunk {i}: {e}")
            errors += 1

    # Final stats
    print("\n" + "=" * 60)
    print("Migration Complete")
    print("=" * 60)
    print(f"  Total chunks processed: {len(old_chunks)}")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")

    final_stats = kb.get_stats()
    print(f"\n  Unified KB now has: {final_stats.get('total_documents', 0)} documents")
    print(f"  By type: {final_stats.get('by_type', {})}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate Journey KB to Unified KB")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--force", action="store_true", help="Re-import existing chunks")

    args = parser.parse_args()
    migrate(dry_run=args.dry_run, force=args.force)
