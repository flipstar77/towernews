"""Training Data Extractor - Extract structured training data from Reddit posts."""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .bulk_scraper import ScrapedPost


class TrainingDataExtractor:
    """Extracts structured training data from Reddit posts for AI training."""

    # Categories of game knowledge
    KNOWLEDGE_CATEGORIES = {
        "progression": "How to progress through the game (tiers, waves, unlocks)",
        "workshop": "Workshop upgrades, priorities, and strategies",
        "cards": "Card collection, leveling, and usage strategies",
        "ultimate_weapons": "Ultimate Weapons (UW) strategies and priorities",
        "labs": "Laboratory research priorities and strategies",
        "perks": "Perk selection and strategies",
        "tournament": "Tournament strategies and tips",
        "coins_farming": "Coin earning and farming strategies",
        "cells_farming": "Cell earning strategies",
        "modules": "Module priorities and strategies",
        "death_wave": "Death wave (DW) strategies",
        "afk_strategies": "AFK and idle play strategies",
        "active_strategies": "Active play strategies",
        "early_game": "Early game tips (Tier 1-8)",
        "mid_game": "Mid game tips (Tier 9-14)",
        "late_game": "Late game tips (Tier 15+)",
        "meta": "Current meta strategies and tier lists",
        "updates": "Game updates and patch notes analysis",
        "general_tips": "General tips and tricks"
    }

    def __init__(self, cache_dir: str = "data/ai/training_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI()

    def extract_from_posts(
        self,
        posts: List[ScrapedPost],
        min_score: int = 5,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Extract training data from a list of posts.

        Args:
            posts: List of ScrapedPost objects
            min_score: Minimum score for posts to include
            progress_callback: Callback function(status, current, total)

        Returns:
            Dictionary with extracted training data
        """
        # Filter by minimum score
        quality_posts = [p for p in posts if p.score >= min_score]
        print(f"[TrainingData] Processing {len(quality_posts)}/{len(posts)} posts (min score: {min_score})")

        # Extract text content
        all_content = []

        for i, post in enumerate(quality_posts):
            if progress_callback and i % 100 == 0:
                progress_callback(f"Extracting content...", i, len(quality_posts))

            content = self._extract_post_content(post)
            if content:
                all_content.append(content)

        print(f"[TrainingData] Extracted {len(all_content)} content items")

        return {
            "extracted_at": datetime.now().isoformat(),
            "source_posts": len(posts),
            "quality_posts": len(quality_posts),
            "content_items": len(all_content),
            "content": all_content
        }

    def _extract_post_content(self, post: ScrapedPost) -> Optional[Dict[str, Any]]:
        """Extract structured content from a single post."""
        # Combine title and body
        text = post.title
        if post.selftext:
            text += "\n\n" + post.selftext

        # Skip if too short
        if len(text) < 50:
            return None

        # Extract comment text
        comment_texts = []
        if post.comments:
            for comment in post.comments:
                body = comment.get("body", "")
                score = comment.get("score", 0)
                if body and len(body) > 20 and score > 0:
                    comment_texts.append({
                        "text": body,
                        "score": score
                    })

        return {
            "post_id": post.id,
            "title": post.title,
            "body": post.selftext,
            "score": post.score,
            "flair": post.flair,
            "created_utc": post.created_utc,
            "comments": comment_texts,
            "full_text": text
        }

    def categorize_content_batch(
        self,
        content_items: List[Dict[str, Any]],
        batch_size: int = 20,
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Use AI to categorize content into knowledge categories.

        Args:
            content_items: List of content items from extract_from_posts
            batch_size: Number of items to process in each AI call
            progress_callback: Callback function(status, current, total)

        Returns:
            List of categorized content items
        """
        categorized = []

        for i in range(0, len(content_items), batch_size):
            batch = content_items[i:i + batch_size]

            if progress_callback:
                progress_callback(f"Categorizing with AI...", i, len(content_items))

            try:
                result = self._categorize_batch_with_ai(batch)
                categorized.extend(result)
            except Exception as e:
                print(f"[TrainingData] Batch categorization failed: {e}")
                # Add uncategorized items
                for item in batch:
                    item["category"] = "general_tips"
                    item["relevance_score"] = 0.5
                    categorized.append(item)

            print(f"[TrainingData] Categorized {len(categorized)}/{len(content_items)}")

        return categorized

    def _categorize_batch_with_ai(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use GPT to categorize a batch of content."""
        categories_desc = "\n".join([
            f"- {k}: {v}" for k, v in self.KNOWLEDGE_CATEGORIES.items()
        ])

        # Build batch prompt
        items_text = ""
        for i, item in enumerate(batch):
            title = item.get("title", "")[:100]
            body = item.get("body", "")[:200]
            items_text += f"\n[{i}] Title: {title}\nBody: {body[:200]}...\n"

        prompt = f"""Analyze these Tower (mobile idle game) Reddit posts and categorize each one.

Available categories:
{categories_desc}

Posts to categorize:
{items_text}

For each post, respond with JSON:
{{
    "categorizations": [
        {{"index": 0, "category": "category_name", "relevance": 0.0-1.0, "key_info": "brief summary of useful info"}},
        ...
    ]
}}

Only include posts that contain useful game strategy/tips (relevance > 0.3).
Skip memes, questions without answers, and off-topic posts."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You categorize Tower game content for AI training."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        categorizations = result.get("categorizations", [])

        # Apply categorizations to items
        categorized_items = []
        for cat in categorizations:
            idx = cat.get("index", -1)
            if 0 <= idx < len(batch):
                item = batch[idx].copy()
                item["category"] = cat.get("category", "general_tips")
                item["relevance_score"] = cat.get("relevance", 0.5)
                item["key_info"] = cat.get("key_info", "")
                categorized_items.append(item)

        return categorized_items

    def extract_strategies(
        self,
        categorized_content: List[Dict[str, Any]],
        min_relevance: float = 0.5,
        progress_callback=None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concrete strategies from categorized content.

        Args:
            categorized_content: List of categorized content items
            min_relevance: Minimum relevance score to include
            progress_callback: Callback function

        Returns:
            Dictionary mapping categories to lists of strategies
        """
        # Filter by relevance
        relevant = [c for c in categorized_content if c.get("relevance_score", 0) >= min_relevance]
        print(f"[TrainingData] Processing {len(relevant)} relevant items")

        # Group by category
        by_category = {}
        for item in relevant:
            cat = item.get("category", "general_tips")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)

        # Extract strategies per category
        strategies = {}
        total_items = sum(len(items) for items in by_category.values())
        processed = 0

        for category, items in by_category.items():
            if progress_callback:
                progress_callback(f"Extracting {category} strategies...", processed, total_items)

            try:
                cat_strategies = self._extract_category_strategies(category, items)
                strategies[category] = cat_strategies
            except Exception as e:
                print(f"[TrainingData] Failed to extract {category} strategies: {e}")
                strategies[category] = []

            processed += len(items)

        return strategies

    def _extract_category_strategies(
        self,
        category: str,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract strategies for a specific category using AI."""
        # Build context from all items
        context_parts = []
        for item in items[:30]:  # Limit to 30 items per category
            title = item.get("title", "")
            body = item.get("body", "")[:500]
            key_info = item.get("key_info", "")
            score = item.get("score", 0)

            comments_text = ""
            for c in item.get("comments", [])[:3]:
                comments_text += f"\n  - {c.get('text', '')[:200]}"

            context_parts.append(f"""
Post (score: {score}): {title}
{body}
Key info: {key_info}
Top comments:{comments_text}
""")

        context = "\n---\n".join(context_parts)

        prompt = f"""Based on these Reddit posts about "{category}" in The Tower game, extract concrete actionable strategies.

Context from player discussions:
{context}

Extract strategies as JSON:
{{
    "strategies": [
        {{
            "title": "Short strategy title",
            "description": "Detailed explanation of the strategy",
            "priority": "high/medium/low",
            "game_stage": "early/mid/late/all",
            "requirements": ["List of requirements or unlocks needed"],
            "tips": ["Specific tips for this strategy"],
            "source_consensus": 0.0-1.0 (how many posts agree on this)
        }}
    ]
}}

Focus on:
1. Strategies mentioned by multiple players
2. Concrete actionable advice
3. Clear explanations of WHY something works
4. Requirements and when to use each strategy"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You extract game strategies from player discussions. Be specific and actionable."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("strategies", [])

    def build_knowledge_base(
        self,
        strategies: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Build a structured knowledge base from extracted strategies.

        Returns:
            Complete knowledge base structure
        """
        knowledge_base = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "categories": {},
            "meta": {
                "total_strategies": 0,
                "categories_count": len(strategies)
            }
        }

        for category, cat_strategies in strategies.items():
            knowledge_base["categories"][category] = {
                "name": self.KNOWLEDGE_CATEGORIES.get(category, category),
                "strategies": cat_strategies,
                "strategy_count": len(cat_strategies)
            }
            knowledge_base["meta"]["total_strategies"] += len(cat_strategies)

        return knowledge_base

    def save_training_data(self, data: Dict[str, Any], filename: str = None) -> str:
        """Save training data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.json"

        filepath = self.cache_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[TrainingData] Saved to {filepath}")
        return str(filepath)

    def load_training_data(self, filename: str) -> Dict[str, Any]:
        """Load training data from file."""
        filepath = self.cache_dir / filename

        if not filepath.exists():
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_training_files(self) -> List[Dict[str, Any]]:
        """List all training data files."""
        files = []
        for filepath in self.cache_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                files.append({
                    "filename": filepath.name,
                    "path": str(filepath),
                    "created_at": data.get("created_at", data.get("extracted_at", "unknown")),
                    "type": "knowledge_base" if "categories" in data else "raw_data"
                })
            except:
                pass
        return sorted(files, key=lambda x: x.get("created_at", ""), reverse=True)
