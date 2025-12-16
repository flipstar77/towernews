"""Metadata Generator - Creates YouTube-ready title, description, and tags."""

import os
from datetime import datetime
from pathlib import Path
from openai import OpenAI


class MetadataGenerator:
    """Generates SEO-optimized metadata for YouTube uploads."""

    def __init__(self, config: dict):
        self.config = config
        self.channel_name = config.get("channel", {}).get("name", "Tower News")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = config.get("llm", {}).get("model", "gpt-4o-mini")

    def generate(
        self,
        articles: list[dict],
        posts: list[dict],
        output_dir: str = None
    ) -> dict:
        """
        Generate YouTube metadata from articles and posts.

        Args:
            articles: List of written articles with post_title, article text
            posts: List of original Reddit posts with urls, scores, etc.
            output_dir: Directory to save metadata file

        Returns:
            Dictionary with title, description, tags, and file_path
        """
        today = datetime.now()
        date_str = today.strftime("%Y-%m-%d")
        date_display = today.strftime("%B %d, %Y")

        if output_dir is None:
            output_dir = Path("output") / date_str
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate title using AI
        title = self._generate_title(articles, date_display)

        # Generate description using AI
        description = self._generate_description(articles, posts, date_display)

        # Generate tags
        tags = self._generate_tags(articles)

        # Save to file
        metadata_path = output_dir / f"metadata_{today.strftime('%H%M%S')}.txt"
        self._save_metadata(metadata_path, title, description, tags, posts)

        return {
            "title": title,
            "description": description,
            "tags": tags,
            "file_path": str(metadata_path),
            "success": True
        }

    def _generate_title(self, articles: list[dict], date_display: str) -> str:
        """Generate an SEO-optimized YouTube title."""
        # Extract headlines for context
        headlines = [a.get("post_title", "") for a in articles if a.get("post_title")]

        prompt = f"""Create a YouTube title for a daily gaming news video about "The Tower" mobile game.

Date: {date_display}
Channel: {self.channel_name}

Today's stories cover:
{chr(10).join(f'- {h}' for h in headlines[:3])}

RULES:
1. Maximum 60 characters (YouTube truncates longer titles)
2. Include the game name "Tower" or "The Tower"
3. Make it click-worthy but NOT clickbait
4. Include the date or "Daily" to show freshness
5. PRESERVE any game abbreviations (CL, SM, PBH, etc.) - do NOT expand them
6. Use | or - as separators if needed

GOOD EXAMPLES:
- Tower News Daily | Dec 16 Update & CL Strategy
- The Tower | Weekly Meta Changes & New Tips
- Tower Game News: What's New This Week

Output ONLY the title, nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )

        return response.choices[0].message.content.strip().strip('"')

    def _generate_description(
        self,
        articles: list[dict],
        posts: list[dict],
        date_display: str
    ) -> str:
        """Generate an SEO-optimized YouTube description."""
        # Build post info for links
        post_info = []
        for i, (article, post) in enumerate(zip(articles, posts), 1):
            title = article.get("post_title", post.get("title", f"Story {i}"))
            url = post.get("url", "")
            score = post.get("score", 0)
            comments = post.get("num_comments", 0)
            post_info.append({
                "num": i,
                "title": title,
                "url": url,
                "score": score,
                "comments": comments
            })

        # Build article summaries
        article_texts = [a.get("article", "")[:200] for a in articles]

        prompt = f"""Create a YouTube description for a daily gaming news video about "The Tower" mobile game.

Date: {date_display}
Channel: {self.channel_name}

Stories covered:
{chr(10).join(f'{i+1}. {a[:150]}...' for i, a in enumerate(article_texts))}

RULES:
1. Start with a compelling 1-2 sentence hook (this shows in search results)
2. Include relevant keywords: Tower game, mobile game, gaming news, strategy, tips
3. Be informative but concise
4. PRESERVE any game abbreviations (CL, SM, PBH, UW, DimCore, etc.) - do NOT expand them
5. Do NOT include timestamps, links, or hashtags (those will be added separately)
6. Maximum 200 words

Output ONLY the description text, nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7
        )

        ai_description = response.choices[0].message.content.strip()

        # Build final description with links
        final_description = f"""{ai_description}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📰 TODAY'S STORIES:
"""

        for p in post_info:
            final_description += f"\n{p['num']}. {p['title']}"
            final_description += f"\n   🔗 {p['url']}"
            final_description += f"\n   👍 {p['score']:,} upvotes | 💬 {p['comments']:,} comments\n"

        final_description += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎮 ABOUT {self.channel_name.upper()}:
Daily news and updates from The Tower game community.
Subscribe for your daily dose of Tower news!

📱 Join the community: reddit.com/r/TheTowerGame

#TheTower #TowerGame #MobileGaming #GamingNews #DailyNews
"""

        return final_description

    def _generate_tags(self, articles: list[dict]) -> list[str]:
        """Generate relevant YouTube tags."""
        base_tags = [
            "The Tower",
            "Tower Game",
            "The Tower Game",
            "mobile game",
            "gaming news",
            "daily news",
            "tower defense",
            "mobile gaming",
            "game tips",
            "game strategy",
            "reddit gaming",
            "gaming community"
        ]

        # Extract potential keywords from article titles
        for article in articles:
            title = article.get("post_title", "")
            # Add relevant words from titles as tags
            words = title.split()
            for word in words:
                # Keep abbreviations and meaningful words
                clean = word.strip(".,!?()[]")
                if len(clean) >= 2 and clean.upper() == clean:
                    # It's an abbreviation like CL, SM, PBH
                    base_tags.append(clean)
                elif len(clean) >= 4 and clean.isalpha():
                    base_tags.append(clean.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in base_tags:
            if tag.lower() not in seen:
                seen.add(tag.lower())
                unique_tags.append(tag)

        return unique_tags[:30]  # YouTube allows max 500 chars, ~30 tags

    def _save_metadata(
        self,
        path: Path,
        title: str,
        description: str,
        tags: list[str],
        posts: list[dict]
    ) -> None:
        """Save metadata to a text file."""
        content = f"""=== YOUTUBE METADATA ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== TITLE ===
{title}

=== DESCRIPTION ===
{description}

=== TAGS ===
{', '.join(tags)}

=== SOURCE LINKS ===
"""
        for i, post in enumerate(posts, 1):
            content += f"{i}. {post.get('url', 'N/A')}\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[MetadataGenerator] Saved metadata to: {path}")

    def run(self, **kwargs) -> dict:
        """Tool interface for pipeline."""
        return self.generate(
            articles=kwargs.get("articles", []),
            posts=kwargs.get("posts", []),
            output_dir=kwargs.get("output_dir")
        )
