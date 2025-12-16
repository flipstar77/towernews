"""Writer Agent - Writes news articles from Reddit posts."""

from .base import BaseAgent


class WriterAgent(BaseAgent):
    """Agent that writes news articles from Reddit posts."""

    def __init__(self, config: dict):
        """
        Initialize the Writer Agent.

        Args:
            config: Configuration dictionary
        """
        super().__init__(
            name="writer",
            config=config,
            tools={}  # Writer doesn't use tools, only writes text
        )

    async def run(self, task: dict) -> dict:
        """
        Write a news article from a Reddit post.

        Args:
            task: Dictionary with:
                - post: Reddit post data (title, selftext, etc.)
                - style_guide: Writing style guidelines (optional)

        Returns:
            Dictionary with:
                - article: Written news article
                - post_id: ID of the processed post
                - success: Whether writing succeeded
        """
        post = task.get("post", {})
        style_guide = task.get("style_guide", "")

        if not post:
            return {
                "article": "",
                "post_id": None,
                "success": False,
                "error": "No post data provided"
            }

        # Build the prompt
        user_message = self._build_prompt(post, style_guide)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Get response from LLM
        response = self.chat(messages, use_tools=False)

        article = response.content.strip()

        return {
            "article": article,
            "post_id": post.get("id"),
            "post_title": post.get("title"),
            "score": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "success": True
        }

    def _build_prompt(self, post: dict, style_guide: str = "") -> str:
        """
        Build the prompt for the LLM.

        Args:
            post: Reddit post data
            style_guide: Additional style guidelines

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "Transform this Reddit post into a SHORT news segment (2-3 sentences max).",
            "",
            f"POST TITLE: {post.get('title', 'N/A')}",
        ]

        if post.get("selftext"):
            # Truncate long posts
            selftext = post.get("selftext", "")[:500]
            prompt_parts.append(f"POST CONTENT: {selftext}")

        if post.get("score"):
            prompt_parts.append(f"UPVOTES: {post.get('score')}")

        if post.get("num_comments"):
            prompt_parts.append(f"COMMENTS: {post.get('num_comments')}")

        if post.get("flair"):
            prompt_parts.append(f"FLAIR: {post.get('flair')}")

        # Add top comments for context
        top_comments = post.get("top_comments", [])
        if top_comments:
            prompt_parts.append("")
            prompt_parts.append("TOP COMMUNITY REACTIONS:")
            for i, comment in enumerate(top_comments[:3], 1):  # Limit to top 3
                body = comment.get("body", "")[:200]  # Truncate long comments
                score = comment.get("score", 0)
                if body and "[deleted]" not in body and "[removed]" not in body:
                    prompt_parts.append(f"  {i}. ({score} upvotes): {body}")

        if style_guide:
            prompt_parts.append("")
            prompt_parts.append(f"ADDITIONAL STYLE GUIDE: {style_guide}")

        prompt_parts.extend([
            "",
            "Remember: Output ONLY the news text, nothing else. Keep it short, speakable, and engaging.",
            "Use the community reactions to gauge sentiment and add context if relevant."
        ])

        return "\n".join(prompt_parts)

    def write_multiple(self, posts: list[dict], style_guide: str = "") -> list[dict]:
        """
        Write articles for multiple posts (synchronous).

        Args:
            posts: List of Reddit post data
            style_guide: Writing style guidelines

        Returns:
            List of article results
        """
        import asyncio

        async def _write_all():
            results = []
            for post in posts:
                result = await self.run({"post": post, "style_guide": style_guide})
                results.append(result)
            return results

        return asyncio.run(_write_all())
