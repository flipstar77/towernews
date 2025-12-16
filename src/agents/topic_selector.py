"""Topic Selector Agent - Intelligently selects newsworthy topics."""

import json
from .base import BaseAgent


class TopicSelectorAgent(BaseAgent):
    """Agent that evaluates and selects the best topics for news coverage."""

    def __init__(self, config: dict):
        """
        Initialize the Topic Selector Agent.

        Args:
            config: Configuration dictionary
        """
        super().__init__(
            name="topic_selector",
            config=config,
            tools={}  # No tools needed, pure LLM analysis
        )

    async def run(self, task: dict) -> dict:
        """
        Evaluate posts and select the best topics.

        Args:
            task: Dictionary with:
                - posts: List of Reddit post data with comments
                - max_topics: Maximum number of topics to select (default 3)

        Returns:
            Dictionary with:
                - selected_posts: List of selected post data
                - evaluations: Full evaluation results
                - success: Whether selection succeeded
        """
        posts = task.get("posts", [])
        max_topics = task.get("max_topics", 3)

        if not posts:
            return {
                "selected_posts": [],
                "evaluations": [],
                "success": False,
                "error": "No posts provided"
            }

        # Build the prompt with all posts
        user_message = self._build_evaluation_prompt(posts)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Get evaluation from LLM
        response = self.chat(messages, use_tools=False)

        # Parse the response
        try:
            evaluations = self._parse_evaluations(response.content)
        except Exception as e:
            print(f"[{self.name}] Failed to parse evaluations: {e}")
            # Fallback: return posts sorted by engagement
            return {
                "selected_posts": posts[:max_topics],
                "evaluations": [],
                "success": True,
                "fallback": True
            }

        # Filter accepted posts and sort by score
        accepted = [e for e in evaluations if e.get("accept", False)]
        accepted.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Get the selected posts
        selected_ids = [e["post_id"] for e in accepted[:max_topics]]
        selected_posts = [p for p in posts if p.get("id") in selected_ids]

        # Maintain order by score
        id_to_score = {e["post_id"]: e.get("score", 0) for e in evaluations}
        selected_posts.sort(key=lambda p: id_to_score.get(p.get("id"), 0), reverse=True)

        # Log selections
        for eval_item in accepted[:max_topics]:
            print(f"[{self.name}] Selected: {eval_item.get('post_id')} "
                  f"(score: {eval_item.get('score')}) - {eval_item.get('reason', '')[:50]}")

        rejected = [e for e in evaluations if not e.get("accept", False)]
        for eval_item in rejected:
            print(f"[{self.name}] Rejected: {eval_item.get('post_id')} - {eval_item.get('reason', '')[:50]}")

        return {
            "selected_posts": selected_posts,
            "evaluations": evaluations,
            "success": True,
            "selected_count": len(selected_posts),
            "rejected_count": len(rejected)
        }

    def _build_evaluation_prompt(self, posts: list[dict]) -> str:
        """
        Build the evaluation prompt with all posts.

        Args:
            posts: List of post data

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "Evaluate the following Reddit posts for news coverage.",
            "Return a JSON array with your evaluation for each post.",
            "",
            "POSTS TO EVALUATE:",
            ""
        ]

        for i, post in enumerate(posts, 1):
            prompt_parts.append(f"--- POST {i} ---")
            prompt_parts.append(f"ID: {post.get('id', 'unknown')}")
            prompt_parts.append(f"TITLE: {post.get('title', 'N/A')}")
            prompt_parts.append(f"FLAIR: {post.get('flair', 'None')}")
            prompt_parts.append(f"UPVOTES: {post.get('score', 0)}")
            prompt_parts.append(f"COMMENTS: {post.get('num_comments', 0)}")

            if post.get("selftext"):
                selftext = post.get("selftext", "")[:300]
                prompt_parts.append(f"CONTENT: {selftext}")

            # Add top comments
            comments = post.get("top_comments", [])
            if comments:
                prompt_parts.append("TOP COMMENTS:")
                for j, comment in enumerate(comments[:3], 1):
                    body = comment.get("body", "")[:150]
                    score = comment.get("score", 0)
                    if body and "[deleted]" not in body:
                        prompt_parts.append(f"  {j}. ({score} pts): {body}")

            prompt_parts.append("")

        prompt_parts.append("Respond with ONLY a JSON array. No explanation before or after.")

        return "\n".join(prompt_parts)

    def _parse_evaluations(self, response: str) -> list[dict]:
        """
        Parse the LLM response into evaluation objects.

        Args:
            response: Raw LLM response

        Returns:
            List of evaluation dictionaries
        """
        # Try to find JSON in the response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        # Parse JSON
        evaluations = json.loads(response)

        # Ensure it's a list
        if isinstance(evaluations, dict):
            evaluations = [evaluations]

        return evaluations

    def select_topics_sync(self, posts: list[dict], max_topics: int = 3) -> list[dict]:
        """
        Synchronous wrapper for topic selection.

        Args:
            posts: List of post data
            max_topics: Maximum topics to select

        Returns:
            List of selected posts
        """
        import asyncio
        result = asyncio.run(self.run({"posts": posts, "max_topics": max_topics}))
        return result.get("selected_posts", [])
