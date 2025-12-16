"""Orchestrator Agent - Coordinates parallel execution of Writer agents."""

import asyncio
from typing import Optional
from .base import BaseAgent
from .writer import WriterAgent


class OrchestratorAgent(BaseAgent):
    """Agent that orchestrates parallel execution of Writer agents."""

    def __init__(self, config: dict):
        """
        Initialize the Orchestrator Agent.

        Args:
            config: Configuration dictionary
        """
        super().__init__(
            name="orchestrator",
            config=config,
            tools={}
        )
        self.max_retries = 2
        self.timeout = 60  # seconds per writer

    async def run(self, task: dict) -> dict:
        """
        Orchestrate parallel writing of news articles.

        Args:
            task: Dictionary with:
                - posts: List of Reddit posts to process
                - style_guide: Writing style guidelines (optional)

        Returns:
            Dictionary with:
                - articles: List of written articles
                - failed: List of failed post IDs
                - success: Overall success status
        """
        posts = task.get("posts", [])
        style_guide = task.get("style_guide", "")

        if not posts:
            return {
                "articles": [],
                "failed": [],
                "success": False,
                "error": "No posts provided"
            }

        # Create tasks for parallel execution
        tasks = []
        for post in posts:
            writer_task = self._create_writer_task(post, style_guide)
            tasks.append(writer_task)

        # Execute all writers in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        articles = []
        failed = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({
                    "post_id": posts[i].get("id"),
                    "error": str(result)
                })
            elif result.get("success"):
                articles.append(result)
            else:
                failed.append({
                    "post_id": posts[i].get("id"),
                    "error": result.get("error", "Unknown error")
                })

        return {
            "articles": articles,
            "failed": failed,
            "success": len(articles) > 0,
            "total": len(posts),
            "successful": len(articles),
            "failed_count": len(failed)
        }

    async def _create_writer_task(
        self,
        post: dict,
        style_guide: str,
        retry_count: int = 0
    ) -> dict:
        """
        Create and execute a writer task with retry logic.

        Args:
            post: Reddit post data
            style_guide: Writing style guidelines
            retry_count: Current retry attempt

        Returns:
            Writer result dictionary
        """
        writer = WriterAgent(self.config)

        try:
            result = await asyncio.wait_for(
                writer.run({"post": post, "style_guide": style_guide}),
                timeout=self.timeout
            )

            if not result.get("success") and retry_count < self.max_retries:
                # Retry on failure
                return await self._create_writer_task(
                    post, style_guide, retry_count + 1
                )

            return result

        except asyncio.TimeoutError:
            if retry_count < self.max_retries:
                return await self._create_writer_task(
                    post, style_guide, retry_count + 1
                )
            return {
                "article": "",
                "post_id": post.get("id"),
                "success": False,
                "error": "Writer timed out"
            }

        except Exception as e:
            if retry_count < self.max_retries:
                return await self._create_writer_task(
                    post, style_guide, retry_count + 1
                )
            return {
                "article": "",
                "post_id": post.get("id"),
                "success": False,
                "error": str(e)
            }

    async def run_with_progress(
        self,
        task: dict,
        progress_callback: Optional[callable] = None
    ) -> dict:
        """
        Run orchestration with progress updates.

        Args:
            task: Task dictionary
            progress_callback: Callback function(completed, total, result)

        Returns:
            Result dictionary
        """
        posts = task.get("posts", [])
        style_guide = task.get("style_guide", "")
        total = len(posts)

        articles = []
        failed = []

        for i, post in enumerate(posts):
            result = await self._create_writer_task(post, style_guide)

            if result.get("success"):
                articles.append(result)
            else:
                failed.append({
                    "post_id": post.get("id"),
                    "error": result.get("error")
                })

            if progress_callback:
                progress_callback(i + 1, total, result)

        return {
            "articles": articles,
            "failed": failed,
            "success": len(articles) > 0,
            "total": total,
            "successful": len(articles),
            "failed_count": len(failed)
        }
