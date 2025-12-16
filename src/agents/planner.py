"""Planner Agent - Strategic planning and quality control."""

import asyncio
import html
from datetime import datetime
from pathlib import Path
from .base import BaseAgent
from .orchestrator import OrchestratorAgent
from .topic_selector import TopicSelectorAgent
from ..tools import RedditScraper, ScreenshotTool, TTSTool, VideoGenerator, MetadataGenerator, YouTubeUploader, HistoryTracker


class PlannerAgent(BaseAgent):
    """Agent that plans and coordinates the entire news pipeline."""

    def __init__(self, config: dict):
        """
        Initialize the Planner Agent.

        Args:
            config: Configuration dictionary
        """
        # Initialize tools
        tools = {
            "reddit_scraper": RedditScraper(config),
            "screenshot": ScreenshotTool(config),
            "tts": TTSTool(config),
            "video_generator": VideoGenerator(config),
            "metadata_generator": MetadataGenerator(config),
            "youtube_uploader": YouTubeUploader(config),
            "history_tracker": HistoryTracker(config)
        }

        super().__init__(
            name="planner",
            config=config,
            tools=tools
        )

        self.orchestrator = OrchestratorAgent(config)
        self.topic_selector = TopicSelectorAgent(config)
        self.channel_name = config.get("channel", {}).get("name", "News")

    async def run(self, task: dict = None) -> dict:
        """
        Run the complete news pipeline.

        Args:
            task: Optional task overrides

        Returns:
            Dictionary with pipeline results
        """
        task = task or {}
        results = {
            "success": False,
            "stages": {},
            "video_path": None,
            "error": None
        }

        try:
            # Stage 1: Scrape Reddit
            print(f"[{self.name}] Stage 1: Scraping Reddit...")
            posts = await self._stage_scrape_reddit(task)
            results["stages"]["scrape"] = {"posts": len(posts), "success": True}

            if not posts:
                results["error"] = "No posts found"
                return results

            # Stage 1b: Filter out already-reported posts
            print(f"[{self.name}] Stage 1b: Filtering already-reported posts...")
            history = self.tools["history_tracker"]
            posts = history.filter_unreported(posts)
            results["stages"]["history_filter"] = {"remaining": len(posts), "success": True}

            if not posts:
                results["error"] = "All posts have already been reported"
                return results

            # Stage 2: Select top posts
            print(f"[{self.name}] Stage 2: Selecting top posts...")
            selected_posts = await self._stage_select_posts(posts, task)
            results["stages"]["select"] = {"selected": len(selected_posts), "success": True}

            # Stage 3: Get images (from Reddit or screenshots)
            print(f"[{self.name}] Stage 3: Collecting images...")
            images = await self._stage_get_images(selected_posts)
            results["stages"]["images"] = {"count": len(images), "success": True}

            # Stage 4: Write articles via Orchestrator
            print(f"[{self.name}] Stage 4: Writing articles...")
            articles = await self._stage_write_articles(selected_posts)
            results["stages"]["articles"] = {
                "count": len(articles),
                "success": len(articles) > 0
            }

            if not articles:
                results["error"] = "No articles written"
                return results

            # Stage 5: Generate audio (segmented for precise timing)
            print(f"[{self.name}] Stage 5: Generating audio...")
            audio_result = await self._stage_generate_audio(articles)
            audio_path = audio_result.get("audio_path")
            audio_segments = audio_result.get("segments", [])
            results["stages"]["audio"] = {
                "path": audio_path,
                "segments": len(audio_segments),
                "success": bool(audio_path)
            }

            if not audio_path:
                results["error"] = "Audio generation failed"
                return results

            # Stage 6: Generate video (with segment timing)
            print(f"[{self.name}] Stage 6: Generating video...")
            video_path = await self._stage_generate_video(
                audio_path, images, articles, audio_segments
            )
            results["stages"]["video"] = {"path": video_path, "success": bool(video_path)}
            results["video_path"] = video_path

            # Stage 7: Quality check
            print(f"[{self.name}] Stage 7: Quality check...")
            quality_ok = await self._stage_quality_check(video_path, articles)
            results["stages"]["quality"] = {"passed": quality_ok, "success": quality_ok}

            # Stage 8: Generate metadata (title, description, links)
            print(f"[{self.name}] Stage 8: Generating metadata...")
            metadata = await self._stage_generate_metadata(articles, selected_posts)
            results["stages"]["metadata"] = {
                "path": metadata.get("file_path"),
                "success": metadata.get("success", False)
            }
            results["metadata_path"] = metadata.get("file_path")

            # Stage 9: Upload to YouTube (optional - controlled by config)
            youtube_config = self.config.get("youtube", {})
            if youtube_config.get("auto_upload", False):
                print(f"[{self.name}] Stage 9: Uploading to YouTube...")
                upload_result = await self._stage_upload_youtube(
                    video_path,
                    metadata.get("file_path"),
                    youtube_config.get("privacy_status", "private")
                )
                results["stages"]["youtube_upload"] = upload_result
                results["youtube_url"] = upload_result.get("url")
            else:
                print(f"[{self.name}] Stage 9: YouTube upload skipped (auto_upload=false)")

            # Stage 10: Mark posts as reported (to avoid duplicates)
            if quality_ok:
                print(f"[{self.name}] Stage 10: Marking posts as reported...")
                history.mark_all_reported(selected_posts)
                results["stages"]["history_mark"] = {"marked": len(selected_posts), "success": True}

            results["success"] = quality_ok
            print(f"[{self.name}] Pipeline complete! Video: {video_path}")

        except Exception as e:
            results["error"] = str(e)
            print(f"[{self.name}] Pipeline failed: {e}")

        return results

    async def _stage_scrape_reddit(self, task: dict) -> list[dict]:
        """Stage 1: Scrape Reddit for posts."""
        scraper = self.tools["reddit_scraper"]
        result = scraper.run(**task.get("scraper_params", {}))
        return result.get("posts", [])

    async def _stage_select_posts(
        self,
        posts: list[dict],
        task: dict
    ) -> list[dict]:
        """Stage 2: Select the best posts using TopicSelector Agent."""
        max_posts = task.get("max_posts", 3)
        use_ai_selection = task.get("use_ai_selection", True)

        if use_ai_selection and len(posts) > max_posts:
            # Use TopicSelector Agent for intelligent selection
            print(f"[{self.name}] Using AI topic selection...")
            result = await self.topic_selector.run({
                "posts": posts,
                "max_topics": max_posts
            })

            if result.get("success") and result.get("selected_posts"):
                return result["selected_posts"]

            # Fallback if AI selection fails
            print(f"[{self.name}] AI selection failed, using engagement score...")

        # Fallback: take top N by engagement
        sorted_posts = sorted(
            posts,
            key=lambda p: p.get("engagement_score", 0),
            reverse=True
        )

        return sorted_posts[:max_posts]

    async def _stage_get_images(self, posts: list[dict]) -> list[str]:
        """Stage 3: Get images from Reddit posts or take screenshots as fallback."""
        images = []
        scraper = self.tools["reddit_scraper"]

        # For each selected post, try to get its image
        for post in posts:
            local_image = post.get("local_image")

            # If already downloaded, use it
            if local_image and Path(local_image).exists():
                images.append(local_image)
                print(f"[{self.name}] Using existing image for post {post.get('id')}")
                continue

            # Try to download image for this specific post
            image_url = post.get("image_url")
            if image_url:
                downloaded = scraper.download_image(image_url, post.get("id"))
                if downloaded:
                    images.append(downloaded)
                    post["local_image"] = downloaded  # Update the post dict
                    print(f"[{self.name}] Downloaded image for post {post.get('id')}")
                    continue

            print(f"[{self.name}] No image available for post {post.get('id')}")

        # If we got images from Reddit, use those
        if images:
            print(f"[{self.name}] Collected {len(images)} images for {len(posts)} posts")
            return images

        # Fallback: try to take screenshots (if Playwright is available)
        print(f"[{self.name}] No Reddit images found, attempting screenshots...")
        screenshot_tool = self.tools["screenshot"]

        # Prepare posts with required fields
        screenshot_posts = [
            {"url": p.get("url"), "id": p.get("id")}
            for p in posts
        ]

        result = screenshot_tool.run(posts=screenshot_posts)
        screenshots = [s for s in result.get("screenshots", []) if s is not None]

        if screenshots:
            print(f"[{self.name}] Took {len(screenshots)} screenshots")
        else:
            print(f"[{self.name}] No screenshots available")

        return screenshots

    async def _stage_write_articles(self, posts: list[dict]) -> list[dict]:
        """Stage 4: Write articles using Orchestrator."""
        result = await self.orchestrator.run({
            "posts": posts,
            "style_guide": "Keep it engaging and suitable for a gaming news show."
        })

        return result.get("articles", [])

    async def _stage_generate_audio(self, articles: list[dict]) -> dict:
        """Stage 5: Generate TTS audio with segment timing."""
        tts_tool = self.tools["tts"]

        # Load templates
        intro = self._load_template("intro")
        outro = self._load_template("outro")

        # Format intro with date
        today = datetime.now().strftime("%B %d, %Y")
        intro = intro.format(channel_name=self.channel_name, date=today)
        outro = outro.format(channel_name=self.channel_name)

        # Extract article texts
        story_texts = [a.get("article", "") for a in articles if a.get("article")]

        if not story_texts:
            return {"audio_path": None, "segments": []}

        # Get intro music path from config
        intro_music_path = self.config.get("audio", {}).get("intro_music")

        # Use segmented audio for precise timing
        result = tts_tool.run(
            intro=intro,
            stories=story_texts,
            outro=outro,
            segmented=True,
            intro_music_path=intro_music_path
        )

        # Log segment durations
        if result.get("segments"):
            for seg in result["segments"]:
                print(f"[{self.name}] Audio segment: {seg['type']} = {seg['duration']:.1f}s")

        return result

    async def _stage_generate_video(
        self,
        audio_path: str,
        screenshots: list[str],
        articles: list[dict],
        audio_segments: list[dict] = None
    ) -> str | None:
        """Stage 6: Generate the final video using segmented approach."""
        video_tool = self.tools["video_generator"]

        # Create ticker text from headlines
        ticker_parts = []
        for article in articles:
            title = article.get("post_title", "")
            if title:
                # Decode HTML entities first
                clean_title = html.unescape(title)
                # Clean title for ticker display (remove problematic chars)
                clean_title = clean_title.replace("'", "").replace('"', "").replace(":", " -")
                ticker_parts.append(f">>> {clean_title}")

        ticker_text = "    ".join(ticker_parts) if ticker_parts else "Tower News - Daily Updates"
        print(f"[{self.name}] Ticker text: {ticker_text[:100]}...")

        # Get presenter image path
        presenter_image = self.config.get("video", {}).get("presenter_image")

        # Always use segmented mode when we have audio_segments (fallback images handled by VideoGenerator)
        if audio_segments:
            # Use SEGMENTED video generation - each segment rendered separately
            # This ensures perfect audio-video sync
            print(f"[{self.name}] Using segmented video generation ({len(audio_segments)} segments)")
            result = video_tool.run(
                audio_segments=audio_segments,
                screenshots=screenshots,
                ticker_text=ticker_text,
                presenter_image=presenter_image,
                articles=articles,
                segmented=True  # Enable segmented mode
            )
        else:
            # Fallback: single video with audio_path
            result = video_tool.run(
                audio_path=audio_path,
                screenshots=screenshots,
                ticker_text=ticker_text,
                presenter_image=presenter_image,
                use_greenscreen=True,
                articles=articles
            )

        return result.get("video_path")

    async def _stage_quality_check(
        self,
        video_path: str | None,
        articles: list[dict]
    ) -> bool:
        """Stage 7: Perform quality checks."""
        if not video_path:
            return False

        if not Path(video_path).exists():
            return False

        if len(articles) == 0:
            return False

        # Check video file size (should be > 100KB)
        file_size = Path(video_path).stat().st_size
        if file_size < 100 * 1024:
            return False

        return True

    async def _stage_generate_metadata(
        self,
        articles: list[dict],
        posts: list[dict]
    ) -> dict:
        """Stage 8: Generate YouTube metadata (title, description, links)."""
        metadata_tool = self.tools["metadata_generator"]

        try:
            result = metadata_tool.run(
                articles=articles,
                posts=posts
            )
            print(f"[{self.name}] Generated metadata: {result.get('title', 'N/A')[:50]}...")
            return result
        except Exception as e:
            print(f"[{self.name}] Metadata generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _stage_upload_youtube(
        self,
        video_path: str,
        metadata_path: str,
        privacy_status: str = "private"
    ) -> dict:
        """Stage 9: Upload video to YouTube."""
        youtube_tool = self.tools["youtube_uploader"]

        try:
            result = youtube_tool.run(
                video_path=video_path,
                metadata_path=metadata_path,
                privacy_status=privacy_status
            )
            if result.get("success"):
                print(f"[{self.name}] YouTube upload successful: {result.get('url')}")
            return result
        except Exception as e:
            print(f"[{self.name}] YouTube upload failed: {e}")
            return {"success": False, "error": str(e)}

    def _load_template(self, template_name: str) -> str:
        """Load a template file."""
        template_path = Path("templates") / f"{template_name}.txt"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        return ""

    def run_pipeline(self, **kwargs) -> dict:
        """
        Synchronous wrapper to run the pipeline.

        Args:
            **kwargs: Task parameters

        Returns:
            Pipeline results
        """
        return asyncio.run(self.run(kwargs))
