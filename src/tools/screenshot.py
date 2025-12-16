"""Screenshot Tool - Takes screenshots of Reddit posts using Playwright."""

import asyncio
from pathlib import Path
from datetime import datetime

# Check if playwright is available
PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    print("[ScreenshotTool] Warning: Playwright not installed. Screenshots will be skipped.")


class ScreenshotTool:
    """Takes screenshots of Reddit posts."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path("output")
        self._browser = None
        self._playwright = None

    async def _ensure_browser(self):
        """Ensure browser is initialized."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright is not installed. Run: pip install playwright && playwright install chromium")

        if self._browser is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)

    async def close(self):
        """Close browser and cleanup."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def take_screenshot(
        self,
        url: str,
        output_path: str | None = None,
        width: int = 1080,
        height: int = 1920,
        wait_for_selector: str | None = None
    ) -> str:
        """
        Take a screenshot of a URL.

        Args:
            url: URL to screenshot
            output_path: Path to save screenshot (optional)
            width: Viewport width
            height: Viewport height
            wait_for_selector: CSS selector to wait for before screenshot

        Returns:
            Path to saved screenshot
        """
        await self._ensure_browser()

        # Create output directory
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_dir / today / "screenshots"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = str(output_dir / f"screenshot_{timestamp}.png")

        page = await self._browser.new_page(viewport={"width": width, "height": height})

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for content to load
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=10000)
            else:
                await asyncio.sleep(2)  # Give page time to render

            # Take screenshot
            await page.screenshot(path=output_path, full_page=False)

            return output_path

        finally:
            await page.close()

    async def screenshot_reddit_post(
        self,
        post_url: str,
        post_id: str,
        output_dir: str | None = None
    ) -> str:
        """
        Take a screenshot of a Reddit post.

        Args:
            post_url: Full URL to the Reddit post
            post_id: ID of the post (for filename)
            output_dir: Directory to save screenshot

        Returns:
            Path to saved screenshot
        """
        await self._ensure_browser()

        # Create output directory
        today = datetime.now().strftime("%Y-%m-%d")
        if output_dir is None:
            output_dir = self.output_dir / today / "screenshots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = str(output_dir / f"post_{post_id}.png")

        # Reddit mobile view for better screenshot
        page = await self._browser.new_page(
            viewport={"width": 800, "height": 1200},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        )

        try:
            # Use old.reddit.com for cleaner screenshots
            clean_url = post_url.replace("www.reddit.com", "old.reddit.com")
            await page.goto(clean_url, wait_until="networkidle", timeout=30000)

            # Wait for post content
            await asyncio.sleep(2)

            # Try to find and screenshot just the post content
            post_selector = ".thing.link, .Post, [data-testid='post-container']"
            try:
                element = await page.wait_for_selector(post_selector, timeout=5000)
                if element:
                    await element.screenshot(path=output_path)
                else:
                    await page.screenshot(path=output_path, full_page=False)
            except Exception:
                # Fallback to full page screenshot
                await page.screenshot(path=output_path, full_page=False)

            return output_path

        finally:
            await page.close()

    async def screenshot_multiple_posts(self, posts: list[dict]) -> list[str]:
        """
        Take screenshots of multiple Reddit posts.

        Args:
            posts: List of post dictionaries with 'url' and 'id' keys

        Returns:
            List of screenshot paths
        """
        screenshots = []

        for post in posts:
            try:
                path = await self.screenshot_reddit_post(
                    post_url=post.get("url", ""),
                    post_id=post.get("id", "unknown")
                )
                screenshots.append(path)
            except Exception as e:
                print(f"Failed to screenshot post {post.get('id')}: {e}")
                screenshots.append(None)

        return screenshots

    def run(self, **kwargs) -> dict:
        """
        Tool interface for agents.

        Args:
            posts: List of post dictionaries with 'url' and 'id' keys
            url: Single URL to screenshot (alternative to posts)

        Returns:
            Dictionary with 'screenshots' list
        """
        # If Playwright isn't available, return empty result
        if not PLAYWRIGHT_AVAILABLE:
            print("[ScreenshotTool] Playwright not available - skipping screenshots")
            return {
                "screenshots": [],
                "count": 0,
                "error": "Playwright not installed"
            }

        posts = kwargs.get("posts", [])
        url = kwargs.get("url")

        async def _run():
            try:
                if posts:
                    screenshots = await self.screenshot_multiple_posts(posts)
                elif url:
                    screenshot = await self.take_screenshot(url)
                    screenshots = [screenshot]
                else:
                    screenshots = []

                return {
                    "screenshots": screenshots,
                    "count": len([s for s in screenshots if s is not None])
                }
            finally:
                await self.close()

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run())
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(_run())
