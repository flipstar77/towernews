"""Clip Scraper - Scrapes video/gif clips from Reddit for compilation."""

import requests
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RedditClip:
    """A video/gif clip from Reddit."""
    id: str
    title: str
    url: str
    permalink: str
    score: int
    num_comments: int
    author: str
    flair: Optional[str]
    created_utc: float

    # Media info
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    gif_url: Optional[str] = None
    thumbnail: Optional[str] = None
    duration: Optional[float] = None

    # Downloaded paths
    local_video: Optional[str] = None
    local_audio: Optional[str] = None

    # Top comments
    comments: List[Dict[str, Any]] = None

    @property
    def full_url(self) -> str:
        return f"https://www.reddit.com{self.permalink}"

    @property
    def has_video(self) -> bool:
        return self.video_url is not None or self.gif_url is not None


class ClipScraper:
    """Scrapes video clips from Reddit subreddits."""

    BASE_URL = "https://www.reddit.com"
    USER_AGENT = "TowerClips/1.0 (Clip Compilation Bot)"

    # Flairs that indicate funny/meme content
    CLIP_FLAIRS = [
        "meme", "memes", "humor", "funny", "clip", "clips",
        "gameplay", "video", "highlight", "wtf", "lol"
    ]

    def __init__(self, cache_dir: str = "data/clips"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    def fetch_clips(
        self,
        subreddit: str,
        sort: str = "top",
        timeframe: str = "week",
        limit: int = 50,
        min_score: int = 10
    ) -> List[RedditClip]:
        """
        Fetch video/gif clips from subreddit.

        Args:
            subreddit: Subreddit name
            sort: Sort method (top, hot, new)
            timeframe: Time filter for top (day, week, month, year, all)
            limit: Max posts to fetch
            min_score: Minimum upvotes

        Returns:
            List of RedditClip objects with video content
        """
        url = f"{self.BASE_URL}/r/{subreddit}/{sort}.json"
        params = {"limit": limit, "raw_json": 1}
        if sort == "top":
            params["t"] = timeframe

        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        clips = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})

            # Skip low score posts
            if post.get("score", 0) < min_score:
                continue

            # Extract video/gif URL
            video_url, audio_url, gif_url, duration = self._extract_media(post)

            # Skip if no video content
            if not video_url and not gif_url:
                continue

            clip = RedditClip(
                id=post.get("id", ""),
                title=post.get("title", ""),
                url=post.get("url", ""),
                permalink=post.get("permalink", ""),
                score=post.get("score", 0),
                num_comments=post.get("num_comments", 0),
                author=post.get("author", "[deleted]"),
                flair=post.get("link_flair_text"),
                created_utc=post.get("created_utc", 0),
                video_url=video_url,
                audio_url=audio_url,
                gif_url=gif_url,
                thumbnail=post.get("thumbnail"),
                duration=duration,
                comments=[]
            )
            clips.append(clip)

        print(f"[ClipScraper] Found {len(clips)} clips from r/{subreddit}")
        return clips

    def _extract_media(self, post: dict) -> tuple:
        """Extract video/audio/gif URLs from post data."""
        video_url = None
        audio_url = None
        gif_url = None
        duration = None

        # Reddit hosted video
        if post.get("is_video"):
            media = post.get("media", {})
            reddit_video = media.get("reddit_video", {})
            if reddit_video:
                video_url = reddit_video.get("fallback_url")
                duration = reddit_video.get("duration")
                # Audio is separate on Reddit
                if video_url:
                    audio_url = video_url.rsplit("/", 1)[0] + "/DASH_audio.mp4"

        # Reddit gif (converted to mp4)
        preview = post.get("preview", {})
        if preview:
            reddit_video_preview = preview.get("reddit_video_preview", {})
            if reddit_video_preview:
                video_url = video_url or reddit_video_preview.get("fallback_url")
                duration = duration or reddit_video_preview.get("duration")

            # Check for gif variant
            images = preview.get("images", [])
            if images:
                variants = images[0].get("variants", {})
                mp4_variant = variants.get("mp4", {})
                if mp4_variant:
                    gif_url = mp4_variant.get("source", {}).get("url", "").replace("&amp;", "&")

        # External gif hosts
        url = post.get("url", "")
        if "imgur.com" in url and ".gif" in url:
            # Convert imgur gif to mp4
            gif_url = url.replace(".gif", ".mp4")
        elif "gfycat.com" in url or "redgifs.com" in url:
            # Would need API call for these
            pass
        elif url.endswith(".gif"):
            gif_url = url

        return video_url, audio_url, gif_url, duration

    def fetch_comments(
        self,
        subreddit: str,
        post_id: str,
        limit: int = 10,
        min_score: int = 5
    ) -> List[Dict[str, Any]]:
        """Fetch top comments for a post."""
        url = f"{self.BASE_URL}/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": limit, "sort": "top", "raw_json": 1}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except:
            return []

        comments = []
        if len(data) > 1:
            for child in data[1].get("data", {}).get("children", []):
                if child.get("kind") != "t1":
                    continue

                comment = child.get("data", {})
                score = comment.get("score", 0)

                if score < min_score:
                    continue

                body = comment.get("body", "")
                # Skip very long comments or empty ones
                if not body or len(body) > 300 or len(body) < 5:
                    continue

                # Skip mod comments, bots, etc
                author = comment.get("author", "")
                if author.lower() in ["automoderator", "[deleted]", "bot"]:
                    continue

                comments.append({
                    "id": comment.get("id", ""),
                    "body": body,
                    "score": score,
                    "author": author
                })

        # Sort by score and return top ones
        comments.sort(key=lambda x: x["score"], reverse=True)
        return comments[:limit]

    def download_clip(
        self,
        clip: RedditClip,
        output_dir: Path = None
    ) -> bool:
        """
        Download video clip with audio.

        Args:
            clip: RedditClip object
            output_dir: Output directory

        Returns:
            True if successful
        """
        if output_dir is None:
            output_dir = self.cache_dir / datetime.now().strftime("%Y-%m-%d")
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = output_dir / f"clip_{clip.id}.mp4"

        # If already downloaded
        if video_path.exists():
            clip.local_video = str(video_path)
            return True

        try:
            if clip.video_url:
                # Download Reddit video (may need to merge with audio)
                temp_video = output_dir / f"temp_video_{clip.id}.mp4"
                temp_audio = output_dir / f"temp_audio_{clip.id}.mp4"

                # Download video
                self._download_file(clip.video_url, temp_video)

                # Try to download audio
                has_audio = False
                if clip.audio_url:
                    try:
                        self._download_file(clip.audio_url, temp_audio)
                        has_audio = temp_audio.exists() and temp_audio.stat().st_size > 1000
                    except:
                        pass

                # Merge or just copy
                if has_audio:
                    # Merge video and audio with ffmpeg
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(temp_video),
                        "-i", str(temp_audio),
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-shortest",
                        str(video_path)
                    ]
                    subprocess.run(cmd, capture_output=True, check=True)
                    temp_video.unlink()
                    temp_audio.unlink()
                else:
                    # Just rename video
                    temp_video.rename(video_path)
                    if temp_audio.exists():
                        temp_audio.unlink()

            elif clip.gif_url:
                # Download gif/mp4 directly
                self._download_file(clip.gif_url, video_path)

            if video_path.exists():
                clip.local_video = str(video_path)
                print(f"[ClipScraper] Downloaded: {clip.title[:40]}...")
                return True

        except Exception as e:
            print(f"[ClipScraper] Failed to download {clip.id}: {e}")
            # Cleanup
            for f in output_dir.glob(f"*{clip.id}*"):
                try:
                    f.unlink()
                except:
                    pass

        return False

    def _download_file(self, url: str, path: Path):
        """Download a file from URL."""
        response = self.session.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def get_clips_with_comments(
        self,
        subreddit: str,
        count: int = 10,
        sort: str = "top",
        timeframe: str = "week",
        min_score: int = 20,
        comments_per_clip: int = 3
    ) -> List[RedditClip]:
        """
        Get clips with their top comments.

        Args:
            subreddit: Subreddit name
            count: Number of clips to get
            sort: Sort method
            timeframe: Time filter
            min_score: Minimum score
            comments_per_clip: Number of comments per clip

        Returns:
            List of clips with comments populated
        """
        # Fetch more than needed since some won't download
        clips = self.fetch_clips(
            subreddit=subreddit,
            sort=sort,
            timeframe=timeframe,
            limit=count * 3,
            min_score=min_score
        )

        successful_clips = []

        for clip in clips:
            if len(successful_clips) >= count:
                break

            # Download clip
            if not self.download_clip(clip):
                continue

            # Fetch comments
            clip.comments = self.fetch_comments(
                subreddit=subreddit,
                post_id=clip.id,
                limit=comments_per_clip,
                min_score=5
            )

            successful_clips.append(clip)
            print(f"[ClipScraper] Got clip {len(successful_clips)}/{count}: {clip.title[:30]}... ({len(clip.comments)} comments)")

        return successful_clips

    def filter_funny_clips(
        self,
        clips: List[RedditClip],
        require_flair: bool = False
    ) -> List[RedditClip]:
        """Filter clips to only include funny/meme content."""
        filtered = []

        for clip in clips:
            flair = (clip.flair or "").lower()
            title = clip.title.lower()

            # Check flair
            is_funny_flair = any(f in flair for f in self.CLIP_FLAIRS)

            # Check title keywords
            funny_keywords = ["lol", "lmao", "wtf", "omg", "rip", "bruh", "xd", "haha", "funny", "meme"]
            has_funny_keyword = any(kw in title for kw in funny_keywords)

            if require_flair and not is_funny_flair:
                continue

            if is_funny_flair or has_funny_keyword:
                filtered.append(clip)
            elif not require_flair:
                # Include all if not requiring flair
                filtered.append(clip)

        return filtered
