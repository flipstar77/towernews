"""Video Generator Tool - Creates news videos using FFmpeg with Greenscreen support."""

import subprocess
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional


class VideoGenerator:
    """Generates news videos with Ken Burns effect, news ticker, and greenscreen keying."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path("output")

        video_config = config.get("video", {})
        self.width = video_config.get("width", 1080)
        self.height = video_config.get("height", 1920)
        self.fps = video_config.get("fps", 30)
        self.ken_burns_zoom = video_config.get("ken_burns_zoom", 1.1)
        self.ticker_speed = video_config.get("ticker_speed", 50)
        # Presenter with greenscreen (for stories WITH B-roll)
        self.presenter_image = video_config.get("presenter_image", "assets/presenter/presenter.png")
        # Fullscreen presenter (for stories WITHOUT B-roll - has studio background)
        self.presenter_fullscreen = video_config.get("presenter_fullscreen")

        # Intro videos (multiple for variety - randomly selected)
        audio_config = config.get("audio", {})
        self.intro_videos = audio_config.get("intro_videos", [])
        # Fallback to single intro_video for backwards compatibility
        if not self.intro_videos and audio_config.get("intro_video"):
            self.intro_videos = [audio_config.get("intro_video")]

        # Outro videos (multiple for variety - randomly selected)
        self.outro_videos = audio_config.get("outro_videos", [])

        # Greeting videos (short ~5s, for intro greeting after jingle)
        self.greeting_videos = audio_config.get("greeting_videos", [])

        # Story videos (longer ~10s, for stories without B-roll - looped to match duration)
        self.story_videos = audio_config.get("story_videos", [])

        # Background music for stories and intro greeting
        self.background_music = audio_config.get("background_music")
        self.background_volume = audio_config.get("background_volume", -20)  # dB

        # Branding
        branding_config = config.get("branding", {})
        self.brand_icon = branding_config.get("icon")

        # Fallback backgrounds for when no Reddit image available
        self.fallback_backgrounds = self._load_fallback_backgrounds(
            video_config.get("fallback_backgrounds", [])
        )

        # Greenscreen/Chroma-key settings
        self.chroma_key_color = video_config.get("chroma_key_color", "0x00ff00")  # Green
        self.chroma_similarity = video_config.get("chroma_similarity", 0.3)  # How similar colors must be
        self.chroma_blend = video_config.get("chroma_blend", 0.1)  # Edge blending

        # Track used videos to avoid repeats (reset per video generation)
        self._used_story_videos = []
        self._last_story_video = None

    def _load_fallback_backgrounds(self, paths: list) -> list[str]:
        """Load all fallback background images from configured paths."""
        backgrounds = []
        for path in paths:
            p = Path(path)
            if p.is_dir():
                # Load all images from directory
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
                    backgrounds.extend([str(f) for f in p.glob(ext)])
            elif p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                backgrounds.append(str(p))
        return backgrounds

    def _get_fallback_background(self) -> Optional[str]:
        """Get a random fallback background image."""
        if self.fallback_backgrounds:
            return random.choice(self.fallback_backgrounds)
        return None

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 60))

    def _get_random_intro_video(self) -> Optional[str]:
        """Get a random intro video from the list."""
        valid_videos = [v for v in self.intro_videos if Path(v).exists()]
        if valid_videos:
            selected = random.choice(valid_videos)
            print(f"[VideoGenerator] Selected intro video: {Path(selected).name}")
            return selected
        return None

    def _get_random_outro_video(self) -> Optional[str]:
        """Get a random outro video from the list."""
        valid_videos = [v for v in self.outro_videos if Path(v).exists()]
        if valid_videos:
            selected = random.choice(valid_videos)
            print(f"[VideoGenerator] Selected outro video: {Path(selected).name}")
            return selected
        return None

    def _get_random_greeting_video(self) -> Optional[str]:
        """Get a random greeting video (short ~5s) from the list."""
        valid_videos = [v for v in self.greeting_videos if Path(v).exists()]
        if valid_videos:
            selected = random.choice(valid_videos)
            print(f"[VideoGenerator] Selected greeting video: {Path(selected).name}")
            return selected
        return None

    def _get_random_story_video(self) -> Optional[str]:
        """Get a random story video from the list, avoiding repeats.

        Each video is used only once per generation run. When all videos
        have been used, the pool resets. Never picks the same video twice
        in a row.
        """
        valid_videos = [v for v in self.story_videos if Path(v).exists()]
        if not valid_videos:
            return None

        # Get available videos (not yet used in this run)
        available = [v for v in valid_videos if v not in self._used_story_videos]

        # If all videos have been used, reset the pool but exclude the last one
        # to prevent back-to-back repeat
        if not available:
            self._used_story_videos = []
            available = [v for v in valid_videos if v != self._last_story_video]
            # Edge case: only one video exists
            if not available:
                available = valid_videos

        # Select random from available
        selected = random.choice(available)

        # Track usage
        self._used_story_videos.append(selected)
        self._last_story_video = selected

        print(f"[VideoGenerator] Selected story video: {Path(selected).name} ({len(self._used_story_videos)}/{len(valid_videos)} used)")
        return selected

    def _reset_story_video_tracking(self):
        """Reset the story video tracking for a new video generation."""
        self._used_story_videos = []
        self._last_story_video = None

    def _generate_intro_video_segment(self, audio_path: str, output_path: str) -> str:
        """
        Generate intro segment using video file with header overlay.
        Video is muted and audio track is overlaid.
        Header with TOWER NEWS branding and icon is added on top.

        Args:
            audio_path: Path to intro music audio
            output_path: Output path for segment

        Returns:
            Path to generated segment
        """
        audio_duration = self._get_audio_duration(audio_path)

        # Select random intro video
        intro_video = self._get_random_intro_video()
        if not intro_video:
            raise RuntimeError("No intro video available")

        # Layout dimensions
        header_height = int(self.height * 0.06) // 2 * 2  # Even number
        ticker_height = int(self.height * 0.10) // 2 * 2  # Even number
        video_height = self.height - header_height - ticker_height

        # Get channel name from config
        channel_name = self.config.get("channel", {}).get("name", "TOWER NEWS").upper()
        header_font_size = max(40, int(header_height * 0.6))

        # Build inputs list
        inputs = [
            "-i", intro_video,      # Input 0: Video
            "-i", audio_path,       # Input 1: Audio
        ]

        # Check if we have a brand icon
        has_icon = self.brand_icon and Path(self.brand_icon).exists()
        if has_icon:
            inputs.extend(["-i", self.brand_icon])  # Input 2: Icon

        # Calculate icon size (fit in header height with padding)
        icon_size = int(header_height * 0.7)

        # Build filter complex with header overlay
        if has_icon:
            filter_complex = (
                # Scale video to fill video area, crop to fit
                f"[0:v]scale={self.width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{video_height},"
                f"setsar=1,fps={self.fps}[video];"

                # Scale icon to fit in header
                f"[2:v]scale={icon_size}:{icon_size}:force_original_aspect_ratio=decrease[icon];"

                # Create header bar with channel name and LIVE indicator
                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={audio_duration},"
                # Red accent on left
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                # Channel name (shifted right to make room for icon)
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x={icon_size + 40}:"  # After icon
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                # Live indicator dot
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                # "LIVE" text
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header_base];"

                # Overlay icon on header
                f"[header_base][icon]overlay=x=20:y=(H-h)/2[header];"

                # Create ticker placeholder
                f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={audio_duration}[ticker_placeholder];"

                # Stack: header, video, ticker
                f"[header][video][ticker_placeholder]vstack=inputs=3,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
            )
        else:
            filter_complex = (
                # Scale video to fill video area, crop to fit
                f"[0:v]scale={self.width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{video_height},"
                f"setsar=1,fps={self.fps}[video];"

                # Create header bar with channel name and LIVE indicator
                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={audio_duration},"
                # Red accent on left
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                # Channel name
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x=30:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                # Live indicator dot
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                # "LIVE" text
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header];"

                # Create ticker placeholder
                f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={audio_duration}[ticker_placeholder];"

                # Stack: header, video, ticker
                f"[header][video][ticker_placeholder]vstack=inputs=3,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
            )

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",  # Use audio from audio input
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-r", str(self.fps),  # Force correct framerate
            "-c:a", "aac",
            "-ac", "2",           # Force stereo (consistent with other segments)
            "-ar", "44100",       # Consistent sample rate
            "-b:a", "128k",
            "-t", str(audio_duration),  # Match audio duration
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"[VideoGenerator] Generating intro video segment (with header overlay)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] Intro video error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"[VideoGenerator] Intro segment generated: {output_path}")
        return output_path

    def _generate_outro_video_segment(self, audio_path: str, output_path: str) -> str:
        """
        Generate outro segment using video file with header overlay.
        Video is muted and audio track (TTS outro) is overlaid.
        Header with TOWER NEWS branding and icon is added on top.

        Args:
            audio_path: Path to outro audio (TTS)
            output_path: Output path for segment

        Returns:
            Path to generated segment
        """
        audio_duration = self._get_audio_duration(audio_path)

        # Select random outro video
        outro_video = self._get_random_outro_video()
        if not outro_video:
            return None  # Fall back to presenter-based outro

        # Get video duration to calculate start time (use last seconds of video)
        video_duration = self._get_audio_duration(outro_video)  # Works for video too

        # Calculate start time: use the END of the video, not the beginning
        # This ensures the final frames (ending animation) are always shown
        start_time = max(0, video_duration - audio_duration)
        print(f"[VideoGenerator] Outro video: using last {audio_duration:.1f}s (starting at {start_time:.1f}s of {video_duration:.1f}s)")

        # Layout dimensions
        header_height = int(self.height * 0.06) // 2 * 2
        ticker_height = int(self.height * 0.10) // 2 * 2
        video_height = self.height - header_height - ticker_height

        # Get channel name from config
        channel_name = self.config.get("channel", {}).get("name", "TOWER NEWS").upper()
        header_font_size = max(40, int(header_height * 0.6))

        # NOTE: Background music is added at the final step over the entire video
        # Not per-segment, to avoid music restarting at each segment
        has_bg_music = False  # Disabled - music added at final step

        # Build inputs list - seek to start_time in video to use the ending
        inputs = [
            "-ss", str(start_time),  # Seek to near the end of video
            "-i", outro_video,       # Input 0: Video
            "-i", audio_path,        # Input 1: Audio (TTS outro)
        ]

        # Add background music for outro
        bg_music_input_idx = None
        next_input_idx = 2
        if has_bg_music:
            inputs.extend(["-stream_loop", "-1", "-i", self.background_music])  # Input 2: Background music
            bg_music_input_idx = next_input_idx
            next_input_idx += 1

        # Check if we have a brand icon
        has_icon = self.brand_icon and Path(self.brand_icon).exists()
        icon_input_idx = None
        if has_icon:
            icon_input_idx = next_input_idx
            inputs.extend(["-i", self.brand_icon])  # Icon input
            next_input_idx += 1

        icon_size = int(header_height * 0.7)

        # Build filter complex (same as intro but for outro)
        if has_icon:
            video_filter = (
                f"[0:v]scale={self.width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{video_height},"
                f"setsar=1,fps={self.fps}[video];"

                f"[{icon_input_idx}:v]scale={icon_size}:{icon_size}:force_original_aspect_ratio=decrease[icon];"

                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={audio_duration},"
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x={icon_size + 40}:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header_base];"

                f"[header_base][icon]overlay=x=20:y=(H-h)/2[header];"

                f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={audio_duration}[ticker_placeholder];"

                f"[header][video][ticker_placeholder]vstack=inputs=3,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
            )
        else:
            video_filter = (
                f"[0:v]scale={self.width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{video_height},"
                f"setsar=1,fps={self.fps}[video];"

                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={audio_duration},"
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x=30:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header];"

                f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={audio_duration}[ticker_placeholder];"

                f"[header][video][ticker_placeholder]vstack=inputs=3,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
            )

        # Build audio filter - mix TTS with background music if available
        if has_bg_music:
            # Mix TTS (full volume) with background music (reduced volume)
            audio_filter = (
                f"[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[tts];"
                f"[{bg_music_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"volume={self.background_volume}dB[bg];"
                f"[tts][bg]amix=inputs=2:duration=first:dropout_transition=2[a]"
            )
            filter_complex = video_filter + ";" + audio_filter
            audio_map = "[a]"
        else:
            filter_complex = video_filter
            audio_map = "1:a"

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", audio_map,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-r", str(self.fps),  # Force correct framerate
            "-c:a", "aac",
            "-ac", "2",
            "-ar", "44100",
            "-b:a", "128k",
            "-t", str(audio_duration),
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"[VideoGenerator] Generating outro video segment (with header overlay)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] Outro video error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"[VideoGenerator] Outro segment generated: {output_path}")
        return output_path

    def _get_all_available_videos(self) -> list[tuple[str, float]]:
        """Get all available videos (story + greeting) with their durations.

        Returns:
            List of (video_path, duration) tuples
        """
        all_videos = []

        # Add story videos
        for v in self.story_videos:
            if Path(v).exists():
                duration = self._get_audio_duration(v)
                all_videos.append((v, duration))

        # Add greeting videos (kling videos)
        for v in self.greeting_videos:
            if Path(v).exists():
                duration = self._get_audio_duration(v)
                all_videos.append((v, duration))

        return all_videos

    def _select_videos_for_duration(self, target_duration: float, exclude_last: Optional[str] = None) -> list[str]:
        """Select multiple videos to fill a target duration with global tracking.

        Prioritizes videos not yet used in this video generation. Once all videos
        have been used, resets and starts fresh. Speed adjustment (±5%) handles
        duration mismatches.

        Args:
            target_duration: Target duration in seconds
            exclude_last: Video path to exclude (to avoid back-to-back repeat)

        Returns:
            List of video paths to use
        """
        import random

        all_videos = self._get_all_available_videos()
        if not all_videos:
            return []

        selected = []
        remaining_duration = target_duration

        while remaining_duration > 0:
            # Get videos not yet used globally in this video generation
            unused_globally = [(v, d) for v, d in all_videos
                              if v not in self._used_story_videos]

            # Also exclude the last selected video to avoid back-to-back repeat
            last_selected = selected[-1] if selected else exclude_last
            candidates = [(v, d) for v, d in unused_globally if v != last_selected]

            # If no candidates (all used globally), reset tracking and try again
            if not candidates:
                if unused_globally:
                    # Just exclude last selected
                    candidates = unused_globally
                else:
                    # All videos used - reset global tracking
                    self._used_story_videos = []
                    candidates = [(v, d) for v, d in all_videos if v != last_selected]
                    if not candidates:
                        candidates = all_videos  # Edge case: only one video

            if candidates:
                # Pick random video from candidates
                video_path, video_duration = random.choice(candidates)
                selected.append(video_path)
                self._used_story_videos.append(video_path)  # Track globally
                remaining_duration -= video_duration

        return selected

    def _generate_story_video_segment(
        self,
        audio_path: str,
        output_path: str,
        article: Optional[dict] = None
    ) -> Optional[str]:
        """
        Generate story segment using multiple video files concatenated with header overlay.
        Multiple different videos are used to avoid repetitive looping.
        Background music is mixed in. Greenscreen is removed and replaced with stats background.

        Args:
            audio_path: Path to story audio (TTS)
            output_path: Output path for segment
            article: Optional article dict with 'score', 'num_comments', 'post_title'

        Returns:
            Path to generated segment, or None if no video available
        """
        audio_duration = self._get_audio_duration(audio_path)

        # Select videos to fill the duration
        selected_videos = self._select_videos_for_duration(audio_duration, self._last_story_video)
        if not selected_videos:
            return None

        # Track last video used (for next segment)
        self._last_story_video = selected_videos[-1]

        # Get durations of selected videos
        video_durations = [self._get_audio_duration(v) for v in selected_videos]

        # Crossfade settings
        crossfade_duration = 0.5  # seconds of overlap between videos
        num_crossfades = max(0, len(selected_videos) - 1)
        total_crossfade_time = num_crossfades * crossfade_duration

        # Calculate total raw video duration
        total_video_duration = sum(video_durations)

        # After crossfades, effective duration is reduced
        effective_video_duration = total_video_duration - total_crossfade_time

        # Calculate speed adjustment to match audio exactly (limit to 95%-105%)
        if effective_video_duration > 0:
            speed_factor = effective_video_duration / audio_duration
            speed_factor = max(0.95, min(1.05, speed_factor))  # Clamp to ±5%
        else:
            speed_factor = 1.0

        # Adjust trim durations based on speed factor
        # Each video plays at speed_factor, so we use full duration
        trim_durations = video_durations.copy()

        video_names = [f"{Path(v).name}({d:.1f}s)" for v, d in zip(selected_videos, trim_durations)]
        speed_info = f" speed={speed_factor:.2f}x" if speed_factor != 1.0 else ""
        print(f"[VideoGenerator] Selected {len(selected_videos)} videos for {audio_duration:.1f}s: {', '.join(video_names)}{speed_info}")

        # Layout dimensions
        header_height = int(self.height * 0.06) // 2 * 2
        ticker_height = int(self.height * 0.10) // 2 * 2
        video_height = self.height - header_height - ticker_height

        # Get channel name from config
        channel_name = self.config.get("channel", {}).get("name", "TOWER NEWS").upper()
        header_font_size = max(40, int(header_height * 0.6))

        # NOTE: Background music is NOT added per-segment anymore
        # It will be added once at the end over the entire video (after ticker overlay)
        has_bg_music = False  # Disabled - music added at final step

        # Build inputs list - add all selected videos
        inputs = []
        for video_path in selected_videos:
            inputs.extend(["-i", video_path])

        # Add audio input
        audio_input_idx = len(selected_videos)
        inputs.extend(["-i", audio_path])

        # Add background music input (looped)
        bg_music_input_idx = None
        next_input_idx = audio_input_idx + 1
        if has_bg_music:
            inputs.extend(["-stream_loop", "-1", "-i", self.background_music])
            bg_music_input_idx = next_input_idx
            next_input_idx += 1

        # Check if we have a brand icon
        has_icon = self.brand_icon and Path(self.brand_icon).exists()
        icon_input_idx = None
        if has_icon:
            icon_input_idx = next_input_idx
            inputs.extend(["-i", self.brand_icon])
            next_input_idx += 1

        icon_size = int(header_height * 0.7)

        # Build video filter complex - concatenate all videos with crossfade
        filter_parts = []

        # Extract article stats for background text
        if article:
            upvotes = article.get("score", 0)
            comments = article.get("num_comments", 0)
            # Format numbers with K for thousands
            upvotes_str = f"{upvotes/1000:.1f}K" if upvotes >= 1000 else str(upvotes)
            comments_str = f"{comments/1000:.1f}K" if comments >= 1000 else str(comments)
            stats_text = f"▲ {upvotes_str} UPVOTES    💬 {comments_str} COMMENTS"
        else:
            stats_text = ""

        # Scale and prepare each video with speed adjustment
        # NOTE: Chromakey and scrolling stats text disabled for now (kept for future use)
        # To enable: set use_chromakey=True and use_scrolling_stats=True
        use_chromakey = False  # Disabled - videos don't have clean greenscreen
        use_scrolling_stats = False  # Disabled - needs better positioning/styling

        for i in range(len(selected_videos)):
            pts_factor = 1.0 / speed_factor
            if use_chromakey:
                # Apply chromakey to remove greenscreen, then scale
                filter_parts.append(
                    f"[{i}:v]setpts={pts_factor:.4f}*PTS,"
                    f"chromakey=color={self.chroma_key_color}:"
                    f"similarity={self.chroma_similarity}:"
                    f"blend={self.chroma_blend},"
                    f"scale={self.width}:{video_height}:force_original_aspect_ratio=increase,"
                    f"crop={self.width}:{video_height},setsar=1,fps={self.fps},format=yuva420p[v{i}]"
                )
            else:
                # Simple scale without chromakey
                filter_parts.append(
                    f"[{i}:v]setpts={pts_factor:.4f}*PTS,"
                    f"scale={self.width}:{video_height}:force_original_aspect_ratio=increase,"
                    f"crop={self.width}:{video_height},setsar=1,fps={self.fps},format=yuv420p[v{i}]"
                )

        # Apply crossfade between videos (if more than 1 video)
        if len(selected_videos) == 1:
            filter_parts.append(f"[v0]trim=duration={audio_duration},setpts=PTS-STARTPTS[video_concat]")
        else:
            # Chain xfade filters: v0 xfade v1 -> xf0, xf0 xfade v2 -> xf1, etc.
            cumulative_offset = 0
            for i in range(len(selected_videos) - 1):
                input_a = f"[v{i}]" if i == 0 else f"[xf{i-1}]"
                input_b = f"[v{i+1}]"
                output = f"[xf{i}]"

                adjusted_duration = trim_durations[i] / speed_factor
                offset = cumulative_offset + adjusted_duration - crossfade_duration
                cumulative_offset = offset

                filter_parts.append(
                    f"{input_a}{input_b}xfade=transition=fade:duration={crossfade_duration}:offset={offset:.3f}{output}"
                )

            final_xf = f"[xf{len(selected_videos)-2}]"
            filter_parts.append(f"{final_xf}trim=duration={audio_duration},setpts=PTS-STARTPTS[video_concat]")

        # Future: Chromakey background and scrolling stats (disabled for now)
        # When enabled, this would overlay chromakeyed video on dark background with stats text
        # if use_chromakey:
        #     filter_parts.append(f"color=c=0x0d0d1a:s={self.width}x{video_height}:d={audio_duration}[bg_solid]")
        #     filter_parts.append("[bg_solid][video_keyed]overlay=0:0:format=auto[video_with_bg]")
        #     if use_scrolling_stats and stats_text:
        #         # Add scrolling stats text
        #         pass

        # Build header with icon if available
        if has_icon:
            filter_parts.append(
                f"[{icon_input_idx}:v]scale={icon_size}:{icon_size}:force_original_aspect_ratio=decrease[icon]"
            )
            filter_parts.append(
                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={audio_duration},"
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x={icon_size + 40}:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header_base]"
            )
            filter_parts.append("[header_base][icon]overlay=x=20:y=(H-h)/2[header]")
        else:
            filter_parts.append(
                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={audio_duration},"
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x=30:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header]"
            )

        # Ticker placeholder
        filter_parts.append(
            f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={audio_duration}[ticker_placeholder]"
        )

        # Stack header, video, ticker
        header_label = "[header]" if has_icon else "[header]"
        filter_parts.append(
            f"{header_label}[video_concat][ticker_placeholder]vstack=inputs=3,"
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
        )

        # Build audio filter - mix TTS with background music if available
        if has_bg_music:
            filter_parts.append(
                f"[{audio_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[tts]"
            )
            filter_parts.append(
                f"[{bg_music_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"volume={self.background_volume}dB[bg]"
            )
            filter_parts.append("[tts][bg]amix=inputs=2:duration=first:dropout_transition=2[a]")
            audio_map = "[a]"
        else:
            audio_map = f"{audio_input_idx}:a"

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", audio_map,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-r", str(self.fps),  # Force correct framerate
            "-c:a", "aac",
            "-ac", "2",
            "-ar", "44100",
            "-b:a", "128k",
            "-t", str(audio_duration),  # Limit to audio duration
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"[VideoGenerator] Generating story video segment (with header)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] Story video error: {result.stderr}")
            return None  # Fall back to presenter

        print(f"[VideoGenerator] Story segment generated: {output_path}")
        return output_path

    def _generate_intro_greeting_segment(self, audio_path: str, output_path: str) -> Optional[str]:
        """
        Generate intro greeting segment using a short kling video with header overlay.
        This is used for the "Welcome to Tower News" segment after the jingle.
        Background music is mixed in at reduced volume.

        Args:
            audio_path: Path to intro greeting audio (TTS)
            output_path: Output path for segment

        Returns:
            Path to generated segment, or None if no video available
        """
        # Use a short greeting video (~5s kling videos)
        greeting_video = self._get_random_greeting_video()
        if not greeting_video:
            return None

        audio_duration = self._get_audio_duration(audio_path)

        # Layout dimensions
        header_height = int(self.height * 0.06) // 2 * 2
        ticker_height = int(self.height * 0.10) // 2 * 2
        video_height = self.height - header_height - ticker_height

        # Get channel name from config
        channel_name = self.config.get("channel", {}).get("name", "TOWER NEWS").upper()
        header_font_size = max(40, int(header_height * 0.6))

        # NOTE: Background music is added at the final step over the entire video
        # Not per-segment, to avoid music restarting at each segment
        has_bg_music = False  # Disabled - music added at final step

        # Build inputs list - use -stream_loop to loop video if needed
        inputs = [
            "-stream_loop", "-1",   # Loop video infinitely (in case audio is longer)
            "-i", greeting_video,   # Input 0: Video (looped)
            "-i", audio_path,       # Input 1: Audio (TTS)
        ]

        # Background music disabled - the intro jingle provides the music
        bg_music_input_idx = None
        next_input_idx = 2

        # Check if we have a brand icon
        has_icon = self.brand_icon and Path(self.brand_icon).exists()
        icon_input_idx = None
        if has_icon:
            icon_input_idx = next_input_idx
            inputs.extend(["-i", self.brand_icon])  # Icon input
            next_input_idx += 1

        icon_size = int(header_height * 0.7)

        # Build video filter complex
        if has_icon:
            video_filter = (
                f"[0:v]scale={self.width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{video_height},"
                f"setsar=1,fps={self.fps}[video];"

                f"[{icon_input_idx}:v]scale={icon_size}:{icon_size}:force_original_aspect_ratio=decrease[icon];"

                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={audio_duration},"
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x={icon_size + 40}:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header_base];"

                f"[header_base][icon]overlay=x=20:y=(H-h)/2[header];"

                f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={audio_duration}[ticker_placeholder];"

                f"[header][video][ticker_placeholder]vstack=inputs=3,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
            )
        else:
            video_filter = (
                f"[0:v]scale={self.width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{video_height},"
                f"setsar=1,fps={self.fps}[video];"

                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={audio_duration},"
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x=30:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header];"

                f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={audio_duration}[ticker_placeholder];"

                f"[header][video][ticker_placeholder]vstack=inputs=3,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
            )

        # Build audio filter - mix TTS with background music if available
        if has_bg_music:
            # Mix TTS (full volume) with background music (reduced volume)
            audio_filter = (
                f"[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[tts];"
                f"[{bg_music_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"volume={self.background_volume}dB[bg];"
                f"[tts][bg]amix=inputs=2:duration=first:dropout_transition=2[a]"
            )
            filter_complex = video_filter + ";" + audio_filter
            audio_map = "[a]"
        else:
            filter_complex = video_filter
            audio_map = "1:a"

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", audio_map,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-r", str(self.fps),  # Force correct framerate
            "-c:a", "aac",
            "-ac", "2",
            "-ar", "44100",
            "-b:a", "128k",
            "-t", str(audio_duration),  # Limit to audio duration
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"[VideoGenerator] Generating intro greeting segment (video + bg music)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] Intro greeting video error: {result.stderr}")
            return None  # Fall back to presenter

        print(f"[VideoGenerator] Intro greeting segment generated: {output_path}")
        return output_path

    def _generate_fullscreen_presenter_segment(
        self,
        audio_path: str,
        presenter_image: str,
        output_path: str,
        duration: float,
        with_background_music: bool = False
    ) -> str:
        """
        Generate segment with presenter image as fullscreen (no B-roll).
        The presenter image is already in 9:16 format and used as-is.
        Includes header with TOWER NEWS branding and ticker placeholder at bottom.

        Args:
            audio_path: Path to audio file
            presenter_image: Path to presenter image (9:16 format)
            output_path: Output path for segment
            duration: Duration in seconds
            with_background_music: Whether to mix in background music

        Returns:
            Path to generated segment
        """
        # Layout: Header (6%), Presenter (84%), Ticker placeholder (10%)
        # Ensure all heights are even numbers (required by libx264)
        header_height = int(self.height * 0.06) // 2 * 2  # Round down to even
        ticker_height = int(self.height * 0.10) // 2 * 2  # Round down to even
        presenter_height = self.height - header_height - ticker_height  # Remaining fills the gap

        # Get channel name from config
        channel_name = self.config.get("channel", {}).get("name", "TOWER NEWS").upper()
        header_font_size = max(40, int(header_height * 0.6))

        # Check if we have a brand icon
        has_icon = self.brand_icon and Path(self.brand_icon).exists()
        icon_size = int(header_height * 0.7)

        # Check if we have background music
        has_bg_music = with_background_music and self.background_music and Path(self.background_music).exists()

        # Build inputs
        inputs = [
            "-loop", "1",
            "-t", str(duration),
            "-i", presenter_image,   # Input 0: Presenter
            "-i", audio_path,        # Input 1: Audio
        ]

        # Track input indices
        next_input_idx = 2

        # Add background music input (looped)
        bg_music_input_idx = None
        if has_bg_music:
            inputs.extend(["-stream_loop", "-1", "-i", self.background_music])  # Background music
            bg_music_input_idx = next_input_idx
            next_input_idx += 1

        # Add icon input
        icon_input_idx = None
        if has_icon:
            inputs.extend(["-i", self.brand_icon])
            icon_input_idx = next_input_idx
            next_input_idx += 1

        # Build video filter complex with header overlay
        if has_icon:
            video_filter = (
                # Scale presenter image to fit presenter area
                f"[0:v]scale={self.width}:{presenter_height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{presenter_height},"
                f"setsar=1,fps={self.fps}[presenter];"

                # Scale icon to fit in header
                f"[{icon_input_idx}:v]scale={icon_size}:{icon_size}:force_original_aspect_ratio=decrease[icon];"

                # Create header bar with channel name and LIVE indicator
                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={duration},"
                # Red accent on left
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                # Channel name (shifted right for icon)
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x={icon_size + 40}:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                # Live indicator dot
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                # "LIVE" text
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header_base];"

                # Overlay icon on header
                f"[header_base][icon]overlay=x=20:y=(H-h)/2[header];"

                # Create ticker placeholder (solid dark bar)
                f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={duration}[ticker_placeholder];"

                # Stack: header on top, presenter in middle, ticker placeholder at bottom
                f"[header][presenter][ticker_placeholder]vstack=inputs=3,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
            )
        else:
            video_filter = (
                # Scale presenter image to fit presenter area
                f"[0:v]scale={self.width}:{presenter_height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{presenter_height},"
                f"setsar=1,fps={self.fps}[presenter];"

                # Create header bar with channel name and LIVE indicator
                f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={duration},"
                # Red accent on left
                f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
                # Channel name
                f"drawtext=text='{channel_name}':"
                f"fontsize={header_font_size}:"
                f"fontcolor=white:"
                f"x=30:"
                f"y=(h-{header_font_size})/2:"
                f"font=Impact,"
                # Live indicator dot
                f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
                # "LIVE" text
                f"drawtext=text='LIVE':"
                f"fontsize={int(header_font_size * 0.6)}:"
                f"fontcolor=white:"
                f"x={self.width - 90}:"
                f"y=(h-{int(header_font_size * 0.6)})/2:"
                f"font=Arial[header];"

                # Create ticker placeholder (solid dark bar)
                f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={duration}[ticker_placeholder];"

                # Stack: header on top, presenter in middle, ticker placeholder at bottom
                f"[header][presenter][ticker_placeholder]vstack=inputs=3,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=disable[v]"
            )

        # Build audio filter - mix TTS with background music if available
        if has_bg_music:
            # Mix TTS (full volume) with background music (reduced volume)
            audio_filter = (
                f"[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[tts];"
                f"[{bg_music_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"volume={self.background_volume}dB[bg];"
                f"[tts][bg]amix=inputs=2:duration=first:dropout_transition=2[a]"
            )
            filter_complex = video_filter + ";" + audio_filter
            audio_map = "[a]"
        else:
            filter_complex = video_filter
            audio_map = "1:a"

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", audio_map,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-r", str(self.fps),  # Force correct framerate
            "-c:a", "aac",
            "-ac", "2",           # Force stereo (consistent with other segments)
            "-ar", "44100",       # Consistent sample rate
            "-b:a", "128k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"[VideoGenerator] Generating fullscreen presenter segment (with header, bg_music={has_bg_music})...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] Fullscreen presenter error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"[VideoGenerator] Fullscreen presenter segment: {output_path}")
        return output_path

    def _create_ken_burns_filter(
        self,
        duration: float,
        zoom_start: float = 1.0,
        zoom_end: float = 1.1
    ) -> str:
        """
        Create Ken Burns zoom filter for FFmpeg.

        Args:
            duration: Duration of the effect in seconds
            zoom_start: Starting zoom level
            zoom_end: Ending zoom level

        Returns:
            FFmpeg filter string
        """
        # Calculate zoom per frame
        frames = int(duration * self.fps)
        zoom_per_frame = (zoom_end - zoom_start) / frames

        return (
            f"zoompan=z='min(zoom+{zoom_per_frame:.8f},{zoom_end})':"
            f"d={frames}:"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"s={self.width}x{int(self.height * 0.55)}"
        )

    def _create_ticker_filter(self, text: str, duration: float) -> str:
        """
        Create scrolling news ticker filter.

        Args:
            text: Ticker text
            duration: Duration in seconds

        Returns:
            FFmpeg drawtext filter string
        """
        # Escape special characters for FFmpeg
        escaped_text = text.replace("'", "'\\''").replace(":", "\\:")

        return (
            f"drawtext=text='{escaped_text}':"
            f"fontsize=36:"
            f"fontcolor=white:"
            f"x=w-mod(t*{self.ticker_speed}\\,w+tw):"
            f"y=h-60:"
            f"font=Arial"
        )

    def _create_chromakey_filter(self, input_label: str, output_label: str) -> str:
        """
        Create FFmpeg chromakey filter to remove greenscreen.

        Args:
            input_label: Input stream label
            output_label: Output stream label

        Returns:
            FFmpeg chromakey filter string
        """
        return (
            f"[{input_label}]chromakey=color={self.chroma_key_color}:"
            f"similarity={self.chroma_similarity}:"
            f"blend={self.chroma_blend}[{output_label}]"
        )

    def generate_video(
        self,
        audio_path: str,
        screenshots: list[str],
        ticker_text: str,
        presenter_image: Optional[str] = None,
        output_path: Optional[str] = None,
        use_greenscreen: bool = True,
        segment_durations: Optional[list[float]] = None,
        intro_duration: float = 0.0,
        articles: Optional[list[dict]] = None
    ) -> str:
        """
        Generate the final news video with presenter overlay and greenscreen keying.

        Layout:
        - Background: Screenshots with Ken Burns effect (full screen behind presenter)
        - Overlay: Presenter with greenscreen removed (bottom portion)
        - Foreground: News ticker scrolling at bottom

        Args:
            audio_path: Path to the audio file
            screenshots: List of screenshot paths
            ticker_text: Text for the scrolling ticker
            presenter_image: Path to presenter image (optional)
            output_path: Path for output video (optional)
            use_greenscreen: Whether to apply chromakey to presenter (default True)
            segment_durations: Optional list of durations for each screenshot (in seconds)
            intro_duration: Duration of intro in seconds (images show first image during intro)
            articles: Optional list of article dicts with stats (score, num_comments, post_title)

        Returns:
            Path to generated video
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not in PATH")

        # Create output directory
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_dir / today
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = str(output_dir / f"news_{timestamp}.mp4")

        presenter_image = presenter_image or self.presenter_image

        # Get audio duration
        duration = self._get_audio_duration(audio_path)

        # Calculate segment durations
        num_screenshots = len(screenshots)
        if num_screenshots == 0:
            raise ValueError("At least one screenshot is required")

        # Use provided segment durations or divide equally
        if segment_durations and len(segment_durations) == num_screenshots:
            # Use exact durations from TTS - these are the real story lengths
            # Add intro_duration to the FIRST image so it shows during the intro
            scaled_durations = segment_durations.copy()
            if intro_duration > 0:
                scaled_durations[0] = scaled_durations[0] + intro_duration
                print(f"[VideoGenerator] First image extended by {intro_duration:.1f}s for intro")

            # Verify total doesn't exceed audio duration
            total_story_time = sum(scaled_durations)
            if total_story_time > duration:
                # Scale down proportionally if stories are longer than audio
                scale_factor = duration / total_story_time
                scaled_durations = [d * scale_factor for d in scaled_durations]
        else:
            # Fallback: divide equally
            scaled_durations = [duration / num_screenshots] * num_screenshots

        # Build inputs
        inputs = []
        input_idx = 0

        # Add presenter image input (will be chromakeyed and overlaid)
        presenter_idx = None
        if Path(presenter_image).exists():
            inputs.extend(["-loop", "1", "-t", str(duration), "-i", presenter_image])
            presenter_idx = input_idx
            input_idx += 1

        # Add screenshot inputs with loop (using individual durations)
        screenshot_start_idx = input_idx
        for i, screenshot in enumerate(screenshots):
            seg_dur = scaled_durations[i]
            inputs.extend(["-loop", "1", "-t", str(seg_dur), "-i", screenshot])
            input_idx += 1

        # Add audio input
        inputs.extend(["-i", audio_path])
        audio_idx = input_idx

        # Build filter graph
        filter_complex = []

        # Layout: Header (6%), Image (46%), Presenter (38%), Ticker (10%)
        header_height = int(self.height * 0.06)  # 6% for header
        ticker_height = int(self.height * 0.10)  # 10% for ticker (more prominent)
        presenter_height = int(self.height * 0.38)  # 38% for presenter
        image_height = self.height - header_height - ticker_height - presenter_height  # ~46% for image

        # Process each screenshot - scale to fit WITHOUT cropping (preserve full image with text)
        concat_inputs = []
        for i, screenshot in enumerate(screenshots):
            idx = screenshot_start_idx + i
            seg_dur = scaled_durations[i]
            # Scale to fit in image area, pad with dark background if needed
            filter_complex.append(
                f"[{idx}:v]scale={self.width}:{image_height}:force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{image_height}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,"
                f"setsar=1,fps={self.fps},trim=duration={seg_dur},setpts=PTS-STARTPTS[bg{i}]"
            )
            concat_inputs.append(f"[bg{i}]")
            print(f"[VideoGenerator] Screenshot {i+1}: {seg_dur:.1f}s")

        # Concatenate all background screenshots
        filter_complex.append(
            f"{''.join(concat_inputs)}concat=n={len(screenshots)}:v=1:a=0[images]"
        )

        # Get channel name from config
        channel_name = self.config.get("channel", {}).get("name", "TOWER NEWS").upper()
        header_font_size = max(40, int(header_height * 0.6))

        # Create header bar with channel name
        filter_complex.append(
            f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={duration},"
            # Red accent on left
            f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
            # Channel name
            f"drawtext=text='{channel_name}':"
            f"fontsize={header_font_size}:"
            f"fontcolor=white:"
            f"x=30:"
            f"y=(h-{header_font_size})/2:"
            f"font=Impact,"
            # Live indicator dot (pulsing effect via modulo)
            f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
            # "LIVE" text
            f"drawtext=text='LIVE':"
            f"fontsize={int(header_font_size * 0.6)}:"
            f"fontcolor=white:"
            f"x={self.width - 90}:"
            f"y=(h-{int(header_font_size * 0.6)})/2:"
            f"font=Arial[header]"
        )

        # Process presenter with greenscreen removal and dynamic background
        if presenter_idx is not None:
            # Create dynamic background for presenter area (replaces greenscreen)
            # Dark gradient with scrolling headlines and stats

            # Build stats text from articles
            stats_text = ""
            if articles:
                stats_parts = []
                for i, article in enumerate(articles):
                    title = article.get("post_title", f"Story {i+1}")[:50]
                    score = article.get("score", 0)
                    comments = article.get("num_comments", 0)
                    stats_parts.append(f"#{i+1} {title} | {score} upvotes | {comments} comments")
                stats_text = "     >>>     ".join(stats_parts)
            else:
                stats_text = "Tower News - Your Daily Gaming Update - Stay Informed - Stay Ahead"

            # Escape for FFmpeg
            escaped_stats = stats_text.replace("'", "'\\''").replace(":", "\\:")

            # Stats font size
            stats_font_size = max(28, int(presenter_height * 0.06))
            stats_scroll_speed = 40  # Slower than main ticker

            # Create presenter background with gradient effect and scrolling stats
            # The text should appear in the GREENSCREEN AREA (upper part where green is)
            # The greenscreen is roughly in the upper 40% of the presenter image
            # The greenscreen/background area is roughly 20-45% from top of presenter section
            # (below the curved header bar, above the presenter's head)
            text_y_top = int(presenter_height * 0.22)  # First row
            text_y_middle = int(presenter_height * 0.32)  # Second row

            filter_complex.append(
                f"color=c=0x1a1a2e:s={self.width}x{presenter_height}:d={duration},"
                # Add subtle gradient overlay (darker at edges)
                f"drawbox=x=0:y=0:w={self.width}:h=60:color=0x000000@0.3:t=fill,"
                # Add glowing border at top
                f"drawbox=x=0:y=0:w={self.width}:h=4:color=0xe63946:t=fill,"
                # Add scrolling headlines/stats - TWO ROWS, very dim so they don't interfere
                f"drawtext=text='{escaped_stats}':"
                f"fontsize={stats_font_size}:"
                f"fontcolor=0x2a3a4a@0.4:"  # Very dim, 40% opacity
                f"x=w-mod(t*{stats_scroll_speed}\\,w+tw):"
                f"y={text_y_top}:"
                f"font=Impact,"
                # Second row scrolling opposite direction
                f"drawtext=text='{escaped_stats}':"
                f"fontsize={stats_font_size}:"
                f"fontcolor=0x2a3a4a@0.3:"  # Even dimmer, 30% opacity
                f"x=-w+mod(t*{stats_scroll_speed * 0.8}\\,w+tw):"
                f"y={text_y_middle}:"
                f"font=Impact[presenter_bg]"
            )

            if use_greenscreen:
                # Apply chromakey to remove greenscreen, then scale
                filter_complex.append(
                    f"[{presenter_idx}:v]chromakey=color={self.chroma_key_color}:"
                    f"similarity={self.chroma_similarity}:"
                    f"blend={self.chroma_blend},"
                    f"scale={self.width}:{presenter_height}:force_original_aspect_ratio=decrease,"
                    f"pad={self.width}:{presenter_height}:(ow-iw)/2:0:color=0x000000@0,"
                    f"format=rgba[presenter_fg]"
                )
                # Overlay presenter on the dynamic background
                filter_complex.append(
                    f"[presenter_bg][presenter_fg]overlay=0:0:format=auto[presenter]"
                )
            else:
                filter_complex.append(
                    f"[{presenter_idx}:v]scale={self.width}:{presenter_height}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={self.width}:{presenter_height}:(ow-iw)/2:0:color=0x1a1a2e,"
                    f"format=rgba[presenter]"
                )

            # Stack: Header on top, then images, then presenter below
            filter_complex.append(
                f"[header][images][presenter]vstack=inputs=3[content]"
            )
            content_output = "content"
        else:
            filter_complex.append(
                f"[header][images]vstack=inputs=2,"
                f"pad={self.width}:{self.height - ticker_height}:0:0:color=0x1a1a2e[content]"
            )
            content_output = "content"

        # Create ticker bar with scrolling text - Professional news style
        # Escape text for FFmpeg - need to handle special chars carefully
        escaped_ticker = ticker_text
        escaped_ticker = escaped_ticker.replace("\\", "")  # Remove backslashes
        escaped_ticker = escaped_ticker.replace("'", "")   # Remove single quotes
        escaped_ticker = escaped_ticker.replace('"', "")   # Remove double quotes
        escaped_ticker = escaped_ticker.replace(":", " -") # Replace colons
        escaped_ticker = escaped_ticker.replace("%", "percent")  # % is special in FFmpeg
        escaped_ticker = escaped_ticker.replace("[", "(")  # Brackets are special
        escaped_ticker = escaped_ticker.replace("]", ")")

        print(f"[VideoGenerator] Ticker text: {escaped_ticker[:80]}...")

        # Use "BREAKING" or "LATEST" prefix
        ticker_prefix = "BREAKING"

        # Calculate ticker font size (responsive to ticker height)
        ticker_font_size = max(32, int(ticker_height * 0.35))  # Smaller for better fit
        badge_font_size = max(36, int(ticker_height * 0.38))  # Badge font
        ticker_prefix_width = 400  # Width reserved for prefix badge (even larger)
        scroll_area_width = self.width - ticker_prefix_width  # Scroll area

        # Create scrolling text area (separate, will be combined with badge)
        filter_complex.append(
            f"color=c=0x0d0d1a:s={scroll_area_width}x{ticker_height}:d={duration},"
            f"drawtext=text='{escaped_ticker}':"
            f"fontsize={ticker_font_size}:"
            f"fontcolor=white:"
            f"x=w-mod(t*{self.ticker_speed}\\,w+tw):"
            f"y=(h-{ticker_font_size})/2:"
            f"font=Arial[ticker_scroll]"
        )

        # Create BREAKING badge (static left part) - with enough space
        filter_complex.append(
            f"color=c=0xe63946:s={ticker_prefix_width}x{ticker_height}:d={duration},"
            # Red accent bar on left edge
            f"drawbox=x=0:y=0:w=8:h={ticker_height}:color=0xcc0000:t=fill,"
            # "BREAKING" text - positioned with padding from left
            f"drawtext=text='{ticker_prefix}':"
            f"fontsize={badge_font_size}:"
            f"fontcolor=white:"
            f"x=20:"
            f"y=(h-{badge_font_size})/2:"
            f"font=Arial[ticker_badge]"
        )

        # Combine badge and scroll area horizontally
        filter_complex.append(
            f"[ticker_badge][ticker_scroll]hstack=inputs=2[ticker]"
        )

        # Stack content and ticker vertically
        filter_complex.append(
            f"[{content_output}][ticker]vstack=inputs=2[video]"
        )

        # Combine filter complex
        filter_string = ";".join(filter_complex)

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            *inputs,
            "-filter_complex", filter_string,
            "-map", "[video]",
            "-map", f"{audio_idx}:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-pix_fmt", "yuv420p",  # Ensure compatibility
            output_path
        ]

        print(f"[VideoGenerator] Running FFmpeg with {len(screenshots)} screenshots...")

        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"[VideoGenerator] Video generated: {output_path}")
        return output_path

    def generate_simple_video(
        self,
        audio_path: str,
        background_image: str,
        ticker_text: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a simple video with single background and ticker.

        Args:
            audio_path: Path to audio file
            background_image: Path to background image
            ticker_text: Text for scrolling ticker
            output_path: Path for output video

        Returns:
            Path to generated video
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not in PATH")

        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_dir / today
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = str(output_dir / f"news_simple_{timestamp}.mp4")

        duration = self._get_audio_duration(audio_path)
        ticker_filter = self._create_ticker_filter(ticker_text, duration)

        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-t", str(duration),
            "-i", background_image,
            "-i", audio_path,
            "-filter_complex",
            f"[0:v]scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,"
            f"{ticker_filter}[video]",
            "-map", "[video]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        return output_path

    def generate_presenter_video(
        self,
        audio_path: str,
        presenter_image: str,
        ticker_text: str,
        background_color: str = "0x1a1a2e",
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate video with presenter (greenscreen removed) on solid background.

        This is used when no screenshots are available - creates a clean
        news presenter look with solid colored background.

        Args:
            audio_path: Path to audio file
            presenter_image: Path to presenter image with greenscreen
            ticker_text: Text for scrolling ticker
            background_color: Hex color for background (default dark blue)
            output_path: Path for output video

        Returns:
            Path to generated video
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not in PATH")

        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_dir / today
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = str(output_dir / f"news_{timestamp}.mp4")

        duration = self._get_audio_duration(audio_path)

        # Calculate dimensions
        ticker_height = int(self.height * 0.10)
        content_height = self.height - ticker_height

        # Escape text for FFmpeg - need to handle special chars carefully
        escaped_ticker = ticker_text
        escaped_ticker = escaped_ticker.replace("\\", "")  # Remove backslashes
        escaped_ticker = escaped_ticker.replace("'", "")   # Remove single quotes
        escaped_ticker = escaped_ticker.replace('"', "")   # Remove double quotes
        escaped_ticker = escaped_ticker.replace(":", " -") # Replace colons
        escaped_ticker = escaped_ticker.replace("%", "percent")  # % is special in FFmpeg
        escaped_ticker = escaped_ticker.replace("[", "(")  # Brackets are special
        escaped_ticker = escaped_ticker.replace("]", ")")

        print(f"[VideoGenerator] Presenter ticker text: {escaped_ticker[:80]}...")

        # Build filter complex:
        # 1. Create animated background with zoom effect
        # 2. Apply chromakey to presenter and overlay
        # 3. Add ticker with BREAKING badge at bottom

        # Ticker dimensions
        ticker_font_size = max(32, int(ticker_height * 0.35))
        badge_font_size = max(36, int(ticker_height * 0.38))
        ticker_prefix_width = 300
        scroll_area_width = self.width - ticker_prefix_width

        filter_complex = (
            # Create gradient background with visual interest
            f"color=c={background_color}:s={self.width}x{content_height}:d={duration},"
            # Add visual interest with gradient boxes
            f"drawbox=x=0:y=0:w={self.width}:h={int(content_height*0.25)}:color=0x252540@0.4:t=fill,"
            f"drawbox=x=0:y={int(content_height*0.75)}:w={self.width}:h={int(content_height*0.25)}:color=0x0d0d1a@0.4:t=fill,"
            # Add subtle glow in center
            f"drawbox=x={int(self.width*0.2)}:y={int(content_height*0.2)}:w={int(self.width*0.6)}:h={int(content_height*0.6)}:color=0x3a4a6a@0.15:t=fill[bg];"
            # Process presenter: chromakey + scale
            f"[0:v]chromakey=color={self.chroma_key_color}:"
            f"similarity={self.chroma_similarity}:"
            f"blend={self.chroma_blend},"
            f"scale={self.width}:{content_height}:force_original_aspect_ratio=decrease,"
            f"format=rgba[presenter];"
            # Overlay presenter on animated background (centered)
            f"[bg][presenter]overlay=x=(W-w)/2:y=(H-h)/2:format=auto[content];"
            # Create BREAKING badge
            f"color=c=0xe63946:s={ticker_prefix_width}x{ticker_height}:d={duration},"
            f"drawbox=x=0:y=0:w=8:h={ticker_height}:color=0xcc0000:t=fill,"
            f"drawtext=text='BREAKING':"
            f"fontsize={badge_font_size}:"
            f"fontcolor=white:"
            f"x=20:"
            f"y=(h-{badge_font_size})/2:"
            f"font=Arial[ticker_badge];"
            # Create scrolling text area
            f"color=c=0x0d0d1a:s={scroll_area_width}x{ticker_height}:d={duration},"
            f"drawtext=text='{escaped_ticker}':"
            f"fontsize={ticker_font_size}:"
            f"fontcolor=white:"
            f"x=w-mod(t*{self.ticker_speed}\\,w+tw):"
            f"y=(h-{ticker_font_size})/2:"
            f"font=Arial[ticker_scroll];"
            # Combine badge and scroll into ticker
            f"[ticker_badge][ticker_scroll]hstack=inputs=2[ticker];"
            # Stack content and ticker
            f"[content][ticker]vstack=inputs=2[video]"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-t", str(duration),
            "-i", presenter_image,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[video]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"[VideoGenerator] Generating presenter video with chromakey...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"[VideoGenerator] Video generated: {output_path}")
        return output_path

    def generate_segment_video(
        self,
        audio_path: str,
        screenshot: Optional[str],
        segment_type: str,
        presenter_image: Optional[str] = None,
        output_path: Optional[str] = None,
        use_greenscreen: bool = True,
        article: Optional[dict] = None
    ) -> str:
        """
        Generate a single video segment with its own audio (WITHOUT ticker).

        Each segment (music, intro, story, outro) is rendered separately
        with exact timing from its audio file. Ticker is added later
        over the final concatenated video to ensure continuous scrolling.

        Args:
            audio_path: Path to audio file for this segment
            screenshot: Path to screenshot (None for intro/outro shows first/last)
            segment_type: Type of segment (music, intro, story, outro)
            presenter_image: Path to presenter image
            output_path: Output path for this segment
            use_greenscreen: Whether to apply chromakey
            article: Article dict for stats display

        Returns:
            Path to generated segment video
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not in PATH")

        # Create output directory
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_dir / today / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%H%M%S%f")
            output_path = str(output_dir / f"segment_{segment_type}_{timestamp}.mp4")

        presenter_image = presenter_image or self.presenter_image
        duration = self._get_audio_duration(audio_path)

        # Special handling for "music" segment with intro video (jingle)
        if segment_type == "music" and self.intro_videos:
            return self._generate_intro_video_segment(audio_path, output_path)

        # Special handling for "intro" segment (greeting after jingle) with short kling video + bg music
        if segment_type == "intro" and self.greeting_videos:
            result = self._generate_intro_greeting_segment(audio_path, output_path)
            if result:
                return result
            # Fall through to fullscreen presenter if greeting video fails

        # Special handling for "outro" segment with outro video
        if segment_type == "outro" and self.outro_videos:
            result = self._generate_outro_video_segment(audio_path, output_path)
            if result:
                return result
            # Fall through to presenter-based outro if no video available

        # Check if we have a real screenshot/B-roll (not fallback)
        has_real_image = screenshot and Path(screenshot).exists()

        # If no real B-roll image for story segments, try story video first
        if not has_real_image and segment_type == "story" and self.story_videos:
            result = self._generate_story_video_segment(audio_path, output_path, article=article)
            if result:
                return result
            # Fall through to presenter if story video fails

        # If no real B-roll image, use fullscreen presenter (with studio background)
        if not has_real_image and self.presenter_fullscreen and Path(self.presenter_fullscreen).exists():
            return self._generate_fullscreen_presenter_segment(
                audio_path, self.presenter_fullscreen, output_path, duration
            )

        # Standard layout with B-roll: Header (6%), Image (46%), Presenter (38%), Ticker area (10%)
        header_height = int(self.height * 0.06)
        ticker_height = int(self.height * 0.10)  # Reserved for ticker overlay later
        presenter_height = int(self.height * 0.38)
        image_height = self.height - header_height - ticker_height - presenter_height

        # Build inputs
        inputs = []
        input_idx = 0

        # Add presenter image if available
        presenter_idx = None
        if presenter_image and Path(presenter_image).exists():
            inputs.extend(["-loop", "1", "-t", str(duration), "-i", presenter_image])
            presenter_idx = input_idx
            input_idx += 1

        # Add screenshot (B-roll)
        screenshot_idx = None
        if has_real_image:
            inputs.extend(["-loop", "1", "-t", str(duration), "-i", screenshot])
            screenshot_idx = input_idx
            input_idx += 1

        # Add audio
        inputs.extend(["-i", audio_path])
        audio_idx = input_idx

        # Build filter graph
        filter_complex = []

        # Create image layer from B-roll
        if screenshot_idx is not None:
            filter_complex.append(
                f"[{screenshot_idx}:v]scale={self.width}:{image_height}:force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{image_height}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,"
                f"setsar=1,fps={self.fps}[images]"
            )
        else:
            # Fallback gradient (shouldn't happen with new logic, but just in case)
            filter_complex.append(
                f"color=c=0x1a1a2e:s={self.width}x{image_height}:d={duration},"
                f"fps={self.fps}[images]"
            )

        # Header
        channel_name = self.config.get("channel", {}).get("name", "TOWER NEWS").upper()
        header_font_size = max(40, int(header_height * 0.6))

        filter_complex.append(
            f"color=c=0x0d0d1a:s={self.width}x{header_height}:d={duration},"
            f"drawbox=x=0:y=0:w=10:h={header_height}:color=0xe63946:t=fill,"
            f"drawtext=text='{channel_name}':"
            f"fontsize={header_font_size}:"
            f"fontcolor=white:"
            f"x=30:"
            f"y=(h-{header_font_size})/2:"
            f"font=Impact,"
            f"drawbox=x={self.width - 120}:y={int(header_height * 0.3)}:w=20:h=20:color=0xe63946:t=fill,"
            f"drawtext=text='LIVE':"
            f"fontsize={int(header_font_size * 0.6)}:"
            f"fontcolor=white:"
            f"x={self.width - 90}:"
            f"y=(h-{int(header_font_size * 0.6)})/2:"
            f"font=Arial[header]"
        )

        # Presenter with greenscreen
        if presenter_idx is not None:
            # Stats text
            stats_text = "Tower News - Your Daily Gaming Update"
            if article:
                title = article.get("post_title", "Story")[:50]
                score = article.get("score", 0)
                comments = article.get("num_comments", 0)
                stats_text = f"{title} | {score} upvotes | {comments} comments"

            escaped_stats = stats_text.replace("'", "'\\''").replace(":", "\\:")
            stats_font_size = max(28, int(presenter_height * 0.06))
            stats_scroll_speed = 40
            text_y_top = int(presenter_height * 0.22)
            text_y_middle = int(presenter_height * 0.32)

            filter_complex.append(
                f"color=c=0x1a1a2e:s={self.width}x{presenter_height}:d={duration},"
                f"drawbox=x=0:y=0:w={self.width}:h=60:color=0x000000@0.3:t=fill,"
                f"drawbox=x=0:y=0:w={self.width}:h=4:color=0xe63946:t=fill,"
                f"drawtext=text='{escaped_stats}':"
                f"fontsize={stats_font_size}:"
                f"fontcolor=0x2a3a4a@0.4:"
                f"x=w-mod(t*{stats_scroll_speed}\\,w+tw):"
                f"y={text_y_top}:"
                f"font=Impact,"
                f"drawtext=text='{escaped_stats}':"
                f"fontsize={stats_font_size}:"
                f"fontcolor=0x2a3a4a@0.3:"
                f"x=-w+mod(t*{stats_scroll_speed * 0.8}\\,w+tw):"
                f"y={text_y_middle}:"
                f"font=Impact[presenter_bg]"
            )

            if use_greenscreen:
                filter_complex.append(
                    f"[{presenter_idx}:v]chromakey=color={self.chroma_key_color}:"
                    f"similarity={self.chroma_similarity}:"
                    f"blend={self.chroma_blend},"
                    f"scale={self.width}:{presenter_height}:force_original_aspect_ratio=decrease,"
                    f"pad={self.width}:{presenter_height}:(ow-iw)/2:0:color=0x000000@0,"
                    f"format=rgba[presenter_fg]"
                )
                filter_complex.append(
                    f"[presenter_bg][presenter_fg]overlay=0:0:format=auto[presenter]"
                )
            else:
                filter_complex.append(
                    f"[{presenter_idx}:v]scale={self.width}:{presenter_height}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={self.width}:{presenter_height}:(ow-iw)/2:0:color=0x1a1a2e,"
                    f"format=rgba[presenter]"
                )

            filter_complex.append(
                f"[header][images][presenter]vstack=inputs=3[content]"
            )
        else:
            filter_complex.append(
                f"[header][images]vstack=inputs=2,"
                f"pad={self.width}:{self.height - ticker_height}:0:0:color=0x1a1a2e[content]"
            )

        # Create empty ticker placeholder (solid color bar)
        # The actual scrolling ticker will be added over the final concatenated video
        filter_complex.append(
            f"color=c=0x0d0d1a:s={self.width}x{ticker_height}:d={duration}[ticker_placeholder]"
        )

        filter_complex.append(
            f"[content][ticker_placeholder]vstack=inputs=2[video]"
        )

        filter_string = ";".join(filter_complex)

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_string,
            "-map", "[video]",
            "-map", f"{audio_idx}:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-ac", "2",           # Force stereo output (fixes mono/stereo concat issues)
            "-ar", "44100",       # Consistent sample rate
            "-b:a", "128k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] Segment FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        print(f"[VideoGenerator] Segment generated: {segment_type} ({duration:.1f}s)")
        return output_path

    def concatenate_videos(
        self,
        video_paths: list[str],
        output_path: Optional[str] = None
    ) -> str:
        """
        Concatenate multiple video segments into one final video.

        Args:
            video_paths: List of video file paths to concatenate
            output_path: Output path for final video

        Returns:
            Path to concatenated video
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not in PATH")

        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_dir / today
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = str(output_dir / f"news_{timestamp}.mp4")

        # Create file list for FFmpeg concat
        file_list_path = str(output_dir / "segments" / f"concat_list_{datetime.now().strftime('%H%M%S')}.txt")
        Path(file_list_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_list_path, "w") as f:
            for video_path in video_paths:
                abs_path = str(Path(video_path).absolute()).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

        # Calculate total audio duration from segments
        total_audio_duration = sum(self._get_audio_duration(v) for v in video_paths)
        print(f"[VideoGenerator] Expected duration: {total_audio_duration:.1f}s")

        # Re-encode with explicit duration limit and framerate
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", file_list_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-r", str(self.fps),  # Force correct framerate
            "-c:a", "aac",
            "-ac", "2",
            "-ar", "44100",
            "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-t", str(total_audio_duration),  # Explicit duration limit
            output_path
        ]

        print(f"[VideoGenerator] Concatenating {len(video_paths)} segments...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] Concat error: {result.stderr}")
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        print(f"[VideoGenerator] Concatenated video: {output_path}")
        return output_path

    def add_ticker_overlay(
        self,
        video_path: str,
        ticker_text: str,
        output_path: Optional[str] = None,
        music_start_time: float = 0.0
    ) -> str:
        """
        Add scrolling ticker overlay and background music to an existing video.

        This is called after concatenating segments to ensure continuous scrolling
        and continuous background music (not restarting per segment).

        Args:
            video_path: Path to input video
            ticker_text: Text for scrolling ticker
            output_path: Path for output video
            music_start_time: When to start background music (in seconds, after jingle)

        Returns:
            Path to video with ticker overlay
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not in PATH")

        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_dir / today
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = str(output_dir / f"news_{timestamp}.mp4")

        # Get video duration
        duration = self._get_audio_duration(video_path)

        # Check if we have background music
        has_bg_music = self.background_music and Path(self.background_music).exists()

        # Ticker dimensions
        ticker_height = int(self.height * 0.10)
        ticker_y = self.height - ticker_height  # Position at bottom

        # Escape text for FFmpeg - need to handle special chars carefully
        # Remove characters that cause FFmpeg filter parsing issues
        escaped_ticker = ticker_text
        escaped_ticker = escaped_ticker.replace("\\", "")  # Remove backslashes
        escaped_ticker = escaped_ticker.replace("'", "")   # Remove single quotes
        escaped_ticker = escaped_ticker.replace('"', "")   # Remove double quotes
        escaped_ticker = escaped_ticker.replace(":", " -") # Replace colons
        escaped_ticker = escaped_ticker.replace("%", "percent")  # % is special in FFmpeg
        escaped_ticker = escaped_ticker.replace("[", "(")  # Brackets are special
        escaped_ticker = escaped_ticker.replace("]", ")")
        ticker_prefix = "BREAKING"

        print(f"[VideoGenerator] Ticker text for overlay: {escaped_ticker[:80]}...")

        # Font sizes
        ticker_font_size = max(32, int(ticker_height * 0.35))
        badge_font_size = max(36, int(ticker_height * 0.38))
        ticker_prefix_width = 400
        scroll_area_width = self.width - ticker_prefix_width

        # Build filter to overlay ticker on bottom of video
        filter_complex = (
            # Create scrolling text area
            f"color=c=0x0d0d1a:s={scroll_area_width}x{ticker_height}:d={duration},"
            f"drawtext=text='{escaped_ticker}':"
            f"fontsize={ticker_font_size}:"
            f"fontcolor=white:"
            f"x=w-mod(t*{self.ticker_speed}\\,w+tw):"
            f"y=(h-{ticker_font_size})/2:"
            f"font=Arial[ticker_scroll];"

            # Create BREAKING badge
            f"color=c=0xe63946:s={ticker_prefix_width}x{ticker_height}:d={duration},"
            f"drawbox=x=0:y=0:w=8:h={ticker_height}:color=0xcc0000:t=fill,"
            f"drawtext=text='{ticker_prefix}':"
            f"fontsize={badge_font_size}:"
            f"fontcolor=white:"
            f"x=20:"
            f"y=(h-{badge_font_size})/2:"
            f"font=Arial[ticker_badge];"

            # Combine badge and scroll
            f"[ticker_badge][ticker_scroll]hstack=inputs=2[ticker];"

            # Overlay ticker on video at bottom
            f"[0:v][ticker]overlay=0:{ticker_y}[video]"
        )

        # Build command with optional background music
        if has_bg_music:
            # Add background music that starts after the jingle
            # Music fades in after music_start_time and plays at reduced volume
            music_duration = duration - music_start_time
            if music_duration > 0:
                # Audio filter: delay music start, reduce volume, mix with original audio
                audio_filter = (
                    f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[original];"
                    f"[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                    f"atrim=0:{music_duration},"  # Trim music to remaining duration
                    f"adelay={int(music_start_time * 1000)}|{int(music_start_time * 1000)},"  # Delay in ms
                    f"volume={self.background_volume}dB,"
                    f"afade=t=in:st={music_start_time}:d=1.0[music];"  # 1s fade in
                    f"[original][music]amix=inputs=2:duration=first:dropout_transition=2[audio_out]"
                )
                filter_complex = filter_complex + ";" + audio_filter

                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-stream_loop", "-1",  # Loop music if needed
                    "-i", self.background_music,
                    "-filter_complex", filter_complex,
                    "-map", "[video]",
                    "-map", "[audio_out]",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-r", str(self.fps),
                    "-c:a", "aac",
                    "-ac", "2",
                    "-ar", "44100",
                    "-b:a", "128k",
                    "-pix_fmt", "yuv420p",
                    "-t", str(duration),
                    output_path
                ]
                print(f"[VideoGenerator] Adding ticker + background music (starting at {music_start_time:.1f}s)...")
            else:
                # No room for music, just add ticker
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-filter_complex", filter_complex,
                    "-map", "[video]",
                    "-map", "0:a",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-r", str(self.fps),
                    "-c:a", "copy",
                    "-pix_fmt", "yuv420p",
                    output_path
                ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-filter_complex", filter_complex,
                "-map", "[video]",
                "-map", "0:a",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-r", str(self.fps),  # Force correct framerate
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                output_path
            ]

        print(f"[VideoGenerator] Adding ticker overlay...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[VideoGenerator] Ticker overlay error: {result.stderr}")
            raise RuntimeError(f"FFmpeg ticker overlay failed: {result.stderr}")

        print(f"[VideoGenerator] Final video with ticker: {output_path}")
        return output_path

    def generate_segmented_video(
        self,
        audio_segments: list[dict],
        screenshots: list[str],
        ticker_text: str,
        presenter_image: Optional[str] = None,
        articles: Optional[list[dict]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate video by creating each segment separately, then concatenating.

        This ensures perfect audio-video sync for each segment.

        Args:
            audio_segments: List of segment dicts with 'type', 'path', 'duration'
            screenshots: List of screenshot paths (one per story)
            ticker_text: Text for scrolling ticker
            presenter_image: Path to presenter image
            articles: List of article dicts for stats
            output_path: Final output path

        Returns:
            Path to final concatenated video
        """
        # Reset story video tracking for this new generation
        self._reset_story_video_tracking()

        segment_videos = []
        story_idx = 0

        # Calculate when background music should start (after jingle segment)
        music_start_time = 0.0
        for segment in audio_segments:
            if segment.get("type") == "music":
                music_start_time = segment.get("duration", 0.0)
                break

        for segment in audio_segments:
            seg_type = segment.get("type")
            audio_path = segment.get("path")

            # Determine which screenshot to use
            screenshot = None
            article = None

            if seg_type == "music":
                # During music intro, show first image
                if screenshots:
                    screenshot = screenshots[0]
                if articles:
                    article = articles[0]
            elif seg_type == "intro":
                # During intro, show first image
                if screenshots:
                    screenshot = screenshots[0]
                if articles:
                    article = articles[0]
            elif seg_type == "story":
                # Each story gets its own image
                if story_idx < len(screenshots):
                    screenshot = screenshots[story_idx]
                if articles and story_idx < len(articles):
                    article = articles[story_idx]
                story_idx += 1
            elif seg_type == "outro":
                # During outro, show last image
                if screenshots:
                    screenshot = screenshots[-1]
                if articles:
                    article = articles[-1]

            # Generate this segment's video (WITHOUT ticker - added later)
            seg_video = self.generate_segment_video(
                audio_path=audio_path,
                screenshot=screenshot,
                segment_type=seg_type,
                presenter_image=presenter_image,
                article=article
            )
            segment_videos.append(seg_video)

        # Concatenate all segments (creates video with placeholder ticker bar)
        today = datetime.now().strftime("%Y-%m-%d")
        concat_path = str(self.output_dir / today / "segments" / f"concat_{datetime.now().strftime('%H%M%S')}.mp4")
        concat_video = self.concatenate_videos(segment_videos, concat_path)

        # Add ticker overlay and background music to the concatenated video
        # Music starts after the jingle segment for continuous playback
        final_video = self.add_ticker_overlay(
            concat_video,
            ticker_text,
            output_path,
            music_start_time=music_start_time
        )
        return final_video

    def run(self, **kwargs) -> dict:
        """
        Tool interface for agents.

        Args:
            audio_path: Path to audio file (for single video mode)
            audio_segments: List of segment dicts (for segmented mode)
            screenshots: List of screenshot paths
            ticker_text: Text for scrolling ticker
            presenter_image: Path to presenter image (optional)
            simple: Use simple video mode (optional)
            background_image: Background for simple mode (optional)
            use_greenscreen: Apply chromakey to presenter image (default True)
            presenter_only: Use presenter-only mode with solid background (optional)
            segmented: Use segmented video generation (default False)
            articles: List of article dicts for stats display

        Returns:
            Dictionary with 'video_path'
        """
        audio_path = kwargs.get("audio_path")
        audio_segments = kwargs.get("audio_segments")  # For segmented mode
        screenshots = kwargs.get("screenshots", [])
        ticker_text = kwargs.get("ticker_text", "")
        presenter_image = kwargs.get("presenter_image") or self.presenter_image
        simple = kwargs.get("simple", False)
        background_image = kwargs.get("background_image")
        use_greenscreen = kwargs.get("use_greenscreen", True)
        presenter_only = kwargs.get("presenter_only", False)
        segment_durations = kwargs.get("segment_durations")
        intro_duration = kwargs.get("intro_duration", 0.0)
        articles = kwargs.get("articles")  # For stats display in presenter background
        segmented = kwargs.get("segmented", False)  # Use segmented video generation

        # Mode 0: Segmented video generation (each segment rendered separately)
        if segmented and audio_segments:
            video_path = self.generate_segmented_video(
                audio_segments=audio_segments,
                screenshots=screenshots,
                ticker_text=ticker_text,
                presenter_image=presenter_image,
                articles=articles
            )
            return {
                "video_path": video_path,
                "success": True
            }

        if not audio_path:
            raise ValueError("audio_path is required")

        # Mode 1: Presenter only (no screenshots) - use solid background with chromakey
        if presenter_only and presenter_image and Path(presenter_image).exists():
            video_path = self.generate_presenter_video(
                audio_path=audio_path,
                presenter_image=presenter_image,
                ticker_text=ticker_text
            )
        # Mode 2: Simple background image
        elif simple and background_image:
            video_path = self.generate_simple_video(
                audio_path=audio_path,
                background_image=background_image,
                ticker_text=ticker_text
            )
        # Mode 3: Full video with screenshots + presenter overlay
        else:
            video_path = self.generate_video(
                audio_path=audio_path,
                screenshots=screenshots,
                ticker_text=ticker_text,
                presenter_image=presenter_image,
                use_greenscreen=use_greenscreen,
                segment_durations=segment_durations,
                intro_duration=intro_duration,
                articles=articles
            )

        return {
            "video_path": video_path,
            "success": True
        }
