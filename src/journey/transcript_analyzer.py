"""YouTube Transcript Analyzer - Fetches and analyzes video transcripts."""

import re
import json
import subprocess
import os
import tempfile
from typing import Dict, Any, List, Optional, Callable
from youtube_transcript_api import YouTubeTranscriptApi
from pathlib import Path
from openai import OpenAI


class TranscriptAnalyzer:
    """Fetches YouTube transcripts and analyzes video structure."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.api = YouTubeTranscriptApi()
        self.cache_dir = Path("data/journey/transcripts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.topic_cache_dir = Path("data/journey/topic_analysis")
        self.topic_cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_cache_dir = Path("data/journey/audio_cache")
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        self.openai_client = None  # Lazy initialization
        self.whisper_model = None  # Lazy initialization

    def _get_openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self.openai_client is None:
            self.openai_client = OpenAI()
        return self.openai_client

    def _get_whisper_model(self, model_name: str = "base"):
        """Lazy initialization of Whisper model."""
        if self.whisper_model is None:
            import whisper
            print(f"[TranscriptAnalyzer] Loading Whisper model '{model_name}'...")
            self.whisper_model = whisper.load_model(model_name)
            print(f"[TranscriptAnalyzer] Whisper model loaded")
        return self.whisper_model

    def download_audio(self, video_id: str, progress_callback: Callable[[str], None] = None) -> Optional[str]:
        """
        Download audio from a YouTube video for transcription.

        Args:
            video_id: YouTube video ID
            progress_callback: Optional callback for progress updates

        Returns:
            Path to downloaded audio file, or None if download failed
        """
        try:
            import yt_dlp

            audio_path = self.audio_cache_dir / f"{video_id}.mp3"

            # Return cached audio if exists
            if audio_path.exists():
                if progress_callback:
                    progress_callback(f"Using cached audio for {video_id}")
                return str(audio_path)

            if progress_callback:
                progress_callback(f"Downloading audio for {video_id}...")

            url = f"https://www.youtube.com/watch?v={video_id}"

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.audio_cache_dir / f"{video_id}.%(ext)s"),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if audio_path.exists():
                if progress_callback:
                    progress_callback(f"Audio downloaded: {audio_path}")
                return str(audio_path)
            else:
                # Try to find the file with a different extension
                for ext in ['m4a', 'webm', 'opus', 'wav']:
                    alt_path = self.audio_cache_dir / f"{video_id}.{ext}"
                    if alt_path.exists():
                        if progress_callback:
                            progress_callback(f"Audio downloaded: {alt_path}")
                        return str(alt_path)

            return None

        except Exception as e:
            print(f"[TranscriptAnalyzer] Audio download failed: {e}")
            if progress_callback:
                progress_callback(f"Audio download failed: {e}")
            return None

    def transcribe_with_whisper(self, video_id: str, model_name: str = "base",
                                 language: str = None,
                                 progress_callback: Callable[[str], None] = None) -> Dict[str, Any]:
        """
        Transcribe a YouTube video using Whisper.

        Args:
            video_id: YouTube video ID
            model_name: Whisper model to use (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'de') or None for auto-detect
            progress_callback: Optional callback for progress updates

        Returns:
            dict with transcript data similar to fetch_transcript format
        """
        video_id = self.extract_video_id(video_id)

        # Check cache first
        cache_file = self.cache_dir / f"{video_id}_whisper.json"
        if cache_file.exists():
            if progress_callback:
                progress_callback(f"Using cached Whisper transcript for {video_id}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Download audio
        audio_path = self.download_audio(video_id, progress_callback)
        if not audio_path:
            return {"error": "Failed to download audio", "video_id": video_id}

        try:
            if progress_callback:
                progress_callback(f"Transcribing with Whisper ({model_name})...")

            model = self._get_whisper_model(model_name)

            # Transcribe
            options = {}
            if language:
                options['language'] = language

            result = model.transcribe(audio_path, **options)

            if progress_callback:
                progress_callback("Processing transcript...")

            # Process into our standard format
            transcript_result = self._process_whisper_result(video_id, result)

            # Cache the result
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_result, f, indent=2, ensure_ascii=False)

            if progress_callback:
                progress_callback(f"Transcription complete: {transcript_result['word_count']} words")

            return transcript_result

        except Exception as e:
            print(f"[TranscriptAnalyzer] Whisper transcription failed: {e}")
            return {"error": str(e), "video_id": video_id}

    def _process_whisper_result(self, video_id: str, whisper_result: Dict) -> Dict[str, Any]:
        """Process Whisper transcription result into standard format."""
        segments = whisper_result.get('segments', [])
        full_text = whisper_result.get('text', '')
        language = whisper_result.get('language', 'unknown')

        # Calculate duration
        total_duration = 0
        if segments:
            last_segment = segments[-1]
            total_duration = last_segment.get('end', 0)

        # Process segments into our format
        processed_segments = []
        for seg in segments:
            processed_segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'start_formatted': self._format_duration(seg.get('start', 0)),
                'text': seg.get('text', '').strip()
            })

        # Create raw entries format (for compatibility)
        raw_entries = []
        for seg in segments:
            raw_entries.append({
                'text': seg.get('text', '').strip(),
                'start': seg.get('start', 0),
                'duration': seg.get('end', 0) - seg.get('start', 0)
            })

        return {
            "video_id": video_id,
            "language": language,
            "duration_seconds": total_duration,
            "duration_formatted": self._format_duration(total_duration),
            "word_count": len(full_text.split()),
            "full_text": full_text.strip(),
            "segments": processed_segments,
            "raw_entries": raw_entries,
            "transcription_method": "whisper"
        }

    # Tower game keywords for content filtering
    TOWER_KEYWORDS = [
        # Game name variants
        "the tower", "thetower", "tower game", "tower idle",
        # Core mechanics
        "tier", "wave", "death wave", "dw", "ultimate weapon", "uw",
        "workshop", "lab", "research", "module", "perk",
        # Resources
        "coins", "cells", "gems", "stones", "golden tower",
        # Gameplay
        "tournament", "tourney", "bracket", "farming", "afk",
        "guardian", "enemy", "boss", "orb", "black hole",
        # Specific items/features
        "spotlight", "thorn", "chain lightning", "critical coin",
        "free pack", "battle pass", "season pass", "event",
        # Community
        "guild", "clan", "f2p", "whale", "progression",
        # Common video types
        "guide", "tips", "strategy", "update", "patch", "changelog"
    ]

    # Keywords that suggest NON-Tower content (exclusion patterns)
    EXCLUDE_KEYWORDS = [
        "minecraft", "fortnite", "valorant", "league of legends", "lol",
        "genshin", "honkai", "call of duty", "cod", "apex legends",
        "pokemon", "clash royale", "clash of clans", "brawl stars",
        "roblox", "among us", "pubg", "free fire",
        "unboxing", "reaction", "vlog", "mukbang", "asmr",
        "shorts compilation", "tiktok", "instagram"
    ]

    def is_tower_related(self, title: str, description: str = "") -> dict:
        """
        Check if a video is likely Tower-related based on title and description.

        Args:
            title: Video title
            description: Video description (optional)

        Returns:
            dict with:
                - is_relevant: bool indicating if likely Tower content
                - confidence: float 0-1 indicating confidence
                - matched_keywords: list of matched keywords
                - reason: explanation string
        """
        text = f"{title} {description}".lower()

        # Check for exclusion keywords first
        excluded_matches = []
        for keyword in self.EXCLUDE_KEYWORDS:
            if keyword in text:
                excluded_matches.append(keyword)

        if excluded_matches:
            return {
                "is_relevant": False,
                "confidence": 0.9,
                "matched_keywords": [],
                "excluded_keywords": excluded_matches,
                "reason": f"Contains non-Tower keywords: {', '.join(excluded_matches[:3])}"
            }

        # Check for Tower keywords
        matched_keywords = []
        for keyword in self.TOWER_KEYWORDS:
            if keyword in text:
                matched_keywords.append(keyword)

        # Calculate confidence based on matches
        if len(matched_keywords) >= 3:
            confidence = 0.95
            is_relevant = True
            reason = f"Strong match: {len(matched_keywords)} Tower keywords found"
        elif len(matched_keywords) >= 2:
            confidence = 0.8
            is_relevant = True
            reason = f"Good match: {len(matched_keywords)} Tower keywords found"
        elif len(matched_keywords) == 1:
            confidence = 0.6
            is_relevant = True
            reason = f"Weak match: 1 Tower keyword found ({matched_keywords[0]})"
        else:
            confidence = 0.3
            is_relevant = False
            reason = "No Tower keywords found"

        return {
            "is_relevant": is_relevant,
            "confidence": confidence,
            "matched_keywords": matched_keywords,
            "excluded_keywords": [],
            "reason": reason
        }

    def filter_tower_videos(self, videos: List[Dict[str, Any]],
                            min_confidence: float = 0.5,
                            use_ai: bool = False,
                            progress_callback: Callable[[str], None] = None) -> List[Dict[str, Any]]:
        """
        Filter a list of videos to only include Tower-related content.

        Args:
            videos: List of video dicts with 'title' and optionally 'description'
            min_confidence: Minimum confidence threshold (0-1)
            use_ai: Use AI for smarter filtering (slower but more accurate)
            progress_callback: Optional callback for progress updates

        Returns:
            List of videos that are likely Tower-related, with relevance info added
        """
        if progress_callback:
            progress_callback(f"Filtering {len(videos)} videos for Tower content...")

        filtered = []
        for i, video in enumerate(videos):
            title = video.get('title', '')
            description = video.get('description', '')

            if use_ai:
                # Use AI for more accurate filtering
                relevance = self._check_relevance_with_ai(title, description)
            else:
                # Use keyword matching
                relevance = self.is_tower_related(title, description)

            video['relevance'] = relevance

            if relevance['is_relevant'] and relevance['confidence'] >= min_confidence:
                filtered.append(video)

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(f"Checked {i + 1}/{len(videos)} videos, {len(filtered)} relevant so far...")

        if progress_callback:
            progress_callback(f"Found {len(filtered)}/{len(videos)} Tower-related videos")

        return filtered

    def _check_relevance_with_ai(self, title: str, description: str = "") -> dict:
        """Use AI to check if content is Tower-related."""
        try:
            prompt = f"""Analyze if this YouTube video is about "The Tower" mobile idle defense game.

Title: {title}
Description: {description[:500] if description else 'No description'}

The Tower is a mobile idle/incremental game where players:
- Build and upgrade a tower to defend against waves of enemies
- Manage resources like coins and cells
- Progress through tiers and waves
- Participate in tournaments
- Upgrade workshop, lab, and research modules

Respond with JSON:
{{
    "is_tower_game": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""

            response = self._get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You analyze YouTube video titles to determine if they're about The Tower mobile game. Be precise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return {
                "is_relevant": result.get("is_tower_game", False),
                "confidence": result.get("confidence", 0.5),
                "matched_keywords": [],
                "excluded_keywords": [],
                "reason": result.get("reason", "AI analysis")
            }

        except Exception as e:
            # Fall back to keyword matching on error
            print(f"[TranscriptAnalyzer] AI filtering failed, using keywords: {e}")
            return self.is_tower_related(title, description)

    def get_channel_videos_filtered(self, channel_url: str, max_videos: int = 100,
                                     min_confidence: float = 0.5,
                                     use_ai: bool = False,
                                     progress_callback: Callable[[str], None] = None) -> List[Dict[str, Any]]:
        """
        Fetch videos from a channel and filter to only Tower-related content.

        This is a convenience method that combines get_channel_videos and filter_tower_videos.

        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum videos to fetch before filtering
            min_confidence: Minimum confidence for Tower relevance
            use_ai: Use AI for smarter filtering
            progress_callback: Optional callback for progress updates

        Returns:
            List of Tower-related videos
        """
        # Fetch all videos
        videos = self.get_channel_videos(channel_url, max_videos, progress_callback)

        if videos and "error" in videos[0]:
            return videos

        # Filter to Tower content
        filtered = self.filter_tower_videos(
            videos,
            min_confidence=min_confidence,
            use_ai=use_ai,
            progress_callback=progress_callback
        )

        return filtered

    def extract_video_id(self, url_or_id: str) -> str:
        """Extract video ID from YouTube URL or return ID directly."""
        # Already an ID
        if len(url_or_id) == 11 and not url_or_id.startswith("http"):
            return url_or_id

        # Extract from various URL formats
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'(?:embed/)([a-zA-Z0-9_-]{11})',
            r'(?:shorts/)([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)

        return url_or_id  # Assume it's an ID

    def get_channel_videos(self, channel_url: str, max_videos: int = 50,
                           progress_callback: Callable[[str], None] = None) -> List[Dict[str, Any]]:
        """
        Fetch all video IDs and metadata from a YouTube channel.

        Args:
            channel_url: YouTube channel URL (e.g., https://www.youtube.com/@ChannelName)
            max_videos: Maximum number of videos to fetch
            progress_callback: Optional callback for progress updates

        Returns:
            List of dicts with video_id, title, duration, upload_date
        """
        try:
            import yt_dlp

            if progress_callback:
                progress_callback("Fetching channel video list...")

            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,  # Don't download, just get metadata
                'playlistend': max_videos,
            }

            # Handle different channel URL formats
            if '/videos' not in channel_url:
                if channel_url.endswith('/'):
                    channel_url = channel_url + 'videos'
                else:
                    channel_url = channel_url + '/videos'

            videos = []
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(channel_url, download=False)

                if result and 'entries' in result:
                    for entry in result['entries']:
                        if entry:
                            video_info = {
                                'video_id': entry.get('id', ''),
                                'title': entry.get('title', 'Unknown'),
                                'duration': entry.get('duration', 0),
                                'duration_formatted': self._format_duration(entry.get('duration', 0)),
                                'url': f"https://www.youtube.com/watch?v={entry.get('id', '')}"
                            }
                            videos.append(video_info)

                            if progress_callback:
                                progress_callback(f"Found {len(videos)} videos...")

            if progress_callback:
                progress_callback(f"Found {len(videos)} videos total")

            return videos

        except Exception as e:
            return [{"error": str(e)}]

    def fetch_transcript(self, video_id: str, languages: List[str] = None) -> Dict[str, Any]:
        """
        Fetch transcript for a YouTube video.

        Args:
            video_id: YouTube video ID or URL
            languages: Preferred languages (default: ['en', 'de'])

        Returns:
            dict with transcript data, timing info, and metadata
        """
        video_id = self.extract_video_id(video_id)
        languages = languages or ['en', 'de']

        # Check cache first
        cache_file = self.cache_dir / f"{video_id}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        try:
            # Fetch transcript
            transcript_list = self.api.list(video_id)

            # Try to get preferred language
            transcript = None
            used_language = None

            for lang in languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    used_language = lang
                    break
                except:
                    continue

            # Fall back to any available transcript
            if not transcript:
                available = list(transcript_list)
                if available:
                    transcript = available[0]
                    used_language = transcript.language_code

            if not transcript:
                return {"error": "No transcript available", "video_id": video_id}

            # Fetch the actual transcript data
            transcript_data = transcript.fetch()

            # Process transcript
            result = self._process_transcript(video_id, transcript_data, used_language)

            # Cache result
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            return result

        except Exception as e:
            return {"error": str(e), "video_id": video_id}

    def fetch_transcript_with_fallback(self, video_id: str, languages: List[str] = None,
                                        use_whisper: bool = True,
                                        whisper_model: str = "base",
                                        progress_callback: Callable[[str], None] = None) -> Dict[str, Any]:
        """
        Fetch transcript, falling back to Whisper if YouTube transcript unavailable.

        This is the recommended method for getting transcripts as it tries
        YouTube's API first (faster, no download needed) and only uses
        Whisper when necessary.

        Args:
            video_id: YouTube video ID or URL
            languages: Preferred languages (default: ['en', 'de'])
            use_whisper: Whether to fall back to Whisper on failure (default: True)
            whisper_model: Whisper model to use (tiny, base, small, medium, large)
            progress_callback: Optional callback for progress updates

        Returns:
            dict with transcript data, timing info, and metadata
        """
        video_id = self.extract_video_id(video_id)

        if progress_callback:
            progress_callback(f"Fetching transcript for {video_id}...")

        # Try YouTube API first
        result = self.fetch_transcript(video_id, languages)

        if "error" not in result:
            result["transcription_method"] = "youtube_api"
            if progress_callback:
                progress_callback(f"Got YouTube transcript: {result.get('word_count', 0)} words")
            return result

        # YouTube failed, try Whisper if enabled
        if use_whisper:
            if progress_callback:
                progress_callback(f"YouTube transcript unavailable, using Whisper...")
            return self.transcribe_with_whisper(
                video_id,
                model_name=whisper_model,
                language=languages[0] if languages else None,
                progress_callback=progress_callback
            )

        return result

    def _process_transcript(self, video_id: str, transcript_data: list, language: str) -> Dict[str, Any]:
        """Process raw transcript into structured data."""
        # Build full text
        full_text = " ".join([entry.text for entry in transcript_data])

        # Calculate timing info
        total_duration = 0
        if transcript_data:
            last_entry = transcript_data[-1]
            total_duration = last_entry.start + last_entry.duration

        # Segment transcript into sections (by time chunks)
        segments = self._segment_transcript(transcript_data)

        return {
            "video_id": video_id,
            "language": language,
            "duration_seconds": total_duration,
            "duration_formatted": self._format_duration(total_duration),
            "word_count": len(full_text.split()),
            "full_text": full_text,
            "segments": segments,
            "raw_entries": [
                {"text": e.text, "start": e.start, "duration": e.duration}
                for e in transcript_data
            ]
        }

    def _segment_transcript(self, transcript_data: list, segment_duration: float = 60) -> List[Dict]:
        """Break transcript into time-based segments."""
        segments = []
        current_segment = {"start": 0, "texts": [], "end": 0}

        for entry in transcript_data:
            segment_index = int(entry.start // segment_duration)
            expected_start = segment_index * segment_duration

            if entry.start >= current_segment["start"] + segment_duration:
                # Save current segment
                if current_segment["texts"]:
                    current_segment["text"] = " ".join(current_segment["texts"])
                    current_segment["end"] = entry.start
                    segments.append({
                        "start": current_segment["start"],
                        "end": current_segment["end"],
                        "start_formatted": self._format_duration(current_segment["start"]),
                        "text": current_segment["text"]
                    })

                # Start new segment
                current_segment = {
                    "start": expected_start,
                    "texts": [entry.text],
                    "end": entry.start + entry.duration
                }
            else:
                current_segment["texts"].append(entry.text)
                current_segment["end"] = entry.start + entry.duration

        # Add final segment
        if current_segment["texts"]:
            current_segment["text"] = " ".join(current_segment["texts"])
            segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"],
                "start_formatted": self._format_duration(current_segment["start"]),
                "text": current_segment["text"]
            })

        return segments

    def _format_duration(self, seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def analyze_structure(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze video structure from transcript.

        Returns insights about:
        - Introduction pattern
        - Main content sections
        - Conclusion pattern
        - Key topics mentioned
        """
        if "error" in transcript:
            return transcript

        full_text = transcript.get("full_text", "")
        segments = transcript.get("segments", [])
        duration = transcript.get("duration_seconds", 0)

        # Analyze intro (first 60 seconds)
        intro_text = ""
        for seg in segments:
            if seg["start"] < 60:
                intro_text += " " + seg["text"]

        # Analyze outro (last 60 seconds)
        outro_text = ""
        for seg in segments:
            if seg["start"] > duration - 60:
                outro_text += " " + seg["text"]

        # Common video elements detection
        has_intro_hook = any(word in intro_text.lower() for word in
                            ["today", "going to", "let's", "welcome", "hey", "what's up"])
        has_cta = any(phrase in full_text.lower() for phrase in
                     ["subscribe", "like", "comment", "notification", "bell"])
        has_outro = any(phrase in outro_text.lower() for phrase in
                       ["thanks for watching", "see you", "next video", "bye", "peace"])

        # Tower game specific keywords
        tower_keywords = [
            "tier", "wave", "coins", "cells", "damage", "workshop",
            "ultimate weapon", "uw", "tournament", "lab", "research",
            "upgrade", "module", "guardian", "bot", "golden bot",
            "death wave", "black hole", "orb", "thorn", "spotlight"
        ]

        mentioned_keywords = []
        for keyword in tower_keywords:
            if keyword in full_text.lower():
                count = full_text.lower().count(keyword)
                mentioned_keywords.append({"keyword": keyword, "count": count})

        mentioned_keywords.sort(key=lambda x: x["count"], reverse=True)

        return {
            "video_id": transcript["video_id"],
            "duration": transcript["duration_formatted"],
            "word_count": transcript["word_count"],
            "segment_count": len(segments),
            "structure": {
                "has_intro_hook": has_intro_hook,
                "has_call_to_action": has_cta,
                "has_outro": has_outro,
                "intro_preview": intro_text[:500] if intro_text else "",
                "outro_preview": outro_text[:500] if outro_text else ""
            },
            "tower_keywords": mentioned_keywords[:15],
            "estimated_words_per_minute": transcript["word_count"] / (duration / 60) if duration > 0 else 0
        }

    def compare_videos(self, video_ids: List[str]) -> Dict[str, Any]:
        """Compare structure of multiple videos to find patterns."""
        analyses = []

        for vid in video_ids:
            transcript = self.fetch_transcript(vid)
            if "error" not in transcript:
                analysis = self.analyze_structure(transcript)
                analyses.append(analysis)

        if not analyses:
            return {"error": "No valid transcripts found"}

        # Aggregate statistics
        avg_duration = sum(a.get("duration_seconds", 0) for a in analyses) / len(analyses)
        avg_words = sum(a.get("word_count", 0) for a in analyses) / len(analyses)
        avg_wpm = sum(a.get("estimated_words_per_minute", 0) for a in analyses) / len(analyses)

        # Common patterns
        intro_hook_pct = sum(1 for a in analyses if a["structure"]["has_intro_hook"]) / len(analyses)
        cta_pct = sum(1 for a in analyses if a["structure"]["has_call_to_action"]) / len(analyses)
        outro_pct = sum(1 for a in analyses if a["structure"]["has_outro"]) / len(analyses)

        # Aggregate keywords
        all_keywords = {}
        for a in analyses:
            for kw in a.get("tower_keywords", []):
                keyword = kw["keyword"]
                all_keywords[keyword] = all_keywords.get(keyword, 0) + kw["count"]

        top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "videos_analyzed": len(analyses),
            "averages": {
                "duration_seconds": avg_duration,
                "word_count": avg_words,
                "words_per_minute": avg_wpm
            },
            "patterns": {
                "intro_hook_percentage": intro_hook_pct * 100,
                "call_to_action_percentage": cta_pct * 100,
                "outro_percentage": outro_pct * 100
            },
            "top_keywords": top_keywords,
            "individual_analyses": analyses
        }

    def save_topic_analysis(self, channel_name: str, analysis: Dict[str, Any]) -> None:
        """Save channel topic analysis to disk."""
        safe_name = re.sub(r'[^\w\-]', '_', channel_name)
        cache_file = self.topic_cache_dir / f"{safe_name}_topics.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

    def load_topic_analysis(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """Load channel topic analysis from disk if exists."""
        safe_name = re.sub(r'[^\w\-]', '_', channel_name)
        cache_file = self.topic_cache_dir / f"{safe_name}_topics.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def list_saved_analyses(self) -> List[str]:
        """List all saved topic analyses."""
        analyses = []
        for f in self.topic_cache_dir.glob("*_topics.json"):
            name = f.stem.replace("_topics", "").replace("_", " ")
            analyses.append(name)
        return analyses

    def extract_topics(self, transcript: Dict[str, Any], video_title: str = "") -> Dict[str, Any]:
        """
        Use AI to extract actual topics and themes from a video transcript.

        Returns:
            - main_topic: The primary subject of the video
            - content_type: Guide, news, strategy, review, etc.
            - topics: List of specific topics discussed
            - tips: Specific advice given
            - questions_answered: What questions does this video answer
            - target_audience: Who is this video for (beginners, advanced, etc.)
        """
        if "error" in transcript:
            return transcript

        video_id = transcript.get("video_id", "")

        # Check cache first
        topic_cache_file = self.topic_cache_dir / f"video_{video_id}.json"
        if topic_cache_file.exists():
            with open(topic_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        full_text = transcript.get("full_text", "")

        # Truncate if too long (GPT has context limits)
        max_chars = 12000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "..."

        prompt = f"""Analyze this YouTube video transcript about "The Tower" mobile game.

Video Title: {video_title}

Transcript:
{full_text}

Extract the following information in JSON format:
{{
    "main_topic": "The primary subject/focus of this video (1 sentence)",
    "content_type": "One of: Guide, Tutorial, News/Update, Strategy, Tier List, Progression Guide, Event Coverage, Tips & Tricks, Review, Comparison, Q&A",
    "topics": ["List of 3-7 specific topics discussed in the video"],
    "tips": ["List of specific tips or advice given (max 5 most important)"],
    "questions_answered": ["What questions does this video answer for viewers? (max 5)"],
    "target_audience": "Who is this for: Beginners, Intermediate, Advanced, or All Players",
    "key_takeaways": ["2-3 main things a viewer would learn from this video"],
    "game_features_covered": ["Specific game features/mechanics discussed: workshop, lab, tournament, etc."]
}}

Be specific and extract actual content from the transcript, not generic descriptions."""

        try:
            response = self._get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing gaming video content. Extract specific, actionable information from transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            result["video_id"] = video_id
            result["video_title"] = video_title
            result["duration"] = transcript.get("duration_formatted", "")

            # Cache the result
            if video_id:
                with open(topic_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            return result

        except Exception as e:
            return {
                "error": str(e),
                "video_id": transcript.get("video_id", ""),
                "video_title": video_title
            }

    def analyze_channel_topics(self, transcripts: List[Dict[str, Any]],
                                video_titles: List[str] = None,
                                progress_callback: Callable[[str], None] = None) -> Dict[str, Any]:
        """
        Analyze multiple videos to find common topics and content patterns.

        Returns aggregated topic analysis across all videos.
        """
        video_titles = video_titles or [""] * len(transcripts)
        topic_analyses = []

        for i, (transcript, title) in enumerate(zip(transcripts, video_titles)):
            if progress_callback:
                progress_callback(f"Analyzing topics in video {i+1}/{len(transcripts)}: {title[:50]}...")

            if "error" not in transcript:
                analysis = self.extract_topics(transcript, title)
                if "error" not in analysis:
                    topic_analyses.append(analysis)

        if not topic_analyses:
            return {"error": "No valid topic analyses"}

        # Aggregate content types
        content_types = {}
        for a in topic_analyses:
            ct = a.get("content_type", "Unknown")
            content_types[ct] = content_types.get(ct, 0) + 1

        # Aggregate all topics
        all_topics = {}
        for a in topic_analyses:
            for topic in a.get("topics", []):
                topic_lower = topic.lower()
                all_topics[topic_lower] = all_topics.get(topic_lower, 0) + 1

        # Aggregate game features
        all_features = {}
        for a in topic_analyses:
            for feature in a.get("game_features_covered", []):
                feature_lower = feature.lower()
                all_features[feature_lower] = all_features.get(feature_lower, 0) + 1

        # Aggregate target audiences
        audiences = {}
        for a in topic_analyses:
            aud = a.get("target_audience", "Unknown")
            audiences[aud] = audiences.get(aud, 0) + 1

        # Collect all tips
        all_tips = []
        for a in topic_analyses:
            all_tips.extend(a.get("tips", []))

        # Sort by frequency
        sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)
        sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
        sorted_content_types = sorted(content_types.items(), key=lambda x: x[1], reverse=True)

        return {
            "videos_analyzed": len(topic_analyses),
            "content_types": sorted_content_types,
            "top_topics": sorted_topics[:20],
            "game_features": sorted_features[:15],
            "target_audiences": audiences,
            "sample_tips": all_tips[:20],
            "individual_analyses": topic_analyses
        }

    def generate_video_template(self, topic_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an improved video script template based on analyzed creator patterns.

        Uses the topic analysis data to create templates that mirror successful
        content patterns from established Tower creators.

        Args:
            topic_analysis: Channel topic analysis (or loads from saved analyses)

        Returns:
            Video template with sections, timing, and content suggestions
        """
        # Load saved analysis if not provided
        if not topic_analysis:
            saved = self.list_saved_analyses()
            if saved:
                topic_analysis = self.load_topic_analysis(saved[0])

        if not topic_analysis:
            return self._get_default_template()

        # Extract patterns from the analysis
        content_types = dict(topic_analysis.get("content_types", []))
        top_topics = [t[0] for t in topic_analysis.get("top_topics", [])[:10]]
        game_features = [f[0] for f in topic_analysis.get("game_features", [])[:10]]
        sample_tips = topic_analysis.get("sample_tips", [])[:10]
        individual_analyses = topic_analysis.get("individual_analyses", [])

        # Determine primary content type
        primary_type = "Progression Guide"
        if content_types:
            primary_type = max(content_types.items(), key=lambda x: x[1])[0]

        # Build recurring topic segments
        recurring_segments = self._build_recurring_segments(top_topics, game_features)

        # Extract common intro/outro patterns from individual analyses
        intro_patterns = self._extract_intro_patterns(individual_analyses)
        outro_patterns = self._extract_outro_patterns(individual_analyses)

        template = {
            "template_name": f"Tower Progress Video ({primary_type})",
            "based_on": f"{topic_analysis.get('videos_analyzed', 0)} analyzed videos",
            "recommended_duration": "15-25 minutes",

            "sections": [
                {
                    "name": "Hook & Greeting",
                    "duration": "0:00 - 0:30",
                    "purpose": "Grab attention, greet viewers",
                    "content_suggestions": intro_patterns,
                    "example": "What's up everyone, welcome back to Tower News! This week we've got some HUGE updates to cover..."
                },
                {
                    "name": "Weekly Stats Overview",
                    "duration": "0:30 - 2:00",
                    "purpose": "Show progress since last video",
                    "content_suggestions": [
                        "Current lifetime coins (compare to last week)",
                        "Stone count progress",
                        "Tier/Wave achievements",
                        "New unlocks or milestones"
                    ],
                    "example": "Let's start with where we are this week. We went from X trillion to Y trillion coins..."
                },
                {
                    "name": "Main Topic Segment",
                    "duration": "2:00 - 8:00",
                    "purpose": "Cover the primary focus of this episode",
                    "content_suggestions": recurring_segments[:5],
                    "tips_to_include": sample_tips[:3]
                },
                {
                    "name": "Game Feature Deep Dive",
                    "duration": "8:00 - 12:00",
                    "purpose": "Detailed look at specific game mechanics",
                    "content_suggestions": game_features[:5],
                    "example": "Now let's look at the workshop - I've been focusing on..."
                },
                {
                    "name": "Tournament/Event Update",
                    "duration": "12:00 - 15:00",
                    "purpose": "Cover competitive/event content",
                    "content_suggestions": [
                        "Tournament performance and ranking",
                        "Current event progress",
                        "Strategy adjustments",
                        "Bracket analysis"
                    ]
                },
                {
                    "name": "Tips & Advice",
                    "duration": "15:00 - 18:00",
                    "purpose": "Share actionable tips with viewers",
                    "content_suggestions": sample_tips[3:6] if len(sample_tips) > 3 else sample_tips
                },
                {
                    "name": "Community/Guild Update",
                    "duration": "18:00 - 20:00",
                    "purpose": "Community engagement",
                    "content_suggestions": [
                        "Guild achievements",
                        "Member shoutouts",
                        "Viewer questions",
                        "Community events"
                    ]
                },
                {
                    "name": "Outro & Preview",
                    "duration": "20:00 - end",
                    "purpose": "Wrap up and tease next video",
                    "content_suggestions": outro_patterns,
                    "example": "That's it for this week! Next time we'll be covering..."
                }
            ],

            "recurring_topics": recurring_segments,

            "content_pillars": [
                {
                    "name": "Progress Tracking",
                    "frequency": "Every video",
                    "elements": ["Coins", "Stones", "Tiers", "Waves", "LTC"]
                },
                {
                    "name": "Tournament Coverage",
                    "frequency": "Every video",
                    "elements": ["Rankings", "Strategies", "Bracket analysis", "Performance review"]
                },
                {
                    "name": "Build/Strategy",
                    "frequency": "Most videos",
                    "elements": ["Mod setups", "Card configurations", "Lab priorities", "Workshop spending"]
                },
                {
                    "name": "Events",
                    "frequency": "When active",
                    "elements": ["Event rewards", "Relic pickups", "Mission progress", "Skin showcases"]
                }
            ],

            "engagement_elements": [
                "Ask viewers about their progress",
                "Reference viewer comments from last video",
                "Pose questions for next video",
                "Shout out guild members or active commenters"
            ],

            "visual_suggestions": [
                "Show stats comparison screens (before/after)",
                "Overlay coin/stone gains",
                "Tournament bracket screenshots",
                "Workshop/Lab progress visuals",
                "Battle report highlights"
            ]
        }

        return template

    def _build_recurring_segments(self, topics: List[str], features: List[str]) -> List[Dict]:
        """Build recurring segment suggestions from topic patterns."""
        segments = []

        # Group related topics
        topic_groups = {
            "tournament": [],
            "economy": [],
            "progression": [],
            "strategy": [],
            "community": []
        }

        for topic in topics:
            topic_lower = topic.lower()
            if any(kw in topic_lower for kw in ["tournament", "ranking", "bracket", "legends", "champion"]):
                topic_groups["tournament"].append(topic)
            elif any(kw in topic_lower for kw in ["coin", "cell", "farm", "earning", "economy"]):
                topic_groups["economy"].append(topic)
            elif any(kw in topic_lower for kw in ["tier", "wave", "milestone", "unlock", "progress"]):
                topic_groups["progression"].append(topic)
            elif any(kw in topic_lower for kw in ["strategy", "build", "setup", "mod", "upgrade"]):
                topic_groups["strategy"].append(topic)
            elif any(kw in topic_lower for kw in ["guild", "member", "community"]):
                topic_groups["community"].append(topic)

        # Build segment suggestions
        if topic_groups["tournament"]:
            segments.append({
                "name": "Tournament Performance",
                "topics": topic_groups["tournament"][:3],
                "suggested_duration": "3-5 minutes"
            })

        if topic_groups["economy"]:
            segments.append({
                "name": "Economy & Farming",
                "topics": topic_groups["economy"][:3],
                "suggested_duration": "2-4 minutes"
            })

        if topic_groups["progression"]:
            segments.append({
                "name": "Progression Updates",
                "topics": topic_groups["progression"][:3],
                "suggested_duration": "3-5 minutes"
            })

        if topic_groups["strategy"]:
            segments.append({
                "name": "Strategy & Builds",
                "topics": topic_groups["strategy"][:3],
                "suggested_duration": "4-6 minutes"
            })

        # Add feature-based segments
        for feature in features[:5]:
            segments.append({
                "name": f"{feature.title()} Update",
                "topics": [feature],
                "suggested_duration": "2-3 minutes"
            })

        return segments

    def _extract_intro_patterns(self, analyses: List[Dict]) -> List[str]:
        """Extract common intro patterns from video analyses."""
        patterns = [
            "Energetic greeting to returning viewers",
            "Quick preview of what's covered in this video",
            "Mention of exciting news or achievements",
            "Reference to previous video or viewer comments"
        ]

        # Look for common intro elements in key_takeaways and main_topic
        common_starts = []
        for a in analyses[:10]:  # Sample first 10
            main_topic = a.get("main_topic", "")
            if "progress" in main_topic.lower():
                common_starts.append("Progress update hook")
            if "update" in main_topic.lower():
                common_starts.append("Weekly/regular update format")
            if "new" in main_topic.lower():
                common_starts.append("New feature or content highlight")

        return patterns + list(set(common_starts))[:3]

    def _extract_outro_patterns(self, analyses: List[Dict]) -> List[str]:
        """Extract common outro patterns from video analyses."""
        return [
            "Summarize key points from the video",
            "Preview next video's content",
            "Call to action (like, subscribe, comment)",
            "Thank viewers for watching",
            "Mention guild recruitment or community",
            "End with a question for viewer engagement"
        ]

    def _get_default_template(self) -> Dict[str, Any]:
        """Return a default template when no analysis data is available."""
        return {
            "template_name": "Tower Progress Video (Default)",
            "based_on": "Default template - analyze creator videos for customized version",
            "recommended_duration": "15-20 minutes",

            "sections": [
                {"name": "Intro & Hook", "duration": "0:00 - 0:30"},
                {"name": "Progress Update", "duration": "0:30 - 3:00"},
                {"name": "Main Content", "duration": "3:00 - 12:00"},
                {"name": "Tournament/Event", "duration": "12:00 - 15:00"},
                {"name": "Tips & Wrap-up", "duration": "15:00 - end"}
            ]
        }

    def generate_script_outline(self, battle_data: Dict[str, Any] = None,
                                 template: Dict[str, Any] = None) -> str:
        """
        Generate a video script outline using the template and battle data.

        Args:
            battle_data: Battle report data to incorporate
            template: Video template to use (or generates one)

        Returns:
            Formatted script outline as string
        """
        if not template:
            template = self.generate_video_template()

        outline = f"# {template.get('template_name', 'Tower Progress Video')}\n\n"
        outline += f"**Duration Target:** {template.get('recommended_duration', '15-20 min')}\n\n"

        for section in template.get("sections", []):
            outline += f"## {section.get('name', 'Section')}\n"
            outline += f"*{section.get('duration', '')}*\n\n"

            if section.get("purpose"):
                outline += f"**Purpose:** {section['purpose']}\n\n"

            if section.get("content_suggestions"):
                outline += "**Cover:**\n"
                for item in section["content_suggestions"][:5]:
                    if isinstance(item, dict):
                        outline += f"- {item.get('name', item)}\n"
                    else:
                        outline += f"- {item}\n"
                outline += "\n"

            if section.get("example"):
                outline += f"*Example:* \"{section['example']}\"\n\n"

            outline += "---\n\n"

        # Add battle data if provided
        if battle_data:
            outline += "## Battle Data to Include\n\n"
            summary = battle_data.get("summary", {})
            if summary:
                outline += f"- **Tier:** {summary.get('tier', '?')}\n"
                outline += f"- **Wave:** {summary.get('wave', '?')}\n"
                outline += f"- **Coins Earned:** {summary.get('coins_earned', '?')}\n"
                outline += f"- **Coins/Hour:** {summary.get('coins_per_hour', '?')}\n"

        return outline
