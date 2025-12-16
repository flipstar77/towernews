"""TTS Tool - Text-to-Speech using 11labs API."""

import os
import random
import requests
from pathlib import Path
from datetime import datetime


# Story transition phrases for variety
STORY_INTROS = [
    # First story options
    ["First up,", "Starting off,", "Kicking things off,", "Let's begin with,", "Our first story,"],
    # Second story options
    ["Next up,", "Moving on,", "In other news,", "Also today,", "Our second story,"],
    # Third story options
    ["And now,", "Next,", "Continuing on,", "Also making headlines,", "Our third story,"],
    # Fourth+ story options
    ["Finally,", "And lastly,", "Wrapping up,", "Last but not least,", "Our final story,"],
]


class TTSTool:
    """Converts text to speech using 11labs API."""

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, config: dict):
        self.config = config
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.output_dir = Path("output")

        elevenlabs_config = config.get("elevenlabs", {})
        self.voice_id = elevenlabs_config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        self.model_id = elevenlabs_config.get("model", "eleven_monolingual_v1")

    def _get_headers(self) -> dict:
        """Get API headers."""
        return {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

    def text_to_speech(
        self,
        text: str,
        output_path: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75
    ) -> str:
        """
        Convert text to speech.

        Args:
            text: Text to convert
            output_path: Path to save audio (optional)
            voice_id: Voice ID to use (optional, uses config default)
            model_id: Model ID to use (optional, uses config default)
            stability: Voice stability (0.0 to 1.0)
            similarity_boost: Similarity boost (0.0 to 1.0)

        Returns:
            Path to saved audio file
        """
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id

        # Create output directory
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_dir / today / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = str(output_dir / f"speech_{timestamp}.mp3")

        url = f"{self.BASE_URL}/text-to-speech/{voice_id}"

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost
            }
        }

        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path

    def get_available_voices(self) -> list[dict]:
        """
        Get list of available voices.

        Returns:
            List of voice dictionaries
        """
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

        url = f"{self.BASE_URL}/voices"
        headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        return [
            {
                "voice_id": v.get("voice_id"),
                "name": v.get("name"),
                "category": v.get("category")
            }
            for v in data.get("voices", [])
        ]

    def generate_news_audio(
        self,
        intro: str,
        stories: list[str],
        outro: str,
        output_path: str | None = None
    ) -> str:
        """
        Generate complete news audio with intro, stories, and outro.

        Args:
            intro: Intro text
            stories: List of story texts
            outro: Outro text
            output_path: Path to save audio (optional)

        Returns:
            Path to saved audio file
        """
        # Combine all text with pauses
        full_text = intro + "\n\n"

        for i, story in enumerate(stories, 1):
            full_text += f"Story number {i}. {story}\n\n"

        full_text += outro

        return self.text_to_speech(full_text, output_path)

    def generate_segmented_audio(
        self,
        intro: str,
        stories: list[str],
        outro: str,
        output_dir: str | None = None,
        intro_music_path: str | None = None
    ) -> dict:
        """
        Generate separate audio files for each segment (intro, stories, outro).

        This allows for precise timing of visuals and segment-specific images.

        Args:
            intro: Intro text
            stories: List of story texts
            outro: Outro text
            output_dir: Directory to save audio files (optional)
            intro_music_path: Path to intro music MP3 file (optional)

        Returns:
            Dictionary with:
                - segments: List of dicts with 'type', 'path', 'duration', 'text'
                - combined_path: Path to combined audio file
                - total_duration: Total duration in seconds
        """
        import subprocess
        import json

        # Create output directory
        today = datetime.now().strftime("%Y-%m-%d")
        if output_dir is None:
            output_dir = self.output_dir / today / "audio"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S")
        segments = []

        def get_audio_duration(path: str) -> float:
            """Get duration of audio file in seconds."""
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))

        # Add intro music as first segment (if provided)
        if intro_music_path and Path(intro_music_path).exists():
            # Normalize intro music: extract audio only, convert to 44100Hz
            # This fixes issues with embedded cover art and different sample rates
            normalized_music_path = str(output_dir / f"music_normalized_{timestamp}.mp3")
            normalize_cmd = [
                "ffmpeg", "-y",
                "-i", intro_music_path,
                "-vn",              # Remove video/cover art streams
                "-c:a", "libmp3lame",
                "-ar", "44100",     # Match TTS sample rate
                "-b:a", "192k",
                normalized_music_path
            ]
            norm_result = subprocess.run(normalize_cmd, capture_output=True, text=True)
            if norm_result.returncode != 0:
                print(f"[TTS] Warning: Could not normalize intro music: {norm_result.stderr}")
                normalized_music_path = intro_music_path  # Fallback to original

            music_duration = get_audio_duration(normalized_music_path)
            segments.append({
                "type": "music",
                "path": normalized_music_path,
                "duration": music_duration,
                "text": ""
            })
            print(f"[TTS] Added intro music (normalized): {music_duration:.1f}s")

        # Generate intro audio
        if intro:
            intro_path = str(output_dir / f"intro_{timestamp}.mp3")
            self.text_to_speech(intro, intro_path)
            intro_duration = get_audio_duration(intro_path)
            segments.append({
                "type": "intro",
                "path": intro_path,
                "duration": intro_duration,
                "text": intro
            })

        # Generate story audio files
        for i, story in enumerate(stories):
            # Select story intro phrase for variety
            is_last = (i == len(stories) - 1) and len(stories) > 1
            if is_last:
                intro_options = STORY_INTROS[3]  # Final story phrases
            elif i < len(STORY_INTROS) - 1:
                intro_options = STORY_INTROS[i]
            else:
                intro_options = STORY_INTROS[2]  # Default to "third story" style

            story_intro = random.choice(intro_options)
            story_text = f"{story_intro} {story}"
            story_path = str(output_dir / f"story_{i+1}_{timestamp}.mp3")
            self.text_to_speech(story_text, story_path)
            story_duration = get_audio_duration(story_path)
            segments.append({
                "type": "story",
                "index": i,
                "path": story_path,
                "duration": story_duration,
                "text": story
            })

        # Generate outro audio
        if outro:
            outro_path = str(output_dir / f"outro_{timestamp}.mp3")
            self.text_to_speech(outro, outro_path)
            outro_duration = get_audio_duration(outro_path)
            segments.append({
                "type": "outro",
                "path": outro_path,
                "duration": outro_duration,
                "text": outro
            })

        # Combine all segments into one audio file using FFmpeg
        combined_path = str(output_dir / f"combined_{timestamp}.mp3")

        # Create file list for FFmpeg concat - use absolute paths for Windows
        file_list_path = str(output_dir / f"filelist_{timestamp}.txt")
        with open(file_list_path, "w") as f:
            for seg in segments:
                # FFmpeg requires forward slashes and absolute paths on Windows
                abs_path = str(Path(seg["path"]).absolute()).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

        # Concatenate audio files with re-encoding to ensure consistent format
        # This prevents audio glitches when intro music has different sample rate/bitrate
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", file_list_path,
            "-c:a", "libmp3lame",
            "-ar", "44100",       # Consistent sample rate
            "-b:a", "192k",       # Consistent bitrate
            combined_path
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[TTS] FFmpeg concat error: {result.stderr}")

        # Calculate total duration
        total_duration = sum(seg["duration"] for seg in segments)

        return {
            "segments": segments,
            "combined_path": combined_path,
            "total_duration": total_duration,
            "file_list": file_list_path
        }

    def run(self, **kwargs) -> dict:
        """
        Tool interface for agents.

        Args:
            text: Text to convert to speech
            intro: Intro text (alternative to text)
            stories: List of story texts (used with intro)
            outro: Outro text (used with intro)
            segmented: Generate separate audio files for each segment (default True)
            intro_music_path: Path to intro music MP3 file (optional)

        Returns:
            Dictionary with 'audio_path' and optionally 'segments'
        """
        text = kwargs.get("text")
        intro = kwargs.get("intro")
        stories = kwargs.get("stories", [])
        outro = kwargs.get("outro", "")
        segmented = kwargs.get("segmented", True)  # Default to segmented
        intro_music_path = kwargs.get("intro_music_path")

        if text:
            audio_path = self.text_to_speech(text)
            return {
                "audio_path": audio_path,
                "success": True
            }
        elif intro and stories:
            if segmented:
                # Use segmented audio generation for precise timing
                result = self.generate_segmented_audio(
                    intro, stories, outro,
                    intro_music_path=intro_music_path
                )
                return {
                    "audio_path": result["combined_path"],
                    "segments": result["segments"],
                    "total_duration": result["total_duration"],
                    "success": True
                }
            else:
                # Use combined audio (less precise but fewer API calls)
                audio_path = self.generate_news_audio(intro, stories, outro)
                return {
                    "audio_path": audio_path,
                    "success": True
                }
        else:
            raise ValueError("Either 'text' or 'intro' + 'stories' must be provided")
