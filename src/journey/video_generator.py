"""Journey Video Generator - Creates Tower Journey videos from battle data."""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from openai import OpenAI


class JourneyVideoGenerator:
    """Generates Tower Journey videos from battle history data."""

    def __init__(self, config: dict = None, knowledge_base=None):
        self.config = config or self._load_config()
        self.openai_client = None  # Lazy initialization
        self.knowledge_base = knowledge_base  # Optional KB for RAG-enhanced scripts

    def _load_config(self) -> dict:
        """Load journey config."""
        config_path = Path("config/journey_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        # Fallback to main config
        config_path = Path("config/config.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self.openai_client is None:
            self.openai_client = OpenAI()
        return self.openai_client

    def format_number(self, n: float) -> str:
        """Format large numbers with suffix."""
        if n >= 1e15:
            return f"{n/1e15:.2f}q"
        elif n >= 1e12:
            return f"{n/1e12:.2f}T"
        elif n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        else:
            return f"{n:.0f}"

    def calculate_weekly_stats(self, battles: list[dict]) -> dict:
        """Calculate aggregate statistics from a list of battles."""
        total_runs = len(battles)
        total_coins = 0
        total_cells = 0
        total_enemies = 0
        total_gb_kills = 0
        coins_per_hour_list = []
        cells_per_hour_list = []
        waves_list = []
        tiers_seen = set()

        for battle in battles:
            values = battle.get("values", {})
            summary = battle.get("summary", {})

            # Coins earned
            coins_val = values.get("coins_earned", {}).get("numeric", 0)
            total_coins += coins_val

            # Cells earned
            cells_val = values.get("cells_earned", {}).get("numeric", 0)
            total_cells += cells_val

            # Enemies and GB kills
            enemies_val = values.get("total_enemies", {}).get("numeric", 0)
            total_enemies += enemies_val
            gb_kills_val = values.get("golden_bot_kills", {}).get("numeric", 0)
            total_gb_kills += gb_kills_val

            # Coins per hour
            cph = values.get("coins_per_hour", {}).get("numeric", 0)
            if cph > 0:
                coins_per_hour_list.append(cph)

            # Cells per hour
            cellph = values.get("cells_per_hour", {}).get("numeric", 0)
            if cellph > 0:
                cells_per_hour_list.append(cellph)

            # Wave and tier
            wave_val = values.get("wave", {}).get("numeric", 0)
            if wave_val > 0:
                waves_list.append(wave_val)
            tier = summary.get("tier", "")
            if tier:
                tiers_seen.add(str(tier))

        # Calculate averages
        avg_coins_per_hour = sum(coins_per_hour_list) / len(coins_per_hour_list) if coins_per_hour_list else 0
        avg_cells_per_hour = sum(cells_per_hour_list) / len(cells_per_hour_list) if cells_per_hour_list else 0
        avg_wave = sum(waves_list) / len(waves_list) if waves_list else 0
        max_wave = max(waves_list) if waves_list else 0
        gb_kill_pct = (total_gb_kills / total_enemies * 100) if total_enemies > 0 else 0

        return {
            "total_runs": total_runs,
            "total_coins": total_coins,
            "total_coins_formatted": self.format_number(total_coins),
            "total_cells": total_cells,
            "total_cells_formatted": self.format_number(total_cells),
            "avg_coins_per_hour": avg_coins_per_hour,
            "avg_coins_per_hour_formatted": self.format_number(avg_coins_per_hour),
            "avg_cells_per_hour": avg_cells_per_hour,
            "avg_cells_per_hour_formatted": self.format_number(avg_cells_per_hour),
            "avg_wave": avg_wave,
            "max_wave": max_wave,
            "max_wave_formatted": self.format_number(max_wave),
            "total_enemies": total_enemies,
            "total_enemies_formatted": self.format_number(total_enemies),
            "total_gb_kills": total_gb_kills,
            "gb_kill_pct": gb_kill_pct,
            "tiers_seen": sorted(tiers_seen)
        }

    def generate_script(
        self,
        stats: dict,
        battles: list[dict],
        include_highlights: bool = True
    ) -> dict:
        """Generate a video script from battle statistics using GPT.

        Returns:
            Dictionary with 'intro', 'stories', 'outro' keys.
        """
        # Build context for GPT
        context = f"""Generate a short, energetic video script for a Tower Journey progress update.

Weekly Statistics:
- Total Runs: {stats['total_runs']}
- Total Coins Earned: {stats['total_coins_formatted']}
- Total Cells Earned: {stats['total_cells_formatted']}
- Average Coins/Hour: {stats['avg_coins_per_hour_formatted']}
- Average Cells/Hour: {stats['avg_cells_per_hour_formatted']}
- Highest Wave: {stats['max_wave_formatted']}
- Golden Bot Kill Rate: {stats['gb_kill_pct']:.1f}%
- Tiers Played: {', '.join(stats['tiers_seen']) if stats['tiers_seen'] else 'Various'}

Generate a script with:
1. A brief intro greeting (1-2 sentences, energetic)
2. 2-3 story segments highlighting the key achievements
3. A closing outro with call to action

Format as JSON with keys: intro, stories (array), outro
Keep each segment concise for TTS (max 30 words per segment).
Be enthusiastic but not over the top. Use gaming terminology naturally."""

        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an energetic gaming content creator making Tower game progress videos. Be concise and engaging."},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            script = json.loads(response.choices[0].message.content)
            return script
        except Exception as e:
            print(f"[JourneyVideoGenerator] Error generating script: {e}")
            # Return fallback script
            return self._generate_fallback_script(stats)

    def _generate_fallback_script(self, stats: dict) -> dict:
        """Generate a fallback script without GPT."""
        return {
            "intro": f"Welcome back to Tower Journey! This week we've completed {stats['total_runs']} runs and earned some serious rewards.",
            "stories": [
                f"We racked up {stats['total_coins_formatted']} coins total, averaging {stats['avg_coins_per_hour_formatted']} per hour.",
                f"Our highest wave this week was {stats['max_wave_formatted']}, pushing through tier {stats['tiers_seen'][-1] if stats['tiers_seen'] else 'multiple tiers'}.",
                f"Cell farming was solid with {stats['total_cells_formatted']} cells collected and a {stats['gb_kill_pct']:.0f}% Golden Bot kill rate."
            ],
            "outro": "That's the progress for this week! Don't forget to like and subscribe for more Tower content. See you in the next one!"
        }

    def _get_relevant_knowledge(self, stats: dict, battles: list[dict]) -> str:
        """Retrieve relevant knowledge from the KB based on battle stats.

        Args:
            stats: Calculated weekly statistics
            battles: List of battle data

        Returns:
            String with relevant knowledge context for script generation
        """
        if not self.knowledge_base:
            return ""

        knowledge_pieces = []

        # Build queries based on the stats and battles
        queries = []

        # Query based on tier
        if stats.get('tiers_seen'):
            for tier in stats['tiers_seen'][:2]:  # Top 2 tiers
                queries.append(f"tier {tier} strategy tips")

        # Query based on wave performance
        if stats.get('max_wave', 0) > 0:
            queries.append(f"high wave strategy wave {stats['max_wave']}")

        # Query based on coins per hour
        if stats.get('avg_coins_per_hour', 0) > 0:
            queries.append("coin farming optimization tips")

        # Query based on Golden Bot kills
        if stats.get('gb_kill_pct', 0) > 50:
            queries.append("golden bot farming strategy")
        elif stats.get('gb_kill_pct', 0) < 30:
            queries.append("how to improve golden bot kills")

        # Query based on cells
        if stats.get('total_cells', 0) > 0:
            queries.append("cell farming tips")

        # Perform searches and collect results
        for query in queries[:4]:  # Limit to 4 queries
            try:
                results = self.knowledge_base.search(query, top_k=2)
                for result in results:
                    content = result.get('content', '')
                    if content and len(content) > 50:
                        # Truncate long content
                        if len(content) > 300:
                            content = content[:300] + "..."
                        knowledge_pieces.append(content)
            except Exception as e:
                print(f"[JourneyVideoGenerator] KB search error: {e}")

        # Deduplicate and format
        unique_knowledge = list(dict.fromkeys(knowledge_pieces))[:5]

        if unique_knowledge:
            return "\n\nRelevant Game Knowledge (use to add tips/context):\n" + "\n---\n".join(unique_knowledge)

        return ""

    def generate_enhanced_script(
        self,
        stats: dict,
        battles: list[dict],
        include_tips: bool = True,
        include_comparisons: bool = True
    ) -> dict:
        """Generate a RAG-enhanced video script using knowledge base context.

        Args:
            stats: Battle statistics
            battles: List of battle data
            include_tips: Whether to include KB tips in the script
            include_comparisons: Whether to add performance comparisons

        Returns:
            Dictionary with 'intro', 'stories', 'outro', and optionally 'tips' keys
        """
        # Get relevant knowledge from KB
        knowledge_context = self._get_relevant_knowledge(stats, battles) if include_tips else ""

        # Build enhanced context for GPT
        context = f"""Generate an informative and engaging video script for a Tower Journey progress update.

Weekly Statistics:
- Total Runs: {stats['total_runs']}
- Total Coins Earned: {stats['total_coins_formatted']}
- Total Cells Earned: {stats['total_cells_formatted']}
- Average Coins/Hour: {stats['avg_coins_per_hour_formatted']}
- Average Cells/Hour: {stats['avg_cells_per_hour_formatted']}
- Highest Wave: {stats['max_wave_formatted']}
- Average Wave: {stats.get('avg_wave', 0):.0f}
- Golden Bot Kill Rate: {stats['gb_kill_pct']:.1f}%
- Total Enemies Killed: {stats['total_enemies_formatted']}
- Tiers Played: {', '.join(stats['tiers_seen']) if stats['tiers_seen'] else 'Various'}
{knowledge_context}

Generate a script with:
1. A brief intro greeting (1-2 sentences, energetic)
2. 3-4 story segments that:
   - Highlight key achievements with specific numbers
   - Include a relevant tip or insight from the game knowledge (if provided)
   - Compare performance metrics (e.g., "our coins per hour is solid for tier X")
3. A closing outro with call to action

Format as JSON with keys: intro, stories (array of strings), outro, featured_tip (optional - one standout tip)
Keep each segment concise for TTS (max 40 words per segment).
Be enthusiastic and use Tower game terminology naturally (coins, cells, waves, tiers, Golden Bot, etc.)."""

        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Tower game content creator. Generate engaging scripts that combine progress stats with helpful tips and insights. Use specific numbers and game terminology."
                    },
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            script = json.loads(response.choices[0].message.content)

            # Add metadata about KB usage
            script['kb_enhanced'] = bool(knowledge_context)

            return script

        except Exception as e:
            print(f"[JourneyVideoGenerator] Error generating enhanced script: {e}")
            return self._generate_fallback_script(stats)

    def generate_video(
        self,
        battles: list[dict],
        screenshots: list[str] = None,
        output_path: str = None,
        include_tts: bool = True,
        use_knowledge_base: bool = True
    ) -> dict:
        """Generate a complete Tower Journey video.

        Args:
            battles: List of battle dictionaries to include
            screenshots: Optional list of screenshot paths
            output_path: Optional output path for the video
            include_tts: Whether to generate TTS voiceover
            use_knowledge_base: Whether to use KB for enhanced scripts (default True)

        Returns:
            Dictionary with video generation results
        """
        from src.tools.tts import TTSTool
        from src.tools.video_generator import VideoGenerator

        # Load main config for video generation
        main_config_path = Path("config/config.yaml")
        with open(main_config_path, 'r') as f:
            main_config = yaml.safe_load(f)

        # Calculate stats
        stats = self.calculate_weekly_stats(battles)

        # Generate script (use enhanced version if KB available)
        print("[JourneyVideoGenerator] Generating script...")
        if use_knowledge_base and self.knowledge_base:
            print("[JourneyVideoGenerator] Using Knowledge Base for enhanced script...")
            script = self.generate_enhanced_script(stats, battles)
        else:
            script = self.generate_script(stats, battles)

        result = {
            "stats": stats,
            "script": script,
            "video_path": None,
            "audio_path": None
        }

        if not include_tts:
            return result

        # Generate TTS audio
        print("[JourneyVideoGenerator] Generating TTS audio...")
        tts = TTSTool(main_config)

        # Get intro music path
        intro_music = main_config.get("audio", {}).get("intro_music")

        audio_result = tts.generate_segmented_audio(
            intro=script.get("intro", ""),
            stories=script.get("stories", []),
            outro=script.get("outro", ""),
            intro_music_path=intro_music
        )

        result["audio_path"] = audio_result["combined_path"]
        result["audio_segments"] = audio_result["segments"]
        result["total_duration"] = audio_result["total_duration"]

        # Generate video
        print("[JourneyVideoGenerator] Generating video...")
        video_gen = VideoGenerator(main_config)

        # Build ticker text from stats
        ticker_text = (
            f"This Week: {stats['total_runs']} runs completed • "
            f"{stats['total_coins_formatted']} coins earned • "
            f"Highest wave: {stats['max_wave_formatted']} • "
            f"{stats['total_cells_formatted']} cells farmed • "
            f"GB Kill Rate: {stats['gb_kill_pct']:.0f}% • "
            "Tower Journey Progress Update"
        )

        # Use screenshots if provided, otherwise use fallback backgrounds
        if not screenshots:
            # Get any images from the battles
            screenshots = []
            for battle in battles:
                img_path = battle.get("image_path")
                if img_path and Path(img_path).exists():
                    screenshots.append(img_path)

            # If still no screenshots, use fallback
            if not screenshots:
                fallback_dir = Path("assets/backgrounds")
                if fallback_dir.exists():
                    screenshots = [str(f) for f in fallback_dir.glob("*.png")][:3]
                    if not screenshots:
                        screenshots = [str(f) for f in fallback_dir.glob("*.jpg")][:3]

        # Ensure we have enough screenshots for the stories
        while len(screenshots) < len(script.get("stories", [])):
            if screenshots:
                screenshots.append(screenshots[-1])  # Duplicate last image
            else:
                break

        # Generate the video
        try:
            video_result = video_gen.run(
                audio_segments=audio_result["segments"],
                screenshots=screenshots,
                ticker_text=ticker_text,
                segmented=True
            )
            result["video_path"] = video_result.get("video_path")
        except Exception as e:
            print(f"[JourneyVideoGenerator] Error generating video: {e}")
            result["error"] = str(e)

        return result

    def generate_weekly_video(
        self,
        days: int = 7,
        output_path: str = None
    ) -> dict:
        """Generate a weekly progress video from recent battles.

        Args:
            days: Number of days to include (default: 7)
            output_path: Optional output path

        Returns:
            Video generation results
        """
        from datetime import datetime, timedelta

        # Load battle history
        history_path = Path("data/journey/battle_history.json")
        if not history_path.exists():
            return {"error": "No battle history found"}

        with open(history_path, 'r') as f:
            history = json.load(f)

        battles = history.get("battles", [])
        if not battles:
            return {"error": "No battles in history"}

        # Filter to recent battles
        cutoff = datetime.now() - timedelta(days=days)
        recent_battles = []

        for battle in battles:
            added_at = battle.get("added_at", "")
            try:
                battle_time = datetime.fromisoformat(added_at)
                if battle_time >= cutoff:
                    recent_battles.append(battle)
            except (ValueError, TypeError):
                # Include if we can't parse the date
                recent_battles.append(battle)

        if not recent_battles:
            return {"error": f"No battles in the last {days} days"}

        return self.generate_video(recent_battles, output_path=output_path)
