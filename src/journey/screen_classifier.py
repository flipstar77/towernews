"""Screen Classifier - Identifies what type of game screen is shown."""

import os
import base64
from pathlib import Path
from openai import OpenAI


class ScreenClassifier:
    """Classifies Tower game screenshots and extracts all relevant values."""

    SCREEN_TYPES = {
        "alltime_stats": {
            "name": "Alltime Stats",
            "description": "Overall game statistics screen",
            "fields": ["total_coins", "highest_wave", "total_kills", "total_damage", "play_time"]
        },
        "labs": {
            "name": "Laboratory",
            "description": "Lab research screen showing completed/ongoing labs",
            "fields": ["labs_completed", "current_lab", "lab_progress"]
        },
        "tournament": {
            "name": "Tournament",
            "description": "Tournament results or ranking screen",
            "fields": ["tournament_rank", "tournament_tier", "waves_reached", "tournament_points"]
        },
        "run_summary": {
            "name": "Run Summary / Battle Report",
            "description": "End of run summary showing coins, waves, damage, kills etc.",
            "fields": ["wave", "tier", "coins_earned", "damage_dealt", "total_enemies", "game_time"]
        },
        "workshop": {
            "name": "Workshop",
            "description": "Workshop upgrades screen",
            "fields": ["workshop_levels"]
        },
        "ultimate_weapons": {
            "name": "Ultimate Weapons",
            "description": "UW screen showing unlocked weapons",
            "fields": ["uw_unlocked", "uw_levels"]
        },
        "unknown": {
            "name": "Unknown",
            "description": "Could not identify screen type",
            "fields": []
        }
    }

    def __init__(self, config: dict):
        self.config = config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = config.get("llm", {}).get("model", "gpt-4o")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def classify_and_extract(self, image_path: str) -> dict:
        """
        Classify the screen type and extract all relevant values.

        Returns:
            dict with 'screen_type', 'screen_name', 'values', 'confidence'
        """
        base64_image = self.encode_image(image_path)

        prompt = """Analyze this Tower (idle tower defense game) screenshot.

1. IDENTIFY the screen type. Common screens are:
   - Alltime Stats: Shows total coins earned, highest wave, total kills, play time
   - Laboratory: Shows lab researches, completed count
   - Tournament: Shows rank, tier, waves reached
   - Run Summary: End of run with coins/waves earned
   - Workshop: Upgrade levels
   - Ultimate Weapons: UW unlocks and levels

2. EXTRACT all visible numerical values with their labels.

Respond in this exact JSON format:
{
    "screen_type": "alltime_stats|labs|tournament|run_summary|workshop|ultimate_weapons|unknown",
    "confidence": 0.0-1.0,
    "values": {
        "field_name": {"value": "displayed value", "numeric": 123456, "label": "Original Label"}
    }
}

For numbers with suffixes (K, M, B, T, q), include both the displayed value and the full numeric value.
Example: "1.5B" -> {"value": "1.5B", "numeric": 1500000000, "label": "Total Coins"}

Extract ALL visible stats, not just the main ones."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        import json
        result = json.loads(response.choices[0].message.content)

        # Add screen info
        screen_type = result.get("screen_type", "unknown")
        screen_info = self.SCREEN_TYPES.get(screen_type, self.SCREEN_TYPES["unknown"])
        result["screen_name"] = screen_info["name"]
        result["expected_fields"] = screen_info["fields"]

        return result

    def get_screen_types(self) -> dict:
        """Get all known screen types."""
        return self.SCREEN_TYPES
