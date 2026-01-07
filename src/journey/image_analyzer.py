"""Image Analyzer - Uses GPT-4o to extract values from game screenshots."""

import os
import base64
from pathlib import Path
from openai import OpenAI


class ImageAnalyzer:
    """Analyzes Tower game screenshots using GPT-4o vision."""

    def __init__(self, config: dict):
        self.config = config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = config.get("llm", {}).get("model", "gpt-4o")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_segment(self, image_path: str, segment_type: str) -> dict:
        """
        Analyze a screenshot and extract the relevant value.

        Args:
            image_path: Path to the screenshot
            segment_type: Type of segment (coins, labs, highest_wave)

        Returns:
            dict with 'value' (raw number), 'formatted' (display string), 'raw_response'
        """
        segment_prompts = {
            "coins": """Look at this Tower game screenshot and find the TOTAL COINS earned value.
                       This is usually displayed as a number with suffix like K, M, B, T, or q.
                       Return ONLY the number and suffix, nothing else. Example: 1.5B or 234M""",

            "labs": """Look at this Tower game screenshot and find the LABS COMPLETED count.
                      This shows how many laboratory researches have been finished.
                      Return ONLY the number, nothing else. Example: 47""",

            "highest_wave": """Look at this Tower game screenshot and find the HIGHEST WAVE reached.
                             This is the best/maximum wave number achieved.
                             Return ONLY the number, nothing else. Example: 12450"""
        }

        prompt = segment_prompts.get(segment_type, f"Find the {segment_type} value in this screenshot. Return ONLY the number.")

        base64_image = self.encode_image(image_path)

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
            max_tokens=100
        )

        raw_value = response.choices[0].message.content.strip()

        # Parse the value
        parsed = self._parse_value(raw_value)

        return {
            "value": parsed["numeric"],
            "formatted": raw_value,
            "raw_response": raw_value
        }

    def _parse_value(self, value_str: str) -> dict:
        """Parse a value string like '1.5B' into numeric value."""
        value_str = value_str.strip().upper()

        multipliers = {
            'K': 1_000,
            'M': 1_000_000,
            'B': 1_000_000_000,
            'T': 1_000_000_000_000,
            'Q': 1_000_000_000_000_000
        }

        # Check for suffix
        for suffix, multiplier in multipliers.items():
            if value_str.endswith(suffix):
                try:
                    num = float(value_str[:-1])
                    return {"numeric": num * multiplier, "suffix": suffix}
                except ValueError:
                    pass

        # Try direct number
        try:
            # Remove commas
            clean = value_str.replace(',', '').replace(' ', '')
            return {"numeric": float(clean), "suffix": None}
        except ValueError:
            return {"numeric": 0, "suffix": None}

    def compare_values(self, old_value: float, new_value: float) -> dict:
        """
        Compare two values and calculate the difference.

        Returns:
            dict with 'difference', 'percentage', 'direction' (up/down/same)
        """
        if old_value == 0:
            percentage = 100 if new_value > 0 else 0
        else:
            percentage = ((new_value - old_value) / old_value) * 100

        difference = new_value - old_value

        if difference > 0:
            direction = "up"
        elif difference < 0:
            direction = "down"
        else:
            direction = "same"

        return {
            "difference": difference,
            "percentage": round(percentage, 1),
            "direction": direction
        }

    def format_number(self, value: float) -> str:
        """Format a large number with appropriate suffix."""
        if value >= 1_000_000_000_000_000:
            return f"{value / 1_000_000_000_000_000:.2f}q"
        elif value >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.2f}T"
        elif value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.2f}K"
        else:
            return str(int(value))
