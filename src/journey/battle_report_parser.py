"""Battle Report Parser - Parses text-based battle reports."""

import re
from typing import Dict, Any


class BattleReportParser:
    """Parses Tower game battle report text into structured data."""

    # ALL fields from battle report organized by category
    FIELD_MAPPINGS = {
        # Run Info
        "battle date": {"field": "battle_date", "category": "run_info", "label": "Battle Date"},
        "game time": {"field": "game_time", "category": "run_info", "label": "Game Time"},
        "real time": {"field": "real_time", "category": "run_info", "label": "Real Time"},
        "tier": {"field": "tier", "category": "run_info", "label": "Tier"},
        "wave": {"field": "wave", "category": "run_info", "label": "Wave"},
        "killed by": {"field": "killed_by", "category": "run_info", "label": "Killed By"},

        # Economy
        "coins earned": {"field": "coins_earned", "category": "economy", "label": "Coins Earned"},
        "coins per hour": {"field": "coins_per_hour", "category": "economy", "label": "Coins/Hour"},
        "cash earned": {"field": "cash_earned", "category": "economy", "label": "Cash Earned"},
        "interest earned": {"field": "interest_earned", "category": "economy", "label": "Interest"},
        "gem blocks tapped": {"field": "gem_blocks", "category": "economy", "label": "Gem Blocks"},
        "cells earned": {"field": "cells_earned", "category": "economy", "label": "Cells Earned"},
        "cells per hour": {"field": "cells_per_hour", "category": "economy", "label": "Cells/Hour"},
        "reroll shards earned": {"field": "reroll_shards", "category": "economy", "label": "Reroll Shards"},

        # Combat - Damage
        "damage dealt": {"field": "damage_dealt", "category": "combat", "label": "Damage Dealt"},
        "damage taken": {"field": "damage_taken", "category": "combat", "label": "Damage Taken"},
        "damage taken wall": {"field": "damage_taken_wall", "category": "combat", "label": "Wall Damage"},
        "damage taken while berserked": {"field": "damage_berserked", "category": "combat", "label": "Berserk Damage"},
        "damage gain from berserk": {"field": "berserk_multiplier", "category": "combat", "label": "Berserk Mult"},
        "death defy": {"field": "death_defy", "category": "combat", "label": "Death Defy"},
        "lifesteal": {"field": "lifesteal", "category": "combat", "label": "Lifesteal"},
        "projectiles damage": {"field": "projectiles_damage", "category": "combat", "label": "Projectile Dmg"},
        "projectiles count": {"field": "projectiles_count", "category": "combat", "label": "Projectiles"},
        "thorn damage": {"field": "thorn_damage", "category": "combat", "label": "Thorn Damage"},
        "orb damage": {"field": "orb_damage", "category": "combat", "label": "Orb Damage"},
        "enemies hit by orbs": {"field": "orb_hits", "category": "combat", "label": "Orb Hits"},
        "land mine damage": {"field": "landmine_damage", "category": "combat", "label": "Land Mine Dmg"},
        "land mines spawned": {"field": "landmines_spawned", "category": "combat", "label": "Mines Spawned"},
        "smart missile damage": {"field": "smart_missile_damage", "category": "combat", "label": "Smart Missile"},
        "inner land mine damage": {"field": "inner_landmine_damage", "category": "combat", "label": "Inner Mine Dmg"},
        "chain lightning damage": {"field": "chain_lightning_damage", "category": "combat", "label": "Chain Lightning"},
        "death wave damage": {"field": "death_wave_damage", "category": "combat", "label": "Death Wave Dmg"},
        "tagged by deathwave": {"field": "deathwave_tagged", "category": "combat", "label": "DW Tagged"},
        "black hole damage": {"field": "black_hole_damage", "category": "combat", "label": "Black Hole Dmg"},

        # Utility
        "waves skipped": {"field": "waves_skipped", "category": "utility", "label": "Waves Skipped"},
        "recovery packages": {"field": "recovery_packages", "category": "utility", "label": "Recovery Pkg"},
        "free attack upgrade": {"field": "free_attack", "category": "utility", "label": "Free Attack"},
        "free defense upgrade": {"field": "free_defense", "category": "utility", "label": "Free Defense"},
        "free utility upgrade": {"field": "free_utility", "category": "utility", "label": "Free Utility"},
        "coins from death wave": {"field": "coins_deathwave", "category": "utility", "label": "DW Coins"},
        "cash from golden tower": {"field": "cash_golden_tower", "category": "utility", "label": "GT Cash"},
        "coins from golden tower": {"field": "coins_golden_tower", "category": "utility", "label": "GT Coins"},
        "coins from black hole": {"field": "coins_black_hole", "category": "utility", "label": "BH Coins"},
        "coins from spotlight": {"field": "coins_spotlight", "category": "utility", "label": "SL Coins"},
        "coins from coin upgrade": {"field": "coins_upgrade", "category": "utility", "label": "Upgrade Coins"},
        "coins from coin bonuses": {"field": "coins_bonuses", "category": "utility", "label": "Bonus Coins"},

        # Enemies
        "total enemies": {"field": "total_enemies", "category": "enemies", "label": "Total Enemies"},
        "basic": {"field": "enemies_basic", "category": "enemies", "label": "Basic"},
        "fast": {"field": "enemies_fast", "category": "enemies", "label": "Fast"},
        "tank": {"field": "enemies_tank", "category": "enemies", "label": "Tank"},
        "ranged": {"field": "enemies_ranged", "category": "enemies", "label": "Ranged"},
        "boss": {"field": "enemies_boss", "category": "enemies", "label": "Boss"},
        "protector": {"field": "enemies_protector", "category": "enemies", "label": "Protector"},
        "total elites": {"field": "enemies_elites", "category": "enemies", "label": "Elites"},
        "vampires": {"field": "elite_vampires", "category": "enemies", "label": "Vampires"},
        "rays": {"field": "elite_rays", "category": "enemies", "label": "Rays"},
        "scatters": {"field": "elite_scatters", "category": "enemies", "label": "Scatters"},
        "destroyed by orbs": {"field": "killed_by_orbs", "category": "enemies", "label": "Orb Kills"},
        "destroyed by thorns": {"field": "killed_by_thorns", "category": "enemies", "label": "Thorn Kills"},
        "destroyed by land mine": {"field": "killed_by_landmine", "category": "enemies", "label": "Mine Kills"},
        "destroyed in spotlight": {"field": "killed_in_spotlight", "category": "enemies", "label": "Spotlight Kills"},

        # Bots
        "golden bot coins earned": {"field": "golden_bot_coins", "category": "bots", "label": "Golden Bot Coins"},
        "destroyed in golden bot": {"field": "golden_bot_kills", "category": "bots", "label": "Golden Bot Kills"},

        # Guardian
        "coins fetched": {"field": "guardian_coins", "category": "guardian", "label": "Guardian Coins"},
        "gems": {"field": "guardian_gems", "category": "guardian", "label": "Guardian Gems"},
        "medals": {"field": "guardian_medals", "category": "guardian", "label": "Guardian Medals"},
        "reroll shards": {"field": "guardian_reroll", "category": "guardian", "label": "Guardian Reroll"},
        "common modules": {"field": "modules_common", "category": "guardian", "label": "Common Modules"},
        "rare modules": {"field": "modules_rare", "category": "guardian", "label": "Rare Modules"},
    }

    # Categories for display
    CATEGORIES = {
        "run_info": "Run Info",
        "economy": "Economy",
        "combat": "Combat",
        "utility": "Utility",
        "enemies": "Enemies Destroyed",
        "bots": "Bots",
        "guardian": "Guardian"
    }

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a battle report text into structured data.

        Args:
            text: The raw battle report text

        Returns:
            dict with 'screen_type', 'values', 'values_by_category', 'summary'
        """
        lines = text.strip().split('\n')
        values = {}
        values_by_category = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to split by tab or multiple spaces
            parts = re.split(r'\t+|\s{2,}', line)
            if len(parts) >= 2:
                label = parts[0].strip().lower()
                value = parts[-1].strip()

                # Check if this is a field we want
                for text_key, field_info in self.FIELD_MAPPINGS.items():
                    if text_key in label:
                        parsed = self._parse_value(value)
                        field_name = field_info["field"]
                        category = field_info["category"]

                        field_data = {
                            "value": value,
                            "numeric": parsed["numeric"],
                            "label": field_info["label"],
                            "category": category
                        }

                        values[field_name] = field_data

                        # Group by category
                        if category not in values_by_category:
                            values_by_category[category] = {}
                        values_by_category[category][field_name] = field_data
                        break

        # Detect if this is a tournament run
        tier_value = values.get("tier", {}).get("value", "?")
        is_tournament = self._is_tournament_run(tier_value)

        # Calculate cells per hour if we have cells earned and real time
        cells_per_hour_value = "?"
        cells_per_hour_numeric = 0
        cells_earned_numeric = values.get("cells_earned", {}).get("numeric", 0)
        real_time_str = values.get("real_time", {}).get("value", "")

        if cells_earned_numeric > 0 and real_time_str:
            hours = self._parse_time_to_hours(real_time_str)
            if hours > 0:
                cells_per_hour_numeric = cells_earned_numeric / hours
                cells_per_hour_value = self._format_number(cells_per_hour_numeric)

                # Add to values
                cells_per_hour_data = {
                    "value": cells_per_hour_value,
                    "numeric": cells_per_hour_numeric,
                    "label": "Cells/Hour",
                    "category": "economy"
                }
                values["cells_per_hour"] = cells_per_hour_data
                if "economy" in values_by_category:
                    values_by_category["economy"]["cells_per_hour"] = cells_per_hour_data

        # Create summary (key stats for card display)
        summary = {
            "tier": tier_value,
            "wave": values.get("wave", {}).get("value", "?"),
            "coins_earned": values.get("coins_earned", {}).get("value", "?"),
            "coins_per_hour": values.get("coins_per_hour", {}).get("value", "?"),
            "cells_earned": values.get("cells_earned", {}).get("value", "?"),
            "cells_per_hour": cells_per_hour_value,
            "battle_date": values.get("battle_date", {}).get("value", "Unknown"),
            "is_tournament": is_tournament,
        }

        return {
            "screen_type": "battle_report",
            "screen_name": "Battle Report",
            "confidence": 1.0,
            "values": values,
            "values_by_category": values_by_category,
            "summary": summary,
            "raw_text": text
        }

    def _is_tournament_run(self, tier_str: str) -> bool:
        """
        Detect if a run is a tournament based on tier.

        Tournament runs are:
        - Tier 17 or higher
        - Any tier with a + sign (e.g., "13+", "17+")
        """
        if not tier_str or tier_str == "?":
            return False

        # Check for + sign (tournament indicator)
        if "+" in str(tier_str):
            return True

        # Check if tier is 17 or higher
        try:
            # Extract numeric part
            tier_num = int(re.sub(r'[^\d]', '', str(tier_str)))
            return tier_num >= 17
        except (ValueError, TypeError):
            return False

    def _parse_value(self, value_str: str) -> Dict[str, Any]:
        """Parse a value string into numeric value."""
        value_str = value_str.strip()

        # Handle time formats (1d 18h 55m 26s)
        time_match = re.match(r'(\d+d\s*)?(\d+h\s*)?(\d+m\s*)?(\d+s)?', value_str)
        if time_match and any(time_match.groups()):
            # Return as string, not numeric
            return {"numeric": 0, "is_time": True}

        # Remove $ sign
        clean = value_str.replace('$', '').replace(',', '').strip()

        # Handle suffixes
        multipliers = {
            'K': 1_000,
            'M': 1_000_000,
            'B': 1_000_000_000,
            'T': 1_000_000_000_000,
            'Q': 1_000_000_000_000_000,  # quadrillion
            'q': 1_000_000_000_000_000_000,  # quintillion
            'S': 1_000_000_000_000_000_000_000,  # sextillion
            'O': 1_000_000_000_000_000_000_000_000,  # octillion
            'N': 1_000_000_000_000_000_000_000_000_000,  # nonillion
            'D': 1_000_000_000_000_000_000_000_000_000_000,  # decillion
        }

        for suffix, multiplier in multipliers.items():
            if clean.upper().endswith(suffix.upper()):
                try:
                    num = float(clean[:-1])
                    return {"numeric": num * multiplier}
                except ValueError:
                    pass

        # Try direct number
        try:
            return {"numeric": float(clean)}
        except ValueError:
            return {"numeric": 0}

    def _parse_time_to_hours(self, time_str: str) -> float:
        """Parse a time string like '6h 35m 40s' or '1d 8h 13m 42s' to hours."""
        hours = 0.0
        time_str = time_str.strip()

        # Match days
        day_match = re.search(r'(\d+)d', time_str)
        if day_match:
            hours += int(day_match.group(1)) * 24

        # Match hours
        hour_match = re.search(r'(\d+)h', time_str)
        if hour_match:
            hours += int(hour_match.group(1))

        # Match minutes
        min_match = re.search(r'(\d+)m', time_str)
        if min_match:
            hours += int(min_match.group(1)) / 60

        # Match seconds
        sec_match = re.search(r'(\d+)s', time_str)
        if sec_match:
            hours += int(sec_match.group(1)) / 3600

        return hours

    def _format_number(self, n: float) -> str:
        """Format a number with K/M suffix."""
        if n >= 1_000_000:
            return f"{n/1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n/1_000:.2f}K"
        else:
            return f"{n:.0f}"

    def get_summary_fields(self) -> list:
        """Get list of fields that should be shown in summary."""
        return [
            "tier", "wave", "coins_earned", "coins_per_hour",
            "damage_dealt", "total_enemies", "waves_skipped"
        ]
