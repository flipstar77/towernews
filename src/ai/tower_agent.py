"""Tower Agent - AI that plans game progression strategies."""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI


class TowerAgent:
    """AI Agent that creates and executes Tower game progression plans."""

    def __init__(self, knowledge_base_path: str = None):
        self.client = OpenAI()
        self.knowledge_base = {}
        self.current_state = {}
        self.progression_plan = {}
        self.plans_dir = Path("data/ai/plans")
        self.plans_dir.mkdir(parents=True, exist_ok=True)

        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)

    def load_knowledge_base(self, path: str):
        """Load knowledge base from file."""
        filepath = Path(path)
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"[TowerAgent] Loaded knowledge base: {self.knowledge_base.get('meta', {}).get('total_strategies', 0)} strategies")
        else:
            print(f"[TowerAgent] Knowledge base not found: {path}")

    def set_current_state(self, state: Dict[str, Any]):
        """
        Set the current game state.

        Expected state format:
        {
            "tier": 12,
            "highest_wave": 5000,
            "total_coins": "500B",
            "coins_per_hour": "10B",
            "workshop": {
                "damage": 100,
                "attack_speed": 80,
                ...
            },
            "cards": {
                "total": 45,
                "max_level": 10
            },
            "ultimate_weapons": ["smart_missiles", "poison_swamp"],
            "labs_completed": 25,
            "perks_unlocked": ["orb_speed", "death_ray"],
            "goals": ["reach_tier_15", "improve_coins_per_hour"]
        }
        """
        self.current_state = state
        print(f"[TowerAgent] Game state updated: Tier {state.get('tier', '?')}, Wave {state.get('highest_wave', '?')}")

    def analyze_state(self) -> Dict[str, Any]:
        """Analyze current state and identify areas for improvement."""
        if not self.current_state:
            return {"error": "No game state set. Use set_current_state() first."}

        # Get relevant strategies from knowledge base
        relevant_strategies = self._get_relevant_strategies()

        prompt = f"""Analyze this Tower game state and identify areas for improvement.

Current Game State:
{json.dumps(self.current_state, indent=2)}

Available Knowledge (strategies from experienced players):
{json.dumps(relevant_strategies, indent=2)}

Provide analysis as JSON:
{{
    "current_stage": "early/mid/late game",
    "overall_assessment": "Brief assessment of current progress",
    "strengths": ["List of things player is doing well"],
    "weaknesses": ["Areas that need improvement"],
    "bottlenecks": ["What's currently limiting progress"],
    "priority_improvements": [
        {{
            "area": "workshop/cards/uw/labs/etc",
            "current": "Current state",
            "target": "Recommended target",
            "impact": "high/medium/low",
            "reason": "Why this matters"
        }}
    ],
    "estimated_tier_potential": "Tier X with current setup optimized"
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Tower game analyst. Provide specific, actionable analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _get_relevant_strategies(self) -> Dict[str, List[Dict]]:
        """Get strategies relevant to current game stage."""
        if not self.knowledge_base:
            return {}

        tier = self.current_state.get("tier", 1)

        # Determine game stage
        if tier <= 8:
            stage = "early"
        elif tier <= 14:
            stage = "mid"
        else:
            stage = "late"

        relevant = {}
        categories = self.knowledge_base.get("categories", {})

        for cat_key, cat_data in categories.items():
            strategies = cat_data.get("strategies", [])
            # Filter by game stage
            relevant_strats = [
                s for s in strategies
                if s.get("game_stage", "all") in [stage, "all"]
            ]
            if relevant_strats:
                relevant[cat_key] = relevant_strats[:3]  # Top 3 per category

        return relevant

    def create_progression_plan(
        self,
        time_horizon: str = "week",
        focus_areas: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a detailed progression plan.

        Args:
            time_horizon: "day", "week", or "month"
            focus_areas: Specific areas to focus on (e.g., ["coins_farming", "workshop"])

        Returns:
            Detailed progression plan
        """
        if not self.current_state:
            return {"error": "No game state set. Use set_current_state() first."}

        # First analyze state
        analysis = self.analyze_state()

        # Get relevant strategies
        relevant_strategies = self._get_relevant_strategies()

        # Filter to focus areas if specified
        if focus_areas:
            relevant_strategies = {
                k: v for k, v in relevant_strategies.items()
                if k in focus_areas
            }

        goals = self.current_state.get("goals", ["progress_tiers", "improve_coins"])

        prompt = f"""Create a detailed progression plan for The Tower game.

Current State Analysis:
{json.dumps(analysis, indent=2)}

Current Game State:
{json.dumps(self.current_state, indent=2)}

Player Goals:
{json.dumps(goals, indent=2)}

Time Horizon: {time_horizon}

Available Strategies from Knowledge Base:
{json.dumps(relevant_strategies, indent=2)}

Create a progression plan as JSON:
{{
    "plan_name": "Descriptive name for this plan",
    "time_horizon": "{time_horizon}",
    "current_tier": {self.current_state.get("tier", 1)},
    "target_tier": X,
    "executive_summary": "One paragraph summary of the plan",

    "immediate_actions": [
        {{
            "action": "Specific action to take",
            "category": "workshop/cards/uw/etc",
            "priority": 1-5,
            "expected_impact": "What this will achieve",
            "details": "Step-by-step how to do this"
        }}
    ],

    "daily_routine": {{
        "active_play": [
            {{"activity": "What to do", "duration": "X minutes", "purpose": "Why"}}
        ],
        "afk_strategy": {{
            "wave_target": X,
            "tier": X,
            "expected_coins_per_hour": "X",
            "settings": "Any specific settings"
        }}
    }},

    "weekly_milestones": [
        {{
            "week": 1,
            "targets": ["Target 1", "Target 2"],
            "focus": "Main focus for this week"
        }}
    ],

    "upgrade_priorities": {{
        "workshop": [
            {{"stat": "stat_name", "current": X, "target": X, "priority": 1-5}}
        ],
        "cards": ["Card priorities"],
        "ultimate_weapons": ["UW priorities"],
        "labs": ["Lab priorities"],
        "perks": ["Perk priorities"]
    }},

    "coins_strategy": {{
        "target_coins_per_hour": "X",
        "farming_method": "Best method for current stage",
        "when_to_prestige": "Condition for prestiging"
    }},

    "tournament_strategy": {{
        "tier_to_play": X,
        "target_wave": X,
        "tips": ["Tournament-specific tips"]
    }},

    "warnings": ["Things to avoid or watch out for"],

    "next_major_unlock": {{
        "what": "Next major game unlock",
        "requirements": "What's needed",
        "impact": "Why it matters"
    }}
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert Tower game strategist. Create detailed, actionable progression plans.

Key game mechanics to consider:
- Coins are the main progression currency, earned per wave
- Cells are earned from Death Wave and used for cards
- Workshop upgrades are permanent and crucial
- Cards provide powerful bonuses but need cells
- Ultimate Weapons (UW) are major power spikes
- Labs provide permanent research bonuses
- Perks are unlocked and upgraded with stones
- Tournaments have fixed tiers and give extra rewards
- Higher tiers = more coins but harder waves
- AFK farming at lower waves can be more efficient than pushing"""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )

        plan = json.loads(response.choices[0].message.content)
        self.progression_plan = plan

        return plan

    def get_next_action(self) -> Dict[str, Any]:
        """Get the single most important next action."""
        if not self.progression_plan:
            return {"error": "No plan created. Use create_progression_plan() first."}

        actions = self.progression_plan.get("immediate_actions", [])
        if actions:
            # Sort by priority
            sorted_actions = sorted(actions, key=lambda x: x.get("priority", 99))
            return {
                "next_action": sorted_actions[0],
                "remaining_actions": len(actions) - 1
            }

        return {"error": "No actions in plan"}

    def update_progress(self, completed_action: str, new_state: Dict[str, Any] = None):
        """Update progress after completing an action."""
        if new_state:
            self.set_current_state(new_state)

        # Remove completed action from plan
        if self.progression_plan:
            actions = self.progression_plan.get("immediate_actions", [])
            self.progression_plan["immediate_actions"] = [
                a for a in actions
                if a.get("action") != completed_action
            ]

            # Log completion
            if "completed_actions" not in self.progression_plan:
                self.progression_plan["completed_actions"] = []
            self.progression_plan["completed_actions"].append({
                "action": completed_action,
                "completed_at": datetime.now().isoformat()
            })

    def save_plan(self, filename: str = None) -> str:
        """Save current plan to file."""
        if not self.progression_plan:
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tier = self.current_state.get("tier", "unknown")
            filename = f"plan_tier{tier}_{timestamp}.json"

        filepath = self.plans_dir / filename

        data = {
            "saved_at": datetime.now().isoformat(),
            "game_state": self.current_state,
            "plan": self.progression_plan
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[TowerAgent] Plan saved to {filepath}")
        return str(filepath)

    def load_plan(self, filename: str) -> Dict[str, Any]:
        """Load a saved plan."""
        filepath = self.plans_dir / filename

        if not filepath.exists():
            return {"error": f"Plan not found: {filename}"}

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.current_state = data.get("game_state", {})
        self.progression_plan = data.get("plan", {})

        return self.progression_plan

    def list_plans(self) -> List[Dict[str, Any]]:
        """List all saved plans."""
        plans = []
        for filepath in self.plans_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                plan = data.get("plan", {})
                state = data.get("game_state", {})
                plans.append({
                    "filename": filepath.name,
                    "saved_at": data.get("saved_at", "unknown"),
                    "tier": state.get("tier", "?"),
                    "plan_name": plan.get("plan_name", "Unnamed"),
                    "time_horizon": plan.get("time_horizon", "?")
                })
            except:
                pass
        return sorted(plans, key=lambda x: x.get("saved_at", ""), reverse=True)

    def ask_question(self, question: str) -> str:
        """Ask the AI agent a question about the game."""
        context = ""

        if self.knowledge_base:
            # Find relevant strategies
            relevant = []
            for cat_key, cat_data in self.knowledge_base.get("categories", {}).items():
                for strat in cat_data.get("strategies", [])[:2]:
                    relevant.append(f"- {strat.get('title', '')}: {strat.get('description', '')[:200]}")

            if relevant:
                context = "Relevant knowledge:\n" + "\n".join(relevant[:10])

        if self.current_state:
            context += f"\n\nPlayer's current state: Tier {self.current_state.get('tier', '?')}, Wave {self.current_state.get('highest_wave', '?')}"

        prompt = f"""Answer this Tower game question based on your knowledge and the context provided.

Question: {question}

{context}

Provide a helpful, specific answer. If the question is about strategy, be concrete about what to do."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Tower game advisor. Give specific, actionable advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def generate_daily_checklist(self) -> List[Dict[str, Any]]:
        """Generate a daily checklist based on current plan."""
        if not self.progression_plan:
            # Generate generic checklist
            return [
                {"task": "Collect free daily rewards", "priority": 1, "category": "daily"},
                {"task": "Check tournament status", "priority": 2, "category": "tournament"},
                {"task": "Run AFK farming session", "priority": 3, "category": "farming"},
                {"task": "Spend accumulated coins on workshop", "priority": 4, "category": "upgrades"},
                {"task": "Open any card packs", "priority": 5, "category": "cards"}
            ]

        checklist = []

        # Add from daily routine
        routine = self.progression_plan.get("daily_routine", {})
        for activity in routine.get("active_play", []):
            checklist.append({
                "task": activity.get("activity", ""),
                "duration": activity.get("duration", ""),
                "priority": len(checklist) + 1,
                "category": "active_play"
            })

        # Add immediate actions
        for action in self.progression_plan.get("immediate_actions", [])[:3]:
            checklist.append({
                "task": action.get("action", ""),
                "priority": action.get("priority", 5),
                "category": action.get("category", "general")
            })

        return sorted(checklist, key=lambda x: x.get("priority", 99))
