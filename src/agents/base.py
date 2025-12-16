"""Base Agent - Foundation class for all agents."""

import json
from abc import ABC, abstractmethod
from typing import Any, Callable
from pathlib import Path
from openai import OpenAI


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(
        self,
        name: str,
        config: dict,
        tools: dict[str, Callable] | None = None,
        system_prompt: str | None = None
    ):
        """
        Initialize the agent.

        Args:
            name: Name of the agent
            config: Configuration dictionary
            tools: Dictionary of tool name -> tool instance
            system_prompt: System prompt for the agent
        """
        self.name = name
        self.config = config
        self.tools = tools or {}
        self._client = None

        # Load system prompt
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load system prompt from file."""
        prompt_path = Path("config/prompts") / f"{self.name.lower()}.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return f"You are a helpful {self.name} agent."

    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def _get_tool_definitions(self) -> list[dict]:
        """
        Get OpenAI function definitions for available tools.

        Returns:
            List of tool definitions for OpenAI API
        """
        definitions = []

        tool_schemas = {
            "reddit_scraper": {
                "name": "reddit_scraper",
                "description": "Fetch top posts from a subreddit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subreddit": {
                            "type": "string",
                            "description": "Name of the subreddit to scrape"
                        },
                        "max_posts": {
                            "type": "integer",
                            "description": "Maximum number of posts to return"
                        }
                    },
                    "required": []
                }
            },
            "screenshot": {
                "name": "screenshot",
                "description": "Take screenshots of Reddit posts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "posts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "id": {"type": "string"}
                                }
                            },
                            "description": "List of posts to screenshot"
                        }
                    },
                    "required": ["posts"]
                }
            },
            "tts": {
                "name": "tts",
                "description": "Convert text to speech audio",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to convert to speech"
                        },
                        "intro": {
                            "type": "string",
                            "description": "Intro text for news"
                        },
                        "stories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of story texts"
                        },
                        "outro": {
                            "type": "string",
                            "description": "Outro text for news"
                        }
                    },
                    "required": []
                }
            },
            "video_generator": {
                "name": "video_generator",
                "description": "Generate the final news video",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file"
                        },
                        "screenshots": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of screenshot paths"
                        },
                        "ticker_text": {
                            "type": "string",
                            "description": "Text for the scrolling ticker"
                        }
                    },
                    "required": ["audio_path", "screenshots", "ticker_text"]
                }
            }
        }

        for tool_name in self.tools.keys():
            if tool_name in tool_schemas:
                definitions.append({
                    "type": "function",
                    "function": tool_schemas[tool_name]
                })

        return definitions

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Call a tool by name.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")

        tool = self.tools[tool_name]
        return tool.run(**arguments)

    def chat(self, messages: list[dict], use_tools: bool = True) -> dict:
        """
        Send a chat completion request.

        Args:
            messages: List of message dictionaries
            use_tools: Whether to include tool definitions

        Returns:
            Response message dictionary
        """
        llm_config = self.config.get("llm", {})
        model = llm_config.get("model", "gpt-4o-mini")

        kwargs = {
            "model": model,
            "messages": messages
        }

        if use_tools and self.tools:
            tools = self._get_tool_definitions()
            if tools:
                kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    def process_tool_calls(self, message) -> list[dict]:
        """
        Process tool calls from a message.

        Args:
            message: Message with potential tool calls

        Returns:
            List of tool result messages
        """
        results = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                try:
                    result = self.call_tool(tool_name, arguments)
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": str(e)})
                    })

        return results

    @abstractmethod
    async def run(self, task: dict) -> dict:
        """
        Run the agent with a task.

        Args:
            task: Task dictionary with input data

        Returns:
            Result dictionary
        """
        pass

    def run_sync(self, task: dict) -> dict:
        """
        Synchronous wrapper for run().

        Args:
            task: Task dictionary

        Returns:
            Result dictionary
        """
        import asyncio
        return asyncio.run(self.run(task))
