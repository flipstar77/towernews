"""
Tower Knowledge Chat - Query the knowledge base interactively.
Uses RAG to answer questions about The Tower game.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()

from openai import OpenAI
from rag import UnifiedKnowledgeBase, QueryProcessor


class TowerChat:
    """Interactive chat interface for Tower knowledge base."""

    def __init__(self):
        self.kb = UnifiedKnowledgeBase()
        self.query_processor = QueryProcessor()
        self.openai = OpenAI()
        self.model = "gpt-4o-mini"
        self.conversation_history = []

        # System prompt for the chat
        self.system_prompt = """You are a knowledgeable assistant for "The Tower" mobile idle game.
You help players with strategies, builds, tips, and game mechanics.

You have access to a knowledge base of community discussions and guides from r/TheTowerGame.
Use the provided context to give accurate, helpful answers. If the context doesn't contain
relevant information, say so and provide general advice based on your knowledge.

Guidelines:
- Be concise but thorough
- Use game terminology correctly (e.g., "Death Wave", "Black Hole", "Labs", "UW")
- Mention specific stats, numbers, or percentages when available
- Credit community insights when relevant
- If unsure, acknowledge uncertainty"""

    def search_context(self, query: str, limit: int = 5) -> str:
        """Search knowledge base for relevant context."""
        # Preprocess query - expands abbreviations like DW -> Death Wave
        processed = self.query_processor.process(query)
        results = self.kb.search(processed.expanded, limit=limit)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            score = result.metadata.get("score", 0)
            post_type = result.metadata.get("post_type", "unknown")
            similarity = result.similarity

            context_parts.append(
                f"[Source {i}] ({post_type}, score: {score}, relevance: {similarity:.2f})\n{result.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        # Search for relevant context
        context = self.search_context(user_message, limit=5)

        # Build the messages
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history (last 10 exchanges)
        for msg in self.conversation_history[-20:]:
            messages.append(msg)

        # Add context and user message
        if context:
            user_content = f"""Context from community knowledge base:

{context}

---

User question: {user_message}"""
        else:
            user_content = f"User question: {user_message}\n\n(No relevant context found in knowledge base)"

        messages.append({"role": "user", "content": user_content})

        # Get response
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        assistant_message = response.choices[0].message.content

        # Store in history (simplified version)
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def get_stats(self) -> dict:
        """Get knowledge base stats."""
        return self.kb.get_stats()


def main():
    """Run the interactive chat."""
    print("=" * 60)
    print("Tower Knowledge Chat")
    print("=" * 60)

    chat = TowerChat()
    stats = chat.get_stats()
    print(f"\nKnowledge base: {stats['total_documents']} documents")
    print(f"Types: {stats['by_type']}")
    print("\nType your questions about The Tower game.")
    print("Commands: 'quit' to exit, 'clear' to reset history, 'stats' for KB stats")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                chat.conversation_history = []
                print("Conversation history cleared.")
                continue

            if user_input.lower() == "stats":
                stats = chat.get_stats()
                print(f"Knowledge base: {stats['total_documents']} documents")
                print(f"Types: {stats['by_type']}")
                continue

            # Get response
            print("\nAssistant:", end=" ")
            response = chat.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
