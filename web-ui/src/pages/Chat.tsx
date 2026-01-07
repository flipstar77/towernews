import { useState, useRef, useEffect, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Send, Trash2, Loader2 } from 'lucide-react';
import { marked } from 'marked';
import { fetchStats, sendChatMessage } from '../api';
import { ChatMessage } from '../components';
import type { ChatMessage as ChatMessageType } from '../types';
import { cn } from '../lib/utils';

const EXAMPLE_QUESTIONS = [
  'What is the best Death Wave build?',
  'How do Labs work?',
  'Best modules for damage?',
  'How to farm coins efficiently?',
];

const WELCOME_MESSAGE: ChatMessageType = {
  role: 'assistant',
  content: `Welcome! I can help you with strategies, builds, and tips for The Tower game.
Ask me anything about Death Wave builds, Lab upgrades, modules, or game mechanics!`,
};

export function Chat() {
  const [messages, setMessages] = useState<ChatMessageType[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
  });

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = async () => {
    const message = input.trim();
    if (!message || isLoading) return;

    // Add user message
    const userMessage: ChatMessageType = { role: 'user', content: message };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Build history for context
      const history = messages
        .filter((m) => m !== WELCOME_MESSAGE)
        .map((m) => ({ role: m.role, content: m.content }));

      const response = await sendChatMessage(message, history);

      if (response.error) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${response.error}` },
        ]);
      } else {
        // Parse markdown
        const htmlContent = await marked.parse(response.response);
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: htmlContent,
            sources: response.sources,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([WELCOME_MESSAGE]);
  };

  const handleExampleClick = (question: string) => {
    setInput(question);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-6 flex flex-col">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl sm:text-3xl font-bold text-blue-400">Tower Knowledge Chat</h1>
          <p className="text-gray-400 mt-1">Ask anything about The Tower game</p>
          {stats && (
            <div className="text-sm text-gray-500">
              Knowledge base: {stats.total_documents.toLocaleString()} documents
            </div>
          )}
        </div>

        {/* Chat Container */}
        <div className="flex-1 overflow-y-auto bg-gray-800 rounded-lg p-4 mb-4 min-h-[400px] max-h-[60vh]">
          <div className="space-y-4">
            {messages.map((message, idx) => (
              <ChatMessage key={idx} message={message} />
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-700 rounded-lg px-4 py-3">
                  <div className="flex items-center gap-2 text-blue-400">
                    <Loader2 className="animate-spin" size={16} />
                    <span>Searching knowledge base...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about The Tower game..."
            className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className={cn(
              'px-4 sm:px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2',
              isLoading || !input.trim()
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            )}
          >
            <Send size={18} />
            <span className="hidden sm:inline">Send</span>
          </button>
          <button
            onClick={handleClear}
            className="px-4 py-3 bg-gray-600 hover:bg-gray-500 text-white rounded-lg transition-colors"
            title="Clear chat"
          >
            <Trash2 size={18} />
          </button>
        </div>

        {/* Example Questions */}
        <div className="mt-4 text-sm text-gray-400 flex flex-wrap items-center gap-2">
          <span>Try:</span>
          {EXAMPLE_QUESTIONS.map((question) => (
            <button
              key={question}
              onClick={() => handleExampleClick(question)}
              className="text-blue-400 hover:underline hover:text-blue-300 transition-colors"
            >
              {question.split(' ').slice(0, 3).join(' ')}...
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
