import { memo } from 'react';
import { User, Bot, ExternalLink } from 'lucide-react';
import type { ChatMessage as ChatMessageType, Source } from '../types';
import { cn, getTypeColor, getFlairColor } from '../lib/utils';

interface ChatMessageProps {
  message: ChatMessageType;
}

function SourceCard({ source }: { source: Source }) {
  return (
    <div className="bg-gray-800 rounded-lg p-2 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={cn('text-white text-xs px-2 py-0.5 rounded', getTypeColor(source.type))}>
          {source.type}
        </span>
        {source.flair && (
          <span className={cn('text-white text-xs px-2 py-0.5 rounded', getFlairColor(source.flair))}>
            {source.flair}
          </span>
        )}
        {source.score > 0 && (
          <span className="text-yellow-400 text-xs">↑{source.score}</span>
        )}
      </div>
      <div className="mt-1">
        {source.url ? (
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300 hover:underline inline-flex items-center gap-1"
          >
            {source.title}
            <ExternalLink size={12} />
          </a>
        ) : (
          <span className="text-gray-300">{source.title}</span>
        )}
      </div>
    </div>
  );
}

export const ChatMessage = memo(function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'rounded-lg px-4 py-3 max-w-[85%] md:max-w-[80%]',
          isUser ? 'bg-blue-600' : 'bg-gray-700'
        )}
      >
        <div className={cn('flex items-center gap-2 text-sm mb-1', isUser ? 'text-blue-200' : 'text-blue-400')}>
          {isUser ? <User size={14} /> : <Bot size={14} />}
          <span>{isUser ? 'You' : 'Assistant'}</span>
        </div>
        <div
          className="message-content prose prose-invert prose-sm max-w-none"
          dangerouslySetInnerHTML={{ __html: message.content }}
        />
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-600">
            <div className="text-xs text-gray-400 mb-2 font-semibold">
              Sources ({message.sources.length}):
            </div>
            <div className="space-y-2">
              {message.sources.map((source, idx) => (
                <SourceCard key={idx} source={source} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
