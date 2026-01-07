import { useQuery } from '@tanstack/react-query';
import { fetchRecentThreads } from '../api';
import { getFlairClass } from '../lib/utils';

export function ThreadBanner() {
  const { data: threads = [] } = useQuery({
    queryKey: ['recent-threads'],
    queryFn: fetchRecentThreads,
    refetchInterval: 60000,
  });

  if (threads.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-800 border-b border-gray-700">
      <div className="max-w-7xl mx-auto px-6 py-2">
        <div className="flex items-center gap-4">
          <span className="text-gray-400 text-sm font-medium shrink-0">
            Recent Threads:
          </span>
          <div className="overflow-hidden whitespace-nowrap flex-1">
            <div className="inline-block animate-scroll-left">
              {threads.map((thread, idx) => (
                <span key={thread.post_id || idx}>
                  <a
                    href={thread.url || '#'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 mx-4 hover:opacity-80 transition"
                  >
                    <span className={`${getFlairClass(thread.flair)} text-white text-xs px-2 py-0.5 rounded`}>
                      {thread.flair}
                    </span>
                    <span className="text-gray-300">{thread.title}</span>
                    <span className="text-gray-500 text-sm">({thread.score}↑)</span>
                  </a>
                  {idx < threads.length - 1 && (
                    <span className="text-gray-600 mx-2">•</span>
                  )}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
