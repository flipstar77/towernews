import { useQuery } from '@tanstack/react-query';
import { History } from 'lucide-react';
import { fetchHistory } from '../api';

export function HistoryList() {
  const { data: history = [], isLoading } = useQuery({
    queryKey: ['history'],
    queryFn: fetchHistory,
  });

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
        <History size={20} />
        Reported Posts History
      </h2>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="bg-gray-700 rounded px-3 py-2 animate-pulse">
                <div className="h-5 bg-gray-600 rounded w-full" />
              </div>
            ))}
          </div>
        ) : history.length === 0 ? (
          <p className="text-gray-500">No posts reported yet</p>
        ) : (
          history.map((item) => (
            <div
              key={item.id}
              className="flex items-center justify-between bg-gray-700 rounded px-3 py-2 hover:bg-gray-650 transition-colors"
            >
              <span className="text-white truncate flex-1">{item.title}</span>
              <span className="text-gray-400 text-sm ml-4 shrink-0">
                {item.video_date || item.reported_at?.split('T')[0] || ''}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
