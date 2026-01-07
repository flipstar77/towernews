import { useQuery } from '@tanstack/react-query';
import { Video, ExternalLink } from 'lucide-react';
import { fetchVideos } from '../api';

export function VideoList() {
  const { data: videos = [], isLoading } = useQuery({
    queryKey: ['videos'],
    queryFn: fetchVideos,
  });

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
        <Video size={20} />
        Recent Videos
      </h2>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-gray-700 rounded-lg p-3 animate-pulse">
                <div className="h-4 bg-gray-600 rounded w-24 mb-2" />
                <div className="h-5 bg-gray-600 rounded w-full" />
              </div>
            ))}
          </div>
        ) : videos.length === 0 ? (
          <p className="text-gray-500">No videos generated yet</p>
        ) : (
          videos.map((video, idx) => (
            <div key={idx} className="bg-gray-700 rounded-lg p-3 hover:bg-gray-650 transition-colors">
              <p className="text-sm text-gray-400">{video.date}</p>
              <p className="text-white font-medium truncate">{video.title}</p>
              {video.youtube_url && (
                <a
                  href={video.youtube_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-blue-400 text-sm hover:underline mt-1"
                >
                  <ExternalLink size={14} />
                  Watch on YouTube
                </a>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
