import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Play, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { fetchPipelineStatus, runPipeline } from '../api';
import { cn } from '../lib/utils';

export function PipelineControl() {
  const queryClient = useQueryClient();
  const [isPolling, setIsPolling] = useState(false);

  const { data: status } = useQuery({
    queryKey: ['pipeline-status'],
    queryFn: fetchPipelineStatus,
    refetchInterval: isPolling ? 2000 : false,
  });

  const mutation = useMutation({
    mutationFn: runPipeline,
    onSuccess: () => {
      setIsPolling(true);
    },
  });

  useEffect(() => {
    if (status && !status.running && isPolling) {
      setIsPolling(false);
      queryClient.invalidateQueries({ queryKey: ['videos'] });
      queryClient.invalidateQueries({ queryKey: ['history'] });
    }
  }, [status, isPolling, queryClient]);

  const getIndicatorClass = () => {
    if (status?.running) return 'bg-yellow-400 animate-pulse-dot';
    if (status?.error) return 'bg-red-500';
    return 'bg-green-500';
  };

  const getStatusIcon = () => {
    if (status?.running) return <Loader2 className="animate-spin" size={16} />;
    if (status?.error) return <XCircle size={16} />;
    return <CheckCircle size={16} />;
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
        <span>Video Pipeline</span>
        <span className={cn('w-3 h-3 rounded-full', getIndicatorClass())} />
      </h2>

      <div className="mb-6">
        <button
          onClick={() => mutation.mutate()}
          disabled={status?.running || mutation.isPending}
          className={cn(
            'w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors',
            status?.running || mutation.isPending
              ? 'bg-gray-600 cursor-not-allowed text-gray-400'
              : 'bg-blue-600 hover:bg-blue-700 text-white'
          )}
        >
          {status?.running ? (
            <>
              <Loader2 className="animate-spin" size={20} />
              Running...
            </>
          ) : (
            <>
              <Play size={20} />
              Generate New Video
            </>
          )}
        </button>
      </div>

      <div className="bg-gray-700 rounded-lg p-4 mb-4">
        <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
          {getStatusIcon()}
          <span>Status</span>
        </div>
        <p className="text-white">{status?.progress || 'Ready'}</p>
      </div>

      {status?.last_video && (
        <div>
          <p className="text-gray-400 text-sm mb-2">Last Generated Video</p>
          <a
            href={status.last_video}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:underline break-all"
          >
            {status.last_video}
          </a>
        </div>
      )}
    </div>
  );
}
