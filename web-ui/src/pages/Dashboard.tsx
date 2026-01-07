import { useQuery } from '@tanstack/react-query';
import { fetchStats } from '../api';
import { StatCard, ThreadBanner, PipelineControl, VideoList, HistoryList } from '../components';

export function Dashboard() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: 30000,
  });

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <ThreadBanner />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6 mb-8">
          <StatCard
            title="Knowledge Base"
            value={stats?.total_documents ?? '-'}
            subtitle="Documents"
            colorClass="text-blue-400"
            loading={isLoading}
          />
          <StatCard
            title="Posts"
            value={stats?.by_type?.post ?? '-'}
            subtitle="Reddit Posts"
            colorClass="text-green-400"
            loading={isLoading}
          />
          <StatCard
            title="Comments"
            value={stats?.by_type?.comment ?? '-'}
            subtitle="Reddit Comments"
            colorClass="text-yellow-400"
            loading={isLoading}
          />
          <StatCard
            title="Wiki"
            value={stats?.by_type?.wiki ?? '-'}
            subtitle="Wiki Pages"
            colorClass="text-purple-400"
            loading={isLoading}
          />
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          <PipelineControl />
          <VideoList />
        </div>

        {/* Post History */}
        <div className="mt-8">
          <HistoryList />
        </div>
      </div>
    </div>
  );
}
