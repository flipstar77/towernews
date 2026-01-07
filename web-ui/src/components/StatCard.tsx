import { cn, formatNumber } from '../lib/utils';

interface StatCardProps {
  title: string;
  value: number | string;
  subtitle: string;
  colorClass?: string;
  loading?: boolean;
}

export function StatCard({ title, value, subtitle, colorClass = 'text-blue-400', loading }: StatCardProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h3 className="text-gray-400 text-sm mb-2">{title}</h3>
      <p className={cn('text-3xl font-bold', colorClass)}>
        {loading ? (
          <span className="inline-block w-16 h-8 bg-gray-700 rounded animate-pulse" />
        ) : (
          typeof value === 'number' ? formatNumber(value) : value
        )}
      </p>
      <p className="text-gray-500 text-sm mt-1">{subtitle}</p>
    </div>
  );
}
