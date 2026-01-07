import { Link, useLocation } from 'react-router-dom';
import { MessageSquare, LayoutDashboard } from 'lucide-react';
import { cn } from '../lib/utils';

export function Navigation() {
  const location = useLocation();

  const links = [
    { to: '/', label: 'RAG Chat', icon: MessageSquare },
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  ];

  return (
    <nav className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold text-blue-400">Tower News</h1>
        <div className="flex gap-2">
          {links.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={cn(
                'flex items-center gap-2 px-4 py-2 rounded-lg transition-colors',
                location.pathname === to
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-100'
              )}
            >
              <Icon size={18} />
              <span className="hidden sm:inline">{label}</span>
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
