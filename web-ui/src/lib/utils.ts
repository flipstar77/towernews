import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getFlairClass(flair: string): string {
  const f = flair.toLowerCase();
  if (f.includes('discussion')) return 'flair-discussion';
  if (f.includes('question') || f.includes('help')) return 'flair-question';
  if (f.includes('guide') || f.includes('tip')) return 'flair-guide';
  if (f.includes('meme') || f.includes('humor')) return 'flair-meme';
  if (f.includes('achievement') || f.includes('flex')) return 'flair-achievement';
  if (f.includes('suggestion') || f.includes('idea')) return 'flair-suggestion';
  if (f.includes('bug') || f.includes('issue')) return 'flair-bug';
  return 'flair-general';
}

export function getTypeColor(type: string): string {
  switch (type) {
    case 'wiki': return 'bg-purple-600';
    case 'post': return 'bg-blue-600';
    case 'comment': return 'bg-green-600';
    default: return 'bg-gray-600';
  }
}

export function getFlairColor(flair?: string): string {
  const f = (flair || '').toLowerCase();
  if (f.includes('discussion')) return 'bg-blue-500';
  if (f.includes('question') || f.includes('help')) return 'bg-green-500';
  if (f.includes('guide') || f.includes('tip')) return 'bg-purple-500';
  if (f.includes('meme') || f.includes('humor')) return 'bg-yellow-500';
  if (f.includes('achievement') || f.includes('flex')) return 'bg-pink-500';
  if (f.includes('info') || f.includes('news')) return 'bg-cyan-500';
  return 'bg-gray-500';
}

export function formatNumber(num: number): string {
  return num.toLocaleString();
}
