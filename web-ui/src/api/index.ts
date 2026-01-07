import type { Stats, ChatResponse, Video, HistoryItem, PipelineStatus, Thread } from '../types';

const API_BASE = '/api';

// Stats
export async function fetchStats(): Promise<Stats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Failed to fetch stats');
  return res.json();
}

// Chat
export async function sendChatMessage(
  message: string,
  history: { role: string; content: string }[]
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) throw new Error('Failed to send message');
  return res.json();
}

// Videos
export async function fetchVideos(): Promise<Video[]> {
  const res = await fetch(`${API_BASE}/videos`);
  if (!res.ok) throw new Error('Failed to fetch videos');
  return res.json();
}

// History
export async function fetchHistory(): Promise<HistoryItem[]> {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

// Pipeline
export async function fetchPipelineStatus(): Promise<PipelineStatus> {
  const res = await fetch(`${API_BASE}/pipeline/status`);
  if (!res.ok) throw new Error('Failed to fetch pipeline status');
  return res.json();
}

export async function runPipeline(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/pipeline/run`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to start pipeline');
  return res.json();
}

// Recent threads
export async function fetchRecentThreads(): Promise<Thread[]> {
  const res = await fetch(`${API_BASE}/recent-threads`);
  if (!res.ok) throw new Error('Failed to fetch recent threads');
  return res.json();
}
