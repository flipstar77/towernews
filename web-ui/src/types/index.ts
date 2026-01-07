// API Types

export interface Stats {
  total_documents: number;
  by_type: {
    post?: number;
    comment?: number;
    wiki?: number;
  };
}

export interface Source {
  type: 'wiki' | 'post' | 'comment';
  title: string;
  flair?: string;
  score: number;
  similarity: number;
  url?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

export interface ChatResponse {
  response: string;
  sources: Source[];
  error?: string;
}

export interface Video {
  date: string;
  title: string;
  path: string;
  youtube_url?: string;
}

export interface HistoryItem {
  id: string;
  title: string;
  reported_at: string;
  video_date: string;
}

export interface PipelineStatus {
  running: boolean;
  progress: string;
  last_run?: string;
  last_video?: string;
  error?: string;
}

export interface Thread {
  title: string;
  flair: string;
  score: number;
  url: string;
  post_id: string;
}
