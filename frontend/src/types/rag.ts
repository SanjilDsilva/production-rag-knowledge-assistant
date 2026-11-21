export interface Citation {
  chunk_id: string;
  document_name: string;
  location: string;
  text?: string | null;
}

export interface QuestionRequest {
  question: string;
}

export interface PipelineStage {
  name: string;
  status: 'done' | 'running' | 'skipped' | 'failed' | 'hit' | 'miss';
  duration_ms: number;
  detail?: string;
}

export interface PipelineTrace {
  total_ms: number;
  cache_hit: boolean;
  stages: PipelineStage[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  degraded?: boolean;
  trace?: PipelineTrace | null;
  timestamp: string;
}

export interface QuestionResponse {
  answer: string;
  citations: Citation[];
  degraded?: boolean;
  degraded_reason?: string | null;
  trace?: PipelineTrace | null;
}

export type DevUser = 'user-a' | 'user-b';

export interface RateLimitError {
  isRateLimited: true;
  retryAfterSeconds: number;
  message: string;
}