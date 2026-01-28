/**
 * Le-Route API Client
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Types
export interface IngestRequest {
  session_id?: string;
  text: string;
  doc_type: 'policy' | 'contract' | 'legal' | 'technical' | 'general';
  title?: string;
}

export interface IngestResponse {
  session_id: string;
  chunks_created: number;
  total_tokens: number;
  doc_type: string;
  status: string;
}

export interface AskRequest {
  session_id: string;
  question: string;
}

export interface ChunkInfo {
  chunk_id: number;
  text: string;
  relevance_score: number;
  highlight_start: number | null;
  highlight_end: number | null;
}

export interface RoutingInfo {
  model_used: string;
  routing_reason: string;
  confidence: number;
  risk_level: 'low' | 'medium' | 'high';
  cost_estimate_usd: number;
  latency_ms: number;
}

export interface AskResponse {
  answer: string;
  citations: ChunkInfo[];
  routing: RoutingInfo;
  abstained: boolean;
}

export interface SessionInfo {
  session_id: string;
  doc_type: string;
  title: string | null;
  chunk_count: number;
  total_tokens: number;
  created_at: string;
}

// API Error
class ApiError extends Error {
  status: number;
  
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

// Helper function
async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_URL}${endpoint}`;
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || 'Request failed');
  }

  return response.json();
}

// API Functions
export async function healthCheck(): Promise<{ status: string; router_type: string; mlp_loaded: boolean }> {
  return fetchApi('/health');
}

export async function ingestDocument(request: IngestRequest): Promise<IngestResponse> {
  return fetchApi('/ingest', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function askQuestion(request: AskRequest): Promise<AskResponse> {
  return fetchApi('/ask', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getSession(sessionId: string): Promise<SessionInfo> {
  return fetchApi(`/session/${sessionId}`);
}

export async function deleteSession(sessionId: string): Promise<void> {
  return fetchApi(`/session/${sessionId}`, {
    method: 'DELETE',
  });
}

export async function listSessions(): Promise<{ sessions: string[] }> {
  return fetchApi('/sessions');
}
