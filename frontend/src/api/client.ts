import { DevUser, QuestionResponse, RateLimitError } from '../types/rag';

export interface DocumentItem {
  document_id: string;
  original_filename: string;
  status: 'queued' | 'processing' | 'ready' | 'failed';
  error_message?: string | null;
  visibility?: 'private' | 'tenant_shared';
  created_at?: string;
}

export interface DocumentListResponse {
  documents: DocumentItem[];
}

export interface UploadDocumentResponse {
  document_id: string;
  filename: string;
  status: string;
}

class ApiClient {
  private baseUrl = '/api';

  /**
   * Helper to execute fetch requests with consistent headers and error handling.
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    user: DevUser = 'user-a'
  ): Promise<T> {
    const headers = new Headers(options.headers || {});
    headers.set('X_Dev_User', user);

    // If body is not FormData, ensure Content-Type is application/json
    if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers,
    });

    if (response.status === 429) {
      const retryAfterHeader = response.headers.get('Retry-After');
      const waitSeconds = retryAfterHeader ? parseInt(retryAfterHeader, 10) : 30;
      const error: RateLimitError = {
        isRateLimited: true,
        retryAfterSeconds: isNaN(waitSeconds) ? 30 : waitSeconds,
        message: 'Rate limit exceeded. Please wait before asking again.',
      };
      throw error;
    }

    if (!response.ok) {
      let errorMessage = `Request failed with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = typeof errorData.detail === 'string' 
            ? errorData.detail 
            : JSON.stringify(errorData.detail);
        }
      } catch {
        // Response was not JSON
      }
      throw new Error(errorMessage);
    }

    return response.json() as Promise<T>;
  }

  /**
   * Ask a question against the user's ready documents.
   */
  async askQuestion(question: string, user: DevUser = 'user-a'): Promise<QuestionResponse> {
    return this.request<QuestionResponse>(
      '/questions',
      {
        method: 'POST',
        body: JSON.stringify({ question }),
      },
      user
    );
  }

  /**
   * List all documents accessible to the current user/tenant.
   */
  async listDocuments(user: DevUser = 'user-a'): Promise<DocumentListResponse> {
    return this.request<DocumentListResponse>('/documents', { method: 'GET' }, user);
  }

  /**
   * Upload a new document file (PDF, TXT, DOCX, CSV).
   */
  async uploadDocument(
    file: File,
    visibility: 'private' | 'tenant_shared' = 'private',
    user: DevUser = 'user-a'
  ): Promise<UploadDocumentResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('visibility', visibility);

    return this.request<UploadDocumentResponse>(
      '/documents',
      {
        method: 'POST',
        body: formData,
      },
      user
    );
  }

  /**
   * Delete a document and all its indexed vector chunks.
   */
  async deleteDocument(documentId: string, user: DevUser = 'user-a'): Promise<{ document_id: string; status: string }> {
    return this.request<{ document_id: string; status: string }>(
      `/documents/${documentId}`,
      { method: 'DELETE' },
      user
    );
  }
}

export const api = new ApiClient();
