import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, DocumentItem } from '../api/client';
import { DevUser } from '../types/rag';

interface DocumentsPageProps {
  currentUser: DevUser;
}

const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MiB
const ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.csv'];

export const DocumentsPage: React.FC<DocumentsPageProps> = ({ currentUser }) => {
  const queryClient = useQueryClient();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [visibility, setVisibility] = useState<'private' | 'tenant_shared'>('private');
  const [clientError, setClientError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);

  // Poll documents every 2 seconds if any document is in 'queued' or 'processing' status
  const { data, isLoading, error: fetchError } = useQuery({
    queryKey: ['documents', currentUser],
    queryFn: () => api.listDocuments(currentUser),
    refetchInterval: (query) => {
      const docs = query.state.data?.documents || [];
      const hasActive = docs.some(
        (doc: DocumentItem) => doc.status === 'queued' || doc.status === 'processing'
      );
      return hasActive ? 2000 : false;
    },
  });

  // Mutation for uploading a document
  const uploadMutation = useMutation({
    mutationFn: ({ file, vis }: { file: File; vis: 'private' | 'tenant_shared' }) =>
      api.uploadDocument(file, vis, currentUser),
    onSuccess: (result) => {
      setSelectedFile(null);
      setUploadSuccess(`Uploaded "${result.filename}" successfully (Status: ${result.status})`);
      setClientError(null);
      queryClient.invalidateQueries({ queryKey: ['documents', currentUser] });
    },
    onError: (err: Error) => {
      setClientError(err.message || 'Upload failed.');
      setUploadSuccess(null);
    },
  });

  // Mutation for deleting a document
  const deleteMutation = useMutation({
    mutationFn: (documentId: string) => api.deleteDocument(documentId, currentUser),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents', currentUser] });
    },
    onError: (err: Error) => {
      setClientError(`Delete failed: ${err.message}`);
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setClientError(null);
    setUploadSuccess(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    // Client-side extension check
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setClientError(
        `Invalid file type "${extension}". Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`
      );
      setSelectedFile(null);
      return;
    }

    // Client-side size check (25MB limit)
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setClientError(
        `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Limit is 25 MB.`
      );
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  const handleUploadSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;
    uploadMutation.mutate({ file: selectedFile, vis: visibility });
  };

  const documents = data?.documents || [];

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      {/* Upload Section */}
      <div className="rounded-xl border border-slate-800 bg-slate-850 p-6 shadow-xl mb-8">
        <h1 className="text-xl font-bold text-white mb-2">Upload Documents</h1>
        <p className="text-xs text-slate-400 mb-6">
          Files are verified for size & format, chunked, embedded with BGE, and indexed into Qdrant in the background.
        </p>

        <form onSubmit={handleUploadSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* File Picker */}
            <div className="md:col-span-2">
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Choose Document (.pdf, .txt, .docx, .csv)
              </label>
              <input
                type="file"
                accept=".pdf,.txt,.docx,.csv"
                onChange={handleFileChange}
                className="w-full text-xs text-slate-400 file:mr-3 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 file:cursor-pointer border border-slate-700 rounded-lg p-1.5 bg-slate-900"
              />
            </div>

            {/* Visibility Selector */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Visibility
              </label>
              <select
                value={visibility}
                onChange={(e) => setVisibility(e.target.value as 'private' | 'tenant_shared')}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 p-2 text-xs text-slate-200 outline-none focus:border-blue-500"
              >
                <option value="private">Private (Only me)</option>
                <option value="tenant_shared">Tenant-Shared (All team)</option>
              </select>
            </div>
          </div>

          {/* Validation / Error Notices */}
          {clientError && (
            <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-300">
              ❌ {clientError}
            </div>
          )}

          {uploadSuccess && (
            <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3 text-xs text-emerald-300">
              ✅ {uploadSuccess}
            </div>
          )}

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={!selectedFile || uploadMutation.isPending}
              className="rounded-lg bg-blue-600 px-5 py-2 text-xs font-semibold text-white shadow-md hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {uploadMutation.isPending ? 'Uploading & Queuing...' : 'Upload & Index'}
            </button>
          </div>
        </form>
      </div>

      {/* Document Status List */}
      <div className="rounded-xl border border-slate-800 bg-slate-850 p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-bold text-white">Your Documents</h2>
            <p className="text-xs text-slate-400">
              Live status updates automatically via background polling.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Total: {documents.length}
          </span>
        </div>

        {isLoading ? (
          <div className="text-center py-8 text-xs text-slate-400">Loading documents...</div>
        ) : fetchError ? (
          <div className="text-center py-8 text-xs text-rose-400">
            Failed to load documents: {(fetchError as Error).message}
          </div>
        ) : documents.length === 0 ? (
          <div className="text-center py-12 border border-dashed border-slate-800 rounded-lg text-slate-400 text-xs">
            No documents uploaded yet for <strong>{currentUser}</strong>. Upload a file above to begin!
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="border-b border-slate-800 text-slate-400 uppercase text-[10px] tracking-wider">
                <tr>
                  <th className="py-2.5 px-3">Filename</th>
                  <th className="py-2.5 px-3">Visibility</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Document ID</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-sans">
                {documents.map((doc) => (
                  <tr key={doc.document_id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-3 px-3 font-medium text-white max-w-[200px] truncate">
                      {doc.original_filename}
                    </td>
                    <td className="py-3 px-3">
                      <span className="rounded px-2 py-0.5 text-[10px] font-medium bg-slate-800 text-slate-300 border border-slate-700">
                        {doc.visibility || 'private'}
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      {doc.status === 'ready' && (
                        <span className="inline-flex items-center space-x-1 text-emerald-400 font-medium">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
                          <span>Ready</span>
                        </span>
                      )}
                      {doc.status === 'processing' && (
                        <span className="inline-flex items-center space-x-1 text-blue-400 font-medium animate-pulse">
                          <span className="h-1.5 w-1.5 rounded-full bg-blue-400"></span>
                          <span>Processing...</span>
                        </span>
                      )}
                      {doc.status === 'queued' && (
                        <span className="inline-flex items-center space-x-1 text-amber-400 font-medium">
                          <span className="h-1.5 w-1.5 rounded-full bg-amber-400"></span>
                          <span>Queued</span>
                        </span>
                      )}
                      {doc.status === 'failed' && (
                        <div className="flex flex-col">
                          <span className="inline-flex items-center space-x-1 text-rose-400 font-medium">
                            <span className="h-1.5 w-1.5 rounded-full bg-rose-400"></span>
                            <span>Failed</span>
                          </span>
                          {doc.error_message && (
                            <span
                              className="text-[10px] text-rose-300/80 mt-0.5 max-w-[220px] truncate"
                              title={doc.error_message}
                            >
                              {doc.error_message}
                            </span>
                          )}
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-3 font-mono text-[10px] text-slate-400">
                      {doc.document_id.slice(0, 12)}...
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        type="button"
                        onClick={() => deleteMutation.mutate(doc.document_id)}
                        disabled={deleteMutation.isPending}
                        className="text-rose-400 hover:text-rose-300 font-semibold text-[11px] hover:underline disabled:opacity-50"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
