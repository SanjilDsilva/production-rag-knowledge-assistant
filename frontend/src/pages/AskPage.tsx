import React from 'react';
import { ChatMessage, DevUser, PipelineTrace } from '../types/rag';
import { CitationChip } from '../components/CitationChip';
import { PipelineView } from '../components/PipelineView';

interface AskPageProps {
  currentUser: DevUser;
  messages: ChatMessage[];
  loading: boolean;
  draftQuestion: string;
  onDraftChange: (text: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  countdown: number | null;
  error: string | null;
  activeTrace: PipelineTrace | null;
  onSelectTrace: (trace: PipelineTrace) => void;
}

const SAMPLE_QUESTIONS = [
  'What are the details of CVE-2021-44228?',
  'What mitigations are recommended for log4j?',
  'Summarize the impact of this vulnerability.',
];

export const AskPage: React.FC<AskPageProps> = ({
  currentUser,
  messages,
  loading,
  draftQuestion,
  onDraftChange,
  onSubmit,
  countdown,
  error,
  activeTrace,
  onSelectTrace,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!loading && draftQuestion.trim() && countdown == null) {
        onSubmit(e);
      }
    }
  };
  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Conversational Chat Interface (7 cols) */}
        <div className="lg:col-span-7 flex flex-col space-y-4">
          {/* Header Info */}
          <div className="rounded-xl border border-slate-800 bg-slate-850 p-4 shadow-sm flex items-center justify-between">
            <div>
              <h1 className="text-base font-bold text-white">Ask Your Documents</h1>
              <p className="text-xs text-slate-400">
                Persona: <span className="font-semibold text-blue-400">{currentUser}</span> • Grounded with citations
              </p>
            </div>
            <div className="text-xs text-slate-400 font-mono">
              {messages.length} {messages.length === 1 ? 'message' : 'messages'}
            </div>
          </div>

          {/* Chat Messages Stream */}
          <div className="flex-1 space-y-4 min-h-[350px] max-h-[550px] overflow-y-auto rounded-xl border border-slate-800 bg-slate-900/60 p-4 shadow-inner">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-72 text-center text-slate-400 text-xs">
                <span className="text-3xl mb-2">💬</span>
                <p className="font-medium text-slate-300">No questions asked yet in this session.</p>
                <p className="mt-1 text-slate-500">Pick a sample question below or type your own to see the pipeline run!</p>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
                >
                  <div
                    className={`max-w-[88%] rounded-2xl p-4 text-xs leading-relaxed shadow-md ${
                      msg.role === 'user'
                        ? 'bg-blue-600 text-white rounded-br-none'
                        : 'bg-slate-800 border border-slate-700/80 text-slate-100 rounded-bl-none'
                    }`}
                  >
                    {/* Role / Name Header */}
                    <div className="flex items-center justify-between mb-1.5 text-[10px] opacity-75 font-mono">
                      <span>{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                      <span>{msg.timestamp}</span>
                    </div>

                    {/* Degraded Alert inside assistant bubble */}
                    {msg.degraded && (
                      <div className="mb-2 rounded bg-amber-500/20 border border-amber-500/30 p-2 text-[11px] text-amber-200">
                        ℹ️ <strong>Degraded Mode:</strong> Global quota cap reached. Displaying direct retrieved passages below.
                      </div>
                    )}

                    {/* Message Body */}
                    <div className="whitespace-pre-wrap">{msg.content}</div>

                    {/* Citations list */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-3 border-t border-slate-700/70 pt-2.5">
                        <div className="text-[10px] font-semibold text-slate-400 mb-1">
                          Retrieved Sources & Citations:
                        </div>
                        <div className="flex flex-wrap">
                          {msg.citations.map((c, i) => (
                            <CitationChip key={c.chunk_id || i} citation={c} index={i} />
                          ))}
                        </div>
                      </div>
                    )}

                    {/* View Trace Button for Assistant messages */}
                    {msg.trace && (
                      <div className="mt-2.5 flex justify-end">
                        <button
                          type="button"
                          onClick={() => onSelectTrace(msg.trace!)}
                          className="text-[10px] text-cyan-400 hover:text-cyan-300 font-mono underline hover:no-underline"
                        >
                          🔍 Inspect execution trace ({msg.trace.total_ms}ms)
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {/* In-Flight Loading Bubble */}
            {loading && (
              <div className="flex items-start">
                <div className="rounded-2xl rounded-bl-none border border-slate-700 bg-slate-800 p-4 text-xs text-slate-300 shadow-md">
                  <div className="flex items-center space-x-2">
                    <span className="h-2 w-2 rounded-full bg-blue-400 animate-ping"></span>
                    <span>Retrieving passages & synthesizing grounded answer...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Rate Limit Alert */}
          {countdown !== null && (
            <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-300">
              ⚠️ <strong>Rate limit active:</strong> Please wait {countdown} seconds before sending another question.
            </div>
          )}

          {/* Error Alert */}
          {error && countdown === null && (
            <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-300">
              ❌ <strong>Error:</strong> {error}
            </div>
          )}

          {/* Sample Questions Pills */}
          <div className="flex flex-wrap gap-1.5 pt-1">
            {SAMPLE_QUESTIONS.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onDraftChange(sample)}
                className="rounded-md border border-slate-700/80 bg-slate-800/80 px-2.5 py-1 text-[11px] text-slate-300 hover:border-blue-500 hover:text-white transition-colors"
              >
                {sample}
              </button>
            ))}
          </div>

          {/* Question Input Form (Pinned at bottom) */}
          <form onSubmit={onSubmit} className="rounded-xl border border-slate-800 bg-slate-850 p-3 shadow-md">
            <textarea
              rows={2}
              maxLength={1000}
              value={draftQuestion}
              onChange={(e) => onDraftChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents (persists across tabs)..."
              className="w-full rounded-lg border border-slate-700 bg-slate-900 p-2.5 text-xs text-slate-100 placeholder-slate-500 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
            />
            <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
              <span>{draftQuestion.length} / 1000 characters</span>
              <button
                type="submit"
                disabled={loading || !draftQuestion.trim() || countdown !== null}
                className="rounded-lg bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white shadow hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Reasoning...' : countdown !== null ? `Wait ${countdown}s` : 'Send Question'}
              </button>
            </div>
          </form>
        </div>

        {/* Right Column: Interactive Pipeline Flowchart (5 cols) */}
        <div className="lg:col-span-5">
          {activeTrace ? (
            <PipelineView trace={activeTrace} />
          ) : (
            <div className="rounded-2xl border border-dashed border-slate-800 bg-slate-900/40 p-12 text-center text-slate-500 text-xs flex flex-col items-center justify-center h-96">
              <span className="text-3xl mb-3 opacity-60">📊</span>
              <p className="font-semibold text-slate-400">Pipeline Flowchart Standby</p>
              <p className="mt-1 max-w-xs text-slate-500 text-[11px]">
                Submit a question to watch requests traverse the security boundary, BGE embeddings, and Qdrant hybrid retrieval in real time.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
