import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { OnboardingModal } from './components/OnboardingModal';
import { AskPage } from './pages/AskPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { DashboardPage } from './pages/DashboardPage';
import { api } from './api/client';
import { ChatMessage, DevUser, PipelineTrace, RateLimitError } from './types/rag';

export default function App() {
  const [currentUser, setCurrentUser] = useState<DevUser>('user-a');
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draftQuestion, setDraftQuestion] = useState<string>('');
  const [activeTrace, setActiveTrace] = useState<PipelineTrace | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Active countdown timer when rate-limited (HTTP 429)
  useEffect(() => {
    if (countdown === null || countdown <= 0) return;
    const timer = setInterval(() => {
      setCountdown((prev) => (prev && prev > 1 ? prev - 1 : null));
    }, 1000);
    return () => clearInterval(timer);
  }, [countdown]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = draftQuestion.trim();
    if (!query || loading || countdown !== null) return;

    const userMessage: ChatMessage = {
      id: String(Date.now()),
      role: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setDraftQuestion('');
    setLoading(true);
    setError(null);

    try {
      const response = await api.askQuestion(query, currentUser);

      const assistantMessage: ChatMessage = {
        id: String(Date.now() + 1),
        role: 'assistant',
        content: response.answer,
        citations: response.citations || [],
        degraded: Boolean(response.degraded),
        trace: response.trace || null,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      if (response.trace) {
        setActiveTrace(response.trace);
      }
    } catch (err: unknown) {
      if (typeof err === 'object' && err !== null && 'isRateLimited' in err) {
        const rateErr = err as RateLimitError;
        setCountdown(rateErr.retryAfterSeconds);
        setError(`Rate limit reached. Please wait ${rateErr.retryAfterSeconds}s.`);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      <Navbar
        currentUser={currentUser}
        onUserChange={setCurrentUser}
        onOpenOnboarding={() => setIsModalOpen(true)}
      />

      <OnboardingModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        currentPersona={currentUser}
        onSelectPersona={setCurrentUser}
      />

      <main className="flex-1">
        <Routes>
          <Route
            path="/"
            element={
              <AskPage
                currentUser={currentUser}
                messages={messages}
                loading={loading}
                draftQuestion={draftQuestion}
                onDraftChange={setDraftQuestion}
                onSubmit={handleSendMessage}
                countdown={countdown}
                error={error}
                activeTrace={activeTrace}
                onSelectTrace={setActiveTrace}
              />
            }
          />
          <Route path="/documents" element={<DocumentsPage currentUser={currentUser} />} />
          <Route
            path="/dashboard"
            element={
              <DashboardPage
                currentUser={currentUser}
                onUserChange={setCurrentUser}
              />
            }
          />
          <Route
            path="/about"
            element={
              <div className="mx-auto max-w-3xl px-4 py-8">
                <div className="rounded-xl border border-slate-800 bg-slate-850 p-6 shadow-xl">
                  <h1 className="text-xl font-bold text-white mb-3">About This RAG Assistant</h1>
                  <p className="text-sm text-slate-300 leading-relaxed mb-4">
                    An upload-first, multi-tenant document question-answering assistant built with
                    FastAPI, Qdrant hybrid vector search, Reciprocal Rank Fusion (RRF), and React.
                  </p>
                  <div className="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3 text-xs text-blue-200">
                    🔒 <strong>Privacy Notice:</strong> Free-tier AI providers may process text according to their terms.
                    Only public or sample benchmark documents are indexed in this demonstration.
                  </div>
                </div>
              </div>
            }
          />
        </Routes>
      </main>
    </div>
  );
}
