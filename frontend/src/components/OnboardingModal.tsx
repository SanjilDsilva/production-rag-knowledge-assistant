import React, { useState } from 'react';
import { DevUser } from '../types/rag';

interface OnboardingModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentPersona: DevUser;
  onSelectPersona: (user: DevUser) => void;
}

export const OnboardingModal: React.FC<OnboardingModalProps> = ({
  isOpen,
  onClose,
  currentPersona,
  onSelectPersona,
}) => {
  const [selected, setSelected] = useState<DevUser>(currentPersona);
  const [ssoMessage, setSsoMessage] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSsoClick = (provider: string) => {
    setSsoMessage(
      `🔐 ${provider} Integration: In an enterprise environment, this redirects to your company's IdP (Identity Provider) and supplies a signed JWT with your tenant_id. For this portfolio evaluation, select a pre-configured sandbox persona to test instantly!`
    );
  };

  const handleContinue = () => {
    onSelectPersona(selected);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm">
      <div className="w-full max-w-2xl rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-5">
          <div className="flex items-center space-x-2">
            <span className="h-3 w-3 rounded-full bg-blue-500 animate-ping"></span>
            <h2 className="text-base font-bold text-white tracking-wide">
              Welcome to Production RAG Assistant
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-white text-lg font-bold"
          >
            ✕
          </button>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed mb-6">
          How would you like to evaluate this application? Choose between an <strong>Instant Recruiter Sandbox</strong> (zero sign-up friction) or previewing <strong>Enterprise SSO Authentication</strong>.
        </p>

        {/* Two Options Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Card 1: Instant Sandbox (Recommended) */}
          <div className="rounded-xl border border-blue-500/40 bg-blue-500/5 p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center space-x-1.5 mb-2">
                <span className="text-sm">⚡</span>
                <span className="text-xs font-bold text-blue-300 uppercase tracking-wider">
                  Option A: Recruiter Sandbox (Instant)
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mb-3">
                No signups or passwords. Jump directly into isolated tenant accounts to test cross-user and cross-company document boundaries:
              </p>

              <div className="space-y-2 text-xs">
                {/* Acme Corp Group */}
                <div className="rounded-lg bg-slate-850 p-2 border border-slate-800">
                  <span className="text-[10px] font-mono font-bold text-indigo-400 uppercase block mb-1">
                    🏢 Acme Corp (Tenant 1)
                  </span>
                  <div className="grid grid-cols-3 gap-1">
                    {(['user-a', 'user-b', 'user-c'] as DevUser[]).map((u) => (
                      <button
                        key={u}
                        type="button"
                        onClick={() => setSelected(u)}
                        className={`rounded px-2 py-1 text-[11px] font-medium border transition-colors ${
                          selected === u
                            ? 'bg-blue-600 text-white border-blue-400'
                            : 'bg-slate-800 text-slate-300 border-slate-700 hover:border-slate-500'
                        }`}
                      >
                        {u} {u === 'user-b' ? '(Admin)' : '(Member)'}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Stark Industries Group */}
                <div className="rounded-lg bg-slate-850 p-2 border border-slate-800">
                  <span className="text-[10px] font-mono font-bold text-amber-400 uppercase block mb-1">
                    🛡️ Stark Industries (Tenant 2)
                  </span>
                  <div className="grid grid-cols-3 gap-1">
                    {(['user-d', 'user-e', 'user-f'] as DevUser[]).map((u) => (
                      <button
                        key={u}
                        type="button"
                        onClick={() => setSelected(u)}
                        className={`rounded px-2 py-1 text-[11px] font-medium border transition-colors ${
                          selected === u
                            ? 'bg-blue-600 text-white border-blue-400'
                            : 'bg-slate-800 text-slate-300 border-slate-700 hover:border-slate-500'
                        }`}
                      >
                        {u} {u === 'user-d' ? '(Admin)' : '(Member)'}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={handleContinue}
              className="mt-4 w-full rounded-lg bg-blue-600 py-2 text-xs font-semibold text-white shadow-md hover:bg-blue-500 transition-colors"
            >
              Enter as {selected} →
            </button>
          </div>

          {/* Card 2: Enterprise SSO / OAuth Preview */}
          <div className="rounded-xl border border-slate-800 bg-slate-850 p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center space-x-1.5 mb-2">
                <span className="text-sm">🔐</span>
                <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  Option B: Enterprise SSO Preview
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mb-4">
                In corporate production, authentication uses signed OAuth2/OIDC JWT tokens issued by your company identity provider:
              </p>

              <div className="space-y-2">
                <button
                  type="button"
                  onClick={() => handleSsoClick('Google Workspace')}
                  className="w-full flex items-center justify-center space-x-2 rounded-lg border border-slate-700 bg-slate-800 p-2 text-xs text-slate-200 hover:border-slate-500 hover:bg-slate-750 transition-colors"
                >
                  <span>🔵</span>
                  <span>Sign in with Google Workspace</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleSsoClick('Okta SSO')}
                  className="w-full flex items-center justify-center space-x-2 rounded-lg border border-slate-700 bg-slate-800 p-2 text-xs text-slate-200 hover:border-slate-500 hover:bg-slate-750 transition-colors"
                >
                  <span>🔷</span>
                  <span>Sign in with Okta / Entra ID</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleSsoClick('Auth0 / Clerk')}
                  className="w-full flex items-center justify-center space-x-2 rounded-lg border border-slate-700 bg-slate-800 p-2 text-xs text-slate-200 hover:border-slate-500 hover:bg-slate-750 transition-colors"
                >
                  <span>🟣</span>
                  <span>Sign in with Auth0 / Clerk</span>
                </button>
              </div>
            </div>

            {ssoMessage && (
              <div className="mt-3 rounded-lg border border-cyan-500/30 bg-cyan-500/10 p-2.5 text-[11px] text-cyan-200 leading-snug">
                {ssoMessage}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
