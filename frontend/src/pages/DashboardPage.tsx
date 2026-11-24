import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, DashboardSummaryResponse, AdminDashboardResponse, MemberDashboardResponse } from '../api/client';
import { DevUser } from '../types/rag';

interface DashboardPageProps {
  currentUser: DevUser;
  onUserChange: (user: DevUser) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ currentUser, onUserChange }) => {
  const { data, isLoading, error, refetch, isFetching } = useQuery<DashboardSummaryResponse, Error>({
    queryKey: ['dashboardSummary', currentUser],
    queryFn: () => api.getDashboardSummary(currentUser),
    refetchInterval: 5000, // Poll every 5 seconds for live quota updates
  });

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-2xl font-bold text-white">
              {data?.role === 'admin' ? 'Enterprise Telemetry & Usage' : 'My Usage & Personal Quotas'}
            </h1>
            <span
              className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold border ${
                data?.role === 'admin'
                  ? 'bg-purple-500/10 border-purple-500/30 text-purple-400'
                  : 'bg-blue-500/10 border-blue-500/30 text-blue-400'
              }`}
            >
              {data?.role === 'admin' ? 'Admin Mode' : 'Member Mode'}
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-400">
            {data?.role === 'admin'
              ? 'Real-time multi-tenant observability: document health, question quotas, cache hit rates, and infrastructure status.'
              : 'Transparent view of your personal document access, per-minute throttling limits, and daily platform capacity.'}
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center space-x-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-750 disabled:opacity-50 transition-colors"
          >
            <span className={isFetching ? 'animate-spin' : ''}>🔄</span>
            <span>{isFetching ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="py-16 text-center text-slate-400 text-sm">
          Loading live usage metrics from database...
        </div>
      ) : error ? (
        <div className="rounded-lg border border-rose-500/30 bg-rose-950/20 p-4 text-xs text-rose-300">
          Failed to load dashboard data: {error.message}
        </div>
      ) : data?.role === 'member' ? (
        /* ================= MEMBER VIEW ================= */
        <MemberView data={data} onUserChange={onUserChange} />
      ) : data?.role === 'admin' ? (
        /* ================= ADMIN VIEW ================= */
        <AdminView data={data} />
      ) : null}
    </div>
  );
};

/* -------------------------------------------------------------------------- */
/*                                MEMBER VIEW                                 */
/* -------------------------------------------------------------------------- */
const MemberView: React.FC<{
  data: MemberDashboardResponse;
  onUserChange: (user: DevUser) => void;
}> = ({ data, onUserChange }) => {
  return (
    <div className="space-y-6">
      {/* 3 KPI Cards Grid for Member */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: My Personal Documents */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider">Accessible Documents</span>
              <span className="font-mono text-[10px] text-blue-400">{data.tenant_id}</span>
            </div>
            <div className="text-3xl font-extrabold text-white">
              {data.personal_documents.total}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-[11px]">
            <span className="text-emerald-400">● {data.personal_documents.ready} Ready</span>
            {data.personal_documents.failed > 0 && (
              <span className="text-rose-400">● {data.personal_documents.failed} Failed</span>
            )}
            <span className="text-slate-500">Tenant-isolated</span>
          </div>
        </div>

        {/* Card 2: Personal Throttling SLA */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider">Your Rate Limit</span>
              <span className="font-mono text-[10px] text-emerald-400">Fixed-Window</span>
            </div>
            <div className="text-3xl font-extrabold text-white">
              {data.rate_limits.user_requests_per_minute}
              <span className="text-xs text-slate-400 font-normal ml-1">req / min</span>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-400">
            <span>HTTP 429 backoff with <code className="text-blue-300">Retry-After</code> countdown</span>
          </div>
        </div>

        {/* Card 3: Organization Daily Pool */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider">Platform Daily Pool</span>
              <span className="font-mono text-[10px] text-purple-400">Shared Quota</span>
            </div>
            <div className="text-3xl font-extrabold text-white">
              {data.rate_limits.daily_pool_remaining}
              <span className="text-xs text-slate-400 font-normal ml-1">/ {data.rate_limits.daily_pool_max}</span>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800">
            <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-purple-500 rounded-full"
                style={{
                  width: `${Math.min(
                    100,
                    ((data.rate_limits.daily_pool_max - data.rate_limits.daily_pool_remaining) /
                      data.rate_limits.daily_pool_max) *
                      100
                  )}%`,
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Admin Privilege Callout */}
      <div className="rounded-xl border border-slate-800 bg-slate-850 p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <span>🔒</span>
              <span>Enterprise Admin Telemetry</span>
            </h3>
            <p className="mt-1 text-xs text-slate-400 leading-relaxed max-w-xl">
              You are currently viewing member-level quotas. Platform-wide cache hit rates,
              failover diagnostics, and all-tenant storage metrics require an <strong>Admin</strong> persona.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => onUserChange('user-b')}
              className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white shadow-md hover:bg-blue-500 transition-colors"
            >
              Acme Admin (user-b)
            </button>
            <button
              type="button"
              onClick={() => onUserChange('user-d')}
              className="rounded-lg bg-purple-600 px-3 py-1.5 text-xs font-semibold text-white shadow-md hover:bg-purple-500 transition-colors"
            >
              Stark Admin (user-d)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

/* -------------------------------------------------------------------------- */
/*                                 ADMIN VIEW                                 */
/* -------------------------------------------------------------------------- */
const AdminView: React.FC<{ data: AdminDashboardResponse }> = ({ data }) => {
  return (
    <div className="space-y-6">
      {/* Top KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Tenant Documents */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider">Tenant Documents</span>
              <span className="font-mono text-[10px] text-blue-400">{data.admin_tenant}</span>
            </div>
            <div className="text-3xl font-extrabold text-white">
              {data.documents.tenant.total}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-[11px]">
            <span className="text-emerald-400">● {data.documents.tenant.ready} Ready</span>
            {data.documents.tenant.failed > 0 && (
              <span className="text-rose-400">● {data.documents.tenant.failed} Failed</span>
            )}
            {data.documents.tenant.queued + data.documents.tenant.processing > 0 && (
              <span className="text-amber-400">
                ● {data.documents.tenant.queued + data.documents.tenant.processing} In-flight
              </span>
            )}
          </div>
        </div>

        {/* Card 2: Question Quotas */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider">Daily Quota</span>
              <span className="font-mono text-[10px] text-purple-400">100 / Day Max</span>
            </div>
            <div className="text-3xl font-extrabold text-white">
              {data.requests.daily_questions_remaining}
              <span className="text-xs text-slate-400 font-normal ml-1">remaining</span>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800">
            <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-purple-500 rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(
                    100,
                    (data.requests.daily_questions_used / data.requests.daily_questions_max) * 100
                  )}%`,
                }}
              />
            </div>
            <div className="mt-1.5 flex justify-between text-[10px] text-slate-400">
              <span>{data.requests.daily_questions_used} used today</span>
              <span>{data.requests.daily_questions_max} cap</span>
            </div>
          </div>
        </div>

        {/* Card 3: Cache Performance */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider">Cache Hit Rate</span>
              <span className="font-mono text-[10px] text-emerald-400">&lt; 2ms SLA</span>
            </div>
            <div className="text-3xl font-extrabold text-emerald-400">
              {data.cache.hit_rate_pct}%
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
            <span>Hits: <strong className="text-emerald-400">{data.cache.hits}</strong></span>
            <span>Misses: <strong className="text-slate-200">{data.cache.misses}</strong></span>
          </div>
        </div>

        {/* Card 4: Infrastructure Health */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider">Infrastructure</span>
              <span className="font-mono text-[10px] text-slate-400">v{data.system.app_version}</span>
            </div>
            <div className="space-y-2 mt-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300">Qdrant Vector DB</span>
                <span className="inline-flex items-center space-x-1 text-emerald-400 font-semibold text-[11px]">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
                  <span>{data.system.qdrant}</span>
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300">BGE Embeddings</span>
                <span className="inline-flex items-center space-x-1 text-blue-400 font-semibold text-[11px]">
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-400"></span>
                  <span>{data.system.embedding_model}</span>
                </span>
              </div>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800 text-[10px] text-slate-500">
            Cross-tenant isolation: strict pre-retrieval
          </div>
        </div>
      </div>

      {/* Detailed Telemetry Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System-wide Document Distribution */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-6 shadow-xl">
          <h3 className="text-sm font-bold text-white mb-1">Document Distribution (SQLite app.db)</h3>
          <p className="text-xs text-slate-400 mb-4">
            Comparison of documents owned by <strong>{data.admin_tenant}</strong> vs the entire multi-tenant system.
          </p>

          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900 border border-slate-800">
              <div>
                <div className="text-xs font-semibold text-slate-200">Current Tenant ({data.admin_tenant})</div>
                <div className="text-[11px] text-slate-400">
                  {data.documents.tenant.ready} Ready · {data.documents.tenant.failed} Failed ·{' '}
                  {data.documents.tenant.processing + data.documents.tenant.queued} In-flight
                </div>
              </div>
              <div className="text-lg font-bold text-blue-400">
                {data.documents.tenant.total} files
              </div>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900 border border-slate-800">
              <div>
                <div className="text-xs font-semibold text-slate-200">Platform-Wide Total (All Tenants)</div>
                <div className="text-[11px] text-slate-400">
                  {data.documents.system_total.ready} Ready · {data.documents.system_total.failed} Failed ·{' '}
                  {data.documents.system_total.processing + data.documents.system_total.queued} In-flight
                </div>
              </div>
              <div className="text-lg font-bold text-slate-300">
                {data.documents.system_total.total} files
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-lg bg-blue-500/10 border border-blue-500/20 p-3 text-[11px] text-blue-300 leading-relaxed">
            💡 <strong>Pre-Retrieval Tenancy Proof:</strong> Even though the platform hosts {data.documents.system_total.total} files, regular members only retrieve chunks tagged with their own <code className="bg-slate-900 px-1 py-0.5 rounded">tenant_id</code> and <code className="bg-slate-900 px-1 py-0.5 rounded">user_id</code>.
          </div>
        </div>

        {/* Reliability & Degraded Mode Audit */}
        <div className="rounded-xl border border-slate-800 bg-slate-850 p-6 shadow-xl">
          <h3 className="text-sm font-bold text-white mb-1">Reliability & Quota Protection</h3>
          <p className="text-xs text-slate-400 mb-4">
            Protects the $0 budget against provider outages and traffic spikes without crashing.
          </p>

          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900 border border-slate-800">
              <div>
                <div className="text-xs font-semibold text-slate-200">Rate Limit Degraded Fallbacks</div>
                <div className="text-[11px] text-slate-400">Triggered when traffic reaches global cap</div>
              </div>
              <div className="font-mono text-sm font-bold text-slate-200">
                {data.reliability.degraded_responses?.question_cap || 0}
              </div>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900 border border-slate-800">
              <div>
                <div className="text-xs font-semibold text-slate-200">Provider Outage Fallbacks</div>
                <div className="text-[11px] text-slate-400">Failover between Gemini and Groq</div>
              </div>
              <div className="font-mono text-sm font-bold text-slate-200">
                {data.reliability.degraded_responses?.provider_failure || 0}
              </div>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900 border border-slate-800">
              <div>
                <div className="text-xs font-semibold text-slate-200">User Rate Limit Rule</div>
                <div className="text-[11px] text-slate-400">Fixed-window per-tenant/per-user quota</div>
              </div>
              <div className="font-mono text-xs font-semibold text-slate-300">
                5 req / min (429 Retry-After)
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
