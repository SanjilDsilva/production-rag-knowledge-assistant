import React, { useState, useEffect } from 'react';
import { PipelineTrace } from '../types/rag';

interface PipelineViewProps {
  trace: PipelineTrace;
}

export const PipelineView: React.FC<PipelineViewProps> = ({ trace }) => {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [isReplaying, setIsReplaying] = useState(false);

  // Step-by-step pulse animation down the tree
  const replayFlow = () => {
    setIsReplaying(true);
    setActiveStep(0);
    const totalSteps = 7;
    let current = 0;
    const interval = setInterval(() => {
      current++;
      if (current >= totalSteps) {
        clearInterval(interval);
        setIsReplaying(false);
      }
      setActiveStep(current);
    }, 400);
  };

  useEffect(() => {
    replayFlow();
  }, [trace]);

  const getStageTiming = (namePattern: string) => {
    const stage = trace.stages.find((s) => s.name.toLowerCase().includes(namePattern.toLowerCase()));
    return stage ? `${stage.duration_ms} ms` : '< 1 ms';
  };

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/95 p-5 shadow-2xl backdrop-blur sticky top-20">
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-5">
        <div>
          <div className="flex items-center space-x-2">
            <span className="h-2 w-2 rounded-full bg-cyan-400 animate-ping"></span>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Pipeline Execution Trace
            </h3>
          </div>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Real boundary timings measured by server events.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={replayFlow}
            disabled={isReplaying}
            className="rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1 text-[11px] font-semibold text-slate-200 hover:border-blue-500 hover:text-white transition-colors disabled:opacity-50"
          >
            {isReplaying ? '⚡ Replaying...' : '▶ Replay'}
          </button>
          <div className="rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1 text-[11px] font-mono font-bold text-cyan-300">
            {trace.total_ms} ms
          </div>
        </div>
      </div>

      {/* Visual Tree / Flowchart */}
      <div className="relative flex flex-col items-center space-y-4 max-w-sm mx-auto py-1">
        {/* Node 1: User Request & Security Scan */}
        <div className="flex flex-col items-center w-full">
          <div
            className={`w-full rounded-lg border p-2.5 text-center transition-all duration-300 shadow-sm ${
              activeStep >= 1
                ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-200 shadow-emerald-500/10'
                : 'border-slate-800 bg-slate-850 text-slate-400'
            }`}
          >
            <div className="text-[9px] font-mono uppercase text-slate-400">Step 1 • Ingestion Boundary</div>
            <div className="text-xs font-bold text-white">🛡️ Query Security Scan</div>
            <div className="mt-0.5 flex items-center justify-center space-x-2 text-[10px] text-slate-300">
              <span className="rounded bg-slate-800 px-1 py-0.2">≤ 1,000 Chars</span>
              <span className="text-cyan-400 font-mono">{getStageTiming('Security')}</span>
            </div>
          </div>
          <div className="h-4 w-0.5 bg-gradient-to-b from-blue-500 to-cyan-500"></div>
          <div className="w-0 h-0 border-x-3 border-x-transparent border-t-3 border-t-cyan-500"></div>
        </div>

        {/* Node 2: Dense Query Embedding */}
        <div className="flex flex-col items-center w-full">
          <div
            className={`w-full rounded-lg border p-2.5 text-center transition-all duration-300 shadow-sm ${
              activeStep >= 2
                ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-200 shadow-emerald-500/10'
                : 'border-slate-800 bg-slate-850 text-slate-400'
            }`}
          >
            <div className="text-[9px] font-mono uppercase text-slate-400">Step 2 • Vectorization</div>
            <div className="text-xs font-bold text-white">🧠 BGE Query Embedding</div>
            <div className="mt-0.5 flex items-center justify-center space-x-2 text-[10px] text-slate-300">
              <span className="rounded bg-slate-800 px-1 py-0.2">384-Dim Vector</span>
              <span className="text-cyan-400 font-mono">{getStageTiming('Embedding')}</span>
            </div>
          </div>
          <div className="h-4 w-0.5 bg-gradient-to-b from-cyan-500 to-indigo-500"></div>
          <div className="w-0 h-0 border-x-3 border-x-transparent border-t-3 border-t-indigo-500"></div>
        </div>

        {/* Node 3: Hybrid Search Parallel Split (Dense + BM25) */}
        <div className="w-full">
          <div className="text-center text-[9px] font-mono uppercase text-indigo-400 mb-1.5">
            Step 3 • Parallel Hybrid Search (Qdrant)
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div
              className={`rounded-lg border p-2 text-center transition-all duration-300 shadow-sm ${
                activeStep >= 3
                  ? 'border-indigo-500/60 bg-indigo-500/10 text-indigo-200'
                  : 'border-slate-800 bg-slate-850 text-slate-400'
              }`}
            >
              <div className="text-[10px] font-bold text-indigo-300">🔍 Vector Search</div>
              <p className="text-[9px] text-slate-400">Dense Meaning</p>
              <div className="text-[9px] font-mono text-cyan-400 mt-0.5">Cosine Match</div>
            </div>

            <div
              className={`rounded-lg border p-2 text-center transition-all duration-300 shadow-sm ${
                activeStep >= 3
                  ? 'border-indigo-500/60 bg-indigo-500/10 text-indigo-200'
                  : 'border-slate-800 bg-slate-850 text-slate-400'
              }`}
            >
              <div className="text-[10px] font-bold text-indigo-300">📝 BM25 Search</div>
              <p className="text-[9px] text-slate-400">Lexical Exact</p>
              <div className="text-[9px] font-mono text-cyan-400 mt-0.5">Frequency Match</div>
            </div>
          </div>

          <div className="flex justify-center mt-1.5">
            <div className="h-4 w-0.5 bg-gradient-to-b from-indigo-500 to-purple-500"></div>
          </div>
          <div className="flex justify-center">
            <div className="w-0 h-0 border-x-3 border-x-transparent border-t-3 border-t-purple-500"></div>
          </div>
        </div>

        {/* Node 4: Reciprocal Rank Fusion (RRF) & Context Budget */}
        <div className="flex flex-col items-center w-full">
          <div
            className={`w-full rounded-lg border p-2.5 text-center transition-all duration-300 shadow-sm ${
              activeStep >= 4
                ? 'border-purple-500/60 bg-purple-500/10 text-purple-200'
                : 'border-slate-800 bg-slate-850 text-slate-400'
            }`}
          >
            <div className="text-[9px] font-mono uppercase text-slate-400">Step 4 • Fusion & Budget</div>
            <div className="text-xs font-bold text-white">⚖️ Reciprocal Rank Fusion (RRF)</div>
            <div className="mt-0.5 flex items-center justify-center space-x-2 text-[10px] text-slate-300">
              <span className="rounded bg-slate-800 px-1 py-0.2">Deduplicated</span>
              <span className="text-cyan-400 font-mono">{getStageTiming('Hybrid')}</span>
            </div>
          </div>
          <div className="h-4 w-0.5 bg-gradient-to-b from-purple-500 to-amber-500"></div>
          <div className="w-0 h-0 border-x-3 border-x-transparent border-t-3 border-t-amber-500"></div>
        </div>

        {/* Node 5: Cache Decision Branch */}
        <div className="w-full">
          <div
            className={`w-full rounded-lg border p-2.5 text-center transition-all duration-300 shadow-sm ${
              trace.cache_hit
                ? 'border-cyan-500/80 bg-cyan-500/15 text-cyan-200 ring-2 ring-cyan-500/40'
                : activeStep >= 5
                ? 'border-slate-700 bg-slate-800 text-slate-200'
                : 'border-slate-800 bg-slate-850 text-slate-400'
            }`}
          >
            <div className="text-[9px] font-mono uppercase text-slate-400">Step 5 • Cache Verification</div>
            <div className="text-xs font-bold text-white">
              {trace.cache_hit ? '⚡ Cache Hit (< 20ms)' : '💾 Cache Miss: Generating'}
            </div>
            <div className="mt-0.5 text-[9px] text-slate-300 font-mono">
              Lookup: {getStageTiming('Cache')}
            </div>
          </div>

          {!trace.cache_hit && (
            <>
              <div className="flex justify-center mt-1.5">
                <div className="h-4 w-0.5 bg-gradient-to-b from-amber-500 to-blue-500"></div>
              </div>
              <div className="flex justify-center">
                <div className="w-0 h-0 border-x-3 border-x-transparent border-t-3 border-t-blue-500"></div>
              </div>
            </>
          )}
        </div>

        {/* Node 6: LLM Generation & Citation Verification (Skipped on Cache Hit) */}
        {!trace.cache_hit ? (
          <div className="flex flex-col items-center w-full">
            <div
              className={`w-full rounded-lg border p-2.5 text-center transition-all duration-300 shadow-sm ${
                activeStep >= 6
                  ? 'border-blue-500/60 bg-blue-500/10 text-blue-200'
                  : 'border-slate-800 bg-slate-850 text-slate-400'
              }`}
            >
              <div className="text-[9px] font-mono uppercase text-slate-400">Step 6 • Generation & Citations</div>
              <div className="text-xs font-bold text-white">🤖 Gemini 3.1 Flash Lite Generation</div>
              <div className="mt-0.5 flex items-center justify-center space-x-2 text-[10px] text-slate-300">
                <span className="rounded bg-slate-800 px-1 py-0.2">Strict ASCII Citations</span>
                <span className="text-cyan-400 font-mono">{getStageTiming('Generation')}</span>
              </div>
            </div>
            <div className="h-4 w-0.5 bg-gradient-to-b from-blue-500 to-emerald-500"></div>
            <div className="w-0 h-0 border-x-3 border-x-transparent border-t-3 border-t-emerald-500"></div>
          </div>
        ) : (
          <div className="w-full text-center py-1">
            <span className="rounded-full border border-dashed border-cyan-500/40 bg-cyan-500/5 px-3 py-1 text-[11px] text-cyan-300">
              ⏩ LLM generation bypassed (Saved ~2s)
            </span>
          </div>
        )}

        {/* Node 7: Output Grounded Answer */}
        <div className="w-full">
          <div
            className={`w-full rounded-lg border p-3 text-center transition-all duration-300 shadow-md ${
              activeStep >= 7 || trace.cache_hit
                ? 'border-emerald-500 bg-emerald-500/15 text-emerald-100 ring-2 ring-emerald-500/30'
                : 'border-slate-800 bg-slate-850 text-slate-400'
            }`}
          >
            <div className="text-[9px] font-mono uppercase text-emerald-400 font-bold">Step 7 • Output Delivered</div>
            <div className="text-xs font-bold text-white">✅ Grounded Answer with Citations</div>
            <p className="text-[10px] text-slate-300 mt-0.5">
              {trace.cache_hit ? 'Served from LRU Cache in < 20ms' : 'Full pipeline verified and logged'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
