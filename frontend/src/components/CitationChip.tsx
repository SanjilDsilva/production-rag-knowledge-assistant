import React, { useState } from 'react';
import { Citation } from '../types/rag';

interface CitationChipProps {
  citation: Citation;
  index: number;
}

export const CitationChip: React.FC<CitationChipProps> = ({ citation, index }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="inline-block relative m-1">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
          isOpen
            ? 'bg-blue-600 text-white border-blue-500 shadow-sm'
            : 'bg-slate-800 text-slate-300 border-slate-700 hover:border-blue-500 hover:text-white'
        }`}
      >
        <span className="font-semibold text-blue-400">[{index + 1}]</span>
        <span className="truncate max-w-[150px]">{citation.document_name}</span>
        <span className="text-slate-400">({citation.location})</span>
      </button>

      {isOpen && (
        <div className="absolute left-0 mt-2 z-20 w-80 max-w-sm rounded-lg border border-slate-700 bg-slate-800 p-3 shadow-2xl text-xs">
          <div className="flex items-center justify-between border-b border-slate-700 pb-1.5 mb-2 font-medium text-slate-300">
            <span>Source Passage</span>
            <span className="text-slate-400 text-[10px] font-mono">{citation.chunk_id.slice(0, 8)}...</span>
          </div>
          <p className="text-slate-200 leading-relaxed whitespace-pre-wrap max-h-48 overflow-y-auto">
            {citation.text || 'Passage text not available.'}
          </p>
        </div>
      )}
    </div>
  );
};