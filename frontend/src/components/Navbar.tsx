import React from 'react';
import { NavLink } from 'react-router-dom';
import { DevUser } from '../types/rag';

interface NavbarProps {
  currentUser: DevUser;
  onUserChange: (user: DevUser) => void;
  onOpenOnboarding: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentUser, onUserChange, onOpenOnboarding }) => {
  const navLinkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? 'bg-blue-600 text-white shadow-sm'
        : 'text-slate-300 hover:text-white hover:bg-slate-800'
    }`;

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-900/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        {/* Brand / Logo */}
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-base font-bold text-white tracking-tight">Production RAG</span>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center space-x-1">
            <NavLink to="/" className={navLinkClass}>
              Ask
            </NavLink>
            <NavLink to="/documents" className={navLinkClass}>
              Documents
            </NavLink>
            <NavLink to="/dashboard" className={navLinkClass}>
              Dashboard
            </NavLink>
            <NavLink to="/about" className={navLinkClass}>
              About
            </NavLink>
          </nav>
        </div>

        {/* Dev Identity Persona Switcher */}
        <div className="flex items-center space-x-2 text-xs">
          <button
            type="button"
            onClick={onOpenOnboarding}
            className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-slate-300 hover:text-white hover:border-blue-500 transition-colors flex items-center space-x-1"
            title="Switch Auth Mode / Persona"
          >
            <span>🔑</span>
            <span className="hidden sm:inline">Auth Mode</span>
          </button>
          <select
            value={currentUser}
            onChange={(e) => onUserChange(e.target.value as DevUser)}
            className="rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1 text-slate-200 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 font-sans"
          >
            <optgroup label="🏢 Acme Corp (Tenant 1)">
              <option value="user-a">user-a (Member)</option>
              <option value="user-b">user-b (Admin)</option>
              <option value="user-c">user-c (Member)</option>
            </optgroup>
            <optgroup label="🛡️ Stark Industries (Tenant 2)">
              <option value="user-d">user-d (Admin)</option>
              <option value="user-e">user-e (Member)</option>
              <option value="user-f">user-f (Member)</option>
            </optgroup>
          </select>
        </div>
      </div>
    </header>
  );
};
