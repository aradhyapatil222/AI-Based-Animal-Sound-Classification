import React from 'react';
import { Cpu, Sparkles } from 'lucide-react';
import type { HealthStatus } from '../types';

interface NavbarProps {
  health: HealthStatus | null;
  onOpenSpecs: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ health, onOpenSpecs }) => {
  return (
    <header className="sticky top-0 z-40 w-full bg-obsidian-950/70 backdrop-blur-md border-b border-slate-800/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand Header */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-electric-indigo to-electric-cyan p-[1px] shadow-glass-glow">
            <div className="w-full h-full bg-obsidian-900 rounded-[11px] flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-electric-cyan" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-1.5">
                WildEcho <span className="text-transparent bg-clip-text bg-gradient-to-r from-electric-cyan to-electric-indigo font-extrabold">AI</span>
              </h1>
              <span className="px-1.5 py-0.5 text-[10px] font-mono font-semibold tracking-wide text-electric-cyan bg-electric-cyan/10 border border-electric-cyan/20 rounded-md">
                v2.0 EDGE
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              Acoustic Wildlife Intelligence Platform
            </p>
          </div>
        </div>

        {/* Status Indicators & Specs Button */}
        <div className="flex items-center gap-3 sm:gap-4">
          {/* Health status dot */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-obsidian-850/80 border border-slate-800/80 text-xs">
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${health?.status === 'online' ? 'bg-emerald-400' : 'bg-amber-400'}`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${health?.status === 'online' ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
            </span>
            <span className="font-mono text-[11px] font-medium text-slate-300 hidden md:inline">
              {health?.model_file || 'TFLite Model Active'}
            </span>
            <span className="font-mono text-[11px] font-medium text-slate-300 md:hidden">
              Online
            </span>
          </div>

          {/* Model Architecture Button */}
          <button
            onClick={onOpenSpecs}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-obsidian-800/80 hover:bg-obsidian-750 border border-slate-700/60 text-slate-200 hover:text-white text-xs font-medium transition-all duration-200 shadow-sm active:scale-95"
          >
            <Cpu className="w-3.5 h-3.5 text-electric-indigo" />
            <span className="hidden sm:inline">Model Specs</span>
          </button>
        </div>

      </div>
    </header>
  );
};
