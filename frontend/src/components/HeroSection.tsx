import React from 'react';
import { Activity } from 'lucide-react';

const ANIMAL_CHIPS = [
  { label: 'Cat', emoji: '🐱' },
  { label: 'Chicken', emoji: '🐔' },
  { label: 'Cow', emoji: '🐮' },
  { label: 'Cricket', emoji: '🦗' },
  { label: 'Dog', emoji: '🐶' },
  { label: 'Donkey', emoji: '🐴' },
  { label: 'Elephant', emoji: '🐘' },
  { label: 'Frog', emoji: '🐸' },
  { label: 'Lion', emoji: '🦁' },
  { label: 'Monkey', emoji: '🐒' },
  { label: 'Pig', emoji: '🐷' },
  { label: 'Sheep', emoji: '🐑' },
  { label: 'Squirrel', emoji: '🐿️' },
];

export const HeroSection: React.FC = () => {
  return (
    <section className="relative overflow-hidden">
      {/* No max-width/padding wrapper here: <main> in App.tsx already applies
          both, and nesting them double-pads the hero on small screens. */}
      <div>
        
        {/* Ambient glow container */}
        <div className="glass-card relative p-6 sm:p-8 overflow-hidden bg-gradient-to-b from-obsidian-850/80 to-obsidian-900/60 border border-slate-800/80 shadow-2xl">
          {/* Subtle background glow circle */}
          <div className="absolute -top-24 -right-24 w-72 h-72 bg-electric-indigo/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -bottom-24 -left-24 w-72 h-72 bg-electric-cyan/10 rounded-full blur-3xl pointer-events-none" />

          <div className="relative z-10 flex flex-col md:flex-row md:items-center md:justify-between gap-6">
            
            {/* Title & Description */}
            <div className="max-w-2xl">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700/60 text-xs font-mono font-medium text-slate-300 mb-3">
                <Activity className="w-3.5 h-3.5 text-electric-cyan" />
                <span>YAMNet Transfer-Learning Pipeline</span>
              </div>

              <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white leading-tight">
                Real-Time Bioacoustic <span className="text-transparent bg-clip-text bg-gradient-to-r from-electric-cyan via-slate-100 to-electric-indigo">Classification</span>
              </h2>

              <p className="mt-2 text-sm text-slate-400 leading-relaxed">
                Google's <span className="font-mono text-slate-300 font-medium">YAMNet</span> encoder turns each
                1-second window into a <span className="font-mono text-slate-300 font-medium">1024-dim embedding</span>,
                classified by a trained TFLite head with sliding-window voting.
                Upload an audio clip or record live to identify the species.
              </p>
            </div>

            {/* Animal Taxonomy Chips */}
            <div className="flex-1 max-w-md">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-micro-label text-slate-400">13 Target Species</span>
                <span className="text-[11px] font-mono text-slate-500">~91% Top-1</span>
              </div>
              <div className="flex flex-wrap gap-1.5 sm:gap-2">
                {ANIMAL_CHIPS.map((chip) => (
                  <span
                    key={chip.label}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-obsidian-800/80 border border-slate-700/50 text-slate-300 hover:text-white hover:bg-obsidian-750 hover:border-slate-600 transition-all duration-200 cursor-default select-none shadow-sm"
                  >
                    <span className="text-sm">{chip.emoji}</span>
                    <span>{chip.label}</span>
                  </span>
                ))}
              </div>
            </div>

          </div>
        </div>

      </div>
    </section>
  );
};
