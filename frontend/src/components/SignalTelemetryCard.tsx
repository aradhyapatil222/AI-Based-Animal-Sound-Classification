import React from 'react';
import type { SignalStats } from '../types';
import { Cpu, Clock, Activity, Zap, Radio } from 'lucide-react';

interface SignalTelemetryCardProps {
  stats: SignalStats | null;
  executionTimeMs: number;
}

export const SignalTelemetryCard: React.FC<SignalTelemetryCardProps> = ({ stats, executionTimeMs }) => {
  if (!stats) return null;

  return (
    <div className="glass-card p-5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider">
          <Radio className="w-4 h-4 text-electric-cyan" />
          Acoustic Signal Diagnostics & Telemetry
        </h4>
        <span className="text-[10px] font-mono text-slate-500">Live Hardware Telemetry</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        
        {/* Duration */}
        <div className="p-3 rounded-xl bg-obsidian-950/60 border border-slate-800/80 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
            <Clock className="w-3 h-3 text-electric-cyan" />
            <span>Clip Duration</span>
          </div>
          <div className="text-base font-extrabold font-mono text-white">
            {stats.duration_seconds}s
          </div>
          <div className="text-[9px] font-mono text-slate-500">
            {stats.num_samples.toLocaleString()} Samples
          </div>
        </div>

        {/* RMS Signal Power */}
        <div className="p-3 rounded-xl bg-obsidian-950/60 border border-slate-800/80 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
            <Activity className="w-3 h-3 text-electric-indigo" />
            <span>RMS Power</span>
          </div>
          <div className="text-base font-extrabold font-mono text-white">
            {stats.rms_power.toFixed(5)}
          </div>
          <div className="text-[9px] font-mono text-slate-500">
            Peak: {stats.peak_amplitude.toFixed(3)}
          </div>
        </div>

        {/* Voting Windows */}
        <div className="p-3 rounded-xl bg-obsidian-950/60 border border-slate-800/80 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
            <Cpu className="w-3 h-3 text-emerald-400" />
            <span>Voting Windows</span>
          </div>
          <div className="text-base font-extrabold font-mono text-white">
            {stats.num_windows_voted} {stats.num_windows_voted === 1 ? 'Win' : 'Wins'}
          </div>
          <div className="text-[9px] font-mono text-slate-500">
            3.0s Sliding Window
          </div>
        </div>

        {/* Inference Latency */}
        <div className="p-3 rounded-xl bg-obsidian-950/60 border border-slate-800/80 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
            <Zap className="w-3 h-3 text-amber-400" />
            <span>Inference Latency</span>
          </div>
          <div className="text-base font-extrabold font-mono text-electric-cyan">
            {executionTimeMs} ms
          </div>
          <div className="text-[9px] font-mono text-slate-500">
            {stats.sample_rate} Hz Mono
          </div>
        </div>

      </div>
    </div>
  );
};
