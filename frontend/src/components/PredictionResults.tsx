import React from 'react';
import { Award, Zap, AlertTriangle, BarChart2 } from 'lucide-react';
import type { ClassifyResponse } from '../types';

interface PredictionResultsProps {
  result: ClassifyResponse | null;
  isLoading: boolean;
}

export const PredictionResults: React.FC<PredictionResultsProps> = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[360px] text-center space-y-4">
        <div className="w-14 h-14 rounded-2xl bg-electric-indigo/10 border border-electric-indigo/30 flex items-center justify-center relative">
          <div className="absolute inset-0 rounded-2xl bg-electric-indigo/20 animate-ping opacity-50" />
          <Zap className="w-7 h-7 text-electric-cyan animate-bounce" />
        </div>
        <div>
          <h3 className="text-base font-bold text-white">Extracting Audio Features & Running TFLite Inference</h3>
          <p className="text-xs text-slate-400 font-mono mt-1">Computing Log-Mel Spectrogram & MFCC stacked feature tensor...</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[360px] text-center border-dashed border-slate-800">
        <div className="w-12 h-12 rounded-2xl bg-obsidian-800 border border-slate-700/60 flex items-center justify-center text-slate-500 mb-3">
          <BarChart2 className="w-6 h-6" />
        </div>
        <h3 className="text-sm font-bold text-slate-300">No Inference Execution Active</h3>
        <p className="text-xs text-slate-500 max-w-sm mt-1">
          Upload an audio clip, record from your microphone, or choose a preset sample above to view prediction results.
        </p>
      </div>
    );
  }

  const top = result.top_prediction;

  // Determine confidence tier
  let confidenceColor = 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
  let confidenceLabel = 'High Confidence Match';
  if (top.percentage < 40) {
    confidenceColor = 'text-rose-400 border-rose-500/30 bg-rose-500/10';
    confidenceLabel = 'Low Confidence / Ambient Noise';
  } else if (top.percentage < 70) {
    confidenceColor = 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    confidenceLabel = 'Moderate Confidence';
  }

  return (
    <div className="space-y-6">
      
      {/* Warning banner if silent or short */}
      {result.warning && (
        <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-2.5 text-xs text-amber-200">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold">Diagnostic Warning: </span>
            <span>{result.warning}</span>
          </div>
        </div>
      )}

      {/* Hero Top-1 Prediction Banner */}
      <div className="glass-card p-6 relative overflow-hidden bg-gradient-to-br from-obsidian-850 via-obsidian-900 to-obsidian-950 border border-slate-800 shadow-2xl">
        <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none text-9xl">
          {top.emoji}
        </div>

        <div className="relative z-10 flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-electric-indigo/30 to-electric-cyan/20 border border-electric-cyan/40 flex items-center justify-center text-4xl shadow-glass-glow shrink-0">
              {top.emoji}
            </div>

            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-micro-label text-electric-cyan">Primary Prediction (Rank #1)</span>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold border ${confidenceColor}`}>
                  {confidenceLabel}
                </span>
              </div>

              <h3 className="text-3xl font-black text-white tracking-tight leading-none flex items-center gap-2">
                {top.label}
              </h3>

              <p className="text-xs text-slate-400 mt-1.5 line-clamp-1">
                {top.description}
              </p>
            </div>
          </div>

          {/* Confidence percentage badge */}
          <div className="flex flex-col sm:items-end justify-center pt-4 sm:pt-0 border-t sm:border-t-0 border-slate-800">
            <span className="text-micro-label text-slate-400 mb-1">Confidence Score</span>
            <div className="text-4xl font-extrabold font-mono text-transparent bg-clip-text bg-gradient-to-r from-white via-electric-cyan to-electric-indigo">
              {top.percentage}%
            </div>
            <span className="text-[10px] font-mono text-slate-500 mt-0.5">
              Softmax Probability ({top.probability.toFixed(4)})
            </span>
          </div>
        </div>
      </div>

      {/* Top-3 Ranked Cards & Full 10-Class Probability Breakdown Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Top-3 Podium Card */}
        <div className="glass-card p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider">
              <Award className="w-4 h-4 text-electric-cyan" />
              Top-3 Ranked Classification
            </h4>
            <span className="text-[10px] font-mono text-slate-500">Argmax Top-3</span>
          </div>

          <div className="space-y-3">
            {result.top_3.map((pred, idx) => (
              <div
                key={pred.label}
                className="p-3 rounded-xl bg-obsidian-950/60 border border-slate-800/80 flex items-center justify-between gap-3 hover:border-slate-700/60 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className={`w-6 h-6 rounded-lg text-xs font-bold font-mono flex items-center justify-center ${
                    idx === 0 ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' :
                    idx === 1 ? 'bg-slate-700/40 text-slate-300 border border-slate-600/40' :
                    'bg-amber-800/20 text-amber-500 border border-amber-800/30'
                  }`}>
                    #{idx + 1}
                  </span>
                  <span className="text-xl">{pred.emoji}</span>
                  <div>
                    <span className="text-xs font-bold text-white">{pred.label}</span>
                    <p className="text-[10px] font-mono text-slate-400">Idx: {pred.index}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 flex-1 max-w-[160px]">
                  <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-electric-indigo to-electric-cyan transition-all duration-500"
                      style={{ width: `${pred.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono font-bold text-white w-12 text-right">
                    {pred.percentage}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Full 10-Class Interactive Bar Distribution */}
        <div className="glass-card p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider">
              <BarChart2 className="w-4 h-4 text-electric-indigo" />
              Complete 10-Class Softmax Array
            </h4>
            <span className="text-[10px] font-mono text-slate-500">10 ESC-50 Classes</span>
          </div>

          <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
            {result.predictions.map((item) => (
              <div key={item.label} className="flex items-center gap-2 text-xs">
                <span className="w-4 text-center">{item.emoji}</span>
                <span className="w-16 font-medium text-slate-300 truncate">{item.label}</span>
                <div className="flex-1 h-2 rounded-full bg-slate-800/80 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      item.rank === 1
                        ? 'bg-gradient-to-r from-electric-cyan to-electric-indigo'
                        : 'bg-slate-600/60'
                    }`}
                    style={{ width: `${Math.max(item.percentage, 2)}%` }}
                  />
                </div>
                <span className={`w-12 text-right font-mono text-[11px] ${item.rank === 1 ? 'font-bold text-electric-cyan' : 'text-slate-400'}`}>
                  {item.percentage}%
                </span>
              </div>
            ))}
          </div>
        </div>

      </div>

    </div>
  );
};
