import React, { useRef, useEffect, useState } from 'react';
import type { SpectrogramData } from '../types';
import { Activity } from 'lucide-react';

interface SpectrogramViewerProps {
  spectrogram: SpectrogramData | null;
}

export const SpectrogramViewer: React.FC<SpectrogramViewerProps> = ({ spectrogram }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [colormap, setColormap] = useState<'cyan' | 'viridis' | 'plasma'>('cyan');
  const [hoverInfo, setHoverInfo] = useState<{ frame: number; mel: number; freqHz: number; db: number } | null>(null);

  // Colormap maps value (0..255) to RGB
  const getColor = (val: number, cmap: string) => {
    const t = val / 255;
    if (cmap === 'viridis') {
      // Approximation of Viridis
      const r = Math.floor(255 * (0.471 * t + 0.177 * Math.pow(t, 2)));
      const g = Math.floor(255 * (0.015 + 0.942 * t - 0.407 * Math.pow(t, 2)));
      const b = Math.floor(255 * (0.334 + 0.404 * t + 0.207 * Math.pow(t, 2)));
      return [r, g, b];
    } else if (cmap === 'plasma') {
      // Approximation of Plasma
      const r = Math.floor(255 * (0.05 + 1.2 * t));
      const g = Math.floor(255 * (0.9 * Math.pow(t, 2)));
      const b = Math.floor(255 * (0.5 - 0.4 * t + 0.5 * Math.pow(1 - t, 2)));
      return [Math.min(255, r), Math.min(255, g), Math.max(0, b)];
    } else {
      // Bespoke Obsidian Electric Cyan theme colormap
      const r = Math.floor(255 * (0.02 + 0.38 * t));
      const g = Math.floor(255 * (0.04 + 0.71 * t));
      const b = Math.floor(255 * (0.10 + 0.83 * t));
      return [r, g, b];
    }
  };

  useEffect(() => {
    if (!spectrogram || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const matrix = spectrogram.matrix; // (128, 130)
    const numMels = spectrogram.num_mels; // 128
    const numFrames = spectrogram.num_frames; // 130

    canvas.width = numFrames;
    canvas.height = numMels;

    const imgData = ctx.createImageData(numFrames, numMels);
    const data = imgData.data;

    // Render matrix: index 0 (low mel) at bottom of canvas
    for (let mel = 0; mel < numMels; mel++) {
      const canvasRow = numMels - 1 - mel; // invert y for low frequency at bottom
      for (let frame = 0; frame < numFrames; frame++) {
        const val = matrix[mel][frame] || 0;
        const [r, g, b] = getColor(val, colormap);
        const idx = (canvasRow * numFrames + frame) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [spectrogram, colormap]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!spectrogram || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const frame = Math.floor((x / rect.width) * spectrogram.num_frames);
    const canvasY = Math.floor((y / rect.height) * spectrogram.num_mels);
    const mel = spectrogram.num_mels - 1 - canvasY;

    if (frame >= 0 && frame < spectrogram.num_frames && mel >= 0 && mel < spectrogram.num_mels) {
      const val = spectrogram.matrix[mel][frame] || 0;
      const db = spectrogram.min_db + (val / 255) * (spectrogram.max_db - spectrogram.min_db);
      const freqHz = Math.round(melBandToHz(mel, spectrogram.num_mels));

      setHoverInfo({
        frame,
        mel,
        freqHz,
        db: roundTo(db, 1),
      });
    }
  };

  const roundTo = (num: number, decimals: number) => {
    return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
  };

  // YAMNet's log-mel front end uses 64 mel bands spaced evenly on the mel scale
  // between 125 Hz and 7500 Hz. Convert a band index back to its centre
  // frequency so the hover readout and axis labels reflect the real model input.
  const YAMNET_MEL_MIN_HZ = 125;
  const YAMNET_MEL_MAX_HZ = 7500;
  const hzToMel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
  const melToHz = (mel: number) => 700 * (Math.pow(10, mel / 2595) - 1);

  const melBandToHz = (band: number, numBands: number) => {
    const lo = hzToMel(YAMNET_MEL_MIN_HZ);
    const hi = hzToMel(YAMNET_MEL_MAX_HZ);
    const frac = numBands > 1 ? band / (numBands - 1) : 0;
    return melToHz(lo + frac * (hi - lo));
  };

  const formatHz = (hz: number) =>
    hz >= 1000 ? `${(hz / 1000).toFixed(1)} kHz` : `${Math.round(hz)} Hz`;

  if (!spectrogram) {
    return null;
  }

  return (
    <div className="glass-card p-5 space-y-4">
      
      {/* Header & Colormap selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="min-w-0">
          <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider">
            <Activity className="w-4 h-4 text-electric-cyan shrink-0" />
            Log-Mel Spectrogram (YAMNet Input)
          </h4>
          <p className="text-[10px] text-slate-400 font-mono mt-0.5">
            {spectrogram.num_mels} Mel Bins × {spectrogram.num_frames} Frames • First 1.0 s window @ 16 kHz
          </p>
        </div>

        {/* Colormap Buttons */}
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-obsidian-950/80 border border-slate-800 self-start sm:self-auto shrink-0">
          {[
            { id: 'cyan', label: 'Obsidian' },
            { id: 'viridis', label: 'Viridis' },
            { id: 'plasma', label: 'Plasma' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setColormap(item.id as any)}
              className={`px-2 py-1 rounded text-[10px] font-mono font-medium transition-all ${
                colormap === item.id
                  ? 'bg-electric-indigo/30 text-white border border-electric-indigo/50'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Spectrogram Canvas with Axes */}
      <div className="relative flex">
        
        {/* Y-Axis Labels (Frequency), derived from YAMNet's actual mel range */}
        <div className="flex flex-col justify-between pr-2 py-1 text-[9px] font-mono text-slate-500 select-none text-right w-12 sm:w-14 shrink-0">
          {[1, 0.66, 0.33, 0].map((frac) => (
            <span key={frac}>
              {formatHz(melBandToHz(frac * (spectrogram.num_mels - 1), spectrogram.num_mels))}
            </span>
          ))}
        </div>

        {/* Canvas Wrapper */}
        <div className="flex-1 relative group">
          <canvas
            ref={canvasRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoverInfo(null)}
            className="w-full h-44 rounded-lg border border-slate-800 image-pixelated cursor-crosshair"
            style={{ imageRendering: 'pixelated' }}
          />

          {/* Hover pixel tooltip overlay */}
          {hoverInfo && (
            <div className="absolute top-2 right-2 px-2.5 py-1.5 rounded-lg bg-obsidian-950/90 border border-electric-cyan/40 text-[10px] font-mono text-slate-200 shadow-xl backdrop-blur-md pointer-events-none space-x-2">
              <span className="text-electric-cyan font-bold">Freq: {hoverInfo.freqHz} Hz</span>
              <span className="text-slate-400">Mel: {hoverInfo.mel}</span>
              <span className="text-slate-400">Frame: {hoverInfo.frame}</span>
              <span className="text-white font-bold">{hoverInfo.db} dB</span>
            </div>
          )}
        </div>

      </div>

      {/* X-Axis Labels (Time) — the displayed window is 1.0 s of 16 kHz mono */}
      <div className="flex justify-between pl-12 sm:pl-14 pr-1 text-[9px] font-mono text-slate-500 select-none">
        <span>0.0s</span>
        <span>0.25s</span>
        <span>0.50s</span>
        <span>0.75s</span>
        <span>1.0s</span>
      </div>

    </div>
  );
};
