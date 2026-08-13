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
      // Rough mel to hz approximation
      const freqHz = Math.round(700 * (Math.pow(10, (mel / 128) * (2595 / 700)) - 1));

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

  if (!spectrogram) {
    return null;
  }

  return (
    <div className="glass-card p-5 space-y-4">
      
      {/* Header & Colormap selector */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider">
            <Activity className="w-4 h-4 text-electric-cyan" />
            Channel 0: Log-Mel Spectrogram Visualization
          </h4>
          <p className="text-[10px] text-slate-400 font-mono mt-0.5">
            128 Mel Bins × 130 Time Frames (n_fft=2048, hop=512, sr=22050Hz)
          </p>
        </div>

        {/* Colormap Buttons */}
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-obsidian-950/80 border border-slate-800">
          {[
            { id: 'cyan', label: 'Electric Obsidian' },
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
        
        {/* Y-Axis Labels (Frequency) */}
        <div className="flex flex-col justify-between pr-2 py-1 text-[9px] font-mono text-slate-500 select-none text-right w-14">
          <span>11.0 kHz</span>
          <span>5.5 kHz</span>
          <span>2.7 kHz</span>
          <span>0 Hz</span>
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

      {/* X-Axis Labels (Time) */}
      <div className="flex justify-between pl-16 pr-1 text-[9px] font-mono text-slate-500 select-none">
        <span>0.0s</span>
        <span>0.75s</span>
        <span>1.50s</span>
        <span>2.25s</span>
        <span>3.00s</span>
      </div>

    </div>
  );
};
