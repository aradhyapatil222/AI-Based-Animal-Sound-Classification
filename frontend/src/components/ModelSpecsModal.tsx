import React from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { X, Cpu, Layers, CheckCircle2, ShieldCheck } from 'lucide-react';
import type { HealthStatus } from '../types';

interface ModelSpecsModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthStatus | null;
}

export const ModelSpecsModal: React.FC<ModelSpecsModalProps> = ({ isOpen, onClose, health }) => {
  return (
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 animate-fade-in" />
        
        <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl bg-obsidian-900 border border-slate-800 rounded-2xl p-6 shadow-2xl z-50 max-h-[90vh] overflow-y-auto animate-fade-in space-y-5">
          
          {/* Header */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-electric-indigo/20 text-electric-indigo border border-electric-indigo/30">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <Dialog.Title className="text-base font-bold text-white">
                  TFLite Neural Architecture & Quantization Specifications
                </Dialog.Title>
                <Dialog.Description className="text-xs text-slate-400 font-mono mt-0.5">
                  MobileNet-inspired Edge Acoustic Convolutional Backbone
                </Dialog.Description>
              </div>
            </div>

            <Dialog.Close asChild>
              <button className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors">
                <X className="w-5 h-5" />
              </button>
            </Dialog.Close>
          </div>

          {/* Model Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded-xl bg-obsidian-950/80 border border-slate-800 space-y-1">
              <span className="text-micro-label text-slate-400">Model Artifact</span>
              <p className="text-xs font-mono font-bold text-electric-cyan truncate">
                {health?.model_file || 'animal_sound_model_quant_final.tflite'}
              </p>
            </div>

            <div className="p-3 rounded-xl bg-obsidian-950/80 border border-slate-800 space-y-1">
              <span className="text-micro-label text-slate-400">Input Tensor Shape</span>
              <p className="text-xs font-mono font-bold text-white">
                [1, 128, 130, 2]
              </p>
            </div>

            <div className="p-3 rounded-xl bg-obsidian-950/80 border border-slate-800 space-y-1">
              <span className="text-micro-label text-slate-400">Quantization</span>
              <p className="text-xs font-mono font-bold text-emerald-400">
                TFLite INT8 / FP32
              </p>
            </div>
          </div>

          {/* Neural Network Layers Overview */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-white flex items-center gap-1.5 uppercase tracking-wider">
              <Layers className="w-3.5 h-3.5 text-electric-cyan" />
              Feature Engineering & Backbone Design
            </h4>

            <div className="p-4 rounded-xl bg-obsidian-950/60 border border-slate-800 space-y-3 text-xs text-slate-300">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-electric-cyan shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-white">Dual-Channel Feature Stacking: </span>
                  <span>
                    Channel 0 computes 128-bin Log-Mel Spectrogram (<code className="font-mono text-electric-cyan text-[11px]">n_fft=2048, hop=512</code>). 
                    Channel 1 stacks 20 MFCCs + Delta + Delta-Delta vertically padded to 128 rows.
                  </span>
                </div>
              </div>

              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-electric-cyan shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-white">Convolutional Topology: </span>
                  <span>
                    2 Separable Convolutional blocks with 64 channels, Batch Normalization, ReLU activation, 
                    96 hidden dense units, and 10-class Softmax output layer.
                  </span>
                </div>
              </div>

              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-electric-cyan shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-white">Standardization Provenance: </span>
                  <span>
                    Global Mean (<code className="font-mono text-slate-200">-25.8526</code>) & Std (<code className="font-mono text-slate-200">34.9650</code>) 
                    computed over 400 full ESC-50 animal training clips and verified on held-out test splits.
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Benchmark Table */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-white flex items-center gap-1.5 uppercase tracking-wider">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              ESC-50 Held-Out Benchmark Performance
            </h4>

            <div className="rounded-xl border border-slate-800 overflow-hidden text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-obsidian-950 text-slate-400 border-b border-slate-800">
                    <th className="py-2 px-3">Metric</th>
                    <th className="py-2 px-3">Value</th>
                    <th className="py-2 px-3">Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80 text-slate-300">
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Top-1 Accuracy</td>
                    <td className="py-2 px-3 text-electric-cyan font-bold">~60 – 75%</td>
                    <td className="py-2 px-3 text-slate-400">Held-out test set ceiling</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Top-3 Inclusion</td>
                    <td className="py-2 px-3 text-emerald-400 font-bold">~88 – 94%</td>
                    <td className="py-2 px-3 text-slate-400">Target species within top 3 predictions</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Model Size</td>
                    <td className="py-2 px-3 text-slate-200">22.5 KB</td>
                    <td className="py-2 px-3 text-slate-400">Quantized TFLite binary file</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
