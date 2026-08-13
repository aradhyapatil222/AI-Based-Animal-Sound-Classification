import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { HeroSection } from './components/HeroSection';
import { AudioInputSection } from './components/AudioInputSection';
import { PredictionResults } from './components/PredictionResults';
import { SpectrogramViewer } from './components/SpectrogramViewer';
import { SignalTelemetryCard } from './components/SignalTelemetryCard';
import { ModelSpecsModal } from './components/ModelSpecsModal';
import type { ClassifyResponse, HealthStatus } from './types';
import { AlertCircle } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isSpecsOpen, setIsSpecsOpen] = useState(false);

  // Fetch API Health on Mount
  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (err) {
      console.warn('Backend server offline or starting up...');
    }
  };

  const handleClassifyFile = async (
    fileOrBlob: File | Blob,
    options: { normOption: string; useVoting: boolean; votingStep: number }
  ) => {
    setIsLoading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', fileOrBlob, 'audio.wav');
    formData.append('norm_option', options.normOption);
    formData.append('use_voting', String(options.useVoting));
    formData.append('voting_step', String(options.votingStep));

    try {
      const res = await fetch(`${API_BASE}/api/classify`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Audio classification failed.');
      }

      const data: ClassifyResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'An unexpected error occurred during classification.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-obsidian-900 text-slate-100 flex flex-col font-sans selection:bg-electric-indigo/30">
      
      {/* Top Glass Navigation Bar */}
      <Navbar health={health} onOpenSpecs={() => setIsSpecsOpen(true)} />

      {/* Main Workspace Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-4 space-y-6">
        
        {/* Hero Banner Section */}
        <HeroSection />

        {/* Global Error Banner if API error occurs */}
        {errorMsg && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-between gap-3 text-xs text-rose-200 animate-fade-in">
            <div className="flex items-center gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-slate-400 hover:text-white font-mono text-[11px] underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* 2-Column Split Workspace Grid (8px grid aligned) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* Left Column: Audio Input Workspace & Controls (5 Cols) */}
          <div className="lg:col-span-5 space-y-6">
            <AudioInputSection
              onClassifyFile={handleClassifyFile}
              isLoading={isLoading}
            />
          </div>

          {/* Right Column: Prediction Results & Spectrogram Intelligence (7 Cols) */}
          <div className="lg:col-span-7 space-y-6">
            <PredictionResults result={result} isLoading={isLoading} />
            
            {result && (
              <>
                <SpectrogramViewer spectrogram={result.spectrogram} />
                <SignalTelemetryCard stats={result.signal_stats} executionTimeMs={result.execution_time_ms} />
              </>
            )}
          </div>

        </div>

      </main>

      {/* Minimal Footer */}
      <footer className="w-full border-t border-slate-800/60 py-6 mt-8 bg-obsidian-950/60">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500 font-mono">
          <div>
            WildEcho AI Edge Neural Classifier • TFLite Quantized Engine
          </div>
          <div>
            10 ESC-50 Species Benchmark • 22.05 kHz Audio Feature Matrix
          </div>
        </div>
      </footer>

      {/* Model Specs & Architecture Modal */}
      <ModelSpecsModal
        isOpen={isSpecsOpen}
        onClose={() => setIsSpecsOpen(false)}
        health={health}
      />

    </div>
  );
}

export default App;
