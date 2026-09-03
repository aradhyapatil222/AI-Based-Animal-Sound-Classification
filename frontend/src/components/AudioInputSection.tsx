import React, { useState, useRef, useEffect } from 'react';
import { Upload, Mic, Play, Pause, Settings, SlidersHorizontal, Volume2, VolumeX, Sparkles, FileAudio } from 'lucide-react';

interface AudioInputSectionProps {
  onClassifyFile: (file: File | Blob, options: { useVoting: boolean; votingStep: number }) => void;
  onRaspberryPiMic: () => Promise<void>;
  isLoading: boolean;
}

export const AudioInputSection: React.FC<AudioInputSectionProps> = ({
  onClassifyFile,
  onRaspberryPiMic,
  isLoading,
}) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'mic'>('upload');
  
  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileAudioUrl, setFileAudioUrl] = useState<string | null>(null);
  const [isPlayingFile, setIsPlayingFile] = useState(false);
  const [fileDuration, setFileDuration] = useState(0);
  const [fileCurrentTime, setFileCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const fileAudioRef = useRef<HTMLAudioElement | null>(null);

  // Advanced settings state
  const [showSettings, setShowSettings] = useState(false);
  const [useVoting, setUseVoting] = useState<boolean>(true);
  const [votingStep, setVotingStep] = useState<number>(0.5);

  // Handle File selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      if (fileAudioUrl) URL.revokeObjectURL(fileAudioUrl);
      const url = URL.createObjectURL(file);
      setFileAudioUrl(url);
      setIsPlayingFile(false);
      setFileCurrentTime(0);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      if (fileAudioUrl) URL.revokeObjectURL(fileAudioUrl);
      const url = URL.createObjectURL(file);
      setFileAudioUrl(url);
      setIsPlayingFile(false);
      setFileCurrentTime(0);
    }
  };

  // Custom Audio Player controls for uploaded file
  const togglePlayFile = () => {
    if (!fileAudioRef.current) return;
    if (isPlayingFile) {
      fileAudioRef.current.pause();
      setIsPlayingFile(false);
    } else {
      fileAudioRef.current.play();
      setIsPlayingFile(true);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setFileCurrentTime(time);
    if (fileAudioRef.current) {
      fileAudioRef.current.currentTime = time;
    }
  };

  const toggleMute = () => {
    if (fileAudioRef.current) {
      fileAudioRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  useEffect(() => {
    return () => {
      if (fileAudioUrl) URL.revokeObjectURL(fileAudioUrl);
    };
  }, [fileAudioUrl]);

  const getOptions = () => ({
    useVoting,
    votingStep,
  });

  return (
    <div className="glass-card p-5 sm:p-6 shadow-2xl relative">
      
      {/* Workspace Tabs & Advanced Settings Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
        
        {/* Tab Buttons */}
        <div className="inline-flex p-1 bg-obsidian-950/80 border border-slate-800 rounded-xl">
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === 'upload'
                ? 'bg-gradient-to-r from-electric-indigo to-electric-indigo/80 text-white shadow-md'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/40'
            }`}
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload File</span>
          </button>

          <button
            onClick={() => setActiveTab('mic')}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === 'mic'
                ? 'bg-gradient-to-r from-electric-indigo to-electric-indigo/80 text-white shadow-md'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/40'
            }`}
          >
            <Mic className="w-3.5 h-3.5" />
            <span>Live Mic</span>
          </button>
        </div>

        {/* Settings Toggle Button */}
        <button
          onClick={() => setShowSettings(!showSettings)}
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-mono transition-all duration-200 ${
            showSettings
              ? 'bg-electric-cyan/15 border-electric-cyan/40 text-electric-cyan'
              : 'bg-obsidian-800/60 border-slate-700/60 text-slate-400 hover:text-slate-200 hover:border-slate-600'
          }`}
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>Inference Tuning</span>
        </button>

      </div>

      {/* Collapsible Advanced Settings Drawer */}
      {showSettings && (
        <div className="mt-4 p-4 rounded-xl bg-obsidian-950/90 border border-slate-800 animate-fade-in space-y-4">
          <div className="flex items-center justify-between gap-2 text-xs font-semibold text-slate-300">
            <span className="flex items-center gap-1.5">
              <Settings className="w-3.5 h-3.5 text-electric-cyan shrink-0" />
              Inference Parameters
            </span>
            <span className="text-[10px] font-mono text-slate-500 shrink-0 hidden sm:inline">TFLite Runtime Config</span>
          </div>

          <div className="pt-2">

            {/* Time-Slice Voting */}
            <div>
              <label className="text-micro-label block mb-1.5 text-slate-400">
                Time-Slice Sliding Window Voting
              </label>
              <div className="space-y-3">
                <label className="flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-obsidian-850/50 border border-slate-800 text-xs cursor-pointer">
                  <span className="text-slate-300 font-medium">Enable Multi-Window Voting</span>
                  <input
                    type="checkbox"
                    checked={useVoting}
                    onChange={(e) => setUseVoting(e.target.checked)}
                    className="rounded border-slate-700 bg-obsidian-900 text-electric-indigo focus:ring-electric-indigo shrink-0"
                  />
                </label>

                {useVoting && (
                  <div className="px-3 py-2 rounded-lg bg-obsidian-850/30 border border-slate-800/80 space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-slate-400">Window Stride Step</span>
                      <span className="text-electric-cyan font-bold">{votingStep}s</span>
                    </div>
                    <input
                      type="range"
                      min="0.25"
                      max="1.5"
                      step="0.25"
                      value={votingStep}
                      onChange={(e) => setVotingStep(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-electric-cyan"
                    />
                    <p className="text-[10px] text-slate-500 leading-snug pt-0.5">
                      Averages predictions over overlapping 1 s windows. Recommended for clips
                      longer than 2 s and for noisy microphone recordings.
                    </p>
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      )}

      {/* Main Interactive Tab Body */}
      <div className="mt-5">
        
        {/* TAB 1: FILE UPLOAD */}
        {activeTab === 'upload' && (
          <div className="space-y-4">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className="relative border-2 border-dashed border-slate-700/80 hover:border-electric-cyan/60 rounded-2xl p-6 sm:p-8 text-center bg-obsidian-950/40 hover:bg-obsidian-900/60 transition-all duration-200 group cursor-pointer"
            >
              <input
                type="file"
                accept="audio/*,.wav,.mp3,.ogg,.webm"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />
              <div className="flex flex-col items-center justify-center gap-3">
                <div className="w-12 h-12 rounded-2xl bg-electric-cyan/10 border border-electric-cyan/20 flex items-center justify-center group-hover:scale-110 transition-transform duration-200">
                  <FileAudio className="w-6 h-6 text-electric-cyan" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">
                    Drop your audio clip here, or <span className="text-electric-cyan underline">browse</span>
                  </p>
                  <p className="text-xs text-slate-400 mt-1 font-mono">
                    Supports .WAV, .MP3, .OGG, .WebM (3–5 seconds recommended)
                  </p>
                </div>
              </div>
            </div>

            {/* Bespoke Audio Player Card when file is uploaded */}
            {selectedFile && fileAudioUrl && (
              <div className="p-5 rounded-2xl bg-obsidian-950/90 border border-slate-800 shadow-2xl space-y-4 animate-fade-in">
                
                {/* Audio HTML element (hidden, controlled programmatically) */}
                <audio
                  ref={fileAudioRef}
                  src={fileAudioUrl}
                  className="hidden"
                  onLoadedMetadata={() => {
                    if (fileAudioRef.current) {
                      setFileDuration(fileAudioRef.current.duration);
                    }
                  }}
                  onTimeUpdate={() => {
                    if (fileAudioRef.current) {
                      setFileCurrentTime(fileAudioRef.current.currentTime);
                    }
                  }}
                  onEnded={() => setIsPlayingFile(false)}
                />

                {/* Header: File info & Listen Audio Badge */}
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-electric-indigo/20 border border-electric-indigo/40 flex items-center justify-center text-electric-indigo shrink-0">
                      <Volume2 className="w-5 h-5" />
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-white truncate max-w-[200px] sm:max-w-[260px]">
                        {selectedFile.name}
                      </h4>
                      <p className="text-[10px] font-mono text-slate-400 mt-0.5">
                        {(selectedFile.size / 1024).toFixed(1)} KB • Audio Clip Loaded
                      </p>
                    </div>
                  </div>

                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-electric-cyan/10 border border-electric-cyan/30 text-electric-cyan text-[11px] font-mono font-medium">
                    <span className="w-1.5 h-1.5 rounded-full bg-electric-cyan animate-pulse" />
                    Listen Mode Active
                  </span>
                </div>

                {/* Audio Player Controls & Seekbar */}
                <div className="flex flex-col sm:flex-row items-center gap-3 sm:gap-4 bg-obsidian-900/80 p-3 rounded-xl border border-slate-800/80">
                  
                  {/* Play / Pause Toggle Button */}
                  <button
                    onClick={togglePlayFile}
                    className="w-10 h-10 rounded-xl bg-electric-indigo hover:bg-electric-indigo/90 text-white flex items-center justify-center shadow-lg shadow-electric-indigo/30 transition-transform active:scale-95 shrink-0"
                    title={isPlayingFile ? 'Pause Audio' : 'Play Audio Sound'}
                  >
                    {isPlayingFile ? (
                      <Pause className="w-5 h-5 fill-white" />
                    ) : (
                      <Play className="w-5 h-5 fill-white ml-0.5" />
                    )}
                  </button>

                  {/* Time Counters & Seekbar */}
                  <div className="flex-1 w-full space-y-1">
                    <div className="flex justify-between text-[11px] font-mono text-slate-400">
                      <span>{formatTime(fileCurrentTime)}</span>
                      <span>{formatTime(fileDuration)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max={fileDuration || 100}
                      step="0.05"
                      value={fileCurrentTime}
                      onChange={handleSeek}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-electric-indigo"
                    />
                  </div>

                  {/* Volume Mute Toggle */}
                  <button
                    onClick={toggleMute}
                    className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors shrink-0"
                    title={isMuted ? 'Unmute' : 'Mute'}
                  >
                    {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4 text-slate-300" />}
                  </button>
                </div>

                {/* Primary Action Button */}
                <div className="flex items-center justify-end pt-1">
                  <button
                    onClick={() => onClassifyFile(selectedFile, getOptions())}
                    disabled={isLoading}
                    className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-gradient-to-r from-electric-indigo via-electric-indigo to-electric-cyan hover:from-electric-indigo/90 hover:to-electric-cyan/90 text-white text-xs font-bold shadow-lg shadow-electric-indigo/25 transition-all duration-200 disabled:opacity-50 active:scale-95 flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <Sparkles className="w-4 h-4 text-electric-cyan" />
                    )}
                    <span>Classify Audio Sound</span>
                  </button>
                </div>

              </div>
            )}
          </div>
        )}

        {/* TAB 2: LIVE MICROPHONE */}
        {activeTab === 'mic' && (
          <div className="space-y-4">
            <div className="p-6 rounded-2xl bg-obsidian-950/60 border border-slate-800 flex flex-col items-center justify-center gap-4 text-center">
              
              {/* Raspberry Pi Info Card */}
              <div className="w-full p-4 rounded-xl bg-obsidian-900 border border-slate-800 flex flex-col items-center justify-center gap-2">
                <div className="w-12 h-12 rounded-2xl bg-rose-600/15 border border-rose-600/30 flex items-center justify-center">
                  <Mic className="w-6 h-6 text-rose-400" />
                </div>
                <p className="text-xs font-semibold text-white">Raspberry Pi UM1002</p>
                <p className="text-[10px] font-mono text-slate-400">
                  3s clip • 44.1 kHz stereo → 16 kHz mono • silence-gated
                </p>
              </div>

              {/* Primary Action Button */}
              <button
                onClick={onRaspberryPiMic}
                disabled={isLoading}
                className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold shadow-lg shadow-rose-600/20 transition-all duration-200 active:scale-95 disabled:opacity-50"
              >
                <Mic className="w-4 h-4" />
                <span>
                  {isLoading
                    ? "Listening on Raspberry Pi..."
                    : "Detect Animal"}
                </span>
              </button>

            </div>
          </div>
        )}

      </div>

    </div>
  );
};
