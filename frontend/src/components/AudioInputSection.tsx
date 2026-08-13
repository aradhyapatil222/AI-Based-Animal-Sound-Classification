import React, { useState, useRef, useEffect } from 'react';
import { Upload, Mic, Play, Pause, Square, Settings, SlidersHorizontal, Volume2, VolumeX, Sparkles, FileAudio } from 'lucide-react';

interface AudioInputSectionProps {
  onClassifyFile: (file: File | Blob, options: { normOption: string; useVoting: boolean; votingStep: number }) => void;
  isLoading: boolean;
}

// 16-bit PCM WAV Encoder in Pure JavaScript
function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  /* RIFF identifier */
  writeString(0, 'RIFF');
  /* RIFF chunk length */
  view.setUint32(4, 36 + samples.length * 2, true);
  /* RIFF type */
  writeString(8, 'WAVE');
  /* format chunk identifier */
  writeString(12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw PCM 1) */
  view.setUint16(20, 1, true);
  /* channel count (mono 1) */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sampleRate * 2) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (2) */
  view.setUint16(32, 2, true);
  /* bits per sample (16) */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(36, 'data');
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  /* float32 to int16 PCM */
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

export const AudioInputSection: React.FC<AudioInputSectionProps> = ({
  onClassifyFile,
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

  // Live microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  
  // Audio PCM & Canvas visualizer refs
  const audioChunksRef = useRef<Blob[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  // Advanced settings state
  const [showSettings, setShowSettings] = useState(false);
  const [normOption, setNormOption] = useState<string>('Global Stats');
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

  // Pure PCM WAV Microphone Recording via Native MediaRecorder + decodeAudioData
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;
      audioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      audioContextRef.current = audioCtx;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 128;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);

      drawWaveform();
    } catch (err) {
      alert('Microphone access was denied or not supported by browser.');
    }
  };

  const stopRecording = () => {
    if (!isRecording || !mediaRecorderRef.current) return;
    setIsRecording(false);

    if (timerRef.current) clearInterval(timerRef.current);
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

    mediaRecorderRef.current.onstop = async () => {
      try {
        const rawBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorderRef.current?.mimeType || 'audio/webm',
        });

        const arrayBuffer = await rawBlob.arrayBuffer();
        const tempCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const decodedBuffer = await tempCtx.decodeAudioData(arrayBuffer);
        const pcmData = decodedBuffer.getChannelData(0);
        const sampleRate = decodedBuffer.sampleRate;
        tempCtx.close();

        // Gain normalization: scale quiet microphone input to 85% peak
        let peak = 0;
        for (let i = 0; i < pcmData.length; i++) {
          const abs = Math.abs(pcmData[i]);
          if (abs > peak) peak = abs;
        }

        if (peak > 0.0001 && peak < 0.8) {
          const scale = 0.85 / peak;
          for (let i = 0; i < pcmData.length; i++) {
            pcmData[i] = pcmData[i] * scale;
          }
        }

        // Encode pristine 16-bit RIFF PCM WAV Blob
        const wavBlob = encodeWAV(pcmData, sampleRate);
        setRecordedBlob(wavBlob);

        if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
        setRecordedAudioUrl(URL.createObjectURL(wavBlob));
      } catch (err) {
        console.error('Failed to decode mic recording:', err);
        const fallbackBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setRecordedBlob(fallbackBlob);
        if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
        setRecordedAudioUrl(URL.createObjectURL(fallbackBlob));
      } finally {
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
        if (audioContextRef.current) {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }
      }
    };

    mediaRecorderRef.current.stop();
  };

  const drawWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      if (!isRecording && !analyserRef.current) return;
      animationFrameRef.current = requestAnimationFrame(render);
      analyserRef.current?.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const barWidth = (canvas.width / bufferLength) * 1.8;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
        gradient.addColorStop(0, '#06b6d4');
        gradient.addColorStop(1, '#6366f1');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 3;
      }
    };

    render();
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (fileAudioUrl) URL.revokeObjectURL(fileAudioUrl);
      if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
    };
  }, []);

  const getOptions = () => ({
    normOption,
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
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <span className="flex items-center gap-1.5">
              <Settings className="w-3.5 h-3.5 text-electric-cyan" />
              Inference & Standardization Parameters
            </span>
            <span className="text-[10px] font-mono text-slate-500">TFLite Runtime Config</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            
            {/* Normalization Strategy */}
            <div>
              <label className="text-micro-label block mb-1.5 text-slate-400">
                Standardization Method
              </label>
              <div className="space-y-1.5">
                {[
                  { id: 'Global Stats', label: 'Global Stats (400-clip ESC-50 verified)' },
                  { id: 'Per-Channel Stats', label: 'Per-Channel Stats (LogMel/MFCC split)' },
                  { id: 'Per-Sample Stats', label: 'Per-Sample Stats (Clip mean/std)' },
                ].map((opt) => (
                  <label
                    key={opt.id}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-xs cursor-pointer transition-all ${
                      normOption === opt.id
                        ? 'bg-electric-indigo/15 border-electric-indigo/50 text-white'
                        : 'bg-obsidian-850/50 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <input
                      type="radio"
                      name="norm_option"
                      value={opt.id}
                      checked={normOption === opt.id}
                      onChange={() => setNormOption(opt.id)}
                      className="hidden"
                    />
                    <span className={`w-3.5 h-3.5 rounded-full border flex items-center justify-center ${normOption === opt.id ? 'border-electric-indigo bg-electric-indigo' : 'border-slate-600'}`}>
                      {normOption === opt.id && <span className="w-1.5 h-1.5 bg-white rounded-full" />}
                    </span>
                    <span>{opt.label}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Time-Slice Voting */}
            <div>
              <label className="text-micro-label block mb-1.5 text-slate-400">
                Time-Slice Sliding Window Voting
              </label>
              <div className="space-y-3">
                <label className="flex items-center justify-between px-3 py-2 rounded-lg bg-obsidian-850/50 border border-slate-800 text-xs cursor-pointer">
                  <span className="text-slate-300 font-medium">Enable Multi-Window Voting</span>
                  <input
                    type="checkbox"
                    checked={useVoting}
                    onChange={(e) => setUseVoting(e.target.checked)}
                    className="rounded border-slate-700 bg-obsidian-900 text-electric-indigo focus:ring-electric-indigo"
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
              
              {/* Live Canvas Visualizer */}
              <div className="w-full h-24 bg-obsidian-900 rounded-xl border border-slate-800/80 overflow-hidden relative flex items-center justify-center">
                <canvas ref={canvasRef} width={500} height={96} className="w-full h-full" />
                {!isRecording && !recordedAudioUrl && (
                  <p className="absolute text-xs text-slate-500 font-mono">
                    Click "Start Recording" to capture live microphone audio
                  </p>
                )}
              </div>

              {/* Controls */}
              <div className="flex items-center gap-4">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold shadow-lg shadow-rose-600/20 transition-all duration-200 active:scale-95"
                  >
                    <Mic className="w-4 h-4" />
                    <span>Start Recording</span>
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold shadow-lg shadow-amber-600/20 transition-all duration-200 active:scale-95 animate-pulse"
                  >
                    <Square className="w-4 h-4 fill-white" />
                    <span>Stop Recording ({recordingTime}s)</span>
                  </button>
                )}
              </div>

              {recordedAudioUrl && (
                <div className="w-full p-4 rounded-xl bg-obsidian-900 border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4 mt-2">
                  <div className="flex items-center gap-2 text-xs font-mono text-slate-300">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
                    <span>Recorded Audio Sound Ready (.WAV)</span>
                  </div>

                  <audio controls src={recordedAudioUrl} className="h-8 max-w-[240px]" />

                  <button
                    onClick={() => recordedBlob && onClassifyFile(recordedBlob, getOptions())}
                    disabled={isLoading || !recordedBlob}
                    className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-gradient-to-r from-electric-indigo to-electric-cyan hover:from-electric-indigo/90 hover:to-electric-cyan/90 text-white text-xs font-bold shadow-lg shadow-electric-indigo/20 transition-all duration-200 disabled:opacity-50 active:scale-95 flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <Play className="w-4 h-4 fill-white" />
                    )}
                    <span>Classify Live Recording</span>
                  </button>
                </div>
              )}

            </div>
          </div>
        )}

      </div>

    </div>
  );
};
