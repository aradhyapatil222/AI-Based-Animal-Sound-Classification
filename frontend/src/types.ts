export interface AnimalPrediction {
  label: string;
  emoji: string;
  probability: number;
  percentage: number;
  description: string;
  index: number;
  rank?: number;
}

export interface SignalStats {
  duration_seconds: number;
  rms_power: number;
  peak_amplitude: number;
  num_samples: number;
  sample_rate: number;
  num_windows_voted: number;
}

export interface SpectrogramData {
  matrix: number[][];
  num_mels: number;
  num_frames: number;
  min_db: number;
  max_db: number;
}

export interface ClassifyResponse {
  success: boolean;
  top_prediction: AnimalPrediction;
  top_3: AnimalPrediction[];
  predictions: AnimalPrediction[];
  signal_stats: SignalStats;
  spectrogram: SpectrogramData;
  execution_time_ms: number;
  warning?: string | null;
}

export interface SampleAudioItem {
  filename: string;
  label: string;
  emoji: string;
  description: string;
  size_bytes?: number;
}

export interface HealthStatus {
  status: string;
  model_file: string;
  classes: string[];
  input_shape: number[];
  channel_stats_available: boolean;
  global_mean: number;
  global_std: number;
}
