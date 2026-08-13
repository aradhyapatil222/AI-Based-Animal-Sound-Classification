import os
import sys
import types
import warnings
import time
import io
from typing import Optional, List

# Suppress TF logs & warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# --- Numba stub for librosa compatibility ---
def _numba_identity(func):
    return func

def _numba_jit(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1:
        return args[0]
    return _numba_identity

def _numba_guvectorize(signatures, layout, *args, **kwargs):
    layout = str(layout)
    left, right = layout.split("->")
    in_groups = [g.strip() for g in left.strip().split(")(")] if left.strip() else []
    in_groups = [g.strip("()") for g in in_groups]
    out_groups = [g.strip("()") for g in right.strip().split(")(")] if right.strip() else []
    in_dims = [d for g in in_groups for d in (g.split(",") if g else [])]

    def decorator(func):
        def wrapper(*inputs):
            outs = []
            for og in out_groups:
                dims = og.split(",") if og else []
                if not dims:
                    shape = ()
                elif all(d in in_dims for d in dims):
                    base = np.asarray(inputs[0])
                    shape = tuple(base.shape[0] for _ in dims)
                else:
                    raise NotImplementedError("numba stub guvectorize: unsupported layout " + layout)
                outs.append(np.zeros(shape, dtype=np.float32))
            result = func(*inputs, *outs)
            if result is not None:
                return result
            return outs[0] if len(outs) == 1 else tuple(outs)
        return wrapper
    return decorator

def _install_numba_stub():
    stub = types.ModuleType("numba")
    stub.__doc__ = "Pure-Python stand-in for numba"
    stub.jit = _numba_jit
    stub.njit = _numba_jit
    stub.stencil = _numba_jit
    stub.vectorize = _numba_jit
    stub.guvectorize = _numba_guvectorize
    stub.prange = range
    sys.modules["numba"] = stub

_install_numba_stub()

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import tensorflow as tf

# Constants
ANIMAL_CLASSES = ['Dog', 'Rooster', 'Pig', 'Cow', 'Frog', 'Cat', 'Hen', 'Insects', 'Sheep', 'Crow']
ANIMAL_EMOJI = {
    'Dog': '🐶', 'Rooster': '🐓', 'Pig': '🐷', 'Cow': '🐮', 'Frog': '🐸',
    'Cat': '🐱', 'Hen': '🐔', 'Insects': '🦗', 'Sheep': '🐑', 'Crow': '🐦',
}
ANIMAL_DESCRIPTIONS = {
    'Dog': 'Canine vocalization (bark, whine, howl)',
    'Rooster': 'Gallinaceous vocalization (crowing)',
    'Pig': 'Porcine vocalization (grunt, squeal)',
    'Cow': 'Bovine vocalization (moo, lowing)',
    'Frog': 'Amphibian vocalization (croak, ribbit)',
    'Cat': 'Feline vocalization (meow, purr)',
    'Hen': 'Gallinaceous female vocalization (cluck)',
    'Insects': 'Arthropod stridulation / buzzing',
    'Sheep': 'Ovine vocalization (bleat, baa)',
    'Crow': 'Corvid vocalization (caw, rattle)',
}

SR = 22050
DURATION = 3
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20

REAL_TRAINING_MEAN = -25.852587
REAL_TRAINING_STD  =  34.965031

GLOBAL_MEAN = REAL_TRAINING_MEAN
GLOBAL_STD  = REAL_TRAINING_STD
CHANNEL_MEAN = None
CHANNEL_STD = None
CHANNEL_STATS_AVAILABLE = False

if os.path.exists("feature_mean.npy") and os.path.exists("feature_std.npy"):
    try:
        GLOBAL_MEAN = float(np.load("feature_mean.npy"))
        GLOBAL_STD = float(np.load("feature_std.npy"))
    except Exception:
        pass

if os.path.exists("feature_mean_ch.npy") and os.path.exists("feature_std_ch.npy"):
    try:
        _ch_m = np.load("feature_mean_ch.npy").reshape(1, 1, 1, 2).astype(np.float32)
        _ch_s = np.load("feature_std_ch.npy").reshape(1, 1, 1, 2).astype(np.float32)
        if _ch_m.shape == (1, 1, 1, 2) and _ch_s.shape == (1, 1, 1, 2):
            CHANNEL_MEAN = _ch_m
            CHANNEL_STD = _ch_s
            CHANNEL_STATS_AVAILABLE = True
    except Exception:
        pass

# Model resolution
MODEL_PATH = None
for candidate in ["animal_sound_model_quant_final.tflite", "animal_sound_model_quant.tflite"]:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if not MODEL_PATH:
    raise RuntimeError("No TFLite model file found.")

# TFLite Interpreter Singleton
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = FastAPI(
    title="WildEcho AI API",
    description="High-performance TFLite Edge Animal Sound Classification Backend",
    version="2.0.0"
)

# CORS setup for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def standardize_features(features: np.ndarray, norm_option: str) -> np.ndarray:
    if norm_option == "Per-Channel Stats":
        if CHANNEL_STATS_AVAILABLE and CHANNEL_MEAN is not None:
            return (features - CHANNEL_MEAN) / (CHANNEL_STD + 1e-6)
        return (features - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)
    elif norm_option == "Per-Sample Stats":
        f_mean = np.mean(features, keepdims=True)
        f_std = np.std(features, keepdims=True)
        return (features - f_mean) / (f_std + 1e-6)
    else:  # Global Stats default
        return (features - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)

def _ffmpeg_available():
    import shutil
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

def _convert_bytes_with_ffmpeg(raw_bytes: bytes, sr: int):
    import shutil
    import subprocess

    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg_bin is None:
        raise FileNotFoundError("FFmpeg executable not found on PATH.")

    proc = subprocess.Popen(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-f",
            "wav",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(raw_bytes)
    if proc.returncode != 0 or not out:
        raise RuntimeError(err.decode("utf-8", "replace").strip() or "FFmpeg failed to convert audio.")
    return out

def _load_audio_bytes(raw_bytes: bytes, sr: int = SR, offset: float = 0.0, duration: Optional[float] = None):
    try:
        return librosa.load(
            io.BytesIO(raw_bytes),
            sr=sr,
            mono=True,
            offset=offset,
            duration=duration
        )
    except Exception as primary_err:
        if _ffmpeg_available():
            try:
                converted_wav = _convert_bytes_with_ffmpeg(raw_bytes, sr)
                return librosa.load(
                    io.BytesIO(converted_wav),
                    sr=sr,
                    mono=True,
                    offset=offset,
                    duration=duration
                )
            except Exception:
                pass

        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(raw_bytes))
            seg = seg.set_frame_rate(sr).set_channels(1)
            wav_io = io.BytesIO()
            seg.export(wav_io, format="wav")
            wav_io.seek(0)
            return librosa.load(wav_io, sr=sr, mono=True, offset=offset, duration=duration)
        except Exception:
            pass

        raise HTTPException(
            status_code=400,
            detail=f"Audio format not recognized ({len(raw_bytes):,} bytes). Please record via Live Mic or upload a standard WAV / MP3 file."
        ) from primary_err

def _extract_window(file_bytes: bytes, offset_sec: float, duration_sec: float):
    y, sr = _load_audio_bytes(file_bytes, sr=SR, offset=offset_sec, duration=duration_sec)
    target_length = sr * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stacked = np.vstack([mfcc, delta, delta2])
    mfcc_resized = np.pad(mfcc_stacked, ((0, 128 - mfcc_stacked.shape[0]), (0, 0)), mode="constant")

    feature_matrix = np.stack([log_mel, mfcc_resized], axis=-1)
    return np.expand_dims(feature_matrix, axis=0).astype(np.float32), log_mel

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_file": MODEL_PATH,
        "classes": ANIMAL_CLASSES,
        "input_shape": [1, 128, 130, 2],
        "channel_stats_available": CHANNEL_STATS_AVAILABLE,
        "global_mean": GLOBAL_MEAN,
        "global_std": GLOBAL_STD
    }

SAMPLE_FILES = [
    {"filename": "1-100032-A-0.wav", "label": "Dog", "emoji": "🐶", "description": "ESC-50 Dog bark clip (Clean)"},
    {"filename": "1-77241-A-3.wav", "label": "Cow", "emoji": "🐮", "description": "ESC-50 Cow lowing clip (High Confidence)"},
    {"filename": "1-15689-A-4.wav", "label": "Frog", "emoji": "🐸", "description": "ESC-50 Frog croak clip"},
    {"filename": "1-155858-D-25.wav", "label": "Sample Clip A", "emoji": "🎵", "description": "ESC-50 Audio Sample"},
    {"filename": "1-28135-B-11.wav", "label": "Sample Clip B", "emoji": "🎵", "description": "ESC-50 Audio Sample"},
    {"filename": "2-73544-A-27.wav", "label": "Sample Clip C", "emoji": "🎵", "description": "ESC-50 Audio Sample"},
]

@app.get("/api/samples")
def get_sample_list():
    existing = []
    for item in SAMPLE_FILES:
        path = os.path.join(".", item["filename"])
        if os.path.exists(path):
            existing.append({
                **item,
                "size_bytes": os.path.getsize(path),
            })
    return existing

@app.get("/api/sample/{filename}")
def stream_sample_audio(filename: str):
    path = os.path.join(".", filename)
    if not os.path.exists(path) or not filename.endswith(".wav"):
        raise HTTPException(status_code=404, detail="Sample audio file not found.")
    return FileResponse(path, media_type="audio/wav")

@app.post("/api/classify")
async def classify_audio(
    file: UploadFile = File(...),
    norm_option: str = Form("Global Stats"),
    use_voting: bool = Form(True),
    voting_step: float = Form(0.5)
):
    start_time = time.time()
    try:
        raw_bytes = await file.read()
        if not raw_bytes or len(raw_bytes) < 44:
            raise HTTPException(status_code=400, detail="Invalid audio file (file empty or corrupt header).")

        # Decode full audio to get signal stats
        y_full, sr = _load_audio_bytes(raw_bytes, sr=SR)
        dur = float(len(y_full) / sr)
        if len(y_full) == 0:
            raise HTTPException(status_code=400, detail="Could not decode audio samples.")

        rms = float(np.sqrt(np.mean(np.square(y_full.astype(np.float64)))))
        peak = float(np.max(np.abs(y_full)))

        warning_msg = None
        if dur < 0.25:
            warning_msg = f"Audio clip duration ({dur:.2f}s) is very short. Accuracy may be affected."
        elif peak < 1e-3 or rms < 1e-4:
            warning_msg = "Audio signal is silent or extremely faint. Model output may default to bias class."

        # Feature extraction & inference
        feat_t0, log_mel = _extract_window(raw_bytes, offset_sec=0.0, duration_sec=DURATION)

        if not use_voting or dur <= DURATION:
            feat_normed = standardize_features(feat_t0, norm_option)
            interpreter.set_tensor(input_details[0]["index"], feat_normed.astype(np.float32))
            interpreter.invoke()
            probs = interpreter.get_tensor(output_details[0]["index"])[0].tolist()
            num_windows = 1
        else:
            windows = []
            t = 0.0
            while t + DURATION <= max(dur, DURATION) + 1e-9:
                windows.append(t)
                t += voting_step
            if not windows:
                windows = [0.0]

            votes = []
            for t0 in windows:
                feat, _ = _extract_window(raw_bytes, offset_sec=t0, duration_sec=DURATION)
                feat_n = standardize_features(feat, norm_option)
                interpreter.set_tensor(input_details[0]["index"], feat_n.astype(np.float32))
                interpreter.invoke()
                votes.append(interpreter.get_tensor(output_details[0]["index"])[0])
            probs = np.mean(votes, axis=0).tolist()
            num_windows = len(windows)

        # Build detailed prediction list
        predictions = []
        for idx, (animal, prob) in enumerate(zip(ANIMAL_CLASSES, probs)):
            predictions.append({
                "label": animal,
                "emoji": ANIMAL_EMOJI.get(animal, "🐾"),
                "probability": float(prob),
                "percentage": round(float(prob) * 100, 2),
                "description": ANIMAL_DESCRIPTIONS.get(animal, ""),
                "index": idx
            })

        # Sort predictions by probability descending
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        for rank, item in enumerate(predictions):
            item["rank"] = rank + 1

        top_pred = predictions[0]
        top_3 = predictions[:3]

        # Downsample / normalize log_mel spectrogram matrix for compact frontend canvas visualization
        # log_mel shape is (128, 130)
        norm_spec = ((log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6) * 255.0).astype(np.uint8)
        spec_matrix = norm_spec.tolist()

        exec_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "success": True,
            "top_prediction": top_pred,
            "top_3": top_3,
            "predictions": predictions,
            "signal_stats": {
                "duration_seconds": round(dur, 2),
                "rms_power": round(rms, 6),
                "peak_amplitude": round(peak, 4),
                "num_samples": len(y_full),
                "sample_rate": sr,
                "num_windows_voted": num_windows
            },
            "spectrogram": {
                "matrix": spec_matrix,
                "num_mels": N_MELS,
                "num_frames": log_mel.shape[1],
                "min_db": float(log_mel.min()),
                "max_db": float(log_mel.max())
            },
            "execution_time_ms": exec_time_ms,
            "warning": warning_msg
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
