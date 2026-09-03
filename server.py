import os
import sys
import types
import warnings
import time
import io
import wave
from typing import Optional, List

# Suppress TF logs & warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Resolve paths relative to this file so the app works no matter what the
# current working directory is (Docker/HF Spaces start it from elsewhere).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache TF-Hub downloads somewhere writable. On Hugging Face Spaces the image
# bakes YAMNet into this directory at build time; locally it is created on
# first run. Without this, TF-Hub falls back to a temp dir and re-downloads.
os.environ.setdefault(
    'TFHUB_CACHE_DIR',
    os.path.join(os.path.expanduser("~"), '.cache', 'tfhub')
)

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

import json
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import tensorflow as tf
import tensorflow_hub as hub

# Microphone capture for /api/live-mic/classify. This is only available on the
# Raspberry Pi where sounddevice + a USB mic are present; on Hugging Face Spaces
# or other backends it is imported lazily so the rest of the server still runs.
try:
    import sounddevice as sd
    from scipy.signal import resample_poly
    _MIC_AVAILABLE = True
except Exception:  # pragma: no cover - sounddevice absent on non-Pi hosts
    sd = None
    resample_poly = None
    _MIC_AVAILABLE = False

# ============================================================
# Model / class configuration
#
# This backend now runs a two-stage pipeline instead of the old
# single spectrogram-CNN:
#   1. YAMNet (frozen, from TF-Hub)  -> converts 1s of 16kHz mono
#      audio into a 1024-dim embedding.
#   2. classifier_head1.tflite       -> small dense head trained on
#      those embeddings, outputs a softmax over ANIMAL_CLASSES.
#
# This must exactly mirror the training notebook's preprocessing
# (16kHz mono, 1s windows, mean-pooled YAMNet embeddings) or the
# model's predictions will be meaningless.
# ============================================================

LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")
CLASSIFIER_HEAD_PATH = os.path.join(BASE_DIR, "classifier_head1.tflite")

if not os.path.exists(CLASSIFIER_HEAD_PATH):
    raise RuntimeError(
        f"Could not find {CLASSIFIER_HEAD_PATH}. Place the trained classifier head "
        f"tflite file next to server.py."
    )

if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH) as f:
        _raw_label_map = json.load(f)
    # JSON keys are strings; sort numerically to get index -> label order right
    ANIMAL_CLASSES = [_raw_label_map[str(i)] for i in range(len(_raw_label_map))]
else:
    # Fallback: matches the label order produced by the training notebook's
    # `sorted(df["label"].unique())` step. Prefer shipping label_map.json instead
    # of relying on this staying in sync.
    ANIMAL_CLASSES = [
        'cat', 'chicken', 'cow', 'cricket', 'dog', 'donkey',
        'elephant', 'frog', 'lion', 'monkey', 'pig', 'sheep', 'squirrel'
    ]

ANIMAL_CLASSES = [c.capitalize() for c in ANIMAL_CLASSES]

ANIMAL_EMOJI = {
    'Cat': '🐱', 'Chicken': '🐔', 'Cow': '🐮', 'Cricket': '🦗', 'Dog': '🐶',
    'Donkey': '🐴', 'Elephant': '🐘', 'Frog': '🐸', 'Lion': '🦁',
    'Monkey': '🐒', 'Pig': '🐷', 'Sheep': '🐑', 'Squirrel': '🐿️',
    'Not_animal': '🔇',
}
ANIMAL_DESCRIPTIONS = {
    'Cat': 'Feline vocalization (meow, purr)',
    'Chicken': 'Gallinaceous vocalization (cluck, crow)',
    'Cow': 'Bovine vocalization (moo, lowing)',
    'Cricket': 'Orthopteran stridulation (chirp)',
    'Dog': 'Canine vocalization (bark, whine, howl)',
    'Donkey': 'Equine vocalization (bray)',
    'Elephant': 'Proboscidean vocalization (trumpet, rumble)',
    'Frog': 'Amphibian vocalization (croak, ribbit)',
    'Lion': 'Felid vocalization (roar, growl)',
    'Monkey': 'Primate vocalization (call, screech)',
    'Pig': 'Porcine vocalization (grunt, squeal)',
    'Sheep': 'Ovine vocalization (bleat, baa)',
    'Squirrel': 'Rodent vocalization (chatter, chirp)',
    'Not_animal': 'Non-animal background sound',
}

# Audio / windowing constants — MUST match the training notebook
SR = 16000
WINDOW_DURATION = 1.0
WINDOW_SAMPLES = int(SR * WINDOW_DURATION)
SILENCE_RMS_THRESHOLD = 0.01

# Raspberry Pi physical microphone configuration
PI_MIC_DEVICE = 1
PI_CAPTURE_SR = 44100
PI_CAPTURE_CHANNELS = 2
PI_MIC_CLIP_SECONDS = 3.0
PI_MIC_SILENCE_RMS = 0.01

# Public-deployment safety limits. The Space is a shared free CPU instance, so
# an oversized upload or a very long clip must not be able to pin it: each
# voting window is a separate YAMNet forward pass.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB
MAX_VOTING_WINDOWS = 60               # ~30 s of audio at the default 0.5 s hop

print("Loading YAMNet from TensorFlow Hub (first run downloads ~15MB, then cached)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded.")

interpreter = tf.lite.Interpreter(model_path=CLASSIFIER_HEAD_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

expected_classes = output_details[0]["shape"][-1]
if expected_classes != len(ANIMAL_CLASSES):
    raise RuntimeError(
        f"classifier_head1.tflite outputs {expected_classes} classes but "
        f"ANIMAL_CLASSES has {len(ANIMAL_CLASSES)} entries ({ANIMAL_CLASSES}). "
        f"Check label_map.json matches the model that was actually converted."
    )

app = FastAPI(
    title="WildEcho AI API",
    description="High-performance TFLite Edge Animal Sound Classification Backend (YAMNet embeddings)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", "-ar", str(sr), "-ac", "1", "-f", "wav", "pipe:1",
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(raw_bytes)
    if proc.returncode != 0 or not out:
        raise RuntimeError(err.decode("utf-8", "replace").strip() or "FFmpeg failed to convert audio.")
    return out

def _load_audio_bytes(raw_bytes: bytes, sr: int = SR, offset: float = 0.0, duration: Optional[float] = None):
    try:
        return librosa.load(io.BytesIO(raw_bytes), sr=sr, mono=True, offset=offset, duration=duration)
    except Exception as primary_err:
        if _ffmpeg_available():
            try:
                converted_wav = _convert_bytes_with_ffmpeg(raw_bytes, sr)
                return librosa.load(io.BytesIO(converted_wav), sr=sr, mono=True, offset=offset, duration=duration)
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

def _is_silent(window: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(window ** 2))) < SILENCE_RMS_THRESHOLD

def _embed_window(window: np.ndarray):
    """Run one 1s (16kHz mono) window through YAMNet and mean-pool its
    per-frame embeddings into a single 1024-dim vector, exactly like the
    training notebook's extract_embedding()."""
    scores, embeddings, spectrogram = yamnet_model(window)
    embedding = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)  # (1024,)
    log_mel = spectrogram.numpy()  # (frames, 64) YAMNet's internal log-mel, used for display only
    return embedding, log_mel

def _classify_embedding(embedding: np.ndarray):
    interpreter.set_tensor(input_details[0]["index"], embedding[np.newaxis, :])
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_file": os.path.basename(CLASSIFIER_HEAD_PATH),
        "input_shape": [1, 1024],
        "classifier_head": os.path.basename(CLASSIFIER_HEAD_PATH),
        "feature_extractor": "YAMNet (tfhub.dev/google/yamnet/1)",
        "classes": ANIMAL_CLASSES,
        "embedding_dim": 1024,
        "window_duration_s": WINDOW_DURATION,
        "sample_rate": SR,
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
        path = os.path.join(BASE_DIR, item["filename"])
        if os.path.exists(path):
            existing.append({**item, "size_bytes": os.path.getsize(path)})
    return existing

# Only filenames from SAMPLE_FILES may be streamed. Serving an arbitrary
# user-supplied path here would let a caller read files outside BASE_DIR
# (e.g. "../../etc/passwd"), so we match against a fixed allow-list.
_ALLOWED_SAMPLES = {item["filename"] for item in SAMPLE_FILES}


@app.get("/api/sample/{filename}")
def stream_sample_audio(filename: str):
    if filename not in _ALLOWED_SAMPLES:
        raise HTTPException(status_code=404, detail="Sample audio file not found.")
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample audio file not found.")
    return FileResponse(path, media_type="audio/wav")

def record_raspberry_pi_mic(
    duration: float = PI_MIC_CLIP_SECONDS,
    device: int = PI_MIC_DEVICE,
):
    """
    Record from the UM1002 connected directly to the Raspberry Pi.

    UM1002 native capture:
        44.1 kHz stereo int16

    Model input:
        16 kHz mono int16
    """

    if not _MIC_AVAILABLE:
        raise RuntimeError(
            "sounddevice is not installed on this host. "
            "This endpoint only works on the Raspberry Pi with a USB microphone "
            "(pip install sounddevice scipy)."
        )

    audio = sd.rec(
        int(duration * PI_CAPTURE_SR),
        samplerate=PI_CAPTURE_SR,
        channels=PI_CAPTURE_CHANNELS,
        dtype="int16",
        device=device,
    )

    sd.wait()

    # Stereo -> mono
    audio = audio.astype(np.float32)
    audio = np.mean(audio, axis=1)

    # Normalize
    audio /= 32768.0

    # Silence detection BEFORE resampling
    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < PI_MIC_SILENCE_RMS:
        return None, rms

    # 44.1 kHz -> 16 kHz
    audio = resample_poly(
        audio,
        SR,
        PI_CAPTURE_SR,
    )

    # Back to int16 WAV samples
    audio = np.clip(
        audio * 32768.0,
        -32768,
        32767,
    ).astype(np.int16)

    return audio, rms


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = SR):
    """Convert mono int16 samples to an in-memory WAV file."""

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())

    buffer.seek(0)

    return buffer


@app.post("/api/live-mic/classify")
async def classify_raspberry_pi_mic(
    use_voting: bool = Form(True),
    voting_step: float = Form(0.5),
):
    """
    Record directly from the microphone attached to the Raspberry Pi,
    then run the existing animal classifier.
    """

    try:
        # Recording is blocking, so run it outside the FastAPI event loop.
        samples, rms = await run_in_threadpool(
            record_raspberry_pi_mic,
            PI_MIC_CLIP_SECONDS,
            PI_MIC_DEVICE,
        )

        # No meaningful sound
        if samples is None:
            return {
                "success": True,
                "silent": True,
                "message": "No significant sound detected.",
                "rms_power": round(rms, 6),
            }

        wav_buffer = samples_to_wav_bytes(
            samples,
            SR,
        )

        # Reuse the EXACT same classifier endpoint logic.
        upload = UploadFile(
            file=wav_buffer,
            filename="raspberry_pi_live.wav",
        )

        result = await classify_audio(
            file=upload,
            use_voting=use_voting,
            voting_step=voting_step,
        )

        result["source"] = "raspberry_pi_um1002"

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Raspberry Pi microphone error: {str(e)}",
        )


@app.post("/api/classify")
async def classify_audio(
    file: UploadFile = File(...),
    use_voting: bool = Form(True),
    voting_step: float = Form(0.5),
):
    """
    use_voting: if True, slide 1s windows (hop = voting_step) across the whole
    clip and average the softmax outputs — mirrors the 50% overlap windowing
    used to build the training set. If False, only the first 1s is classified.
    """
    start_time = time.time()
    try:
        raw_bytes = await file.read()
        if not raw_bytes or len(raw_bytes) < 44:
            raise HTTPException(status_code=400, detail="Invalid audio file (file empty or corrupt header).")
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file is too large ({len(raw_bytes) / 1_048_576:.1f} MB). "
                       f"Maximum is {MAX_UPLOAD_BYTES // 1_048_576} MB — please upload a short clip (3–5 s is ideal)."
            )

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

        def _window_at(offset_sec: float) -> np.ndarray:
            seg, _ = _load_audio_bytes(raw_bytes, sr=SR, offset=offset_sec, duration=WINDOW_DURATION)
            if len(seg) < WINDOW_SAMPLES:
                seg = np.pad(seg, (0, WINDOW_SAMPLES - len(seg)))
            else:
                seg = seg[:WINDOW_SAMPLES]
            return seg.astype(np.float32)

        first_window = _window_at(0.0)
        first_embedding, display_log_mel = _embed_window(first_window)

        if not use_voting or dur <= WINDOW_DURATION:
            probs = _classify_embedding(first_embedding).tolist()
            num_windows = 1
        else:
            offsets = []
            t = 0.0
            while t + WINDOW_DURATION <= max(dur, WINDOW_DURATION) + 1e-9:
                offsets.append(t)
                t += voting_step
            if not offsets:
                offsets = [0.0]

            # Cap the work per request: sample evenly across the whole clip
            # instead of truncating, so long uploads are still classified using
            # audio from beginning to end.
            if len(offsets) > MAX_VOTING_WINDOWS:
                stride = len(offsets) / MAX_VOTING_WINDOWS
                offsets = [offsets[int(i * stride)] for i in range(MAX_VOTING_WINDOWS)]

            votes = []
            for i, t0 in enumerate(offsets):
                window = first_window if i == 0 else _window_at(t0)
                if _is_silent(window):
                    continue
                embedding, _ = _embed_window(window)
                votes.append(_classify_embedding(embedding))
            if not votes:
                votes = [_classify_embedding(first_embedding)]
            probs = np.mean(votes, axis=0).tolist()
            num_windows = len(votes)

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

        predictions.sort(key=lambda x: x["probability"], reverse=True)
        for rank, item in enumerate(predictions):
            item["rank"] = rank + 1

        top_pred = predictions[0]
        top_3 = predictions[:3]

        # Display-only spectrogram: YAMNet's own internal log-mel for the first window
        spec = display_log_mel.T  # (mel_bins, frames)
        norm_spec = ((spec - spec.min()) / (spec.max() - spec.min() + 1e-6) * 255.0).astype(np.uint8)
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
                "num_mels": spec.shape[0],
                "num_frames": spec.shape[1],
                "min_db": float(spec.min()),
                "max_db": float(spec.max())
            },
            "execution_time_ms": exec_time_ms,
            "warning": warning_msg
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# ============================================================
# Static frontend
#
# In the deployed (Docker / Hugging Face Spaces) image the compiled React app
# lives in frontend/dist and is served from this same process, so the UI and the
# API share one origin and one URL — no CORS hop, no second service.
#
# This mount is declared LAST on purpose: it is a catch-all for "/", so any
# route registered after it would be shadowed. During local development the
# directory simply does not exist and the API runs on its own while Vite serves
# the UI on port 5173.
# ============================================================

FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    print(f"Serving compiled frontend from {FRONTEND_DIST}")
else:
    print(
        "No compiled frontend found at frontend/dist — running in API-only mode.\n"
        "For local development start the Vite dev server separately "
        "(cd frontend && npm run dev)."
    )

if __name__ == "__main__":
    import uvicorn

    # Hugging Face Spaces routes external traffic to $PORT (7860 for Docker
    # spaces) and requires binding 0.0.0.0; both default sensibly for local runs.
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
