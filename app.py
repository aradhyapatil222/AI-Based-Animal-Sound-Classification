import os
import sys
import types
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


# --- numba stub ----------------------------------------------------------
# Windows' Application Control policy intermittently blocks numba's compiled
# `_typeconv` DLL, which makes `import numba` (and therefore librosa) fail with
# a cryptic "DLL load failed" error. librosa only uses numba as a JIT
# accelerator for a few utility functions; running those as plain Python gives
# identical results (just slightly slower). Installing this stub BEFORE librosa
# imports makes the app deterministic on machines where numba is blocked.
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
                    raise NotImplementedError(
                        "numba stub guvectorize: unsupported layout " + layout
                    )
                outs.append(np.zeros(shape, dtype=np.float32))
            result = func(*inputs, *outs)
            if result is not None:
                return result
            return outs[0] if len(outs) == 1 else tuple(outs)
        return wrapper
    return decorator


def _install_numba_stub():
    stub = types.ModuleType("numba")
    stub.__doc__ = "Pure-Python stand-in for the numba JIT package (see app.py)."
    stub.jit = _numba_jit
    stub.njit = _numba_jit
    stub.stencil = _numba_jit
    stub.vectorize = _numba_jit
    stub.guvectorize = _numba_guvectorize
    stub.prange = range
    sys.modules["numba"] = stub


_install_numba_stub()

import streamlit as st
import numpy as np

LIBROSA_IMPORT_ERROR = None
try:
    import librosa
except Exception as err:
    librosa = None
    LIBROSA_IMPORT_ERROR = err

class LibrosaImportError(RuntimeError):
    pass


def _ensure_librosa():
    if LIBROSA_IMPORT_ERROR is None:
        return librosa
    msg = str(LIBROSA_IMPORT_ERROR)
    if "Numba needs NumPy 2.4 or less" in msg:
        raise LibrosaImportError(
            "Librosa failed to import because NumPy is too new for the installed librosa/numba stack. "
            "Install `numpy==2.4.6` in the app environment and restart Streamlit."
        ) from LIBROSA_IMPORT_ERROR
    raise LibrosaImportError(
        "Librosa failed to import. Please install compatible versions of librosa, numba, and numpy."
    ) from LIBROSA_IMPORT_ERROR


def _render_librosa_import_error(exc):
    return (
        "⚠️ Librosa import failed. Audio decoding and feature extraction require a compatible "
        "librosa install in this environment.\n\n"
        "Detected error: `" + str(exc) + "`\n\n"
        "Fix by running:\n"
        "`pip install numpy==2.4.6 numba==0.66.0 librosa==0.11.0`\n"
        "Then restart the Streamlit app."
    )

# --- Page Configuration ---
st.set_page_config(
    page_title="WildEcho | Animal Sound Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Global theme / styles ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    :root {
        --leaf: #2ecc71;
        --leaf-deep: #1f9d58;
        --mint: #a8e6cf;
        --gold: #f5b942;
    }

    .stApp {
        background: linear-gradient(160deg, #071410 0%, #0d2f24 45%, #11412b 72%, #071410 100%);
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', system-ui, -apple-system, sans-serif;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, rgba(46,204,113,.16), rgba(245,185,66,.10));
        border: 1px solid rgba(46,204,113,.28);
        border-radius: 24px;
        padding: 2.1rem 2.4rem 1.7rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 14px 44px rgba(0,0,0,.38);
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -.02em;
        margin: 0 0 .35rem 0;
        color: #ffffff;
        line-height: 1.1;
    }
    .hero .accent { color: var(--leaf); }
    .hero p {
        font-size: 1.05rem;
        color: #cfe8dd;
        margin: 0 0 1.1rem 0;
        font-weight: 300;
    }
    .hero .model-badge {
        display: inline-block;
        background: rgba(0,0,0,.25);
        border: 1px solid rgba(255,255,255,.12);
        color: var(--mint);
        font-size: .78rem;
        font-weight: 600;
        padding: .25rem .75rem;
        border-radius: 999px;
        margin-top: .2rem;
        letter-spacing: .04em;
    }

    /* ---------- Animal chips ---------- */
    .chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .3rem; }
    .chip {
        background: rgba(46,204,113,.12);
        border: 1px solid rgba(46,204,113,.35);
        color: #d7f5e6;
        font-size: .8rem;
        font-weight: 600;
        padding: .3rem .85rem;
        border-radius: 999px;
        letter-spacing: .02em;
        transition: transform .15s ease, background .15s ease;
    }
    .chip:hover { transform: translateY(-2px); background: rgba(46,204,113,.24); }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: .55rem;
        background: rgba(255,255,255,.05);
        padding: .4rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,.08);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #bfe6d4;
        font-weight: 600;
        padding: .5rem 1.3rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--leaf), var(--leaf-deep)) !important;
        color: #04120c !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(46,204,113,.55);
        background: linear-gradient(135deg, #243c31, #182a22);
        color: #eaf6f0;
        padding: .55rem 1.4rem;
        transition: all .2s ease;
        box-shadow: 0 4px 14px rgba(0,0,0,.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(46,204,113,.3);
        background: linear-gradient(135deg, #2d4a3c, #1e352a);
        color: #ffffff;
        border-color: rgba(46,204,113,.8);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,.08);
        color: #eaf6f0;
        border: 1px solid rgba(255,255,255,.16);
        box-shadow: none;
    }
    .stButton > button[kind="secondary"]:hover { background: rgba(255,255,255,.15); color: #fff; }

    /* ---------- Inputs ---------- */
    .stFileUploader, .stAudioInput { border-radius: 16px; }
    .stFileUploader > div:first-child, .stAudioInput > div:first-child {
        background: #16271f !important;
        border: 1px solid rgba(46,204,113,.6) !important;
        border-radius: 16px !important;
    }
    .stFileUploader > div:first-child:hover, .stAudioInput > div:first-child:hover {
        border-color: rgba(46,204,113,.95) !important;
        background: #1a2d24 !important;
    }

    /* ---------- Result cards ---------- */
    .result-card {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 30px rgba(0,0,0,.3);
        height: 100%;
    }
    .result-card .label {
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .16em;
        color: var(--leaf);
        text-transform: uppercase;
        margin-bottom: .4rem;
    }
    .result-card .big {
        font-size: 2.3rem;
        font-weight: 800;
        color: #fff;
        line-height: 1.15;
        margin-bottom: .35rem;
    }
    .result-card .sub { color: #cfe8dd; font-size: .95rem; line-height: 1.6; }

    /* ---------- Rank rows (top-3) ---------- */
    .rank-row { display: flex; align-items: center; gap: .7rem; margin: .55rem 0; }
    .rank { font-weight: 800; color: var(--gold); font-size: .9rem; width: 1.1rem; }
    .rank-name { font-weight: 600; color: #eaf6f0; width: 7.2rem; }
    .rank-bar { flex: 1; height: 9px; border-radius: 999px; background: rgba(255,255,255,.08); overflow: hidden; }
    .rank-bar > span { display: block; height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, var(--leaf), var(--gold)); }
    .rank-pct { font-weight: 700; color: #fff; width: 3.4rem; text-align: right; font-size: .9rem; }

    /* ---------- Expander ---------- */
    .streamlit-expanderHeader { font-weight: 600; color: #cfe8dd; }
    .streamlit-expander {
        background: rgba(255,255,255,.035);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,.08);
    }

    /* ---------- Alerts ---------- */
    .stAlert { border-radius: 14px; }

    /* ---------- Misc ---------- */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #fff;
        margin: 1.2rem 0 .6rem;
    }
    footer, #MainMenu { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }

    /* ---------- Global text colors (dark theme) ---------- */
    .stApp { color: #eaf6f0; }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] td,
    [data-testid="stMarkdownContainer"] strong {
        color: #eaf6f0;
    }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #ffffff;
    }
    [data-testid="stMarkdownContainer"] a { color: #7ee2a8; }
    [data-testid="stMarkdownContainer"] code {
        color: #d7f5e6;
        background: rgba(46,204,113,.12);
        border-radius: 6px;
    }
    .stSubheader { color: #ffffff !important; }
    .stCaption, .stCaption p { color: #9fb9ad; }
    [data-baseweb="radio"] label,
    [data-baseweb="radio"] div,
    [data-baseweb="checkbox"] label,
    [data-baseweb="checkbox"] span,
    [data-baseweb="slider"] label {
        color: #eaf6f0;
    }
    .stSlider [data-testid="stMarkdownContainer"] { color: #eaf6f0; }
    .streamlit-expanderHeader p { color: #cfe8dd; }
    [data-testid="stFileUploader"] * { color: #eaf6f0; }
    [data-testid="stFileUploader"] small { color: #9fb9ad; }
    [data-testid="stAudioInput"] * { color: #eaf6f0; }
    hr { border-color: rgba(255,255,255,.12); }
</style>
""", unsafe_allow_html=True)

# --- Hero header ---
st.markdown("""
<div class="hero">
  <h1>🐾 WildEcho <span class="accent">AI</span></h1>
  <p>Real-time <b>animal voice detection &amp; classification</b> running on a quantized
     <b>Edge TFLite</b> model. Upload an audio clip or record live from your microphone —
     get an instant prediction.</p>
  <div class="chips">
    <span class="chip">🐶 Dog</span><span class="chip">🐓 Rooster</span><span class="chip">🐷 Pig</span>
    <span class="chip">🐮 Cow</span><span class="chip">🐸 Frog</span><span class="chip">🐱 Cat</span>
    <span class="chip">🐔 Hen</span><span class="chip">🦗 Insects</span><span class="chip">🐑 Sheep</span>
    <span class="chip">🐦 Crow</span>
  </div>
</div>
""", unsafe_allow_html=True)

# --- Constants & Classes ---
ANIMAL_CLASSES = ['Dog', 'Rooster', 'Pig', 'Cow', 'Frog', 'Cat', 'Hen', 'Insects', 'Sheep', 'Crow']
ANIMAL_EMOJI = {
    'Dog': '🐶', 'Rooster': '🐓', 'Pig': '🐷', 'Cow': '🐮', 'Frog': '🐸',
    'Cat': '🐱', 'Hen': '🐔', 'Insects': '🦗', 'Sheep': '🐑', 'Crow': '🐦',
}
SR = 22050
DURATION = 3
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20

# Defaults used when the (hidden-by-default) advanced settings are untouched.
NORM_DEFAULT = "Global Stats (400-clip training, VERIFIED default)"

# --- Model Selection: prefer FINAL model which passes validation ---
MODEL_CANDIDATES = [
    "animal_sound_model_quant_final.tflite",
    "animal_sound_model_quant.tflite",
]
MODEL_PATH = None
MODEL_NAME = None
for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        MODEL_NAME = candidate.replace(".tflite", "")
        break

if MODEL_PATH is None:
    st.error(f"❌ No TFLite model file found. Looked for: {MODEL_CANDIDATES}")
    st.stop()

# --- Real training-set statistics (computed over 400 ESC-50 animal clips) ---
REAL_TRAINING_MEAN = -25.852587
REAL_TRAINING_STD  =  34.965031

# --- Load standardization stats (GLOBAL scalar + CHANNEL-WISE variants) ---
GLOBAL_MEAN = None
GLOBAL_STD = None
CHANNEL_MEAN = None
CHANNEL_STD = None
CHANNEL_STATS_AVAILABLE = False
STATS_FILE_SOURCE = None

if os.path.exists("feature_mean.npy") and os.path.exists("feature_std.npy"):
    try:
        GLOBAL_MEAN = float(np.load("feature_mean.npy"))
        GLOBAL_STD = float(np.load("feature_std.npy"))
        STATS_FILE_SOURCE = "feature_mean.npy / feature_std.npy (400-clip verified)"
    except Exception:
        GLOBAL_MEAN = None
        GLOBAL_STD = None

if os.path.exists("feature_mean_ch.npy") and os.path.exists("feature_std_ch.npy"):
    try:
        _ch_m = np.load("feature_mean_ch.npy").reshape(1, 1, 1, 2).astype(np.float32)
        _ch_s = np.load("feature_std_ch.npy").reshape(1, 1, 1, 2).astype(np.float32)
        if _ch_m.shape == (1, 1, 1, 2) and _ch_s.shape == (1, 1, 1, 2):
            CHANNEL_MEAN = _ch_m
            CHANNEL_STD = _ch_s
            CHANNEL_STATS_AVAILABLE = True
    except Exception:
        CHANNEL_MEAN = None
        CHANNEL_STD = None
        CHANNEL_STATS_AVAILABLE = False

if GLOBAL_MEAN is None:
    GLOBAL_MEAN = REAL_TRAINING_MEAN
    GLOBAL_STD  = REAL_TRAINING_STD
    STATS_FILE_SOURCE = (
        "CODE FALLBACK — real training global mean/std: "
        "mean={:.4f}, std={:.4f}"
    ).format(REAL_TRAINING_MEAN, REAL_TRAINING_STD)

# --- Standardization dispatcher (reads current widget value from session state) ---
def standardize_features(features):
    norm = st.session_state.get("norm_option", NORM_DEFAULT)
    if norm.startswith("Global Stats"):
        return (features - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)
    elif norm.startswith("Per-Channel Stats"):
        if not CHANNEL_STATS_AVAILABLE:
            st.error("Per-Channel stats unavailable on this machine; falling back to Global.")
            return (features - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)
        return (features - CHANNEL_MEAN) / (CHANNEL_STD + 1e-6)
    else:  # Per-Sample Standardization (NOT recommended)
        f_mean = np.mean(features, keepdims=True)
        f_std = np.std(features, keepdims=True)
        return (features - f_mean) / (f_std + 1e-6)

# --- Audio-format detection from magic bytes ---
def detect_audio_mime(raw_bytes):
    """Return (mime, label) inferred from the first 16 bytes of the file, or fallback ('application/octet-stream','unknown')."""
    if raw_bytes is None or len(raw_bytes) < 4:
        return "application/octet-stream", "unknown (<4B header)"
    head = bytes(raw_bytes[:16])
    if head.startswith(b"RIFF") and len(head) >= 12 and head[8:12] == b"WAVE":
        return "audio/wav", "WAV (RIFF/WAVE)"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio/webm", "WebM/Matroska (EBML)"
    if head.startswith(b"fLaC") or head.startswith(b"OggS"):
        return "audio/ogg", "OGG/FLAC"
    if head.startswith(b"ID3") or (len(head) >= 2 and (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)):
        return "audio/mpeg", "MP3"
    if head.startswith(b"\x00\x00\x00") and len(head) >= 12 and head[4:8] in (b"ftyp", b"moov", b"wide", b"mdat"):
        return "audio/mp4", "MP4/M4A"
    ext_guess = "unknown (first4B=" + " ".join(f"{b:02X}" for b in head[:4]) + ")"
    return "application/octet-stream", ext_guess

# --- Audio decoding helpers --------------------------------------------------

def _ffmpeg_available():
    import shutil
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def _convert_bytes_with_ffmpeg(raw_bytes, sr):
    import io
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


def _load_audio_bytes(raw_bytes, sr=SR, offset=None, duration=None):
    import io
    import shutil

    librosa = _ensure_librosa()

    try:
        return librosa.load(
            io.BytesIO(raw_bytes),
            sr=sr,
            mono=True,
            offset=offset if offset is not None else 0.0,
            duration=duration,
        )
    except Exception:
        if _ffmpeg_available():
            converted = _convert_bytes_with_ffmpeg(raw_bytes, sr)
            return librosa.load(
                io.BytesIO(converted),
                sr=sr,
                mono=True,
                offset=offset if offset is not None else 0.0,
                duration=duration,
            )
        raise


# --- Signal sanity check ---
def audio_signal_stats(raw_bytes):
    """Try to load audio via librosa and return (duration_s, rms, peak_abs, n_samples).
    Returns (None, None, None, None) if loading fails entirely."""
    try:
        y, sr = _load_audio_bytes(raw_bytes, sr=SR)
        if len(y) == 0:
            return 0.0, 0.0, 0.0, 0
        rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))
        peak = float(np.max(np.abs(y)))
        dur = float(len(y) / sr)
        return dur, rms, peak, int(len(y))
    except LibrosaImportError:
        raise
    except Exception:
        return None, None, None, None

# --- Cached TFLite Interpreter ---
@st.cache_resource
def load_tflite_model():
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# --- Feature extraction for a single (offset, duration) segment ---
def _extract_window(file_bytes, offset_sec, duration_sec):
    librosa = _ensure_librosa()
    y, sr = _load_audio_bytes(file_bytes, sr=SR, offset=offset_sec, duration=duration_sec)

    target_length = sr * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stacked = np.vstack([mfcc, delta, delta2])

    mfcc_resized = np.pad(
        mfcc_stacked, ((0, 128 - mfcc_stacked.shape[0]), (0, 0)), mode="constant"
    )

    feature_matrix = np.stack([log_mel, mfcc_resized], axis=-1)
    return np.expand_dims(feature_matrix, axis=0).astype(np.float32)

# --- Legacy single-window extractor (still used for spectrogram display) ---
def extract_dual_channel_features(file_bytes):
    return _extract_window(file_bytes, offset_sec=0.0, duration_sec=DURATION)

# --- Full classification: features (optionally multi-window voting) -> probabilities ---
def run_inference(file_bytes):
    import io

    librosa = _ensure_librosa()
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    use_voting = st.session_state.get("use_voting", True)
    voting_step = st.session_state.get("voting_step", 0.5)

    # Always compute the t=0 single window for spectrogram display
    feat_t0 = extract_dual_channel_features(file_bytes)
    feat_t0_normed = standardize_features(feat_t0)

    if not use_voting:
        interpreter.set_tensor(input_details[0]["index"], feat_t0_normed)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]["index"])[0]
        return probs, feat_t0_normed, 1

    # Voting: find actual audio duration, slide DURATION-s windows
    y_full, _ = _load_audio_bytes(file_bytes, sr=SR)
    total_dur = len(y_full) / SR

    windows = []
    t = 0.0
    while t + DURATION <= max(total_dur, DURATION) + 1e-9:
        windows.append(t)
        t += voting_step
    if not windows:
        windows = [0.0]

    votes = []
    for t0 in windows:
        feat = _extract_window(file_bytes, offset_sec=t0, duration_sec=DURATION)
        feat_n = standardize_features(feat).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], feat_n)
        interpreter.invoke()
        votes.append(interpreter.get_tensor(output_details[0]["index"])[0])

    avg_probs = np.mean(votes, axis=0)
    return avg_probs, feat_t0_normed, len(windows)

# --- Classification pipeline (shared by upload + record paths) ---
def classify_audio_bytes(file_bytes):
    import plotly.express as px
    import io

    mime, mime_label = detect_audio_mime(file_bytes)

    # 1) Try to decode the whole thing and collect stats first (cheap, catches 99%)
    try:
        dur, rms, peak, n_samples = audio_signal_stats(file_bytes)
    except LibrosaImportError as exc:
        st.error(_render_librosa_import_error(exc))
        return

    if dur is None:
        st.error(
            f"**Failed to decode audio.** librosa could not load the {len(file_bytes):,} byte buffer.\n\n"
            f"Detected container from magic bytes: **`{mime_label}`** — MIME guessed `{mime}`.\n\n"
            "**Most likely cause (if using Mic tab):** your browser actually recorded **WebM/Opus** "
            "despite `format=\"wav\"` — MediaRecorder API support for WAV is spotty. "
            "To decode WebM you need **FFmpeg** on PATH.\n\n"
            "**Fixes:**\n"
            "1. Install FFmpeg on Windows:  `winget install --id=Gyan.FFmpeg -e`  then **close & reopen the terminal** before restarting the app.\n"
            "2. Use **Chrome or Edge** rather than Firefox; they often give true WAV.\n"
            "3. Until then, use the **Upload** tab with a WAV recorded in Audacity / Voice Recorder / Phone."
        )
        return

    # 2) Silent / nearly-silent input → the model will output its bias class (often Crow)
    NEAR_SILENCE_RMS = 1e-4
    VERY_SHORT_S = 0.25
    if dur < VERY_SHORT_S:
        st.warning(
            f"⚠️ Audio loaded but it is only **{dur:.2f} s** — too short for classification "
            f"(need at least ~{VERY_SHORT_S:.1f} s of sound). The model will guess its bias class "
            f"(often Crow), not what you recorded. **Re-record with at least 1–2 s of actual sound.**"
        )
        return
    if peak < 1e-3 or rms < NEAR_SILENCE_RMS:
        st.warning(
            "⚠️ **The decoded audio is essentially silent.**  \n"
            f"Signal stats: length={dur:.2f}s, peak={peak:.2e}, RMS={rms:.2e}  \n"
            "This almost always produces a fake **Crow** prediction (the network's highest-bias class), "
            "NOT a real classification.  \n\n"
            "**Likely causes & fixes:**  \n"
            "- Wrong mic selected in the browser (e.g. a disabled/muted device). → Click the 🎤 icon in the "
            "  address bar and pick a different input / grant mic permission again.  \n"
            "- Mic hardware gain at zero or physical mic mute switch on.  \n"
            f"- Wrong container format: we detected `{mime_label}`. If it says WebM but you expected WAV, "
            "  that's a decode path issue — install FFmpeg (see above) so librosa gets real samples.  \n"
            f"- Browser wrote a silent WAV header (can happen on some OSes) → re-record in Chrome/Edge "
            "or use the **Upload** tab."
        )
        return

    with st.spinner(
        f"Extracting features & running inference...  \n"
        f"(container={mime_label}, signal={dur:.2f}s RMS={rms:.4f} peak={peak:.3f})"
    ):
        probabilities, features_normed, n_windows = run_inference(file_bytes)

    top_class_idx = int(np.argmax(probabilities))
    top_class_name = ANIMAL_CLASSES[top_class_idx]
    confidence = float(probabilities[top_class_idx]) * 100
    top_emoji = ANIMAL_EMOJI.get(top_class_name, "🎯")

    top3_idx = np.argsort(probabilities)[-3:][::-1]

    # --- Result cards ---
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        windows_note = f"averaged over {n_windows} windows" if n_windows > 1 else "single 3s window"
        st.markdown(
            f"""
            <div class="result-card">
              <div class="label">Prediction</div>
              <div class="big">{top_emoji} {top_class_name}</div>
              <div class="sub">Confidence <b>{confidence:.1f}%</b> · {windows_note}</div>
              <div class="sub">Signal {dur:.2f}s · RMS {rms:.4f} · Peak {peak:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        rows = "".join(
            (
                f'<div class="rank-row">'
                f'<span class="rank">{i + 1}</span>'
                f'<span class="rank-name">{ANIMAL_EMOJI.get(ANIMAL_CLASSES[j], "🐾")} {ANIMAL_CLASSES[j]}</span>'
                f'<span class="rank-bar"><span style="width:{probabilities[j] * 100:.1f}%"></span></span>'
                f'<span class="rank-pct">{probabilities[j] * 100:.1f}%</span>'
                f'</div>'
            )
            for i, j in enumerate(top3_idx)
        )
        st.markdown(
            f'<div class="result-card"><div class="label">Top Predictions</div>{rows}</div>',
            unsafe_allow_html=True,
        )

    # --- Confidence / bias feedback ---
    crow_bias_flag = (top_class_name == "Crow") and (confidence < 65 or rms < 5e-3)
    if crow_bias_flag:
        st.warning(
            "⚠️ **Crow was predicted with weak signal — this is suspicious.**  \n"
            f"Confidence={confidence:.1f}%, input RMS={rms:.4f}. Crow is this network's "
            "**bias class** on silence / noise / failed decodes.  \n"
            "👉 Re-record with louder, longer sound or verify mic input; otherwise this is "
            "almost certainly NOT a real Crow."
        )
    elif confidence < 50:
        st.warning(
            "⚠️ **Low confidence** (<50%).  \n"
            "Try: cleaner recording, longer duration, moving the mic closer, "
            "or switching the standardization method in the advanced settings."
        )
    elif confidence < 70:
        st.info(
            "💡 **Moderate confidence.** Consider verifying with a clearer / longer sample."
        )
    else:
        st.balloons()

    # --- Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Log Mel-Spectrogram (Channel 0)")
        fig_mel = px.imshow(
            features_normed[0, :, :, 0],
            origin="lower",
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"x": "Time frames", "y": "Mel bins", "color": "Normed dB"},
        )
        st.plotly_chart(fig_mel, use_container_width=True)

    with col2:
        st.subheader("Class Probabilities")
        prob_fig = px.bar(
            x=ANIMAL_CLASSES,
            y=probabilities,
            labels={"x": "Animal Class", "y": "Probability"},
            title="Classification Confidence Breakdown",
            color=probabilities,
            color_continuous_scale="Blues",
        )
        prob_fig.update_layout(yaxis_range=[0, max(1.0, float(np.max(probabilities)) * 1.1)])
        st.plotly_chart(prob_fig, use_container_width=True)

# --- Streamlit UI Controls: Tabs for Upload vs Microphone ---
tab_upload, tab_mic = st.tabs(["📁 Upload Audio File", "🎙️ Record from Microphone"])

with tab_upload:
    audio_file = st.file_uploader(
        "Upload Audio File (.wav, .mp3, .ogg)",
        type=["wav", "mp3", "ogg"],
        key="upload_audio",
    )

    if audio_file is not None:
        file_bytes = audio_file.read()

        if not file_bytes:
            st.warning("⚠️ Uploaded file seems empty. Please try a different file.")
        else:
            audio_mime, audio_label = detect_audio_mime(file_bytes)
            # Fallback to extension-based mime if magic bytes were inconclusive
            if audio_mime == "application/octet-stream":
                ext = os.path.splitext(audio_file.name)[1].lower()
                if ext == ".mp3":
                    audio_mime = "audio/mpeg"
                elif ext == ".ogg":
                    audio_mime = "audio/ogg"
                else:
                    audio_mime = "audio/wav"
            st.audio(file_bytes, format=audio_mime)
            st.caption(f"📦 Detected: `{audio_label}` — playing as `{audio_mime}`")

            if st.button("🔍 Classify Sound", key="classify_upload", type="primary"):
                classify_audio_bytes(file_bytes)

with tab_mic:
    st.markdown(
        "### How to record\n"
        "1. Click the **microphone button** below — your browser will ask for permission.\n"
        "2. Make the animal sound (or play it near the mic).\n"
        "3. Click **stop** when you're done.\n"
        "4. Confirm the **🔊 Decoded signal** line shows RMS > 0.001 (not silence!).\n"
        "5. Then press **Classify Recorded Sound**."
    )

    if "mic_recorded_bytes" not in st.session_state:
        st.session_state.mic_recorded_bytes = None
    if "mic_input_nonce" not in st.session_state:
        st.session_state.mic_input_nonce = 0

    recorded = st.audio_input(
        "Record from microphone",
        sample_rate=SR,
        key=f"mic_audio_input_{st.session_state.mic_input_nonce}",
    )
    if recorded is not None:
        raw = recorded.getvalue()
        if raw:
            st.session_state.mic_recorded_bytes = bytes(raw)

    with st.expander("Alternative: upload a recording from another app"):
        mic_uploaded = st.file_uploader(
            "Recorded audio file",
            type=["wav", "mp3", "ogg", "m4a"],
            key="mic_fallback_uploader",
        )
        if mic_uploaded is not None:
            raw = mic_uploaded.read()
            if raw:
                st.session_state.mic_recorded_bytes = bytes(raw)

    with st.expander("🛠️ Troubleshooting (if RMS ≈ 0 or playback silent)"):
        st.markdown(
            "1. **Mic permission:** Click the 🔒/🎤 icon in the browser address bar. "
            "Ensure the site is **Allowed** to use your microphone and the correct input device is picked.\n"
            "2. **OS input level:** Right-click 🔊 in tray → Sound settings → Input. "
            "Speak into the mic; the input meter must visibly move. If not, the mic is muted / wrong device.\n"
            "3. **Physical mute switch:** Many laptops and USB mics have a hardware mute button (lit red).\n"
            "4. **Browser choice:** Chrome/Edge work best for microphone capture.\n"
            "5. **Still stuck?** Record in Windows Voice Recorder, save WAV, and upload via the alternative uploader above."
        )

    mic_bytes = st.session_state.mic_recorded_bytes

    if mic_bytes is None:
        st.info(
            "👆 No recording yet. Use the controls above, then check signal stats below the audio player "
            "for RMS > 0.001 before you press Classify."
        )
    else:
        if len(mic_bytes) < 1024:
            st.warning("⚠️ Recording is too small (<1 KB). Please re-record.")
            if st.button("🗑️ Clear Recording", key="clear_mic_short"):
                st.session_state.mic_recorded_bytes = None
                st.session_state.mic_input_nonce += 1
                st.rerun()
        else:
            actual_mime, actual_label = detect_audio_mime(mic_bytes)
            # Correct playback mime — browser plays what it ACTUALLY is, not what we wished for.
            st.audio(mic_bytes, format=actual_mime)

            try:
                dur, rms, peak, _ = audio_signal_stats(mic_bytes)
            except LibrosaImportError as exc:
                st.error(_render_librosa_import_error(exc))
                st.stop()

            if dur is None:
                st.error(
                    f"⚠️ Captured {len(mic_bytes):,} bytes (container: `{actual_label}`) **but librosa "
                    f"cannot decode them.** Re-record with the microphone widget or upload a WAV file."
                )
            else:
                silent_flag = (peak < 1e-3 or rms < 1e-4)
                short_flag = (dur < 0.25)
                cap = (
                    f"📦 Container: `{actual_label}` — playing as `{actual_mime}`  \n"
                    f"🔊 Decoded signal: **{dur:.2f} s**, peak={peak:.4f}, **RMS={rms:.4f}**  \n"
                )
                if silent_flag:
                    cap += "🚩 **RMS/peak ≈ 0 → SILENCE captured.** Classifier will fake-Crow. See troubleshooting above."
                    st.warning(cap)
                elif short_flag:
                    cap += f"🚩 Only {dur:.2f} s. Re-record with at least 1–2 s of actual sound."
                    st.warning(cap)
                elif rms < 0.005:
                    cap += "ℹ️ Quiet recording (RMS < 0.005). Speak louder / move mic closer for reliable classification."
                    st.caption(cap)
                else:
                    cap += "✅ Signal looks good — you can press Classify."
                    st.caption(cap)

            col_classify, col_clear = st.columns([2, 1])
            with col_classify:
                if st.button("🔍 Classify Recorded Sound", key="classify_mic", type="primary"):
                    classify_audio_bytes(mic_bytes)
            with col_clear:
                if st.button("🗑️ Clear & Re-record", key="clear_mic"):
                    st.session_state.mic_recorded_bytes = None
                    st.session_state.mic_input_nonce += 1
                    st.rerun()

# --- Advanced settings (moved out of the sidebar into a collapsed expander) ---
with st.expander("⚙️ Advanced Inference Settings", expanded=False):
    st.markdown(
        f'<span class="model-badge">✅ Model: {MODEL_NAME}</span>',
        unsafe_allow_html=True,
    )

    norm_options = [NORM_DEFAULT]
    if CHANNEL_STATS_AVAILABLE:
        norm_options.append("Per-Channel Stats (from 6-file .npy, EXPERIMENTAL)")
    norm_options.append("Per-Sample Standardization (NOT recommended)")

    st.radio(
        "Standardization Method",
        options=norm_options,
        index=0,
        key="norm_option",
        help=(
            "How features are normalized before feeding the TFLite model.\n\n"
            "- Global Stats [DEFAULT, VERIFIED]: single scalar mean/std computed over\n"
            "  all 400 training clips (mean=-25.8526, std=34.9650). Checked against\n"
            "  80 held-out clips the model never saw — this is the safest choice.\n"
            "- Per-Channel Stats [EXPERIMENTAL, shown only if .npy files exist]:\n"
            "  separate stats for LogMel vs MFCC channels. Derived from only 6 files.\n"
            "  Anecdotally better on some inputs, worse on others; use for A/B tests.\n"
            "- Per-Sample Standardization: NEVER the training pipeline — destroys\n"
            "  absolute magnitude info that the BatchNorm layers depend on."
        ),
    )

    st.checkbox(
        "Enable Time-Slice Voting",
        value=True,
        key="use_voting",
        help=(
            "If audio is longer than 3s, split it into overlapping 3s windows, "
            "classify each, and average probabilities. Improves microphone "
            "recordings by looking at the whole clip, not just the first 3s."
        ),
    )
    st.slider(
        "Voting Window Step (seconds)",
        min_value=0.25,
        max_value=1.0,
        value=0.5,
        step=0.25,
        key="voting_step",
        help="Smaller step = more windows = slower but smoother average.",
    )

    st.divider()
    st.info(
        "📐 Input: `[1,128,130,2]`\n\n"
        "```\n"
        "Model architecture (from TFLite graph):\n"
        "  - 2 DepthwiseConv + PointwiseConv blocks\n"
        "  - Channels: 32 -> 64\n"
        "  - GlobalAvgPool -> Dense(96) -> Dense(10)\n"
        "  - Held-out eval: ~60-75% top-1 (small network)\n"
        "```"
    )
    st.caption(f"Stats source: {STATS_FILE_SOURCE}")

    with st.expander("Loaded normalization stats"):
        st.write(
            f"**Global (400-clip, VERIFIED)**  \n"
            f"`mean = {GLOBAL_MEAN:.6f}`, `std = {GLOBAL_STD:.6f}`"
        )
        if CHANNEL_STATS_AVAILABLE:
            cm_flat = CHANNEL_MEAN.flatten()
            cs_flat = CHANNEL_STD.flatten()
            st.write(
                "**Per-Channel (6-file proxy, EXPERIMENTAL)**  \n"
                f"  - Ch0 (LogMel):     mean={cm_flat[0]:.3f}, std={cs_flat[0]:.3f}  \n"
                f"  - Ch1 (MFCC+Δ+ΔΔ):  mean={cm_flat[1]:.3f}, std={cs_flat[1]:.3f}  \n"
                f"  ⚠️ These were NOT regenerated over the full 400-clip training set."
            )
        else:
            st.write("**Per-Channel:** *unavailable* — no valid `feature_mean_ch.npy` / `feature_std_ch.npy` on disk.")

with st.expander("🔬 Accuracy notes, model limits, and suggested improvements"):
    st.markdown(
        f"""
### Normalization stats (defaults to VERIFIED global scalar)

The stats used as the app default are computed from **the full 400-clip ESC-50 animal
training split** using the exact feature extractor from the training pipeline, and
were verified on **80 held-out clips** the model never saw:

- **Global scalar [DEFAULT, VERIFIED]:**
  `mean = {REAL_TRAINING_MEAN:.6f}`, `std = {REAL_TRAINING_STD:.6f}`
- The old 3-file guess of `(-23.1244, 34.2621)` is gone from every code path.
- **Per-Channel mode** (shown in the advanced settings radio only if the `.npy` files
  exist on disk) is a 6-file proxy set, NOT regenerated over the full 400-clip training
  run. Use it only for A/B tests, not as the default.

### If your results are suddenly way off after an app update

Check the first radio option in the advanced settings: it should be **`Global Stats
(400-clip training, VERIFIED default)`**. If you previously used an older version of this
app that defaulted to an experimental Per-Channel mode, that was switched to fabricated
fallback numbers and produced low-confidence random guesses on some machines. Flip back
to Global (the current default) and re-classify.

### Why top-1 accuracy still won't be 90%+

The deployed TFLite model's actual flatbuffer compute graph is:

- **Input:** `[1, 128, 130, 2]` (LogMel + stacked MFCC/Δ/ΔΔ)
- **Backbone:** 2× (DepthwiseConv2D 3×3 → Conv2D 1×1 → BN → ReLU → MaxPool)
  - Channels: 32 → 64
- **Head:** GlobalAveragePool → Dense(96, ReLU) → Dense(10) → Softmax

This is a **very small network** for 10-way audio classification. Held-out eval puts the
realistic ceiling at roughly **60–75 % top-1** even with the corrected normalization —
so some misclassifications (especially on noisy mic recordings) are baked into the
architecture, not a bug you can fix at inference time.

### Known weak class: Sheep (~25 % on held-out)

From the 80-clip eval, **Sheep** is the class dragging the average down the most.
Typical confusions are with Cat, Crow, Cow, Pig, or Insects. If Sheep matters for your
use case, the most effective fix is:
1. Visually compare a few Sheep mel-spectrograms vs the classes they are confused with —
   if there's overlap in the patterns, the network simply can't tell them apart.
2. Add targeted augmentation on the Sheep class during retraining: ±semitone pitch-shift,
   ±10 % time-stretch, light background-noise injection, and SpecAug-style time/freq
   masking. This usually helps much more than tweaking normalization on a fixed model.

### Inference-time things to try (already built in!)

1. **Standardization method** in the advanced settings. Keep the default **Global Stats
   (VERIFIED)** unless you are actively A/B testing. The Per-Channel option is only 6-file
   proxy stats and may be better or worse depending on the specific clip.
2. **Time-Slice Voting ON** (default): on 4–5 s mic recordings, sliding 3 s windows
   and averaging probabilities reliably gains several % because the model no longer only
   inspects the first 3 s (which may be pre-sound silence).
3. Always check the **Top-3** predictions, not just Top-1 — for this small network,
   the correct class is very often rank-2 or rank-3 even when Top-1 is wrong.

### Pushing past ~75 %: the real fix

All of the above are band-aids on a fixed under-capacity model. To go beyond:

- **Retrain with a deeper backbone:** 3 separable-conv blocks (channels 64→128→128 or
  128→256→256), GlobalAvgPool → Dense(128, ReLU, dropout) → Dense(10). This directly
  lifts the representational ceiling before any data tuning is done.
- **Bake a normalization layer into the exported graph** (e.g. a preprocessing
  `Rescaling` / `Normalization` Keras layer at the input) so every deployment of the
  `.tflite` uses the correct stats by construction — no sidebars, no `.npy` files to
  lose.
- If your real-world input is *microphone* recordings (not curated ESC-50 WAVs),
  fine-tune on a pretrained audio backbone (YAMNet / PANNs / Audio Spectrogram
  Transformer) with a few hundred in-domain mic examples — domain mismatch between
  ESC-50-studio and laptop-mic is responsible for a bigger accuracy drop than
  normalization noise.
        """
    )
