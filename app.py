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

# --- Constants & Classes ---
# NOTE: this app now runs YAMNet embeddings -> classifier_head1.tflite instead
# of the old spectrogram-CNN. Class list is loaded from label_map.json (produced
# by the training notebook) so it always matches the model actually shipped.
import json as _json

_LABEL_MAP_PATH = "label_map.json"
MODEL_PATH = "classifier_head1.tflite"
MODEL_NAME = "classifier_head1"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Could not find {MODEL_PATH}. Place the trained classifier head tflite file next to app.py.")
    st.stop()

if os.path.exists(_LABEL_MAP_PATH):
    with open(_LABEL_MAP_PATH) as _f:
        _raw_label_map = _json.load(_f)
    ANIMAL_CLASSES = [_raw_label_map[str(i)].capitalize() for i in range(len(_raw_label_map))]
else:
    ANIMAL_CLASSES = [
        'Cat', 'Chicken', 'Cow', 'Cricket', 'Dog', 'Donkey',
        'Elephant', 'Frog', 'Lion', 'Monkey', 'Pig', 'Sheep', 'Squirrel'
    ]

ANIMAL_EMOJI = {
    'Cat': '🐱', 'Chicken': '🐔', 'Cow': '🐮', 'Cricket': '🦗', 'Dog': '🐶',
    'Donkey': '🐴', 'Elephant': '🐘', 'Frog': '🐸', 'Lion': '🦁',
    'Monkey': '🐒', 'Pig': '🐷', 'Sheep': '🐑', 'Squirrel': '🐿️',
    'Not_animal': '🔇',
}

# Audio / windowing constants — MUST match the training notebook
SR = 16000
WINDOW_DURATION = 1.0
WINDOW_SAMPLES = int(SR * WINDOW_DURATION)
DURATION = WINDOW_DURATION  # kept for compatibility with helper functions below
SILENCE_RMS_THRESHOLD = 0.01

# --- Hero header ---
st.markdown("""
<div class="hero">
  <h1>🐾 WildEcho <span class="accent">AI</span></h1>
  <p>Real-time <b>animal voice detection &amp; classification</b> running on a quantized
     <b>Edge TFLite</b> model. Upload an audio clip or record live from your microphone —
     get an instant prediction.</p>
  <div class="chips">
    {chips}
  </div>
</div>
""".format(chips="".join(f'<span class="chip">{ANIMAL_EMOJI.get(c, "🐾")} {c}</span>' for c in ANIMAL_CLASSES)), unsafe_allow_html=True)

# No standardization step needed: YAMNet embeddings are fed to the classifier
# head as-is (this matches how the head was trained in the notebook).
def standardize_features(features):
    return features

@st.cache_resource
def load_yamnet_model():
    import tensorflow_hub as hub
    return hub.load("https://tfhub.dev/google/yamnet/1")

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

def _is_silent(window):
    return float(np.sqrt(np.mean(window ** 2))) < SILENCE_RMS_THRESHOLD

# --- Feature extraction: one 1s window -> YAMNet embedding (+ log-mel for display) ---
def _extract_window(file_bytes, offset_sec, duration_sec=WINDOW_DURATION):
    y, sr = _load_audio_bytes(file_bytes, sr=SR, offset=offset_sec, duration=duration_sec)
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    else:
        y = y[:WINDOW_SAMPLES]
    y = y.astype(np.float32)

    yamnet_model = load_yamnet_model()
    scores, embeddings, spectrogram = yamnet_model(y)
    import tensorflow as tf
    embedding = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)  # (1024,)
    log_mel = spectrogram.numpy()  # (frames, 64) — for spectrogram display only
    return embedding[np.newaxis, :], log_mel, y

# --- Legacy single-window extractor (still used for spectrogram display) ---
def extract_dual_channel_features(file_bytes):
    embedding, log_mel, _ = _extract_window(file_bytes, offset_sec=0.0, duration_sec=WINDOW_DURATION)
    return log_mel  # used only by the spectrogram viewer, not fed to the model

# --- Full classification: features (optionally multi-window voting) -> probabilities ---
def run_inference(file_bytes):
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    use_voting = st.session_state.get("use_voting", True)
    voting_step = st.session_state.get("voting_step", 0.5)

    # Always compute the t=0 embedding for spectrogram display + fallback prediction
    first_embedding, first_log_mel, first_window = _extract_window(file_bytes, offset_sec=0.0)

    y_full, _ = _load_audio_bytes(file_bytes, sr=SR)
    total_dur = len(y_full) / SR

    if not use_voting or total_dur <= WINDOW_DURATION:
        interpreter.set_tensor(input_details[0]["index"], first_embedding)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]["index"])[0]
        return probs, first_log_mel, 1

    # Voting: slide 1s windows (hop = voting_step) across the whole clip,
    # matching the 50% overlap windowing used to build the training set.
    offsets = []
    t = 0.0
    while t + WINDOW_DURATION <= max(total_dur, WINDOW_DURATION) + 1e-9:
        offsets.append(t)
        t += voting_step
    if not offsets:
        offsets = [0.0]

    votes = []
    for i, t0 in enumerate(offsets):
        if i == 0:
            embedding, window = first_embedding, first_window
        else:
            embedding, _, window = _extract_window(file_bytes, offset_sec=t0)
        if _is_silent(window):
            continue
        interpreter.set_tensor(input_details[0]["index"], embedding)
        interpreter.invoke()
        votes.append(interpreter.get_tensor(output_details[0]["index"])[0])

    if not votes:
        interpreter.set_tensor(input_details[0]["index"], first_embedding)
        interpreter.invoke()
        votes = [interpreter.get_tensor(output_details[0]["index"])[0]]

    avg_probs = np.mean(votes, axis=0)
    return avg_probs, first_log_mel, len(votes)

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

    # 2) Silent / nearly-silent input → prediction will be unreliable / essentially random
    NEAR_SILENCE_RMS = 1e-4
    VERY_SHORT_S = 0.25
    if dur < VERY_SHORT_S:
        st.warning(
            f"⚠️ Audio loaded but it is only **{dur:.2f} s** — too short for classification "
            f"(need at least ~{VERY_SHORT_S:.1f} s of sound). **Re-record with at least 1–2 s of actual sound.**"
        )
        return
    if peak < 1e-3 or rms < NEAR_SILENCE_RMS:
        st.warning(
            "⚠️ **The decoded audio is essentially silent.**  \n"
            f"Signal stats: length={dur:.2f}s, peak={peak:.2e}, RMS={rms:.2e}  \n"
            "Classifying silence produces an unreliable, essentially random prediction, "
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
        windows_note = f"averaged over {n_windows} windows" if n_windows > 1 else "single 1s window"
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

    # --- Confidence feedback ---
    weak_signal_flag = confidence < 65 and (rms < 5e-3 or dur < 0.5)
    if weak_signal_flag:
        st.warning(
            "⚠️ **Prediction made with a weak/short signal — treat with caution.**  \n"
            f"Confidence={confidence:.1f}%, input RMS={rms:.4f}, duration={dur:.2f}s.  \n"
            "👉 Re-record with a louder, longer (1s+) sound or verify mic input."
        )
    elif confidence < 50:
        st.warning(
            "⚠️ **Low confidence** (<50%).  \n"
            "Try: cleaner recording, longer duration, or moving the mic closer."
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
        st.subheader("YAMNet Log Mel-Spectrogram (first 1s window)")
        # features_normed here is YAMNet's own internal log-mel, shape (frames, 64)
        fig_mel = px.imshow(
            features_normed.T,
            origin="lower",
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"x": "Time frames", "y": "Mel bins", "color": "dB"},
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
                    cap += "🚩 **RMS/peak ≈ 0 → SILENCE captured.** Prediction will be unreliable. See troubleshooting above."
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

    st.checkbox(
        "Enable Time-Slice Voting",
        value=True,
        key="use_voting",
        help=(
            "If audio is longer than 1s, split it into overlapping 1s windows "
            "(matching how the training set was built), classify each, and average "
            "probabilities. Improves microphone recordings by looking at the whole "
            "clip instead of just the first second."
        ),
    )
    st.slider(
        "Voting Window Step (seconds)",
        min_value=0.25,
        max_value=1.0,
        value=0.5,
        step=0.25,
        key="voting_step",
        help="Smaller step = more windows = slower but smoother average. 0.5s matches training (50% overlap).",
    )

    st.divider()
    st.info(
        "📐 Pipeline: 16kHz mono audio → 1s windows → **YAMNet** (frozen, TF-Hub) "
        "→ mean-pooled 1024-dim embedding → **classifier_head1.tflite**\n\n"
        f"```\n"
        f"Classifier head:\n"
        f"  Input:  (1024,) YAMNet embedding\n"
        f"  Dense(256, relu) -> Dropout(0.3)\n"
        f"  Dense(128, relu) -> Dropout(0.2)\n"
        f"  Dense({len(ANIMAL_CLASSES)}, softmax)\n"
        f"```"
    )
    st.caption(
        "No feature standardization is applied — YAMNet embeddings are fed to the "
        "classifier head as-is, matching how the head was trained."
    )

with st.expander("🔬 Accuracy notes, model limits, and suggested improvements"):
    st.markdown(
        f"""
### How this model differs from the old spectrogram-CNN version

This app now uses **transfer learning on YAMNet** instead of a small CNN trained from
scratch on log-mel/MFCC spectrograms. YAMNet is a large model pretrained on ~2M YouTube
clips (AudioSet), frozen here and used purely as a feature extractor; only the small
dense head on top ({len(ANIMAL_CLASSES)}-way classifier) is actually trained on your
animal-sound clips. This generally gives much better accuracy per training clip than
training a CNN from scratch, especially with a modest dataset size.

### Current classes

`{', '.join(ANIMAL_CLASSES)}`

### Known limitations

- **Class imbalance:** if some classes have far fewer training clips than others, expect
  lower precision/recall on those specific classes — check the confusion matrix from your
  training notebook to see which ones need more data.
- **Domain mismatch:** if the training clips came from a curated dataset (e.g. ESC-50)
  rather than real microphone recordings, live mic accuracy may be lower than the
  notebook's held-out test accuracy. Fine-tuning with real in-domain mic clips closes
  this gap the most.
- **Window granularity:** predictions are made per 1-second window (matching training);
  Time-Slice Voting averages across the whole clip, but very brief sounds shorter than
  1s will be padded with silence, which can dilute the signal.

### Inference-time things to try (already built in!)

1. **Time-Slice Voting ON** (default): on longer mic recordings, sliding 1s windows with
   50% overlap and averaging probabilities reliably improves results over classifying
   only the first second.
2. Always check the **Top-3** predictions, not just Top-1 — the correct class is often
   rank-2 or rank-3 even when Top-1 is wrong, especially for visually/acoustically
   similar animal pairs.

### Pushing accuracy further

- **Add more clips for your weakest classes** (check the classification report from
  training) rather than uniformly expanding every class.
- **Add a `not_animal` / background-noise class** during training so the model can
  reject non-animal sounds instead of being forced to pick its closest animal match.
- **Re-run training with class weights** (`sklearn.utils.class_weight.compute_class_weight`)
  if your dataset is imbalanced, so common classes don't dominate the loss.
        """
    )
