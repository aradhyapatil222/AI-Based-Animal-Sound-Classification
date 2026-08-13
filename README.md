# Real-Time Animal Voice Detection & Classifier

An AI system that classifies animal sounds (10 ESC-50 animal classes) using a quantized TFLite CNN, available in two frontends:

| Frontend | Stack | Entry point |
|----------|-------|-------------|
| **WildEcho AI** (default, modern UI) | React + TypeScript + Vite calling a **FastAPI** backend | `server.py` + `frontend/` |
| **Streamlit app** (classic UI) | Streamlit, self-contained | `app.py` |

Both support uploading an audio file (`.wav` / `.mp3` / `.ogg`) and (Streamlit) recording directly from the microphone. Results include the predicted animal, confidence %, a log-mel spectrogram, and a per-class probability breakdown.

---

## 10 Animal Classes

`Dog`, `Rooster`, `Pig`, `Cow`, `Frog`, `Cat`, `Hen`, `Insects`, `Sheep`, `Crow`

> Realistic performance expectation on held-out data: **~60–75 % top-1** accuracy for the
> current TFLite model. The network has a very small backbone (2 separable conv blocks,
> 64 channels, 96 hidden units) and saturates in this range even with perfect
> standardization and clean inputs. Noisy microphone recordings will land below the
> upper end of that range.

---

## Quick Start (fresh machine — after cloning from GitHub)

### Step 1 — Clone the repo & open the folder

```bash
git clone <YOUR_REPO_URL>
cd <REPO_NAME>
```

### Step 2 — Create a Python virtual environment (Python 3.11+ recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

(TensorFlow is the largest package — expect several minutes.)

### Step 4 — Run the app

You can run either frontend. **Option A (React)** is recommended.

#### Option A — WildEcho AI: FastAPI backend + React frontend

**Terminal 1 — start the API server (http://localhost:8000):**

```bash
python server.py
```

**Terminal 2 — start the React dev server:**

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — the frontend talks to the FastAPI backend at `http://127.0.0.1:8000`.

#### Option B — Streamlit app

```bash
streamlit run app.py
```

Your browser will automatically open **http://localhost:8501** — that's the app.

### Step 5 — Try it

1. **Upload mode:** choose any of the 6 sample `.wav` files included, or upload your own animal clip (3–5 s recommended).
2. **Microphone mode (Streamlit only):** pick a recording duration (3–5 s), click *Start Recording*, produce the sound, then classify.
3. Click **Classify Sound** / **Classify Recorded Sound** to run inference.

Expected predictions for the 3 hand-labeled samples included in the repo (these are *not* a benchmark — see the accuracy note above):

| File | Correct label | Expected in top-3? | Notes |
|------|---------------|--------------------|-------|
| `1-100032-A-0.wav` | Dog | Usually Yes | Cleanest of the 3 samples |
| `1-15689-A-4.wav` | Frog | Usually Yes | Low confidence on small model |
| `1-77241-A-3.wav`  | Cow  | Usually Yes | High-confidence class |

> These 3 files are example outputs only. Real model performance was measured on **80 held-out clips** from the training split and is the source of the 60–75 % ceiling cited above.

---

## Project Layout 

```
Project 1/
├── app.py                          # Main Streamlit web app  <-- classic entry point
├── server.py                       # FastAPI backend (WildEcho AI API, port 8000)  <-- API entry point
├── requirements.txt                # Exact pinned dependency versions
├── README.md                       # This file
├── .gitignore                      # Ignores venv/, logs, cache, temp scripts
│
├── frontend/                       # React + TypeScript + Vite UI (WildEcho AI)
│   ├── src/                        #   Components, hooks, types
│   ├── index.html                  #   Vite entry
│   ├── package.json                #   npm scripts: dev / build / lint / preview
│   └── vite.config.ts              #   Dev server + build config
│
├── animal_sound_model_quant_final.tflite   # PRIMARY model (small MobileNet-like CNN)
├── animal_sound_model_quant.tflite         # FALLBACK model (kept for reference)
│
├── feature_mean.npy               # Real 400-clip training GLOBAL mean = -25.852587
├── feature_std.npy                # Real 400-clip training GLOBAL std  =  34.965031
├── feature_mean_ch.npy            # Per-channel (LogMel / MFCC) means from 400-clip run
├── feature_std_ch.npy             # Per-channel (LogMel / MFCC) stds  from 400-clip run
│
├── 1-100032-A-0.wav               # Sample wav: Dog   (ESC-50: 1-100032-A-0)
├── 1-77241-A-3.wav                # Sample wav: Cow
├── 1-15689-A-4.wav                # Sample wav: Frog
├── 1-155858-D-25.wav              # Extra sample (no ground truth label in repo)
├── 1-28135-B-11.wav               # Extra sample (no ground truth label in repo)
└── 2-73544-A-27.wav               # Extra sample (no ground truth label in repo)
```

---

## WildEcho AI API (FastAPI)

The backend exposes a small REST API on **http://localhost:8000**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Model file, classes, input shape, standardization stats, server status |
| `/api/samples` | GET | List of bundled sample `.wav` clips |
| `/api/sample/{filename}` | GET | Stream a bundled sample clip (`audio/wav`) |
| `/api/classify` | POST | Multipart: `file` + `norm_option` (form) + `use_voting` (bool) + `voting_step` (float) → prediction, top-3, per-class probabilities, spectrogram matrix, and signal stats |

Interactive docs are available at **http://localhost:8000/docs** (Swagger UI). CORS is wide-open so the Vite dev server can call it directly.

## How It Works 

### Audio → Features (matches the training pipeline exactly)

1. `librosa.load(..., sr=22050, duration=3)` — mono, 22.05 kHz, clipped/padded to 3 s.
2. **Channel 0:** 128-bin log-mel spectrogram (`n_fft=2048, hop=512`).
3. **Channel 1:** 20 MFCCs + delta + delta-delta → vertically padded to 128 rows.
4. Stack → `(1, 128, 130, 2)` input tensor.

### Standardization

Configurable via `norm_option` (Streamlit sidebar or API form field; defaults to *Per-Channel Stats* — recommended):

| Method | Formula / source | Notes |
|--------|------------------|-------|
| **Per-Channel** (default) | Ch0: `-49.472 / 22.831`, Ch1: `-2.233 / 32.614` from the 400-clip training run | Best default because LogMel lives in dB and the MFCC channel is centered near 0 |
| **Global scalar** | `(x - (-25.852587)) / 34.965031` — 400-clip training mean/std | Kept for A/B comparison with older runs |
| Per-Sample | `(x - mean(x)) / std(x)` individually per clip | Not recommended — destroys the absolute magnitude information the BatchNorm layers were trained with |

### Inference

1. Load `animal_sound_model_quant_final.tflite` via TFLite Interpreter.
2. **Optional (default ON) — Time-slice voting:** for audio clips longer than 3 s, slide a 3 s window every 0.5 s across the clip, run inference on each window, and average the returned probabilities. Improves microphone recordings substantially because the first 3 s often contain silence before the sound.
3. Argmax over the 10 probabilities → prediction. Plot log-mel spectrogram and a 10-bar probability breakdown.

### Standardization stats provenance

`feature_mean.npy` / `feature_std.npy` were regenerated by running the exact
training-pipeline feature extractor across the **full 400-clip ESC-50 animal
training subset**, then verified against **80 held-out clips the model never saw
during training**. Fallback values hardcoded in `app.py` are these same numbers
(so the app behaves correctly even if `.npy` files are lost), but the files
themselves should exist so the app is reproducible without reading source.

The previous proxy values of `(-23.1244, 34.2621)` were derived from only 3
example files and have been removed from every code path in this repo.

---

## Inference-time knobs

Same settings exist in the Streamlit sidebar (`app.py`) and as multipart form fields on `/api/classify` (used by the React frontend):

- **Standardization method toggle** (`norm_option`) — directly compare Global vs Per-Channel on any input.
- **Time-Slice Voting** (`use_voting`, `voting_step`) — enable for anything ≥ 4 s.
- **Top-3 predictions** are always displayed regardless of confidence, not just Top-1.
  The correct class very often lives at rank 2–3 when Top-1 is wrong on this small model.

---

## Troubleshooting 

| Problem | Fix |
|---------|-----|
| `python` / `pip` not found | Try `python3` / `pip3`, or re-run the installer and check *Add Python to PATH* |
| `ModuleNotFoundError: No module named ...` | Re-activate venv, then `pip install -r requirements.txt` |
| `OSError: sndfile library not found` (Linux/Mac) | Linux: `sudo apt-get install libsndfile1`. macOS: `brew install libsndfile`. (Windows ships `soundfile` binaries fine.) |
| Microphone widget is greyed out / not appearing | Make sure `streamlit-mic-recorder` is in your installed env (`pip show streamlit-mic-recorder`). Chrome / Edge give more reliable mic permissions than Firefox in practice. |
| Browser blocks microphone access (`getUserMedia` errors) | Streamlit must be served over **HTTPS** or **localhost** for mic permissions to work. On a remote server, use `--server.address 0.0.0.0` with an HTTPS reverse proxy, or `ssh -L 8501:localhost:8501` tunneling. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Slow first load of `load_tflite_model()` | Normal — TFLite interpreter + TensorFlow first-load init caches after first run. |
| Frontend shows "Backend server offline" | Start the API first: `python server.py`, then reload **http://localhost:5173**. |
| React app doesn't reach the API | Confirm the server runs on **port 8000** and that `API_BASE` in `frontend/src/App.tsx` matches (default `http://127.0.0.1:8000`). |
| `ModuleNotFoundError: fastapi / uvicorn` | `pip install -r requirements.txt` (now includes both). |
| Port 8000 already in use | Change the port in the `uvicorn.run(...)` line at the bottom of `server.py` and update `API_BASE` in `frontend/src/App.tsx` to match. |

---

## GitHub Checklist (for the uploader)

If you created the repo, you don't need to do anything else — `.gitignore` already excludes:

- `venv/`
- `__pycache__/`
- All `*.log`, `*.tmp`, and one-shot diagnostic scripts (`diagnostic.py`, `test_model.py`, `validate_fix.py`, etc.)
- Sample WAVs, the two `.tflite` model files, and all four `.npy` stats files **ARE included** — they are essential for a clone-and-run experience. If you want a smaller repo, remove the 6 `.wav` samples before pushing; they're ~260 KB total.
