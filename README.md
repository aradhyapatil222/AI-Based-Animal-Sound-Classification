# Real-Time Animal Voice Detection & Classifier

An AI system that classifies animal sounds (13 classes) using a **transfer-learning pipeline** — Google's **YAMNet** audio feature extractor (frozen) feeding a small trained TFLite classifier head — available in two frontends:

| Frontend | Stack | Entry point |
|----------|-------|-------------|
| **WildEcho AI** (default, modern UI) | React + TypeScript + Vite calling a **FastAPI** backend | `server.py` + `frontend/` |
| **Streamlit app** (classic UI) | Streamlit, self-contained | `app.py` |

Both support uploading an audio file (`.wav` / `.mp3` / `.ogg`) and (Streamlit) recording directly from the microphone. Results include the predicted animal, confidence %, YAMNet's internal log-mel spectrogram, and a per-class probability breakdown.

---

## 13 Animal Classes

`Cat`, `Chicken`, `Cow`, `Cricket`, `Dog`, `Donkey`, `Elephant`, `Frog`, `Lion`, `Monkey`, `Pig`, `Sheep`, `Squirrel`

> Measured performance: **~91 % top-1 accuracy** on the held-out test split of the training notebook
> (`classifier_head1.tflite` = dense head trained on mean-pooled YAMNet embeddings).
> Because the heavy lifting is done by YAMNet's ImageNet-scale pretraining, this pipeline
> generalizes far better to noisy microphone recordings than the old from-scratch spectrogram CNN (~60–75 %).

> **First-run note:** YAMNet is pulled from TensorFlow Hub on startup (~15 MB, then cached).
> An internet connection is required the first time you run `server.py` or `app.py`.

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

Expected predictions for the 3 hand-labeled samples included in the repo (measured with the current pipeline, voting ON):

| File | Correct label | Result | Confidence | Notes |
|------|---------------|--------|------------|-------|
| `1-100032-A-0.wav` | Dog | ✅ Dog | ~92 % | Cleanest of the 3 samples |
| `1-77241-A-3.wav`  | Cow  | ✅ Cow | ~87 % | High-confidence class |
| `1-15689-A-4.wav`  | Frog | ✅ Frog | ~73 % | Croak overlaps with other nature sounds |

> These 3 files are example outputs only. Real model performance was measured on the
> held-out test split in the training notebook (~91 % top-1).

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
├── classifier_head1.tflite           # PRIMARY model: trained dense head over YAMNet embeddings (13 classes)
├── label_map.json                    # Class-index → animal-name mapping (must match the head's output order)
│
├── animal_sound_model_quant_final.tflite   # OLD end-to-end CNN (kept for reference, no longer used)
├── animal_sound_model_quant.tflite         # OLD fallback model (kept for reference)
│
├── feature_mean.npy / feature_std.npy        # OLD pipeline stats (kept for reference)
├── feature_mean_ch.npy / feature_std_ch.npy  # OLD per-channel stats (kept for reference)
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
| `/api/health` | GET | Model file, classes, embedding dim, window/sample-rate config, server status |
| `/api/samples` | GET | List of bundled sample `.wav` clips |
| `/api/sample/{filename}` | GET | Stream a bundled sample clip (`audio/wav`) |
| `/api/classify` | POST | Multipart: `file` + `use_voting` (bool) + `voting_step` (float) → prediction, top-3, per-class probabilities, spectrogram matrix, and signal stats |

Interactive docs are available at **http://localhost:8000/docs** (Swagger UI). CORS is wide-open so the Vite dev server can call it directly.

## How It Works 

### Two-stage inference pipeline (mirrors the training notebook exactly)

1. **Resample** every input to 16 kHz mono with librosa.
2. **Window:** slide a 1.0 s window across the clip (hop = 0.5 s when voting is on); windows whose RMS < 0.01 are treated as silence and skipped.
3. **YAMNet (frozen, TF-Hub):** each 1 s window → per-frame 1024-dim embeddings → mean-pooled over frames → one 1024-dim vector per window (identical to the notebook's `extract_embedding()`).
4. **classifier_head1.tflite:** softmax over the 13 classes for each window's embedding.
5. **Vote:** average the per-window softmax outputs → final probabilities.

### Display

- YAMNet's internal log-mel spectrogram (64 mel bins) is returned for the first window and rendered as a heatmap.
- Per-class probability bars show all 13 classes; top-3 are highlighted.

---

## Inference-time knobs

Same settings exist in the Streamlit UI (`app.py`) and as multipart form fields on `/api/classify` (used by the React frontend):

- **Time-Slice Voting** (`use_voting`, `voting_step`) — averages predictions across overlapping 1 s windows; strongly recommended for anything ≥ 2 s or noisy mic recordings.
- **Top-3 predictions** are always displayed regardless of confidence, not just Top-1.

---

## Troubleshooting 

| Problem | Fix |
|---------|-----|
| `python` / `pip` not found | Try `python3` / `pip3`, or re-run the installer and check *Add Python to PATH* |
| `ModuleNotFoundError: No module named ...` | Re-activate venv, then `pip install -r requirements.txt` |
| `OSError: sndfile library not found` (Linux/Mac) | Linux: `sudo apt-get install libsndfile1`. macOS: `brew install libsndfile`. (Windows ships `soundfile` binaries fine.) |
| Microphone widget is greyed out / not appearing | The mic tab uses Streamlit's built-in `st.audio_input` widget (Chrome / Edge give more reliable mic permissions than Firefox in practice). |
| Browser blocks microphone access (`getUserMedia` errors) | Streamlit must be served over **HTTPS** or **localhost** for mic permissions to work. On a remote server, use `--server.address 0.0.0.0` with an HTTPS reverse proxy, or `ssh -L 8501:localhost:8501` tunneling. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| First request after startup is slow | Normal — TF init + YAMNet load + TFLite interpreter warm-up caches after first run. |
| `hub.load` fails / first run needs internet | YAMNet downloads from TensorFlow Hub on first launch (~15 MB) and is cached afterwards. Re-run once online; subsequent runs work offline. |
| `ModuleNotFoundError: tensorflow_hub` or `pkg_resources` | `pip install -r requirements.txt` (includes `tensorflow-hub` and `setuptools<81`, which still ships `pkg_resources`). |
| `RuntimeError: classifier_head1.tflite outputs N classes but ANIMAL_CLASSES has M entries` | `label_map.json` doesn't match the converted model. Re-export both from the training notebook together. |
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
- Sample WAVs, `classifier_head1.tflite`, `label_map.json`, the two legacy `.tflite` models, and the four `.npy` stats files **ARE included** — they are essential for a clone-and-run experience. If you want a smaller repo, remove the 6 `.wav` samples (~260 KB) and/or the 2 legacy `.tflite` + 4 `.npy` files before pushing.
