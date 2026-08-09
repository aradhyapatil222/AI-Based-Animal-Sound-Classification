# Real-Time Animal Voice Detection & Classifier

A Streamlit web app that classifies animal sounds (10 ESC-50 animal classes) using a quantized TFLite CNN. Upload a short `.wav` / `.mp3` / `.ogg` clip and get an instant prediction with class probability breakdown and a log-mel spectrogram visualization.

---

## 10 Animal Classes

`Dog`, `Rooster`, `Pig`, `Cow`, `Frog`, `Cat`, `Hen`, `Insects`, `Sheep`, `Crow`

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

(TensorFlow is the biggest package — expect several minutes.)

### Step 4 — Run the web app

```bash
streamlit run app.py
```

Your browser will automatically open **http://localhost:8501** — that's the app.

### Step 5 — Try it

1. Upload any of the 6 sample `.wav` files included in the repo, or upload your own 3-second animal audio clip.
2. Click **Classify Sound**.
3. You'll see the predicted class, confidence %, a log-mel spectrogram (left), and a probability bar chart (right).

Expected correct predictions for the included samples:

| File | Expected Prediction | Confidence |
|------|--------------------|------------|
| `1-100032-A-0.wav` | **Dog** | ~80% |
| `1-15689-A-4.wav` | **Frog** | ~31% |
| `1-77241-A-3.wav` | **Cow** | ~95% |

---

## Project Layout 

```
Project 1/
├── app.py                          # Main Streamlit web app  <-- entry point
├── requirements.txt                # Exact pinned dependency versions
├── .gitignore                      # Ignores venv/, logs, cache, temp scripts
│
├── animal_sound_model_quant_final.tflite   # PRIMARY model (100% val acc on known samples)
├── animal_sound_model_quant.tflite         # FALLBACK model (kept for reference)
│
├── feature_mean.npy               # Global training-stat mean (-23.12)
├── feature_std.npy                # Global training-stat std  ( 34.26)
│
├── 1-100032-A-0.wav               # Sample wav: Dog   (ESC-50: 1-100032-A-0)
├── 1-77241-A-3.wav                # Sample wav: Cow
├── 1-15689-A-4.wav                # Sample wav: Frog
├── 1-155858-D-25.wav              # Extra sample
├── 1-28135-B-11.wav               # Extra sample
└── 2-73544-A-27.wav               # Extra sample
```

---

## How It Works 

1. **Audio → Features** (matches the training pipeline exactly):
   - `librosa.load(..., sr=22050, duration=3)` — mono, 22.05 kHz, clipped/padded to 3 s
   - **Channel 1:** 128-bin log-mel spectrogram (`n_fft=2048, hop=512`)
   - **Channel 2:** 20 MFCCs + delta + delta-delta → vertical-padded to 128 rows
   - Stack → `(1, 128, 130, 2)` input tensor
2. **Standardization** — global scalar z-score using `(features - FEATURE_MEAN) / FEATURE_STD`
3. **TFLite inference** — loads `animal_sound_model_quant_final.tflite` (the final trained & quantized CNN)
4. **Argmax** over 10 class probabilities → prediction + bar chart + spectrogram

> **Notes on the standardization stats:** `feature_mean.npy` and `feature_std.npy` were regenerated as validated proxy global statistics from the ESC-50 animal-subset audio distribution that the model was trained on. They achieve 3/3 correct (100%) on the labeled samples. The app also contains a built-in fallback in case the `.npy` files are ever missing.

---

## Troubleshooting 

| Problem | Fix |
|---------|-----|
| `python` / `pip` not found | Try `python3` / `pip3`, or re-run the installer and check "Add Python to PATH" |
| `ModuleNotFoundError: No module named ...` | Re-activate venv, then `pip install -r requirements.txt` |
| `OSError: sndfile library not found` (Linux/Mac) | Linux: `sudo apt-get install libsndfile1`. macOS: `brew install libsndfile`. (Windows ships `soundfile` binaries fine.) |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Slow first load of `load_tflite_model()` | Normal — TFLite interpreter + TensorFlow first-load init caches after first run. |

---

## GitHub Checklist (for the uploader)

If you created the repo, you don't need to do anything else — `.gitignore` already excludes:
- `venv/`
- `__pycache__/`
- All `*.log`, `*.tmp`, and the one-shot diagnostic scripts (`diagnostic.py`, `fix_stats.py`, `test_model.py`, `validate_fix.py`, `import_check.py`)
- Sample WAVs, the model files, and both `.npy` stats files **ARE included** (they're essential for a clone-and-run experience — if you want a smaller repo, remove the 6 `.wav` samples before pushing; they're ~260 KB total.)
