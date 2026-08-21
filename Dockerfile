# ============================================================
# WildEcho AI — single-container image for Hugging Face Spaces
#
# Stage 1 builds the React/Vite frontend into static files.
# Stage 2 installs the Python backend and serves BOTH the API and those static
# files from one FastAPI process on port 7860 (the port Hugging Face expects
# for Docker SDK spaces).
#
# If the build ever fails resolving a wheel, the Python minor version below is
# the first thing to adjust (3.12 has the widest wheel coverage for these pins).
#
# Verified against PyPI for linux/cp312: all 67 resolved packages have binary
# wheels. numpy and tensorflow ship manylinux_2_27/2_28 wheels, i.e. they need
# glibc >= 2.28 — bookworm (Debian 12) provides 2.36, so the base image below
# must stay on bookworm or newer.
# ============================================================

# ---------- Stage 1: build the React frontend ----------
# Node 22 (LTS): Vite 8 requires "^20.19.0 || >=22.12.0". The node:20 tag only
# satisfies that on its very latest patches, so 22 is the safer branch.
FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /build

# Install deps first so this layer stays cached when only source changes.
# Not --omit=dev: vite, tsc and tailwind are devDependencies and the build needs them.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build
# -> /build/dist


# ---------- Stage 2: Python backend + compiled frontend ----------
FROM python:3.12-slim-bookworm

# libsndfile: soundfile/librosa decoding.
# ffmpeg: fallback decoder for mp3/ogg/webm uploads libsndfile cannot read.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps as root, system-wide. Doing this before switching users
# avoids depending on pip's "site-packages not writable" fallback behaviour.
COPY requirements-deploy.txt /tmp/requirements-deploy.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-deploy.txt \
    && rm /tmp/requirements-deploy.txt

# Hugging Face Spaces runs the container as UID 1000. Creating that user and
# owning $HOME means the TF-Hub cache is writable at both build and run time.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TFHUB_CACHE_DIR=/home/user/.cache/tfhub \
    PORT=7860

USER user
WORKDIR /home/user/app

# Bake YAMNet (~15 MB) into the image so the Space boots fast and does not
# depend on TF-Hub being reachable at runtime.
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1'); print('YAMNet cached OK')"

# Backend + model artifacts
COPY --chown=user:user server.py ./
COPY --chown=user:user classifier_head1.tflite label_map.json ./

# Bundled demo clips served by /api/samples
COPY --chown=user:user *.wav ./

# Compiled frontend from stage 1 — server.py mounts this directory at "/"
COPY --chown=user:user --from=frontend-builder /build/dist ./frontend/dist

EXPOSE 7860

CMD ["python", "server.py"]
