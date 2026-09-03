"""
Live mic -> WildEcho AI backend, running standalone on the Raspberry Pi.

This does NOT reimplement YAMNet or the classifier head — it just records
short clips from your USB mic (Ultraprolink, seen as a normal ALSA/PortAudio
device) and POSTs them to your existing server.py's /api/classify endpoint,
exactly like the browser's "Live Mic" section does. This lets you run
detection headlessly on the Pi with no browser needed.

Prerequisites:
    1. server.py running (on this same Pi, or reachable over the network):
           python3 server.py
       (classifier_head1.tflite and label_map.json must sit next to it, and
        the first run needs internet once to download YAMNet from TF-Hub.)
    2. pip install sounddevice numpy requests scipy

Usage:
    python3 live_mic_client.py                       # loop forever, local server
    python3 live_mic_client.py --api http://192.168.1.50:8000
    python3 live_mic_client.py --device 1             # force a specific mic index
    python3 live_mic_client.py --once                 # record & classify one clip, then exit
"""

import argparse
import io
import time
import wave

import numpy as np
import requests
import sounddevice as sd
from scipy.signal import resample_poly


CAPTURE_SR = 44100
MODEL_SR = 16000
CAPTURE_CHANNELS = 2
CLIP_SECONDS = 3.0
DEFAULT_DEVICE = 1
SILENCE_RMS_THRESHOLD = 0.01


def record_clip(duration, device=None):
    """Record from the UM1002 at its native format: 44.1 kHz stereo."""
    audio = sd.rec(
        int(duration * CAPTURE_SR),
        samplerate=CAPTURE_SR,
        channels=CAPTURE_CHANNELS,
        dtype="int16",
        device=device,
    )
    sd.wait()

    # Stereo -> mono
    audio = audio.astype(np.float32)
    audio = np.mean(audio, axis=1)

    # Normalize to [-1, 1]
    audio /= 32768.0

    # 44.1 kHz -> 16 kHz
    audio = resample_poly(audio, MODEL_SR, CAPTURE_SR)

    # Convert back to int16
    audio = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

    return audio


def to_wav_bytes(samples_int16, sr):
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int16.tobytes())

    buf.seek(0)
    return buf


def classify_clip(
    api_base,
    wav_buf,
    use_voting=True,
    voting_step=0.5,
    timeout=30,
):
    files = {
        "file": ("clip.wav", wav_buf, "audio/wav")
    }

    data = {
        "use_voting": str(use_voting),
        "voting_step": str(voting_step),
    }

    resp = requests.post(
        f"{api_base}/api/classify",
        files=files,
        data=data,
        timeout=timeout,
    )

    resp.raise_for_status()
    return resp.json()


def is_silent(samples_int16):
    x = samples_int16.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x ** 2))) < SILENCE_RMS_THRESHOLD


def check_health(api_base):
    resp = requests.get(
        f"{api_base}/api/health",
        timeout=10,
    )

    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000",
        help="Base URL of server.py",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_DEVICE,
        help="sounddevice input device index",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=CLIP_SECONDS,
        help="Clip length per capture",
    )

    parser.add_argument(
        "--voting-step",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--no-voting",
        action="store_true",
        help="Classify only the first 1s window",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Record & classify a single clip, then exit",
    )

    args = parser.parse_args()

    print(f"Checking backend at {args.api} ...")

    try:
        health = check_health(args.api)

        print(
            f"Backend online. Classes: {health['classes']}"
        )

    except Exception as e:
        print(
            f"Could not reach backend at {args.api}: {e}"
        )

        print(
            "Make sure `python3 server.py` is running."
        )

        return

    print(
        f"Listening on device={args.device} "
        f"| {args.seconds}s clips "
        f"| capture={CAPTURE_SR}Hz stereo "
        f"| model={MODEL_SR}Hz mono "
        f"| Ctrl+C to stop"
    )

    try:
        while True:

            samples = record_clip(
                args.seconds,
                device=args.device,
            )

            if is_silent(samples):

                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    "... quiet, skipping"
                )

            else:

                wav_buf = to_wav_bytes(
                    samples,
                    MODEL_SR,
                )

                try:

                    result = classify_clip(
                        args.api,
                        wav_buf,
                        use_voting=not args.no_voting,
                        voting_step=args.voting_step,
                    )

                    top = result["top_prediction"]

                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{top['emoji']} "
                        f"{top['label']} "
                        f"({top['percentage']}%) "
                        f"| exec "
                        f"{result['execution_time_ms']}ms "
                        f"| "
                        f"{result['signal_stats']['num_windows_voted']} "
                        f"windows voted"
                    )

                    if result.get("warning"):
                        print(
                            f"    warning: "
                            f"{result['warning']}"
                        )

                except requests.RequestException as e:

                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"request failed: {e}"
                    )

            if args.once:
                break

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
