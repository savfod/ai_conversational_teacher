#!/usr/bin/env python3
"""Read tests/speech.wav and play it using sounddevice.OutputStream.

This script prefers the third-party `soundfile` library for reading audio
and uses `sounddevice.OutputStream` to send the audio to the default output
device.

Run:
    python tests/read_wav.py

"""
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read an audio file using soundfile.

    Args:
        path: Path to the WAV file.

    Returns:
        A tuple of (data, samplerate). Data is a float32 numpy array with shape
        (frames, channels).
    """
    data, original_samplerate = sf.read(str(path), dtype="float32")
    # Resample to 16kHz if needed
    if original_samplerate != 16000:
        data = librosa.resample(data.T, orig_sr=original_samplerate, target_sr=16000).T
    samplerate = 16000
    # Ensure data is 2-D: (frames, channels)
    if data.ndim == 1:
        data = data[:, None]
    return data, samplerate


def play_via_outputstream(data: np.ndarray, samplerate: int) -> None:
    """Play audio data using sounddevice.OutputStream.

    This uses blocking writes to the stream so the function will return only
    after playback completes.

    Args:
        data: float32 numpy array shaped (frames, channels).
        samplerate: Sample rate in Hz.
    """
    channels = int(data.shape[1])

    
    with sd.OutputStream(samplerate=samplerate, channels=channels, dtype="float32") as stream:
        stream.write(data)

import librosa

from aiteacher.audio.input_stream import MicrophoneInputStream


def main() -> int:
    """Main entry point.

    Looks for `tests/speech.wav` relative to the repository root (this file's
    parent directory).
    """
    mic_stream = MicrophoneInputStream(sample_rate=16000)
    mic_stream.start()
    import time
    time.sleep(1)
    

    # path = Path(__file__).parent / "speech.wav"
    path = Path(__file__).parent.parent / "audio" / "speech_20251030_172228.wav"
    if not path.exists():
        print(f"Audio file not found: {path}", file=sys.stderr)
        return 2

    data, sr = read_wav(path)
    try:
        play_via_outputstream(data, sr)
    except Exception as exc:  # Let errors surface but provide a helpful message
        print(f"Error during playback: {exc!r}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
