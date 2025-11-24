"""
Output stream helpers.

This module provides a public API to read and save audio files.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def save_audio(
    audio_data: np.ndarray, output_path: str | Path, sample_rate: int = 16000
) -> None:
    """Public function to save audio data as WAV file.

    Args:
        audio_data: Audio data as numpy array
        output_path: Output file path (will be converted to .wav extension)
        sample_rate: Audio sample rate
    """
    output_path = Path(output_path)
    assert output_path.suffix.lower() == ".wav", (
        f'Currently only "wav" format is supported, "{output_path.suffix}" given.'
    )

    # Ensure output directory exists
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(output_path, audio_data, sample_rate, format="WAV")
        print(
            f"Saved audio chunk: ({len(audio_data) / sample_rate:.2f}s) to the file:"
            f"\nfile://{output_path}"
        )

    except Exception as e:
        print(f"Error saving file {output_path}: {e}")


def read_audio(file_path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Read audio data from a file.

    Args:
        file_path: Path to the audio file.
        sample_rate: Desired sample rate.

    Returns:
        Audio data as a 1-D numpy array of dtype float32 (mono).
    """
    # Check if file exists first
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load audio file using librosa (supports MP3, WAV, FLAC, etc.)
    audio_data, effective_sr = librosa.load(
        str(file_path),
        sr=sample_rate,  # Resample to target sample rate
        mono=True,  # Convert to mono
    )

    assert effective_sr == sample_rate, (
        f"Unexpected sample rate after loading: {effective_sr} Hz, expected {sample_rate} Hz."
    )
    duration = len(audio_data) / sample_rate
    print(
        f"Loaded audio file: {file_path.name}, {duration:.2f} seconds, "
        f"{sample_rate} Hz, {len(audio_data)} samples"
    )
    return audio_data.astype(np.float32)
