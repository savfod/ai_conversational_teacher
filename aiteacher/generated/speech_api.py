"""Simple speech utilities using OpenAI for STT and TTS.

This module provides two convenience functions:
- speech_to_text: transcribe audio bytes to text using an OpenAI model
- text_to_speech: synthesize speech bytes from text using an OpenAI model

The OpenAI API key is loaded via dotenv from the environment variable
`OPENAI_API_KEY`.

Type hints and Google-style docstrings are used.
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

import librosa
import openai
from io import BytesIO
import wave
import struct
import soundfile as sf

# Load environment variables from a .env file if present
load_dotenv()


def _get_api_key() -> Optional[str]:
    """Return the OpenAI API key from the environment, if available.

    Returns:
        The API key string or None if not set.
    """
    return os.getenv("OPENAI_API_KEY")


def _ensure_openai_available() -> None:
    """Ensure the openai package and API key are available.

    Raises:
        RuntimeError: If the openai package is not installed or API key is missing.
    """
    # Verify import succeeded and that an API key exists in the env.
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")


def _get_client():
    """Create and return an OpenAI v1 client instance.

    This function expects the installed `openai` package to expose an
    `OpenAI` client class (openai.OpenAI) as introduced in openai>=1.0.0.

    Raises:
        RuntimeError: If the OpenAI client class is not present.
    """
    api_key = _get_api_key()
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception as exc:
        # Provide a clear error message guiding the user to install a
        # compatible openai package or pin to the older version.
        raise RuntimeError(
            "Unable to construct OpenAI client. Ensure openai>=1.0.0 is installed or pin to an older SDK."
        ) from exc


def speech_to_text(audio_bytes: bytes, language: Optional[str] = "en", sample_rate: int = 16000, *, model: str = "gpt-4o-mini-transcribe") -> str:
    """Transcribe audio chunk to text using an OpenAI model.

    Args:
        audio_bytes: Raw audio bytes (e.g., WAV, MP3, or supported format).
        language: Optional BCP-47 language code hint (e.g., "en", "es").
        model: Model name to use for transcription. Default chosen to a recent
            speech-to-text capable model. Change if required.

    Returns:
        The transcribed text.

    Raises:
        RuntimeError: If the OpenAI package or API key is missing, or if
            the API returns an error.
    """
    _ensure_openai_available()

    try:

        # Normalize different input types into a WAV bytes buffer that the
        # OpenAI API understands. Accept either raw audio bytes (already a
        # file like WAV/MP3) or a numeric array (e.g., numpy.ndarray of
        # float32 samples).
        client = _get_client()

        wav_buffer: BytesIO
        if isinstance(audio_bytes, (bytes, bytearray)):
            # Assume already a file-like audio container (mp3/wav) and pass
            # through directly. Set a filename on the buffer so the client
            # sends a sensible Content-Disposition with an extension the
            # server can use to detect format.
            wav_buffer = BytesIO(audio_bytes)
            head = bytes(audio_bytes[:12])
            # simple magic checks
            if head.startswith(b"RIFF"):
                wav_buffer.name = "audio.wav"
            elif head.startswith(b"ID3") or head[:2] == b"\xff\xfb":
                wav_buffer.name = "audio.mp3"
            else:
                # default to wav
                wav_buffer.name = "audio.wav"
        else:
            # Try to convert a numeric array (numpy) to 16-bit PCM WAV.
            try:
                import numpy as _np

                arr = _np.asarray(audio_bytes)
            except Exception:
                raise RuntimeError("Unsupported audio input type - provide raw bytes or a numpy array of samples")

            # Convert to mono if multi-channel by taking first channel
            if arr.ndim > 1:
                arr = arr[:, 0]

            # If floats, assume in range [-1, 1]. If integers, scale to int16.
            if _np.issubdtype(arr.dtype, _np.floating):
                clipped = _np.clip(arr, -1.0, 1.0)
                int_samples = (clipped * 32767.0).astype(_np.int16)
            else:
                # Cast to int16, but beware of range
                int_samples = arr.astype(_np.int16)

            wav_buffer = BytesIO()
            with wave.open(wav_buffer, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                # wave expects bytes in little-endian signed 16-bit
                w.writeframes(int_samples.tobytes())

            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

        # Use the v1 client audio transcription endpoint
        params = {"file": wav_buffer, "model": model}
        if language:
            params["language"] = language

        resp = client.audio.transcriptions.create(**params)

        # Prefer attribute access which is the usual shape for the v1 SDK
        try:
            return resp.text
        except Exception:
            # Fallback to dict-like access for compatibility
            if isinstance(resp, dict):
                return resp.get("text", "")
            raise
    except Exception as exc:
        raise RuntimeError(f"speech_to_text failed: {exc}") from exc


def text_to_speech(text: str, *, model: str = "gpt-4o-mini-tts", voice: Optional[str] = None) -> bytes:
    """Synthesize speech bytes from text using an OpenAI model.

    Args:
        text: The input text to synthesize.
        model: The TTS model name to use. Default set to a recent TTS-capable model.
        voice: Optional voice selection string, if supported by the model.

    Returns:
        Raw audio bytes (e.g., WAV or MP3) as returned by the API.

    Raises:
        RuntimeError: If the OpenAI package or API key is missing, or if
            the API returns an error.
    """
    _ensure_openai_available()
    if voice is None: 
        voice = "coral"  # default voice
    
    client = _get_client()
    from pathlib import Path
    from datetime import datetime
    speech_file_path = Path(__file__).parent.parent.parent / "audio" / f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav",
        instructions="Speak in a cheerful and positive tone.",
    ) as response:
        response.stream_to_file(speech_file_path)
    

    # Read the generated audio file and return as bytes
    data, samplerate = sf.read(str(speech_file_path), dtype="float32")
    # Ensure data is 1-D: (frames, channels)
    if data.ndim == 2:
        data = data[:, 0]

    if samplerate != 16000:
        data = librosa.resample(data.T, orig_sr=samplerate, target_sr=16000).T
    

    # Clean up the temporary file
    # speech_file_path.unlink()
    
    return data

    # try:
    #     client = _get_client()

    #     # Use the v1 client TTS endpoint
    #     params = {"model": model, "input": text}
    #     if voice:
    #         params["voice"] = voice

    #     resp = client.audio.speech.create(**params)

    #     # If the SDK returns raw bytes directly
    #     if isinstance(resp, (bytes, bytearray)):
    #         return bytes(resp)

    #     # Prefer attribute access
    #     try:
    #         return bytes(resp.audio)
    #     except Exception:
    #         # Fallback to dict-like access
    #         if isinstance(resp, dict):
    #             for key in ("audio", "audio_content", "audio_base64"):
    #                 if key in resp:
    #                     val = resp[key]
    #                     if isinstance(val, str):
    #                         import base64

    #                         return base64.b64decode(val)
    #                     return bytes(val)
    #         raise
    # except Exception as exc:
    #     raise RuntimeError(f"text_to_speech failed: {exc}") from exc
