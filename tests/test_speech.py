"""Tests for the simple speech utilities.

These tests check the basic error behavior when the OPENAI_API_KEY is not set.
"""
from __future__ import annotations

import importlib
import os

import pytest
import aiteacher.generated.speech_api as speech
from aiteacher.generated.speech_api import speech_to_text, text_to_speech
from aiteacher.audio.input_stream import AudioFileInputStream
import time


def test_functions_raise_when_no_api_key(monkeypatch):
    """If OPENAI_API_KEY is not present, functions should raise RuntimeError."""
    # Ensure the environment variable is not present
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Import fresh to ensure dotenv/load happens with the variable missing
    importlib.reload(speech)

    with pytest.raises(RuntimeError):
        speech.speech_to_text(b"\x00\x01")

    with pytest.raises(RuntimeError):
        speech.text_to_speech("hello")


@pytest.mark.slow
def test_speech_real():
    stream = AudioFileInputStream("aiteacher/audio/start_stop.mp3")
    stream.start()
    time.sleep(5)
    data = stream.get_unprocessed_chunk()
    text = speech_to_text(data)
    print(text)
    

if __name__ == "__main__":
    # manual test :(



    import sounddevice as sd
    output_stream = sd.RawOutputStream(
        # samplerate=16000, 
        # blocksize=2048,
        # channels=1, 
        # dtype='float32',
    )
    # Optionally, convert text back to speech
    tts_audio = text_to_speech("Hello world, this is a test!!")
    output_stream.write(tts_audio)

    print(f"Generated TTS audio of length: {len(tts_audio)} bytes")

