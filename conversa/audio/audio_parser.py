"""Audio parser for processing speech chunks and detecting start/stop commands."""

import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import vosk

State = Literal["listening", "waiting"]


class AudioParser:
    """Processes audio chunks to detect start/stop commands and extract speech intervals.

    This class maintains an internal buffer of audio chunks and uses Vosk speech recognition
    to detect "start" and "stop" commands. When a complete speech interval is detected
    (from start to stop), it returns the accumulated audio data.
    """

    def __init__(self, model_path: str, sample_rate: int = 16000) -> None:
        """Initialize the audio parser.

        Args:
            model_path: Path to the Vosk model directory.
            sample_rate: Audio sample rate in Hz.
        """
        # Initialize Vosk
        vosk.SetLogLevel(-1)  # Disable Vosk logging for testing

        # Resolve model path
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            # Look for model in the model directory
            generated_dir = Path(__file__).parent.parent.parent / "models"
            model_path_obj = generated_dir / model_path_obj

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Vosk model not found at: {model_path_obj}")

        self.model = vosk.Model(str(model_path_obj))
        self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        self._audio_buffer: List[np.ndarray] = []
        self._vosk_parsed_buffer: List[str] = []  # todo: remove

        self._state: State = "waiting"

    @staticmethod
    def _has_start_seq(text: str) -> bool:
        """Check if the text contains a start command.

        Args:
            text: Recognized text to inspect.

        Returns:
            True if a start-like token is present, False otherwise.
        """
        # "start" and words that "start" is sometimes misrecognized as
        start_words = [
            "start",
            "go",
            "begin",
            "that",
            "startup",
        ]  # not too accurate on 'start'
        return any(word in text.lower() for word in start_words)

    @staticmethod
    def _has_stop_seq(text: str) -> bool:
        """Check if the text contains a stopping command.

        Args:
            text: Recognized text to inspect.

        Returns:
            True if a stop-like token pattern is present, False otherwise.
        """
        # stop command is "stop stop" to avoid false positives
        return text.lower().count("stop") > 1

    def add_chunk(
        self, audio_chunk: np.ndarray
    ) -> Tuple[State, Optional[np.ndarray], bool]:
        """Add an audio chunk and process it for start/stop detection.

        Args:
            audio_chunk: Audio data as numpy array (float32, mono).

        Returns:
            Tuple of (status, optional_audio, status_changed):
            - status: "listening" if recording speech, "waiting" if waiting for start command
            - optional_audio: Complete speech interval if stop was just detected, None otherwise
            - status_changed: True if the parser status changed during this call
        """
        optional_audio: Optional[np.ndarray] = None
        status_changed = False

        self._audio_buffer.append(audio_chunk.copy())
        detected_text = self._add_vosk_chunk(audio_chunk.copy())

        # Detect start command
        if self._state == "waiting" and self._has_start_seq(detected_text):
            print("START command detected - now listening for speech")
            status_changed = True
            self.reset(to_state="listening")

        # Detect stop command
        elif self._state == "listening" and self._has_stop_seq(detected_text):
            print("\nSTOP command detected - processing speech interval")
            if self._audio_buffer:
                optional_audio = np.concatenate(self._audio_buffer)
            else:
                optional_audio = np.array([], dtype=np.float32)
                print("Warning: no audio buffered between start and stop")

            print(
                f"Returning speech interval: {len(optional_audio) / self.sample_rate:.2f} seconds"
            )
            status_changed = True
            self.reset(to_state="waiting")

        return self._state, optional_audio, status_changed

    @staticmethod
    def _preprocess_vosk_chunk(audio_chunk: np.ndarray) -> bytes:
        """Convert audio chunk to 16-bit PCM bytes for Vosk."""
        assert audio_chunk.dtype == np.float32

        # Ensure mono audio
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk[:, 0]

        int16_chunk = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        return int16_chunk.astype("<i2", copy=False).tobytes()  # little-endian

    def _add_vosk_chunk(self, audio_chunk: np.ndarray) -> str:
        """Process a new audio chunk through Vosk and return recognized text.

        Args:
            audio_chunk: Audio samples as a numpy array (float32, mono or multi-channel).

        Returns:
            Recognized text (possibly empty string) produced by Vosk for the
            supplied chunk. Partial results are returned as a short string
            starting with a space; full results are returned as the text.
        """
        # vosk processes new chunk, saving partial results, and removing buffer with saving full results
        # todo: switch to manual logic with recognize(), current logic adds it on the vosk side
        audio_data = self._preprocess_vosk_chunk(audio_chunk)

        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
            # print(f"[VOSK] Full result: {result}")  # Print full Vosk result for testing
            self._vosk_parsed_buffer.append(text)
            return text
        else:
            # Also check for partial results during testing
            partial_result = json.loads(self.recognizer.PartialResult())
            partial_text = partial_result.get("partial", "").strip()
            # if partial_text:
            #     print(f"[VOSK] Partial: {partial_text}")
            cur_text = " " + partial_text
            return cur_text

        # It may be reasonable or not to combine with previously calculated buffer.
        # text = " ".join(self._vosk_parsed_buffer) + cur_text
        # print(f"[VOSK] Text part:", text)

    @property
    def status(self) -> State:
        """Get current parser status."""
        return self._state

    @property
    def buffered_duration(self) -> float:
        """Get duration of currently buffered audio in seconds."""
        if not self._audio_buffer:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        return total_samples / self.sample_rate

    def reset(self, to_state: State = "waiting") -> None:
        """Delete all buffered audio and recognized text."""
        self._audio_buffer = []
        self._state = to_state

        # Reset Vosk recognizer to clear any partial recognition state.
        # todo: try to avoid with direct vosk.recognize() interface if possible
        self._vosk_parsed_buffer = []
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
