"""Audio parser for processing speech chunks and detecting start/stop commands."""

import json
import numpy as np
import vosk
from pathlib import Path
from typing import Tuple, Optional, List, Literal
from enum import Enum

Status = Literal["listening", "waiting"]


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
        vosk.SetLogLevel(0)  # Enable Vosk logging for testing
        
        # Resolve model path
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            # Look for model in the generated directory
            generated_dir = Path(__file__).parent.parent / "generated"
            model_path_obj = generated_dir / model_path_obj
        
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Vosk model not found at: {model_path_obj}")
        
        self.model = vosk.Model(str(model_path_obj))
        self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        self._audio_buffer: List[np.ndarray] = []
        self._vosk_parsed_buffer: List[str] = []  # todo: remove
    
        self._status: Status = "waiting"

    @staticmethod
    def _has_start_seq(text: str) -> bool:
        """Check if the text contains a start command."""
        start_words = ["start", "go", "begin", "that", "startup"]  # not too accurate on 'start'
        return any(word in text.lower() for word in start_words)

    @staticmethod
    def _has_stop_seq(text: str) -> bool:
        """Check if the text contains a stop command."""
        return text.lower().count("stop") > 1

    def add_chunk(self, audio_chunk: np.ndarray) -> Tuple[Status, Optional[np.ndarray]]:
        """Add an audio chunk and process it for start/stop detection.
        
        Args:
            audio_chunk: Audio data as numpy array (float32, mono).
            
        Returns:
            Tuple of (status, optional_audio):
            - status: "listening" if recording speech, "waiting" if waiting for start command
            - optional_audio: Complete speech interval if stop was just detected, None otherwise
        """
        self._audio_buffer.append(audio_chunk.copy())
        detected_text = self._add_vosk_chunk(audio_chunk.copy())
        
        if self._status == "waiting" and self._has_start_seq(detected_text):    
            print("START command detected - now listening for speech")
            self._status = "listening"
            self._audio_buffer = []  # Clear any previous buffer
            self._reset_vosk()
        
        elif self._status == "listening" and self._has_stop_seq(detected_text):
            print("STOP command detected - processing speech interval")
            self._status = "waiting"
        
            if self._audio_buffer:
                complete_audio = np.concatenate(self._audio_buffer)
            else:
                complete_audio = np.array([], dtype=np.float32)
                print("Warning: no audio buffered between start and stop")
            
            print(f"Returning speech interval: {len(complete_audio) / self.sample_rate:.2f} seconds")
            self._reset_vosk()
            return self._status, complete_audio
        
        return self._status, None

    @staticmethod
    def _preprocess_vosk_chunk(audio_chunk: np.ndarray) -> bytes:
        """Convert audio chunk to 16-bit PCM bytes for Vosk."""
        assert audio_chunk.dtype == np.float32

        # Ensure mono audio
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk[:, 0]

        int16_chunk = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        return int16_chunk.tobytes()

    def _add_vosk_chunk(self, audio_chunk: np.ndarray) -> str:
        """Process new audio chunk through Vosk and return recognized text. 
        
        Args:
            audio_data: Raw audio data as bytes (16-bit PCM).
            
        Returns:
            Recognized text if smth was detected, None otherwise.
        """
        # vosk processes new chunk, saving partial results, and removing buffer with saving full results
        # todo: switch to manual logic with recognize(), not add_chunk()
        audio_data = self._preprocess_vosk_chunk(audio_chunk)

        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            text = result.get('text', '').strip()
            print(f"[VOSK] Full result: {result}")  # Print full Vosk result for testing
            self._vosk_parsed_buffer.append(text)
            return text
        else:
            # Also check for partial results during testing
            partial_result = json.loads(self.recognizer.PartialResult())
            partial_text = partial_result.get('partial', '').strip()
            if partial_text:
                print(f"[VOSK] Partial: {partial_text}")
            cur_text = " " + partial_text
            return cur_text

        # It may be reasonable or not to combine with previously calculated buffer.
        # text = " ".join(self._vosk_parsed_buffer) + cur_text
        # print(f"[VOSK] Text part:", text)
    
    
    
    def _reset_vosk(self) -> None:
        """Reset Vosk recognizer to clear any partial recognition state."""
        self._vosk_parsed_buffer = []
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
    
    @property
    def status(self) -> Status:
        """Get current parser status."""
        return self._status
    
    @property
    def buffered_duration(self) -> float:
        """Get duration of currently buffered audio in seconds."""
        if not self._audio_buffer:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        return total_samples / self.sample_rate
    
    def reset(self) -> None:
        """Reset the parser to initial state."""
        print("Resetting audio parser state")
        self._reset_state()