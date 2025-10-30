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
        
        # State management
        self._status: Status = "waiting"
        self._audio_buffer: List[np.ndarray] = []
        self._start_detected = False
        self._stop_detected = False
    
    @staticmethod
    def _start_seq(text: str) -> bool:
        """Check if the text contains a start command."""
        start_words = ["start", "go", "begin", "that", "startup"]  # not too accurate on 'start'
        return any(word in text.lower() for word in start_words)

    @staticmethod
    def _stop_seq(text: str) -> bool:
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
        # Convert audio chunk to format expected by Vosk (16-bit PCM bytes)
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Ensure mono audio
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk[:, 0]
        
        # Convert to 16-bit PCM bytes for Vosk
        # self._audio_buffer.append((np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16))
        # normalize
        audio_chunk = audio_chunk
        self._audio_buffer.append((np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16).copy())

        
        # joined_audio = np.concatenate(self._audio_buffer)
        joined_audio = self._audio_buffer[-1]  # vosk has that already :/

        # print(f"Processing audio chunk of {len(joined_audio) / self.sample_rate:.3f} seconds")
        # looking max 2 seconds of audio for commands
        joined_audio = joined_audio[-2 * self.sample_rate:]

        # Process audio through Vosk recognizer
        detected_text = self._process_vosk_audio(joined_audio.tobytes())
        
        # Update state based on detected commands
        if detected_text:
            print(f"Detected speech: '{detected_text}'")
            

            start_words = ["start", "go", "begin", "startup", "stuff"]
            if any(word in detected_text.lower() for word in start_words) and self._status == "waiting":
                self._start_detected = True
                self._status = "listening"
                self._audio_buffer = []  # Clear any previous buffer
                print("START command detected - now listening for speech")
                
            elif "stop" in detected_text.lower() and self._status == "listening":
                self._stop_detected = True
                print("STOP command detected - processing speech interval")
        
        # Handle audio buffering based on status
        if self._status == "listening":
            # Store the original float32 audio data
            self._audio_buffer.append(audio_chunk.copy())
            
            # Check if we just detected stop
            if self._stop_detected:
                # Concatenate all buffered audio and return it
                if self._audio_buffer:
                    complete_audio = np.concatenate(self._audio_buffer)
                    print(f"Returning speech interval: {len(complete_audio) / self.sample_rate:.2f} seconds")
                else:
                    complete_audio = None
                    print("No audio buffered between start and stop")
                
                # Reset state
                self._reset_state()
                return "waiting", complete_audio
        
        return self._status, None
    
    def _process_vosk_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio data through Vosk and return recognized text.
        
        Args:
            audio_data: Raw audio data as bytes (16-bit PCM).
            
        Returns:
            Recognized text if speech was detected, None otherwise.
        """
        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            text = result.get('text', '').strip()
            print(f"[VOSK] Full result: {result}")  # Print full Vosk result for testing
            return text if text else None
        else:
            # Also check for partial results during testing
            partial_result = json.loads(self.recognizer.PartialResult())
            partial_text = partial_result.get('partial', '').strip()
            if partial_text:
                print(f"[VOSK] Partial: {partial_text}")
            return partial_text if partial_text else None
    
    def _reset_state(self) -> None:
        """Reset the parser state after processing a complete speech interval."""
        self._status = "waiting"
        self._audio_buffer = []
        self._start_detected = False
        self._stop_detected = False
        
        # Reset Vosk recognizer to clear any partial recognition state
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