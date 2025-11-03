"""Voice command detection for start/stop functionality using Vosk."""

import json
import queue
from typing import Dict, Optional

import numpy as np
import sounddevice as sd
import vosk


class VoiceCommandDetector:
    """Detects voice commands for starting and stopping conversation sessions.
    
    This class uses the Vosk speech recognition library to listen for voice commands
    and detect when the user says "start" or "stop" to control conversation sessions.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the voice command detector.
        
        Args:
            model_name: Path to the Vosk model directory.
        """
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(model_name)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.commands: Dict[str, bool] = {
            "start": False,
            "end": False
        }

    def detect_commands(self, audio_chunk: bytes) -> Optional[str]:
        """Detect voice commands in an audio chunk.
        
        Args:
            audio_chunk: Raw audio data as bytes.
            
        Returns:
            Detected text if speech was recognized, None otherwise.
        """
        if self.recognizer.AcceptWaveform(audio_chunk):
            result = json.loads(self.recognizer.Result())
            text = result.get('text', '').strip().lower()
            
            if "start" in text:
                self.commands["start"] = True
            if "stop" in text:
                self.commands["end"] = True
            
            return text
        return None

    def listen_continuously(self) -> None:
        """Start continuous listening for voice commands.
        
        This method starts an audio stream and continuously listens for voice commands.
        When "start" or "stop" commands are detected, it prints a message and resets
        the command flag.
        
        Note:
            This method runs indefinitely. Use Ctrl+C to stop.
        """
        def audio_callback(indata: np.ndarray, frames: int, time: sd.CallbackFlags, status: sd.CallbackFlags) -> None:
            """Callback function for processing audio input.
            
            Args:
                indata: Input audio data as numpy array.
                frames: Number of frames in the input.
                time: Timing information.
                status: Status information about the stream.
            """
            if status:
                print(status)
            
            # Convert to mono if stereo
            if indata.ndim > 1:
                indata = indata[:, 0]
            
            # Convert to 16-bit PCM
            audio_data = (indata * 32767).astype(np.int16).tobytes()
            
            # Detect commands
            detected_text = self.detect_commands(audio_data)
            if detected_text:
                print(f"Detected: {detected_text}")

        # Start listening
        with sd.InputStream(callback=audio_callback, 
                            channels=1, 
                            samplerate=16000, 
                            dtype='float32'):
            while True:
                if self.commands["start"]:
                    print("Start command detected!")
                    # Your start logic here
                    self.commands["start"] = False
                
                if self.commands["end"]:
                    print("End command detected!")
                    # Your end logic here
                    self.commands["end"] = False


if __name__ == "__main__":
    detector = VoiceCommandDetector("vosk-model-small-en-us-0.15")
    detector.listen_continuously()