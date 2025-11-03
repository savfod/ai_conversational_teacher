"""
Speech Interface Stub Implementation

This module provides a threaded speech interface system that can handle both
microphone input and MP3 file processing without blocking the main thread.
"""

import abc
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioBuffer:
    """Thread-safe audio buffer with automatic size management."""
    
    def __init__(self, max_duration_seconds: float = 30.0, sample_rate: int = 16000):
        """
        Initialize audio buffer.
        
        Args:
            max_duration_seconds: Maximum buffer duration before trimming
            sample_rate: Audio sample rate
        """
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self._buffer = []
        self._lock = threading.Lock()
        
    def add_audio(self, audio_data: np.ndarray) -> None:
        """Add audio data to buffer with automatic trimming."""
        with self._lock:
            self._buffer.extend(audio_data.flatten())
            
            # Trim buffer if it exceeds maximum size
            if len(self._buffer) > self.max_samples:
                trim_amount = int(len(self._buffer) * 0.2)  # Remove 20%
                self._buffer = self._buffer[trim_amount:]
                print(f"Warning: Audio buffer trimmed by {trim_amount} samples "
                      f"({trim_amount / self.sample_rate:.2f} seconds)")
    
    def get_and_clear(self) -> np.ndarray:
        """Get all buffered audio and clear the buffer."""
        with self._lock:
            if not self._buffer:
                return np.array([])
            
            audio_data = np.array(self._buffer, dtype=np.float32)
            self._buffer.clear()
            return audio_data
    
    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self._lock:
            return len(self._buffer) / self.sample_rate if self._buffer else 0.0


class AbstractAudioInputStream(abc.ABC):
    """Abstract base class for audio input streams."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio input stream.
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._buffer = AudioBuffer(sample_rate=sample_rate)
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
    
    @abc.abstractmethod
    def _audio_processing_loop(self) -> None:
        """Internal audio processing loop to be implemented by subclasses."""
        pass
    
    def start(self) -> None:
        """Start the audio input stream."""
        if self._is_running:
            return
            
        self._is_running = True
        self._thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self._thread.start()
        print(f"Started {self.__class__.__name__}")
    
    def stop(self) -> None:
        """Stop the audio input stream."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        print(f"Stopped {self.__class__.__name__}")
    
    def get_unprocessed_chunk(self) -> Optional[np.ndarray]:
        """
        Get currently buffered audio data.
        
        Returns:
            Audio chunk as numpy array, or None if no data available
        """
        chunk = self._buffer.get_and_clear()
        return chunk if len(chunk) > 0 else None
    
    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return self._buffer.get_duration()


class MicrophoneInputStream(AbstractAudioInputStream):
    """Microphone input stream implementation using sounddevice."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 block_size: int = 1024, device: Optional[int] = None):
        """
        Initialize microphone input stream.
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            block_size: Audio block size for processing
            device: Specific audio device ID (None for default)
        """
        super().__init__(sample_rate, channels)
        self.block_size = block_size
        self.device = device
        self._stream: Optional[sd.InputStream] = None
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                       time_info, status) -> None:
        """Callback function for sounddevice stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add audio data to buffer
        self._buffer.add_audio(indata)
    
    def _audio_processing_loop(self) -> None:
        """Audio processing loop for microphone input."""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                device=self.device,
                callback=self._audio_callback,
                dtype=np.float32
            ) as self._stream:
                print(f"Microphone stream started (device: {self.device}, "
                      f"rate: {self.sample_rate}, channels: {self.channels})")
                
                while self._is_running:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting
                    
        except Exception as e:
            print(f"Error in microphone processing: {e}")
            self._is_running = False


class AudioFileInputStream(AbstractAudioInputStream):
    """Audio file input stream that simulates real-time processing."""
    
    def __init__(self, file_path: str, sample_rate: int = 16000, 
                 channels: int = 1, chunk_duration: float = 0.1):
        """
        Initialize audio file input stream.
        
        Args:
            file_path: Path to audio file (MP3, WAV, FLAC, etc.)
            sample_rate: Target sample rate
            channels: Target number of channels
            chunk_duration: Duration of each chunk in seconds
        """
        super().__init__(sample_rate, channels)
        self.file_path = Path(file_path)
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Note: File existence will be checked in _load_and_convert_mp3()
        # This allows for fallback behavior if file doesn't exist
    
    def _load_and_convert_mp3(self) -> np.ndarray:
        """
        Load and convert MP3 file to numpy array using librosa.
        
        Returns:
            Audio data as numpy array
        """
        # Check if file exists first
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")
        
        # Load audio file using librosa (supports MP3, WAV, FLAC, etc.)
        audio_data, original_sr = librosa.load(
            str(self.file_path), 
            sr=self.sample_rate,  # Resample to target sample rate
            mono=True  # Convert to mono
        )
        
        duration = len(audio_data) / self.sample_rate
        print(f"Loaded audio file: {self.file_path.name}, {duration:.2f} seconds, "
                f"{self.sample_rate} Hz, {len(audio_data)} samples")
        
        return audio_data.astype(np.float32)
                
    def _audio_processing_loop(self) -> None:
        """Audio processing loop for MP3 file input."""
        try:
            # Load the entire audio file
            audio_data = self._load_and_convert_mp3()
            total_samples = len(audio_data)
            current_position = 0
            
            print(f"Starting MP3 playback simulation: {total_samples / self.sample_rate:.2f} seconds")
            
            while self._is_running and current_position < total_samples:
                # Calculate chunk end position
                chunk_end = min(current_position + self.chunk_size, total_samples)
                
                # Extract chunk
                chunk = audio_data[current_position:chunk_end]
                
                # Add chunk to buffer
                if len(chunk) > 0:
                    self._buffer.add_audio(chunk)
                
                current_position = chunk_end
                
                # Sleep to simulate real-time playback
                time.sleep(self.chunk_duration)
            
            print("MP3 file processing completed")
            
        except Exception as e:
            print(f"Error in MP3 processing: {e}")
            self._is_running = False


def _simulate_wait() -> None:
    start = datetime.now()
    print(f"[{start}] Processing audio chunk...")
    # simulate processing delay 
    while datetime.now() - start < timedelta(seconds=3):
        a = np.zeros([3000,3000]) @ np.zeros([3000,3000])
    time.sleep(1)


def _stub_wav_processor(input_stream: AbstractAudioInputStream, output_dir: str = "out") -> None:
    """
    Stub WAV processor that continuously processes audio chunks.
    
    Args:
        input_stream: Audio input stream instance
        output_dir: Output directory for saved chunks
    """
    print("Starting stub WAV processor...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    i = 0
    def _save_chunk(chunk: np.ndarray | None) -> None:
        nonlocal i
        if chunk is not None and len(chunk) > 0:
            i += 1
            output_path = f"out/chunk_{i:04d}.wav"
            
            # save chunk
            sf.write(output_path, chunk, input_stream.sample_rate, format='WAV')
            print(f"Saved audio chunk: {output_path} ({len(chunk) / input_stream.sample_rate:.2f}s)")

            duration = len(chunk) / input_stream.sample_rate
            buffer_duration = input_stream.get_buffer_duration()
            print(f"Processed chunk {i}: {duration:.2f}s, "
                f"buffer: {buffer_duration:.2f}s")
        
        else:
            pass # no chunk to save

    try:
        while True:
            _simulate_wait()
            _save_chunk(input_stream.get_unprocessed_chunk())

    except KeyboardInterrupt:
        _save_chunk(input_stream.get_unprocessed_chunk())
        print("\nStopping WAV processor...")
    
    finally:
        input_stream.stop()


def _test_online_speech_saver():
    """Main function demonstrating the speech interface usage."""
    print("Speech Interface Stub Demo")
    print("=" * 40)
    
    USE_MICROPHONE = False  # Set to True to use microphone, False to use audio file
    
    if USE_MICROPHONE:
        # Example 1: Microphone input
        print("\n1. Testing Microphone Input")
        stream = MicrophoneInputStream()
        stream.start()

    else:
        # Example 2: Audio file input (supports MP3, WAV, FLAC, etc.)
        print("\n2. Testing Audio File Input")
        # Replace with path to your actual audio file
        audio_file_path = "audio.mp3"  # Put your audio file here (MP3, WAV, FLAC, etc.)
        print(f"Using audio file: {audio_file_path}")
        
        stream = AudioFileInputStream(audio_file_path)
        stream.start()
    
    _stub_wav_processor(stream, output_dir="out")

if __name__ == "__main__":    
    _test_online_speech_saver()