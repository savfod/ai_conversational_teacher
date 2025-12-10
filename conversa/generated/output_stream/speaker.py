"""
Speaker output stream implementation for audio playback.
"""

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from conversa.generated.output_stream.base import AbstractAudioOutputStream


class SpeakerOutputStream(AbstractAudioOutputStream):
    """Speaker output stream implementation using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
    ):
        """
        Initialize speaker output stream.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            device: Specific audio device ID (None for default)
        """
        super().__init__(sample_rate, channels)
        self.device = device
        self._playback_queue: queue.Queue = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

    def _playback_loop(self) -> None:
        """Playback loop that processes audio chunks from the queue."""
        while not self._stop_event.is_set():
            try:
                # Wait for audio chunk with timeout to check stop event
                audio_chunk = self._playback_queue.get(timeout=0.1)

                if audio_chunk is None:  # Sentinel value to stop
                    break

                # Play audio using sd.play()
                sd.play(
                    audio_chunk,
                    samplerate=self.sample_rate,
                    device=self.device,
                )
                sd.wait()  # Wait for playback to finish

                self._playback_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in playback loop: {e}")
                self._playback_queue.task_done()

    def play_chunk(self, audio_data: np.ndarray) -> None:
        """Play audio chunk without blocking.

        Args:
            audio_data: Audio data as numpy array to play

        Returns:
            None
        """
        assert audio_data is not None, "Audio data cannot be None"
        assert len(audio_data) > 0, "Audio data cannot be empty"

        # Ensure correct shape for channels
        if self.channels == 1 and audio_data.ndim == 1:
            # Keep as 1D for mono
            pass
        elif self.channels > 1 and audio_data.ndim == 1:
            # Duplicate mono to all channels
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, self.channels))

        # Start playback thread if not already running
        if not self._started:
            self._stop_event.clear()
            self._playback_thread = threading.Thread(
                target=self._playback_loop, daemon=True
            )
            self._playback_thread.start()
            self._started = True
            print(
                f"Speaker stream started (device: {self.device}, "
                f"rate: {self.sample_rate}, channels: {self.channels})"
            )

        self._playback_queue.put(audio_data)

    def stop(self) -> None:
        """Stop the audio output stream.

        Returns:
            None
        """
        if not self._started:
            return

        # Stop current playback
        sd.stop()

        # Signal thread to stop
        self._stop_event.set()

        # Clear the queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
                self._playback_queue.task_done()
            except queue.Empty:
                break

        # Wait for thread to finish
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

        self._started = False
        print("Speaker stream stopped")

    def wait(self) -> None:
        """Block until all audio playback is finished.

        Returns:
            None
        """
        # Wait for queue to be empty
        self._playback_queue.join()
