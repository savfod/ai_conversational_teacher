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
        blocksize_sec: float = 0.5,
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
        self.blocksize = int(sample_rate * blocksize_sec)
        self._playback_queue: queue.Queue = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._output_stream: Optional[sd.OutputStream] = None

    def _playback_loop(self, stop_event: threading.Event) -> None:
        """Playback loop that processes audio chunks from the queue."""
        # Create the output stream
        self._output_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            blocksize=self.blocksize,
        )
        self._output_stream.start()

        try:
            while not stop_event.is_set():
                try:
                    # Wait for audio chunk with timeout to check stop event
                    audio_chunk = self._playback_queue.get(timeout=0.01)

                    if audio_chunk is None:  # Sentinel value to stop
                        break

                    # Write audio to output stream
                    self._writing = True
                    try:
                        self._output_stream.write(audio_chunk)
                    except Exception as e:
                        print(f"Error writing to output stream: {e}")
                    finally:
                        self._writing = False

                    self._playback_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in playback loop: {e}")
                    self._writing = False
                    self._playback_queue.task_done()
        finally:
            # Clean up the output stream
            if self._output_stream:
                self._output_stream.stop()
                self._output_stream.close()
                self._output_stream = None

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
            self._current_stop_event = threading.Event()
            self._playback_thread = threading.Thread(
                target=self._playback_loop,
                args=(self._current_stop_event,),
                daemon=True,
            )
            self._playback_thread.start()
            self._started = True
            # print(
            #     f"Speaker stream started (device: {self.device}, "
            #     f"rate: {self.sample_rate}, channels: {self.channels})"
            # )

        self._playback_queue.put(audio_data)

    def stop(self) -> None:
        """Stop the audio output stream.

        Returns:
            None
        """
        if not self._started:
            return

        # Signal thread to stop
        if hasattr(self, "_current_stop_event"):
            self._current_stop_event.set()

        # Put sentinel value to unblock the queue
        self._playback_queue.put(None)

        # Wait for thread to finish
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

        # Clear any remaining items in the queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
                self._playback_queue.task_done()
            except queue.Empty:
                break

        self._started = False
        print("Speaker stream stopped")

    def wait(self) -> None:
        """Block until all audio playback is finished.

        Returns:
            None
        """
        if not self._started:
            return

        # Wait for queue to be empty
        self._playback_queue.join()

    def is_playing(self) -> bool:
        """Check if audio is currently playing.

        Returns:
            True if audio is playing (queue not empty or currently writing), False otherwise
        """
        # We consider it playing if the queue has items or if the thread is currently writing
        # Note: This is an approximation. The underlying stream implementation details
        # determine exactly when sound stops, but this tracks our application-level buffer.
        return not self._playback_queue.empty() or getattr(self, "_writing", False)
