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
        self._queue: queue.Queue = queue.Queue()
        self._current_chunk: Optional[np.ndarray] = None
        self._current_chunk_idx: int = 0
        self._chunk_lock = threading.Lock()

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=int(sample_rate * 0.1),  # 100ms block size
            device=device,
            channels=channels,
            dtype="float32",
            callback=self._callback,
            finished_callback=self._finished_callback,
        )
        self._stream_started = False

    def _callback(self, outdata, frames, time, status):
        """Audio callback function."""
        if status:
            print(f"Status: {status}")

        chunk_size = len(outdata)
        out_idx = 0

        # Fill outdata from queue
        while out_idx < chunk_size:
            # Get new chunk if needed
            if self._current_chunk is None:
                try:
                    # Get next chunk - non-blocking here as we want to fill silence if empty
                    # but actually we probably want to block? No, sounddevice callback shouldn't block too long.
                    # But if we don't have data, we just write zeros.
                    data = self._queue.get_nowait()

                    # Handle stop sentinel if we were to support one, but stop() clears queue.
                    # Just standard data here.
                    self._current_chunk = data
                    self._current_chunk_idx = 0
                except queue.Empty:
                    # No more data, fill rest with zeros
                    outdata[out_idx:] = 0
                    return

            # Copy data
            remaining = chunk_size - out_idx
            chunk_remaining = len(self._current_chunk) - self._current_chunk_idx

            to_copy = min(remaining, chunk_remaining)

            outdata[out_idx : out_idx + to_copy] = self._current_chunk[
                self._current_chunk_idx : self._current_chunk_idx + to_copy
            ]

            out_idx += to_copy
            self._current_chunk_idx += to_copy

            # Check if chunk finished
            if self._current_chunk_idx >= len(self._current_chunk):
                self._current_chunk = None
                self._queue.task_done()

    def _finished_callback(self):
        """Called when stream finishes."""
        pass  # We don't auto-stop the stream usually, unless we want to?

    def play_chunk(self, audio_data: np.ndarray) -> None:
        """Play audio chunk without blocking.

        Args:
            audio_data: Audio data as numpy array to play
        """
        assert audio_data is not None, "Audio data cannot be None"
        assert len(audio_data) > 0, "Audio data cannot be empty"

        # Ensure correct shape for channels
        if self.channels == 1 and audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
        elif self.channels > 1 and audio_data.ndim == 1:
            # Duplicate mono to all channels
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, self.channels))

        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        self._queue.put(audio_data)

        if not self._stream_started:
            self._stream.start()
            self._stream_started = True

    def stop(self) -> None:
        """Stop the audio output stream immediately."""
        if not self._stream_started:
            return

        # Stop stream immediately
        self._stream.stop()
        self._stream_started = False

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        # Clear current chunk
        with self._chunk_lock:
            # Note: We can't easily clear _current_chunk inside callback from here safely without lock
            # But since we stopped stream, callback won't run.
            # Reset state for next play
            if self._current_chunk is not None:
                # We need to manually task_done if we are discarding a chunk that was halfway played?
                # Actually, if we pulled it from valid queue item, we should task_done it if we discard it.
                # But typically we just reset.
                self._current_chunk = None
                self._current_chunk_idx = 0
                # If we broke in middle of chunk, that chunk was already 'get' from queue.
                # So we should call task_done if we aren't going to finish it?
                # The queue.join() relies on task_done.
                # If callback pulled it, it's responsible.
                # If callback didn't finish it, we must finish it?
                # Actually, simpler: we stopped stream. Next start will create new stream or reuse?
                # Sounddevice streams can be restarted.
                pass

        # Re-create stream or just stop/start?
        # stop() makes it inactive. start() resumes? or restarts?
        # SD docs: "stopped stream can be restarted".
        # However, buffer state in callback might be stale?
        # We manually reset _current_chunk = None above, so next callback start fresh from queue.
        # But wait, if we stopped in middle of callback?
        # callback doesn't run while stopped.

        # Crucially: we need to ensure play_chunk works again.
        # If we just _stream.stop(), we can _stream.start() later.

        # But we also need to make sure `wait()` doesn't hang if we stopped.
        # `wait()` waits for queue.join().
        # If we cleared queue, we called task_down for all items in queue?
        # Yes, in the while loop above.
        # But what about the item currently in `_current_chunk`?
        # That item was popped from queue. task_done() is called only when finished.
        # If we interrupt, we must call task_done() for it too!
        if self._current_chunk is not None:
            self._queue.task_done()
            self._current_chunk = None

        print("Speaker stream stopped")

    def wait(self) -> None:
        """Block until all audio playback is finished."""
        if not self._stream_started and self._queue.empty():
            return

        self._queue.join()

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if not self._stream_started:
            return False
        return not self._queue.empty() or self._current_chunk is not None
