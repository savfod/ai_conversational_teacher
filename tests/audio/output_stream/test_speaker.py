"""
Tests for SpeakerOutputStream class.
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from conversa.generated.output_stream.speaker import SpeakerOutputStream


class TestSpeakerOutputStream:
    """Test suite for SpeakerOutputStream class."""

    def test_initialization(self):
        """Test SpeakerOutputStream initialization with default parameters."""
        stream = SpeakerOutputStream()

        assert stream.sample_rate == 16000
        assert stream.channels == 1
        assert stream.device is None
        assert not stream._started

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        stream = SpeakerOutputStream(sample_rate=8000, channels=2, device=1)

        assert stream.sample_rate == 8000
        assert stream.channels == 2
        assert stream.device == 1

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    def test_play_chunk_starts_stream(self, mock_wait, mock_play):
        """Test that play_chunk starts the playback thread."""
        stream = SpeakerOutputStream()
        audio_data = np.random.randn(1600).astype(np.float32)

        stream.play_chunk(audio_data)
        time.sleep(0.2)  # Give thread time to start and process

        # Verify play was called
        assert mock_play.call_count >= 1
        assert stream._started

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    def test_play_chunk_with_correct_parameters(self, mock_wait, mock_play):
        """Test that sd.play is called with correct parameters."""
        stream = SpeakerOutputStream(sample_rate=8000, channels=2, device=3)
        audio_data = np.random.randn(800).astype(np.float32)

        stream.play_chunk(audio_data)
        time.sleep(0.2)

        # Verify sd.play was called with correct parameters
        assert mock_play.call_count >= 1
        call_kwargs = mock_play.call_args[1]
        assert call_kwargs["samplerate"] == 8000
        assert call_kwargs["device"] == 3

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    def test_play_multiple_chunks(self, mock_wait, mock_play):
        """Test playing multiple audio chunks."""
        stream = SpeakerOutputStream()

        # Play multiple chunks
        for _ in range(3):
            audio_data = np.random.randn(800).astype(np.float32)
            stream.play_chunk(audio_data)

        time.sleep(0.3)  # Give time to process

        # Verify sd.play was called multiple times
        assert mock_play.call_count >= 1

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    @patch("conversa.generated.output_stream.speaker.sd.stop")
    def test_stop(self, mock_sd_stop, mock_wait, mock_play):
        """Test stopping the stream."""
        stream = SpeakerOutputStream()
        audio_data = np.random.randn(1600).astype(np.float32)

        stream.play_chunk(audio_data)
        time.sleep(0.1)

        stream.stop()

        # Verify sd.stop was called
        assert mock_sd_stop.call_count >= 1
        assert not stream._started

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    def test_wait(self, mock_wait, mock_play):
        """Test waiting for playback to finish."""
        stream = SpeakerOutputStream()

        # Play some chunks
        for _ in range(2):
            audio_data = np.random.randn(800).astype(np.float32)
            stream.play_chunk(audio_data)

        # Wait for completion
        time.sleep(0.3)
        stream.wait()

        # Queue should be empty after wait
        assert stream._playback_queue.empty()

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    def test_play_chunk_mono_audio(self, mock_wait, mock_play):
        """Test playing mono audio."""
        stream = SpeakerOutputStream(channels=1)

        # Play 1D mono audio
        audio_mono = np.random.randn(1600).astype(np.float32)
        stream.play_chunk(audio_mono)

        time.sleep(0.2)

        # Verify play was called
        assert mock_play.call_count >= 1

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    def test_play_chunk_stereo_audio(self, mock_wait, mock_play):
        """Test playing stereo audio (mono duplicated to stereo)."""
        stream = SpeakerOutputStream(channels=2)

        # Play 1D mono audio (should be duplicated to stereo)
        audio_mono = np.random.randn(1600).astype(np.float32)
        stream.play_chunk(audio_mono)

        time.sleep(0.2)

        # Verify play was called
        assert mock_play.call_count >= 1
        if mock_play.call_count > 0:
            played_audio = mock_play.call_args[0][0]
            assert played_audio.ndim == 2
            assert played_audio.shape[1] == 2

        stream.stop()

    def test_play_chunk_empty_array_assertion(self):
        """Test that empty arrays raise an assertion error."""
        stream = SpeakerOutputStream()

        with pytest.raises(AssertionError):
            stream.play_chunk(np.array([]))

    def test_play_chunk_none_assertion(self):
        """Test that None raises an assertion error."""
        stream = SpeakerOutputStream()

        with pytest.raises(AssertionError):
            stream.play_chunk(None)

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    @patch("conversa.generated.output_stream.speaker.sd.stop")
    def test_stop_clears_queue(self, mock_sd_stop, mock_wait, mock_play):
        """Test that stop clears the playback queue."""
        stream = SpeakerOutputStream()

        # Add many chunks to queue
        for _ in range(10):
            audio_data = np.random.randn(800).astype(np.float32)
            stream.play_chunk(audio_data)

        # Stop immediately (before they can all play)
        stream.stop()

        # Queue should be cleared
        assert stream._playback_queue.empty()

    @patch("conversa.generated.output_stream.speaker.sd.play")
    @patch("conversa.generated.output_stream.speaker.sd.wait")
    def test_exception_handling_in_playback_loop(self, mock_wait, mock_play, capsys):
        """Test that exceptions in playback loop are handled gracefully."""
        mock_play.side_effect = Exception("Test error")

        stream = SpeakerOutputStream()
        audio_data = np.random.randn(1600).astype(np.float32)

        stream.play_chunk(audio_data)
        time.sleep(0.3)

        captured = capsys.readouterr()
        assert "Error in playback loop:" in captured.out

        stream.stop()

    def test_stop_before_play(self):
        """Test that stopping before playing doesn't cause errors."""
        stream = SpeakerOutputStream()

        # Should not raise an exception
        stream.stop()

        assert not stream._started

    def test_wait_with_empty_queue(self):
        """Test wait with empty queue completes immediately."""
        stream = SpeakerOutputStream()

        # Should complete without blocking
        stream.wait()

        # No errors should occur
        assert True
