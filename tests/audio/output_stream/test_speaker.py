"""
Tests for SpeakerOutputStream class.
"""

import time
from unittest.mock import MagicMock, patch

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

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_play_chunk_starts_stream(self, mock_output_stream):
        """Test that play_chunk starts the playback thread."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream()
        audio_data = np.random.randn(1600).astype(np.float32)

        stream.play_chunk(audio_data)
        time.sleep(0.2)  # Give thread time to start and process

        # Verify OutputStream was created and started
        assert mock_output_stream.call_count >= 1
        assert mock_stream_instance.start.called
        assert stream._started

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_play_chunk_with_correct_parameters(self, mock_output_stream):
        """Test that sd.OutputStream is created with correct parameters."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream(sample_rate=8000, channels=2, device=3)
        audio_data = np.random.randn(800).astype(np.float32)

        stream.play_chunk(audio_data)
        time.sleep(0.2)

        # Verify sd.OutputStream was created with correct parameters
        assert mock_output_stream.call_count >= 1
        call_kwargs = mock_output_stream.call_args[1]
        assert call_kwargs["samplerate"] == 8000
        assert call_kwargs["device"] == 3
        assert call_kwargs["channels"] == 2

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_play_multiple_chunks(self, mock_output_stream):
        """Test playing multiple audio chunks."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream()

        # Play multiple chunks
        for _ in range(3):
            audio_data = np.random.randn(800).astype(np.float32)
            stream.play_chunk(audio_data)

        time.sleep(0.3)  # Give time to process

        # Verify write was called multiple times
        assert mock_stream_instance.write.call_count >= 3

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_stop(self, mock_output_stream):
        """Test stopping the stream."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream()
        audio_data = np.random.randn(1600).astype(np.float32)

        stream.play_chunk(audio_data)
        time.sleep(0.1)

        stream.stop()

        # Verify stream was stopped and closed
        assert mock_stream_instance.stop.called
        assert mock_stream_instance.close.called
        assert not stream._started

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_wait(self, mock_output_stream):
        """Test waiting for playback to finish."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

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

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_play_chunk_mono_audio(self, mock_output_stream):
        """Test playing mono audio."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream(channels=1)

        # Play 1D mono audio
        audio_mono = np.random.randn(1600).astype(np.float32)
        stream.play_chunk(audio_mono)

        time.sleep(0.2)

        # Verify write was called
        assert mock_stream_instance.write.call_count >= 1

        stream.stop()

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_play_chunk_stereo_audio(self, mock_output_stream):
        """Test playing stereo audio (mono duplicated to stereo)."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream(channels=2)

        # Play 1D mono audio (should be duplicated to stereo)
        audio_mono = np.random.randn(1600).astype(np.float32)
        stream.play_chunk(audio_mono)

        time.sleep(0.2)

        # Verify write was called
        assert mock_stream_instance.write.call_count >= 1
        if mock_stream_instance.write.call_count > 0:
            written_audio = mock_stream_instance.write.call_args[0][0]
            assert written_audio.ndim == 2
            assert written_audio.shape[1] == 2

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

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_stop_clears_queue(self, mock_output_stream):
        """Test that stop clears the playback queue."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream()

        # Add many chunks to queue
        for _ in range(10):
            audio_data = np.random.randn(800).astype(np.float32)
            stream.play_chunk(audio_data)

        # Stop immediately (before they can all play)
        stream.stop()

        # Queue should be cleared
        assert stream._playback_queue.empty()

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_exception_handling_in_playback_loop(self, mock_output_stream, capsys):
        """Test that exceptions in playback loop are handled gracefully."""
        mock_stream_instance = MagicMock()
        mock_stream_instance.write.side_effect = Exception("Test error")
        mock_output_stream.return_value = mock_stream_instance

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
