import time
from unittest.mock import MagicMock, patch

from conversa.generated.output_stream.speaker import SpeakerOutputStream


def test_speaker_race_condition():
    """Test race condition during SpeakerOutputStream restart."""

    # Mock sounddevice
    with patch("conversa.generated.output_stream.speaker.sd") as mock_sd:
        # Create a stream mock that behaves somewhat realistically
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream

        # Simulate slow close to potentially trigger overlap if join times out or logic is loose
        def slow_close():
            time.sleep(0.2)

        mock_stream.close.side_effect = slow_close

        print("\n--- Starting Stream ---")
        speaker = SpeakerOutputStream(sample_rate=16000)

        # Start playback
        import numpy as np

        data = np.zeros(1600, dtype=np.float32)
        speaker.play_chunk(data)

        # Verify started
        assert speaker._playback_thread.is_alive()

        print("--- Stopping Stream ---")
        # Stop
        speaker.stop()

        assert not speaker._playback_thread.is_alive()
        print("Stream stopped successfully.")

        print("--- Restarting Stream ---")
        # Restart immediately
        speaker.play_chunk(data)

        assert speaker._playback_thread.is_alive()
        print("Stream restarted successfully.")

        speaker.stop()


if __name__ == "__main__":
    test_speaker_race_condition()
