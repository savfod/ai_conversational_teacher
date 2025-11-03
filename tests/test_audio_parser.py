"""Test audio parser functionality with start/stop commands."""

import sys
import time
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the audio module to the path
sys.path.append(str(Path(__file__).parent.parent / "aiteacher" / "audio"))
sys.path.append(str(Path(__file__).parent.parent / "aiteacher" / "generated"))

import soundfile as sf
from audio_parser import AudioParser

class TestAudioParser:
    """Test suite for AudioParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use mock Vosk model for testing to avoid dependency on actual model files
        self.mock_model_path = "vosk-model-small-en-us-0.15"
        self.sample_rate = 16000
        
    @patch('audio_parser.vosk.Model')
    @patch('audio_parser.vosk.KaldiRecognizer')
    def test_parser_initialization(self, mock_recognizer, mock_model):
        """Test that AudioParser initializes correctly."""
        # Mock the Path.exists() to return True
        with patch('audio_parser.Path.exists', return_value=True):
            parser = AudioParser(self.mock_model_path, self.sample_rate)
            
            assert parser.sample_rate == self.sample_rate
            assert parser.status == "waiting"
            assert parser.buffered_duration == 0.0
    
    @patch('audio_parser.vosk.Model')
    @patch('audio_parser.vosk.KaldiRecognizer')
    def test_start_command_detection(self, mock_recognizer, mock_model):
        """Test detection of start command."""
        with patch('audio_parser.Path.exists', return_value=True):
            parser = AudioParser(self.mock_model_path)
            
            # Mock Vosk to return start command
            mock_recognizer_instance = mock_recognizer.return_value
            mock_recognizer_instance.AcceptWaveform.return_value = True
            mock_recognizer_instance.Result.return_value = '{"text": "start recording"}'
            
            # Create dummy audio chunk
            audio_chunk = np.random.normal(0, 0.1, 1600).astype(np.float32)  # 0.1 seconds
            
            status, audio = parser.add_chunk(audio_chunk)
            
            assert status == "listening"
            assert audio is None  # No complete interval yet
            assert parser.status == "listening"
    
    @patch('audio_parser.vosk.Model')
    @patch('audio_parser.vosk.KaldiRecognizer')
    def test_stop_command_detection(self, mock_recognizer, mock_model):
        """Test detection of stop command and audio interval extraction."""
        with patch('audio_parser.Path.exists', return_value=True):
            parser = AudioParser(self.mock_model_path)
            
            mock_recognizer_instance = mock_recognizer.return_value
            
            # First, simulate start command
            mock_recognizer_instance.AcceptWaveform.return_value = True
            mock_recognizer_instance.Result.return_value = '{"text": "start now"}'
            
            audio_chunk1 = np.random.normal(0, 0.1, 1600).astype(np.float32)
            status, audio = parser.add_chunk(audio_chunk1)
            assert status == "listening"
            assert audio is None
            
            # Add some audio chunks while listening
            mock_recognizer_instance.AcceptWaveform.return_value = False  # No speech detected
            mock_recognizer_instance.Result.return_value = '{"text": ""}'
            
            audio_chunk2 = np.random.normal(0, 0.2, 1600).astype(np.float32)
            audio_chunk3 = np.random.normal(0, 0.15, 1600).astype(np.float32)
            
            parser.add_chunk(audio_chunk2)
            parser.add_chunk(audio_chunk3)
            
            # Now simulate stop command
            mock_recognizer_instance.AcceptWaveform.return_value = True
            mock_recognizer_instance.Result.return_value = '{"text": "stop recording"}'
            
            audio_chunk4 = np.random.normal(0, 0.1, 1600).astype(np.float32)
            status, returned_audio = parser.add_chunk(audio_chunk4)
            
            assert status == "waiting"
            assert returned_audio is not None
            assert len(returned_audio) > 0
            # Should contain all chunks from start to stop
            expected_length = len(audio_chunk1) + len(audio_chunk2) + len(audio_chunk3) + len(audio_chunk4)
            assert len(returned_audio) == expected_length
    
    @patch('audio_parser.vosk.Model')
    @patch('audio_parser.vosk.KaldiRecognizer')
    def test_waiting_state_ignores_non_start_commands(self, mock_recognizer, mock_model):
        """Test that parser ignores non-start commands when in waiting state."""
        with patch('audio_parser.Path.exists', return_value=True):
            parser = AudioParser(self.mock_model_path)
            
            mock_recognizer_instance = mock_recognizer.return_value
            mock_recognizer_instance.AcceptWaveform.return_value = True
            mock_recognizer_instance.Result.return_value = '{"text": "stop recording"}'
            
            audio_chunk = np.random.normal(0, 0.1, 1600).astype(np.float32)
            status, audio = parser.add_chunk(audio_chunk)
            
            # Should remain in waiting state
            assert status == "waiting"
            assert audio is None
            assert parser.status == "waiting"
    
    @patch('audio_parser.vosk.Model')
    @patch('audio_parser.vosk.KaldiRecognizer')
    def test_buffered_duration_calculation(self, mock_recognizer, mock_model):
        """Test calculation of buffered audio duration."""
        with patch('audio_parser.Path.exists', return_value=True):
            parser = AudioParser(self.mock_model_path, sample_rate=16000)
            
            mock_recognizer_instance = mock_recognizer.return_value
            
            # Start listening
            mock_recognizer_instance.AcceptWaveform.return_value = True
            mock_recognizer_instance.Result.return_value = '{"text": "start"}'
            
            audio_chunk = np.random.normal(0, 0.1, 16000).astype(np.float32)  # 1 second
            parser.add_chunk(audio_chunk)
            
            assert abs(parser.buffered_duration - 1.0) < 0.01  # Should be close to 1 second
            
            # Add another chunk
            mock_recognizer_instance.AcceptWaveform.return_value = False
            parser.add_chunk(audio_chunk)
            
            assert abs(parser.buffered_duration - 2.0) < 0.01  # Should be close to 2 seconds
    
    @patch('audio_parser.vosk.Model')
    @patch('audio_parser.vosk.KaldiRecognizer')
    def test_reset_functionality(self, mock_recognizer, mock_model):
        """Test parser reset functionality."""
        with patch('audio_parser.Path.exists', return_value=True):
            parser = AudioParser(self.mock_model_path)
            
            mock_recognizer_instance = mock_recognizer.return_value
            
            # Start listening and add some audio
            mock_recognizer_instance.AcceptWaveform.return_value = True
            mock_recognizer_instance.Result.return_value = '{"text": "start"}'
            
            audio_chunk = np.random.normal(0, 0.1, 1600).astype(np.float32)
            parser.add_chunk(audio_chunk)
            
            assert parser.status == "listening"
            assert parser.buffered_duration > 0
            
            # Reset parser
            parser.reset()
            
            assert parser.status == "waiting"
            assert parser.buffered_duration == 0.0


def test_with_actual_audio_file():
    """Integration test using the actual start_stop.mp3 file with input stream."""
    # Use the existing start_stop.mp3 file
    test_audio_path = Path(__file__).parent.parent / "aiteacher" / "audio" / "start_stop.mp3"
    
    if not test_audio_path.exists():
        print(f"Warning: Test audio file not found at {test_audio_path}")
        return
    
    print(f"Testing with audio file: {test_audio_path}")
    
    # Import input stream classes
    sys.path.append(str(Path(__file__).parent.parent / "aiteacher" / "audio"))
    try:
        from input_stream import AudioFileInputStream
    except ImportError as e:
        print(f"Warning: Could not import input_stream: {e}")
        # Fallback to direct audio file loading
        try:
            audio_data, sample_rate = sf.read(str(test_audio_path))
            print(f"Loaded test audio directly: {test_audio_path.name}, {len(audio_data) / sample_rate:.1f} seconds")
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return
        
        # Convert to mono if necessary
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Use the original test approach
        _test_with_direct_audio_data(audio_data, sample_rate)
        return
    
    # Test with input stream
    print("Using AudioFileInputStream for testing...")
    
    # Initialize input stream
    input_stream = AudioFileInputStream(
        file_path=str(test_audio_path),
        sample_rate=16000,
        chunk_duration=0.1  # 100ms chunks
    )
    
    # Test audio parsing with mocked Vosk
    with patch('audio_parser.vosk.Model'), \
         patch('audio_parser.vosk.KaldiRecognizer'), \
         patch('audio_parser.Path.exists', return_value=True):
        
        parser = AudioParser("vosk-model-small-en-us-0.15", 16000)
        
        # Mock Vosk responses to simulate start and stop detection
        mock_recognizer = parser.recognizer
        
        def mock_accept_waveform(audio_bytes):
            # Simulate detecting speech periodically
            return True
        
        # Use a class to maintain state instead of function attributes
        class MockResult:
            def __init__(self):
                self.call_count = 0
            
            def __call__(self):
                result = self.call_count
                self.call_count += 1
                
                if result == 0:
                    return '{"text": "start recording please"}'
                elif result > 10:  # After more chunks for input stream
                    return '{"text": "stop recording now"}'
                else:
                    return '{"text": ""}'
        
        mock_recognizer.AcceptWaveform = mock_accept_waveform
        mock_recognizer.Result = MockResult()
        
        # Start input stream
        input_stream.start()
        
        # Process audio chunks from input stream
        speech_intervals = []
        chunk_count = 0
        
        try:
            for _ in range(50):  # Process up to 50 chunks
                chunk = input_stream.get_unprocessed_chunk()
                if chunk is not None and len(chunk) > 0:
                    chunk_count += 1
                    status, audio_interval = parser.add_chunk(chunk)
                    
                    if audio_interval is not None:
                        speech_intervals.append(audio_interval)
                        print(f"Captured speech interval: {len(audio_interval) / 16000:.2f} seconds")
                
                time.sleep(0.05)  # Small delay between chunk processing
        
        finally:
            input_stream.stop()
        
        assert len(speech_intervals) > 0, "Should have captured at least one speech interval"
        print(f"Successfully captured {len(speech_intervals)} speech interval(s) using input stream")


def _test_with_direct_audio_data(audio_data, sample_rate):
    """Fallback test method using direct audio data."""
    print("Using direct audio data for testing...")
    
    # Test audio parsing with mocked Vosk
    with patch('audio_parser.vosk.Model'), \
         patch('audio_parser.vosk.KaldiRecognizer'), \
         patch('audio_parser.Path.exists', return_value=True):
        
        parser = AudioParser("vosk-model-small-en-us-0.15", int(sample_rate))
        
        # Mock Vosk responses to simulate start and stop detection
        mock_recognizer = parser.recognizer
        
        def mock_accept_waveform(audio_bytes):
            # Simulate detecting speech periodically
            return True
        
        # Use a class to maintain state instead of function attributes
        class MockResult:
            def __init__(self):
                self.call_count = 0
            
            def __call__(self):
                result = self.call_count
                self.call_count += 1
                
                if result == 0:
                    return '{"text": "start recording please"}'
                elif result > 5:  # After several chunks
                    return '{"text": "stop recording now"}'
                else:
                    return '{"text": ""}'
        
        mock_recognizer.AcceptWaveform = mock_accept_waveform
        mock_recognizer.Result = MockResult()
        
        # Process audio in chunks
        chunk_size = int(sample_rate) // 10  # 0.1 second chunks
        speech_intervals = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size].astype(np.float32)
            status, audio_interval = parser.add_chunk(chunk)
            
            if audio_interval is not None:
                speech_intervals.append(audio_interval)
                print(f"Captured speech interval: {len(audio_interval) / sample_rate:.2f} seconds")
        
        assert len(speech_intervals) > 0, "Should have captured at least one speech interval"
        print(f"Successfully captured {len(speech_intervals)} speech interval(s)")


if __name__ == "__main__":
    # Run the integration test
    print("Running integration test with start_stop.mp3...")
    test_with_actual_audio_file()
    print("Test completed.")