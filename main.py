import sys
import time

import numpy as np
import sounddevice as sd

from aiteacher.audio.audio_parser import AudioParser
from aiteacher.audio.input_stream import AudioFileInputStream, MicrophoneInputStream
from aiteacher.generated.llm import answer
from aiteacher.generated.speech_api import speech_to_text, text_to_speech
from aiteacher.scenario.find_errors import check_for_errors


def send_tone_signal(output_stream, signal: str):
    """Play a short tone to signal state change"""

    def _generate_tone(freq: float, duration: float = 0.15, sample_rate: int = 16000, amplitude: float = 0.25) -> np.ndarray:
        """Generate a simple sine tone (mono) as float32 numpy array.

        Args:
            freq: Frequency in Hz.
            duration: Duration in seconds.
            sample_rate: Sample rate in Hz.
            amplitude: Peak amplitude (0..1).

        Returns:
            Numpy array of shape (n,) dtype float32.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * freq * t)
        return tone.astype(np.float32)

    if signal == 'listening':
        # start tone (higher pitch, short)
        tone = _generate_tone(freq=880.0, duration=0.12)
    else:
        # stop tone (lower pitch, slightly longer)
        tone = _generate_tone(freq=440.0, duration=0.18)
    try:
        output_stream.write(tone)
    except Exception as e:
        print(f"Warning: failed to play tone: {e}")



def main():
    """Main function to demonstrate MicrophoneInputStream usage."""
    print("Starting MicrophoneInputStream...")

    if sys.argv[1:] and sys.argv[1] == "file":
        file_path = sys.argv[2] if len(sys.argv) > 2 else "data/test_audio/error1.wav"
        input_stream = AudioFileInputStream(
            file_path=file_path,
        )
        input_stream.start()
        print("Audio file stream started.")
    else:
        input_stream = MicrophoneInputStream(sample_rate=16000)
        input_stream.start()
        print("Microphone stream started. Press Ctrl+C to stop.")

    audio_parser = AudioParser(model_path="vosk-model-small-en-us-0.15", sample_rate=16000)

    output_stream = sd.OutputStream(
        samplerate=16000, 
        # blocksize=2048,
        channels=1, 
        dtype='float32',
    )
    output_stream.start()


    prev_status = None
    try:
        while True:
            time.sleep(0.5)
            chunk = input_stream.get_unprocessed_chunk()
            if chunk is None:
                continue
            assert len(chunk) > 0

            status, speech, status_changed = audio_parser.add_chunk(chunk)
            if status_changed:
                send_tone_signal(output_stream, status)

            if status == 'listening':
                print(".", end="", flush=True)

            if speech is not None:
                print("\nSpeech interval detected. Transcribing...")
                transcription = speech_to_text(speech, language='en')
                print(f"Transcription: {transcription}")

                errs_message = check_for_errors(transcription)
                if errs_message:
                    print(f"Errors found:\n{errs_message}")
                    output_stream.write(text_to_speech(errs_message, instructions="Speak in a strict and instructive teacher tone."))

                reply = answer(transcription)
                print(f"LLM Reply: {reply}")

                # Optionally, convert text back to speech
                tts_audio = text_to_speech(reply)
                output_stream.write(tts_audio)

                # avoid parsing text said during blocking playbacks, e.g. TTS output
                input_stream.get_unprocessed_chunk()
                audio_parser._reset_vosk() 

                print(f"Generated TTS audio of length: {len(tts_audio)} bytes")



    
    
    except KeyboardInterrupt:
        print("Stopping microphone stream...")
    finally:
        input_stream.stop()
        output_stream.stop()
        output_stream.close()
        print("Microphone stream stopped.")




if __name__ == "__main__":
    main()
