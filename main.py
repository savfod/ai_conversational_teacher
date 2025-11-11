import argparse
import time

import numpy as np
import sounddevice as sd

from conversa.audio.audio_parser import AudioParser
from conversa.audio.input_stream import AudioFileInputStream, MicrophoneInputStream
from conversa.generated.llm import answer
from conversa.generated.speech_api import speech_to_text, text_to_speech
from conversa.scenario.find_errors import check_for_errors
from conversa.util.logs import setup_logging


def send_tone_signal(output_stream: "sd.OutputStream", signal: str) -> None:
    """Play a short tone to signal state change.

    Args:
        output_stream: A sounddevice OutputStream used to write the generated tone.
        signal: A textual signal, e.g. 'listening' or other values to select tone.

    Returns:
        None
    """

    def _generate_tone(
        freq: float,
        duration: float = 0.15,
        sample_rate: int = 16000,
        amplitude: float = 0.25,
    ) -> np.ndarray:
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

    if signal == "listening":
        # start tone (higher pitch, short)
        tone = _generate_tone(freq=880.0, duration=0.12)
    else:
        # stop tone (lower pitch, slightly longer)
        tone = _generate_tone(freq=440.0, duration=0.18)
    try:
        output_stream.write(tone)
    except Exception as e:
        print(f"Warning: failed to play tone: {e}")


def main(language: str, file_path: str | None = None) -> None:
    """Main entry point for demo application.

    This function starts either a microphone input stream or an audio-file-based
    input stream, processes audio to detect speech intervals, sends them to a
    speech-to-text service, checks for language errors, and uses an LLM to
    produce replies which are played back.

    Args:
        language: Language code for speech recognition and processing (e.g., "en" for English).
        file_path: Optional path to an audio file. If provided, the file input
            stream is used; otherwise the microphone is used.

    Returns:
        None
    """
    if file_path is not None:
        print("Starting AudioFileInputStream...")
        input_stream = AudioFileInputStream(
            file_path=file_path,
        )
        input_stream.start()
        print("Audio file stream started.")
    else:
        print("Starting MicrophoneInputStream...")
        input_stream = MicrophoneInputStream(sample_rate=16000)
        input_stream.start()
        print("Microphone stream started. Press Ctrl+C to stop.")

    audio_parser = AudioParser(
        model_path="vosk-model-small-en-us-0.15", sample_rate=16000
    )

    output_stream = sd.OutputStream(
        samplerate=16000,
        # blocksize=2048,
        channels=1,
        dtype="float32",
    )
    output_stream.start()

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

            if status == "listening":
                print(".", end="", flush=True)

            if speech is not None:
                print("\nSpeech interval detected. Transcribing...")
                transcription = speech_to_text(speech, language=language)
                print(f"Transcription: {transcription}")

                errs_message = check_for_errors(transcription)
                if errs_message:
                    print(f"Errors found:\n{errs_message}")
                    output_stream.write(
                        text_to_speech(
                            errs_message,
                            instructions="Speak in a strict and instructive teacher tone.",
                        )
                    )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Conversational Teacher")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to an audio file to process instead of using the microphone.",
    )
    parser.add_argument(
        "language",
        default="en",
        nargs="?",
        help="Language code for speech recognition and processing (default: en).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging(level="INFO")

    args = parse_args()
    main(language=args.language, file_path=args.file)
