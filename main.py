import argparse
import datetime
import time

import numpy as np

from conversa.audio.audio_io import save_audio
from conversa.audio.audio_parser import AudioParser
from conversa.audio.input_stream import AudioFileInputStream, MicrophoneInputStream
from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.features.find_errors import check_for_errors
from conversa.generated.output_stream.base import AbstractAudioOutputStream
from conversa.generated.output_stream.speaker import SpeakerOutputStream
from conversa.generated.speech_api import speech_to_text, text_to_speech
from conversa.scenarios.answer import teacher_answer
from conversa.util.io import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_CONVERSATIONS_FILE,
    append_to_jsonl_file,
)
from conversa.util.logs import setup_logging


def _save(message: dict, time_str: str) -> None:
    """Save a message to the conversations log file.

    Args:
        message: A dictionary representing the message to save.
        time_str: Timestamp string for logging purposes.

    Returns:
        None
    """
    message = {"timestamp": time_str, **message}
    append_to_jsonl_file(message, DEFAULT_CONVERSATIONS_FILE)


def send_tone_signal(output_stream: AbstractAudioOutputStream, signal: str) -> None:
    """Play a short tone to signal state change.

    Args:
        output_stream: Audio output stream used to play the generated tone.
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
        output_stream.play_chunk(tone)
        output_stream.wait()
    except Exception as e:
        print(f"Warning: failed to play tone: {e}")


def main(
    language: str,
    input_stream: AbstractAudioInputStream,
    output_stream: AbstractAudioOutputStream,
) -> None:
    """Main entry point for demo application.

    This function processes audio from the provided input stream, detects speech,
    sends it to speech-to-text, checks for errors, and uses an LLM to produce
    replies which are played back via the output stream.

    Args:
        language: Language code for speech recognition and processing.
        input_stream: Configured audio input stream.
        output_stream: Configured audio output stream.

    Returns:
        None
    """
    input_stream.start()
    print("Stream started.")

    audio_parser = AudioParser(
        model_path="vosk-model-small-en-us-0.15", sample_rate=16000
    )

    history = []
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
                print(f"\nSpeech interval detected. Transcribing ({language})...")
                time_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                audio_file_path = DEFAULT_AUDIO_DIR / f"speech_{time_id}.wav"
                save_audio(speech, audio_file_path, sample_rate=16000)

                transcription = speech_to_text(speech, language=language)
                new_message = {"role": "user", "content": transcription}
                _save(new_message, time_str=time_id)
                history.append(new_message)
                print(f"Transcription: {transcription}")

                errs_message = check_for_errors(transcription, time_str=time_id)
                if errs_message:
                    print(f"Errors found:\n{errs_message}")
                    output_stream.play_chunk(
                        text_to_speech(
                            errs_message,
                            instructions="Speak in a strict and instructive teacher tone.",
                        )
                    )

                reply = teacher_answer(transcription, history=history)
                new_message = {"role": "assistant", "content": reply}
                _save(new_message, time_str=time_id)
                history.append(new_message)
                print(f"LLM Reply: {reply}")

                # Optionally, convert text back to speech
                tts_audio = text_to_speech(reply)
                output_stream.play_chunk(tts_audio)
                output_stream.wait()

                # avoid parsing text said during blocking playbacks, e.g. TTS output
                input_stream.get_unprocessed_chunk()
                audio_parser.reset()

                print(f"Generated TTS audio of length: {len(tts_audio)} bytes")

    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping...")

    finally:
        input_stream.stop()
        output_stream.stop()
        # output_stream.close()
        # todo: add close method to AbstractAudioOutputStream
        print("Input stream stopped.")
        print(
            "Text of the conversation can be found in the following file:"
            f"\nfile://{DEFAULT_CONVERSATIONS_FILE}"
        )
        print(
            f"Audio files saved in the following directory:\nfile://{DEFAULT_AUDIO_DIR}"
        )


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

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)

    # Create streams based on arguments
    if args.file:
        print("Starting AudioFileInputStream...")
        input_stream = AudioFileInputStream(file_path=args.file)
    else:
        print("Starting MicrophoneInputStream...")
        input_stream = MicrophoneInputStream(sample_rate=16000)

    output_stream = SpeakerOutputStream(sample_rate=16000)

    try:
        main(
            language=args.language,
            input_stream=input_stream,
            output_stream=output_stream,
        )
    finally:
        input_stream.stop()
        output_stream.stop()
