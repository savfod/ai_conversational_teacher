"""File reader with translation and text-to-speech support.

This module provides functionality to:
1. Load text files (with future support for epub)
2. Process content in parts
3. Translate to simplified language versions
4. Read content using text-to-speech
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np
import sounddevice as sd
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from conversa.audio.audio_parser import AudioParser
from conversa.audio.input_stream import MicrophoneInputStream
from conversa.features.llm_api import call_llm
from conversa.generated.speech_api import speech_to_text, text_to_speech
from conversa.util.io import DEFAULT_READING_STATUS, read_json, write_json


class CommandListener:
    """Listens for voice commands and triggers callbacks.

    This class monitors microphone input for voice commands (specifically 'start')
    and executes callbacks when commands are detected or stop conditions are met.
    """

    def __init__(self, input_stream: MicrophoneInputStream, audio_parser: AudioParser):
        """Initialize the command listener.

        Args:
            input_stream: Microphone input stream for capturing audio
            audio_parser: Audio parser for detecting voice commands
        """
        self.input_stream = input_stream
        self.audio_parser = audio_parser

    def start_waiting_loop(
        self,
        stop_condition_callback: Callable[[], bool],
        on_start_detected_callback: Callable[[], None],
    ) -> bool:
        """Wait for either stop condition or 'start' command detection.

        This method runs a polling loop that:
        1. Checks if stop_condition_callback returns True (e.g., playback finished)
        2. Processes microphone input for voice commands
        3. Calls on_start_detected_callback if 'start' command is detected

        Args:
            stop_condition_callback: Function that returns True when waiting should stop
                                    (e.g., lambda: not sd.get_stream().active)
            on_start_detected_callback: Function to call when 'start' command detected
                                       (e.g., lambda: sd.stop())

        Returns:
            True if 'start' command was detected, False if stopped by stop condition

        """
        while not stop_condition_callback():
            time.sleep(0.1)
            print("Listening for 'start' command to pause... or stop condition met.")

            # Get and process microphone input
            chunk = self.input_stream.get_unprocessed_chunk()
            if chunk is None or len(chunk) == 0:
                continue

            status, speech, status_changed = self.audio_parser.add_chunk(chunk)

            # If 'start' command detected, trigger callback and wait for 'stop stop'
            if status == "listening":
                print("\n[Voice Control] Paused. Say 'stop stop' to resume...")
                on_start_detected_callback()
                return True

        return False

    def wait_for_stop_command(self) -> np.ndarray:
        """Wait for 'stop stop' command to resume from pause.

        This is a blocking method that continuously processes microphone input
        until the AudioParser detects transition back to 'waiting' status.
        """
        while True:
            time.sleep(0.1)
            print("Waiting for 'stop stop' command to resume...")
            chunk = self.input_stream.get_unprocessed_chunk()
            if chunk is None or len(chunk) == 0:
                continue

            status, speech, status_changed = self.audio_parser.add_chunk(chunk)
            if status == "waiting":
                # 'stop stop' detected - exit pause state
                assert speech is not None
                return speech


class ReadingStatus:
    """Class to manage reading status persistence."""

    def __init__(self, file_id: str, status_file: str | Path = DEFAULT_READING_STATUS):
        self.file_id = file_id
        self.status_file = Path(status_file)
        self.status = self._load_status()

    def _load_status(self) -> dict:
        """Load reading status from the JSON file."""
        if self.status_file.exists():
            return read_json(self.status_file)
        return {}

    def get_last_position(self) -> int:
        """Get the last read position for a given file."""
        return self.status.get(self.file_id, 0)

    def update_position(self, position: int) -> None:
        """Update the last read position for a given file."""
        self.status[self.file_id] = position
        write_json(self.status, self.status_file)

    def print_info(self):
        """Print current reading status info."""
        last_pos = self.get_last_position()
        print(
            f"File ID: {self.file_id}, read position: {last_pos}."
            "\nThe setting can be found in the file:"
            f"\nfile://{self.status_file}"
        )


@dataclass
class ChunkInfo:
    i: int
    text: str
    audio: np.ndarray | None
    end_position: int


class FileReader:
    """Reads files and processes them with optional translation and TTS."""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 500,
        simplify: bool = False,
        target_language: str = "English",
        simplification_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = "B1",
        enable_voice_control: bool = False,
    ):
        """Initialize the file reader.

        Args:
            file_path: Path to the file to read
            chunk_size: Number of characters per chunk
            simplify: Whether to simplify the text
            target_language: Target language for simplification
            simplification_level: CEFR level for simplification (A1-C2)
            enable_voice_control: Enable voice commands for pause/resume control
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.simplify = simplify
        self.target_language = target_language
        self.simplification_level = simplification_level
        self.reading_status = ReadingStatus(str(self.file_path))
        self.enable_voice_control = enable_voice_control

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def _load_txt(self) -> str:
        """Load content from a text file.

        Returns:
            The full text content of the file
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_epub(self) -> str:
        """Load content from an epub file.

        Returns:
            The full text content of the epub
        """
        all_text = []
        book = epub.read_epub(self.file_path)
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                content = item.get_content()
                soup = BeautifulSoup(content, "html.parser")
                all_text.append(soup.get_text())

        full_text = "\n\n".join(all_text)
        return full_text

    def load_data(self) -> str:
        """Load data from the file based on its extension.

        Returns:
            The full text content

        Raises:
            ValueError: If the file format is not supported
        """
        extension = self.file_path.suffix.lower()

        if extension == ".txt":
            return self._load_txt()
        elif extension == ".epub":
            return self._load_epub()
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _split_into_chunks(self, text: str) -> Iterator[tuple[str, int]]:
        """Split text into chunks of approximately chunk_size characters.

        Tries to break at sentence boundaries when possible.

        Args:
            text: The text to split

        Yields:
            Text chunks
        """
        if not text:
            return

        start = 0
        text_length = len(text)
        last_position = self.reading_status.get_last_position()
        if last_position < text_length:
            print(f"Resuming from saved position {last_position}/{text_length}")
            self.reading_status.print_info()
            start = last_position

        while start < text_length:
            end = start + self.chunk_size

            # If we're not at the end, try to find a sentence boundary
            if end < text_length:
                # Look for sentence endings in the next 100 characters
                search_end = min(end + 100, text_length)
                chunk = text[start:search_end]

                # Find the last sentence ending
                for separator in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    last_sep = chunk.rfind(separator)
                    if last_sep != -1:
                        end = start + last_sep + len(separator)
                        break

            yield text[start:end].strip(), end

            start = end

    def _simplify_text(self, text: str) -> str:
        """Translate text if required and simplify to the specified language level.

        Args:
            text: The text to simplify

        Returns:
            Simplified text
        """

        print(
            f"Simplifying text... to language {self.target_language}, level {self.simplification_level}"
        )
        user_prompt = f"""Translate the following text if required and simplify it to the {self.simplification_level} level of language {self.target_language}.
Keep the meaning intact but use simpler vocabulary and sentence structures appropriate for {self.simplification_level} learners. If required, use longer explanations. Very rare words could be described in English also. Include only the final simplified text in your response.

Text to simplify:
{text}"""

        sys_prompt = "You are a language teaching assistant that simplifies texts for language learners."

        print("Original text to simplify:")
        print(text)
        simplified = call_llm(query=user_prompt, sys_prompt=sys_prompt)
        return simplified.strip()

    def _text_to_speech(self, chunk: str) -> np.ndarray:
        """Convert text chunk to speech audio."""
        return text_to_speech(
            chunk,
            instructions=f"Read the text in clear {self.target_language} pronunciation, please. Use speed appropriate for language learners on {self.simplification_level} level. Be expressive and cheerful.",
        )

    def _prepare_chunk(
        self, text_chunk: str, chunk_index: int
    ) -> tuple[str, np.ndarray]:
        """Prepare a chunk for playback: simplify if needed and convert to audio.

        Args:
            text_chunk: Text to prepare
            chunk_index: Index of the current chunk being processed

        Returns:
            Tuple of (text, audio_data)
        """
        print(f"\n--- Chunk {chunk_index} ---")
        if self.simplify:
            text_chunk = self._simplify_text(text_chunk)
            print("--")
            print("Translated & Simplified text:")

        print(text_chunk)
        print()

        audio = self._text_to_speech(text_chunk)
        return text_chunk, audio

    def _start_play(self, audio: np.ndarray) -> None:
        """Start playing audio chunk.

        Args:
            audio: Audio data to play
        """
        sd.play(audio, samplerate=16000)

    def _finalize_chunk(self, position: int) -> None:
        """Finalize chunk by updating reading status.

        Args:
            position: Text position to save
        """
        self.reading_status.update_position(position)

    def read_file(self) -> None:
        """Read the entire file, processing chunks with optional TTS.

        Architecture:
        1. Prepare next chunk (simplify + TTS)
        2. Start playback of current chunk
        3. Wait for playback to finish OR 'start' command (which pauses)
        4. Finalize current chunk (save position)
        5. Repeat
        """
        command_listener = None

        if self.enable_voice_control:
            print("Starting microphone for voice control...")
            print("Say 'start' to pause, then 'stop stop' to resume")
            input_stream = MicrophoneInputStream(sample_rate=16000)
            input_stream.start()
            audio_parser = AudioParser(
                model_path="vosk-model-small-en-us-0.15", sample_rate=16000
            )
            command_listener = CommandListener(input_stream, audio_parser)

        try:
            chunks_stream = enumerate(
                self._split_into_chunks(self.load_data()), start=1
            )

            processed_chunk_info = None

            def _get_next_chunk_info():
                data = next(chunks_stream, None)
                if data is not None:
                    i, (chunk_text, end) = data
                    return ChunkInfo(i=i, text=chunk_text, audio=None, end_position=end)
                return None

            next_chunk_info = _get_next_chunk_info()

            while processed_chunk_info or next_chunk_info:
                # 1. Pull the next chunk from the stream
                if next_chunk_info is None:
                    next_chunk_info = _get_next_chunk_info()
                    print("Next chunk generated")

                # 2: Start playback
                play_started = False
                if (
                    processed_chunk_info is not None
                    and processed_chunk_info.audio is not None
                ):
                    print(f"Playing chunk {processed_chunk_info.i}...")
                    self._start_play(processed_chunk_info.audio)
                    play_started = True

                # 3: During playback, prepare the next chunk (simplify and convert to audio)
                if next_chunk_info is not None and next_chunk_info.audio is None:
                    _text, audio = self._prepare_chunk(
                        next_chunk_info.text, next_chunk_info.i
                    )
                    next_chunk_info = ChunkInfo(
                        i=next_chunk_info.i,
                        text=_text,
                        audio=audio,
                        end_position=next_chunk_info.end_position,
                    )
                    print("Next chunk prepared")

                # 4: Wait for playback to finish or voice command")
                processed_chunk_played = False
                # breakpoint()
                if play_started:
                    if command_listener:
                        print(
                            "Voice control enabled - waiting for 'start' command or playback finish..."
                        )
                        start_detected = command_listener.start_waiting_loop(
                            stop_condition_callback=lambda: not sd.get_stream().active,
                            on_start_detected_callback=lambda: sd.stop(),
                        )
                        if start_detected:
                            speech = command_listener.wait_for_stop_command()
                            transcription = speech_to_text(
                                speech
                            )  # todo: language code
                            print("Transcription of the command:", transcription)
                            if "resume" in transcription.lower():
                                print("[Voice Control] Resuming playback...")
                                # we need to repeat the chunk, as it was paused mid-playback

                            else:
                                llm_response = call_llm(
                                    query=f"""The user said the following (text to speech may have errors): "{transcription}" about.""",
                                    sys_prompt="You are a helpful assistant.",
                                )
                                audio = text_to_speech(
                                    llm_response,
                                    instructions="Read the text in a clear and friendly tone.",
                                )
                                sd.play(audio, samplerate=16000)
                                sd.wait()

                    else:
                        # No voice control - just wait for playback to finish
                        sd.wait()
                        processed_chunk_played = True

                # Step 5: Finalize chunk (update reading position, switch to next one) or add it
                if processed_chunk_played:
                    assert processed_chunk_info is not None
                    self._finalize_chunk(processed_chunk_info.end_position)
                    processed_chunk_info, next_chunk_info = next_chunk_info, None

                elif (
                    processed_chunk_info is None
                    and next_chunk_info is not None
                    and next_chunk_info.audio is not None
                ):
                    processed_chunk_info, next_chunk_info = next_chunk_info, None

        finally:
            if command_listener:
                command_listener.input_stream.stop()
                print("Microphone stopped.")


def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Read and process text files with optional simplification and TTS"
    )
    parser.add_argument("file_path", type=str, help="Path to the file to read")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of characters per chunk (default: 500)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the text to a target language level",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Target language for simplification (default: English)",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["A1", "A2", "B1", "B2", "C1", "C2"],
        default="B1",
        help="CEFR level for simplification (default: B1)",
    )
    parser.add_argument(
        "--voice-control",
        action="store_true",
        help="Enable voice commands to pause/resume playback (say 'start' to pause, 'stop stop' to resume)",
    )

    args = parser.parse_args()

    reader = FileReader(
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        simplify=args.simplify,
        target_language=args.language,
        simplification_level=args.level,
        enable_voice_control=args.voice_control,
    )

    reader.read_file()


if __name__ == "__main__":
    main()
