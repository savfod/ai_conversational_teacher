"""File reader with translation and text-to-speech support.

This module provides functionality to:
1. Load text files (with future support for epub)
2. Process content in parts
3. Translate to simplified language versions
4. Read content using text-to-speech
"""

import argparse
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import sounddevice as sd
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from conversa.audio.audio_parser import AudioParser
from conversa.audio.input_stream import MicrophoneInputStream
from conversa.features.llm_api import call_llm
from conversa.generated.speech_api import text_to_speech
from conversa.util.io import DEFAULT_READING_STATUS, read_json, write_json


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

    def _check_voice_commands(
        self,
        input_stream: MicrophoneInputStream,
        audio_parser: AudioParser,
    ) -> bool:
        """Check for voice commands during playback.

        Args:
            input_stream: The microphone input stream
            audio_parser: The audio parser for command detection

        Returns:
            True if "start" command detected (resume), False to continue waiting
        """
        chunk = input_stream.get_unprocessed_chunk()
        if chunk is None or len(chunk) == 0:
            return False

        status, speech, status_changed = audio_parser.add_chunk(chunk)

        if status == "listening":
            # Start command detected - wait for stop to resume
            print("\n[Voice Control] Paused. Say 'stop stop' to resume...")
            sd.stop()
            while True:
                import time

                time.sleep(0.1)
                chunk = input_stream.get_unprocessed_chunk()
                if chunk is None or len(chunk) == 0:
                    continue

                status, speech, status_changed = audio_parser.add_chunk(chunk)
                if status == "waiting":
                    # Stop command detected - resume playback
                    print("[Voice Control] Resuming playback...")
                    return True

        return False

    def read_file(self) -> None:
        """Read the entire file, processing chunks with optional TTS."""
        input_stream = None
        audio_parser = None

        if self.enable_voice_control:
            print("Starting microphone for voice control...")
            print("Say 'start' to pause, then 'stop stop' to resume")
            input_stream = MicrophoneInputStream(sample_rate=16000)
            input_stream.start()
            audio_parser = AudioParser(
                model_path="vosk-model-small-en-us-0.15", sample_rate=16000
            )

        try:
            prev_end = None
            for i, (chunk, end) in enumerate(
                self._split_into_chunks(self.load_data()), 1
            ):
                if self.simplify:
                    chunk = self._simplify_text(chunk)

                print(f"\n--- Chunk {i} ---")
                print(chunk)
                print()

                # Read aloud
                audio = self._text_to_speech(chunk)
                sd.wait()  # Wait for previous audio to finish
                sd.play(audio, samplerate=16000)

                # Monitor for voice commands during playback
                if self.enable_voice_control and input_stream and audio_parser:
                    import time

                    # Check periodically while audio is playing
                    while sd.get_stream().active:
                        time.sleep(0.1)
                        if self._check_voice_commands(input_stream, audio_parser):
                            # User requested pause - stop current playback
                            sd.stop()
                            break

                # Update reading status on previous end
                if prev_end is not None:
                    self.reading_status.update_position(prev_end)
                prev_end = end

        finally:
            if input_stream:
                input_stream.stop()
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
