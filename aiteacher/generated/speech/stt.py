"""Speech-to-text functionality using OpenAI Whisper API."""

import io
from pathlib import Path
from typing import Optional, Union

from openai import OpenAI


class SpeechToText:
    """Speech-to-text converter using OpenAI's Whisper API.

    This class provides an interface to convert audio files or audio data
    into text using OpenAI's Whisper model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        language: Optional[str] = None,
    ):
        """Initialize the speech-to-text converter.

        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY env var.
            model: Whisper model to use (default: "whisper-1")
            language: Optional language code (e.g., "en", "es", "fr") to improve
                     accuracy and speed. If None, language will be auto-detected.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.language = language

    def transcribe_file(
        self,
        audio_file_path: Union[str, Path],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0,
    ) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_file_path: Path to the audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)
            language: Optional language code to override the default
            prompt: Optional text to guide the model's style or continue a previous segment
            response_format: Format of the response ("text", "json", "srt", "verbose_json", "vtt")
            temperature: Sampling temperature (0.0 to 1.0). Lower is more deterministic.

        Returns:
            Transcribed text

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            Exception: If transcription fails
        """
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language or self.language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )

        # Handle different response formats
        if response_format == "text":
            return transcript
        else:
            # For json, verbose_json, srt, vtt formats
            return transcript.text if hasattr(transcript, "text") else str(transcript)

    def transcribe_audio_data(
        self,
        audio_data: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0,
    ) -> str:
        """Transcribe audio data (bytes) to text.

        Args:
            audio_data: Audio data as bytes
            filename: Filename to use for the audio data (must include extension)
            language: Optional language code to override the default
            prompt: Optional text to guide the model's style
            response_format: Format of the response ("text", "json", "srt", "verbose_json", "vtt")
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Transcribed text

        Raises:
            Exception: If transcription fails
        """
        audio_file = io.BytesIO(audio_data)
        audio_file.name = filename

        transcript = self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            language=language or self.language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )

        # Handle different response formats
        if response_format == "text":
            return transcript
        else:
            return transcript.text if hasattr(transcript, "text") else str(transcript)

    def translate_to_english(
        self,
        audio_file_path: Union[str, Path],
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0,
    ) -> str:
        """Translate audio in any language to English text.

        This uses OpenAI's translation endpoint, which transcribes and translates
        the audio to English in a single step.

        Args:
            audio_file_path: Path to the audio file
            prompt: Optional text to guide the model's style
            response_format: Format of the response ("text", "json", "srt", "verbose_json", "vtt")
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Translated English text

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            Exception: If translation fails
        """
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        with open(audio_path, "rb") as audio_file:
            translation = self.client.audio.translations.create(
                model=self.model,
                file=audio_file,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )

        # Handle different response formats
        if response_format == "text":
            return translation
        else:
            return translation.text if hasattr(translation, "text") else str(translation)
