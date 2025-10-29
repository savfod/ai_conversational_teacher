"""Text-to-speech functionality using OpenAI TTS API."""

import io
from pathlib import Path
from typing import Literal, Optional, Union

from openai import OpenAI

# Type aliases for TTS parameters
TTSModel = Literal["tts-1", "tts-1-hd"]
TTSVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
TTSFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class TextToSpeech:
    """Text-to-speech converter using OpenAI's TTS API.

    This class provides an interface to convert text into natural-sounding
    speech using OpenAI's text-to-speech models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: TTSModel = "tts-1",
        voice: TTSVoice = "alloy",
    ):
        """Initialize the text-to-speech converter.

        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY env var.
            model: TTS model to use:
                   - "tts-1": Standard quality, faster, lower latency
                   - "tts-1-hd": High definition quality, slower, higher latency
            voice: Voice to use for speech. Available voices:
                   - "alloy": Neutral, balanced voice
                   - "echo": Male voice
                   - "fable": British accent
                   - "onyx": Deep male voice
                   - "nova": Female voice
                   - "shimmer": Soft female voice
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Optional[TTSVoice] = None,
        response_format: TTSFormat = "mp3",
        speed: float = 1.0,
    ) -> Path:
        """Convert text to speech and save to a file.

        Args:
            text: The text to convert to speech (max 4096 characters)
            output_path: Path where the audio file will be saved
            voice: Voice to use (overrides default if provided)
            response_format: Audio format ("mp3", "opus", "aac", "flac", "wav", "pcm")
            speed: Speed of speech (0.25 to 4.0). 1.0 is normal speed.

        Returns:
            Path to the saved audio file

        Raises:
            ValueError: If text is too long or speed is out of range
            Exception: If synthesis fails
        """
        if len(text) > 4096:
            raise ValueError("Text must be 4096 characters or less")

        if not 0.25 <= speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            response_format=response_format,
            speed=speed,
        )

        # Stream the response to file
        response.stream_to_file(output_path)

        return output_path

    def synthesize_to_bytes(
        self,
        text: str,
        voice: Optional[TTSVoice] = None,
        response_format: TTSFormat = "mp3",
        speed: float = 1.0,
    ) -> bytes:
        """Convert text to speech and return audio data as bytes.

        Args:
            text: The text to convert to speech (max 4096 characters)
            voice: Voice to use (overrides default if provided)
            response_format: Audio format ("mp3", "opus", "aac", "flac", "wav", "pcm")
            speed: Speed of speech (0.25 to 4.0). 1.0 is normal speed.

        Returns:
            Audio data as bytes

        Raises:
            ValueError: If text is too long or speed is out of range
            Exception: If synthesis fails
        """
        if len(text) > 4096:
            raise ValueError("Text must be 4096 characters or less")

        if not 0.25 <= speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            response_format=response_format,
            speed=speed,
        )

        # Read the response content
        return response.content

    def synthesize_streaming(
        self,
        text: str,
        voice: Optional[TTSVoice] = None,
        response_format: TTSFormat = "mp3",
        speed: float = 1.0,
    ):
        """Convert text to speech with streaming response.

        This method returns an iterator that yields audio chunks as they become
        available, allowing for lower latency in playback.

        Args:
            text: The text to convert to speech (max 4096 characters)
            voice: Voice to use (overrides default if provided)
            response_format: Audio format ("mp3", "opus", "aac", "flac", "wav", "pcm")
            speed: Speed of speech (0.25 to 4.0). 1.0 is normal speed.

        Returns:
            Iterator yielding audio data chunks

        Raises:
            ValueError: If text is too long or speed is out of range
            Exception: If synthesis fails
        """
        if len(text) > 4096:
            raise ValueError("Text must be 4096 characters or less")

        if not 0.25 <= speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            response_format=response_format,
            speed=speed,
        )

        # Return the streaming response
        return response.iter_bytes()

    def synthesize_to_stream(
        self,
        text: str,
        voice: Optional[TTSVoice] = None,
        response_format: TTSFormat = "mp3",
        speed: float = 1.0,
    ) -> io.BytesIO:
        """Convert text to speech and return as a BytesIO stream.

        This is useful for in-memory audio processing without saving to disk.

        Args:
            text: The text to convert to speech (max 4096 characters)
            voice: Voice to use (overrides default if provided)
            response_format: Audio format ("mp3", "opus", "aac", "flac", "wav", "pcm")
            speed: Speed of speech (0.25 to 4.0). 1.0 is normal speed.

        Returns:
            BytesIO stream containing audio data

        Raises:
            ValueError: If text is too long or speed is out of range
            Exception: If synthesis fails
        """
        audio_bytes = self.synthesize_to_bytes(
            text=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )

        return io.BytesIO(audio_bytes)


# Voice descriptions for reference
VOICE_DESCRIPTIONS = {
    "alloy": "Neutral and balanced voice, suitable for most content",
    "echo": "Male voice with clear articulation",
    "fable": "British accent, expressive and warm",
    "onyx": "Deep male voice, authoritative",
    "nova": "Female voice, friendly and engaging",
    "shimmer": "Soft female voice, gentle and soothing",
}


def get_voice_description(voice: TTSVoice) -> str:
    """Get a description of a voice.

    Args:
        voice: Voice name

    Returns:
        Description of the voice
    """
    return VOICE_DESCRIPTIONS.get(voice, "Unknown voice")
