"""Configuration management for AI conversational teacher."""

from typing import Literal

from pydantic import BaseModel, Field


class LanguageConfig(BaseModel):
    """Configuration for language learning settings."""

    language: str = Field(
        default="Spanish",
        description="Target language to practice (e.g., English, Spanish, French, German, Japanese)",
    )

    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        default="A2", description="Current proficiency level"
    )

    native_language: str = Field(
        default="English",
        description="User's native language for translations and explanations",
    )


class VoiceConfig(BaseModel):
    """Configuration for voice interface settings."""

    enabled: bool = Field(
        default=True, description="Enable voice interface (hands-free mode)"
    )

    auto_listen: bool = Field(
        default=True, description="Automatically start listening after AI responds"
    )

    voice_activation_threshold: float = Field(
        default=0.5, description="Voice activation detection threshold (0.0-1.0)"
    )


class StatisticsConfig(BaseModel):
    """Configuration for statistics tracking."""

    track_errors: bool = Field(
        default=True, description="Track errors made during practice"
    )

    track_vocabulary: bool = Field(
        default=True, description="Track new vocabulary encountered"
    )

    export_to_anki: bool = Field(
        default=True, description="Enable Anki export functionality"
    )


class AppConfig(BaseModel):
    """Main application configuration."""

    language: LanguageConfig = Field(default_factory=LanguageConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)

    openai_api_key: str = Field(
        default="", description="OpenAI API key for LLM functionality"
    )

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_file(cls, filepath: str) -> "AppConfig":
        """Load configuration from a JSON file."""
        with open(filepath, "r") as f:
            return cls.model_validate_json(f.read())


def create_config_interactive() -> AppConfig:
    """Create configuration interactively."""
    print("\nConfiguration Setup")
    print("=" * 60)

    # Language settings
    language = (
        input("Target language (e.g., Spanish, French, German) [Spanish]: ").strip()
        or "Spanish"
    )
    level = (
        input("Your level (beginner/intermediate/advanced) [beginner]: ").strip()
        or "beginner"
    )
    native_language = input("Your native language [English]: ").strip() or "English"

    # Voice settings
    voice_enabled_input = input("Enable voice interface? (y/n) [n]: ").strip().lower()
    voice_enabled = voice_enabled_input in ["y", "yes"]

    auto_listen = False
    if voice_enabled:
        auto_listen_input = (
            input("Auto-listen after AI responds? (y/n) [n]: ").strip().lower()
        )
        auto_listen = auto_listen_input in ["y", "yes"]

    # Statistics settings
    track_errors_input = input("Track errors? (y/n) [y]: ").strip().lower()
    track_errors = track_errors_input not in ["n", "no"]

    track_vocabulary_input = input("Track vocabulary? (y/n) [y]: ").strip().lower()
    track_vocabulary = track_vocabulary_input not in ["n", "no"]

    export_to_anki_input = input("Enable Anki export? (y/n) [y]: ").strip().lower()
    export_to_anki = export_to_anki_input not in ["n", "no"]

    # OpenAI API key (optional for now)
    api_key = input("OpenAI API key (optional, press Enter to skip): ").strip()

    # Create config
    config = AppConfig(
        language=LanguageConfig(
            language=language, level=level, native_language=native_language
        ),
        voice=VoiceConfig(enabled=voice_enabled, auto_listen=auto_listen),
        statistics=StatisticsConfig(
            track_errors=track_errors,
            track_vocabulary=track_vocabulary,
            export_to_anki=export_to_anki,
        ),
        openai_api_key=api_key,
    )

    print("\nConfiguration created successfully!")
    return config

