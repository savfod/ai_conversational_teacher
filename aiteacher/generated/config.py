"""Configuration management for AI conversational teacher."""

from typing import Literal
from pydantic import BaseModel, Field


class LanguageConfig(BaseModel):
    """Configuration for language learning settings."""

    language: str = Field(
        default="Spanish",
        description="Target language to practice (e.g., Spanish, French, German, Japanese)",
    )

    level: Literal["beginner", "intermediate", "advanced"] = Field(
        default="beginner", description="Current proficiency level"
    )

    native_language: str = Field(
        default="English",
        description="User's native language for translations and explanations",
    )


class VoiceConfig(BaseModel):
    """Configuration for voice interface settings."""

    enabled: bool = Field(
        default=False, description="Enable voice interface (hands-free mode)"
    )

    auto_listen: bool = Field(
        default=False, description="Automatically start listening after AI responds"
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
