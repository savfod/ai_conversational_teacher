"""
Example: Creating and using a custom configuration
"""

from config import AppConfig, LanguageConfig, VoiceConfig, StatisticsConfig

# Create a custom configuration
config = AppConfig(
    language=LanguageConfig(
        language="Japanese", level="beginner", native_language="English"
    ),
    voice=VoiceConfig(enabled=True, auto_listen=True, voice_activation_threshold=0.6),
    statistics=StatisticsConfig(
        track_errors=True, track_vocabulary=True, export_to_anki=True
    ),
    openai_api_key="your-api-key-here",
)

# Save configuration
config.save_to_file("japanese_config.json")
print("Configuration saved to japanese_config.json")

# Load and use configuration
loaded_config = AppConfig.load_from_file("japanese_config.json")
print(
    f"Loaded config: {loaded_config.language.language} at {loaded_config.language.level} level"
)
