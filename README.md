# AI Conversational Teacher

Speaking with AI to practice languages - an interactive language learning tool that provides conversational practice with AI-powered error checking, statistics tracking, and vocabulary export.

## Features

### ✅ Implemented (Stub/Foundation)

- **Voice Interface**: Hands-free speaking mode (stub implementation ready for integration)
  - Enable/disable voice input
  - Auto-listen after AI responses
  - Configurable voice activation threshold
  
- **Configuration System**: Comprehensive settings for personalized learning
  - Language selection (Spanish, French, German, Japanese, etc.)
  - Proficiency level (beginner, intermediate, advanced)
  - Native language for explanations
  - Voice and statistics preferences
  
- **Error Checking with Structured LLM Output**: Detailed grammar and error analysis
  - Grammar error detection with structured output
  - Error categorization by type and severity
  - Contextual explanations in native language
  - Fluency scoring
  
- **Statistics Tracking**: Comprehensive progress monitoring
  - Session-based tracking
  - Error statistics by type and severity
  - Vocabulary tracking by difficulty
  - Historical session data
  
- **Anki Export**: Build vocabulary decks for spaced repetition
  - Export vocabulary to Anki-compatible CSV format
  - Automatic tagging by language and difficulty
  - Context included in flashcards

## Installation

1. Clone the repository:
```bash
git clone https://github.com/savfod/ai_conversational_teacher.git
cd ai_conversational_teacher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Quick Start

Run with default settings:
```bash
python run.py
```

Or use the CLI for more options:
```bash
python cli.py start
```

### Create Custom Configuration

Interactive configuration setup:
```bash
python cli.py config --create --output my_config.json
```

Start with custom configuration:
```bash
python cli.py start --config my_config.json
```

### Export Vocabulary to Anki

After a practice session:
```bash
python cli.py export --stats-file statistics.json --output vocabulary.csv
```

Then import the CSV file into Anki using File → Import.

### View Statistics

```bash
python cli.py stats --stats-file statistics.json
```

## Configuration

Configuration is managed through JSON files. Example configuration:

```json
{
  "language": {
    "language": "Spanish",
    "level": "intermediate",
    "native_language": "English"
  },
  "voice": {
    "enabled": false,
    "auto_listen": false,
    "voice_activation_threshold": 0.5
  },
  "statistics": {
    "track_errors": true,
    "track_vocabulary": true,
    "export_to_anki": true
  },
  "openai_api_key": ""
}
```

## Architecture

The application consists of several modular components:

- **config.py**: Configuration management with Pydantic models
- **voice_interface.py**: Voice input/output handling (stub)
- **error_checker.py**: Grammar and error checking with structured LLM output
- **statistics.py**: Session and progress tracking
- **anki_exporter.py**: Vocabulary export to Anki format
- **main.py**: Main application logic and conversation loop
- **cli.py**: Command-line interface

## Current Implementation Status

### Fully Implemented (Stub/Ready for Integration)
- ✅ Configuration system with all settings
- ✅ Voice interface stub (ready for speech_recognition integration)
- ✅ Error checker structure with Pydantic models
- ✅ Statistics tracking with persistence
- ✅ Anki CSV export functionality
- ✅ CLI interface for all features
- ✅ Session management

### Requires OpenAI API Integration
- ⚠️ Actual LLM-based error checking (currently returns stub data)
- ⚠️ AI response generation (currently returns placeholder)
- ⚠️ Contextual conversation (structure ready)

### Optional Enhancements
- 🔄 Voice recognition integration (requires speech_recognition + pyaudio)
- 🔄 Text-to-speech integration (requires pyttsx3 or gTTS)
- 🔄 Anki .apkg package export (requires genanki)

## Example Session

```
============================================================
Welcome to AI Conversational Teacher!
============================================================
Session ID: session_20251029_120000
Language: Spanish
Level: beginner
Voice Interface: Disabled
============================================================

Start practicing! (Type 'quit', 'exit', or 'q' to end session)

You: Hola, como estas?

AI: [AI Response - STUB MODE]
Thank you for practicing Spanish with me! In a full implementation, 
I would respond contextually to your message.
Your message: 'Hola, como estas?'

============================================================
Session Summary
============================================================
Messages exchanged: 1
Total errors: 0
Words learned: 0
============================================================
```

## Development

### Adding Real LLM Integration

To integrate with OpenAI's API, update `error_checker.py`:

```python
import openai

def check_errors(self, user_input: str, context: str = "") -> ErrorCheckResult:
    prompt = self.get_correction_prompt(user_input)
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result_json = response.choices[0].message.content
    return ErrorCheckResult.model_validate_json(result_json)
```

### Adding Voice Recognition

To add real voice recognition, install dependencies:

```bash
pip install SpeechRecognition pyaudio
```

Then update `voice_interface.py`:

```python
import speech_recognition as sr

def get_voice_input(self, timeout: int = 10) -> Optional[str]:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=timeout)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return None
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.
