# AI Conversational Teacher

Speaking with AI to practice languages - an interactive language learning tool that provides conversational practice with AI-powered error checking.

Planned: statistics tracking, and vocabulary export.

## Features

### ✅ Implemented, ❌ Not Implemented, but planned

- **Voice Interface**: Hands-free speaking mode
  - ✅ Enable/disable voice input with "start" and "stop stop" (twice for reliability) commands

- **Configuration System**: Comprehensive settings for personalized learning
  - ✅ Language (use 'uv run conversa CODE', where CODE is a language code from ISO 639-1, [supported](https://platform.openai.com/docs/guides/speech-to-text/supported-languages#supported-languages) by OpenAI)
  - ❌ Proficiency level (beginner, intermediate, advanced)
  - ❌ Native language for explanations

- **Error Checking with Structured LLM Output**: Detailed grammar and error analysis
  - ✅ Grammar error detection
  - ✅ Contextual explanations in native language

- **Statistics**:
  - ❌ Error statistics by type and severity

- **Anki Export**: Build vocabulary decks for spaced repetition
  - ❌ Export vocabulary to Anki-compatible CSV format


## Installation

1. Clone the repository:
```bash
git clone https://github.com/savfod/ai_conversational_teacher.git
cd ai_conversational_teacher
```

2. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

or write in the .env file.


## Usage

### Quick Start

Run with default settings:

```bash
uv run main.py [language]
```


Replace `[language]` with the desired language code (e.g., 'es' for Spanish).
Say "start" to begin voice input and "stop stop" to end it.
