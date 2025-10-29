# Contributing to AI Conversational Teacher

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ai_conversational_teacher.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`

## Development Setup

### Prerequisites
- Python 3.8 or higher
- pip for package management

### Installing Development Dependencies
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
# Test individual modules
python -c "from config import AppConfig; print('Config OK')"
python -c "from statistics import StatisticsTracker; print('Statistics OK')"

# Or create comprehensive tests
python test_app.py  # if test file exists
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep functions focused and concise

## Project Structure

```
ai_conversational_teacher/
├── config.py              # Configuration management
├── voice_interface.py     # Voice input/output (stub)
├── error_checker.py       # Error checking with LLM
├── statistics.py          # Statistics tracking
├── anki_exporter.py       # Anki export functionality
├── main.py                # Main application logic
├── cli.py                 # Command-line interface
├── run.py                 # Simple runner script
└── example_*.py           # Usage examples
```

## Areas for Contribution

### High Priority
1. **OpenAI API Integration**
   - Implement actual LLM calls in `error_checker.py`
   - Add conversation generation in `main.py`
   - Add structured output parsing

2. **Voice Recognition**
   - Integrate speech_recognition library
   - Add real-time voice input in `voice_interface.py`
   - Implement text-to-speech output

3. **Testing**
   - Add comprehensive unit tests
   - Add integration tests
   - Add end-to-end tests

### Medium Priority
4. **UI/UX Improvements**
   - Add a web interface (Flask/FastAPI)
   - Add a GUI (tkinter/PyQt)
   - Improve CLI experience

5. **Additional Features**
   - Support for more languages
   - Pronunciation feedback
   - Progress visualization
   - Anki .apkg export (using genanki)

### Low Priority
6. **Documentation**
   - Add more examples
   - Create video tutorials
   - Improve API documentation

## Making Changes

### For New Features
1. Discuss the feature in an issue first
2. Create a branch from `main`
3. Implement the feature with tests
4. Update documentation
5. Submit a pull request

### For Bug Fixes
1. Create an issue describing the bug
2. Create a branch from `main`
3. Fix the bug with a test that proves the fix
4. Submit a pull request

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update examples if API changes
3. Ensure all tests pass
4. Follow the code style guidelines
5. Write clear commit messages
6. Reference related issues in your PR description

## Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat: Add OpenAI API integration for error checking

Implement actual LLM calls using OpenAI's structured output
feature to parse errors and provide feedback.

Closes #123
```

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

Feel free to open an issue for any questions or clarifications needed.

Thank you for contributing! 🎉
