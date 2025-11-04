# AI Conversational Teacher

Speaking with AI to practice language - an interactive tool for language learning through conversation.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/savfod/ai_conversational_teacher.git
cd ai_conversational_teacher
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
pytest
```

### Code Quality

This project uses several tools to maintain code quality:

- **Ruff**: Fast Python linter and formatter
- **mypy**: Static type checker
- **pytest**: Testing framework
- **pre-commit**: Git hooks for automated checks

To run linting manually:
```bash
ruff check .
ruff format .
```

To run type checking:
```bash
mypy src
```

### Running the Application

```bash
python -m src.ai_conversational_teacher
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.
