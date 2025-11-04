# Contributing to AI Conversational Teacher

Thank you for your interest in contributing to AI Conversational Teacher! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see README.md)
4. Create a new branch for your changes

## Development Workflow

### Before Making Changes

1. Ensure all tests pass: `pytest`
2. Verify code quality: `ruff check .`
3. Check types: `mypy src`

### Making Changes

1. Write clear, concise commit messages
2. Follow the existing code style
3. Add tests for new functionality
4. Update documentation as needed
5. Run pre-commit hooks: `pre-commit run --all-files`

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and small
- Maximum line length: 100 characters

### Testing

- Write tests for all new features
- Maintain or improve test coverage
- Use descriptive test names
- Test edge cases and error conditions

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep first line under 72 characters
- Reference issues and pull requests when relevant

## Submitting Changes

1. Push your changes to your fork
2. Submit a pull request to the main repository
3. Describe your changes in detail
4. Link any related issues
5. Wait for review and address feedback

## Code Review Process

- All submissions require review
- Maintainers may request changes
- Once approved, changes will be merged
- Continuous integration must pass

## Questions?

Feel free to open an issue for any questions or concerns.

## License

By contributing, you agree that your contributions will be licensed under the GNU Affero General Public License v3.0.
