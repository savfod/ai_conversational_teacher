.PHONY: help install test lint format typecheck clean pre-commit run

help:
	@echo "Available commands:"
	@echo "  make install     - Install development dependencies"
	@echo "  make test        - Run tests with coverage"
	@echo "  make lint        - Run linter (ruff)"
	@echo "  make format      - Format code with ruff"
	@echo "  make typecheck   - Run type checker (mypy)"
	@echo "  make pre-commit  - Run pre-commit hooks"
	@echo "  make clean       - Remove build artifacts and cache"
	@echo "  make run         - Run the application"

install:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

pre-commit:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ dist/ build/

run:
	python -m src.ai_conversational_teacher
