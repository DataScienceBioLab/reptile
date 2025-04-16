# Contributing to Reptile

Thank you for your interest in contributing to Reptile! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/reptile.git
   cd reptile
   ```
3. Install Poetry and dependencies:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   poetry install
   ```
4. Set up pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards:
   - Use type hints
   - Follow PEP 8 guidelines
   - Write docstrings for public functions
   - Add tests for new functionality

3. Run tests and linting:
   ```bash
   poetry run pytest
   poetry run black .
   poetry run isort .
   poetry run flake8
   poetry run mypy .
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

## Code Standards

### Python Code
- Use Python 3.10+ features
- Follow PEP 8 guidelines
- Use type hints for all function definitions
- Write docstrings using Google style
- Maximum line length: 88 characters (Black default)

### Testing
- Write unit tests for all new functionality
- Maintain minimum 80% test coverage
- Use pytest for testing
- Include both positive and negative test cases

### Documentation
- Update README.md if needed
- Document new features
- Include examples in docstrings
- Keep documentation up to date

### Git Commits
- Use semantic commit messages:
  - feat: New feature
  - fix: Bug fix
  - docs: Documentation changes
  - style: Code style changes
  - refactor: Code refactoring
  - test: Test updates
  - chore: Maintenance tasks

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if needed
3. Ensure all tests pass
4. Request review from maintainers

## Questions?

Feel free to open an issue for any questions or concerns about contributing to Reptile. 