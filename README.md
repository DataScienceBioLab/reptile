# Reptile: Data Science Project Seed

A robust, standardized foundation for data science projects using Python. This template provides a structured environment for data analysis, machine learning, and visualization tasks.

## Features

- Modern Python data science stack (Pandas, Polars, scikit-learn, etc.)
- Standardized project structure
- Development tools and best practices
- Comprehensive documentation
- Environment management with Poetry

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/reptile.git
   cd reptile
   ```

2. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

5. Start developing:
   ```bash
   # Run tests
   pytest
   
   # Start Jupyter notebook
   jupyter notebook
   ```

## Project Structure

```
reptile/
├── data/                 # Data directory
│   ├── raw/             # Original, immutable data
│   ├── interim/         # Intermediate processing data
│   └── processed/       # Final, analysis-ready data
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── data/           # Data processing scripts
│   ├── features/       # Feature engineering code
│   ├── models/         # Model training and prediction
│   └── visualization/  # Visualization utilities
└── tests/              # Test suite
```

## Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Format code with Black
- Sort imports with isort

### Testing
- Write unit tests with pytest
- Maintain minimum 80% test coverage
- Run tests with `pytest`

### Documentation
- Use Google-style docstrings
- Keep README files up to date
- Document data transformations
- Track analysis steps in notebooks

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by various data science project templates
- Built with modern Python data science tools
- Designed for use with Cursor IDE 