# Markov and Mouse House

This project combines Markov chain analysis with a mouse house simulation, exploring probabilistic systems and emergent behavior.

## Project Structure

```
markov_mouse/
├── src/          # Source code
│   ├── markov/   # Markov chain implementation
│   └── mouse/    # Mouse house simulation
├── tests/        # Test cases
└── docs/         # Documentation
```

## Components

### Markov Chain Analysis
- Implementation of Markov chains
- State transition analysis
- Probability calculations
- Chain visualization

### Mouse House Simulation
- Mouse behavior modeling
- Environment simulation
- State tracking
- Visualization tools

## Development Guidelines

1. Follow the Python development standards defined in the root `.cursor/rules/` directory
2. Use type hints and docstrings for all functions
3. Include comprehensive test cases
4. Document mathematical models and assumptions
5. Provide visualization examples

## Getting Started

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Run tests:
   ```bash
   poetry run pytest
   ```

3. Run simulations:
   ```bash
   poetry run python -m src.mouse.simulate
   poetry run python -m src.markov.analyze
   ```

## Dependencies

- Python 3.10+
- NumPy for numerical computations
- Matplotlib for visualization
- Pytest for testing
- Poetry for dependency management

## Documentation

Detailed documentation can be found in the `docs/` directory:
- Mathematical models
- Simulation parameters
- Visualization guides
- API reference 