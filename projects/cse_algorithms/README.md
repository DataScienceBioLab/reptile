# CSE Algorithms

This directory contains implementations of algorithms and data structures for the CSE Algorithms class.

## Structure

```
cse_algorithms/
├── problems/     # Individual algorithm problem implementations
└── shared/       # Shared utilities and common code
```

## Project Organization

Each algorithm problem should be implemented in its own directory under `problems/` with the following structure:

```
problems/problem_name/
├── src/          # Source code
├── tests/        # Test cases
├── README.md     # Problem description and solution approach
└── pyproject.toml # Project dependencies
```

## Development Guidelines

1. Follow the Python development standards defined in the root `.cursor/rules/` directory
2. Use type hints and docstrings for all functions
3. Include comprehensive test cases
4. Document time and space complexity
5. Provide example usage in README.md

## Getting Started

1. Create a new directory under `problems/` for your algorithm
2. Copy the template from `shared/template/` if available
3. Implement your solution following the project standards
4. Add tests and documentation
5. Submit your solution

## Shared Resources

The `shared/` directory contains:
- Common utility functions
- Testing frameworks
- Project templates
- Shared documentation 