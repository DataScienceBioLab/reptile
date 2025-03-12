# Reptile: Data Science Project Seed Specifications

## Overview

Reptile is a Cursor seed project designed to provide a robust, standardized foundation for data science projects using Python. It serves as a template that can be quickly cloned and customized for various data science tasks while maintaining compatibility with modern data science libraries and best practices.

## Core Objectives

- **Reproducibility**: Ensure consistent environment setup across different machines
- **Modularity**: Enable easy integration of various data science libraries
- **Scalability**: Support projects from small analyses to large-scale data processing
- **Best Practices**: Enforce coding standards and project organization
- **Documentation**: Maintain clear documentation and analysis tracking

## Technical Stack

### Core Dependencies

- **Python Version**: 3.10+ (for modern type hints and features)
- **Package Management**: 
  - Poetry for dependency management
  - pyproject.toml for modern Python packaging

### Primary Libraries

#### Data Processing
- Pandas (2.0+) for traditional DataFrame operations
- Polars (0.20+) for high-performance data processing
- NumPy (1.24+) for numerical operations

#### Machine Learning
- scikit-learn (1.3+) for traditional ML algorithms
- XGBoost (2.0+) for gradient boosting
- LightGBM for high-performance gradient boosting

#### Database Integration
- SQLAlchemy (2.0+) for database operations
- psycopg2-binary for PostgreSQL support
- pymysql for MySQL support

#### Visualization
- Plotly (5.0+) for interactive visualizations
- Matplotlib for static plots
- Seaborn for statistical visualizations

#### Development Tools
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for static type checking
- pytest for testing

## Project Structure

```
reptile/
├── .cursor/              # Cursor-specific configurations
├── .github/              # GitHub Actions and templates
├── data/                 # Data directory (gitignored)
│   ├── raw/             # Original, immutable data
│   ├── interim/         # Intermediate processing data
│   └── processed/       # Final, analysis-ready data
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── data/           # Data processing scripts
│   ├── features/       # Feature engineering code
│   ├── models/         # Model training and prediction
│   └── visualization/  # Visualization utilities
├── tests/              # Test suite
├── .env.example        # Environment variable template
├── .gitignore         
├── pyproject.toml      # Project dependencies and metadata
└── README.md          
```

## Development Standards

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all function definitions
- Document functions and classes using Google-style docstrings
- Maximum line length: 88 characters (Black default)

### Version Control
- Use semantic versioning
- Branch naming convention:
  - feature/[feature-name]
  - bugfix/[bug-name]
  - release/[version]

### Testing
- Minimum 80% test coverage
- Unit tests for all core functionality
- Integration tests for data pipelines
- Property-based testing for data transformations

## Data Management

### Data Storage
- Local data stored in `data/` directory
- Support for remote data sources:
  - S3-compatible storage
  - Database connections
  - API integrations

### Data Processing Guidelines
- Implement data validation checks
- Track data lineage
- Document data transformations
- Handle missing values explicitly

## Documentation Requirements

### Code Documentation
- Function and class docstrings
- Inline comments for complex logic
- Type hints for all functions
- README files in each major directory

### Analysis Documentation
- Jupyter notebooks for exploratory analysis
- Markdown cells explaining analysis steps
- Requirements tracking in issues
- Results documentation in project wiki

## Environment Management

### Local Development
- Poetry for dependency management
- Pre-commit hooks for code quality
- Environment variables via .env files
- Virtual environment isolation

### Reproducibility
- Lock files for exact dependency versions
- Docker support for containerization
- Environment export capabilities
- Seed values for random operations

## Security Considerations

- Secure credential management
- Data encryption at rest
- Access control implementation
- Dependency vulnerability scanning

## Performance Guidelines

- Use Polars for large dataset operations
- Implement proper memory management
- Support for parallel processing
- Caching strategies for repeated operations

## Version History

### 1.0.0 (Initial Release)
- Basic project structure
- Core library integration
- Documentation setup
- Testing framework

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details. 