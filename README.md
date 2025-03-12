# Interactive Educational Form

An interactive educational web application built with Streamlit that helps users learn about probability distributions through visualization and hands-on exploration.

## Features

- Interactive form for user input and customization
- Real-time visualization of probability distributions
- Dynamic parameter adjustment
- Educational content and explanations
- Interactive quizzes
- Statistical analysis and insights

## Supported Distributions

- Normal (Gaussian) Distribution
- Uniform Distribution
- Exponential Distribution

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

1. Activate the virtual environment:
```bash
poetry shell
```

2. Run the Streamlit app:
```bash
streamlit run src/app.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Project Structure

```
.
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── components/         # Reusable UI components
│   ├── styles/            # CSS and styling
│   └── utils/             # Utility functions
│       └── data_utils.py  # Data generation and processing
├── data/                  # Data directory
├── docs/                  # Documentation
├── tests/                # Test suite
├── pyproject.toml        # Project dependencies
└── README.md            # Project documentation
```

## Dependencies

- Python 3.10+
- Streamlit 1.32.0
- Plotly 5.19.0
- NumPy 1.24.0
- Pandas 2.0.0
- Poetry for dependency management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details. 