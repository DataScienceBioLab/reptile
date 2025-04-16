"""Utility functions for data generation and processing."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

def generate_distribution_data(
    distribution: str,
    num_points: int,
    params: Dict[str, float] = None
) -> Tuple[np.ndarray, str]:
    """Generate data points for a given distribution.

    Args:
        distribution: Type of distribution ("Normal", "Uniform", or "Exponential")
        num_points: Number of data points to generate
        params: Optional parameters for the distribution

    Returns:
        Tuple of (data array, distribution title)
    """
    if params is None:
        params = {}

    if distribution == "Normal":
        mu = params.get("mu", 0)
        sigma = params.get("sigma", 1)
        data = np.random.normal(mu, sigma, num_points)
        title = f"Normal Distribution (μ={mu}, σ={sigma})"
    elif distribution == "Uniform":
        low = params.get("low", -3)
        high = params.get("high", 3)
        data = np.random.uniform(low, high, num_points)
        title = f"Uniform Distribution [{low}, {high}]"
    else:  # Exponential
        scale = params.get("scale", 1)
        data = np.random.exponential(scale, num_points)
        title = f"Exponential Distribution (λ={1/scale})"

    return data, title

def calculate_distribution_stats(data: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for the distribution.

    Args:
        data: Array of numerical data

    Returns:
        Dictionary containing basic statistics
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data))
    }

def get_distribution_info(distribution: str) -> Dict[str, str]:
    """Get educational information about a distribution.

    Args:
        distribution: Type of distribution

    Returns:
        Dictionary containing educational content
    """
    info = {
        "Normal": {
            "description": """
            The Normal (or Gaussian) distribution is a fundamental probability distribution
            that appears frequently in various natural phenomena.
            """,
            "characteristics": [
                "Symmetric, bell-shaped curve",
                "Mean, median, and mode are all equal",
                "Approximately 68% of data falls within one standard deviation"
            ],
            "examples": [
                "Height and weight distributions in populations",
                "Measurement errors in scientific experiments",
                "IQ scores and other standardized tests"
            ]
        },
        "Uniform": {
            "description": """
            The Uniform distribution represents a constant probability across all values
            in a given range.
            """,
            "characteristics": [
                "Equal probability for all values in the range",
                "Rectangular shape",
                "Mean is the average of the minimum and maximum values"
            ],
            "examples": [
                "Random number generators",
                "Rounding errors in computations",
                "Time of arrival within a fixed interval"
            ]
        },
        "Exponential": {
            "description": """
            The Exponential distribution models the time between events in a Poisson process.
            """,
            "characteristics": [
                "Always non-negative",
                "Memoryless property",
                "Decreasing probability density"
            ],
            "examples": [
                "Time between customer arrivals",
                "Length of phone calls",
                "Time until equipment failure"
            ]
        }
    }
    return info.get(distribution, {}) 