"""Educational app for learning about probability distributions."""

__version__ = "0.1.0"

from .app import main
from .utils.data_utils import (
    generate_distribution_data,
    calculate_distribution_stats,
    get_distribution_info
)

__all__ = [
    "main",
    "generate_distribution_data",
    "calculate_distribution_stats",
    "get_distribution_info"
] 