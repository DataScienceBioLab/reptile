"""Utility functions for matrix operations.

This module provides utility functions for working with matrices, including:
- Probability matrix validation
- Matrix normalization
- Special matrix properties
- Symmetric matrix operations
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

def is_stochastic_matrix(matrix: npt.NDArray) -> bool:
    """Check if a matrix is a valid stochastic matrix.

    A stochastic matrix must have:
    1. All elements between 0 and 1
    2. Each row sum equal to 1

    Args:
        matrix: Matrix to validate

    Returns:
        True if matrix is stochastic, False otherwise
    """
    if not isinstance(matrix, np.ndarray):
        return False
    
    # Check values between 0 and 1
    if not np.all((matrix >= 0) & (matrix <= 1)):
        return False
    
    # Check row sums equal to 1 (within numerical precision)
    row_sums = np.sum(matrix, axis=1)
    return np.allclose(row_sums, 1.0)

def normalize_matrix_rows(matrix: npt.NDArray) -> npt.NDArray:
    """Normalize matrix rows to sum to 1.

    Args:
        matrix: Matrix to normalize

    Returns:
        Normalized matrix where each row sums to 1

    Raises:
        ValueError: If matrix contains a row of all zeros
    """
    row_sums = np.sum(matrix, axis=1)
    if np.any(row_sums == 0):
        raise ValueError("Cannot normalize matrix with zero-sum rows")
    return matrix / row_sums[:, np.newaxis]

def is_symmetric_matrix(matrix: npt.NDArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """Check if a matrix is symmetric.

    A matrix is symmetric if it equals its transpose: A = A^T

    Args:
        matrix: Matrix to check
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Returns:
        True if matrix is symmetric, False otherwise
    """
    if not isinstance(matrix, np.ndarray):
        return False
    
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

def is_positive_definite(matrix: npt.NDArray) -> bool:
    """Check if a matrix is positive definite.

    A symmetric matrix is positive definite if all its eigenvalues are positive.

    Args:
        matrix: Matrix to check

    Returns:
        True if matrix is positive definite, False otherwise
    """
    if not is_symmetric_matrix(matrix):
        return False
    
    try:
        # Try Cholesky decomposition (faster than eigenvalue computation)
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def make_symmetric(matrix: npt.NDArray) -> npt.NDArray:
    """Make a matrix symmetric by averaging with its transpose.

    The resulting matrix A' = (A + A^T)/2 is guaranteed to be symmetric.

    Args:
        matrix: Matrix to make symmetric

    Returns:
        Symmetric matrix

    Raises:
        ValueError: If input matrix is not square
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cannot make non-square matrix symmetric")
    
    return (matrix + matrix.T) / 2

def clip_probability_matrix(
    matrix: npt.NDArray,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> npt.NDArray:
    """Clip matrix values to probability range and renormalize rows.

    This is useful for numerical stability in probability calculations.

    Args:
        matrix: Matrix to clip and normalize
        min_value: Minimum allowed value (default: 0.0)
        max_value: Maximum allowed value (default: 1.0)

    Returns:
        Clipped and normalized matrix

    Raises:
        ValueError: If min_value >= max_value or if values are outside [0,1]
    """
    if not 0 <= min_value < max_value <= 1:
        raise ValueError(
            f"Invalid probability bounds: [{min_value}, {max_value}]"
        )
    
    # Clip values to bounds
    clipped = np.clip(matrix, min_value, max_value)
    
    # Normalize rows
    return normalize_matrix_rows(clipped) 