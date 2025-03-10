"""Unit tests for matrix operation utilities."""

import pytest
import numpy as np
from markov_rl.utils.matrix_ops import (
    is_stochastic_matrix,
    normalize_matrix_rows,
    is_symmetric_matrix,
    is_positive_definite,
    make_symmetric,
    clip_probability_matrix
)

@pytest.mark.unit
class TestMatrixOperations:
    """Test suite for matrix operation utilities."""

    def test_stochastic_matrix_validation_valid(self) -> None:
        """Test stochastic matrix validation with valid matrices."""
        # Simple 2x2 stochastic matrix
        matrix = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        assert is_stochastic_matrix(matrix)

        # Identity matrix is stochastic
        assert is_stochastic_matrix(np.eye(3))

        # Uniform transition matrix
        assert is_stochastic_matrix(np.ones((4, 4)) / 4)

    def test_stochastic_matrix_validation_invalid(self) -> None:
        """Test stochastic matrix validation with invalid matrices."""
        # Negative probabilities
        matrix = np.array([
            [1.1, -0.1],
            [0.5, 0.5]
        ])
        assert not is_stochastic_matrix(matrix)

        # Probabilities > 1
        matrix = np.array([
            [1.2, 0.3],
            [0.4, 0.6]
        ])
        assert not is_stochastic_matrix(matrix)

        # Row sums != 1
        matrix = np.array([
            [0.7, 0.2],
            [0.4, 0.4]
        ])
        assert not is_stochastic_matrix(matrix)

        # Not a numpy array
        assert not is_stochastic_matrix([[0.5, 0.5], [0.5, 0.5]])

    def test_stochastic_matrix_validation_edge_cases(self) -> None:
        """Test stochastic matrix validation with edge cases."""
        # Single state
        assert is_stochastic_matrix(np.array([[1.0]]))

        # Zero probabilities
        matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        assert is_stochastic_matrix(matrix)

        # Close to 1 sum (floating point precision)
        matrix = np.array([
            [0.333333333, 0.333333333, 0.333333334],
            [0.5, 0.4, 0.1]
        ])
        assert is_stochastic_matrix(matrix)

    def test_matrix_normalization_valid(self) -> None:
        """Test matrix row normalization with valid inputs."""
        # Simple matrix
        matrix = np.array([
            [1.0, 2.0],
            [3.0, 3.0]
        ])
        normalized = normalize_matrix_rows(matrix)
        assert np.allclose(normalized.sum(axis=1), 1.0)
        assert np.allclose(
            normalized,
            np.array([
                [1/3, 2/3],
                [0.5, 0.5]
            ])
        )

        # Already normalized
        stochastic = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        assert np.array_equal(normalize_matrix_rows(stochastic), stochastic)

        # Different scales
        matrix = np.array([
            [10.0, 10.0],
            [1.0, 3.0]
        ])
        normalized = normalize_matrix_rows(matrix)
        assert np.allclose(normalized.sum(axis=1), 1.0)
        assert np.allclose(
            normalized,
            np.array([
                [0.5, 0.5],
                [0.25, 0.75]
            ])
        )

    def test_matrix_normalization_invalid(self) -> None:
        """Test matrix row normalization with invalid inputs."""
        # Zero-sum row
        matrix = np.array([
            [1.0, 1.0],
            [0.0, 0.0]
        ])
        with pytest.raises(ValueError):
            normalize_matrix_rows(matrix)

        # All zeros
        with pytest.raises(ValueError):
            normalize_matrix_rows(np.zeros((2, 2)))

    def test_matrix_normalization_edge_cases(self) -> None:
        """Test matrix row normalization with edge cases."""
        # Single element
        assert np.array_equal(
            normalize_matrix_rows(np.array([[2.0]])),
            np.array([[1.0]])
        )

        # Very small values
        matrix = np.array([
            [1e-10, 2e-10],
            [3e-10, 3e-10]
        ])
        normalized = normalize_matrix_rows(matrix)
        assert np.allclose(normalized.sum(axis=1), 1.0)

        # Very large values
        matrix = np.array([
            [1e10, 2e10],
            [3e10, 3e10]
        ])
        normalized = normalize_matrix_rows(matrix)
        assert np.allclose(normalized.sum(axis=1), 1.0)

    def test_symmetric_matrix_validation_valid(self) -> None:
        """Test symmetric matrix validation with valid matrices."""
        # Identity matrix is symmetric
        assert is_symmetric_matrix(np.eye(3))
        
        # Symmetric 2x2 matrix
        matrix = np.array([
            [1.0, 0.5],
            [0.5, 2.0]
        ])
        assert is_symmetric_matrix(matrix)
        
        # Zero matrix is symmetric
        assert is_symmetric_matrix(np.zeros((4, 4)))
        
        # Matrix with small numerical differences
        matrix = np.array([
            [1.0, 0.1 + 1e-10],
            [0.1, 2.0]
        ])
        assert is_symmetric_matrix(matrix, atol=1e-9)

    def test_symmetric_matrix_validation_invalid(self) -> None:
        """Test symmetric matrix validation with invalid matrices."""
        # Non-symmetric matrix
        matrix = np.array([
            [1.0, 0.5],
            [0.6, 2.0]
        ])
        assert not is_symmetric_matrix(matrix)
        
        # Non-square matrix
        matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 2.0, 0.1]
        ])
        assert not is_symmetric_matrix(matrix)
        
        # Not a numpy array
        assert not is_symmetric_matrix([[1.0, 0.5], [0.5, 1.0]])

    def test_positive_definite_validation(self) -> None:
        """Test positive definite matrix validation."""
        # Positive definite matrix
        matrix = np.array([
            [2.0, 0.5],
            [0.5, 2.0]
        ])
        assert is_positive_definite(matrix)
        
        # Identity matrix is positive definite
        assert is_positive_definite(np.eye(3))
        
        # Not positive definite (eigenvalue = 0)
        matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        assert not is_positive_definite(matrix)
        
        # Not positive definite (negative eigenvalue)
        matrix = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])
        assert not is_positive_definite(matrix)
        
        # Not symmetric
        matrix = np.array([
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        assert not is_positive_definite(matrix)

    def test_make_symmetric(self) -> None:
        """Test making matrices symmetric."""
        # Already symmetric
        matrix = np.array([
            [1.0, 0.5],
            [0.5, 2.0]
        ])
        result = make_symmetric(matrix)
        assert np.array_equal(result, matrix)
        
        # Make non-symmetric matrix symmetric
        matrix = np.array([
            [1.0, 0.2],
            [0.8, 2.0]
        ])
        result = make_symmetric(matrix)
        assert is_symmetric_matrix(result)
        assert np.allclose(result, np.array([
            [1.0, 0.5],
            [0.5, 2.0]
        ]))
        
        # Non-square matrix
        with pytest.raises(ValueError):
            make_symmetric(np.array([[1.0, 0.5, 0.3], [0.5, 2.0, 0.1]]))

    def test_clip_probability_matrix(self) -> None:
        """Test clipping and normalizing probability matrices."""
        # Matrix with values outside [0,1]
        matrix = np.array([
            [1.2, -0.1],
            [0.8, 1.5]
        ])
        result = clip_probability_matrix(matrix)
        assert is_stochastic_matrix(result)
        
        # Test custom bounds
        matrix = np.array([
            [0.0, 0.1],
            [0.2, 0.0]
        ])
        result = clip_probability_matrix(matrix, min_value=0.1, max_value=0.9)
        assert np.all(result >= 0.1)
        assert np.all(result <= 0.9)
        assert np.allclose(result.sum(axis=1), 1.0)
        
        # Invalid bounds
        with pytest.raises(ValueError):
            clip_probability_matrix(matrix, min_value=0.5, max_value=0.3)
        
        with pytest.raises(ValueError):
            clip_probability_matrix(matrix, min_value=-0.1, max_value=1.1) 