"""Unit tests for MarkovChain implementation."""

import pytest
import numpy as np
from typing import List, Tuple
from markov_rl.core.markov_chain import MarkovChain

@pytest.fixture
def simple_chain() -> Tuple[List[str], np.ndarray]:
    """Create a simple Markov chain for testing."""
    states = ["A", "B", "C"]
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])
    return states, transition_matrix

@pytest.mark.unit
class TestMarkovChain:
    """Test suite for MarkovChain implementation."""

    def test_initialization_with_matrix(self, simple_chain: Tuple[List[str], np.ndarray]) -> None:
        """Test MarkovChain initialization with transition matrix."""
        states, transition_matrix = simple_chain
        chain = MarkovChain(states, transition_matrix)
        
        assert chain.states == states
        assert np.array_equal(chain.transition_matrix, transition_matrix)

    def test_initialization_without_matrix(self) -> None:
        """Test MarkovChain initialization without transition matrix."""
        states = ["A", "B", "C"]
        chain = MarkovChain(states)
        
        assert chain.states == states
        assert chain.transition_matrix.shape == (3, 3)
        assert np.allclose(chain.transition_matrix, 1/3)  # Uniform distribution

    def test_transition_matrix_validation(self) -> None:
        """Test transition matrix validation."""
        states = ["A", "B"]
        
        # Invalid shapes
        with pytest.raises(ValueError):
            MarkovChain(states, np.array([[0.5]]))  # Too small
        with pytest.raises(ValueError):
            MarkovChain(states, np.array([[0.5, 0.5, 0], [0.5, 0.5, 0]]))  # Wrong shape
            
        # Invalid probabilities
        with pytest.raises(ValueError):
            MarkovChain(states, np.array([[1.5, -0.5], [0.5, 0.5]]))  # Out of range
            
        # Rows don't sum to 1
        with pytest.raises(ValueError):
            MarkovChain(states, np.array([[0.5, 0.3], [0.5, 0.3]]))

    def test_valid_transition_matrix(self, simple_chain: Tuple[List[str], np.ndarray]) -> None:
        """Test transition matrix validation with valid matrix."""
        states, transition_matrix = simple_chain
        chain = MarkovChain(states)
        
        assert chain._is_valid_transition_matrix(transition_matrix)

    def test_eigenvalues_vectors(self, simple_chain: Tuple[List[str], np.ndarray]) -> None:
        """Test eigenvalue and eigenvector computation."""
        states, transition_matrix = simple_chain
        chain = MarkovChain(states, transition_matrix)
        
        eigenvals, eigenvecs = chain.get_eigenvalues_vectors()
        
        # Should have correct shape
        assert len(eigenvals) == len(states)
        assert eigenvecs.shape == (len(states), len(states))
        
        # Should have a eigenvalue close to 1
        assert np.isclose(max(np.abs(eigenvals)), 1.0)
        
        # Verify eigenvector equation
        principal_idx = np.argmax(np.abs(eigenvals))
        principal_eigenvec = eigenvecs[:, principal_idx]
        assert np.allclose(
            transition_matrix @ principal_eigenvec,
            eigenvals[principal_idx] * principal_eigenvec
        )

    def test_stationary_distribution(self, simple_chain: Tuple[List[str], np.ndarray]) -> None:
        """Test stationary distribution computation."""
        states, transition_matrix = simple_chain
        chain = MarkovChain(states, transition_matrix)
        
        # Test with default parameters
        stationary = chain.get_stationary_distribution()
        assert len(stationary) == len(states)
        assert np.isclose(sum(stationary), 1.0)  # Should sum to 1
        assert np.allclose(transition_matrix.T @ stationary, stationary)  # Should be fixed point
        
        # Test with custom initial state
        initial_state = np.array([1.0, 0.0, 0.0])
        stationary_custom = chain.get_stationary_distribution(
            initial_state=initial_state,
            n_iterations=1000
        )
        assert np.allclose(stationary, stationary_custom)  # Should converge to same distribution

    def test_distribution_evolution(self, simple_chain: Tuple[List[str], np.ndarray]) -> None:
        """Test state distribution evolution."""
        states, transition_matrix = simple_chain
        chain = MarkovChain(states, transition_matrix)
        
        n_steps = 10
        distributions = chain.evolve_distribution(n_steps)
        
        # Check shape
        assert distributions.shape == (n_steps, len(states))
        
        # Check that each distribution sums to 1
        assert np.allclose(distributions.sum(axis=1), 1.0)
        
        # Check evolution equation
        for t in range(1, n_steps):
            assert np.allclose(
                distributions[t],
                transition_matrix.T @ distributions[t-1]
            )
        
        # Test with custom initial state
        initial_state = np.array([1.0, 0.0, 0.0])
        distributions_custom = chain.evolve_distribution(
            n_steps,
            initial_state=initial_state
        )
        assert np.array_equal(distributions_custom[0], initial_state)

    def test_convergence_to_stationary(self, simple_chain: Tuple[List[str], np.ndarray]) -> None:
        """Test that distribution evolution converges to stationary distribution."""
        states, transition_matrix = simple_chain
        chain = MarkovChain(states, transition_matrix)
        
        n_steps = 1000
        distributions = chain.evolve_distribution(n_steps)
        stationary = chain.get_stationary_distribution()
        
        # Final distribution should be close to stationary
        assert np.allclose(distributions[-1], stationary, atol=1e-6)
        
        # Convergence should be monotonic
        diffs = np.linalg.norm(distributions - stationary, axis=1)
        assert np.all(diffs[1:] <= diffs[:-1]) 