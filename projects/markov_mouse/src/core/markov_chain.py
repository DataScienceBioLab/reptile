"""Core implementation of Markov Chain functionality.

This module provides the fundamental implementation of Markov Chains, including:
- Basic Markov Chain class
- Transition matrix operations
- State space management
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from numpy import linalg as LA

class MarkovChain:
    """A class representing a finite Markov Chain.

    This class implements the core functionality for working with finite Markov chains,
    including state transitions, probability calculations, and chain properties.

    Attributes:
        states (List[str]): List of possible states in the Markov chain
        transition_matrix (np.ndarray): Matrix of transition probabilities
    """

    def __init__(self, states: List[str], transition_matrix: Optional[npt.NDArray] = None) -> None:
        """Initialize a Markov Chain.

        Args:
            states: List of state names
            transition_matrix: Optional transition probability matrix.
                If None, initializes with uniform probabilities.

        Raises:
            ValueError: If transition matrix is invalid
        """
        self.states = states
        self.n_states = len(states)
        self._validate_and_set_transition_matrix(transition_matrix)

    def _validate_and_set_transition_matrix(self, matrix: Optional[npt.NDArray]) -> None:
        """Validate and set the transition matrix.

        Args:
            matrix: Transition probability matrix to validate and set

        Raises:
            ValueError: If matrix shape or values are invalid
        """
        if matrix is None:
            # Initialize with uniform probabilities
            self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
            return

        # Convert to numpy array if needed
        matrix = np.asarray(matrix)

        # Check shape
        if matrix.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"Transition matrix shape {matrix.shape} does not match "
                f"number of states {self.n_states}"
            )

        # Validate probabilities
        if not self._is_valid_transition_matrix(matrix):
            raise ValueError(
                "Invalid transition matrix. Probabilities must be between "
                "0 and 1, and rows must sum to 1."
            )

        self.transition_matrix = matrix

    def _is_valid_transition_matrix(self, matrix: np.ndarray) -> bool:
        """Check if matrix is a valid transition matrix.
        
        Args:
            matrix: Matrix to validate
            
        Returns:
            bool: True if matrix is valid transition matrix
        """
        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            return False
            
        # Check if probabilities are between 0 and 1
        if not np.all((matrix >= 0) & (matrix <= 1)):
            return False
            
        # Check if rows sum to 1
        row_sums = matrix.sum(axis=1)
        return np.allclose(row_sums, 1.0)

    def get_eigenvalues_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of transition matrix.
        
        Returns:
            Tuple containing:
            - Array of eigenvalues
            - Array of eigenvectors as columns
        """
        return LA.eig(self.transition_matrix)

    def get_stationary_distribution(
        self,
        n_iterations: int = 100,
        initial_state: Optional[np.ndarray] = None,
        tol: float = 1e-8
    ) -> np.ndarray:
        """Find stationary distribution by iteration or from principal eigenvector.
        
        Args:
            n_iterations: Number of iterations for power method
            initial_state: Initial state distribution (uniform if None)
            tol: Convergence tolerance
            
        Returns:
            Stationary distribution vector
        """
        if initial_state is None:
            initial_state = np.ones(self.n_states) / self.n_states
        else:
            # Validate initial state
            if len(initial_state) != self.n_states:
                raise ValueError("Initial state length must match number of states")
            if not np.isclose(sum(initial_state), 1.0):
                raise ValueError("Initial state must sum to 1")
            if not np.all(initial_state >= 0):
                raise ValueError("Initial state probabilities must be non-negative")
            
        state = initial_state.copy()
        for _ in range(n_iterations):
            next_state = self.transition_matrix.T @ state
            if np.linalg.norm(next_state - state) < tol:
                break
            state = next_state
            
        return state

    def evolve_distribution(
        self,
        n_steps: int,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Evolve state distribution over time.
        
        Args:
            n_steps: Number of time steps
            initial_state: Initial state distribution (uniform if None)
            
        Returns:
            Array of shape (n_steps, n_states) containing distribution at each step

        Raises:
            ValueError: If initial state is invalid
        """
        if initial_state is None:
            initial_state = np.ones(self.n_states) / self.n_states
        else:
            # Validate initial state
            if len(initial_state) != self.n_states:
                raise ValueError("Initial state length must match number of states")
            if not np.isclose(sum(initial_state), 1.0):
                raise ValueError("Initial state must sum to 1")
            if not np.all(initial_state >= 0):
                raise ValueError("Initial state probabilities must be non-negative")
            
        distributions = np.zeros((n_steps, self.n_states))
        state = initial_state.copy()
        
        for t in range(n_steps):
            distributions[t] = state
            state = self.transition_matrix.T @ state
            
        return distributions 