"""Markov chain analysis module."""

from typing import Dict, List, Optional, Tuple

__version__ = "0.1.0"

class MarkovChain:
    """Implementation of a Markov chain."""
    
    def __init__(self, transition_matrix: List[List[float]]) -> None:
        """Initialize Markov chain with transition matrix.
        
        Args:
            transition_matrix: Square matrix of transition probabilities
        """
        self.transition_matrix = transition_matrix
        self.states = list(range(len(transition_matrix)))
    
    def next_state(self, current_state: int) -> int:
        """Get next state based on transition probabilities.
        
        Args:
            current_state: Current state index
            
        Returns:
            Next state index
        """
        # Implementation will go here
        pass
    
    def stationary_distribution(self) -> List[float]:
        """Calculate stationary distribution.
        
        Returns:
            List of probabilities for each state
        """
        # Implementation will go here
        pass 