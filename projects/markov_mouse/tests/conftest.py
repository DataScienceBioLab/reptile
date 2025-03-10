"""Test configuration and shared fixtures."""

import pytest
import numpy as np
from typing import List, Tuple, Generator
from markov_rl.models.q_learning import QLearning

class SimpleGridEnv:
    """Simple 2D grid environment for testing Q-Learning.
    
    A 3x3 grid where the agent can move in four directions.
    Goal is at (2, 2), starting position is (0, 0).
    """
    
    def __init__(self) -> None:
        self.grid_size = 3
        self.actions = ["up", "right", "down", "left"]
        self.reset()
        
    def reset(self) -> int:
        """Reset environment to starting state."""
        self.position = [0, 0]
        return self._get_state()
        
    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take a step in the environment.
        
        Args:
            action: Action index (0: up, 1: right, 2: down, 3: left)
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Get movement direction
        if action == 0:  # up
            self.position[0] = max(0, self.position[0] - 1)
        elif action == 1:  # right
            self.position[1] = min(self.grid_size - 1, self.position[1] + 1)
        elif action == 2:  # down
            self.position[0] = min(self.grid_size - 1, self.position[0] + 1)
        else:  # left
            self.position[1] = max(0, self.position[1] - 1)
            
        # Check if goal reached
        done = self.position == [2, 2]
        reward = 1.0 if done else -0.1
        
        return self._get_state(), reward, done
        
    def _get_state(self) -> int:
        """Convert position to state index."""
        return self.position[0] * self.grid_size + self.position[1]

@pytest.fixture
def simple_env() -> SimpleGridEnv:
    """Create a simple grid environment for testing."""
    return SimpleGridEnv()

@pytest.fixture
def state_names() -> List[str]:
    """Create list of state names for testing."""
    return [f"State_{i}" for i in range(9)]  # 3x3 grid

@pytest.fixture
def action_names() -> List[str]:
    """Create list of action names for testing."""
    return ["up", "right", "down", "left"]

@pytest.fixture
def q_learning(state_names: List[str], action_names: List[str]) -> QLearning:
    """Create Q-Learning instance with test configuration."""
    return QLearning(
        states=state_names,
        actions=action_names,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=0.995,
        temperature=1.0
    )

@pytest.fixture
def transition_matrix() -> np.ndarray:
    """Create a test transition matrix."""
    matrix = np.array([
        [0.7, 0.3, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.2, 0.8]
    ])
    return matrix

@pytest.fixture
def distribution_sequence() -> np.ndarray:
    """Create a test sequence of distributions."""
    sequence = np.array([
        [1.0, 0.0, 0.0],
        [0.7, 0.3, 0.0],
        [0.55, 0.4, 0.05],
        [0.45, 0.45, 0.1]
    ])
    return sequence 