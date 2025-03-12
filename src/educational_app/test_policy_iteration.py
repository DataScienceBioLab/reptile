"""Tests for the Policy Iteration implementation."""

import pytest
import numpy as np
from .policy_iteration import GridWorld

def test_grid_world_initialization():
    """Test that the grid world is initialized correctly."""
    env = GridWorld()
    
    # Test grid size
    assert env.size == 5
    assert env.grid.shape == (5, 5)
    
    # Test reward values
    assert env.grid[4, 4] == 10  # Goal state
    assert env.grid[1, 1] == -5  # First trap
    assert env.grid[3, 2] == -5  # Second trap
    
    # Test initial policy and values
    assert env.policy.shape == (5, 5)
    assert env.values.shape == (5, 5)
    assert np.all(env.values == 0)  # All values start at 0

def test_valid_state_checking():
    """Test the is_valid_state method."""
    env = GridWorld()
    
    # Test valid states
    assert env.is_valid_state((0, 0))  # Start state
    assert env.is_valid_state((4, 4))  # Goal state
    
    # Test invalid states
    assert not env.is_valid_state((-1, 0))  # Outside grid
    assert not env.is_valid_state((5, 5))   # Outside grid
    assert not env.is_valid_state((0, 2))   # Wall state
    assert not env.is_valid_state((2, 1))   # Wall state
    assert not env.is_valid_state((2, 3))   # Wall state

def test_next_state_transitions():
    """Test state transitions."""
    env = GridWorld()
    
    # Test valid moves
    assert env.get_next_state((0, 0), 0) == (0, 1)  # Right
    assert env.get_next_state((0, 0), 1) == (1, 0)  # Down
    
    # Test moves into walls (should stay in place)
    assert env.get_next_state((0, 1), 0) == (0, 1)  # Right into wall
    
    # Test moves outside grid (should stay in place)
    assert env.get_next_state((0, 0), 2) == (0, 0)  # Left
    assert env.get_next_state((0, 0), 3) == (0, 0)  # Up

def test_policy_evaluation():
    """Test policy evaluation step."""
    env = GridWorld()
    
    # Set a simple deterministic policy (all actions = right)
    env.policy = np.zeros((5, 5), dtype=int)
    
    # Run policy evaluation
    old_values = env.values.copy()
    env.policy_evaluation()
    
    # Values should have changed
    assert not np.array_equal(env.values, old_values)
    
    # Goal state should have highest value
    assert env.values[4, 4] > env.values[0, 0]

def test_policy_improvement():
    """Test policy improvement step."""
    env = GridWorld()
    
    # Set initial values to make goal state attractive
    env.values[4, 4] = 10
    
    # Run policy improvement
    old_policy = env.policy.copy()
    policy_changed = env.policy_improvement()
    
    # Policy should have changed
    assert policy_changed
    assert not np.array_equal(env.policy, old_policy)

def test_policy_iteration_convergence():
    """Test that policy iteration converges."""
    env = GridWorld()
    
    # Run policy iteration
    env.policy_iteration()
    
    # Values should be non-zero
    assert not np.all(env.values == 0)
    
    # Goal state should have highest value
    assert env.values[4, 4] >= env.values.max()
    
    # Trap states should have negative values
    assert env.values[1, 1] < 0
    assert env.values[3, 2] < 0 