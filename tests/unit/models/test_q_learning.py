"""Unit tests for Q-Learning implementation."""

import pytest
import numpy as np
from typing import List
from markov_rl.models.q_learning import QLearning, TrainingMetrics

@pytest.mark.unit
class TestQLearning:
    """Test suite for Q-Learning implementation."""

    def test_initialization(self, q_learning: QLearning) -> None:
        """Test Q-Learning initialization."""
        assert q_learning.learning_rate == 0.1
        assert q_learning.discount_factor == 0.95
        assert q_learning.epsilon == 1.0
        assert q_learning.min_epsilon == 0.01
        assert q_learning.epsilon_decay == 0.995
        assert q_learning.temperature == 1.0
        assert q_learning.q_table.shape == (9, 4)  # 9 states, 4 actions

    def test_q_table_initialization(self, q_learning: QLearning) -> None:
        """Test Q-table initialization."""
        assert np.all(q_learning.q_table >= 0)
        assert np.all(q_learning.q_table <= 0.1)
        assert q_learning.q_table.shape == (9, 4)

    def test_epsilon_greedy_exploration(
        self,
        q_learning: QLearning,
        state_names: List[str]
    ) -> None:
        """Test ε-greedy exploration."""
        # Set epsilon to 1.0 for pure exploration
        q_learning.epsilon = 1.0
        actions = [q_learning.get_action_epsilon_greedy(0) for _ in range(1000)]
        
        # Check that all actions are selected with roughly equal probability
        action_counts = np.bincount(actions)
        expected_count = 1000 / 4  # 4 actions
        assert all(abs(count - expected_count) < 100 for count in action_counts)

    def test_epsilon_greedy_exploitation(
        self,
        q_learning: QLearning,
        state_names: List[str]
    ) -> None:
        """Test ε-greedy exploitation."""
        # Set epsilon to 0.0 for pure exploitation
        q_learning.epsilon = 0.0
        
        # Set up Q-table with a clear best action
        q_learning.q_table[0] = [0.1, 0.5, 0.2, 0.3]
        
        # Check that the best action is always selected
        actions = [q_learning.get_action_epsilon_greedy(0) for _ in range(100)]
        assert all(action == 1 for action in actions)

    def test_softmax_policy(
        self,
        q_learning: QLearning,
        state_names: List[str]
    ) -> None:
        """Test softmax action selection."""
        # Set up Q-table with different values
        q_learning.q_table[0] = [0.1, 0.2, 0.3, 0.4]
        q_learning.temperature = 0.1  # Low temperature for more exploitation
        
        actions = [q_learning.get_action_softmax(0) for _ in range(1000)]
        action_counts = np.bincount(actions)
        
        # Higher Q-values should be selected more frequently
        assert action_counts[3] > action_counts[2]
        assert action_counts[2] > action_counts[1]
        assert action_counts[1] > action_counts[0]

    def test_q_value_update(self, q_learning: QLearning) -> None:
        """Test Q-value update calculation."""
        old_value = q_learning.q_table[0, 0]
        change = q_learning.update(
            state_idx=0,
            action_idx=0,
            reward=1.0,
            next_state_idx=1
        )
        
        # Check that the Q-value was updated
        assert q_learning.q_table[0, 0] != old_value
        assert change > 0

    def test_epsilon_decay(self, q_learning: QLearning) -> None:
        """Test epsilon decay."""
        initial_epsilon = q_learning.epsilon
        q_learning.decay_epsilon()
        assert q_learning.epsilon == initial_epsilon * q_learning.epsilon_decay
        
        # Test minimum epsilon
        q_learning.epsilon = 0.005  # Just above min_epsilon
        q_learning.decay_epsilon()
        assert q_learning.epsilon == q_learning.min_epsilon

    def test_training_loop(
        self,
        q_learning: QLearning,
        simple_env: 'SimpleGridEnv'
    ) -> None:
        """Test training loop execution."""
        metrics = q_learning.train(
            env=simple_env,
            n_episodes=10,
            max_steps=100,
            policy='epsilon-greedy'
        )
        
        assert isinstance(metrics, TrainingMetrics)
        assert len(metrics.episode_rewards) == 10
        assert len(metrics.episode_lengths) == 10
        assert len(metrics.q_value_changes) == 10
        
        # Check that some learning occurred
        assert any(reward > 0 for reward in metrics.episode_rewards)
        assert any(changes > 0 for changes in metrics.q_value_changes)

    def test_policy_extraction(
        self,
        q_learning: QLearning,
        state_names: List[str]
    ) -> None:
        """Test policy extraction from Q-table."""
        # Set up Q-table with known best actions
        q_learning.q_table = np.array([
            [0.1, 0.2, 0.3, 0.4],  # Best action: 3
            [0.4, 0.3, 0.2, 0.1],  # Best action: 0
            [0.2, 0.4, 0.1, 0.3]   # Best action: 1
        ])
        
        policy = q_learning.get_policy()
        assert np.array_equal(policy, [3, 0, 1])

    def test_state_value_calculation(
        self,
        q_learning: QLearning,
        state_names: List[str]
    ) -> None:
        """Test state value calculation."""
        # Set up Q-table with known values
        q_learning.q_table = np.array([
            [0.1, 0.2, 0.3, 0.4],  # Max: 0.4
            [0.4, 0.3, 0.2, 0.1],  # Max: 0.4
            [0.2, 0.4, 0.1, 0.3]   # Max: 0.4
        ])
        
        state_values = q_learning.get_state_values()
        assert np.array_equal(state_values, [0.4, 0.4, 0.4]) 