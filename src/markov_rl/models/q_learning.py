"""Q-Learning implementation for reinforcement learning.

This module provides the implementation of Q-Learning algorithm, including:
- Q-Table management
- Action-value function updates
- Policy derivation
- Training loop management
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Metrics collected during Q-Learning training.

    Attributes:
        episode_rewards: List of total rewards per episode
        episode_lengths: List of steps per episode
        q_value_changes: Average Q-value change per episode
    """
    episode_rewards: List[float]
    episode_lengths: List[int]
    q_value_changes: List[float]

class QLearning:
    """A class implementing the Q-Learning algorithm.

    This class provides functionality for Q-Learning, including state-action value
    updates, policy generation, and learning rate management.

    Attributes:
        states (List[str]): List of possible states
        actions (List[str]): List of possible actions
        q_table (np.ndarray): Q-value table for state-action pairs
        learning_rate (float): Learning rate for Q-value updates
        discount_factor (float): Discount factor for future rewards
        epsilon (float): Exploration rate for ε-greedy policy
        min_epsilon (float): Minimum exploration rate
        epsilon_decay (float): Rate at which epsilon decays
        temperature (float): Temperature parameter for softmax policy
    """

    def __init__(
        self,
        states: List[str],
        actions: List[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.995,
        temperature: float = 1.0
    ) -> None:
        """Initialize Q-Learning agent.

        Args:
            states: List of state names
            actions: List of possible actions
            learning_rate: Rate at which new information overrides old (default: 0.1)
            discount_factor: Weight of future rewards (default: 0.95)
            epsilon: Initial exploration rate (default: 1.0)
            min_epsilon: Minimum exploration rate (default: 0.01)
            epsilon_decay: Rate at which epsilon decays (default: 0.995)
            temperature: Temperature parameter for softmax policy (default: 1.0)
        """
        self.states = states
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self._initialize_q_table()

    def _initialize_q_table(self) -> None:
        """Initialize the Q-table with small random values."""
        self.q_table = np.random.uniform(
            low=0, high=0.1, size=(len(self.states), len(self.actions))
        )

    def get_action_epsilon_greedy(self, state_idx: int) -> int:
        """Select action using ε-greedy policy.

        Args:
            state_idx: Index of current state

        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        return int(np.argmax(self.q_table[state_idx]))

    def get_action_softmax(self, state_idx: int) -> int:
        """Select action using softmax policy.

        Args:
            state_idx: Index of current state

        Returns:
            Selected action index
        """
        q_values = self.q_table[state_idx]
        exp_q = np.exp(q_values / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        return int(np.random.choice(len(self.actions), p=probabilities))

    def update(
        self, 
        state_idx: int, 
        action_idx: int, 
        reward: float, 
        next_state_idx: int
    ) -> float:
        """Update Q-value for state-action pair.

        Args:
            state_idx: Current state index
            action_idx: Chosen action index
            reward: Immediate reward received
            next_state_idx: Next state index

        Returns:
            Magnitude of Q-value change
        """
        old_value = self.q_table[state_idx, action_idx]
        next_max = np.max(self.q_table[next_state_idx])
        
        # Q-Learning update rule
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state_idx, action_idx] = new_value
        return abs(new_value - old_value)

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(
        self,
        env: 'Environment',  # Type hint for environment interface
        n_episodes: int,
        max_steps: int,
        policy: str = 'epsilon-greedy'
    ) -> TrainingMetrics:
        """Train the Q-Learning agent.

        Args:
            env: Environment that implements reset() and step()
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            policy: Action selection policy ('epsilon-greedy' or 'softmax')

        Returns:
            Training metrics
        """
        metrics = TrainingMetrics([], [], [])
        action_selector = (
            self.get_action_epsilon_greedy if policy == 'epsilon-greedy'
            else self.get_action_softmax
        )

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            q_changes = []

            for step in range(max_steps):
                action = action_selector(state)
                next_state, reward, done = env.step(action)
                
                q_change = self.update(state, action, reward, next_state)
                q_changes.append(q_change)
                total_reward += reward
                state = next_state

                if done:
                    break

            # Update metrics
            metrics.episode_rewards.append(total_reward)
            metrics.episode_lengths.append(step + 1)
            metrics.q_value_changes.append(np.mean(q_changes))

            # Decay exploration rate
            if policy == 'epsilon-greedy':
                self.decay_epsilon()

            # Log progress
            if (episode + 1) % 100 == 0:
                logger.info(
                    f"Episode {episode + 1}/{n_episodes}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Steps: {step + 1}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        return metrics

    def get_policy(self) -> npt.NDArray[np.int_]:
        """Extract deterministic policy from Q-table.

        Returns:
            Array of best actions for each state
        """
        return np.argmax(self.q_table, axis=1)

    def get_state_values(self) -> npt.NDArray[np.float_]:
        """Get state values (max Q-value for each state).

        Returns:
            Array of state values
        """
        return np.max(self.q_table, axis=1) 