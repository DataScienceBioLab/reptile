"""Unit tests for visualization utilities."""

import pytest
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from markov_rl.utils.visualization import (
    plot_distribution_evolution,
    plot_transition_matrix,
    plot_q_table,
    plot_learning_curves,
    plot_policy
)
from markov_rl.models.q_learning import TrainingMetrics

@pytest.mark.unit
class TestVisualization:
    """Test suite for visualization utilities."""

    def setup_method(self) -> None:
        """Set up test environment."""
        plt.close('all')  # Close any open figures

    def test_distribution_evolution_plot(self, distribution_sequence: np.ndarray) -> None:
        """Test distribution evolution plotting."""
        plot_distribution_evolution(distribution_sequence)
        fig = plt.gcf()
        
        # Check plot components
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Steps'
        assert ax.get_ylabel() == 'Probability'
        assert len(ax.lines) == distribution_sequence.shape[1]

    def test_transition_matrix_plot(self, transition_matrix: np.ndarray) -> None:
        """Test transition matrix plotting."""
        plot_transition_matrix(transition_matrix)
        fig = plt.gcf()
        
        # Check plot components
        assert len(fig.axes) == 2  # Main plot and colorbar
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'To State'
        assert ax.get_ylabel() == 'From State'
        
        # Check annotations
        texts = [t for t in ax.texts if t.get_text()]
        assert len(texts) == transition_matrix.size

    def test_q_table_plot(self, q_learning: 'QLearning') -> None:
        """Test Q-table plotting."""
        plot_q_table(q_learning.q_table)
        fig = plt.gcf()
        
        # Check plot components
        assert len(fig.axes) == 2  # Main plot and colorbar
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Actions'
        assert ax.get_ylabel() == 'States'

    def test_learning_curves_plot(self) -> None:
        """Test learning curves plotting."""
        metrics = TrainingMetrics(
            episode_rewards=[1.0, 2.0, 3.0],
            episode_lengths=[10, 8, 6],
            q_value_changes=[0.5, 0.3, 0.1]
        )
        
        plot_learning_curves(metrics)
        fig = plt.gcf()
        
        # Check plot components
        assert len(fig.axes) == 3  # Three subplots
        assert fig.axes[0].get_ylabel() == 'Total Reward'
        assert fig.axes[1].get_ylabel() == 'Steps'
        assert fig.axes[2].get_ylabel() == 'Average Q-Value Change'

    def test_policy_plot(self) -> None:
        """Test policy visualization."""
        policy = np.array([0, 1, 2, 3])  # Four states with different actions
        state_coords = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ])
        
        plot_policy(policy, state_coords)
        fig = plt.gcf()
        
        # Check plot components
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        
        # Check scatter points and arrows
        assert len(ax.collections) == 1  # Scatter points
        assert len(ax.patches) == len(policy)  # Arrows

    def test_custom_labels(self, transition_matrix: np.ndarray) -> None:
        """Test plotting with custom labels."""
        state_labels = ['A', 'B', 'C']
        plot_transition_matrix(
            transition_matrix,
            state_labels=state_labels,
            title='Custom Title'
        )
        
        fig = plt.gcf()
        ax = fig.axes[0]
        assert ax.get_title() == 'Custom Title'
        
        # Check that custom labels are used
        x_ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert x_ticks == state_labels

    def test_figure_size(self, distribution_sequence: np.ndarray) -> None:
        """Test custom figure size."""
        figsize = (12, 8)
        plot_distribution_evolution(
            distribution_sequence,
            figsize=figsize
        )
        
        fig = plt.gcf()
        assert fig.get_size_inches().tolist() == list(figsize)

    def teardown_method(self) -> None:
        """Clean up after tests."""
        plt.close('all') 