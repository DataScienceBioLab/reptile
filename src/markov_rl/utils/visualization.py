"""Visualization utilities for Markov chain analysis and Q-Learning.

This module provides plotting functions for visualizing:
- State distribution evolution
- Convergence to stationary distribution
- Transition matrices
- Q-value tables
- Learning curves
- Policy visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import seaborn as sns
from ..models.q_learning import TrainingMetrics

def plot_distribution_evolution(distributions: np.ndarray,
                              stationary_dist: Optional[np.ndarray] = None,
                              state_labels: Optional[List[str]] = None,
                              title: str = 'State Distribution Evolution',
                              figsize: tuple = (10, 6)) -> None:
    """Plot evolution of state distributions over time.
    
    Args:
        distributions: Array of shape (n_steps, n_states) with distribution at each step
        stationary_dist: Optional stationary distribution to show as horizontal lines
        state_labels: Optional list of state labels
        title: Plot title
        figsize: Figure size tuple
    """
    n_steps, n_states = distributions.shape
    
    if state_labels is None:
        state_labels = [f'State {i+1}' for i in range(n_states)]
        
    plt.figure(figsize=figsize)
    
    # Plot evolution for each state
    for i in range(n_states):
        plt.plot(range(n_steps), distributions[:, i], 
                label=state_labels[i])
                
    # Add stationary distribution lines if provided
    if stationary_dist is not None:
        for i in range(n_states):
            plt.axhline(y=stationary_dist[i], color=f'C{i}',
                       linestyle='--', alpha=0.3)
                       
    plt.xlabel('Steps')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_transition_matrix(transition_matrix: np.ndarray,
                         state_labels: Optional[List[str]] = None,
                         title: str = 'Transition Matrix',
                         figsize: tuple = (8, 6)) -> None:
    """Plot transition matrix as a heatmap.
    
    Args:
        transition_matrix: Square transition probability matrix
        state_labels: Optional list of state labels
        title: Plot title
        figsize: Figure size tuple
    """
    n_states = transition_matrix.shape[0]
    
    if state_labels is None:
        state_labels = [f'State {i+1}' for i in range(n_states)]
        
    plt.figure(figsize=figsize)
    plt.imshow(transition_matrix, cmap='YlOrRd')
    
    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            plt.text(j, i, f'{transition_matrix[i,j]:.2f}',
                    ha='center', va='center')
                    
    plt.xticks(range(n_states), state_labels)
    plt.yticks(range(n_states), state_labels)
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.title(title)
    plt.colorbar(label='Transition Probability')
    plt.show()

def plot_q_table(q_table: np.ndarray,
                 state_labels: Optional[List[str]] = None,
                 action_labels: Optional[List[str]] = None,
                 title: str = 'Q-Value Table',
                 figsize: tuple = (10, 6)) -> None:
    """Plot Q-table as a heatmap.
    
    Args:
        q_table: Q-value table of shape (n_states, n_actions)
        state_labels: Optional list of state labels
        action_labels: Optional list of action labels
        title: Plot title
        figsize: Figure size tuple
    """
    n_states, n_actions = q_table.shape
    
    if state_labels is None:
        state_labels = [f'State {i+1}' for i in range(n_states)]
    if action_labels is None:
        action_labels = [f'Action {i+1}' for i in range(n_actions)]
        
    plt.figure(figsize=figsize)
    sns.heatmap(q_table, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=action_labels, yticklabels=state_labels)
    
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.title(title)
    plt.show()

def plot_learning_curves(metrics: TrainingMetrics,
                        figsize: tuple = (15, 5)) -> None:
    """Plot learning curves from training metrics.
    
    Args:
        metrics: Training metrics containing rewards and other data
        figsize: Figure size tuple
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot episode rewards
    ax1.plot(metrics.episode_rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(metrics.episode_lengths)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.grid(True)
    
    # Plot Q-value changes
    ax3.plot(metrics.q_value_changes)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Q-Value Change')
    ax3.set_title('Learning Progress')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_policy(policy: np.ndarray,
                state_coordinates: np.ndarray,
                action_vectors: Optional[List[Tuple[float, float]]] = None,
                title: str = 'Policy Visualization',
                figsize: tuple = (8, 8)) -> None:
    """Visualize policy as arrows in 2D space.
    
    Args:
        policy: Array of action indices for each state
        state_coordinates: Array of (x, y) coordinates for each state
        action_vectors: Optional list of (dx, dy) vectors for each action
        title: Plot title
        figsize: Figure size tuple
    """
    if action_vectors is None:
        action_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # NSEW
        
    plt.figure(figsize=figsize)
    
    # Plot state points
    plt.scatter(state_coordinates[:, 0], state_coordinates[:, 1],
                c='blue', alpha=0.5)
    
    # Plot policy arrows
    for state_idx, action_idx in enumerate(policy):
        x, y = state_coordinates[state_idx]
        dx, dy = action_vectors[action_idx]
        plt.arrow(x, y, dx*0.2, dy*0.2, head_width=0.1,
                 head_length=0.1, fc='red', ec='red')
    
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show() 