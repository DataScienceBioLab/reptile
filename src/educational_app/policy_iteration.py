"""Policy Iteration implementation for a customizable grid world environment.

This module implements the Policy Iteration algorithm for solving a Markov Decision Process
(MDP) in a grid world environment with customizable:
- Grid dimensions
- Goal state and reward
- Trap states and penalties
- Wall placements
- Discount factor
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Define color scheme
COLORS = {
    'goal': '#90EE90',  # Light green
    'trap': '#FFB6C1',  # Light red
    'wall': '#D3D3D3',  # Light gray
    'normal': '#FFFFFF',  # White
    'text': '#2F4F4F',  # Dark slate gray
    'arrow': '#4682B4',  # Steel blue
    'grid': '#000000'   # Black
}

class GridWorld:
    """A customizable grid world environment for reinforcement learning."""
    
    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        gamma: float = 0.9,
        goal_states: List[Tuple[int, int]] = None,
        goal_rewards: List[float] = None,
        trap_states: List[Tuple[int, int]] = None,
        trap_penalties: List[float] = None,
        wall_states: List[Tuple[int, int]] = None,
        max_eval_iterations: int = 1000,
        eval_theta: float = 0.01
    ) -> None:
        """Initialize the grid world environment.
        
        Args:
            height: Grid height
            width: Grid width
            gamma: Discount factor
            goal_states: List of goal state positions
            goal_rewards: List of rewards for goal states
            trap_states: List of trap state positions
            trap_penalties: List of penalties for trap states
            wall_states: List of wall positions
            max_eval_iterations: Maximum iterations for policy evaluation
            eval_theta: Convergence threshold for policy evaluation
        """
        self.height = height
        self.width = width
        self.gamma = gamma
        self.max_eval_iterations = max_eval_iterations
        self.eval_theta = eval_theta
        
        # Initialize grid
        self.grid = np.zeros((height, width))
        
        # Set rewards
        self.goal_states = goal_states or [(height-1, width-1)]
        self.goal_rewards = goal_rewards or [10.0]
        self.trap_states = trap_states or [(1, 1), (3, 2)]
        self.trap_penalties = trap_penalties or [-5.0, -5.0]
        self.wall_states = wall_states or [(0, 2), (2, 1), (2, 3)]
        
        # Validate and set rewards
        for state, reward in zip(self.goal_states, self.goal_rewards):
            if self._is_valid_position(state):
                self.grid[state] = reward
                
        for state, penalty in zip(self.trap_states, self.trap_penalties):
            if self._is_valid_position(state):
                self.grid[state] = penalty
            
        # Define possible actions: Right, Down, Left, Up
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.action_names = ['→', '↓', '←', '↑']
        
        # Initialize random policy
        self.policy = np.random.randint(0, 4, (height, width))
        self.values = np.zeros((height, width))
        
        # Store iteration history for visualization
        self.history = []
        
        # Store evaluation iterations for visualization
        self.eval_iterations = []
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if a state is valid (within grid and not a wall)."""
        return self._is_valid_position(state) and state not in self.wall_states
    
    def get_next_state(self, state: Tuple[int, int], action_idx: int) -> Tuple[int, int]:
        """Get the next state given current state and action."""
        action = self.actions[action_idx]
        next_state = (state[0] + action[0], state[1] + action[1])
        
        if self.is_valid_state(next_state):
            return next_state
        return state  # Stay in current state if next state is invalid
    
    def policy_evaluation(self) -> int:
        """Evaluate the current policy until convergence. Returns number of iterations."""
        iterations = 0
        while iterations < self.max_eval_iterations:
            iterations += 1
            delta = 0
            for i in range(self.height):
                for j in range(self.width):
                    if (i, j) in self.wall_states:
                        continue
                        
                    old_value = self.values[i, j]
                    
                    # Get next state under current policy
                    action_idx = self.policy[i, j]
                    next_state = self.get_next_state((i, j), action_idx)
                    
                    # Update value using Bellman equation
                    self.values[i, j] = self.grid[next_state] + self.gamma * self.values[next_state]
                    
                    delta = max(delta, abs(old_value - self.values[i, j]))
            
            if delta < self.eval_theta:
                break
        
        return iterations
    
    def policy_improvement(self) -> bool:
        """Improve the policy based on current values. Returns True if policy changed."""
        policy_stable = True
        
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.wall_states:
                    continue
                    
                old_action = self.policy[i, j]
                
                # Evaluate all actions
                action_values = []
                for action_idx in range(len(self.actions)):
                    next_state = self.get_next_state((i, j), action_idx)
                    value = self.grid[next_state] + self.gamma * self.values[next_state]
                    action_values.append(value)
                
                # Choose best action
                self.policy[i, j] = np.argmax(action_values)
                
                if old_action != self.policy[i, j]:
                    policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self, max_iterations: int = 100) -> None:
        """Run the policy iteration algorithm."""
        self.history = []  # Reset history
        self.eval_iterations = []  # Reset evaluation iterations
        
        for i in range(max_iterations):
            # Store current state
            self.history.append({
                'iteration': i,
                'values': self.values.copy(),
                'policy': self.policy.copy()
            })
            
            # 1. Policy Evaluation
            eval_iters = self.policy_evaluation()
            self.eval_iterations.append(eval_iters)
            
            # 2. Policy Improvement
            policy_stable = self.policy_improvement()
            
            if policy_stable:
                # Store final state
                self.history.append({
                    'iteration': i + 1,
                    'values': self.values.copy(),
                    'policy': self.policy.copy()
                })
                st.success(f"Policy converged after {i+1} iterations")
                break
    
    def create_visualization(self, values: np.ndarray, policy: np.ndarray, show_heatmap: bool = True) -> go.Figure:
        """Create an interactive visualization of the grid world state.
        
        Args:
            values: Array of state values.
            policy: Array of policies.
            show_heatmap: Whether to show the value heatmap.
        
        Returns:
            A plotly figure object.
        """
        # Create a figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Create heatmap for values
        if show_heatmap:
            heatmap = go.Heatmap(
                z=values,
                colorscale='RdBu',
                showscale=True,
                name='State Values',
                hoverongaps=False,
                hovertemplate='Value: %{z:.2f}<extra></extra>'
            )
            fig.add_trace(heatmap, secondary_y=False)
        
        # Add text annotations for values
        for i in range(self.height):
            for j in range(self.width):
                # Determine cell type and style
                if (i, j) in self.goal_states:
                    cell_text = f"G: {values[i, j]:.2f}"
                    font_color = 'white'
                    bgcolor = 'rgba(0, 255, 0, 0.3)'
                elif (i, j) in self.trap_states:
                    cell_text = f"T: {values[i, j]:.2f}"
                    font_color = 'white'
                    bgcolor = 'rgba(255, 0, 0, 0.3)'
                elif (i, j) in self.wall_states:
                    cell_text = "WALL"
                    font_color = 'black'
                    bgcolor = 'rgba(128, 128, 128, 0.5)'
                else:
                    cell_text = f"{values[i, j]:.2f}"
                    font_color = 'black'
                    bgcolor = 'rgba(255, 255, 255, 0.7)'
                
                # Add text annotation
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=cell_text,
                    showarrow=False,
                    font=dict(color=font_color, size=12),
                    bgcolor=bgcolor,
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.8
                )
                
                # Add policy arrows for non-wall states
                if (i, j) not in self.wall_states:
                    policy_probs = policy[i, j]
                    max_action = np.argmax(policy_probs)
                    
                    # Arrow parameters
                    arrow_length = 0.3
                    arrow_head_length = 0.1
                    
                    if max_action == 0:  # Up
                        dx, dy = 0, arrow_length
                    elif max_action == 1:  # Right
                        dx, dy = arrow_length, 0
                    elif max_action == 2:  # Down
                        dx, dy = 0, -arrow_length
                    else:  # Left
                        dx, dy = -arrow_length, 0
                    
                    # Add arrow
                    fig.add_annotation(
                        x=j,
                        y=i,
                        ax=j + dx,
                        ay=i + dy,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=3,
                        arrowcolor='black'
                    )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Grid World State",
                x=0.5,
                y=0.95
            ),
            showlegend=False,
            width=800,
            height=800,
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridwidth=2,
                gridcolor='black',
                range=[-0.5, self.width - 0.5],
                tickmode='array',
                ticktext=[str(i) for i in range(self.width)],
                tickvals=list(range(self.width))
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=2,
                gridcolor='black',
                range=[-0.5, self.height - 0.5],
                tickmode='array',
                ticktext=[str(i) for i in range(self.height)],
                tickvals=list(range(self.height))
            ),
            plot_bgcolor='white'
        )
        
        return fig

def main() -> None:
    """Run a Streamlit app demonstrating policy iteration."""
    st.set_page_config(layout="wide")
    
    st.title("Policy Iteration in Grid World")
    
    st.markdown("""
    This application demonstrates the Policy Iteration algorithm in a customizable grid world environment.
    Use the controls in the sidebar to modify the environment and visualization parameters.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Environment Settings")
        
        # Grid size controls
        st.subheader("Grid Size")
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("Height", min_value=3, max_value=10, value=5)
        with col2:
            width = st.number_input("Width", min_value=3, max_value=10, value=5)
        
        # Discount factor
        gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.9, 0.01)
        
        # Algorithm parameters
        st.subheader("Algorithm Parameters")
        max_iterations = st.number_input("Max Policy Iterations", 10, 1000, 100)
        max_eval_iterations = st.number_input("Max Evaluation Iterations", 10, 1000, 100)
        eval_theta = st.number_input("Evaluation Threshold (θ)", 0.001, 0.1, 0.01, format="%.3f")
        
        # Reward settings
        st.subheader("Rewards")
        num_goals = st.number_input("Number of Goal States", 1, 5, 1)
        goal_reward = st.number_input("Goal Reward", 0.0, 100.0, 10.0)
        
        num_traps = st.number_input("Number of Trap States", 0, 5, 2)
        trap_penalty = st.number_input("Trap Penalty", -100.0, 0.0, -5.0)
        
        num_walls = st.number_input("Number of Wall States", 0, 10, 3)
        
        # Visualization settings
        st.header("Visualization")
        show_heatmap = st.checkbox("Show Value Heatmap", value=True)
        animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)
        
        # Actions
        st.header("Actions")
        initialize = st.button("Initialize New Environment")
        run = st.button("Run Policy Iteration")
    
    # Create or update environment
    if 'env' not in st.session_state or initialize:
        # Generate random positions for goals, traps, and walls
        all_positions = [(i, j) for i in range(height) for j in range(width)]
        np.random.shuffle(all_positions)
        
        goal_states = all_positions[:num_goals]
        trap_states = all_positions[num_goals:num_goals + num_traps]
        wall_states = all_positions[num_goals + num_traps:num_goals + num_traps + num_walls]
        
        st.session_state.env = GridWorld(
            height=height,
            width=width,
            gamma=gamma,
            goal_states=goal_states,
            goal_rewards=[goal_reward] * num_goals,
            trap_states=trap_states,
            trap_penalties=[trap_penalty] * num_traps,
            wall_states=wall_states,
            max_eval_iterations=max_eval_iterations,
            eval_theta=eval_theta
        )
        st.session_state.initialized = False
    
    if run:
        st.session_state.env.policy_iteration(max_iterations=max_iterations)
        st.session_state.initialized = True
    
    # Main content
    if st.session_state.initialized and len(st.session_state.env.history) > 0:
        # Add play/pause controls
        col1, col2 = st.columns([3, 1])
        with col1:
            iteration = st.slider(
                "Iteration",
                0,
                len(st.session_state.env.history) - 1,
                0
            )
        with col2:
            auto_play = st.button("Play Animation")
        
        # Get current state
        state = st.session_state.env.history[iteration]
        
        # Create container for visualization
        viz_container = st.empty()
        
        # Create columns for statistics and evaluation info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ### Statistics for Iteration {iteration}
            - Maximum Value: {state['values'].max():.2f}
            - Minimum Value: {state['values'].min():.2f}
            - Average Value: {state['values'].mean():.2f}
            """)
        
        with col2:
            if iteration < len(st.session_state.env.eval_iterations):
                st.markdown(f"""
                ### Evaluation Details
                - Policy Evaluation Iterations: {st.session_state.env.eval_iterations[iteration]}
                - Evaluation Threshold (θ): {st.session_state.env.eval_theta}
                """)
        
        # Auto-play animation if requested
        if auto_play:
            for i in range(iteration, len(st.session_state.env.history)):
                state = st.session_state.env.history[i]
                fig = st.session_state.env.create_visualization(
                    state['values'],
                    state['policy'],
                    show_heatmap
                )
                viz_container.plotly_chart(fig, use_container_width=True)
                time.sleep(1 / animation_speed)
        else:
            # Display current state
            fig = st.session_state.env.create_visualization(
                state['values'],
                state['policy'],
                show_heatmap
            )
            viz_container.plotly_chart(fig, use_container_width=True)
    else:
        # Show initial state
        fig = st.session_state.env.create_visualization(
            st.session_state.env.values,
            st.session_state.env.policy,
            show_heatmap=True
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 