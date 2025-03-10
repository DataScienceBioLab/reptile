"""Mouse house simulation module."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from ..markov import MarkovChain

__version__ = "0.1.0"

@dataclass
class MouseState:
    """Represents the state of a mouse in the simulation."""
    position: Tuple[int, int]
    energy: float
    direction: Tuple[float, float]
    age: int

class MouseHouse:
    """Simulation of mouse behavior in a house environment."""
    
    def __init__(self, width: int, height: int) -> None:
        """Initialize mouse house environment.
        
        Args:
            width: Width of the house in grid units
            height: Height of the house in grid units
        """
        self.width = width
        self.height = height
        self.mice: List[MouseState] = []
        self.food_locations: List[Tuple[int, int]] = []
    
    def add_mouse(self, position: Tuple[int, int]) -> None:
        """Add a mouse to the simulation.
        
        Args:
            position: Initial position of the mouse
        """
        mouse = MouseState(
            position=position,
            energy=100.0,
            direction=(0.0, 0.0),
            age=0
        )
        self.mice.append(mouse)
    
    def add_food(self, position: Tuple[int, int]) -> None:
        """Add food to the environment.
        
        Args:
            position: Position of the food
        """
        self.food_locations.append(position)
    
    def step(self) -> None:
        """Advance the simulation by one time step."""
        # Implementation will go here
        pass

class MouseSimulation:
    """Simulation of mouse movement between rooms.
    
    The mouse has:
    - Configurable probability of staying in current room (default 0.6)
    - Equal probability of moving to each adjacent room (default 0.2)
    
    Attributes:
        n_rooms: Number of rooms (5)
        stay_prob: Probability of staying in current room
        transition_matrix: Transition probability matrix
        markov_chain: MarkovChain instance for analysis
    """
    
    def __init__(self, stay_prob: float = 0.6):
        """Initialize mouse simulation with 5 rooms.
        
        Args:
            stay_prob: Probability of staying in current room (default 0.6)
        """
        self.n_rooms = 5
        self.stay_prob = stay_prob
        self.transition_matrix = self._create_transition_matrix()
        self.markov_chain = MarkovChain(self.transition_matrix)
        
    def _create_transition_matrix(self) -> np.ndarray:
        """Create transition matrix for mouse movement.
        
        Returns:
            5x5 transition probability matrix
        """
        P = np.zeros((self.n_rooms, self.n_rooms))
        
        # Set diagonal (stay probabilities)
        np.fill_diagonal(P, self.stay_prob)
        
        # Calculate probability of moving to adjacent rooms
        move_prob = (1 - self.stay_prob) / 2
        
        # Set adjacent room probabilities
        for i in range(self.n_rooms):
            next_room = (i + 1) % self.n_rooms
            prev_room = (i - 1) % self.n_rooms
            P[i, next_room] = move_prob
            P[i, prev_room] = move_prob
            
        return P
        
    def simulate_trajectory(self, n_steps: int, start_room: int = 0) -> List[int]:
        """Simulate mouse movement for given number of steps.
        
        Args:
            n_steps: Number of time steps to simulate
            start_room: Initial room (0-4)
            
        Returns:
            List of room indices visited
        """
        trajectory = [start_room]
        current_room = start_room
        
        for _ in range(n_steps - 1):
            # Sample next room based on transition probabilities
            next_room = np.random.choice(self.n_rooms, 
                                       p=self.transition_matrix[current_room])
            trajectory.append(next_room)
            current_room = next_room
            
        return trajectory
        
    def compute_visit_frequencies(self, trajectory: List[int]) -> np.ndarray:
        """Compute frequency of visits to each room.
        
        Args:
            trajectory: List of room indices visited
            
        Returns:
            Array of visit frequencies for each room
        """
        visits = np.zeros(self.n_rooms)
        for room in trajectory:
            visits[room] += 1
        return visits / len(trajectory)
        
    def analyze_and_visualize(self, n_steps: int = 1000, start_room: int = 0):
        """Run simulation and create visualizations.
        
        Args:
            n_steps: Number of time steps
            start_room: Initial room
        """
        # Simulate trajectory
        trajectory = self.simulate_trajectory(n_steps, start_room)
        visit_freqs = self.compute_visit_frequencies(trajectory)
        
        # Get theoretical stationary distribution
        stationary_dist = self.markov_chain.get_stationary_distribution()
        
        # Create initial state distribution
        initial_state = np.zeros(self.n_rooms)
        initial_state[start_room] = 1
        
        # Get state distribution evolution
        distributions = self.markov_chain.evolve_distribution(
            n_steps=n_steps,
            initial_state=initial_state
        )
        
        # Visualizations
        room_labels = [f'Room {i+1}' for i in range(self.n_rooms)]
        
        # Plot transition matrix
        from ..utils.visualization import plot_transition_matrix
        plot_transition_matrix(
            self.transition_matrix,
            state_labels=room_labels,
            title='Mouse Movement Transition Probabilities'
        )
        
        # Plot distribution evolution
        from ..utils.visualization import plot_distribution_evolution
        plot_distribution_evolution(
            distributions,
            stationary_dist=stationary_dist,
            state_labels=room_labels,
            title='Mouse Location Distribution Over Time'
        )
        
        # Print comparison of empirical vs theoretical distributions
        print("\nEmpirical vs Theoretical Distributions:")
        print("Room | Empirical | Theoretical")
        print("-" * 35)
        for i in range(self.n_rooms):
            print(f"{i+1:4d} | {visit_freqs[i]:9.3f} | {stationary_dist[i]:10.3f}") 