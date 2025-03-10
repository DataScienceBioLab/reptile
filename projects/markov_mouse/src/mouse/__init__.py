"""Mouse house simulation module."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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