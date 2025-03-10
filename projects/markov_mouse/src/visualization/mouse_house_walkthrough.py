"""Interactive walkthrough of the Mouse House Markov Chain problem.

This module provides an educational walkthrough of the Mouse House problem,
demonstrating Markov chain concepts through interactive visualizations and
step-by-step explanations.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Optional
import sympy as sp
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.models.mouse_simulation import MouseSimulation
from src.utils.matrix_ops import is_stochastic_matrix

def create_room_layout(
    highlighted_room: Optional[int] = None,
    transitions: Optional[Dict[tuple, float]] = None
) -> go.Figure:
    """Create a visualization of the circular room layout."""
    fig = go.Figure()
    
    # Room positions in a circle
    angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points on a circle
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Add rooms
    for i, (xi, yi) in enumerate(zip(x, y)):
        color = "red" if i == highlighted_room else "lightblue"
        fig.add_trace(go.Scatter(
            x=[xi],
            y=[yi],
            mode="markers+text",
            marker=dict(size=40, color=color),
            text=f"Room {i+1}",
            name=f"Room {i+1}",
            hoverinfo="text"
        ))
    
    # Add transition arrows if provided
    if transitions:
        for (start, end), prob in transitions.items():
            start_x, start_y = x[start-1], y[start-1]
            end_x, end_y = x[end-1], y[end-1]
            
            # Create curved arrow for self-transitions
            if start == end:
                fig.add_annotation(
                    x=start_x,
                    y=start_y,
                    ax=start_x + 0.2,
                    ay=start_y + 0.2,
                    text=f"{prob:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="gray"
                )
            else:
                fig.add_annotation(
                    x=end_x,
                    y=end_y,
                    ax=start_x,
                    ay=start_y,
                    text=f"{prob:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="gray"
                )
    
    fig.update_layout(
        showlegend=False,
        width=600,
        height=600,
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
        title="Mouse House Layout"
    )
    
    return fig

def plot_visit_distribution(
    visits: np.ndarray,
    title: str = "Room Visit Distribution"
) -> go.Figure:
    """Plot the distribution of room visits."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"Room {i+1}" for i in range(len(visits))],
        y=visits / visits.sum(),
        name="Visit Probability"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Room",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        width=600,
        height=400
    )
    
    return fig

def render_latex(expr):
    """Render LaTeX expression in Streamlit."""
    st.latex(sp.latex(expr))

def mouse_house_walkthrough() -> None:
    """Main Streamlit application for Mouse House walkthrough."""
    st.title("Mouse in a House: Interactive Markov Chain Walkthrough")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Choose a section:",
        [
            "1. Problem Introduction",
            "2. Mathematical Setup",
            "3. Transition Analysis",
            "4. Simulation",
            "5. Stationary Distribution",
            "6. Convergence Analysis",
            "7. Applications"
        ]
    )
    
    if section == "1. Problem Introduction":
        st.header("Mouse in a House Problem")
        
        st.markdown("""
        ### Problem Description 🐭
        
        Imagine a mouse moving between five rooms arranged in a circle. The mouse follows
        specific movement rules:
        
        - 60% chance of staying in the current room
        - 20% chance of moving to each adjacent room
        - 0% chance of moving to non-adjacent rooms
        
        This creates an interesting Markov chain with both practical and theoretical implications.
        """)
        
        # Show room layout
        fig = create_room_layout()
        st.plotly_chart(fig)
        
        st.markdown("""
        ### Key Questions 🤔
        
        1. Where will the mouse spend most of its time?
        2. How quickly does it explore all rooms?
        3. What patterns emerge in its movement?
        
        Let's explore these questions through mathematical analysis and simulation!
        """)
        
    elif section == "2. Mathematical Setup":
        st.header("Mathematical Formulation")
        
        st.markdown("""
        ### State Space 📊
        
        Our Markov chain has 5 states (rooms), with specific transition rules:
        """)
        
        # Show transition probabilities for a selected room
        room = st.selectbox("Select a room to see transition probabilities:", [1, 2, 3, 4, 5])
        
        transitions = {}
        for i in range(1, 6):
            if i == room:
                transitions[(room, i)] = 0.6
            elif abs(i - room) == 1 or abs(i - room) == 4:  # Adjacent rooms (including wrap-around)
                transitions[(room, i)] = 0.2
        
        fig = create_room_layout(room-1, transitions)
        st.plotly_chart(fig)
        
        st.markdown("""
        ### Transition Matrix P
        
        The movement probabilities are represented in a 5×5 matrix:
        """)
        
        P = np.array([
            [0.6, 0.2, 0.0, 0.0, 0.2],
            [0.2, 0.6, 0.2, 0.0, 0.0],
            [0.0, 0.2, 0.6, 0.2, 0.0],
            [0.0, 0.0, 0.2, 0.6, 0.2],
            [0.2, 0.0, 0.0, 0.2, 0.6]
        ])
        
        render_latex(sp.Matrix(P))
        
    elif section == "3. Transition Analysis":
        st.header("Analyzing Transitions")
        
        st.markdown("""
        ### Matrix Properties 📐
        
        The transition matrix has several important properties:
        
        1. **Stochastic**: Each row sums to 1
        2. **Symmetric**: P[i,j] = P[j,i]
        3. **Irreducible**: All states communicate
        4. **Aperiodic**: Has self-transitions
        """)
        
        # Interactive transition probability demonstration
        st.subheader("Explore Transitions")
        
        steps = st.slider("Number of steps to simulate", 1, 20, 5)
        start_room = st.selectbox("Starting room", [1, 2, 3, 4, 5])
        
        # Simulate and show path
        sim = MouseSimulation()
        path = sim.simulate_trajectory(steps, start_room-1)
        
        st.write(f"Path: {' → '.join(str(r+1) for r in path)}")
        
        # Show visit distribution
        visits = np.zeros(5)
        for room in path:
            visits[room] += 1
            
        fig = plot_visit_distribution(visits, f"Room Visits (First {steps} Steps)")
        st.plotly_chart(fig)
        
    elif section == "4. Simulation":
        st.header("Mouse Movement Simulation")
        
        st.markdown("""
        ### Interactive Simulation 🎮
        
        Watch how the mouse moves through the house and observe emerging patterns.
        """)
        
        # Simulation parameters
        n_steps = st.slider("Number of steps", 100, 1000, 500)
        start = st.selectbox("Starting room", [1, 2, 3, 4, 5], key="sim_start")
        
        # Run simulation
        sim = MouseSimulation()
        path = sim.simulate_trajectory(n_steps, start-1)
        
        # Analyze results
        visits = np.zeros(5)
        for room in path:
            visits[room] += 1
        
        # Show visit distribution
        fig = plot_visit_distribution(visits)
        st.plotly_chart(fig)
        
        # Show statistics
        st.markdown("### Statistics 📊")
        st.write(f"Average steps per room: {n_steps/5:.1f}")
        st.write(f"Most visited room: Room {np.argmax(visits)+1}")
        st.write(f"Least visited room: Room {np.argmin(visits)+1}")
        
    elif section == "5. Stationary Distribution":
        st.header("Stationary Distribution")
        
        st.markdown("""
        ### Theory 📚
        
        The stationary distribution π satisfies:
        """)
        
        render_latex(r"\pi P = \pi")
        
        st.markdown("""
        Due to the symmetric structure of our transition matrix,
        the stationary distribution is uniform:
        """)
        
        render_latex(r"\pi = [0.2, 0.2, 0.2, 0.2, 0.2]")
        
        st.markdown("""
        ### Verification through Simulation 🔍
        
        Let's verify this theoretical result through simulation:
        """)
        
        # Run long simulation
        sim = MouseSimulation()
        path = sim.simulate_trajectory(10000, 0)
        visits = np.zeros(5)
        for room in path:
            visits[room] += 1
        
        fig = plot_visit_distribution(visits, "Long-term Visit Distribution")
        st.plotly_chart(fig)
        
    elif section == "6. Convergence Analysis":
        st.header("Convergence Analysis")
        
        st.markdown("""
        ### Convergence Properties 📈
        
        The speed of convergence to the stationary distribution depends on:
        1. The second-largest eigenvalue of P
        2. The initial distribution
        3. The mixing time of the chain
        """)
        
        # Demonstrate convergence
        st.subheader("Convergence Demonstration")
        
        n_steps = st.slider("Steps for convergence", 10, 100, 50, key="conv_steps")
        start = st.selectbox("Initial room", [1, 2, 3, 4, 5], key="conv_start")
        
        # Run multiple simulations
        n_sims = 100
        all_visits = np.zeros((n_sims, 5))
        
        for i in range(n_sims):
            sim = MouseSimulation()
            path = sim.simulate_trajectory(n_steps, start-1)
            visits = np.zeros(5)
            for room in path:
                visits[room] += 1
            all_visits[i] = visits / n_steps
        
        # Plot average distribution
        avg_dist = all_visits.mean(axis=0)
        fig = plot_visit_distribution(avg_dist, f"Average Distribution after {n_steps} steps")
        st.plotly_chart(fig)
        
    elif section == "7. Applications":
        st.header("Real-world Applications")
        
        st.markdown("""
        ### Applications of the Mouse House Model 🌟
        
        This simple model has several practical applications:
        
        1. **Animal Behavior** 🐁
           - Understanding movement patterns
           - Habitat utilization
           - Space exploration strategies
        
        2. **Robot Navigation** 🤖
           - Room exploration algorithms
           - Coverage optimization
           - Movement planning
        
        3. **Human Movement** 👤
           - Building utilization analysis
           - Traffic flow modeling
           - Space design optimization
        """)
        
        # Interactive application example
        st.subheader("Example: Space Utilization")
        
        # Simulate different movement patterns
        pattern = st.selectbox(
            "Select movement pattern:",
            ["Standard (60-20)", "Explorative (40-30)", "Sticky (80-10)"]
        )
        
        if pattern == "Standard (60-20)":
            stay_prob = 0.6
        elif pattern == "Explorative (40-30)":
            stay_prob = 0.4
        else:  # Sticky
            stay_prob = 0.8
        
        move_prob = (1 - stay_prob) / 2
        
        sim = MouseSimulation(stay_prob=stay_prob)
        path = sim.simulate_trajectory(1000, 0)
        visits = np.zeros(5)
        for room in path:
            visits[room] += 1
        
        fig = plot_visit_distribution(visits, f"Visit Distribution ({pattern})")
        st.plotly_chart(fig)

if __name__ == "__main__":
    mouse_house_walkthrough() 