"""Interactive walkthrough of Markov chain properties and concepts.

This module provides an educational walkthrough of Markov chain properties using
interactive visualizations, step-by-step explanations, and hands-on exercises.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Optional, Tuple, List, Dict
import sympy as sp
from src.markov_rl.core.markov_chain import MarkovChain
from src.markov_rl.utils.matrix_ops import (
    is_stochastic_matrix,
    normalize_matrix_rows,
    is_symmetric_matrix,
    make_symmetric
)

def create_matrix_heatmap(
    matrix: np.ndarray,
    title: str = "Matrix Visualization",
    show_values: bool = True,
    colorscale: str = "RdYlBu_r",
    height: int = 400
) -> go.Figure:
    """Create an interactive heatmap visualization of a matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        text=matrix if show_values else None,
        texttemplate="%{text:.3f}" if show_values else None,
        colorscale=colorscale,
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Next State",
        yaxis_title="Current State",
        width=600,
        height=height
    )
    
    return fig

def plot_state_distribution(
    distributions: np.ndarray,
    states: List[str],
    title: str = "State Distribution Evolution"
) -> go.Figure:
    """Plot the evolution of state distributions over time."""
    fig = go.Figure()
    
    for i, state in enumerate(states):
        fig.add_trace(go.Scatter(
            y=distributions[:, i],
            mode='lines+markers',
            name=f'State {state}',
            hovertemplate=f'State {state}: %{{y:.3f}}<br>Step: %{{x}}'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        width=800,
        height=400,
        showlegend=True
    )
    
    return fig

def create_quiz_section(quiz_data: Dict) -> None:
    """Create an interactive quiz section."""
    st.subheader("📝 Knowledge Check")
    
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    
    for q_id, data in quiz_data.items():
        st.write(f"**{data['question']}**")
        user_answer = st.radio(
            "Select your answer:",
            data['options'],
            key=f"quiz_{q_id}"
        )
        
        check = st.button("Check Answer", key=f"check_{q_id}")
        if check:
            if user_answer == data['correct']:
                st.success("✅ Correct! " + data['explanation'])
            else:
                st.error("❌ Try again! " + data['hint'])
            st.session_state.quiz_answers[q_id] = user_answer

def create_interactive_example(title: str, description: str, callback_fn) -> None:
    """Create an interactive example section with description and visualization."""
    st.subheader(f"🔬 Interactive Example: {title}")
    st.markdown(description)
    callback_fn()

def render_latex(expr):
    """Render LaTeX expression in Streamlit."""
    st.latex(sp.latex(expr))

def plot_probability_distribution(
    x: np.ndarray,
    p: np.ndarray,
    title: str = "Probability Distribution"
) -> go.Figure:
    """Plot a probability distribution."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=p, name="Probability"))
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        width=600,
        height=400
    )
    return fig

def markov_walkthrough_app() -> None:
    """Main Streamlit application for Markov chain walkthrough."""
    st.title("Markov Chain Properties Interactive Walkthrough")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Choose a section:",
        [
            "1. Introduction",
            "2. Probability Foundations",
            "3. Transition Matrices",
            "4. State Evolution",
            "5. Stationary Distribution",
            "6. Eigenvalue Analysis",
            "7. Convergence Properties",
            "8. Bayesian Inference",
            "9. Real-world Applications"
        ]
    )
    
    if section == "1. Introduction":
        st.header("Introduction to Markov Chains")
        
        st.markdown("""
        ### What is a Markov Chain? 🤔
        
        A Markov chain is a mathematical model that describes a sequence of events where
        the probability of each event depends only on the state of the previous event.
        
        ### Key Concepts:
        
        1. **Memoryless Property (Markov Property)** 🧠
           - The future only depends on the present, not the past
           - Like a "goldfish" with no memory of previous states
        
        2. **States and Transitions** 🔄
           - System moves between discrete states
           - Transitions occur with fixed probabilities
        
        3. **Time Steps** ⏱️
           - Changes occur at discrete time intervals
           - Each step represents one transition
        """)
        
        # Simple example
        create_interactive_example(
            "Weather Model",
            """
            Let's model daily weather transitions between Sunny ☀️ and Rainy 🌧️ days.
            
            Try adjusting the probabilities below and observe how they affect the transition matrix:
            """,
            lambda: weather_example()
        )
        
        # Add quiz
        quiz_data = {
            'q1': {
                'question': "What is the key characteristic of a Markov chain?",
                'options': [
                    "It always leads to the same final state",
                    "Future state depends only on the current state",
                    "It requires knowledge of all past states",
                    "Transitions are always deterministic"
                ],
                'correct': "Future state depends only on the current state",
                'explanation': "This is the Markov property - the system's future only depends on its current state.",
                'hint': "Think about the 'memoryless' property we discussed."
            }
        }
        create_quiz_section(quiz_data)
        
    elif section == "2. Probability Foundations":
        st.header("Probability Foundations")
        
        st.markdown("""
        ### Mathematical Foundations 📐
        
        Before diving into Markov chains, let's understand the key probability concepts:
        
        1. **Probability Spaces**
           - Sample space (Ω): All possible outcomes
           - Events (F): Subsets of outcomes
           - Probability measure (P): Assignment of probabilities
        
        2. **Conditional Probability**
           - P(A|B) = P(A ∩ B) / P(B)
           - Foundation for transition probabilities
        
        3. **Independence**
           - P(A ∩ B) = P(A)P(B)
           - Contrast with Markov dependence
        """)
        
        # Interactive probability example
        create_interactive_example(
            "Conditional Probability",
            """
            Explore how conditional probabilities work in a simple scenario.
            Adjust the probabilities and see how they affect the joint distribution.
            """,
            lambda: conditional_probability_example()
        )
        
        st.markdown("""
        ### Mathematical Notation 📝
        
        For a Markov chain {Xₙ}, the key property is:
        """)
        
        render_latex(r"P(X_{n+1} = j | X_n = i, X_{n-1} = i_{n-1}, ..., X_0 = i_0) = P(X_{n+1} = j | X_n = i)")
        
        st.markdown("""
        This is formalized in the transition matrix P where:
        """)
        
        render_latex(r"P_{ij} = P(X_{n+1} = j | X_n = i)")
        
    elif section == "3. Transition Matrices":
        st.header("Transition Matrices")
        
        st.markdown("""
        A transition matrix P represents the probabilities of transitioning from each
        state to every other state. For a matrix to be a valid transition matrix:
        
        1. All elements must be non-negative (probabilities)
        2. Each row must sum to 1 (probability distribution)
        """)
        
        # Interactive transition matrix creation
        st.subheader("Create Your Own Transition Matrix")
        n_states = st.number_input("Number of states", min_value=2, max_value=4, value=2)
        
        matrix_input = []
        for i in range(n_states):
            row = []
            cols = st.columns(n_states)
            for j, col in enumerate(cols):
                val = col.number_input(
                    f"P({i}->{j})",
                    value=1.0/n_states,
                    min_value=0.0,
                    max_value=1.0,
                    key=f"matrix_{i}_{j}"
                )
                row.append(val)
            matrix_input.append(row)
        
        matrix = np.array(matrix_input)
        
        # Validate and visualize
        is_valid = is_stochastic_matrix(matrix)
        if is_valid:
            st.success("✅ Valid transition matrix!")
        else:
            st.error("❌ Not a valid transition matrix. Check that rows sum to 1.")
        
        fig = create_matrix_heatmap(matrix, title="Your Transition Matrix")
        st.plotly_chart(fig)
        
        # Add practical examples
        st.markdown("""
        ### Real-world Examples of Transition Matrices:
        
        1. **Customer Loyalty** 💳
           - States: Different brands/products
           - Transitions: Customer switching behavior
        
        2. **Web Navigation** 🌐
           - States: Different web pages
           - Transitions: User clicking patterns
        
        3. **Population Migration** 🌍
           - States: Different cities/regions
           - Transitions: Migration probabilities
        """)
        
        # Add interactive exercise
        create_interactive_example(
            "Brand Loyalty Model",
            """
            Create a transition matrix for customer brand loyalty between three products:
            - Premium (P)
            - Standard (S)
            - Budget (B)
            
            Think about realistic transition probabilities based on typical customer behavior.
            """,
            lambda: brand_loyalty_example()
        )
        
        st.markdown("""
        ### Matrix Properties 🔢
        
        A transition matrix P must satisfy:
        """)
        
        render_latex(r"P_{ij} \geq 0 \quad \forall i,j")
        render_latex(r"\sum_{j} P_{ij} = 1 \quad \forall i")
        
    elif section == "4. State Evolution":
        st.header("State Evolution")
        
        st.markdown("""
        Given an initial state distribution and a transition matrix, we can compute
        how the state probabilities evolve over time.
        """)
        
        # Example with predefined matrix
        st.subheader("Example: Three-State System")
        matrix = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5]
        ])
        
        states = ["A", "B", "C"]
        chain = MarkovChain(states, matrix)
        
        # Initial state selection
        st.write("Choose initial state probabilities:")
        cols = st.columns(3)
        initial_state = np.array([
            cols[0].number_input("P(A)", value=1.0, min_value=0.0, max_value=1.0),
            cols[1].number_input("P(B)", value=0.0, min_value=0.0, max_value=1.0),
            cols[2].number_input("P(C)", value=0.0, min_value=0.0, max_value=1.0)
        ])
        
        # Normalize initial state
        initial_state = initial_state / initial_state.sum()
        
        # Evolution
        n_steps = st.slider("Number of steps", min_value=5, max_value=50, value=20)
        distributions = chain.evolve_distribution(n_steps, initial_state)
        
        fig = plot_state_distribution(distributions, states)
        st.plotly_chart(fig)
        
    elif section == "5. Stationary Distribution":
        st.header("Stationary Distribution")
        
        st.markdown("""
        A stationary distribution π is a probability distribution that remains unchanged
        after applying the transition matrix: π = πP
        
        Properties:
        1. Exists for any irreducible and aperiodic Markov chain
        2. Unique when it exists
        3. Can be found by:
           - Taking the limit of state distributions
           - Finding the left eigenvector with eigenvalue 1
        """)
        
        # Interactive example
        st.subheader("Find Stationary Distribution")
        matrix = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])
        
        chain = MarkovChain(["A", "B"], matrix)
        stationary = chain.get_stationary_distribution()
        
        fig = create_matrix_heatmap(matrix, "Transition Matrix")
        st.plotly_chart(fig)
        
        st.write("Stationary Distribution:")
        st.write(f"π = [{', '.join(f'{x:.3f}' for x in stationary)}]")
        
        # Verify
        st.write("Verification: πP = π")
        result = stationary @ matrix
        st.write(f"πP = [{', '.join(f'{x:.3f}' for x in result)}]")
        
    elif section == "6. Eigenvalue Analysis":
        st.header("Eigenvalue Analysis")
        
        st.markdown("""
        The eigenvalues and eigenvectors of a transition matrix provide important
        information about the long-term behavior of the Markov chain.
        
        Key Properties:
        1. Largest eigenvalue is always 1
        2. All eigenvalues have magnitude ≤ 1
        3. The eigenvector for eigenvalue 1 gives the stationary distribution
        """)
        
        # Example
        matrix = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5]
        ])
        
        chain = MarkovChain(["A", "B", "C"], matrix)
        eigenvals, eigenvecs = chain.get_eigenvalues_vectors()
        
        st.write("Transition Matrix:")
        fig = create_matrix_heatmap(matrix)
        st.plotly_chart(fig)
        
        st.write("Eigenvalues:")
        for i, val in enumerate(eigenvals):
            st.write(f"λ{i+1} = {val:.3f}")
        
        st.write("Eigenvectors:")
        for i in range(len(eigenvals)):
            st.write(f"v{i+1} = [{', '.join(f'{x:.3f}' for x in eigenvecs[:, i])}]")
        
    elif section == "7. Convergence Properties":
        st.header("Convergence Properties")
        
        st.markdown("""
        The convergence of a Markov chain to its stationary distribution depends on:
        1. **Irreducibility**: All states can be reached from all other states
        2. **Aperiodicity**: The chain doesn't cycle deterministically
        3. **Mixing time**: How quickly the chain converges
        """)
        
        # Interactive convergence demonstration
        st.subheader("Convergence Demonstration")
        
        # Create two different chains
        fast_mixing = np.array([
            [0.6, 0.4],
            [0.4, 0.6]
        ])
        
        slow_mixing = np.array([
            [0.95, 0.05],
            [0.05, 0.95]
        ])
        
        chain_type = st.radio(
            "Select mixing speed:",
            ["Fast mixing", "Slow mixing"]
        )
        
        matrix = fast_mixing if chain_type == "Fast mixing" else slow_mixing
        chain = MarkovChain(["A", "B"], matrix)
        
        n_steps = st.slider(
            "Number of steps",
            min_value=10,
            max_value=100,
            value=50,
            key="conv_steps"
        )
        
        initial_state = np.array([1.0, 0.0])
        distributions = chain.evolve_distribution(n_steps, initial_state)
        
        fig = plot_state_distribution(
            distributions,
            ["A", "B"],
            f"{chain_type} Chain Evolution"
        )
        st.plotly_chart(fig)
        
        stationary = chain.get_stationary_distribution()
        st.write("Stationary Distribution:")
        st.write(f"π = [{', '.join(f'{x:.3f}' for x in stationary)}]")

    elif section == "8. Bayesian Inference":
        st.header("Bayesian Inference in Markov Chains")
        
        st.markdown("""
        ### Bayesian Framework 🎯
        
        Markov chains are closely related to Bayesian inference:
        
        1. **Prior Distribution**
           - Initial state probabilities
           - Based on prior knowledge
        
        2. **Likelihood Function**
           - Transition probabilities
           - Based on observed data
        
        3. **Posterior Distribution**
           - Updated state probabilities
           - Combines prior and likelihood
        """)
        
        # Interactive Bayesian example
        create_interactive_example(
            "Bayesian Update",
            """
            See how Bayesian updates work in a Markov chain context.
            Observe how new evidence updates our beliefs about the system state.
            """,
            lambda: bayesian_update_example()
        )
        
        st.markdown("""
        ### Mathematical Framework
        
        The Bayesian update in Markov chains follows:
        """)
        
        render_latex(r"P(X_n | Y_{1:n}) \propto P(Y_n | X_n) P(X_n | Y_{1:n-1})")
        
        st.markdown("""
        Where:
        - X_n is the state at time n
        - Y_{1:n} are observations up to time n
        """)

    elif section == "9. Real-world Applications":
        st.header("Real-world Applications of Markov Chains")
        
        st.markdown("""
        ### Practical Applications 🌟
        
        Markov chains are used in many real-world scenarios:
        
        1. **Google's PageRank Algorithm** 🔍
           - Web pages as states
           - Links as transitions
           - Ranks pages by importance
        
        2. **Natural Language Processing** 📝
           - Words/characters as states
           - Text generation and analysis
           - Predictive typing
        
        3. **Financial Markets** 📈
           - Market states (Bull/Bear)
           - Stock price movements
           - Risk analysis
        
        4. **Biological Processes** 🧬
           - Gene sequences
           - Population dynamics
           - Disease spread
        """)
        
        # Interactive PageRank example
        create_interactive_example(
            "Simple PageRank",
            """
            Explore how PageRank works with a simple web network.
            Create links between pages and see how it affects their importance scores.
            """,
            lambda: pagerank_example()
        )

def weather_example():
    """Interactive weather transition example."""
    sunny_to_sunny = st.slider(
        "Probability of Sunny → Sunny",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    rainy_to_rainy = st.slider(
        "Probability of Rainy → Rainy",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )
    
    weather_matrix = np.array([
        [sunny_to_sunny, 1-sunny_to_sunny],
        [1-rainy_to_rainy, rainy_to_rainy]
    ])
    
    fig = create_matrix_heatmap(
        weather_matrix,
        title="Weather Transition Matrix",
        height=300
    )
    st.plotly_chart(fig)
    
    # Show prediction
    if st.button("Predict Next 5 Days"):
        chain = MarkovChain(["Sunny", "Rainy"], weather_matrix)
        initial_state = np.array([1, 0])  # Start with Sunny
        distributions = chain.evolve_distribution(5, initial_state)
        
        fig = plot_state_distribution(
            distributions,
            ["Sunny", "Rainy"],
            "Weather Probability Evolution"
        )
        st.plotly_chart(fig)

def brand_loyalty_example():
    """Interactive brand loyalty example."""
    st.write("Enter transition probabilities (each row must sum to 1):")
    
    matrix = np.zeros((3, 3))
    labels = ["Premium", "Standard", "Budget"]
    
    for i, from_brand in enumerate(labels):
        st.write(f"From {from_brand}:")
        cols = st.columns(3)
        total = 0
        for j, to_brand in enumerate(labels):
            val = cols[j].number_input(
                f"To {to_brand}",
                min_value=0.0,
                max_value=1.0,
                value=1/3,
                key=f"brand_{i}_{j}"
            )
            matrix[i, j] = val
            total += val
        
        if not np.isclose(total, 1.0):
            st.warning(f"Row {i+1} sum = {total:.2f} (should be 1.0)")
    
    fig = create_matrix_heatmap(
        matrix,
        title="Brand Loyalty Transition Matrix",
        height=400
    )
    st.plotly_chart(fig)
    
    if st.button("Analyze Long-term Market Share"):
        chain = MarkovChain(labels, matrix)
        stationary = chain.get_stationary_distribution()
        
        st.write("Long-term Market Share Prediction:")
        for brand, share in zip(labels, stationary):
            st.write(f"{brand}: {share:.1%}")

def pagerank_example():
    """Interactive PageRank example."""
    n_pages = 4
    st.write("Create links between web pages (click cells to toggle connections):")
    
    # Initialize adjacency matrix
    if "adjacency" not in st.session_state:
        st.session_state.adjacency = np.zeros((n_pages, n_pages))
    
    # Create clickable grid
    cols = st.columns(n_pages)
    for i in range(n_pages):
        cols[i].write(f"Page {i+1}")
    
    for i in range(n_pages):
        cols = st.columns(n_pages)
        for j in range(n_pages):
            if i != j:  # No self-loops
                if cols[j].button(
                    "🔗" if st.session_state.adjacency[i,j] else "⭕",
                    key=f"link_{i}_{j}"
                ):
                    st.session_state.adjacency[i,j] = 1 - st.session_state.adjacency[i,j]
    
    # Convert to transition matrix
    matrix = st.session_state.adjacency.copy()
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums[:, np.newaxis]
    
    fig = create_matrix_heatmap(
        matrix,
        title="Web Link Transition Matrix",
        height=400
    )
    st.plotly_chart(fig)
    
    if st.button("Calculate PageRank"):
        chain = MarkovChain([f"Page {i+1}" for i in range(n_pages)], matrix)
        ranks = chain.get_stationary_distribution()
        
        st.write("PageRank Scores:")
        for i, rank in enumerate(ranks):
            st.write(f"Page {i+1}: {rank:.3f}")

def conditional_probability_example():
    """Interactive example of conditional probability."""
    st.write("Consider a two-state system with conditional probabilities:")
    
    # Prior probabilities
    p_a = st.slider("P(A)", 0.0, 1.0, 0.5, 0.1)
    p_b_given_a = st.slider("P(B|A)", 0.0, 1.0, 0.7, 0.1)
    p_b_given_not_a = st.slider("P(B|not A)", 0.0, 1.0, 0.3, 0.1)
    
    # Calculate joint probabilities
    p_a_and_b = p_a * p_b_given_a
    p_a_and_not_b = p_a * (1 - p_b_given_a)
    p_not_a_and_b = (1 - p_a) * p_b_given_not_a
    p_not_a_and_not_b = (1 - p_a) * (1 - p_b_given_not_a)
    
    # Create probability table
    data = np.array([
        [p_a_and_b, p_a_and_not_b],
        [p_not_a_and_b, p_not_a_and_not_b]
    ])
    
    fig = create_matrix_heatmap(
        data,
        title="Joint Probability Matrix",
        height=300
    )
    st.plotly_chart(fig)
    
    # Calculate marginal probability of B
    p_b = p_a_and_b + p_not_a_and_b
    st.write(f"Marginal probability P(B) = {p_b:.3f}")

def bayesian_update_example():
    """Interactive example of Bayesian updates in Markov chains."""
    st.write("Consider a hidden Markov model with two states (Sunny/Rainy) and noisy observations.")
    
    # Prior probabilities
    prior_sunny = st.slider("Prior P(Sunny)", 0.0, 1.0, 0.5, 0.1)
    
    # Likelihood probabilities
    p_observe_correct = st.slider("P(Observe correctly)", 0.5, 1.0, 0.7, 0.1)
    
    # Observation
    observation = st.radio("Observation", ["Sunny", "Rainy"])
    
    # Calculate posterior
    if observation == "Sunny":
        likelihood_ratio = p_observe_correct / (1 - p_observe_correct)
    else:
        likelihood_ratio = (1 - p_observe_correct) / p_observe_correct
    
    posterior_odds = (prior_sunny / (1 - prior_sunny)) * likelihood_ratio
    posterior_sunny = posterior_odds / (1 + posterior_odds)
    
    # Plot prior and posterior
    states = ["Sunny", "Rainy"]
    prior = np.array([prior_sunny, 1 - prior_sunny])
    posterior = np.array([posterior_sunny, 1 - posterior_sunny])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=states,
        y=prior,
        name="Prior"
    ))
    fig.add_trace(go.Bar(
        x=states,
        y=posterior,
        name="Posterior"
    ))
    
    fig.update_layout(
        title="Bayesian Update",
        xaxis_title="State",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        barmode='group'
    )
    
    st.plotly_chart(fig)
    
    st.write(f"Posterior P(Sunny) = {posterior_sunny:.3f}")

if __name__ == "__main__":
    markov_walkthrough_app() 