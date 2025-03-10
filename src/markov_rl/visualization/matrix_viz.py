"""Interactive visualization for matrix operations.

This module provides interactive visualization tools for matrix operations using
Streamlit and Plotly. It allows users to:
1. Input and visualize matrices
2. Apply various matrix operations
3. See step-by-step transformations
4. Validate stochastic properties
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Optional, Tuple, List
from ..utils.matrix_ops import (
    is_stochastic_matrix,
    normalize_matrix_rows,
    is_symmetric_matrix,
    make_symmetric,
    clip_probability_matrix
)

def create_matrix_heatmap(
    matrix: np.ndarray,
    title: str = "Matrix Visualization",
    show_values: bool = True,
    colorscale: str = "RdYlBu_r"
) -> go.Figure:
    """Create an interactive heatmap visualization of a matrix.
    
    Args:
        matrix: The matrix to visualize
        title: Title for the heatmap
        show_values: Whether to show values in cells
        colorscale: Plotly colorscale to use
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        text=matrix if show_values else None,
        texttemplate="%{text:.3f}" if show_values else None,
        colorscale=colorscale,
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        width=600,
        height=400
    )
    
    return fig

def visualize_matrix_operation(
    input_matrix: np.ndarray,
    operation_name: str,
    result_matrix: np.ndarray,
    intermediate_steps: Optional[List[Tuple[str, np.ndarray]]] = None
) -> None:
    """Visualize a matrix operation with intermediate steps.
    
    Args:
        input_matrix: Original matrix
        operation_name: Name of the operation being performed
        result_matrix: Final result matrix
        intermediate_steps: Optional list of (step_name, matrix) tuples
    """
    st.subheader(f"Matrix {operation_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Input Matrix")
        fig_input = create_matrix_heatmap(
            input_matrix,
            title="Input Matrix",
            colorscale="Viridis"
        )
        st.plotly_chart(fig_input)
        
        if is_stochastic_matrix(input_matrix):
            st.success("✅ Input is a valid stochastic matrix")
        else:
            st.warning("⚠️ Input is not a stochastic matrix")
    
    with col2:
        st.write("Result Matrix")
        fig_result = create_matrix_heatmap(
            result_matrix,
            title=f"After {operation_name}",
            colorscale="Viridis"
        )
        st.plotly_chart(fig_result)
        
        if is_stochastic_matrix(result_matrix):
            st.success("✅ Result is a valid stochastic matrix")
        else:
            st.warning("⚠️ Result is not a stochastic matrix")
    
    if intermediate_steps:
        st.subheader("Intermediate Steps")
        for step_name, step_matrix in intermediate_steps:
            st.write(f"Step: {step_name}")
            fig_step = create_matrix_heatmap(
                step_matrix,
                title=step_name,
                colorscale="Viridis"
            )
            st.plotly_chart(fig_step)

def matrix_operation_app() -> None:
    """Main Streamlit application for matrix operations visualization."""
    st.title("Matrix Operations Visualization")
    
    # Matrix input
    st.header("Input Matrix")
    n_rows = st.number_input("Number of rows", min_value=1, max_value=5, value=2)
    n_cols = st.number_input("Number of columns", min_value=1, max_value=5, value=2)
    
    # Create input matrix
    matrix_input = []
    for i in range(n_rows):
        row = []
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            val = col.number_input(
                f"Value [{i},{j}]",
                value=1.0/(n_cols),  # Initialize with uniform probabilities
                key=f"matrix_{i}_{j}"
            )
            row.append(val)
        matrix_input.append(row)
    
    matrix = np.array(matrix_input)
    
    # Operation selection
    operation = st.selectbox(
        "Select Operation",
        ["Normalize Rows", "Make Symmetric", "Clip Probabilities"]
    )
    
    if st.button("Apply Operation"):
        try:
            if operation == "Normalize Rows":
                result = normalize_matrix_rows(matrix)
                visualize_matrix_operation(matrix, "Row Normalization", result)
                
            elif operation == "Make Symmetric":
                if matrix.shape[0] != matrix.shape[1]:
                    st.error("Matrix must be square for symmetrization")
                else:
                    result = make_symmetric(matrix)
                    steps = [
                        ("Original", matrix),
                        ("Transpose", matrix.T),
                        ("Average", result)
                    ]
                    visualize_matrix_operation(
                        matrix,
                        "Symmetrization",
                        result,
                        intermediate_steps=steps
                    )
                    
            elif operation == "Clip Probabilities":
                min_val = st.slider("Minimum Value", 0.0, 0.5, 0.0, 0.1)
                max_val = st.slider("Maximum Value", 0.5, 1.0, 1.0, 0.1)
                result = clip_probability_matrix(matrix, min_val, max_val)
                steps = [
                    ("Original", matrix),
                    ("Clipped", np.clip(matrix, min_val, max_val)),
                    ("Normalized", result)
                ]
                visualize_matrix_operation(
                    matrix,
                    "Probability Clipping",
                    result,
                    intermediate_steps=steps
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    matrix_operation_app() 