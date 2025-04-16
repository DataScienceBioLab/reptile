"""Main Streamlit application for educational content."""

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Any

from educational_app.utils.data_utils import (
    generate_distribution_data,
    calculate_distribution_stats,
    get_distribution_info
)

# Configure the Streamlit page
st.set_page_config(
    page_title="Educational App",
    page_icon="📚",
    layout="wide",
)

# Add custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton button {
        width: 100%;
    }
    .stTextInput input {
        border-radius: 5px;
    }
    .stats-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .example-container {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    st.title("Interactive Educational Form")
    st.markdown("### Learn through interactive visualization")

    # Create a form
    with st.form("educational_form"):
        # User inputs
        name = st.text_input("Your Name", key="name")
        age = st.number_input("Your Age", min_value=1, max_value=120, value=25)
        education_level = st.selectbox(
            "Education Level",
            ["High School", "Bachelor's", "Master's", "PhD", "Other"]
        )
        
        # Interactive visualization parameters
        st.subheader("Customize Your Learning")
        num_points = st.slider("Number of Data Points", 10, 1000, 100)
        distribution = st.selectbox(
            "Select Distribution",
            ["Normal", "Uniform", "Exponential"]
        )

        # Distribution parameters
        st.subheader("Distribution Parameters")
        params: Dict[str, float] = {}
        
        if distribution == "Normal":
            params["mu"] = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
            params["sigma"] = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)
        elif distribution == "Uniform":
            params["low"] = st.slider("Lower Bound", -10.0, 0.0, -3.0, 0.5)
            params["high"] = st.slider("Upper Bound", 0.0, 10.0, 3.0, 0.5)
        else:  # Exponential
            params["scale"] = st.slider("Scale (1/λ)", 0.1, 5.0, 1.0, 0.1)

        # Submit button
        submitted = st.form_submit_button("Generate Visualization")

        if submitted:
            # Generate data
            data, title = generate_distribution_data(distribution, num_points, params)
            
            # Create DataFrame
            df = pd.DataFrame({
                "Values": data,
                "Count": range(len(data))
            })

            # Create interactive plot
            fig = px.histogram(
                df,
                x="Values",
                title=title,
                template="plotly_white",
                marginal="box"  # Add box plot on the margin
            )
            fig.update_layout(
                showlegend=False,
                title_x=0.5,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Display statistics
            stats = calculate_distribution_stats(data)
            st.markdown("### Distribution Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
                st.metric("Standard Deviation", f"{stats['std']:.2f}")
            with col2:
                st.metric("Median", f"{stats['median']:.2f}")
                st.metric("Minimum", f"{stats['min']:.2f}")
            with col3:
                st.metric("Maximum", f"{stats['max']:.2f}")

            # Educational content
            info = get_distribution_info(distribution)
            st.markdown("### Understanding the Distribution")
            st.write(info["description"])
            
            st.markdown("#### Key Characteristics")
            for char in info["characteristics"]:
                st.markdown(f"- {char}")
            
            st.markdown("#### Real-world Examples")
            with st.container():
                st.markdown('<div class="example-container">', unsafe_allow_html=True)
                for example in info["examples"]:
                    st.markdown(f"- {example}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Interactive quiz
            st.markdown("### Quick Quiz")
            correct_answer = info["characteristics"][0]
            options = [
                correct_answer,
                "Always positive skewness",
                "Always negative kurtosis"
            ]
            # Shuffle options
            np.random.shuffle(options)
            
            quiz_answer = st.radio(
                f"What is a key characteristic of the {distribution} distribution?",
                options,
                key="quiz"
            )
            
            if st.button("Check Answer"):
                if quiz_answer == correct_answer:
                    st.success("🎉 Correct! Well done!")
                else:
                    st.error("Not quite right. Try again!")
                    st.info(f"Hint: The correct answer is related to the shape of the distribution.")

if __name__ == "__main__":
    main() 