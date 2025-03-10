import streamlit as st
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), "src")
logger.debug(f"Adding to Python path: {src_path}")
sys.path.append(src_path)

def main():
    st.sidebar.title("Select App")
    app_choice = st.sidebar.radio(
        "Choose which app to run:",
        ["Markov Chain Walkthrough", "Mouse House Simulation"]
    )
    
    try:
        if app_choice == "Markov Chain Walkthrough":
            logger.debug("Importing markov_walkthrough_app")
            from markov_rl.visualization.markov_walkthrough import markov_walkthrough_app
            logger.debug("Running markov_walkthrough_app")
            markov_walkthrough_app()
        else:
            logger.debug("Importing mouse house app")
            from markov_rl.visualization.mouse_house_walkthrough import mouse_house_walkthrough
            logger.debug("Running mouse house app")
            mouse_house_walkthrough()
    except Exception as e:
        logger.error(f"Error running app: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 