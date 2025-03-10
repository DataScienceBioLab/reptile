import streamlit as st
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add projects directory to Python path
projects_path = os.path.join(os.path.dirname(__file__), "projects")
logger.debug(f"Adding to Python path: {projects_path}")
sys.path.append(projects_path)

def main():
    st.sidebar.title("Select Project")
    project_choice = st.sidebar.radio(
        "Choose which project to run:",
        ["Markov Mouse", "CSE Algorithms", "Cursor RL"]
    )
    
    try:
        if project_choice == "Markov Mouse":
            st.sidebar.title("Select App")
            app_choice = st.sidebar.radio(
                "Choose which app to run:",
                ["Markov Chain Walkthrough", "Mouse House Simulation"]
            )
            
            if app_choice == "Markov Chain Walkthrough":
                logger.debug("Importing markov_walkthrough")
                from markov_mouse.src.visualization.markov_walkthrough import markov_walkthrough
                logger.debug("Running markov_walkthrough")
                markov_walkthrough()
            else:
                logger.debug("Importing mouse_house_walkthrough")
                from markov_mouse.src.visualization.mouse_house_walkthrough import mouse_house_walkthrough
                logger.debug("Running mouse_house_walkthrough")
                mouse_house_walkthrough()
                
        elif project_choice == "CSE Algorithms":
            st.info("CSE Algorithms project selection coming soon...")
            
        else:  # Cursor RL
            st.info("Cursor RL project selection coming soon...")
            
    except Exception as e:
        logger.error(f"Error running app: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 