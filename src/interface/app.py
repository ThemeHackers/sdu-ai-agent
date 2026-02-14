import streamlit as st
import time
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.brain import SmartBrain
from src.interface.components.styles import get_custom_css
from src.interface.components.sidebar import render_sidebar
from src.interface.components.chat_interface import render_hero_section, render_chat_history, handle_user_input

st.set_page_config(
    page_title="SDU AI Agent | พี่สวนดุสิต",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_brain():
    return SmartBrain()

try:
    brain = load_brain()
except Exception as e:
    st.error(f"Failed to initialize AI Brain: {e}")
    st.stop()


st.markdown(get_custom_css(), unsafe_allow_html=True)


render_sidebar()


render_hero_section()
render_chat_history()
handle_user_input(brain)
