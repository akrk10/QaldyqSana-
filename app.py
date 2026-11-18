import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

st.set_page_config(
page_title="QaldyqSana",
page_icon="♻️",
layout="wide"
)

st.markdown("""

""", unsafe_allow_html=True)

st.title("♻️ QaldyqSana")
st.caption("AI-Powered Waste Classification Platform")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
"Choose section",
["Dashboard", "AI Scanner", "About"]
)

if page == "Dashboard":
    st.header("Waste Analytics Dashboard")
st.info("Upload waste images in AI Scanner to begin.")

elif page == "AI Scanner":
    st.header("AI Waste Scanner")
uploaded = st.file_uploader(
"Upload a waste image",
type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image",
             use_container_width=True)
    st.warning("YOLO11n model integration is coming next.")
elif page == "About":
    st.header("About QaldyqSana")
st.write(
"An AI-powered platform designed to "
"classify waste into five recyclable categories."
)
st.write(
"Plastic | Paper/Cardboard | Metal | Glass | Other"
)
