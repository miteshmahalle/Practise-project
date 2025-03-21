import os
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(page_title="MedVision's Predictive System", layout="wide", page_icon="🧑‍⚕️")

# Hide sidebar completely
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# Get the directory of the script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define model paths correctly
diabetes_model_path = os.path.join(working_dir, "saved_models", "diabetes_model.sav")
heart_disease_model_path = os.path.join(working_dir, "saved_models", "heart_disease_model.sav")
parkinsons_model_path = os.path.join(working_dir, "saved_models", "parkinsons_model.sav")

# Load models
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))
parkinsons_model = pickle.load(open(parkinsons_model_path, 'rb'))

# Correct CSV paths
diabetes_csv_path = os.path.join(working_dir, "dataset", "diabetes.csv")
heart_disease_csv_path = os.path.join(working_dir, "dataset", "heart.csv")
parkinsons_csv_path = os.path.join(working_dir, "dataset", "parkinsons.csv")

# Load datasets
try:
    diabetes_dataset = pd.read_csv(diabetes_csv_path)
    heart_disease_dataset = pd.read_csv(heart_disease_csv_path)
    parkinsons_dataset = pd.read_csv(parkinsons_csv_path)
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please make sure the CSV files are in the 'data/' directory.")

# Create a layout with the "Home" button at the top right
col1, col2 = st.columns([8, 1])

with col2:
    if st.button("Home", key="home_btn", help="Go to MedVision's Predictive System", use_container_width=True, type="primary"):
        st.switch_page("app.py")  # Redirect to the main page

st.title("MedVision's Predictive System")
st.markdown("### Select a disease prediction below:")

col1, col2, col3 = st.columns(3)

# 🔷 Diabetes Prediction Box
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2927/2927495.png", width=80)
    st.markdown("### Diabetes Prediction")
    if st.button("Check Now", key="diabetes_btn", help="Go to Diabetes Prediction Page", use_container_width=True, type="primary"):
        st.switch_page("pages/diabetes.py")

# ❤️ Heart Disease Prediction Box (Updated with Red Heart Icon & Fancy Look)
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=80)  # Red Heart Icon
    st.markdown("### **Heart Disease Prediction**")
    if st.button("Check Now", key="heart_btn", help="Go to Heart Disease Prediction Page", use_container_width=True, type="primary"):
        st.switch_page("pages/heart.py")

# 🧠 Parkinson's Prediction Box
with col3:
    st.image("https://cdn-icons-png.flaticon.com/512/4140/4140047.png", width=80)
    st.markdown("### Parkinson's Prediction")
    if st.button("Check Now", key="parkinson_btn", help="Go to Parkinson's Prediction Page", use_container_width=True, type="primary"):
        st.switch_page("pages/parkinson.py")
