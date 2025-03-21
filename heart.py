#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st


# Page Configuration (Removing Sidebar)
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Hide Sidebar Completely
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Home Button (Top Right)
col1, col2 = st.columns([8, 1])

with col2:
    if st.button("Home", key="home_btn", help="Go to MedVision's Predictive System", use_container_width=True, type="primary"):
        st.switch_page("app.py")  # Redirect to the main page

# Data Collection and Processing
# Define the correct absolute path for the dataset
heart_csv_path = r"C:\Users\HP\multiple-disease-prediction\dataset\heart.csv"

# Load dataset with error handling
try:
    heart_data = pd.read_csv(heart_csv_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {heart_csv_path}. Please check the file location.")
    exit()

# Splitting the Features and Target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting the Data into Training & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training - Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# Saving the trained model
model_filename = "heart_disease_model.sav"
pickle.dump(model, open(model_filename, 'wb'))

# Streamlit UI for Heart Disease Prediction
st.title("Heart Disease Prediction")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.number_input("Chest Pain types", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.number_input("Resting Electrocardiographic results", min_value=0, max_value=2, value=1)
thalach = st.number_input("Maximum Heart Rate achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.number_input("Slope of the peak exercise ST segment", min_value=0, max_value=2, value=1)
ca = st.number_input("Major vessels colored by fluoroscopy", min_value=0, max_value=4, value=0)
thal = st.number_input("Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect", min_value=0, max_value=3, value=1)

# Predict button
if st.button("Test the Result"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success("The person does not have Heart Disease")
    else:
        st.error("The person has Heart Disease")
