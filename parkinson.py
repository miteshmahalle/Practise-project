#!/usr/bin/env python  
# coding: utf-8

# Importing the Dependencies
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
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

# Data Collection & Analysis
parkinsons_csv_path = r"C:\Users\HP\multiple-disease-prediction\dataset\parkinsons.csv"

try:
    parkinsons_data = pd.read_csv(parkinsons_csv_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {parkinsons_csv_path}. Please check the file location.")
    exit()

print("First 5 rows:\n", parkinsons_data.head())
print("Dataset shape:", parkinsons_data.shape)
print("Dataset info:")
parkinsons_data.info()
print("Missing values count:\n", parkinsons_data.isnull().sum())
print("Dataset statistics:\n", parkinsons_data.describe())
print("Target value counts:\n", parkinsons_data['status'].value_counts())

numeric_columns = parkinsons_data.select_dtypes(include=['number']).columns
print("Mean values grouped by status:\n", parkinsons_data[numeric_columns].groupby(parkinsons_data['status']).mean())

# Data Preprocessing
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

print("Features:\n", X.head())
print("Target:\n", Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print("Shapes:", X.shape, X_train.shape, X_test.shape)

# Model Training - Support Vector Machine Model
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Training Accuracy:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Test Accuracy:', test_data_accuracy)

# Saving the trained model
model_filename = "parkinsons_model.sav"
pickle.dump(model, open(model_filename, 'wb'))
print("Model saved successfully!")

# Load the saved model
loaded_model = pickle.load(open(model_filename, 'rb'))

# Streamlit UI for Parkinson's Prediction
st.title("Parkinson's Disease Prediction")
st.markdown("### Enter the following parameters:")

MDVP_Fo = st.number_input("MDVP:Fo(Hz) (85 - 260)")
MDVP_Fhi = st.number_input("MDVP:Fhi(Hz) (100 - 600)")
MDVP_Flo = st.number_input("MDVP:Flo(Hz) (50 - 250)")
MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%) (0.001 - 0.03)")
MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs) (0.00001 - 0.002)")
MDVP_RAP = st.number_input("MDVP:RAP (0.001 - 0.02)")
MDVP_PPQ = st.number_input("MDVP:PPQ (0.001 - 0.02)")
Jitter_DDP = st.number_input("Jitter:DDP (0.002 - 0.06)")
MDVP_Shimmer = st.number_input("MDVP:Shimmer (0.01 - 1.0)")
MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB) (0.01 - 1.0)")
Shimmer_APQ3 = st.number_input("Shimmer:APQ3 (0.005 - 0.05)")
Shimmer_APQ5 = st.number_input("Shimmer:APQ5 (0.005 - 0.06)")
MDVP_APQ = st.number_input("MDVP:APQ (0.005 - 0.08)")
Shimmer_DDA = st.number_input("Shimmer:DDA (0.01 - 0.15)")
NHR = st.number_input("NHR (0.001 - 0.3)")
HNR = st.number_input("HNR (10 - 35)")
RPDE = st.number_input("RPDE (0.2 - 0.7)")
DFA = st.number_input("DFA (0.5 - 1.0)")
spread1 = st.number_input("spread1 (-8 - -2)")
spread2 = st.number_input("spread2 (0 - 0.6)")
D2 = st.number_input("D2 (1.5 - 3.5)")
PPE = st.number_input("PPE (0.02 - 0.6)")

if st.button("Test the Result"):
    input_data = np.asarray([MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, 
                             Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, 
                             Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]).reshape(1, -1)
    prediction = loaded_model.predict(input_data)
    
    if prediction[0] == 0:
        st.success("The person does not have Parkinson's Disease.")
    else:
        st.error("The person has Parkinson's Disease.")
