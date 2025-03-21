from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Get the directory of the script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define model paths
diabetes_model_path = os.path.join(working_dir, "models", "diabetes_model.sav")
heart_disease_model_path = os.path.join(working_dir, "models", "heart_disease_model.sav")
parkinsons_model_path = os.path.join(working_dir, "models", "parkinsons_model.sav")

# Load models
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))
parkinsons_model = pickle.load(open(parkinsons_model_path, 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    model_type = request.form.get("model")

    if model_type == "diabetes":
        prediction = diabetes_model.predict(np.array(data).reshape(1, -1))
    elif model_type == "heart":
        prediction = heart_disease_model.predict(np.array(data).reshape(1, -1))
    elif model_type == "parkinsons":
        prediction = parkinsons_model.predict(np.array(data).reshape(1, -1))
    else:
        return jsonify({"error": "Invalid model type"}), 400

    result = "Positive" if prediction[0] == 1 else "Negative"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
