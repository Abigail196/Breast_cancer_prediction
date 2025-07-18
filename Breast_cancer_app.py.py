import streamlit as st
import numpy as np
import joblib

# Load saved pipeline
model = joblib.load("logreg_breastcancer_model.pkl")

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🧬 Breast Cancer Diagnosis App")

st.markdown("""
This tool uses a logistic regression model to predict whether a tumor is **cancerous** or **non-cancerous** based on input features from a medical dataset.
""")

# Define input features
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.subheader("Enter Tumor Measurements")

# Split inputs into two columns
cols = st.columns(2)
inputs = []

for i, feature in enumerate(features):
    col = cols[i % 2]
    value = col.number_input(f"{feature}", min_value=0.0, value=1.0, format="%.4f")
    inputs.append(value)

if st.button("🔍 Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][1]

    result = "🟥 CANCEROUS" if prediction == 1 else "🟩 NON-CANCEROUS"
    st.success(f"Prediction: {result}")
    st.write(f"Confidence Score: `{round(confidence, 4)}`")

    # Optional: Add a visual gauge
    st.progress(confidence if prediction == 1 else 1 - confidence)

