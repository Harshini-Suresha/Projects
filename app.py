import streamlit as st
import pandas as pd
import joblib

# Load the trained model and data
model = joblib.load("model.pkl")  # Pre-trained ML model
disease_list = joblib.load("diseases_list.pkl")  # List of unique diseases
label_encoders = joblib.load("label_encoders.pkl")  # Encoders for categorical columns

st.set_page_config(page_title="Drug & Dosage Predictor", layout="centered")

# App Title
st.title("Drug and Dosage Predictor")
st.markdown("Use this app to predict the recommended drug and dosage based on the patient's disease.")

# Sidebar for input
st.sidebar.header("Disease Input")

# Dropdown to select disease
selected_disease = st.sidebar.selectbox("Select a Disease", sorted(disease_list))

# Predict button
if st.sidebar.button("Predict Drug & Dosage"):
    # Encode input
    input_data = pd.DataFrame({"Diagnosis": [selected_disease]})
    for col in input_data.columns:
        le = label_encoders.get(col)
        if le:
            input_data[col] = le.transform(input_data[col])

    # Predict
    prediction = model.predict(input_data)[0]

    # Decode prediction if it was label-encoded
    decoded_prediction = prediction
    if "Target" in label_encoders:
        decoded_prediction = label_encoders["Target"].inverse_transform([prediction])[0]

    st.success(f"**Recommended Drug/Dosage:** {decoded_prediction}")

# Optional: show disease list
if st.checkbox("Show All Diseases"):
    st.write(sorted(disease_list))
