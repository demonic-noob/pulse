import os
import joblib
import pandas as pd
import streamlit as st
import xgboost

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model_mars_storm.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("Mars Global Storm Predictor")
st.write("Adjust the atmospheric telemetry parameters below to predict the likelihood of a global dust storm within the next 3 sols.")

st.subheader("Telemetry Inputs")
col1, col2 = st.columns(2)

with col1:
    latitude = st.slider("Latitude", -90.0, 90.0, -15.5)
    longitude = st.slider("Longitude", 0.0, 360.0, 120.0)
    ls = st.slider("Solar Longitude (Ls)", 0.0, 360.0, 145.2)

with col2:
    surf_pressure = st.number_input("Surface Pressure (Pa)", min_value=100.0, max_value=1000.0, value=610.5)
    temp_differential = st.number_input("Temperature Differential (K)", min_value=0.0, max_value=100.0, value=15.2)
    opacity_tau = st.number_input("Dust Opacity (Tau)", min_value=0.0, max_value=10.0, value=0.8)

input_data = pd.DataFrame({
    "latitude": [latitude],
    "longitude": [longitude],
    "ls": [ls],
    "surf_pressure": [surf_pressure],
    "temp_differential": [temp_differential],
    "opacity_tau": [opacity_tau]
})

if st.button("Run Prediction", type="primary"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction == 1.0:
        confidence = probabilities[1] * 100
        st.error(f"**Global Storm Predicted** with **{confidence:.2f}%** confidence!")
    else:
        confidence = probabilities[0] * 100
        st.success(f"**Clear Weather** with **{confidence:.2f}%** confidence!")

# to run the app, use the command: streamlit run predict.py
