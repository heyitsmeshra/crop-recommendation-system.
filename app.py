
import streamlit as st
import joblib
import pandas as pd
import numpy as np

model=joblib.load("random_forest_model.pkl")
encoder=joblib.load("label_encoder.pkl")

st.title("🌱 Crop Recommendation System")
st.write("Enter the soil and weather parameters to find the top 3 suitable crops.")

N=st.number_input("Nitrogen (N)",min_value=0.0,value=50.0)
P=st.number_input("Phosphorus (P)",min_value=0.0,value=50.0)
K=st.number_input("Potassium (K)",min_value=0.0,value=50.0)
temperature=st.number_input("Temperature (°C)",value=25.0)
humidity=st.number_input("Humidity (%)",min_value=0.0,max_value=100.0,value=70.0)
ph=st.number_input("Soil pH",min_value=0.0,max_value=14.0,value=6.5)
rainfall=st.number_input("Rainfall (mm)",min_value=0.0,value=100.0)

if st.button("Recommend Top 3 Crops"):
    input_data=pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]],columns=["N","P","K","temperature","humidity","ph","rainfall"])
    probabilities=model.predict_proba(input_data)[0]
    top_indices=np.argsort(probabilities)[::-1][:3]
    top_class_labels=model.classes_[top_indices]
    top_crops=encoder.inverse_transform(top_class_labels)
    top_probabilities=probabilities[top_indices]
    st.subheader("🌾 Top 3 Crop Recommendations")
    for rank,(crop,probability) in enumerate(zip(top_crops,top_probabilities),start=1):
        st.success(f"{rank}. {crop.title()} — Model score: {probability*100:.2f}%")
    st.caption("These percentages are model-predicted probabilities, not a guarantee of crop suitability or yield.")
