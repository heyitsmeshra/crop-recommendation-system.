import streamlit as st
import joblib
import pandas as pd

model=joblib.load("random_forest_model.pkl")
encoder=joblib.load("label_encoder.pkl")

st.title("🌱 Crop Recommendation System")
st.write("Enter the soil and weather parameters to find a suitable crop.")

N=st.number_input("Nitrogen (N)",min_value=0.0,value=50.0)
P=st.number_input("Phosphorus (P)",min_value=0.0,value=50.0)
K=st.number_input("Potassium (K)",min_value=0.0,value=50.0)
temperature=st.number_input("Temperature (°C)",value=25.0)
humidity=st.number_input("Humidity (%)",min_value=0.0,max_value=100.0,value=70.0)
ph=st.number_input("Soil pH",min_value=0.0,max_value=14.0,value=6.5)
rainfall=st.number_input("Rainfall (mm)",min_value=0.0,value=100.0)

if st.button("Recommend Crop"):
    input_data=pd.DataFrame(
        [[N,P,K,temperature,humidity,ph,rainfall]],
        columns=["N","P","K","temperature","humidity","ph","rainfall"]
    )

    prediction=model.predict(input_data)
    crop=encoder.inverse_transform(prediction)[0]

    st.success(f"Recommended Crop: {crop.title()}")
