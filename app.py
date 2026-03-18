import streamlit as st
import pickle
import numpy as np

with open("model.pkl", "rb") as file:
    model = pickle.load(file)
st.title("House Price Prediction App")

area = st.number_input("Area (sq ft)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")

if st.button("Predict Price"):

    features = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(features)

    st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")