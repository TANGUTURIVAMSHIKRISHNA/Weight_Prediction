import streamlit as st
import joblib

# Load the trained model
model = joblib.load('weight_predictor.pkl')

# Streamlit UI
st.markdown("<h3 style='text-align: center; color: orange;'>Weight Prediction App</h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: purple;'>Enter your height in feet to predict your weight.</h5>", unsafe_allow_html=True)

# Input from user
height = st.number_input("Enter the height in feet:", min_value=3.0, max_value=8.0, step=0.01, value=5.80)

# Predict button
if st.button("Predict"):
    weight_pred = model.predict([[height]])
    st.success(f"Predicted Weight: {weight_pred[0]:.2f} kg")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
