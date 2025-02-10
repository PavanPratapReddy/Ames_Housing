import streamlit as st
import torch
import numpy as np
import json

# Load your trained PyTorch model (ensure you have the model file in the same directory)
MODEL_PATH = "model.pth"  # Update this path with your actual model file
model = torch.load(MODEL_PATH)
model.eval()  # Set to evaluation mode

# Define a function to make predictions
def predict_house_price(input_data):
    input_tensor = torch.tensor(list(input_data.values()), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

# Streamlit UI
st.title("House Price Prediction Service")
st.write("Enter the house features below:")

# Define input fields
LotArea = st.number_input("Lot Area", value=8450)
YearBuilt = st.number_input("Year Built", value=2003)
OverallQual = st.number_input("Overall Quality (1-10)", value=7)
TotalBsmtSF = st.number_input("Total Basement SF", value=856)
GrLivArea = st.number_input("Above Grade Living Area SF", value=1710)
GarageCars = st.number_input("Garage Capacity (Cars)", value=2)

# Collect inputs
input_data = {
    "LotArea": LotArea,
    "YearBuilt": YearBuilt,
    "OverallQual": OverallQual,
    "TotalBsmtSF": TotalBsmtSF,
    "GrLivArea": GrLivArea,
    "GarageCars": GarageCars
}

# Make prediction on button click
if st.button("Predict Price"):
    predicted_price = predict_house_price(input_data)
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")

