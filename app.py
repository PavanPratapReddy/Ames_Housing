import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import io

# Define the same model architecture (Must match the saved model)
class HousePriceModel(nn.Module):
    def __init__(self):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = HousePriceModel()

# Load trained model
MODEL_PATH = "model.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
    st.sidebar.success("✅ Model loaded successfully")
except FileNotFoundError:
    st.sidebar.error("❌ Model file not found! Please check deployment.")

# Load the scaler for preprocessing
SCALER_PATH = "scaler.pkl"
try:
    scaler = joblib.load(SCALER_PATH)
    st.sidebar.success("✅ Scaler loaded successfully")
except FileNotFoundError:
    st.sidebar.error("❌ Scaler file not found!")

# Streamlit UI
st.title("🏡 House Price Prediction")
st.write("Upload a dataset or enter house details below to predict the price.")

# 📂 Upload Dataset Section
uploaded_file = st.file_uploader("📂 Upload a CSV file (optional)", type=["csv"])
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of uploaded dataset:")
    st.write(df.head())

    # Save file locally (if needed)
    with open("uploaded_dataset.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

# 🔢 Manual Input Section
st.subheader("Or enter house details manually:")
LotArea = st.number_input("Lot Area (sq ft)", value=8450)
YearBuilt = st.number_input("Year Built", value=2003)
OverallQual = st.number_input("Overall Quality (1-10)", value=7)
TotalBsmtSF = st.number_input("Total Basement SF", value=856)
GrLivArea = st.number_input("Above Grade Living Area SF", value=1710)
GarageCars = st.number_input("Garage Capacity (Cars)", value=2)

# Prepare input data
input_data = np.array([[LotArea, YearBuilt, OverallQual, TotalBsmtSF, GrLivArea, GarageCars]], dtype=np.float32)

# Normalize using pre-saved scaler
if "scaler" in locals():
    input_data_scaled = scaler.transform(input_data)
else:
    st.error("❌ Scaler not loaded. Unable to process input.")
    input_data_scaled = input_data  # Use raw data if scaler is missing

# Convert to tensor
input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

# Predict price
if st.button("🔮 Predict Price"):
    if "model" in locals():
        with torch.no_grad():
            predicted_price = model(input_tensor).item()
        st.success(f"🏠 Predicted House Price: ${predicted_price:,.2f}")
    else:
        st.error("❌ Model not loaded. Cannot make predictions.")


