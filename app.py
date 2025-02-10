import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the same model architecture
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

# Load trained model
model = HousePriceModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()  # Set to evaluation mode

# Streamlit UI
st.title("🏡 House Price Prediction")
st.write("Enter house details below to predict the price.")

# Input fields
LotArea = st.number_input("Lot Area (sq ft)", value=8450)
YearBuilt = st.number_input("Year Built", value=2003)
OverallQual = st.number_input("Overall Quality (1-10)", value=7)
TotalBsmtSF = st.number_input("Total Basement SF", value=856)
GrLivArea = st.number_input("Above Grade Living Area SF", value=1710)
GarageCars = st.number_input("Garage Capacity (Cars)", value=2)

# Prepare input data
input_data = np.array([[LotArea, YearBuilt, OverallQual, TotalBsmtSF, GrLivArea, GarageCars]], dtype=np.float32)

# Normalize using pre-saved scaler values (use the same scaling as training)
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Convert to tensor
input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

# Predict price
if st.button("Predict Price"):
    with torch.no_grad():
        predicted_price = model(input_tensor).item()
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")


