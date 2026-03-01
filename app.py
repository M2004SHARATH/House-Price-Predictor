import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ==============================
# Load Model & Columns
# ==============================
model = joblib.load("random_forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ==============================
# Custom CSS Styling
# ==============================
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        color: #1f4e79;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        font-size: 24px;
        font-weight: bold;
        color: #0b8457;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Header
# ==============================
st.markdown('<div class="title">🏠 House Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict property prices using Machine Learning</div>', unsafe_allow_html=True)

# ==============================
# Input Section
# ==============================
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Living Area (Square Feet)", min_value=300, max_value=6000, value=1500)
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)

with col2:
    age = st.number_input("House Age (Years)", min_value=0, max_value=200, value=10)
    
    # Extract location names from column list
    location_columns = [col for col in model_columns if col.startswith("Location_")]
    locations = [col.replace("Location_", "") for col in location_columns]
    
    selected_location = st.selectbox("Select Location", locations)

# ==============================
# Prediction Logic
# ==============================
if st.button("Predict Price"):

    # Create input dictionary
    input_data = {
        "GrLivArea": area,
        "BedroomAbvGr": bedrooms,
        "Age": age
    }

    # Add all location columns as 0
    for col in location_columns:
        input_data[col] = 0

    # Set selected location to 1
    location_column_name = "Location_" + selected_location
    input_data[location_column_name] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure correct column order
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]

    # Display Result
    st.markdown(f"""
        <div class="prediction-box">
            Estimated House Price: ₹ {prediction:,.2f}
        </div>
    """, unsafe_allow_html=True)