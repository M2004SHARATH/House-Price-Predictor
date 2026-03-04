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

/* App Background */
.stApp {
    background: linear-gradient(to right, #e3f2fd, #ffffff);
}

/* Main Title */
.title {
    text-align: center;
    font-size: 48px;   /* Increased */
    font-weight: 700;
    color: #0d47a1;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 22px;   /* Increased */
    color: #444;
    margin-bottom: 25px;
}

/* Input Labels */
label {
    font-size: 38px !important;
    font-weight: 800;
}

/* Input Boxes */
input, select {
    font-size: 38px !important;
}

/* Button Style */
button[kind="primary"] {
    background-color: #185a9d;
    color: white;
    border-radius: 20px;
    height: 6em;
    width: 100%;
    font-size:28px;
}

/* Prediction Box */
.prediction-box {
    font-size: 40px;   /* Bigger result */
    padding: 25px;
    border-radius: 25px;
    text-align: center;
    font-weight: bold;
    color: white;
    width:60%;
    margin:auto;
    background: linear-gradient(135deg,#43cea2,#185a9d);
}
/* Sidebar input labels */
section[data-testid="stSidebar"] label {
    font-size: 22px !important;
    font-weight: 600;
}

/* Sidebar input boxes */
section[data-testid="stSidebar"] input {
    font-size: 22px !important;
    height: 50px !important;
}

/* Selectbox dropdown */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    font-size: 22px !important;
}

/* Number input +/- buttons */
section[data-testid="stSidebar"] button {
    font-size: 20px !important;
}
/* Sidebar Background */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #f8fbff,
        #e3f2fd,
        #dbeafe
    );
    padding-top: 25px;
}

/* Sidebar Title */
section[data-testid="stSidebar"] h2 {
    color: #0d47a1;
    font-size: 28px;
    font-weight: bold;
}

/* Sidebar Labels */
section[data-testid="stSidebar"] label {
    font-size: 20px;
    font-weight: 600;
    color: #1a237e;
}
section[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
}
/* Predict Button */
button[kind="primary"] {
    font-size: 24px !important;
    height: 60px !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] {
    width: 340px !important;
}
.stApp {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
    background-position: center;
    background-attachment: fixed;
}
.main {
    background-color: rgba(255,255,255,0.9);
    padding: 20px;
    border-radius: 15px;
}
/* Animated Predict Button */
button[kind="primary"] {
    background: linear-gradient(45deg, #4facfe, #00f2fe);
    color: white;
    font-size: 22px !important;
    height: 60px !important;
    border-radius: 12px !important;
    border: none;
    transition: all 0.3s ease-in-out;
}

/* Hover Animation */
button[kind="primary"]:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 20px rgba(0, 150, 255, 0.6);
}

/* Click Animation */
button[kind="primary"]:active {
    transform: scale(0.95);
}

</style>
""", unsafe_allow_html=True)
# ==============================
# Header
# ==============================
st.markdown('<div class="title">🏡 Smart House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"> Real Estate Valuation System</div>', unsafe_allow_html=True)

# ==============================
# Sidebar Dashboard
# ==============================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/619/619034.png",
    width=100
)

st.sidebar.title("🏠 Property Inputs")
# ==============================
# Input Section
# ==============================
# Sidebar Inputs

area = st.sidebar.number_input(
    "📐 Living Area (Sqft)", 300, 6000, 1500
)

bedrooms = st.sidebar.number_input(
    "🛏 Bedrooms", 0, 10, 3
)

age = st.sidebar.number_input(
    "🏗 House Age", 0, 200, 10
)

# Location List
location_columns = [
    col for col in model_columns if col.startswith("Location_")
]

locations = [
    col.replace("Location_", "") for col in location_columns
]

selected_location = st.sidebar.selectbox(
    "📍 Location",
    locations
)

# ==============================
# Prediction Logic
# ==============================
if st.button("🔮 Predict Price",type = "primary"):
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
    price = f"${prediction :,.2f}"

    # Display Result
    st.markdown(f"""
        <div class="prediction-box">
            Estimated House Price:{price}
        </div>
    """, unsafe_allow_html=True)
