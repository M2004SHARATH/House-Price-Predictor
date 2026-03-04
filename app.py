import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)



st.markdown("""
<style>

body {
    background-color: white;
}

.main-title{
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:#0b3d91;
}

.sub-title{
    text-align:center;
    font-size:18px;
    color:#333;
    margin-bottom:30px;
}

.stButton>button{
    background:#0b3d91;
    color:white;
    height:48px;
    width:100%;
    border-radius:6px;
    font-size:18px;
    border:none;
}

.stButton>button:hover{
    background:#174db8;
}

.result-box{
    background:white;
    padding:35px;
    border-radius:10px;
    border:1px solid #e6e6e6;
    text-align:center;
}

.price-value{
    font-size:42px;
    font-weight:bold;
    color:#0b3d91;
}

</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown('<div class="main-title">House Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Based Real Estate Valuation</div>', unsafe_allow_html=True)


# CHECK MODEL FILES

if not os.path.exists("model_columns.pkl"):
    st.error("Model files missing. Train the model first.")
    st.stop()

model_columns = joblib.load("pickle files/Random Forest/model_columns.pkl")


# MODEL SELECTION

st.markdown("<h3 style='color:#0b3d91'>Select Prediction Model</h3>", unsafe_allow_html=True)

model_choice = st.selectbox(
    "",
    ["Random Forest (Recommended)", "Linear Regression"]
)

if model_choice == "Random Forest (Recommended)":
    model = joblib.load("pickle files/Random Forest/random_forest_model.pkl")
else:
    model = joblib.load("pickle files/Linear Regression/linear_regression_model.pkl")

# =====================================
# PROPERTY DETAILS
# =====================================
st.markdown("<h2 style='color:#0b3d91'>Property Details</h2>", unsafe_allow_html=True)

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Living Area (sq ft)", 300, 6000, 1500)

with col2:
    bedrooms = st.number_input("Bedrooms", 0, 10, 3)

with col3:
    age = st.number_input("House Age (years)", 0, 200, 10)

# Row 2
col4, col5, col6 = st.columns(3)

with col4:
    quality = st.slider("Overall Quality", 1, 10, 5)

with col5:
    bathrooms = st.number_input("Total Bathrooms", 0.0, 10.0, 2.0)

with col6:
    garage = st.number_input("Garage Capacity", 0, 5, 2)

# Row 3
col7, col8 = st.columns(2)

with col7:
    basement = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

with col8:
    location_columns = [c for c in model_columns if c.startswith("Location_")]
    locations = [c.replace("Location_", "") for c in location_columns]

    location = st.selectbox("Location", locations)


# PREDICT BUTTON

predict_button = st.button("Predict House Price")


# RESULT (BOTTOM)

if predict_button:

    input_data = {
        "GrLivArea": area,
        "BedroomAbvGr": bedrooms,
        "Age": age,
        "OverallQual": quality,
        "TotalBathrooms": bathrooms,
        "GarageCars": garage,
        "TotalBsmtSF": basement
    }

    for col in location_columns:
        input_data[col] = 0

    input_data["Location_" + location] = 1

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.markdown("<h2 style='color:#0b3d91'>Estimated Property Value</h2>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-box">
        <div style="font-size:18px;color:#555">Predicted Market Price</div>
        <div class="price-value">$ {prediction:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)