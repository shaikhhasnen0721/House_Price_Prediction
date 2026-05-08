import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(
    page_title="🏡 House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ----------------------------------------
# Load Resources
# ----------------------------------------
@st.cache_data
def load_resources():

    # Load Model
    with open("banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)

    # Load Columns
    with open("columns.json", "r") as f:
        data_columns = json.load(f)["data_columns"]

    # Load Dataset
    df = pd.read_csv("hpp_excet.csv")

    # Clean Column Names
    df.columns = df.columns.str.strip().str.lower()

    # Required Columns Check
    required_columns = ["location", "category"]

    for col in required_columns:
        if col not in df.columns:
            st.error(f"❌ Required column '{col}' not found in CSV file.")
            st.write("Available Columns:", df.columns.tolist())
            st.stop()

    # Clean Data
    df["location"] = df["location"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.lower().str.strip()

    return model, data_columns, df


# Load Everything
model, data_columns, df = load_resources()

# ----------------------------------------
# Prediction Function
# ----------------------------------------
def predict_price(location, sqft, bhk, area_type):

    area_type = area_type.lower().strip()

    # Find Selected Location
    sample = df[df["location"].str.lower() == location.lower()]

    # Location Validation
    if len(sample) == 0:
        return "❌ Location not found in dataset!"

    # Get Actual Category
    actual_category = sample["category"].iloc[0]

    # Check Urban / Rural
    if actual_category != area_type:
        return f"❌ This location is not {area_type.title()}. Please select {actual_category.title()} location."

    # Create Input Vector
    x = np.zeros(len(data_columns))

    # Numeric Features
    if "total_sqft" in data_columns:
        x[data_columns.index("total_sqft")] = sqft

    if "bhk" in data_columns:
        x[data_columns.index("bhk")] = bhk

    # Location One Hot Encoding
    loc = location.lower().strip()

    if loc in data_columns:
        x[data_columns.index(loc)] = 1

    # Area Type Encoding
    area_col = f"area_{area_type}"

    if area_col in data_columns:
        x[data_columns.index(area_col)] = 1

    # Prediction
    prediction = model.predict([x])[0]

    return round(prediction, 2)


# ----------------------------------------
# UI
# ----------------------------------------
st.title("🏡 Bangalore House Price Prediction")

st.markdown("### Enter Property Details")

# ----------------------------------------
# User Inputs
# ----------------------------------------

sqft = st.number_input(
    "📐 Total Square Feet",
    min_value=300,
    max_value=10000,
    step=10
)

bhk = st.selectbox(
    "🏠 Select BHK",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)

locations = sorted(df["location"].unique())

location = st.selectbox(
    "📍 Select Location",
    locations
)

area_type = st.radio(
    "🌍 Select Area Type",
    ["Urban", "Rural"]
)

# Predict Button
predict_btn = st.button("🔍 Predict Price")

# ----------------------------------------
# Prediction Result
# ----------------------------------------
if predict_btn:

    result = predict_price(location, sqft, bhk, area_type)

    st.markdown("## 🏁 Prediction Result")

    # If Error Message
    if isinstance(result, str):
        st.error(result)

    # If Prediction Success
    else:
        st.success(f"### 💰 Estimated Price: ₹ {result} Lakhs")

        st.write("---")

        st.info(f"""
        📍 Location: {location}

        🏠 BHK: {bhk}

        📐 Total Sqft: {sqft}

        🌍 Area Type: {area_type}

        💰 Estimated Price: ₹ {result} Lakhs
        """)

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")