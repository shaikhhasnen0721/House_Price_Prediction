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

    # Check Required Columns
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


# Load All Resources
model, data_columns, df = load_resources()

# ----------------------------------------
# Prediction Function
# ----------------------------------------
def predict_price(location, sqft, bhk, area_type):

    area_type = area_type.lower().strip()

    x = np.zeros(len(data_columns))

    # Numeric Features
    if "total_sqft" in data_columns:
        x[data_columns.index("total_sqft")] = sqft

    if "bhk" in data_columns:
        x[data_columns.index("bhk")] = bhk

    # Location One-Hot Encoding
    loc = location.lower().strip()

    if loc in data_columns:
        x[data_columns.index(loc)] = 1

    # Area Type One-Hot Encoding
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

# User Inputs
sqft = st.number_input(
    "Total Square Feet",
    min_value=300,
    max_value=10000,
    step=10
)

bhk = st.selectbox(
    "BHK",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)

locations = sorted(df["location"].unique())

location = st.selectbox(
    "Select Location",
    locations
)

area_type = st.radio(
    "Area Type",
    ["Urban", "Rural"]
)

# Predict Button
predict_btn = st.button("Predict Price 🔍")

# ----------------------------------------
# Prediction Result
# ----------------------------------------
if predict_btn:

    result = predict_price(location, sqft, bhk, area_type)

    st.markdown("## 🏁 Prediction Result")

    st.success(f"### Estimated Price: ₹ {result} Lakhs")

    st.write("---")

    st.info(f"""
    📍 Location: {location}

    🏠 BHK: {bhk}

    📐 Total Sqft: {sqft}

    🌍 Area Type: {area_type}

    💰 Estimated Price: ₹ {result} Lakhs
    """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")