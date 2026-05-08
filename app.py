import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

@st.cache_data
def load_resources():

    # Load Model
    with open("banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)

    # Load Columns
    with open("columns.json", "r") as f:
        data_columns = json.load(f)["data_columns"]

    # Load CSV
    df = pd.read_csv("hpp_excet.csv")

    # Clean Columns
    df.columns = df.columns.str.strip().str.lower()

    # Check Required Columns
    if "location" not in df.columns:
        st.error("❌ 'location' column not found")
        st.stop()

    if "category" not in df.columns:
        st.error("❌ 'category' column not found")
        st.stop()

    # Clean Data
    df["location"] = df["location"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.lower().str.strip()

    return model, data_columns, df


# Load Resources
model, data_columns, df = load_resources()


# Prediction Function
def predict_price(location, sqft, bhk, area_type):

    area_type = area_type.lower()

    # Find location row
    sample = df[df["location"].str.lower() == location.lower()]

    if len(sample) == 0:
        return "❌ Location not found"

    actual_category = sample["category"].iloc[0]

    # Validation
    if actual_category != area_type:
        return f"❌ This location is not {area_type.title()}. Please select {actual_category.title()} location."

    x = np.zeros(len(data_columns))

    # Numeric Features
    if "total_sqft" in data_columns:
        x[data_columns.index("total_sqft")] = sqft

    if "bhk" in data_columns:
        x[data_columns.index("bhk")] = bhk

    # Location Encoding
    loc = location.lower()

    if loc in data_columns:
        x[data_columns.index(loc)] = 1

    # Area Type Encoding
    area_col = f"area_{area_type}"

    if area_col in data_columns:
        x[data_columns.index(area_col)] = 1

    prediction = model.predict([x])[0]

    return round(prediction, 2)


# UI
st.markdown("""
<h1 style='text-align: center; color: #00ADB5;'>
🏡 Bangalore House Price Prediction
</h1>

<p style='text-align: center;'>
Predict house prices using Machine Learning
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input(
        "📐 Total Square Feet",
        min_value=300,
        max_value=10000,
        step=10
    )

with col2:
    bhk = st.selectbox(
        "🏠 BHK",
        [1,2,3,4,5,6,7,8,9,10]
    )

bhk = st.selectbox(
    "BHK",
    [1,2,3,4,5,6,7,8,9,10]
)

location = st.selectbox(
    "Select Location",
    sorted(df["location"].unique())
)

area_type = st.radio(
    "Area Type",
    ["Urban", "Rural"]
)

if st.button("Predict Price"):

    result = predict_price(location, sqft, bhk, area_type)

    st.markdown("## 🏁 Prediction Result")

    # If validation error
    if type(result) == str:

        st.error(result)

    # If prediction success
    else:

        st.success(f"💰 Estimated Price: ₹ {result} Lakhs")

        st.info(f"""
        📍 Location: {location}

        🏠 BHK: {bhk}

        📐 Total Sqft: {sqft}

        🌍 Area Type: {area_type}
        """)