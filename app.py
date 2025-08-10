import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
import kagglehub
import joblib
import os

# Download the dataset from Kaggle
path = kagglehub.dataset_download("madhushivam/walmart-pkl")

# Assuming the pkl file is inside the downloaded folder
model_path = os.path.join(path, "walmart_sales_model.pkl")

# Load the model
model = joblib.load(model_path)

print("Model loaded successfully!")

# App configuration
st.set_page_config(page_title="📈 Walmart Sales Predictor", layout="wide")

# ---------- Custom title with color ----------
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>
        📊 Walmart Weekly Sales Forecasting
    </h1>
    <h4 style='text-align: center; color: grey;'>
        Enter store information below to predict sales
    </h4>
""", unsafe_allow_html=True)



# Sidebar: Info
with st.sidebar:
    st.header("📘 Model Info")
    st.markdown("""
    **Random Forest Regressor** trained on Walmart historical data.
    
    🔢 Features:
    - Store, Department, Markdowns
    - Temperature, Fuel Price, CPI
    - Store Type & Size, Year/Month/Week
    - Holiday Indicator
    
    📅 Developed: 2025   
    📊 Input → Predict Weekly Sales
    """)
    st.info("Customize inputs on the main page and click **Predict**.")

# 

# Input columns
col1, col2, col3 = st.columns(3)

with col1:
    store = st.number_input("🏬 Store ID", min_value=1, max_value=50, step=1)
    dept = st.number_input("🏷️ Department", min_value=1, max_value=100, step=1)
    store_type = st.selectbox("🏢 Store Type", ["A", "B", "C"])
    store_size = st.number_input("📏 Store Size (sq ft)", value=150000)

with col2:
    temperature = st.number_input("🌡️ Temperature (°F)", value=60.0)
    fuel_price = st.number_input("⛽ Fuel Price", value=3.0)
    cpi = st.number_input("📈 CPI", value=220.0)
    unemployment = st.number_input("📉 Unemployment", value=7.5)

with col3:
    markdown1 = st.number_input("💲 MarkDown1", value=0.0)
    markdown2 = st.number_input("💲 MarkDown2", value=0.0)
    markdown3 = st.number_input("💲 MarkDown3", value=0.0)
    markdown4 = st.number_input("💲 MarkDown4", value=0.0)
    markdown5 = st.number_input("💲 MarkDown5", value=0.0)

# Additional inputs
st.markdown("---")
col4, col5, col6 = st.columns(3)

with col4:
    is_holiday = st.selectbox("🎉 Is Holiday", ["No", "Yes"])
with col5:
    year = st.number_input("📆 Year", min_value=2010, max_value=2025, value=2012)
with col6:
    month = st.number_input("📅 Month", min_value=1, max_value=12, value=6)
    week = st.number_input("📈 Week Number", min_value=1, max_value=52, value=25)

# Encode categorical data
type_encoding = {"A": 0, "B": 1, "C": 2}
is_holiday_val = 1 if is_holiday == "Yes" else 0

# Arrange input
input_data = np.array([[store, dept, temperature, fuel_price,
                        markdown1, markdown2, markdown3, markdown4, markdown5,
                        cpi, unemployment, is_holiday_val, store_size,
                        type_encoding[store_type], year, month, week]])

# Prediction
if st.button("🚀 Predict Weekly Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ **Predicted Weekly Sales:**  \n💰 **${prediction:,.2f}**")

    # Forecast visualization placeholder
    st.markdown("---")
    st.markdown("<h4 style='color: darkgreen;'>📊 Forecast Preview</h4>", unsafe_allow_html=True)

    # Simulated forecast for next 5 weeks
    future_weeks = [f"Week {week+i}" for i in range(1, 6)]
    simulated_sales = [prediction + np.random.randint(-1000, 1000) for _ in range(5)]
    forecast_df = pd.DataFrame({
        "Week": future_weeks,
        "Predicted Sales": simulated_sales
    })

    st.line_chart(forecast_df.set_index("Week"))

    with st.expander("📘 What this chart means"):
        st.markdown("""
        - This forecast preview simulates how sales **may vary** in upcoming weeks.  
        - It uses slight fluctuations around the current prediction to help visualize future trends.  
        - Useful for inventory planning, staffing, and promotions 💡.
        """)
    # Download results section

st.markdown("---")

