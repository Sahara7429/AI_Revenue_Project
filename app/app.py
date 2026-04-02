import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "revenue_model.pkl")
subscription_encoder = joblib.load(BASE_DIR / "models" / "subscription_encoder.pkl")
month_encoder = joblib.load(BASE_DIR / "models" / "month_encoder.pkl")

st.title("AI-Powered SaaS Revenue Prediction")
st.write("Predict monthly revenue based on customer details")

subscription = st.selectbox(
    "Subscription Type",
    ['Basic', 'Premium', 'Enterprise']
)

churn = st.number_input("Churn Rate", min_value=0.0)
active_users = st.number_input("Active Users", min_value=0)
marketing = st.number_input("Marketing Spend", min_value=0)
month = st.selectbox(
    "Month",
    ['Jan', 'Feb', 'Mar']
)

if st.button("Predict Revenue"):
    sub_encoded = subscription_encoder.transform([subscription])[0]
    month_encoded = month_encoder.transform([month])[0]

    input_data = pd.DataFrame([[
        sub_encoded,
        churn,
        active_users,
        marketing,
        month_encoded
    ]], columns=[
        'Subscription_Type',
        'Churn_Rate',
        'Active_Users',
        'Marketing_Spend',
        'Month'
    ])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Monthly Revenue: ₹{prediction:,.2f}")