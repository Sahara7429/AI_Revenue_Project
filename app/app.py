import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="AI-Powered Revenue Insight Dashboard",
    layout="wide"
)

# --- Title ---
st.title("AI-Powered Revenue Insight and Prediction System for SaaS Platforms")
st.write("Explore SaaS revenue trends and predict future revenue.")

# --- Load Data ---
@st.cache_data
def load_data():
    # Automatically find dataset relative to app.py
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "saas_revenue.csv"
    df = pd.read_csv(data_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

# --- Load Model ---
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "revenue_model.pkl"

    if model_path.exists():
        return joblib.load(model_path)
    return None

# --- Main Data ---
df = load_data()
model = load_model()

# --- Raw Data ---
st.subheader("Raw Data")
st.dataframe(df.head())

# --- Summary ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Summary")
    st.dataframe(df.describe())

with col2:
    st.subheader("Columns in Dataset")
    st.write(df.columns.tolist())

# --- Filter by Product ---
filtered_df = df.copy()

if "Product" in df.columns:
    product = st.selectbox("Select Product", ["All"] + list(df["Product"].unique()))

    if product != "All":
        filtered_df = df[df["Product"] == product]

# --- KPI Section ---
st.subheader("Key Metrics")

kpi1, kpi2 = st.columns(2)

if "Revenue" in filtered_df.columns:
    total_revenue = filtered_df["Revenue"].sum()
    avg_revenue = filtered_df["Revenue"].mean()

    with kpi1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")

    with kpi2:
        st.metric("Average Revenue", f"${avg_revenue:,.2f}")

# --- Revenue Trend Chart ---
st.subheader("Revenue Trend")

if "Month" in filtered_df.columns and "Revenue" in filtered_df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(
        data=filtered_df,
        x="Month",
        y="Revenue",
        marker="o",
        ax=ax
    )

    ax.set_title("Revenue Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45)

    st.pyplot(fig)
else:
    st.warning("Month or Revenue column not found in dataset.")

# --- Prediction Section ---
st.subheader("Future Revenue Prediction")

if model is not None:
    st.write("Enter values to predict revenue:")

    input_data = {}

    # Use all columns except Revenue as inputs
    feature_columns = [col for col in df.columns if col != "Revenue"]

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            input_data[col] = st.number_input(
                f"{col}",
                value=float(df[col].mean())
            )

    if st.button("Predict Revenue"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Revenue: ${prediction:,.2f}")

else:
    st.info("Train your model first. After `revenue_model.pkl` is created, prediction will work automatically.")