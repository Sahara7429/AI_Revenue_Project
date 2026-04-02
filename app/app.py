import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Revenue Insight System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

.metric-box {
    background: linear-gradient(135deg, #1F2937, #111827);
    padding: 20px;
    border-radius: 15px;
    border-left: 6px solid #3B82F6;
    margin-bottom: 10px;
}

/* Main page title */
h1 {
    color: white;
    font-size: 30px !important;
}

/* Section headings */
h2 {
    color: white;
    font-size: 22px !important;
}

h3 {
    color: white;
    font-size: 18px !important;
}

/* Normal text only on main page */
p {
    color: white;
    font-size: 14px !important;
}

/* Metric box text */
.metric-box h2 {
    font-size: 20px !important;
}

.metric-box h3 {
    font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "revenue_model.pkl")
subscription_encoder = joblib.load(BASE_DIR / "models" / "subscription_encoder.pkl")
month_encoder = joblib.load(BASE_DIR / "models" / "month_encoder.pkl")

# ---------------- TITLE ----------------
st.title("📈 AI-Powered SaaS Revenue Insight System")
st.write("Predict SaaS revenue using subscription details, churn, users and marketing spend.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input Customer Details")

subscription = st.sidebar.selectbox(
    "Subscription Type",
    ["Basic", "Premium", "Enterprise"]
)

month = st.sidebar.selectbox(
    "Month",
    ["Jan", "Feb", "Mar"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Credit Card", "UPI", "Net Banking", "PayPal"]
)

renewal_status = st.sidebar.selectbox(
    "Renewal Status",
    ["Renewed", "Pending", "Expired"]
)

upgrade_count = st.sidebar.number_input("Number of Upgrades", min_value=0, value=0)

downgrade_count = st.sidebar.number_input("Number of Downgrades", min_value=0, value=0)

cancelled = st.sidebar.selectbox(
    "Cancellation Status",
    ["No", "Yes"]
)

refund_amount = st.sidebar.number_input(
    "Refund Amount (₹)",
    min_value=0,
    value=0
)

failed_transactions = st.sidebar.number_input(
    "Failed Transactions",
    min_value=0,
    value=0
)

churn = st.sidebar.slider("Churn Rate (%)", 0.0, 20.0, 5.0)
active_users = st.sidebar.number_input("Active Users", min_value=0, value=100)
marketing = st.sidebar.number_input("Marketing Spend (₹)", min_value=0, value=500)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Revenue"):

    # Encode inputs
    sub_encoded = subscription_encoder.transform([subscription])[0]
    month_encoded = month_encoder.transform([month])[0]

    # Create input dataframe
    input_data = pd.DataFrame([[
        sub_encoded,
        churn,
        active_users,
        marketing,
        month_encoded
    ]], columns=[
        "Subscription_Type",
        "Churn_Rate",
        "Active_Users",
        "Marketing_Spend",
        "Month"
    ])

    # Predict
    prediction = model.predict(input_data)[0]

    # ---------------- METRIC CARDS ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Predicted Revenue</h3>
            <h2>₹{prediction:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Active Users</h3>
            <h2>{active_users}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        risk = "Low" if churn < 5 else "Medium" if churn < 10 else "High"

        risk_color = (
            "#10B981" if risk == "Low"
            else "#F59E0B" if risk == "Medium"
            else "#EF4444"
        )

        st.markdown(f"""
        <div class='metric-box'>
            <h3>Churn Risk</h3>
            <h2 style='color:{risk_color};'>{risk}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.success(f"Estimated Monthly Revenue: ₹{prediction:,.2f}")

    # ---------------- PIE CHART ----------------
    st.subheader("📊 Revenue vs Marketing Spend")

    chart_df = pd.DataFrame({
        "Category": ["Marketing Spend", "Predicted Revenue"],
        "Value": [marketing, prediction]
    })

    fig1, ax1 = plt.subplots(figsize=(2, 2))
    colors = ["#8B5CF6", "#3B82F6"]

    ax1.pie(
        chart_df["Value"],
        labels=chart_df["Category"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"color": "white", "fontsize": 4}
    )

    ax1.set_facecolor("#0E1117")
    fig1.patch.set_facecolor("#0E1117")

    st.pyplot(fig1)

    # ---------------- BAR CHART ----------------
st.subheader("📉 Input Values Summary")

compare_df = pd.DataFrame({
    "Metric": ["Churn Rate", "Active Users", "Marketing Spend"],
    "Value": [churn, active_users, marketing]
})

# Smaller figure
fig2, ax2 = plt.subplots(figsize=(4, 2.5))  # smaller width & height

sns.barplot(
    data=compare_df,
    x="Metric",
    y="Value",
    palette=["#EF4444", "#10B981", "#3B82F6"],
    ax=ax2
)

# Customize titles and labels with smaller font
ax2.set_title("Input Values Summary", color="white", fontsize=10)
ax2.set_xlabel("")
ax2.set_ylabel("Value", color="white", fontsize=6)

# Tick labels smaller
ax2.tick_params(axis='x', colors="white", labelsize=4)
ax2.tick_params(axis='y', colors="white", labelsize=4)

# Spine colors
for spine in ax2.spines.values():
    spine.set_color("white")

# Background colors for dark theme
ax2.set_facecolor("#0E1117")
fig2.patch.set_facecolor("#0E1117")

st.pyplot(fig2)

    # ---------------- EXTRA ANALYTICS ----------------
st.subheader("📈 Revenue Efficiency")

efficiency = prediction / marketing if marketing > 0 else 0

st.markdown(f"""
    <div class='metric-box'>
        <h3>Revenue Generated Per ₹1 Marketing Spend</h3>
        <h2>₹{efficiency:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- BUSINESS INSIGHT ----------------
st.subheader("💡 Business Insight")

if prediction > 5000:
        st.info(
            "Premium and Enterprise customers are generating strong revenue. "
            "You can increase marketing spend to scale even faster."
        )
elif prediction > 2500:
        st.warning(
            "Revenue is moderate. Improving customer retention and increasing active users may help."
        )
else:
        st.error(
            "Revenue is low. Focus on reducing churn and increasing active users before scaling marketing."
        )