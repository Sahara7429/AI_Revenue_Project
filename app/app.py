import streamlit as st

st.set_page_config(
    page_title="AI Revenue Insight System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
#Imports
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

#  Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

#  Custom CSS  (dark-premium theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0B0F1A; color: #E2E8F0; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1629 0%, #0B0F1A 100%);
    border-right: 1px solid #1E2D4A;
}

/* Sidebar text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #94A3B8 !important;
    font-size: 13px !important;
    font-weight: 500;
}

h1 { color: #F1F5F9 !important; font-size: 26px !important; font-weight: 700 !important; }
h2 { color: #CBD5E1 !important; font-size: 20px !important; font-weight: 600 !important; }
h3 { color: #94A3B8 !important; font-size: 15px !important; font-weight: 500 !important; }
p  { color: #94A3B8 !important; font-size: 14px !important; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #141C2E 0%, #0F1629 100%);
    border: 1px solid #1E3A5F;
    border-left: 4px solid #3B82F6;
    padding: 22px 20px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.kpi-card.green  { border-left-color: #10B981; }
.kpi-card.amber  { border-left-color: #F59E0B; }
.kpi-card.red    { border-left-color: #EF4444; }
.kpi-card.purple { border-left-color: #8B5CF6; }

.kpi-label { font-size: 12px; color: #64748B; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #F1F5F9; }
.kpi-sub   { font-size: 12px; color: #64748B; margin-top: 4px; }

/* Insight banner */
.insight-banner {
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 8px;
    font-size: 14px;
    line-height: 1.6;
}
.insight-banner.success { background:#052e16; border-left:4px solid #10B981; color:#6EE7B7; }
.insight-banner.warning { background:#1c1400; border-left:4px solid #F59E0B; color:#FDE68A; }
.insight-banner.danger  { background:#1a0606; border-left:4px solid #EF4444; color:#FCA5A5; }

/* Divider */
hr { border-color: #1E2D4A !important; }

/* Remove default streamlit padding from pyplot */
.stPlotlyChart, .element-container { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

#  Helper — dark-styled matplotlib figure
BG = "#0B0F1A"
CARD = "#141C2E"
GRID = "#1E2D4A"
BLUE = "#3B82F6"
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
PURPLE = "#8B5CF6"
TEXT = "#94A3B8"

def dark_fig(w=7, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("#E2E8F0")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    return fig, ax

#  Load datasets
@st.cache_data
def load_datasets():
    dfs = {}
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("*.csv"):
            try:
                dfs[f.stem] = pd.read_csv(f)
            except Exception:
                pass
    return dfs

data_dict = load_datasets()

#  Load models
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_DIR / "revenue_model.pkl")
    sub_enc = joblib.load(MODEL_DIR / "subscription_encoder.pkl")
    mon_enc = joblib.load(MODEL_DIR / "month_encoder.pkl")
    return model, sub_enc, mon_enc

try:
    model, subscription_encoder, month_encoder = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    model_error = str(e)

#  Sidebar — inputs
from sklearn.preprocessing import LabelEncoder

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

month_encoder = LabelEncoder()
month_encoder.fit(MONTHS)  # Fit on all possible months

with st.sidebar:
    st.markdown("## ⚙️ Customer Parameters")
    st.markdown("---")

    st.markdown("**📦 Subscription**")
    subscription = st.selectbox("Subscription Type", ["Basic", "Premium", "Enterprise"], label_visibility="collapsed")

    month = st.selectbox("Month", MONTHS)

    st.markdown("**💳 Payment & Status**")
    payment_method = st.selectbox("Payment Method", ["Credit Card", "UPI", "Net Banking", "PayPal"])
    renewal_status = st.selectbox("Renewal Status", ["Renewed", "Pending", "Expired"])
    cancelled = st.selectbox("Cancellation Status", ["No", "Yes"])

    st.markdown("**🔄 Account Activity**")
    upgrade_count   = st.number_input("Upgrades",   min_value=0, value=0)
    downgrade_count = st.number_input("Downgrades", min_value=0, value=0)

    st.markdown("**💰 Financials**")
    refund_amount       = st.number_input("Refund Amount (₹)",    min_value=0, value=0, step=100)
    failed_transactions = st.number_input("Failed Transactions",  min_value=0, value=0)
    marketing           = st.number_input("Marketing Spend (₹)",  min_value=0, value=500, step=100)

    st.markdown("**📊 Performance**")
    churn        = st.slider("Churn Rate (%)",  0.0, 20.0, 5.0, 0.1)
    active_users = st.number_input("Active Users", min_value=0, value=100, step=10)

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Revenue", use_container_width=True)

#  Header
st.title("📈 AI-Powered SaaS Revenue Insight System")
st.markdown("<p>Enter customer parameters in the sidebar and click <strong style='color:#3B82F6'>Predict Revenue</strong> to generate insights.</p>", unsafe_allow_html=True)
st.markdown("---")

#  Available datasets
if data_dict:
    with st.expander(f"📂 Loaded Datasets ({len(data_dict)})", expanded=False):
        for name, df in data_dict.items():
            st.markdown(f"**{name}** — {df.shape[0]:,} rows × {df.shape[1]} cols")
            st.dataframe(df.head(5), use_container_width=True)

#  Prediction block
if predict_btn:

    if not models_ok:
        st.error(f"❌ Could not load models: {model_error}")
        st.stop()

    # Encode
    sub_encoded = subscription_encoder.transform([subscription])[0]
    mon_encoded = month_encoder.transform([month])[0]

    input_df = pd.DataFrame([[
        sub_encoded,
        churn,
        active_users,
        marketing,
        mon_encoded,
    ]], columns=["Subscription_Type", "Churn_Rate", "Active_Users", "Marketing_Spend", "Month"])

    prediction = float(model.predict(input_df)[0])
    efficiency = prediction / marketing if marketing > 0 else 0.0


    #  Churn risk label 

    if churn < 5:
        risk, risk_col, card_cls = "Low", GREEN, "green"
    elif churn < 10:
        risk, risk_col, card_cls = "Medium", AMBER, "amber"
    else:
        risk, risk_col, card_cls = "High", RED, "red"

    # KPI cards 
    st.subheader("📌 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Predicted Revenue</div>
            <div class='kpi-value'>₹{prediction:,.0f}</div>
            <div class='kpi-sub'>{month} · {subscription}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='kpi-card green'>
            <div class='kpi-label'>Active Users</div>
            <div class='kpi-value'>{active_users:,}</div>
            <div class='kpi-sub'>↕ {upgrade_count} up / {downgrade_count} down</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='kpi-card {card_cls}'>
            <div class='kpi-label'>Churn Risk</div>
            <div class='kpi-value' style='color:{risk_col}'>{risk}</div>
            <div class='kpi-sub'>Churn rate: {churn:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='kpi-card purple'>
            <div class='kpi-label'>Marketing ROI</div>
            <div class='kpi-value'>₹{efficiency:.2f}</div>
            <div class='kpi-sub'>per ₹1 spent</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Charts 
    st.subheader("📊 Visual Analytics")
    col_l, col_r = st.columns(2)

    # Pie: Revenue vs Marketing 
    with col_l:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        fig1.patch.set_facecolor(CARD)
        ax1.set_facecolor(CARD)
        vals   = [marketing, max(prediction - marketing, 0), min(prediction, marketing) * 0 ]
        labels = ["Marketing Spend", "Net Revenue Gain"]
        sizes  = [marketing, max(prediction - marketing, 1)]
        colors = [PURPLE, BLUE]
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            startangle=90, colors=colors,
            textprops={"color": TEXT, "fontsize": 10},
            wedgeprops={"linewidth": 2, "edgecolor": CARD},
        )
        for at in autotexts:
            at.set_color("#F1F5F9")
            at.set_fontsize(10)
        ax1.set_title("Revenue vs Marketing Spend", color="#E2E8F0", fontsize=12, pad=12)
        fig1.tight_layout()
        st.pyplot(fig1)

    # Bar: Input summary
    with col_r:
        fig2, ax2 = dark_fig(5, 3.5)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        metrics = ["Churn Rate", "Active Users", "Mktg Spend (÷100)"]
        values  = [churn, active_users, marketing / 100]
        bars = ax2.bar(metrics, values, color=[RED, GREEN, BLUE], width=0.5,
                       edgecolor=CARD, linewidth=1.5)
        for bar, val in zip(bars, [churn, active_users, marketing]):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(values) * 0.02,
                     f"{val:,.1f}" if isinstance(val, float) else f"{val:,}",
                     ha="center", va="bottom", color="#E2E8F0", fontsize=9)
        ax2.set_title("Input Parameters Overview", color="#E2E8F0", fontsize=12)
        ax2.set_ylabel("")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax2.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
        ax2.set_axisbelow(True)
        fig2.tight_layout()
        st.pyplot(fig2)

    # Revenue trend simulation 
    st.markdown("#### 📈 Simulated Revenue Trend (±20% churn sensitivity)")
    churn_range = np.linspace(max(0, churn - 5), min(20, churn + 5), 30)
    rev_trend   = []
    for c in churn_range:
        row = pd.DataFrame([[sub_encoded, c, active_users, marketing, mon_encoded]],
                           columns=input_df.columns)
        rev_trend.append(float(model.predict(row)[0]))

    fig3, ax3 = dark_fig(10, 3)
    ax3.plot(churn_range, rev_trend, color=BLUE, linewidth=2.5)
    ax3.axvline(churn, color=AMBER, linewidth=1.5, linestyle="--", label=f"Current churn {churn:.1f}%")
    ax3.fill_between(churn_range, rev_trend, alpha=0.15, color=BLUE)
    ax3.set_xlabel("Churn Rate (%)", color=TEXT, fontsize=10)
    ax3.set_title("Revenue Sensitivity to Churn Rate", color="#E2E8F0", fontsize=12)
    ax3.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    ax3.grid(color=GRID, linewidth=0.4, alpha=0.6)
    ax3.set_axisbelow(True)
    fig3.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")

    # Financial summary table
    st.subheader("🧾 Financial Summary")
    summary = pd.DataFrame({
        "Metric": [
            "Predicted Revenue",
            "Marketing Spend",
            "Net Revenue (after marketing)",
            "Refund Deductions",
            "Marketing ROI (₹ per ₹1 spent)",
            "Churn Risk Level",
            "Renewal Status",
        ],
        "Value": [
            f"₹{prediction:,.2f}",
            f"₹{marketing:,.2f}",
            f"₹{max(prediction - marketing, 0):,.2f}",
            f"₹{refund_amount:,.2f}",
            f"₹{efficiency:.3f}",
            f"{risk} ({churn:.1f}%)",
            renewal_status,
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Business insight 
    st.subheader("💡 Business Insight")

    if prediction > 5000:
        st.markdown(f"""
        <div class='insight-banner success'>
        ✅ <strong>Strong revenue detected.</strong> {subscription} customers are driving solid returns.
        Your marketing ROI of <strong>₹{efficiency:.2f}</strong> per ₹1 spent is healthy.
        Consider increasing marketing spend 10–20% to accelerate growth while churn remains {risk.lower()}.
        </div>""", unsafe_allow_html=True)
    elif prediction > 2500:
        st.markdown(f"""
        <div class='insight-banner warning'>
        ⚠️ <strong>Moderate revenue.</strong> There's room to grow.
        With a churn rate of {churn:.1f}% ({risk} risk), improving retention by just 2–3 percentage points
        could meaningfully boost revenue. Focus on customer success before scaling spend.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='insight-banner danger'>
        🔴 <strong>Low revenue alert.</strong> Current metrics suggest underlying issues.
        Churn at {churn:.1f}% is eating into your base.
        Prioritize retention, re-engagement campaigns, and product improvements
        before increasing marketing spend of ₹{marketing:,}.
        </div>""", unsafe_allow_html=True)

    # Actionable recommendations 
    st.markdown("#### 🎯 Recommendations")
    recs = []

    if churn >= 10:
        recs.append("🔴 **Critical:** Implement churn-prevention workflows (loyalty rewards, proactive support).")
    elif churn >= 5:
        recs.append("🟡 **Watch:** Churn is creeping up — review offboarding surveys and NPS scores.")

    if renewal_status in ["Pending", "Expired"]:
        recs.append(f"📧 **Renewal {renewal_status}:** Send automated renewal reminder emails with a discount incentive.")

    if downgrade_count > upgrade_count:
        recs.append("📉 **More downgrades than upgrades:** Investigate feature gaps causing plan downgrades.")

    if failed_transactions > 0:
        recs.append(f"💳 **{failed_transactions} failed transactions:** Set up automated payment retry logic to recover revenue.")

    if refund_amount > 0:
        recs.append(f"💸 **Refunds of ₹{refund_amount:,}:** Analyse refund reasons — a pattern may indicate product-market fit issues.")

    if efficiency < 1:
        recs.append("📊 **Marketing ROI < 1:** You're spending more on marketing than you're earning. Pause and optimise campaigns.")

    if not recs:
        recs.append("✅ **All metrics look healthy.** Keep monitoring and consider A/B testing new growth channels.")

    for r in recs:
        st.markdown(f"- {r}")

else:
    # Placeholder when no prediction made 
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Revenue** to get started.")

    if data_dict:
        # Show a quick overview chart from loaded data if numeric columns exist
        first_df = list(data_dict.values())[0]
        num_cols = first_df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            st.markdown("#### 📂 Quick Data Overview")
            fig0, ax0 = dark_fig(10, 3.5)
            ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            first_df[num_cols[:2]].dropna().plot(ax=ax0, color=[BLUE, GREEN])
            ax0.set_title(f"Dataset Preview — {list(data_dict.keys())[0]}", color="#E2E8F0", fontsize=12)
            ax0.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
            ax0.grid(color=GRID, linewidth=0.4, alpha=0.6)
            fig0.tight_layout()
            st.pyplot(fig0)
