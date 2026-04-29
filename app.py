import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Preeclampsia Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------------------------
# Custom CSS
# ------------------------------------------------
st.markdown("""
<style>
.big-font {font-size:20px; font-weight:600;}
.risk-low {color: green; font-size:22px; font-weight:700;}
.risk-mid {color: orange; font-size:22px; font-weight:700;}
.risk-high {color: red; font-size:22px; font-weight:700;}
.card {
    padding:20px;
    border-radius:12px;
    background-color:#f8f9fa;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.title("🩺 Clinical Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Model Performance", "🧬 Feature Importance", "📈 Temporal Risk Prediction"]
)
st.sidebar.info("AI Decision Support System\n(Not Medical Advice)")

# ------------------------------------------------
# Load Data
# ------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("preeclampsia_data.csv")

data = load_data()
X = data.drop("RiskLevel", axis=1)
y = data["RiskLevel"]


# ------------------------------------------------
# Train Models
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

for model in models.values():
    model.fit(X_train, y_train)

# ------------------------------------------------
# Overview
# ------------------------------------------------
if page == "🏠 Overview":
    st.title("Preeclampsia Risk Prediction System")
    st.write("### AI-Powered Temporal Clinical Decision Support")

    c1, c2, c3 = st.columns(3)
    c1.metric("ML Models", "3")
    c2.metric("Dataset Records", data.shape[0])
    c3.metric("Clinical Features", X.shape[1])

    st.markdown("""
    <div class="card">
    <p class="big-font">
    This system predicts preeclampsia risk using ensemble machine learning
    and monitors patient condition over time using <b>Temporal AI</b>.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# Model Performance
# ------------------------------------------------
elif page == "📊 Model Performance":
    st.title("Model Accuracy")

    cols = st.columns(3)
    for col, (name, model) in zip(cols, models.items()):
        acc = accuracy_score(y_test, model.predict(X_test))
        col.metric(name, f"{acc:.2f}")

# ------------------------------------------------
# Feature Importance
# ------------------------------------------------
elif page == "🧬 Feature Importance":
    st.title("Key Clinical Risk Factors")

    rf = models["Random Forest"]
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    st.dataframe(imp_df, use_container_width=True)

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

# ------------------------------------------------
# Temporal Risk Prediction
# ------------------------------------------------
elif page == "📈 Temporal Risk Prediction":
    st.title("Temporal AI – Trend-Based Risk Assessment")

    prev_file = st.file_uploader("Upload Previous Visit Report", type="csv")
    curr_file = st.file_uploader("Upload Current Visit Report", type="csv")

    if prev_file and curr_file:
        prev = pd.read_csv(prev_file)
        curr = pd.read_csv(curr_file)

        st.subheader("Report Comparison")
        st.dataframe(pd.concat(
            {"Previous": prev, "Current": curr},
            axis=0
        ))

        prev_scaled = scaler.transform(prev)
        curr_scaled = scaler.transform(curr)

        prev_risk = np.mean([m.predict_proba(prev_scaled)[0][1] for m in models.values()])
        curr_risk = np.mean([m.predict_proba(curr_scaled)[0][1] for m in models.values()])

        trend = curr_risk - prev_risk

        if curr_risk < 0.3:
            css, level = "risk-low", "Low Risk"
        elif curr_risk < 0.6:
            css, level = "risk-mid", "Moderate Risk"
        else:
            css, level = "risk-high", "High Risk"

        st.subheader("AI Risk Analysis")

        c1, c2, c3 = st.columns(3)
        c1.metric("Previous Risk", f"{prev_risk:.2f}")
        c2.metric("Current Risk", f"{curr_risk:.2f}")
        c3.metric("Risk Change", f"{trend:+.2f}")

        st.markdown(f"<p class='{css}'>{level}</p>", unsafe_allow_html=True)

        if trend > 0.1:
            st.error("⚠️ AI Alert: Worsening clinical trend detected")
        else:
            st.success("✅ Stable clinical trend")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.caption("© 2026 Temporal AI Preeclampsia Prediction System | Academic Project")
