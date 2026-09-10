from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).with_name("European_Bank.csv")
TARGET = "Exited"
DROP_COLUMNS = ["CustomerId", "Surname", "Year", TARGET]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERICAL_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "BalanceSalaryRatio",
    "ProductDensity",
    "EngagementProduct",
    "AgeTenureInteraction",
]


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["BalanceSalaryRatio"] = data["Balance"] / data["EstimatedSalary"].clip(lower=1)
    data["ProductDensity"] = data["NumOfProducts"] / data["Tenure"].replace(0, 1)
    data["EngagementProduct"] = data["IsActiveMember"] * data["NumOfProducts"]
    data["AgeTenureInteraction"] = data["Age"] * data["Tenure"]
    return data


def build_features(data):
    return data.drop(columns=DROP_COLUMNS)


@st.cache_resource
def train_models(data):
    features = build_features(data)
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERICAL_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1500, random_state=42)),
        ]
    )
    forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=250, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]
    )
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    probabilities = forest.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }
    importance = permutation_importance(
        forest, x_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    importance_frame = pd.DataFrame({"Feature": x_test.columns, "Importance": importance.importances_mean})
    importance_frame = importance_frame.sort_values("Importance", ascending=False).head(10)
    return logistic, forest, metrics, importance_frame


def risk_band(probability):
    if probability >= 0.65:
        return "High risk", "#d95d39"
    if probability >= 0.35:
        return "Medium risk", "#d4a72c"
    return "Low risk", "#3b8c6e"


st.markdown(
    """
    <style>
    :root { --ink: #1d2a2b; --muted: #687778; --paper: #f5f3ee; --teal: #1f7772; --coral: #d95d39; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #173b3c; }
    [data-testid="stSidebar"] * { color: #f5f3ee; }
    .hero { padding: 1.3rem 0 1rem; border-bottom: 1px solid #d8d6cf; margin-bottom: 1.2rem; }
    .eyebrow { color: var(--coral); font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { color: var(--ink); font-family: Georgia, serif; font-size: clamp(2.2rem, 5vw, 4.7rem); line-height: .95; margin: .25rem 0 .8rem; }
    .hero p { color: var(--muted); max-width: 680px; font-size: 1.05rem; }
    div[data-testid="stMetric"] { background: #fffdf8; border: 1px solid #d8d6cf; padding: 1rem; border-radius: 6px; }
    h2, h3 { color: var(--ink); font-family: Georgia, serif; }
    </style>
    <div class="hero">
      <div class="eyebrow">European retail banking / predictive operations</div>
      <h1>Know who needs<br>attention next.</h1>
      <p>Churn probability, customer-level risk scoring, and practical what-if analysis for focused retention work.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}")
    st.stop()

data = load_data()
logistic_model, model, metrics, importance_frame = train_models(data)

with st.sidebar:
    st.markdown("## Bank Churn Intelligence")
    page = st.radio("View", ["Overview", "Risk calculator", "What-if simulator"], label_visibility="collapsed")
    st.caption(f"Scored population: {len(data):,} customers")
    st.caption("Model: balanced random forest · stratified 80/20 split")

if page == "Overview":
    churn_rate = data[TARGET].mean()
    high_risk_count = int((model.predict_proba(build_features(data))[:, 1] >= 0.65).sum())
    metric_columns = st.columns(4)
    metric_columns[0].metric("Customers", f"{len(data):,}")
    metric_columns[1].metric("Observed churn", f"{churn_rate:.1%}")
    metric_columns[2].metric("High-risk customers", f"{high_risk_count:,}")
    metric_columns[3].metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Risk distribution")
        scored = data[["Geography", "Gender", TARGET]].copy()
        scored["Churn probability"] = model.predict_proba(build_features(data))[:, 1]
        histogram = px.histogram(
            scored, x="Churn probability", color=TARGET, nbins=24,
            color_discrete_map={0: "#72aaa3", 1: "#d95d39"},
            labels={TARGET: "Exited"},
        )
        histogram.update_layout(bargap=0.08, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(histogram, width="stretch")
    with right:
        st.subheader("Model scorecard")
        st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Score"]).assign(Score=lambda frame: frame["Score"].map(lambda value: f"{value:.3f}")), hide_index=True, width="stretch")
        st.caption("Metrics are measured on the held-out stratified test set.")

    st.subheader("What drives risk?")
    importance_chart = px.bar(
        importance_frame.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=[[0, "#b9d8d0"], [1, "#d95d39"]],
    )
    importance_chart.update_layout(showlegend=False, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(importance_chart, width="stretch")

elif page == "Risk calculator":
    st.subheader("Customer churn risk calculator")
    st.caption("Enter a customer profile to receive a probability and a practical risk band.")
    left, right = st.columns(2)
    with left:
        geography = st.selectbox("Geography", sorted(data["Geography"].unique()))
        gender = st.selectbox("Gender", sorted(data["Gender"].unique()))
        credit_score = st.slider("Credit score", 350, 850, 650)
        age = st.slider("Age", 18, 95, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 300000.0, 75000.0, step=500.0)
    with right:
        products = st.slider("Number of products", 1, 4, 1)
        card = st.checkbox("Has credit card", value=True)
        active = st.checkbox("Active member", value=True)
        salary = st.number_input("Estimated salary", 0.0, 250000.0, 100000.0, step=500.0)
    customer = pd.DataFrame([{
        "CreditScore": credit_score, "Geography": geography, "Gender": gender, "Age": age,
        "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
        "HasCrCard": int(card), "IsActiveMember": int(active), "EstimatedSalary": salary,
    }])
    customer["BalanceSalaryRatio"] = customer["Balance"] / customer["EstimatedSalary"].clip(lower=1)
    customer["ProductDensity"] = customer["NumOfProducts"] / customer["Tenure"].replace(0, 1)
    customer["EngagementProduct"] = customer["IsActiveMember"] * customer["NumOfProducts"]
    customer["AgeTenureInteraction"] = customer["Age"] * customer["Tenure"]
    probability = float(model.predict_proba(customer)[0, 1])
    band, color = risk_band(probability)
    st.markdown(f"<div style='border-left: 6px solid {color}; background:#fffdf8; padding:1rem 1.2rem; margin-top:1rem'><div style='font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; color:#687778'>Predicted churn probability</div><div style='font: 3rem Georgia; color:#1d2a2b'>{probability:.1%}</div><strong style='color:{color}'>{band}</strong></div>", unsafe_allow_html=True)

else:
    st.subheader("What-if scenario simulator")
    st.caption("See how engagement and product changes alter the same customer's risk score.")
    selected_id = st.selectbox("Customer", data["CustomerId"].astype(str).head(500).tolist())
    baseline = data[data["CustomerId"].astype(str) == selected_id].iloc[0].copy()
    active_now = st.checkbox("Active member", value=bool(baseline["IsActiveMember"]))
    products_now = st.slider("Number of products", 1, 4, int(baseline["NumOfProducts"]))
    scenario = baseline.to_frame().T
    scenario["IsActiveMember"] = int(active_now)
    scenario["NumOfProducts"] = products_now
    scenario["ProductDensity"] = scenario["NumOfProducts"] / scenario["Tenure"].replace(0, 1)
    scenario["EngagementProduct"] = scenario["IsActiveMember"] * scenario["NumOfProducts"]
    current_probability = float(model.predict_proba(build_features(baseline.to_frame().T))[0, 1])
    scenario_probability = float(model.predict_proba(build_features(scenario))[0, 1])
    first, second, third = st.columns(3)
    first.metric("Baseline risk", f"{current_probability:.1%}")
    second.metric("Scenario risk", f"{scenario_probability:.1%}", delta=f"{scenario_probability - current_probability:.1%}")
    third.metric("Risk band", risk_band(scenario_probability)[0])
    comparison = pd.DataFrame({"State": ["Baseline", "Scenario"], "Churn probability": [current_probability, scenario_probability]})
    chart = px.bar(comparison, x="State", y="Churn probability", color="State", color_discrete_sequence=["#72aaa3", "#d95d39"], text_auto=".1%")
    chart.update_yaxes(range=[0, 1], tickformat=".0%")
    chart.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(chart, width="stretch")
