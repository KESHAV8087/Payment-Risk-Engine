# app/app.py
# Streamlit dashboard for the Payment Risk Engine.
# Run from the project root:  streamlit run app/app.py

import sys
from pathlib import Path

# Make src/ importable regardless of where streamlit is launched from
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import risk_engine

st.set_page_config(page_title="Payment Risk Engine", page_icon="💳", layout="centered")

# Load artifacts once and cache them across reruns
@st.cache_resource
def _load():
    calibrated, xgb_raw, operating, feature_columns = risk_engine.load_artifacts()
    client = risk_engine.get_groq_client()
    return calibrated, xgb_raw, operating, feature_columns, client

calibrated, xgb_raw, operating, feature_columns, client = _load()
threshold = operating["threshold"]

st.title("AI-Powered Payment Risk Engine")
st.caption(
    f"Calibrated XGBoost risk score with a cost-sensitive threshold of {threshold:.3f}, "
    "SHAP-based factor attribution, and an LLM explanation with a faithfulness check."
)

CATEGORIES = [
    "shopping_net", "misc_net", "grocery_pos", "shopping_pos", "gas_transport",
    "misc_pos", "grocery_net", "travel", "entertainment", "personal_care",
    "kids_pets", "food_dining", "home", "health_fitness",
]

st.subheader("Transaction details")
col1, col2 = st.columns(2)
with col1:
    amt = st.number_input("Amount ($)", min_value=0.0, value=899.99, step=10.0)
    category = st.selectbox("Merchant category", CATEGORIES)
    gender = st.radio("Cardholder gender", ["F", "M"], horizontal=True)
    
import datetime
with col2:
    date = st.date_input(
        "Transaction date",
        value=datetime.date(2020, 6, 15),
        min_value=datetime.date(2019, 1, 1),
        max_value=datetime.date(2020, 12, 31),
    )
    time = st.time_input("Transaction time", value=datetime.time(23, 14))
    dob = st.date_input(
        "Cardholder date of birth",
        value=datetime.date(1991, 4, 2),
        min_value=datetime.date(1920, 1, 1),
        max_value=datetime.date(2007, 1, 1),
    )
    city_pop = st.number_input("City population", min_value=1, value=120000, step=1000)

if st.button("Assess transaction", type="primary"):
    raw = {
        "trans_date_trans_time": f"{date} {time}",
        "amt": float(amt),
        "category": category,
        "gender": gender,
        "city_pop": int(city_pop),
        "dob": str(dob),
    }
    with st.spinner("Scoring and generating explanation..."):
        result = risk_engine.score_transaction(
            raw, calibrated, xgb_raw, feature_columns, threshold, client, explain=True
        )

    # Headline decision
    decision = result["decision"]
    score = result["risk_score"]
    if decision == "approve":
        st.success(f"APPROVE  |  risk score {score:.3f}  (threshold {threshold:.3f})")
    else:
        st.error(f"REVIEW / DECLINE  |  risk score {score:.3f}  (threshold {threshold:.3f})")

    # Explanation
    st.subheader("Explanation")
    st.write(result["explanation"])
    st.caption(f"Faithfulness score: {result['faithfulness']:.2f} "
               "(fraction of top factors the explanation references)")

    # Factor breakdown
    st.subheader("Top contributing factors")
    for f in result["context"]["top_factors"]:
        direction = "increases risk" if f["direction"] == "toward_fraud" else "lowers risk"
        st.write(f"- **{f['label']}**: {f['detail']}  ({direction}, strength {f['strength']})")

st.divider()
st.caption(
    "Model: calibrated XGBoost. Explanation: Groq Llama 3.3 70B. "
    "For academic demonstration; trained on the Sparkov simulated dataset."
)