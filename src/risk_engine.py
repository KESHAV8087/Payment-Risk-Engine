# src/risk_engine.py
# Shared scoring + explanation logic for the Payment Risk Engine.
# Both the notebook and the Streamlit app import from here so they score identically.

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]     # project root (one level up from src/)
MODELS_DIR = ROOT / "models"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Features whose information is redundant or not human-verbalizable; excluded from
# the top-factor explanation (log_amt duplicates amt; hour_sin/cos duplicate is_night).
EXCLUDE = ("log_amt", "hour_sin", "hour_cos")

MONTHS = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
          7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}

# ---------------------------------------------------------------------------
# Artifact loading (done once)
# ---------------------------------------------------------------------------
def load_artifacts():
    calibrated = joblib.load(MODELS_DIR / "calibrated_xgb.joblib")
    xgb_raw = joblib.load(MODELS_DIR / "xgb_raw.joblib")
    with open(MODELS_DIR / "operating_point.json") as f:
        operating = json.load(f)
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return calibrated, xgb_raw, operating, feature_columns

# ---------------------------------------------------------------------------
# Feature engineering (must match the notebook's build_features exactly)
# ---------------------------------------------------------------------------
def build_features(df, feature_columns):
    """Build the 25-feature modeling row(s) from raw transaction fields,
    aligned to the exact saved column order (missing one-hots filled with 0)."""
    out = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["trans_date_trans_time"])
    hour = dt.dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = hour.isin([22, 23, 0, 1, 2, 3]).astype(int)
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["amt"] = df["amt"].values
    out["log_amt"] = np.log1p(df["amt"].values)
    dob = pd.to_datetime(df["dob"])
    out["age"] = (dt - dob).dt.days / 365.25
    out["city_pop"] = df["city_pop"].values
    cat = pd.get_dummies(df["category"], prefix="cat").astype(int)
    gen = pd.get_dummies(df["gender"], prefix="gender").astype(int)
    out = pd.concat([out, cat, gen], axis=1)
    # Align to the exact training column set/order; fill any absent dummies with 0
    out = out.reindex(columns=feature_columns, fill_value=0)
    return out

# ---------------------------------------------------------------------------
# Human-readable factor translation
# ---------------------------------------------------------------------------
def humanize(feature, value):
    if feature.startswith("cat_"):
        return ("merchant category", feature[4:].replace("_", " "))
    if feature.startswith("gender_"):
        return ("cardholder gender", feature[7:])
    if feature == "is_night":
        return ("time of day", "late night (22:00-04:00)" if value >= 0.5 else "daytime")
    if feature == "amt":
        return ("transaction amount", f"${value:,.2f}")
    if feature == "day_of_week":
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return ("day of week", days[int(value)] if 0 <= int(value) <= 6 else str(value))
    if feature == "month":
        return ("month", str(int(value)))
    if feature == "age":
        return ("cardholder age", f"{value:.0f} years")
    if feature == "city_pop":
        return ("city population", f"{int(value):,}")
    return (feature, f"{value:.4f}")

# ---------------------------------------------------------------------------
# SHAP explanation context
# ---------------------------------------------------------------------------
def build_context(X_row, shap_row, feature_columns, proba, threshold, top_k=6):
    contribs = []
    for name, val, sv in zip(feature_columns, X_row.values[0], shap_row):
        if name in EXCLUDE:
            continue
        contribs.append({"feature": name, "value": float(val), "shap": float(sv),
                         "direction": "toward_fraud" if sv > 0 else "toward_legit"})
    contribs.sort(key=lambda d: abs(d["shap"]), reverse=True)
    factors = []
    for c in contribs[:top_k]:
        label, detail = humanize(c["feature"], c["value"])
        factors.append({"label": label, "detail": detail, "direction": c["direction"],
                        "strength": round(abs(c["shap"]), 3),
                        "_feature": c["feature"], "_value": c["value"]})
    return {"risk_score": round(float(proba), 4), "threshold": round(float(threshold), 4),
            "decision": "review/decline" if proba >= threshold else "approve",
            "top_factors": factors}

# ---------------------------------------------------------------------------
# LLM natural-language explanation (Groq / Llama)
# ---------------------------------------------------------------------------
def get_groq_client():
    load_dotenv(ROOT / ".env")
    return Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_explanation(context, client, model=GROQ_MODEL):
    factor_lines = []
    for f in context["top_factors"]:
        push = "increases fraud risk" if f["direction"] == "toward_fraud" else "lowers fraud risk"
        factor_lines.append(f'- {f["label"]}: {f["detail"]} ({push})')
    factors_text = "\n".join(factor_lines)
    system = ("You are a fraud-analysis assistant. Explain a model's decision in 2-3 plain "
              "sentences for a human reviewer. Use ONLY the factors provided. Do not invent any "
              "details, numbers, or factors that are not listed. Do not mention machine learning "
              "internals or SHAP.")
    user = (f"A transaction received a fraud risk score of {context['risk_score']:.3f} "
            f"(threshold {context['threshold']:.3f}), so the recommendation is: {context['decision']}.\n\n"
            f"The factors that most influenced this decision were:\n{factors_text}\n\n"
            f"Write a short explanation of why the transaction was {context['decision']}.")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2, max_tokens=180,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Faithfulness check
# ---------------------------------------------------------------------------
def check_faithfulness(context, explanation_text):
    text = explanation_text.lower()
    results = []
    for f in context["top_factors"]:
        cues = set()
        cues.update(f["label"].lower().split())
        cues.update(str(f["detail"]).lower().replace("$", "").replace(",", "").split())
        if f.get("_feature") == "month":
            cues.add(MONTHS.get(int(f["_value"]), ""))
        cues -= {"of", "the", "a", "log", "years", "(log)", ""}
        cues = {c for c in cues if len(c) > 2}
        mentioned = any(cue in text for cue in cues)
        results.append({"label": f["label"], "detail": f["detail"], "direction": f["direction"],
                        "strength": f["strength"], "mentioned": mentioned})
    total_w = sum(r["strength"] for r in results) or 1.0
    covered_w = sum(r["strength"] for r in results if r["mentioned"])
    return covered_w / total_w, results

# ---------------------------------------------------------------------------
# End-to-end scoring for a single raw transaction (the app's main entry point)
# ---------------------------------------------------------------------------
def score_transaction(raw, calibrated, xgb_raw, feature_columns, threshold, client, explain=True):
    """raw: dict of one transaction's raw fields. Returns risk score, decision, and explanation."""
    df = pd.DataFrame([raw])
    X = build_features(df, feature_columns)
    proba = float(calibrated.predict_proba(X)[:, 1][0])

    result = {"risk_score": round(proba, 4),
              "decision": "review/decline" if proba >= threshold else "approve",
              "threshold": threshold}
    if explain:
        explainer = shap.TreeExplainer(xgb_raw)
        shap_row = explainer.shap_values(X)[0]
        context = build_context(X, shap_row, feature_columns, proba, threshold)
        explanation = generate_explanation(context, client)
        faith, factor_results = check_faithfulness(context, explanation)
        result.update({"context": context, "explanation": explanation,
                       "faithfulness": round(faith, 3), "factors": factor_results})
    return result