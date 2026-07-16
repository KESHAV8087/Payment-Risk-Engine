# AI-Powered Payment Risk Engine

CS6140 Machine Learning, Final Project
Keshav Girish Adkar and Atharva Mahadik

A hybrid fraud-detection system: a calibrated XGBoost classifier produces a risk
score, a cost-sensitive threshold turns it into an approve / review-decline decision,
SHAP attributions identify the driving factors, and an LLM (Groq Llama 3.3 70B)
generates a plain-language explanation that is checked for faithfulness against the
SHAP factors.

## Repository layout

    Payment Risk Engine/
    ├── data/                 (not tracked; download separately, see below)
    ├── notebooks/
    │   └── 01_eda.ipynb       full analysis: EDA, model, calibration, SHAP, faithfulness
    ├── src/
    │   └── risk_engine.py     shared scoring + explanation logic
    ├── app/
    │   └── app.py             Streamlit dashboard
    ├── models/                saved model artifacts and evaluation outputs
    └── README.md

## 1. Dataset setup

Uses the Credit Card Transactions Fraud Detection dataset (Sparkov simulator):
https://www.kaggle.com/datasets/kartik2112/fraud-detection

The CSV files are not tracked in git (they exceed GitHub size limits). Download the
dataset, then place both files in a `data/` folder at the project root:

    data/fraudTrain.csv
    data/fraudTest.csv

## 2. Environment

Python 3.12. Note the NumPy pin: SHAP's Numba dependency requires NumPy below 2.5.

    pip install "numpy<2.5" pandas matplotlib scikit-learn xgboost shap imbalanced-learn
    pip install streamlit groq python-dotenv joblib

## 3. API key (required for the explanation layer)

The explanation layer calls the Groq API. Create a file named `.env` at the project
root containing:

    GROQ_API_KEY=your_groq_key_here

The `.env` file is gitignored and must not be committed. Get a free key at
https://console.groq.com

## 4. Running

Notebook: open `notebooks/01_eda.ipynb` and Run All (rebuilds all state).

App (run from the project root so paths resolve):

    streamlit run app/app.py

The dashboard opens at http://localhost:8501. Enter a transaction and click
"Assess transaction" to see the risk score, decision, factor breakdown, LLM
explanation, and faithfulness score.

## Key results

- XGBoost PR-AUC 0.87 on the temporal test set (vs 0.18 logistic baseline)
- Cost-sensitive operating threshold: 0.029 (assumes $5 per false positive)
- Explanation faithfulness: mean 0.974 across evaluated flagged transactions