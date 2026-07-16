# Project Context: AI-Powered Payment Risk Engine

**How to use this file:** Paste it into a new Claude conversation to bring it up to speed, or keep it in the repo as the shared source of truth. Updated after the coding phase is complete.

---

## 1. Project overview

- **Course:** CS6140 Machine Learning, Final Project.
- **Team:** Keshav Girish Adkar, Atharva Mahadik (both second-year MS Computer Science, expected May 2027).
- **Deliverables:** 1-page proposal (submitted). Final report due July 26, scientific-paper format, max 8 pages, 11 pt single-spaced, plus a public GitHub repo.
- **Repo:** https://github.com/KESHAV8087/Payment-Risk-Engine
- **Status: coding phase complete.** Remaining work is the written report.

**What was built:** a hybrid fraud risk engine. A calibrated XGBoost classifier produces a risk score; a cost-sensitive threshold converts it to an approve / review-decline decision; SHAP attributions identify the driving factors; an LLM (Groq Llama 3.3 70B) generates a plain-language explanation; and a faithfulness check verifies the explanation references the SHAP-identified factors. Deployed as a Streamlit dashboard.

---

## 2. Working conventions

- **Cell by cell.** One notebook cell at a time; verify pasted output before the next cell.
- **Concept before code.** Each code cell is preceded by a short plain-language explanation.
- **No em-dashes or en-dashes** in any deliverable. Use commas, colons, or parentheses.
- **Log every prompt** in the AI Prompt Log (required for grading); note who ran it.
- **Guard against leakage.** Features are label-free; resampling/calibration fit on training data only.
- **Git rhythm.** Ctrl+S to save the notebook before committing (unsaved cells are not written to disk). Then add, commit, push. Pull before starting a session.
- **Save before running.** VS Code does not auto-save; imports read from disk, so unsaved edits are invisible.

---

## 3. Dataset

- **Source:** Sparkov Credit Card Transactions Fraud Detection dataset, Kaggle: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Not in git** (~470 MB, exceeds GitHub limits). Each person downloads it and places `fraudTrain.csv` and `fraudTest.csv` in a local `data/` folder. Notebook uses relative paths (`../data/...`).
- **Shape:** train 1,296,675 rows, test 555,719 rows, 23 columns, no missing values.
- **Fraud rate:** train 0.579%, test 0.386% (a real distribution shift; train is earlier, test later). Splits kept separate for temporal validation.

---

## 4. Environment note (important)

SHAP's Numba dependency requires **NumPy below 2.5**. Install with `pip install "numpy<2.5"`. The Groq explanation layer needs a `GROQ_API_KEY` in a gitignored `.env` file at the project root. scikit-learn 1.9 removed `cv="prefit"`; we use `FrozenEstimator` for calibration instead.

---

## 5. Repository layout

```
Payment Risk Engine/
|-- data/                 (gitignored: fraudTrain.csv, fraudTest.csv)
|-- notebooks/
|   `-- 01_eda.ipynb       full analysis, cells 1-30
|-- src/
|   `-- risk_engine.py     shared scoring + explanation module (notebook and app import this)
|-- app/
|   `-- app.py             Streamlit dashboard
|-- models/
|   |-- calibrated_xgb.joblib      final risk scorer
|   |-- xgb_raw.joblib             raw model for SHAP
|   |-- operating_point.json       threshold + assumptions + metrics
|   |-- feature_columns.json       exact feature order
|   |-- faithfulness_eval.csv      per-transaction explanations + scores
|   `-- faithfulness_summary.json  aggregate faithfulness stats
|-- .env                  (gitignored: GROQ_API_KEY)
|-- .gitignore
`-- README.md
```

---

## 6. v1 feature set (25 features, label-free)

Built identically for train and test by `build_features`:
- Temporal: `hour_sin`, `hour_cos`, `is_night` (22:00-03:59), `day_of_week`, `month`
- Amount: `amt`, `log_amt`
- `age` (from dob at transaction date), `city_pop`
- One-hot: 14 `category` dummies, 2 `gender` dummies

Dropped: high-cardinality strings (merchant 693, city 894, job 494, state 51), PII/ids, and `distance_km` (no signal, simulator artifact). `log_amt`, `hour_sin`, `hour_cos` are excluded from the explanation top-factors as redundant/un-verbalizable encodings.

---

## 7. Key results (all in the notebook and models/)

**EDA signals (validated):** amount (fraud median $396 vs legit $47, bimodal); hour (10-25x fraud-rate spike at 22:00-03:59); category (11x spread, shopping_net 1.76% highest, grocery_pos surprisingly high at 1.41%). Distance rejected (fraud and legit distributions identical; Sparkov places merchants randomly around the cardholder regardless of fraud).

**Models (temporal test set):**
- Logistic regression baseline: PR-AUC 0.180 (no-skill baseline 0.0039), ROC-AUC 0.962
- XGBoost (scale_pos_weight 171.75): PR-AUC 0.866, ROC-AUC 0.998, a 381% PR-AUC improvement
- After isotonic calibration: PR-AUC 0.848, Brier score improved from 0.00377 to 0.00119, mean predicted probability matches the actual 0.0039 fraud rate

**Cost-sensitive threshold:** optimal t = 0.029 (assuming $5 per false positive; missed fraud costs the transaction amount). Cuts total expected cost ~70% vs the default 0.5 threshold. Recall 0.929, precision 0.319 at the operating point.

**Explanation faithfulness:** mean 0.974, median 1.0, 70% fully faithful across 30 evaluated flagged transactions. Metric is lexical and conservative. Remaining sub-1.0 cases are the LLM omitting low-strength factors (month, city population, day-of-week) for brevity, not fabrications. Metric checks factor coverage, not causal-direction correctness (noted as a limitation).

**Notable nuance:** SHAP local attributions can differ from global EDA rates (e.g., shopping_net can show as risk-lowering for a specific transaction given learned interactions). Good Analysis-section point.

**Provider change from proposal:** the proposal named the Claude API; the implementation uses Groq/Llama 3.3 70B for cost (free tier). To be documented in the report Methods. The faithfulness method is model-agnostic.

---

## 8. What is left: the report

Map sections to the rubric point values:
- **Introduction** (motivation, target, 2-3 challenges), **Proposed Method** (1 paragraph), **Related Work** (11 pts: 5-6 published papers), **Related Kaggle implementations** (11 pts: 2-3 kernels on this dataset with URLs), **Data Analysis** (8 pts: plots + formulas), **Proposed Method technical** (8 pts: how XGBoost/SHAP work, formulas), **Analysis** (15 pts, largest: why the features/model work, generalization), **Experimental Setup** (8 pts: split, no-leakage, cost model), **Results** (11 pts: table + interpretation + app screenshots), **Conclusion** (4 pts), **References** (4 pts), **Statement of Contributions**.
- Highest-value, least-started items: Related Work and Related Kaggle implementations (need literature/kernel hunting), and the Analysis section.
- App screenshots (input form + REVIEW/DECLINE result) are captured for the Results section.
