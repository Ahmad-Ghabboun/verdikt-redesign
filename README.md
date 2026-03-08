# LLM Preference Predictor — LMSYS Chatbot Arena

## What This Project Does

This project builds a complete machine learning pipeline to predict **human preference
in LLM head-to-head comparisons**. Using the LMSYS Chatbot Arena dataset from Kaggle,
we train six models to classify whether humans prefer Response A, Response B, or consider
them a tie — based on prompt characteristics, response lengths, model identities, and
semantic embeddings of the prompt text.

### Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1 | Data Cleaning & Feature Engineering | `X_features.parquet`, `y_target.parquet`, `models/pca.joblib`, `embeddings.npy` |
| 2 | Descriptive Analytics (EDA) | 6 PNG plots in `eda_plots/` |
| 3 | Predictive Modeling (5-fold CV) | 6 `.joblib` model files, `model_comparison.csv` |
| 4 | SHAP Explainability | 3 SHAP plots in `shap_plots/` |
| 5 | Streamlit Dashboard | `app.py` with 4 interactive tabs |

### Features Used

- **Structured**: `prompt_length`, `response_a_length`, `response_b_length`,
  `length_difference`, `length_ratio`, `prompt_word_count`, `is_tie`
- **Categorical**: one-hot encoded `model_a` and `model_b`
- **Semantic**: 50-dimensional PCA reduction of 384-dim `all-MiniLM-L6-v2` embeddings

### Models Compared

Logistic Regression · Lasso (L1) · Ridge (L2) · CART Decision Tree ·
Random Forest · LightGBM — all evaluated with 5-fold stratified cross-validation.

---

## Dataset Source

**LMSYS Chatbot Arena** — Kaggle Competition
<https://www.kaggle.com/competitions/lmsys-chatbot-arena>

Each row contains a user prompt, two LLM responses (A and B), the model names,
and a human-preference label (`winner_model_a`, `winner_model_b`, or `winner_tie`).
The dataset has ~57,000 rows; we sample 30,000 for this pipeline.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (Steps 1–4)

```bash
python pipeline.py
```

> On the first run this downloads and applies `all-MiniLM-L6-v2` locally to embed
> 30,000 prompts (no API key needed). Embeddings are cached in `embeddings.npy` —
> subsequent runs skip this step and load the cache directly.

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Project Structure

```
├── pipeline.py              # Steps 1–4: complete ML pipeline
├── app.py                   # Step 5: Streamlit dashboard
├── requirements.txt         # Python dependencies
├── embeddings.npy           # Cached 384-dim prompt embeddings (created by pipeline)
├── X_features.parquet       # Final feature matrix (created by pipeline)
├── y_target.parquet         # Encoded target labels (created by pipeline)
├── model_comparison.csv     # CV results for all 6 models (created by pipeline)
├── models/
│   ├── pca.joblib                        # Fitted PCA (384→50)
│   ├── top_models.joblib                 # Top-5 model names for Streamlit UI
│   ├── all_feature_columns.joblib        # Ordered column list for prediction
│   ├── model_dummies_columns.joblib      # One-hot column names
│   ├── structured_feature_columns.joblib # Structured-only column names
│   ├── shap_insights.joblib              # Top SHAP feature + best model name
│   ├── Logistic_Regression.joblib
│   ├── Lasso_L1.joblib
│   ├── Ridge_L2.joblib
│   ├── CART_Decision_Tree.joblib
│   ├── Random_Forest.joblib
│   ├── LightGBM.joblib
│   └── best_struct_<ModelName>.joblib    # Best model retrained on structured features (SHAP)
├── eda_plots/
│   ├── 01_target_distribution.png
│   ├── 02_response_length_distribution.png
│   ├── 03_length_difference_boxplot.png
│   ├── 04_top_model_pairs.png
│   ├── 05_prompt_length_violin.png
│   └── 06_correlation_heatmap.png
└── shap_plots/
    ├── 01_shap_beeswarm.png
    ├── 02_shap_bar.png
    └── 03_shap_waterfall.png
```

---

## Connection to Dual-LLM Validation Systems

This project's findings apply directly to **dual-LLM validation pipelines** — a pattern
increasingly used in production AI systems for quality control:

```
Content → LLM Judge A ──┐
                         ├── Agreement? → High confidence (no human review)
Content → LLM Judge B ──┘  Disagree?  → Escalate to human review
```

Understanding *what drives human preference* provides three practical benefits:

1. **Threshold calibration** — length ratio and response completeness predict human
   preference with measurable F1; these signals can set automated review thresholds.

2. **Judge model selection** — SHAP reveals which model identity features matter most;
   this informs which model to deploy as the "judge" for a given content domain.

3. **Confidence scoring** — the predicted class probabilities (Model A / B / Tie) can
   serve as a disagreement score between the two judges, replacing arbitrary thresholds
   with data-driven ones.
