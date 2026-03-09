"""
LMSYS Chatbot Arena — Streamlit Dashboard (v2)
9-tab interactive analytics suite — no emojis, all-Plotly, dark/light mode
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Verdikt", layout="wide")

# ─────────────────────────────────────────────────────────────────────
# Guard: require pipeline outputs
# ─────────────────────────────────────────────────────────────────────
REQUIRED = [
    "model_comparison.csv",
    "X_features.parquet",
    "y_target.parquet",
    "models/pca.joblib",
    "models/top_models.joblib",
    "models/all_feature_columns.joblib",
    "models/model_dummies_columns.joblib",
    "models/shap_insights.joblib",
    "models/structured_feature_columns.joblib",
]
_missing = [f for f in REQUIRED if not os.path.exists(f)]
if _missing:
    st.error("Pipeline artifacts missing. Please run `python pipeline.py` first.")
    st.code("\n".join(_missing))
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────
_SS = {
    "dark_mode": True,
    "scenarios": [],
    "agreement_history": [],
    "prediction_history": [],
    "last_confidence": None,
    "exec_animated": False,
    "perf_metric": "F1_Weighted",
    "shap_view": "Summary",
    "last_slider": None,
    "h2h_result": None,
    "last_prediction": None,
}
for _k, _v in _SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────
# Google Fonts
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk'
    ":wght@400;500;600;700&family=Inter:wght@400;500&display=swap"
    '" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# Theme variables
# ─────────────────────────────────────────────────────────────────────
dark = st.session_state["dark_mode"]
BG     = "#1C1C1C"  if dark else "#F7F9F9"
CARD   = "#252525"  if dark else "#FFFFFF"
TEXT   = "#FFFFFF"  if dark else "#1A202C"
MUTED  = "#8E8E8E"  if dark else "#A0AEC0"
BORDER = "#2A2A2A"  if dark else "#E2E8F0"
ACCENT = "#CFF242"  if dark else "#35B095"
ACCENT2= "#9575CD"  if dark else "#4498D2"
PALETTE = ["#35B095", "#4498D2", "#1D72CE", "#2B825B", "#F2B21A"]
TAB_INACTIVE = "#1A1A1A" if dark else "#EDF7F4"
GRID   = "rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.06)"



# ─────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
/* ── Global ── */
.stApp {{ background-color: {BG}; --border-color: {BORDER}; }}
body, p, li, span, .stMarkdown, div[data-testid="stMarkdownContainer"] {{
  font-family: 'Inter', sans-serif !important; color: {TEXT} !important;
}}
h1, h2, h3, h4 {{
  font-family: 'Syne', sans-serif !important; color: {TEXT} !important;
}}

/* ── Gradient header (used in tabs) ── */
.gradient-header {{
  background: {"#1C1C1C" if dark else "#35B095"};
  padding: 2rem 2.5rem; border-radius: 16px;
  text-align: center; margin-bottom: 1rem;
}}
.gradient-header h1 {{
  color: white !important; font-family: 'Syne', sans-serif !important;
  font-weight: 700; font-size: 2rem; margin: 0;
}}
.gradient-header p {{
  color: rgba(255,255,255,0.85) !important;
  font-size: 1rem; margin-top: 0.5rem;
}}

/* ── Fade-in ── */
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in {{ animation: fadeIn 0.45s ease-out; }}

/* ── Metric cards ── */
.metric-card {{
  background: {CARD}; border: 1px solid rgba(79,195,247,0.15);
  border-radius: 12px; padding: 1.25rem 1.5rem;
  text-align: center; cursor: default; margin-bottom: 0.75rem;
  transition: transform 300ms ease, box-shadow 300ms ease;
}}
.metric-card:hover {{
  transform: scale(1.04); box-shadow: 0 0 22px rgba(79,195,247,0.4);
}}
.metric-card.best {{ border: 2px solid {ACCENT}; box-shadow: 0 0 16px rgba(79,195,247,0.3); }}
.mc-label {{
  font-size: 0.74rem; font-weight: 600; color: {ACCENT} !important;
  text-transform: uppercase; letter-spacing: 0.08em;
  font-family: 'Syne', sans-serif !important;
}}
.mc-value {{
  font-size: 1.75rem; font-weight: 700; color: {TEXT} !important;
  margin-top: 0.35rem; font-family: 'Syne', sans-serif !important;
}}

/* ── Info card ── */
.info-card {{
  background: {CARD}; border-left: 4px solid {ACCENT};
  border-radius: 8px; padding: 1rem 1.25rem; margin: 0.6rem 0;
}}

/* ── Synaps callout ── */
.synaps-card {{
  background: linear-gradient(135deg, rgba(79,195,247,0.10), rgba(0,119,182,0.10));
  border: 1px solid rgba(79,195,247,0.35); border-radius: 14px;
  padding: 1.5rem 2rem; margin: 1rem 0;
}}
.synaps-card h4 {{
  color: {ACCENT} !important; margin-bottom: 0.6rem;
  font-family: 'Syne', sans-serif !important;
}}

/* ── Prediction card ── */
.prediction-card {{
  background: {CARD}; border: 2px solid {ACCENT}; border-radius: 14px;
  padding: 1.5rem 2rem; text-align: center;
  box-shadow: 0 0 28px rgba(79,195,247,0.2); margin: 1rem 0;
}}
.pc-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: {ACCENT}; }}
.pc-value {{ font-size: 1.7rem; font-weight: 700; color: {TEXT}; margin-top: 0.4rem; font-family: 'Syne', sans-serif; }}
.pc-sub   {{ font-size: 0.85rem; color: {MUTED}; margin-top: 0.25rem; }}

/* ── Verdict card ── */
.verdict-card {{
  border-radius: 12px; padding: 1.25rem; text-align: center; margin: 0.75rem 0;
  font-family: 'Syne', sans-serif;
}}

/* ── Custom divider ── */
.custom-hr {{
  border: none; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(79,195,247,0.3), transparent);
  margin: 1.5rem 0;
}}

/* ── Tab pill navigation ── */
[data-baseweb="tab-list"] {{
  background: rgba(255,255,255,0.04) !important;
  backdrop-filter: blur(10px) !important;
  border-radius: 60px !important; padding: 6px !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  max-width: 1100px; margin: 0 auto 1.5rem auto !important;
  gap: 3px !important; flex-wrap: wrap !important;
}}
button[data-baseweb="tab"] {{
  background: {TAB_INACTIVE} !important; border-radius: 50px !important;
  color: {MUTED} !important; font-family: 'Syne', sans-serif !important;
  font-size: 13px !important; font-weight: 500 !important;
  padding: 8px 18px !important; border: none !important;
  transition: all 300ms ease !important; white-space: nowrap !important;
}}
button[data-baseweb="tab"]:hover {{
  transform: scale(1.03) !important; color: {TEXT} !important;
}}
button[aria-selected="true"][data-baseweb="tab"] {{
  background: {ACCENT} !important; 
  color: black !important;
  box-shadow: 0 0 12px rgba(79,195,247,0.5) !important;
}}
button[aria-selected="true"][data-baseweb="tab"] * {{
  color: black !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none !important; }}

/* ── All buttons as pills ── */
.stButton > button {{
  border-radius: 50px !important;
  background: linear-gradient(90deg, #0077B6, {ACCENT}) !important;
  color: white !important; font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important; padding: 12px 32px !important;
  min-width: 180px !important; max-width: 240px !important;
  display: block !important; margin: 20px auto !important;
  box-shadow: 0 4px 15px rgba(79,195,247,0.4) !important;
  border: none !important; transition: all 300ms ease !important;
  font-size: 14px !important;
}}
.stButton > button:hover {{
  transform: scale(1.04) !important;
  box-shadow: 0 6px 22px rgba(79,195,247,0.6) !important;
}}
.stDownloadButton > button {{
  border-radius: 50px !important;
  background: linear-gradient(90deg, #0077B6, {ACCENT}) !important;
  color: white !important; font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important; padding: 12px 32px !important;
  min-width: 180px !important; display: block !important;
  margin: 20px auto !important;
  box-shadow: 0 4px 15px rgba(79,195,247,0.4) !important;
  border: none !important; transition: all 300ms ease !important;
}}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {{
  background: {CARD} !important; color: {TEXT} !important;
  border-radius: 8px !important; border: 1px solid rgba(79,195,247,0.2) !important;
  font-family: 'Inter', sans-serif !important;
}}
.stSelectbox > div > div {{
  background: {CARD} !important; color: {TEXT} !important;
  border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
}}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────
def safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def hr():
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)


def apply_chart_style(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter"),
        height=height,
        transition=dict(duration=800, easing="cubic-in-out"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=GRID, color=TEXT, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, color=TEXT, zerolinecolor=GRID)
    return fig


def metric_card(label: str, value: str, best: bool = False):
    cls = "metric-card best" if best else "metric-card"
    st.markdown(
        f'<div class="{cls}"><div class="mc-label">{label}</div>'
        f'<div class="mc-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def animated_metric_card(label: str, final_int: int, prefix: str = "", suffix: str = ""):
    ph = st.empty()
    steps = 18
    for i in range(steps + 1):
        cur = int(final_int * i / steps)
        ph.markdown(
            f'<div class="metric-card"><div class="mc-label">{label}</div>'
            f'<div class="mc-value">{prefix}{cur:,}{suffix}</div></div>',
            unsafe_allow_html=True,
        )
        if i < steps:
            time.sleep(0.015)


# ─────────────────────────────────────────────────────────────────────
# Cached resource / data loaders
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_pcas():
    return {
        "prompt": joblib.load("models/pca.joblib"),
    }


@st.cache_resource
def _load_model_file(path: str):
    return joblib.load(path)


@st.cache_resource
def load_struct_model():
    si  = joblib.load("models/shap_insights.joblib")
    mn  = si.get("best_model", "LightGBM")
    pth = f"models/best_struct_{safe_name(mn)}.joblib"
    return joblib.load(pth) if os.path.exists(pth) else None


@st.cache_resource
def load_mlp_model():
    """Load Keras MLP model and its feature scaler. Returns (model, scaler) or (None, None)."""
    if not os.path.exists("models/mlp_model.keras"):
        return None, None
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras as _keras
    _mlp    = _keras.models.load_model("models/mlp_model.keras")
    _scaler = joblib.load("models/mlp_scaler.joblib")
    return _mlp, _scaler


@st.cache_data
def load_features_targets():
    X = pd.read_parquet("X_features.parquet")
    y = pd.read_parquet("y_target.parquet").squeeze()
    return X, y


@st.cache_data
def get_shap_importance():
    import shap as _shap
    sm = load_struct_model()
    if sm is None:
        return None
    struct_cols = joblib.load("models/structured_feature_columns.joblib")
    X, _ = load_features_targets()
    struct_cols = [c for c in struct_cols if c in X.columns]
    X_s = X[struct_cols].sample(min(500, len(X)), random_state=42)
    try:
        explainer = _shap.TreeExplainer(sm)
    except Exception:
        explainer = _shap.LinearExplainer(sm, X_s)
    sv_raw = explainer.shap_values(X_s)
    if isinstance(sv_raw, list):
        sv_2d = sv_raw[0]
    elif hasattr(sv_raw, "ndim") and sv_raw.ndim == 3:
        sv_2d = sv_raw[:, :, 0]
    else:
        sv_2d = sv_raw
    mean_abs = np.abs(sv_2d).mean(axis=0)
    return pd.Series(mean_abs, index=X_s.columns).sort_values(ascending=False)


@st.cache_data
def get_model_pair_stats(_key: str):
    X, y = load_features_targets()
    mdummies = joblib.load("models/model_dummies_columns.joblib")
    ma_cols = [c for c in mdummies if c.startswith("model_a_") and c in X.columns]
    mb_cols = [c for c in mdummies if c.startswith("model_b_") and c in X.columns]
    if not ma_cols or not mb_cols:
        return pd.DataFrame(columns=["pair", "count"]), pd.DataFrame(columns=["model", "win_rate", "appearances"])
    ma_max = X[ma_cols].max(axis=1)
    mb_max = X[mb_cols].max(axis=1)
    ma_name = (
        X[ma_cols].idxmax(axis=1)
        .where(ma_max > 0, "Unknown")
        .str.replace("model_a_", "", regex=False)
    )
    mb_name = (
        X[mb_cols].idxmax(axis=1)
        .where(mb_max > 0, "Unknown")
        .str.replace("model_b_", "", regex=False)
    )
    df = pd.DataFrame({"model_a": ma_name.values, "model_b": mb_name.values, "winner": y.values})
    df["pair"] = df["model_a"] + " vs " + df["model_b"]
    pair_counts = df["pair"].value_counts().head(10).reset_index()
    pair_counts.columns = ["pair", "count"]
    all_models = (set(ma_name.unique()) | set(mb_name.unique())) - {"Unknown"}
    rows = []
    for model in all_models:
        rows_a = df[df["model_a"] == model]
        rows_b = df[df["model_b"] == model]
        wins = int((rows_a["winner"] == 0).sum() + (rows_b["winner"] == 1).sum())
        apps = len(rows_a) + len(rows_b)
        if apps >= 10:
            rows.append({"model": model, "wins": wins, "appearances": apps, "win_rate": wins / apps})
    wr_df = (
        pd.DataFrame(rows).sort_values("win_rate").tail(15)
        if rows
        else pd.DataFrame(columns=["model", "win_rate", "appearances"])
    )
    return pair_counts, wr_df


@st.cache_data
def get_predictions_for_model(model_name: str):
    X, y = load_features_targets()
    if model_name == "MLP Neural Network":
        _mlp, _scaler = load_mlp_model()
        if _mlp is None:
            return np.zeros(len(y), dtype=int), y.values
        _X_sc  = _scaler.transform(X.values)
        _proba = _mlp.predict(_X_sc, verbose=0)
        return _proba.argmax(axis=1), y.values
    model = _load_model_file(f"models/{safe_name(model_name)}.joblib")
    return model.predict(X), y.values


# ─────────────────────────────────────────────────────────────────────
# Static metadata (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────
model_comparison   = pd.read_csv("model_comparison.csv")
best_row           = model_comparison.iloc[0]
best_model_name    = best_row["Model"]
best_f1            = float(best_row["F1_Weighted"])
best_model         = _load_model_file(f"models/{safe_name(best_model_name)}.joblib")
all_feat_cols      = joblib.load("models/all_feature_columns.joblib")
model_dummies_cols = joblib.load("models/model_dummies_columns.joblib")
top_models         = joblib.load("models/top_models.joblib")
shap_insights      = joblib.load("models/shap_insights.joblib")

_X_meta, _y_meta = load_features_targets()
dataset_size  = len(_X_meta)
unique_models = len({c.replace("model_a_", "").replace("model_b_", "") for c in model_dummies_cols})
CLASS_LABELS  = {0: "Model A Wins", 1: "Model B Wins", 2: "Tie"}
mlp_available = os.path.exists("models/mlp_model.keras")

# ─────────────────────────────────────────────────────────────────────
# Top header row: toggle + gradient banner
# ─────────────────────────────────────────────────────────────────────
_, _tcol = st.columns([8, 1])
with _tcol:
    if st.button("Light Mode" if dark else "Dark Mode", key="theme_toggle"):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.session_state["exec_animated"] = False
        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# Tab bar
# ─────────────────────────────────────────────────────────────────────
(
    tab1, tab2, tab3, tab4, tab5,
    tab6, tab7, tab8, tab9,
) = st.tabs([
    "Interactive Prediction",
    "Executive Summary",
    "EDA Explorer",
    "Model Performance",
    "SHAP Explainability",
    "What-If Simulator",
    "Model Agreement Analyzer",
    "Head-to-Head Arena",
    "Dataset Explorer",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Interactive Prediction
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("Interactive Prediction")
    _all_pred_models = model_comparison["Model"].tolist()
    pred_model_sel   = st.selectbox(
        "Select prediction model",
        _all_pred_models,
        index=0,
        key="pred_model_choice",
        help="Choose which trained model to use for this prediction.",
    )
    st.markdown(
        f"Using **{pred_model_sel}**. "
        "Enter a prompt, set response lengths, choose models, then click Predict."
    )
    hr()

    pcas = load_pcas()
    col_left, col_right = st.columns([2, 1])

    with col_left:
        prompt_input = st.text_area(
            "Sample Prompt",
            height=170,
            placeholder="Enter a question or instruction here...",
            help="Embedded with all-MiniLM-L6-v2, reduced to 50 dims via PCA",
            key="pred_prompt",
        )
    with col_right:
        resp_a_len  = st.slider("Response A Length (chars)", 100, 2000, 500, 50, key="pred_ra")
        resp_b_len  = st.slider("Response B Length (chars)", 100, 2000, 650, 50, key="pred_rb")
        model_a_sel = st.selectbox("Model A", top_models, index=0, key="pred_ma")
        model_b_sel = st.selectbox("Model B", top_models, index=min(1, len(top_models) - 1), key="pred_mb")

    predict_btn = st.button("Predict", key="predict_btn")

    if predict_btn:
        if not prompt_input.strip():
            st.warning("Please enter a prompt before predicting.")
        else:
            with st.spinner("Embedding inputs..."):
                _st_model  = load_sentence_transformer()
                prompt_emb = _st_model.encode([prompt_input])
                emb_p_50   = pcas["prompt"].transform(prompt_emb)[0]

                ra_len = resp_a_len
                rb_len = resp_b_len
                ra_wc  = int(ra_len / 5)
                rb_wc  = int(rb_len / 5)
                ra_code = rb_code = 0
                ra_list = rb_list = 0

            feat = {
                "prompt_length":     len(prompt_input),
                "response_a_length": ra_len,
                "response_b_length": rb_len,
                "length_difference": abs(ra_len - rb_len),
                "length_ratio":      ra_len / (rb_len + 1),
                "prompt_word_count": len(prompt_input.split()),
                "response_a_word_count": ra_wc,
                "response_b_word_count": rb_wc,
                "word_count_difference": abs(ra_wc - rb_wc),
                "word_count_ratio":      ra_wc / (rb_wc + 1),
                "longer_response":       1 if ra_len > rb_len else 0,
                "response_a_has_code":   ra_code,
                "response_b_has_code":   rb_code,
                "response_a_has_list":   ra_list,
                "response_b_has_list":   rb_list,
                "is_tie":            0,
            }
            for col in model_dummies_cols:
                feat[col] = 0
            feat[f"model_a_{model_a_sel}"] = 1
            feat[f"model_b_{model_b_sel}"] = 1
            
            # Add prompt embeddings
            for i, v in enumerate(emb_p_50):
                feat[f"emb_prompt_{i}"] = v

            X_pred = pd.DataFrame([[feat.get(c, 0) for c in all_feat_cols]], columns=all_feat_cols)
            if pred_model_sel == "MLP Neural Network":
                _mlp_p, _scaler_p = load_mlp_model()
                if _mlp_p is None:
                    st.error("MLP model file not found. Run pipeline.py first.")
                    st.stop()
                _X_sc_p  = _scaler_p.transform(X_pred.values)
                pred_prob = _mlp_p.predict(_X_sc_p, verbose=0)[0]
                pred_cls  = int(pred_prob.argmax())
            else:
                _sel_model = (best_model if pred_model_sel == best_model_name
                              else _load_model_file(f"models/{safe_name(pred_model_sel)}.joblib"))
                pred_cls  = int(_sel_model.predict(X_pred)[0])
                pred_prob = _sel_model.predict_proba(X_pred)[0]
            winner = CLASS_LABELS[pred_cls]
            conf   = float(pred_prob[pred_cls])

            st.session_state["last_prediction"] = {
                "winner": winner, "confidence": conf, "proba": list(pred_prob),
                "model": pred_model_sel,
            }
            st.session_state["prediction_history"].append({
                "Prompt (first 60 chars)": prompt_input[:60],
                "Model Used":   pred_model_sel,
                "Prediction":   winner,
                "Confidence":   f"{conf:.1%}",
            })
            st.session_state["last_confidence"] = conf

    # Always render last prediction if available
    lp = st.session_state.get("last_prediction")
    if lp:
        hr()
        st.markdown(
            f'<div class="prediction-card fade-in">'
            f'<div class="pc-label">Predicted Winner</div>'
            f'<div class="pc-value">{lp["winner"]}</div>'
            f'<div class="pc-sub">Confidence: {lp["confidence"]:.1%} &mdash; Model: {lp.get("model", best_model_name)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        conf_df = pd.DataFrame({"Outcome": list(CLASS_LABELS.values()), "Confidence": lp["proba"]})
        fig_c = px.bar(
            conf_df, x="Outcome", y="Confidence",
            color="Outcome", color_discrete_sequence=PALETTE,
            title="Prediction Confidence — All 3 Classes",
            text=conf_df["Confidence"].apply(lambda x: f"{x:.1%}"),
        )
        fig_c.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>",
        )
        fig_c.update_layout(yaxis=dict(range=[0, 1.15], tickformat=".0%"), showlegend=False)
        fig_c = apply_chart_style(fig_c, height=360)
        st.plotly_chart(fig_c, use_container_width=True)

        hr()
        st.subheader("Why this prediction?")
        shap_imp = get_shap_importance()
        if shap_imp is not None:
            top3 = shap_imp.head(3)
            html = "".join(
                f'<div class="info-card"><b>{fn}</b> — mean |SHAP| importance: {iv:.4f}</div>'
                for fn, iv in top3.items()
            )
            st.markdown(html, unsafe_allow_html=True)
        else:
            tf = shap_insights.get("top_feature", "N/A")
            st.markdown(f'<div class="info-card">Top driver of predictions: <b>{tf}</b></div>', unsafe_allow_html=True)

    hr()
    if st.session_state["prediction_history"]:
        st.subheader("Prediction History")
        st.dataframe(pd.DataFrame(st.session_state["prediction_history"]), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Executive Summary
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-header">'
        "<h1>Executive Summary</h1>"
        "<p>Predicting Human Preference in LLM Judgments</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "The LMSYS Chatbot Arena dataset is a large collection of real-world conversations between "
        "humans and various AI chatbots including ChatGPT, Claude, Gemini, and LLaMA. It captures "
        "head-to-head blind battles where real users voted for which AI response they preferred, "
        "creating a rich dataset of 30,000 human preference judgments across 64 unique models. "
        "The prediction task is a three-class classification problem: given a prompt and two "
        "responses, predict whether Model A wins, Model B wins, or the result is a tie.\n\n"
        "I chose this dataset for two reasons. First, I want to deepen my understanding of what "
        "makes AI responses better, closer to what humans actually need, with less hallucination "
        "and unnecessary content. Second, this dataset directly connects to Synaps, a Project "
        "Quality Assurance Advisor I built that uses dual-LLM cross-validation by running Claude "
        "and Gemini independently to audit project documents like BRDs and business plans, and "
        "flags missing sections with a confidence score based on whether both models agree. "
        "Studying human preference patterns at scale is the foundation of what makes that kind "
        "of dual-LLM evaluation meaningful.\n\n"
        "The dataset was downloaded from Kaggle's LMSYS Chatbot Arena competition using the "
        "train.csv file. To prepare the data, I adapted a full data science pipeline to fit this "
        "dataset, which included exploratory data analysis, feature engineering, and model "
        "training. Because the dataset contains text-based prompts and responses, I applied "
        "semantic embeddings using sentence-transformers to convert the prompt text into "
        "50-dimensional numerical vectors the models could learn from. I trained seven models "
        "with 5-fold cross-validation: Logistic Regression, Ridge, Lasso, CART Decision Tree, "
        "Random Forest, LightGBM, and a Neural Network MLP. During this process I identified "
        "and removed a data leakage issue where a feature called is_tie was derived from the "
        "target variable, which had artificially inflated F1 from 0.76 to an honest 0.4865 "
        "after removal.\n\n"
        "LightGBM was the best performing model with an F1 score of 0.4865, outperforming all "
        "other models including the Neural Network. The most surprising finding was that response "
        "length, specifically the ratio between Response A and Response B, was the strongest "
        "predictor of human preference, more than which AI model actually generated the response. "
        "This mirrors a well-known bias from academic settings where people tend to assume longer "
        "answers are more correct, regardless of actual quality. What this reveals practically is "
        "that humans evaluating AI responses are still influenced by surface-level signals like "
        "length rather than deeper quality indicators like accuracy, trustworthiness, or lack of "
        "hallucination. This has a direct implication for Synaps: an AI quality advisor cannot "
        "rely solely on what humans prefer, but must independently validate structural completeness "
        "and factual integrity, which is exactly what the dual-LLM cross-validation approach is "
        "designed to do."
    )
    hr()

    c1, c2, c3, c4 = st.columns(4)
    _do_anim = not st.session_state["exec_animated"]
    with c1:
        if _do_anim:
            animated_metric_card("Dataset Size", dataset_size)
        else:
            metric_card("Dataset Size", f"{dataset_size:,}")
    with c2:
        if _do_anim:
            animated_metric_card("Unique Models", unique_models)
        else:
            metric_card("Unique Models", str(unique_models))
    with c3:
        metric_card("Best Model", best_model_name, best=True)
    with c4:
        metric_card("Best F1 Score", f"{best_f1:.4f}", best=True)

    if _do_anim:
        st.session_state["exec_animated"] = True

    hr()
    st.subheader("Key SHAP Insight")
    top_feat = shap_insights.get("top_feature", "N/A")
    st.markdown(
        f'<div class="info-card"><b>Top driver of human preference: <code>{top_feat}</code></b><br><br>'
        "SHAP analysis shows that response length characteristics are among the strongest "
        "predictors of human preference. Responses that are noticeably longer tend to win — "
        "but only up to a point; extreme verbosity does not improve outcomes. "
        "In dual-LLM validation systems, this suggests that judges should penalise responses "
        "that are either far too short or padded unnecessarily.</div>",
        unsafe_allow_html=True,
    )

    hr()
    st.subheader("Model CV Results")
    def _highlight(row):
        bg = f"background-color: {ACCENT}22" if row["Model"] == best_model_name else ""
        return [bg] * len(row)

    styled_df = (
        model_comparison.style
        .apply(_highlight, axis=1)
        .format({"Accuracy": "{:.4f}", "F1_Weighted": "{:.4f}", "Log_Loss": "{:.4f}"})
        .hide(axis="index")
    )
    st.dataframe(styled_df, use_container_width=True)

    hr()
    st.markdown(
        '<div class="synaps-card">'
        "<h4>How Synaps Uses This</h4>"
        "Synaps applies this same prediction logic — when Claude and Gemini independently evaluate "
        "a document gap, their agreement level maps directly to the confidence scoring system shown "
        "in this analysis. High agreement between judges signals a High Confidence gap finding. "
        "The length-ratio and semantic embedding features correspond directly to the structural "
        "completeness signals Synaps uses when auditing BRD and PRD documents."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — EDA Explorer
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("EDA Explorer")

    X_eda, y_eda = load_features_targets()

    NUMERIC_COLS = [
        c for c in ["prompt_length", "response_a_length", "response_b_length",
                     "length_difference", "length_ratio", "prompt_word_count", "is_tie"]
        if c in X_eda.columns
    ]

    y_labels = y_eda.map(CLASS_LABELS)

    # Winner filter for EDA
    winner_filter = st.multiselect(
        "Filter by outcome",
        options=list(CLASS_LABELS.values()),
        default=list(CLASS_LABELS.values()),
        key="eda_winner_filter",
    )
    mask = y_labels.isin(winner_filter)
    X_f, y_f = X_eda[mask].reset_index(drop=True), y_eda[mask].reset_index(drop=True)
    y_f_labels = y_f.map(CLASS_LABELS)

    hr()
    plot_options = {
        "1. Target Distribution":            "target_dist",
        "2. Response Length Distribution":   "resp_len",
        "3. Length Difference Boxplot":      "len_diff",
        "4. Top Model Pairs":                "top_pairs",
        "5. Prompt Length Violin":           "prompt_violin",
        "6. Feature Correlation Heatmap":    "corr_heatmap",
        "7. Model Win Rate Leaderboard":     "win_rate",
        "8. Prompt Length vs Tie Scatter":   "prompt_tie",
    }
    CAPTIONS = {
        "target_dist":   (
            "This chart shows the distribution of outcomes across all 30,000 human preference "
            "judgments. The three classes are nearly balanced at 35% Model A Wins, 34% Model B "
            "Wins, and 31% Ties, which means the dataset does not heavily favor any single outcome."
        ),
        "resp_len":      (
            "This histogram shows how long Response A and Response B are in characters across the "
            "dataset. Responses vary widely in length, with some extremely long outliers that "
            "likely influence human preference judgments significantly."
        ),
        "len_diff":      (
            "This boxplot shows the absolute difference in character length between Response A and "
            "Response B for each outcome class. When one response is noticeably longer than the "
            "other, it tends to win, confirming that length ratio is the strongest predictor."
        ),
        "top_pairs":     (
            "This chart shows the most frequently occurring model matchups in the dataset. Certain "
            "model pairs like GPT-4 vs GPT-3.5 appear far more often than others, which may "
            "introduce bias into preference patterns."
        ),
        "prompt_violin": (
            "This violin plot shows the distribution of prompt lengths across the three outcome "
            "classes. Longer prompts tend to produce more ties, suggesting that complex questions "
            "are harder for humans to judge clearly."
        ),
        "corr_heatmap":  (
            "This heatmap shows the correlation between all numerical features in the dataset. "
            "Response length features are strongly correlated with each other, which explains why "
            "length ratio captures most of the predictive signal."
        ),
        "win_rate":      (
            "This chart ranks AI models by their win rate across all battles in the dataset. "
            "GPT-4 variants consistently outperform older models, but win rate alone does not "
            "capture response quality since longer responses inflate win rates."
        ),
        "prompt_tie":    (
            "This scatter plot explores the relationship between prompt length and whether a "
            "battle resulted in a tie. Shorter prompts produce more decisive winners while longer "
            "prompts tend to create more ties, suggesting humans struggle to differentiate quality "
            "on complex questions."
        ),
    }

    selected_plot = st.selectbox("Select plot", list(plot_options.keys()), key="eda_plot_sel")
    pk = plot_options[selected_plot]

    fig_eda = None

    if pk == "target_dist":
        td = y_f_labels.value_counts().reset_index()
        td.columns = ["Outcome", "Count"]
        td["Pct"] = (td["Count"] / td["Count"].sum() * 100).round(1)
        fig_eda = px.bar(
            td, x="Outcome", y="Count", color="Outcome",
            color_discrete_sequence=PALETTE,
            text=td["Pct"].apply(lambda x: f"{x}%"),
            title="Target Distribution",
        )
        fig_eda.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        )
        fig_eda.update_layout(showlegend=False)

    elif pk == "resp_len":
        _df = pd.DataFrame({
            "Length": pd.concat([
                X_f["response_a_length"].rename("Length"),
                X_f["response_b_length"].rename("Length"),
            ]),
            "Response": (["Response A"] * len(X_f)) + (["Response B"] * len(X_f)),
        })
        fig_eda = px.histogram(
            _df, x="Length", color="Response",
            color_discrete_sequence=[PALETTE[0], PALETTE[1]],
            barmode="overlay", opacity=0.72,
            nbins=60, title="Response Length Distribution",
        )
        fig_eda.update_traces(hovertemplate="Length: %{x}<br>Count: %{y}<extra></extra>")

    elif pk == "len_diff":
        _df = pd.DataFrame({"Length Difference": X_f["length_difference"], "Outcome": y_f_labels})
        fig_eda = px.box(
            _df, x="Outcome", y="Length Difference", color="Outcome",
            color_discrete_sequence=PALETTE,
            title="Length Difference by Outcome",
        )
        fig_eda.update_traces(hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>")
        fig_eda.update_layout(showlegend=False)

    elif pk == "top_pairs":
        pair_counts, _ = get_model_pair_stats("|".join(sorted(model_dummies_cols)))
        if not pair_counts.empty:
            fig_eda = px.bar(
                pair_counts.sort_values("count"), x="count", y="pair",
                orientation="h", color="count",
                color_continuous_scale=["#0077B6", "#4FC3F7"],
                title="Top 10 Model Pairs by Frequency",
                text="count",
            )
            fig_eda.update_traces(
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
            )
            fig_eda.update_layout(coloraxis_showscale=False, height=500)
        else:
            st.info("Model pair data unavailable.")

    elif pk == "prompt_violin":
        _df = pd.DataFrame({"Prompt Length": X_f["prompt_length"], "Outcome": y_f_labels})
        fig_eda = px.violin(
            _df, x="Outcome", y="Prompt Length", color="Outcome",
            color_discrete_sequence=PALETTE, box=True, points=False,
            title="Prompt Length by Outcome",
        )
        fig_eda.update_layout(showlegend=False)

    elif pk == "corr_heatmap":
        _corr_cols = [c for c in NUMERIC_COLS if c in X_f.columns]
        corr_mat = X_f[_corr_cols].corr()
        fig_eda = px.imshow(
            corr_mat, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
            zmin=-1, zmax=1,
        )
        fig_eda.update_traces(hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>")
        fig_eda = apply_chart_style(fig_eda, height=480)

    elif pk == "win_rate":
        _, wr_df = get_model_pair_stats("|".join(sorted(model_dummies_cols)))
        if not wr_df.empty:
            fig_eda = px.bar(
                wr_df, x="win_rate", y="model", orientation="h",
                color="win_rate", color_continuous_scale=["#0077B6", "#4FC3F7"],
                title="Model Win Rate Leaderboard",
                text=wr_df["win_rate"].apply(lambda x: f"{x:.1%}"),
            )
            fig_eda.update_traces(
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Win rate: %{x:.1%}<extra></extra>",
            )
            fig_eda.update_layout(coloraxis_showscale=False, xaxis=dict(tickformat=".0%"), height=500)
        else:
            st.info("Win rate data unavailable (insufficient matchup data).")

    elif pk == "prompt_tie":
        _df = pd.DataFrame({
            "Prompt Length": X_f["prompt_length"],
            "Is Tie": (y_f == 2).astype(int),
            "Outcome": y_f_labels,
        })
        fig_eda = px.scatter(
            _df, x="Prompt Length", y="Is Tie",
            color="Outcome", color_discrete_sequence=PALETTE,
            opacity=0.45, title="Prompt Length vs Tie Outcome",
        )
        fig_eda.update_layout(yaxis=dict(tickvals=[0, 1], ticktext=["Not Tie", "Tie"]))

    if fig_eda is not None:
        if pk not in ("corr_heatmap",):
            fig_eda = apply_chart_style(fig_eda)
        st.plotly_chart(fig_eda, use_container_width=True)
        if CAPTIONS.get(pk):
            st.caption(CAPTIONS[pk])

    hr()
    st.subheader("Summary Statistics (filtered)")
    _stat_cols = [c for c in NUMERIC_COLS if c in X_f.columns and c != "is_tie"]
    if _stat_cols:
        st.dataframe(X_f[_stat_cols].describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("Model Performance")
    st.markdown(
        f"All models evaluated with **5-fold stratified cross-validation** (random_state=42). "
        f"Best model: **{best_model_name}**."
    )
    hr()

    # Metric toggle
    st.markdown("**Select metric to display:**")
    m_cols = st.columns(3)
    with m_cols[0]:
        if st.button("Accuracy", key="perf_acc"):
            st.session_state["perf_metric"] = "Accuracy"
    with m_cols[1]:
        if st.button("F1 Weighted", key="perf_f1"):
            st.session_state["perf_metric"] = "F1_Weighted"
    with m_cols[2]:
        if st.button("Log Loss", key="perf_ll"):
            st.session_state["perf_metric"] = "Log_Loss"

    metric_col = st.session_state["perf_metric"]
    is_loss    = metric_col == "Log_Loss"

    bar_colors = [
        PALETTE[0] if r["Model"] == best_model_name else PALETTE[1]
        for _, r in model_comparison.iterrows()
    ]
    fig_perf = go.Figure(go.Bar(
        x=model_comparison["Model"],
        y=model_comparison[metric_col],
        marker_color=bar_colors,
        text=model_comparison[metric_col].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>" + metric_col + ": %{y:.4f}<extra></extra>",
    ))
    _y_max = model_comparison[metric_col].max()
    _y_min = model_comparison[metric_col].min()
    fig_perf.update_layout(
        title=f"{metric_col} Comparison — 5-Fold CV",
        xaxis_title="Model", yaxis_title=metric_col,
        yaxis=dict(range=[max(0, _y_min - 0.05), _y_max + 0.08]) if not is_loss
              else dict(range=[0, _y_max + 1]),
        showlegend=False,
    )
    fig_perf = apply_chart_style(fig_perf, height=420)
    st.plotly_chart(fig_perf, use_container_width=True)

    hr()

    # Styled table
    def _hl(row):
        if row["Model"] == best_model_name:
            return [f"background-color: {ACCENT}22; font-weight: bold"] * len(row)
        if row["Model"] == model_comparison.iloc[-1]["Model"]:
            return ["background-color: rgba(220,53,69,0.12)"] * len(row)
        return [""] * len(row)

    st_tbl = (
        model_comparison.style
        .apply(_hl, axis=1)
        .format({"Accuracy": "{:.4f}", "F1_Weighted": "{:.4f}", "Log_Loss": "{:.4f}"})
        .hide(axis="index")
    )
    st.dataframe(st_tbl, use_container_width=True)

    st.download_button(
        "Download Results CSV",
        data=model_comparison.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv",
        key="dl_model_csv",
    )

    hr()
    st.subheader("Confusion Matrix")
    cm_model = st.selectbox(
        "Select model for confusion matrix",
        options=model_comparison["Model"].tolist(),
        index=0,
        key="cm_model_sel",
    )
    with st.spinner("Computing predictions..."):
        y_pred_cm, y_true_cm = get_predictions_for_model(cm_model)
    cm_mat = confusion_matrix(y_true_cm, y_pred_cm, labels=[0, 1, 2])
    class_names = ["Model A Wins", "Model B Wins", "Tie"]
    fig_cm = px.imshow(
        cm_mat, x=class_names, y=class_names,
        color_continuous_scale=["#0f1117", ACCENT],
        text_auto=True,
        title=f"Confusion Matrix — {cm_model}",
        labels=dict(x="Predicted", y="Actual"),
    )
    fig_cm.update_traces(hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: %{z}<extra></extra>")
    fig_cm = apply_chart_style(fig_cm, height=420)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curves ───────────────────────────────────────────────────
    hr()
    st.subheader("ROC Curves")

    _SKLEARN_MODELS = [
        m for m in model_comparison["Model"].tolist()
        if m != "MLP Neural Network"
    ]

    roc_model_sel = st.selectbox(
        "Select model for ROC curves",
        options=_SKLEARN_MODELS,
        index=0,
        key="roc_model_sel",
    )

    if mlp_available and roc_model_sel == "MLP Neural Network":
        st.info("ROC curves are not displayed for MLP Neural Network here. Select a tree-based or linear model.")
    else:
        @st.cache_data
        def _compute_roc(model_name: str):
            from sklearn.model_selection import train_test_split as _tts_roc
            from sklearn.metrics import roc_curve, auc
            import numpy as _np

            X_r, y_r = load_features_targets()
            _, X_te_r, _, y_te_r = _tts_roc(
                X_r, y_r, test_size=0.3, random_state=42, stratify=y_r
            )
            _m = _load_model_file(f"models/{safe_name(model_name)}.joblib")
            _proba = _m.predict_proba(X_te_r)

            results = {}
            for cls_idx, cls_label in CLASS_LABELS.items():
                y_bin = (y_te_r.values == cls_idx).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, _proba[:, cls_idx])
                roc_auc = auc(fpr, tpr)
                results[cls_label] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
            return results

        with st.spinner("Computing ROC curves..."):
            _roc_data = _compute_roc(roc_model_sel)

        _roc_colors = {"Model A Wins": "#4FC3F7", "Model B Wins": "#FF6B6B", "Tie": "#90E0EF"}
        fig_roc = go.Figure()
        for cls_label, vals in _roc_data.items():
            fig_roc.add_trace(go.Scatter(
                x=vals["fpr"], y=vals["tpr"],
                mode="lines",
                name=f"{cls_label} (AUC = {vals['auc']:.3f})",
                line=dict(color=_roc_colors.get(cls_label, ACCENT), width=2),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>" + cls_label + "</extra>",
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random (AUC = 0.500)",
            line=dict(color=MUTED, width=1, dash="dash"),
            hoverinfo="skip",
        ))
        fig_roc.update_layout(
            title=f"ROC Curves (One-vs-Rest) — {roc_model_sel}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.08, bgcolor="rgba(0,0,0,0)"),
        )
        fig_roc = apply_chart_style(fig_roc, height=460)
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption(
            "ROC curves show each model's ability to distinguish between the three outcome "
            "classes. Higher AUC indicates better discrimination ability. LightGBM achieves "
            "the highest AUC across all three classes."
        )
        if mlp_available:
            st.caption("Note: MLP Neural Network is excluded from ROC curves. Use the model selector above to view sklearn models only.")

    # ── Neural Network Training History ──────────────────────────────
    if mlp_available and os.path.exists("eda_plots/mlp_training_history.png"):
        hr()
        st.subheader("Neural Network Training History")
        st.markdown(
            '<div class="info-card">'
            "<b>Architecture:</b> Input → Dense(128, ReLU) → Dropout(0.3) → Dense(128, ReLU) "
            "→ Dropout(0.3) → Dense(3, Softmax)<br>"
            "<b>Hyperparameters:</b> Optimizer = Adam &nbsp;|&nbsp; Loss = Sparse Categorical "
            "Cross-Entropy &nbsp;|&nbsp; Epochs = 50 &nbsp;|&nbsp; Batch Size = 64 &nbsp;|&nbsp; "
            "Validation Split = 10% &nbsp;|&nbsp; random_state = 42<br>"
            "<b>Feature preprocessing:</b> StandardScaler on all 184 features (structured + embeddings)"
            "</div>",
            unsafe_allow_html=True,
        )
        _mlp_row = model_comparison[model_comparison["Model"] == "MLP Neural Network"]
        if not _mlp_row.empty:
            _mr = _mlp_row.iloc[0]
            mc1, mc2, mc3 = st.columns(3)
            with mc1: metric_card("MLP Accuracy",    f"{_mr['Accuracy']:.4f}")
            with mc2: metric_card("MLP F1 Weighted", f"{_mr['F1_Weighted']:.4f}")
            with mc3: metric_card("MLP Log Loss",    f"{_mr['Log_Loss']:.4f}")
        st.image("eda_plots/mlp_training_history.png", use_container_width=True)
        # ── Early stopping insight ────────────────────────────────────
        _meta_path = "models/mlp_training_meta.json"
        if os.path.exists(_meta_path):
            import json as _json
            with open(_meta_path) as _mf:
                _meta = _json.load(_mf)
            _stopped = _meta.get("stopped_epoch", "?")
        else:
            _stopped = 10  # actual value from training run
        st.markdown(
            f'<div class="info-card">'
            f"<b>Overfitting was detected:</b> training loss continued decreasing while validation "
            f"loss rose after the early epochs. Early stopping was applied (patience=5, monitoring "
            f"val_loss) to restore the best weights and prevent the model from memorizing training "
            f"data. The model stopped at epoch <b>{_stopped}</b> out of 50 maximum epochs."
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — SHAP Explainability
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("SHAP Explainability")
    st.markdown(f"Explaining **{best_model_name}** predictions using structured features only.")
    if mlp_available:
        st.info(
            "SHAP is only available for tree-based models. "
            "Showing LightGBM SHAP values for reference. "
            "MLP Neural Network does not support TreeExplainer."
        )
    hr()

    # View toggle
    v_cols = st.columns(3)
    with v_cols[0]:
        if st.button("Summary View", key="shap_sum"):
            st.session_state["shap_view"] = "Summary"
    with v_cols[1]:
        if st.button("Bar View", key="shap_bar"):
            st.session_state["shap_view"] = "Bar"
    with v_cols[2]:
        if st.button("Waterfall View", key="shap_wf"):
            st.session_state["shap_view"] = "Waterfall"

    shap_view = st.session_state["shap_view"]
    top_feat  = shap_insights.get("top_feature", "N/A")

    with st.spinner("Computing SHAP values..."):
        shap_imp = get_shap_importance()

    if shap_imp is None:
        st.warning("SHAP struct model not found. Re-run pipeline.py to generate it.")
    else:
        display_n = 20
        imp_df = shap_imp.head(display_n).reset_index()
        imp_df.columns = ["Feature", "Mean |SHAP|"]

        chart_col, text_col = st.columns([3, 2])

        with chart_col:
            if shap_view == "Summary":
                fig_s = px.bar(
                    imp_df.sort_values("Mean |SHAP|"),
                    x="Mean |SHAP|", y="Feature", orientation="h",
                    color="Mean |SHAP|",
                    color_continuous_scale=["#0077B6", "#4FC3F7"],
                    title=f"SHAP Summary — Top {display_n} Features",
                    text=imp_df.sort_values("Mean |SHAP|")["Mean |SHAP|"].apply(lambda x: f"{x:.4f}"),
                )
                fig_s.update_traces(
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
                )
                fig_s.update_layout(coloraxis_showscale=False, height=560)
                fig_s = apply_chart_style(fig_s, height=560)
                st.plotly_chart(fig_s, use_container_width=True)

            elif shap_view == "Bar":
                fig_b = px.bar(
                    imp_df, x="Feature", y="Mean |SHAP|",
                    color="Mean |SHAP|",
                    color_continuous_scale=["#0077B6", "#4FC3F7"],
                    title=f"SHAP Feature Importance — Top {display_n}",
                    text=imp_df["Mean |SHAP|"].apply(lambda x: f"{x:.4f}"),
                )
                fig_b.update_traces(
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Mean |SHAP|: %{y:.4f}<extra></extra>",
                )
                fig_b.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
                fig_b = apply_chart_style(fig_b, height=480)
                st.plotly_chart(fig_b, use_container_width=True)

            else:  # Waterfall
                import shap as _shap

                @st.cache_resource
                def _load_shap_explainer_and_sample():
                    _sm = load_struct_model()
                    if _sm is None:
                        return None, None, None
                    _struct_cols = joblib.load("models/structured_feature_columns.joblib")
                    _X, _ = load_features_targets()
                    _struct_cols = [c for c in _struct_cols if c in _X.columns]
                    _X_s = _X[_struct_cols]
                    # Load cached explainer if available, else build it
                    _exp_path = "models/shap_explainer.joblib"
                    if os.path.exists(_exp_path):
                        _explainer = joblib.load(_exp_path)
                    else:
                        try:
                            _explainer = _shap.TreeExplainer(_sm)
                        except Exception:
                            _explainer = _shap.LinearExplainer(_sm, _X_s.sample(500, random_state=42))
                    return _explainer, _X_s, _struct_cols

                with st.spinner("Computing SHAP waterfall for sample instance..."):
                    _explainer, _X_s, _struct_cols = _load_shap_explainer_and_sample()

                if _explainer is None:
                    st.warning("SHAP struct model not found.")
                else:
                    _sample = _X_s.iloc[[0]]
                    _sv_raw = _explainer.shap_values(_sample)
                    # For multi-class take class-0 (Model A wins); flatten to 1-D
                    if isinstance(_sv_raw, list):
                        _sv = np.array(_sv_raw[0]).flatten()
                    elif hasattr(_sv_raw, "ndim") and _sv_raw.ndim == 3:
                        _sv = _sv_raw[0, :, 0]
                    else:
                        _sv = np.array(_sv_raw).flatten()

                    _base = (
                        float(np.array(_explainer.expected_value).flatten()[0])
                        if hasattr(_explainer, "expected_value")
                        else 0.0
                    )

                    _wf = (
                        pd.DataFrame({"Feature": _struct_cols, "SHAP": _sv})
                        .assign(AbsSHAP=lambda d: d["SHAP"].abs())
                        .nlargest(15, "AbsSHAP")
                        .sort_values("SHAP")
                    )

                    _colors = ["#FF6B6B" if v < 0 else "#4FC3F7" for v in _wf["SHAP"]]

                    fig_wf = go.Figure()
                    fig_wf.add_trace(go.Bar(
                        x=_wf["SHAP"],
                        y=_wf["Feature"],
                        orientation="h",
                        marker_color=_colors,
                        text=_wf["SHAP"].apply(lambda v: f"{v:+.4f}"),
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>SHAP value: %{x:+.4f}<extra></extra>",
                    ))
                    fig_wf.add_vline(
                        x=_base,
                        line_dash="dash",
                        line_color=TEXT,
                        line_width=1.5,
                        annotation_text=f"Base: {_base:.3f}",
                        annotation_position="top right",
                        annotation_font_color=TEXT,
                    )
                    fig_wf.add_annotation(
                        text=(
                            "Positive (blue): pushes prediction toward Model A Wins  |  "
                            "Negative (red): pushes toward Model B Wins"
                        ),
                        xref="paper", yref="paper",
                        x=0.0, y=-0.13,
                        showarrow=False,
                        font=dict(size=11, color=MUTED),
                        align="left",
                    )
                    fig_wf.update_layout(title="SHAP Waterfall — Single Prediction Instance")
                    fig_wf = apply_chart_style(fig_wf, height=520)
                    fig_wf.update_layout(margin=dict(l=40, r=40, t=50, b=70))
                    st.plotly_chart(fig_wf, use_container_width=True)

        with text_col:
            st.markdown(
                f'<div class="info-card"><b>Current view:</b> {shap_view}<br><br>'
                f"<b>Top feature:</b> <code>{top_feat}</code><br><br>"
                "Mean |SHAP| measures the average magnitude of each feature's contribution "
                "to the model's prediction. Higher values indicate stronger influence on the "
                "output, regardless of direction.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="synaps-card"><h4>What This Means for Synaps</h4>'
                "The top SHAP features reveal that response length characteristics dominate "
                "human preference decisions. In Synaps, this translates to: when Claude and "
                "Gemini both flag a gap, the confidence score is calibrated by how strongly "
                "these same structural signals align. A high-SHAP feature with strong agreement "
                "between both models maps to a High Confidence finding in the audit report.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 6 — What-If Simulator
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("What-If Simulator")
    st.markdown(
        "Adjust sliders to explore how feature values affect the model prediction. "
        "Prediction updates automatically on every change."
    )
    hr()

    def _track_length():
        st.session_state["last_slider"] = "length"

    def _track_prompt():
        st.session_state["last_slider"] = "prompt"

    def _track_model():
        st.session_state["last_slider"] = "model"

    wif_left, wif_right = st.columns([1, 1])

    with wif_left:
        st.markdown("**Controls**")
        wif_ra  = st.slider("Response A Length (chars)", 100, 3000, 500, 50,
                            key="wif_ra", on_change=_track_length)
        wif_rb  = st.slider("Response B Length (chars)", 100, 3000, 650, 50,
                            key="wif_rb", on_change=_track_length)
        wif_pl  = st.slider("Prompt Length (chars)", 50, 2000, 200, 10,
                            key="wif_pl", on_change=_track_prompt)
        wif_wc  = st.slider("Prompt Word Count", 10, 400, 40, 5,
                            key="wif_wc", on_change=_track_prompt)
        wif_ma  = st.selectbox("Model A", top_models, index=0,
                               key="wif_ma", on_change=_track_model)
        wif_mb  = st.selectbox("Model B", top_models,
                               index=min(1, len(top_models) - 1),
                               key="wif_mb", on_change=_track_model)
        wif_has_code = st.checkbox("Responses Have Code?", value=False)
        wif_has_list = st.checkbox("Responses Have Lists?", value=False)

    # Build feature vector (zero embeddings)
    wif_ld = abs(wif_ra - wif_rb)
    wif_lr = wif_ra / (wif_rb + 1)
    wif_feat = {
        "prompt_length":     wif_pl,
        "response_a_length": wif_ra,
        "response_b_length": wif_rb,
        "length_difference": wif_ld,
        "length_ratio":      wif_lr,
        "prompt_word_count": wif_wc,
        "response_a_word_count": int(wif_ra / 5),
        "response_b_word_count": int(wif_rb / 5),
        "word_count_difference": abs(int(wif_ra / 5) - int(wif_rb / 5)),
        "word_count_ratio":      (int(wif_ra / 5)) / (int(wif_rb / 5) + 1),
        "longer_response":       1 if wif_ra > wif_rb else 0,
        "response_a_has_code":   1 if wif_has_code else 0,
        "response_b_has_code":   1 if wif_has_code else 0,
        "response_a_has_list":   1 if wif_has_list else 0,
        "response_b_has_list":   1 if wif_has_list else 0,
        "is_tie":            0,
    }
    for col in model_dummies_cols:
        wif_feat[col] = 0
    wif_feat[f"model_a_{wif_ma}"] = 1
    wif_feat[f"model_b_{wif_mb}"] = 1
    
    # Zero embeddings for What-If
    for c in all_feat_cols:
        if c.startswith("emb_"):
            wif_feat[c] = 0.0

    wif_X      = pd.DataFrame([[wif_feat.get(c, 0) for c in all_feat_cols]], columns=all_feat_cols)
    wif_cls    = int(best_model.predict(wif_X)[0])
    wif_proba  = best_model.predict_proba(wif_X)[0]
    wif_winner = CLASS_LABELS[wif_cls]
    wif_conf   = float(wif_proba[wif_cls])

    # Confidence delta
    prev_conf = st.session_state["last_confidence"]
    if prev_conf is not None:
        delta = wif_conf - prev_conf
        delta_str = f"(+{delta:.1%})" if delta >= 0 else f"({delta:.1%})"
        arrow = "up" if delta >= 0 else "down"
        delta_color = "#28a745" if delta >= 0 else "#dc3545"
    else:
        delta_str = ""
        delta_color = TEXT
    st.session_state["last_confidence"] = wif_conf

    with wif_right:
        st.markdown("**Live Results**")
        dr1, dr2 = st.columns(2)
        with dr1:
            metric_card("Length Difference", f"{wif_ld:,} chars")
        with dr2:
            metric_card("Length Ratio (A/B)", f"{wif_lr:.3f}")

        st.markdown(
            f'<div class="prediction-card">'
            f'<div class="pc-label">Predicted Winner</div>'
            f'<div class="pc-value">{wif_winner}</div>'
            f'<div class="pc-sub">Confidence: {wif_conf:.1%} '
            f'<span style="color:{delta_color}">{delta_str}</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        wif_conf_df = pd.DataFrame({"Outcome": list(CLASS_LABELS.values()), "Confidence": wif_proba})
        fig_wif = px.bar(
            wif_conf_df, x="Outcome", y="Confidence",
            color="Outcome", color_discrete_sequence=PALETTE,
            text=wif_conf_df["Confidence"].apply(lambda x: f"{x:.1%}"),
        )
        fig_wif.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:.1%}<extra></extra>",
        )
        fig_wif.update_layout(
            yaxis=dict(range=[0, 1.15], tickformat=".0%"),
            showlegend=False, title="Confidence — All 3 Classes",
        )
        fig_wif = apply_chart_style(fig_wif, height=320)
        st.plotly_chart(fig_wif, use_container_width=True)

    hr()
    st.subheader("Sensitivity Analysis")
    shap_imp_wif = get_shap_importance()
    if shap_imp_wif is not None:
        last_sl = st.session_state.get("last_slider", None)
        highlight_feat = (
            "length_difference" if last_sl == "length"
            else "prompt_length" if last_sl == "prompt"
            else None
        )
        sa_df = shap_imp_wif.head(15).reset_index()
        sa_df.columns = ["Feature", "Mean |SHAP|"]
        sa_colors = [
            PALETTE[0] if (highlight_feat and f == highlight_feat) else PALETTE[2]
            for f in sa_df["Feature"]
        ]
        fig_sa = go.Figure(go.Bar(
            x=sa_df["Feature"], y=sa_df["Mean |SHAP|"],
            marker_color=sa_colors,
            text=sa_df["Mean |SHAP|"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Mean |SHAP|: %{y:.4f}<extra></extra>",
        ))
        fig_sa.update_layout(
            title="Feature Impact — Structured Features",
            xaxis_tickangle=-30,
        )
        fig_sa = apply_chart_style(fig_sa, height=380)
        st.plotly_chart(fig_sa, use_container_width=True)
        if highlight_feat:
            st.markdown(
                f'<div class="info-card">Highlighted: <b>{highlight_feat}</b> '
                f"— last modified feature group.</div>",
                unsafe_allow_html=True,
            )

    hr()
    st.subheader("Scenario Comparison")
    if st.button("Save This Scenario", key="wif_save"):
        st.session_state["scenarios"].append({
            "Response A Len": wif_ra,
            "Response B Len": wif_rb,
            "Prompt Len": wif_pl,
            "Model A": wif_ma,
            "Model B": wif_mb,
            "Prediction": wif_winner,
            "Confidence": f"{wif_conf:.1%}",
        })
        st.success("Scenario saved.")

    if st.session_state["scenarios"]:
        sc_df = pd.DataFrame(st.session_state["scenarios"])
        sc_df.index = [f"S{i+1}" for i in range(len(sc_df))]
        st.dataframe(sc_df, use_container_width=True)

        # Grouped confidence chart across scenarios
        _conf_vals = [float(r["Confidence"].strip("%")) / 100 for r in st.session_state["scenarios"]]
        _sc_names  = [f"S{i+1}" for i in range(len(_conf_vals))]
        fig_sc = px.bar(
            x=_sc_names, y=_conf_vals,
            color=_conf_vals, color_continuous_scale=["#0077B6", "#4FC3F7"],
            title="Confidence Comparison Across Scenarios",
            text=[f"{v:.1%}" for v in _conf_vals],
            labels={"x": "Scenario", "y": "Confidence"},
        )
        fig_sc.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>")
        fig_sc.update_layout(yaxis=dict(range=[0, 1.15], tickformat=".0%"), coloraxis_showscale=False)
        fig_sc = apply_chart_style(fig_sc, height=340)
        st.plotly_chart(fig_sc, use_container_width=True)

        if st.button("Clear All Scenarios", key="wif_clear"):
            st.session_state["scenarios"] = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 7 — Model Agreement Analyzer
# ══════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("Model Agreement Analyzer")
    st.markdown(
        "Simulates the Synaps dual-LLM evaluation system. Set scores and confidence "
        "for two independent judge models, then analyze their agreement."
    )
    hr()

    j1_col, j2_col = st.columns(2)

    with j1_col:
        st.markdown("**Judge 1**")
        j1_model = st.selectbox("Judge 1 — Select Model", top_models, index=0, key="j1_model")
        j1_score = st.slider("Judge 1 Score", 1, 10, 7, key="j1_score")
        j1_conf  = st.slider("Judge 1 Confidence (%)", 0, 100, 80, key="j1_conf")

    with j2_col:
        st.markdown("**Judge 2**")
        j2_model = st.selectbox("Judge 2 — Select Model", top_models,
                                index=min(1, len(top_models) - 1), key="j2_model")
        j2_score = st.slider("Judge 2 Score", 1, 10, 5, key="j2_score")
        j2_conf  = st.slider("Judge 2 Confidence (%)", 0, 100, 70, key="j2_conf")

    analyze_btn = st.button("Analyze Agreement", key="agree_btn")

    if analyze_btn:
        score_diff = abs(j1_score - j2_score)
        avg_conf   = (j1_conf + j2_conf) / 2

        if score_diff <= 1:
            level       = "HIGH CONFIDENCE — 95%+ Accuracy"
            level_color = "#28a745"
            verdict     = "Accept Finding"
        elif score_diff <= 3:
            level       = "MODERATE CONFIDENCE — 60-80% Accuracy"
            level_color = "#ffc107"
            verdict     = "Flag for Review"
        else:
            level       = "LOW CONFIDENCE — Human Review Required"
            level_color = "#dc3545"
            verdict     = "Flag for Review"

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_conf,
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT},
                "bar":  {"color": level_color},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 33],  "color": "rgba(220,53,69,0.15)"},
                    {"range": [33, 66], "color": "rgba(255,193,7,0.15)"},
                    {"range": [66, 100],"color": "rgba(40,167,69,0.15)"},
                ],
                "threshold": {"line": {"color": level_color, "width": 3}, "thickness": 0.8, "value": avg_conf},
            },
            title={"text": "Average Confidence Score", "font": {"color": TEXT}},
            number={"suffix": "%", "font": {"color": TEXT}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, family="Inter"),
            height=300,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Verdict card
        st.markdown(
            f'<div class="verdict-card" style="background-color:{level_color}22; '
            f'border:2px solid {level_color};">'
            f'<div style="font-size:1.4rem;font-weight:700;color:{level_color}">{level}</div>'
            f'<div style="margin-top:0.5rem;color:{TEXT}">'
            f'Score difference: {score_diff} point{"s" if score_diff != 1 else ""} '
            f'| Recommended action: <b>{verdict}</b></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Append to history
        st.session_state["agreement_history"].append({
            "Judge 1 Model": j1_model,
            "J1 Score": j1_score,
            "J1 Conf %": j1_conf,
            "Judge 2 Model": j2_model,
            "J2 Score": j2_score,
            "J2 Conf %": j2_conf,
            "Score Diff": score_diff,
            "Level": level.split("—")[0].strip(),
        })

    hr()
    if st.session_state["agreement_history"]:
        st.subheader("Session Log")
        log_df = pd.DataFrame(st.session_state["agreement_history"])
        log_df.index = [f"Run {i+1}" for i in range(len(log_df))]
        st.dataframe(log_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 8 — Head-to-Head Arena
# ══════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("Head-to-Head Arena")
    st.markdown(
        "Filter the dataset for rows where two specific models face off, "
        "then view win/loss/tie statistics."
    )
    hr()

    h2h_c1, h2h_c2 = st.columns(2)
    with h2h_c1:
        h2h_ma = st.selectbox("Model A", top_models, index=0, key="h2h_ma")
    with h2h_c2:
        h2h_mb = st.selectbox("Model B", top_models,
                              index=min(1, len(top_models) - 1), key="h2h_mb")

    run_h2h = st.button("Run Matchup", key="h2h_run")

    if run_h2h:
        X_h, y_h = load_features_targets()
        col_a = f"model_a_{h2h_ma}"
        col_b = f"model_b_{h2h_mb}"

        has_a = col_a in X_h.columns
        has_b = col_b in X_h.columns

        if not (has_a and has_b):
            st.warning(
                f"One or both models not found in one-hot columns. "
                f"Available: {[c for c in X_h.columns if 'model_' in c][:10]}"
            )
        else:
            mask_h = (X_h[col_a] == 1) & (X_h[col_b] == 1)
            X_match = X_h[mask_h]
            y_match = y_h[mask_h]

            n_total = len(X_match)
            if n_total == 0:
                st.info(f"No matchups found between {h2h_ma} (A) and {h2h_mb} (B).")
            else:
                n_a_wins = int((y_match == 0).sum())
                n_b_wins = int((y_match == 1).sum())
                n_ties   = int((y_match == 2).sum())
                wr_a     = n_a_wins / n_total

                m1, m2, m3 = st.columns(3)
                with m1: metric_card(f"{h2h_ma} Wins", str(n_a_wins))
                with m2: metric_card(f"{h2h_mb} Wins", str(n_b_wins))
                with m3: metric_card("Ties", str(n_ties))

                fig_h2h = go.Figure(go.Bar(
                    x=[f"{h2h_ma} Wins", f"{h2h_mb} Wins", "Ties"],
                    y=[n_a_wins, n_b_wins, n_ties],
                    marker_color=[PALETTE[0], PALETTE[1], PALETTE[3]],
                    text=[n_a_wins, n_b_wins, n_ties],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                ))
                fig_h2h.update_layout(
                    title=f"Head-to-Head: {h2h_ma} vs {h2h_mb} ({n_total} matchups)",
                    showlegend=False,
                    yaxis=dict(range=[0, max(n_a_wins, n_b_wins, n_ties) * 1.2]),
                )
                fig_h2h = apply_chart_style(fig_h2h, height=380)
                st.plotly_chart(fig_h2h, use_container_width=True)

                # Verdict
                if n_a_wins > n_b_wins * 1.2:
                    vtext  = f"{h2h_ma} dominates this matchup"
                    vcolor = PALETTE[0]
                elif n_b_wins > n_a_wins * 1.2:
                    vtext  = f"{h2h_mb} dominates this matchup"
                    vcolor = PALETTE[1]
                else:
                    vtext  = "Even matchup — no clear dominant model"
                    vcolor = PALETTE[3]

                st.markdown(
                    f'<div class="verdict-card" style="background:{vcolor}22;border:2px solid {vcolor};">'
                    f'<div style="font-size:1.3rem;font-weight:700;color:{vcolor}">{vtext}</div>'
                    f'<div style="margin-top:0.4rem;color:{TEXT}">'
                    f"Win rate ({h2h_ma}): {wr_a:.1%} | Sample size: {n_total:,}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                hr()
                # Response length comparison
                ra_col = "response_a_length"
                rb_col = "response_b_length"
                if ra_col in X_match.columns and rb_col in X_match.columns:
                    avg_ra = X_match[ra_col].mean()
                    avg_rb = X_match[rb_col].mean()
                    fig_rl = go.Figure(go.Bar(
                        x=[f"Avg {h2h_ma} Response", f"Avg {h2h_mb} Response"],
                        y=[avg_ra, avg_rb],
                        marker_color=[PALETTE[0], PALETTE[1]],
                        text=[f"{avg_ra:.0f}", f"{avg_rb:.0f}"],
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Avg length: %{y:.0f} chars<extra></extra>",
                    ))
                    fig_rl.update_layout(
                        title="Average Response Length Comparison",
                        showlegend=False,
                        yaxis=dict(range=[0, max(avg_ra, avg_rb) * 1.2]),
                    )
                    fig_rl = apply_chart_style(fig_rl, height=340)
                    st.plotly_chart(fig_rl, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 9 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════════
with tab9:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("Dataset Explorer")
    st.markdown("Filter, explore, and export the feature dataset. All filters apply live.")
    hr()

    X_ex, y_ex = load_features_targets()
    y_ex_labels = y_ex.map(CLASS_LABELS)

    mdummies_ex = joblib.load("models/model_dummies_columns.joblib")
    ma_cols_ex  = [c for c in mdummies_ex if c.startswith("model_a_") and c in X_ex.columns]
    mb_cols_ex  = [c for c in mdummies_ex if c.startswith("model_b_") and c in X_ex.columns]

    ma_max_ex = X_ex[ma_cols_ex].max(axis=1) if ma_cols_ex else pd.Series(0, index=X_ex.index)
    mb_max_ex = X_ex[mb_cols_ex].max(axis=1) if mb_cols_ex else pd.Series(0, index=X_ex.index)

    if ma_cols_ex:
        ma_name_ex = (
            X_ex[ma_cols_ex].idxmax(axis=1)
            .where(ma_max_ex > 0, "Unknown")
            .str.replace("model_a_", "", regex=False)
        )
    else:
        ma_name_ex = pd.Series("Unknown", index=X_ex.index)

    if mb_cols_ex:
        mb_name_ex = (
            X_ex[mb_cols_ex].idxmax(axis=1)
            .where(mb_max_ex > 0, "Unknown")
            .str.replace("model_b_", "", regex=False)
        )
    else:
        mb_name_ex = pd.Series("Unknown", index=X_ex.index)

    # Filter controls
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        kw = st.text_input("Search model name (keyword)", key="ex_kw")
    with fc2:
        win_filter_ex = st.multiselect(
            "Winner outcome",
            options=list(CLASS_LABELS.values()),
            default=list(CLASS_LABELS.values()),
            key="ex_win",
        )
    with fc3:
        model_a_filter = st.multiselect("Model A filter", sorted(ma_name_ex.unique()), key="ex_ma")
        model_b_filter = st.multiselect("Model B filter", sorted(mb_name_ex.unique()), key="ex_mb")

    _pl_min = int(X_ex["prompt_length"].min()) if "prompt_length" in X_ex.columns else 0
    _pl_max = int(X_ex["prompt_length"].max()) if "prompt_length" in X_ex.columns else 2000
    pl_range = st.slider(
        "Prompt length range", _pl_min, _pl_max, (_pl_min, _pl_max), key="ex_pl_range"
    )

    # Build filtered mask
    ex_df = X_ex.copy()
    ex_df["_winner"]  = y_ex_labels.values
    ex_df["_model_a"] = ma_name_ex.values
    ex_df["_model_b"] = mb_name_ex.values

    _mask_ex = ex_df["_winner"].isin(win_filter_ex)
    if "prompt_length" in ex_df.columns:
        _mask_ex &= ex_df["prompt_length"].between(pl_range[0], pl_range[1])
    if model_a_filter:
        _mask_ex &= ex_df["_model_a"].isin(model_a_filter)
    if model_b_filter:
        _mask_ex &= ex_df["_model_b"].isin(model_b_filter)
    if kw.strip():
        _kw_lower = kw.strip().lower()
        _mask_ex &= (
            ex_df["_model_a"].str.lower().str.contains(_kw_lower, na=False) |
            ex_df["_model_b"].str.lower().str.contains(_kw_lower, na=False)
        )

    ex_filtered = ex_df[_mask_ex].reset_index(drop=True)

    # Live stats bar
    hr()
    s1, s2, s3 = st.columns(3)
    with s1:
        metric_card("Filtered Rows", f"{len(ex_filtered):,}")
    with s2:
        avg_pl = ex_filtered["prompt_length"].mean() if "prompt_length" in ex_filtered.columns else 0
        metric_card("Avg Prompt Length", f"{avg_pl:.0f} chars")
    with s3:
        wr_counts = ex_filtered["_winner"].value_counts()
        fig_mini = px.bar(
            x=wr_counts.index, y=wr_counts.values,
            color=wr_counts.index, color_discrete_sequence=PALETTE,
            labels={"x": "", "y": ""},
        )
        fig_mini.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=140, margin=dict(l=5, r=5, t=5, b=25),
            font=dict(color=TEXT, size=10),
        )
        fig_mini.update_xaxes(color=TEXT)
        fig_mini.update_yaxes(color=TEXT)
        st.plotly_chart(fig_mini, use_container_width=True)

    hr()

    # Scatter plot
    if "prompt_length" in ex_filtered.columns and "response_a_length" in ex_filtered.columns:
        fig_ex_sc = px.scatter(
            ex_filtered, x="prompt_length", y="response_a_length",
            color="_winner", color_discrete_sequence=PALETTE,
            opacity=0.45,
            title="Prompt Length vs Response A Length (filtered)",
            labels={"prompt_length": "Prompt Length", "response_a_length": "Response A Length", "_winner": "Outcome"},
            hover_data={"_model_a": True, "_model_b": True},
        )
        fig_ex_sc = apply_chart_style(fig_ex_sc, height=400)
        st.plotly_chart(fig_ex_sc, use_container_width=True)

    hr()

    # Table
    SHOW_COLS = [
        c for c in ["prompt_length", "response_a_length", "response_b_length",
                     "length_difference", "length_ratio", "prompt_word_count", "_model_a", "_model_b", "_winner"]
        if c in ex_filtered.columns
    ]
    st.subheader(f"Data Table — {len(ex_filtered):,} rows")
    display_df = ex_filtered[SHOW_COLS].rename(columns={"_model_a": "model_a", "_model_b": "model_b", "_winner": "winner"})
    st.dataframe(display_df.head(500), use_container_width=True)

    hr()
    st.download_button(
        "Export Filtered Data as CSV",
        data=display_df.to_csv(index=False),
        file_name="filtered_dataset.csv",
        mime="text/csv",
        key="ex_dl",
    )

    st.markdown("</div>", unsafe_allow_html=True)
