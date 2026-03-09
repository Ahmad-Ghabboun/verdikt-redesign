import re
import os

with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update _SS state and Theme Variables
text = text.replace(
'''_SS = {
    "dark_mode": True,''',
'''_SS = {
    "dark_mode": False,
    "current_nav": "Executive Summary",'''
)

text = text.replace(
'''dark = st.session_state["dark_mode"]
BG     = "#1C1C1C"  if dark else "#F7F9F9"
CARD   = "#252525"  if dark else "#FFFFFF"
TEXT   = "#FFFFFF"  if dark else "#1A202C"
MUTED  = "#8E8E8E"  if dark else "#A0AEC0"
BORDER = "#2A2A2A"  if dark else "#E2E8F0"
ACCENT = "#CFF242"  if dark else "#35B095"
ACCENT2= "#9575CD"  if dark else "#4498D2"
PALETTE = ["#35B095", "#4498D2", "#1D72CE", "#2B825B", "#F2B21A"]
TAB_INACTIVE = "#1A1A1A" if dark else "#EDF7F4"
GRID   = "rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.06)"''',
'''dark = False
BG     = "#F8F9FA"
CARD   = "#FFFFFF"
TEXT   = "#333333"
MUTED  = "#8B95A5"
BORDER = "#E5E9F2"
ACCENT = "#2CC9B6"
ACCENT2= "#80CBC4"
PALETTE = ["#00796B", "#4DD0E1", "#FFF176", "#81C784", "#64B5F6"]
TAB_INACTIVE = "#FFFFFF"
GRID   = "rgba(0,0,0,0.05)"'''
)

# 2. Top header row toggle removal
text = text.replace(
'''# ─────────────────────────────────────────────────────────────────────
# Top header row: toggle + gradient banner
# ─────────────────────────────────────────────────────────────────────
_, _tcol = st.columns([8, 1])
with _tcol:
    if st.button("Light Mode" if dark else "Dark Mode", key="theme_toggle"):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.session_state["exec_animated"] = False
        st.rerun()''',
'''# Top header row logic removed'''
)

# 3. Sidebar Tab Replacement
text = text.replace(
'''# ─────────────────────────────────────────────────────────────────────
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
])''',
'''# ─────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2CC9B6; font-family: Inter, sans-serif; margin-bottom: 2rem;'>Verdikt</h2>", unsafe_allow_html=True)
    nav_options = {
        "📉 Summary": "Executive Summary",
        "🎯 Predict": "Interactive Prediction",
        "🔍 EDA": "EDA Explorer",
        "🏆 Model": "Model Performance",
        "🧠 SHAP": "SHAP Explainability",
        "🕹️ What-If": "What-If Simulator",
        "🤝 Analyzer": "Model Agreement Analyzer",
        "⚔️ Arena": "Head-to-Head Arena",
        "📂 Data": "Dataset Explorer"
    }
    selected_nav_label = st.radio(
        "Navigation",
        options=list(nav_options.keys()),
        index=list(nav_options.values()).index(st.session_state.get("current_nav", "Executive Summary")),
        label_visibility="collapsed"
    )
    selected_nav = nav_options[selected_nav_label]
    st.session_state["current_nav"] = selected_nav

tab1_v = selected_nav == "Interactive Prediction"
tab2_v = selected_nav == "Executive Summary"
tab3_v = selected_nav == "EDA Explorer"
tab4_v = selected_nav == "Model Performance"
tab5_v = selected_nav == "SHAP Explainability"
tab6_v = selected_nav == "What-If Simulator"
tab7_v = selected_nav == "Model Agreement Analyzer"
tab8_v = selected_nav == "Head-to-Head Arena"
tab9_v = selected_nav == "Dataset Explorer"'''
)

# Replace 'with tabX:' with 'if tabX_v:' (regex format to catch properly)
text = re.sub(r'with tab1:', 'if tab1_v:', text)
text = re.sub(r'with tab2:', 'if tab2_v:', text)
text = re.sub(r'with tab3:', 'if tab3_v:', text)
text = re.sub(r'with tab4:', 'if tab4_v:', text)
text = re.sub(r'with tab5:', 'if tab5_v:', text)
text = re.sub(r'with tab6:', 'if tab6_v:', text)
text = re.sub(r'with tab7:', 'if tab7_v:', text)
text = re.sub(r'with tab8:', 'if tab8_v:', text)
text = re.sub(r'with tab9:', 'if tab9_v:', text)

# Update Helper Functions
old_helpers = '''def apply_chart_style(fig, height=420):
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
            time.sleep(0.015)'''

new_helpers = '''def apply_chart_style(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter"),
        height=height,
        transition=dict(duration=800, easing="cubic-in-out"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.2,
            xanchor="center", x=0.5
        )
    )
    fig.update_xaxes(gridcolor=GRID, color=MUTED, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, color=MUTED, zerolinecolor=GRID)
    return fig


def metric_card(label: str, value: str, best: bool = False, icon: str = "📊"):
    cls = "metric-card best" if best else "metric-card"
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="mc-icon">{icon}</div>'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def animated_metric_card(label: str, final_int: int, prefix: str = "", suffix: str = "", icon: str = "📊"):
    ph = st.empty()
    steps = 18
    for i in range(steps + 1):
        cur = int(final_int * i / steps)
        ph.markdown(
            f'<div class="metric-card">'
            f'<div class="mc-icon">{icon}</div>'
            f'<div class="mc-label">{label}</div>'
            f'<div class="mc-value">{prefix}{cur:,}{suffix}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if i < steps:
            time.sleep(0.015)'''

text = text.replace(old_helpers, new_helpers)

# Add custom icons to Executive Summary cards
text = text.replace('''animated_metric_card("Dataset Size", dataset_size)''', '''animated_metric_card("Dataset Size", dataset_size, icon="📂")''')
text = text.replace('''metric_card("Dataset Size", f"{dataset_size:,}")''', '''metric_card("Dataset Size", f"{dataset_size:,}", icon="📂")''')
text = text.replace('''animated_metric_card("Unique Models", unique_models)''', '''animated_metric_card("Unique Models", unique_models, icon="🧠")''')
text = text.replace('''metric_card("Unique Models", str(unique_models))''', '''metric_card("Unique Models", str(unique_models), icon="🧠")''')
text = text.replace('''metric_card("Best Model", best_model_name, best=True)''', '''metric_card("Best Model", best_model_name, best=True, icon="🏆")''')
text = text.replace('''metric_card("Best F1 Score", f"{best_f1:.4f}", best=True)''', '''metric_card("Best F1 Score", f"{best_f1:.4f}", best=True, icon="🎯")''')

# Now the CSS rewrite
css_pattern = re.compile(r'_CSS = f"""\n<style>\n.*?</style>\n"""', re.DOTALL)
new_css = '''_CSS = f"""
<style>
/* ── Global ── */
.stApp {{ background-color: {BG}; --border-color: {BORDER}; }}
.stApp > header {{ display: none !important; }}  /* Hide default header */
body, p, li, span, .stMarkdown, div[data-testid="stMarkdownContainer"] {{
  font-family: 'Inter', sans-serif !important; color: {TEXT} !important;
}}
h1, h2, h3, h4 {{
  font-family: 'Inter', sans-serif !important; color: {TEXT} !important; font-weight: 600 !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background-color: #E6F4EA !important; /* light pastel green */
  min-width: 90px !important;
  max-width: 90px !important;
  border-right: 1px solid {BORDER} !important;
}}
/* Hide completely the sidebar text/labels in radio but keep icons */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {{
  font-size: 0px !important; 
  display: flex !important;
  justify-content: center !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p::first-letter {{
  font-size: 24px !important; /* Show just the emoji/icon */
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
  justify-content: center !important;
  padding: 16px 0 !important;
  margin-bottom: 8px !important;
  border-radius: 12px !important;
  cursor: pointer !important;
  background-color: transparent !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
  background-color: rgba(44, 201, 182, 0.15) !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-selected="true"] {{
  background-color: #D4EDDA !important;
  border-left: 4px solid {ACCENT} !important;
  border-radius: 0 12px 12px 0 !important;
}}
/* Hide radio circle */
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
    display: none !important;
}}

/* ── Gradient header (used in tabs) - Now standard header ── */
.gradient-header {{
  padding: 1rem 0;
  text-align: left; margin-bottom: 1.5rem;
  border-bottom: 1px solid {BORDER};
}}
.gradient-header h1 {{
  color: {TEXT} !important; font-family: 'Inter', sans-serif !important;
  font-weight: 700; font-size: 2.2rem; margin: 0;
}}
.gradient-header p {{
  color: {MUTED} !important;
  font-size: 1.1rem; margin-top: 0.5rem;
}}

/* ── Fade-in ── */
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in {{ animation: fadeIn 0.45s ease-out; }}

/* ── Metric cards ── */
.metric-card {{
  background: {CARD}; border: 1px solid {BORDER};
  border-radius: 16px; padding: 1.5rem;
  text-align: center; cursor: default; margin-bottom: 1rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.03);
  transition: transform 300ms ease, box-shadow 300ms ease;
  position: relative;
}}
.metric-card:hover {{
  transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}}
.metric-card.best {{ border: 2px solid {ACCENT}; }}
.mc-icon {{
  width: 48px; height: 48px; background: #E8F5E9; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 12px auto; color: {ACCENT}; font-size: 1.5rem;
}}
.mc-label {{
  font-size: 0.85rem; font-weight: 600; color: {MUTED} !important;
  text-transform: uppercase; letter-spacing: 0.05em;
}}
.mc-value {{
  font-size: 2rem; font-weight: 700; color: {TEXT} !important;
  margin-top: 0.25rem;
}}

/* ── Info card & Synaps callout ── */
.info-card, .synaps-card {{
  background: {CARD}; border: 1px solid {BORDER};
  border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
  box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}}
.info-card {{ border-left: 4px solid {ACCENT}; }}
.synaps-card h4 {{ color: {ACCENT} !important; margin-bottom: 0.8rem; font-weight: 600 !important; }}

/* ── Prediction card ── */
.prediction-card {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px;
  padding: 1.5rem 2rem; text-align: center;
  box-shadow: 0 8px 24px rgba(0,0,0,0.06); margin: 1rem 0;
}}
.pc-label {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: {MUTED}; }}
.pc-value {{ font-size: 2rem; font-weight: 700; color: {ACCENT}; margin-top: 0.5rem; }}
.pc-sub   {{ font-size: 0.9rem; color: {TEXT}; margin-top: 0.5rem; }}

/* ── Verdict card ── */
.verdict-card {{
  background: {CARD}; border-radius: 16px; padding: 1.5rem; text-align: center; margin: 1rem 0;
  box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}}

/* ── Custom divider ── */
.custom-hr {{
  border: none; height: 1px; background: {BORDER}; margin: 2rem 0;
}}

/* ── Hide Tabs natively ── */
[data-baseweb="tab-list"] {{ display: none !important; }}

/* ── All buttons ── */
.stButton > button, .stDownloadButton > button {{
  border-radius: 8px !important;
  background: {CARD} !important;
  color: {TEXT} !important;
  font-weight: 500 !important; padding: 10px 24px !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
  border: 1px solid {BORDER} !important; transition: all 200ms ease !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: {ACCENT} !important; color: {ACCENT} !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}}

/* ── Chart Cards ── */
.js-plotly-plot {{
  background: {CARD} !important;
  border-radius: 16px !important;
  padding: 1rem !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
  border: 1px solid {BORDER} !important;
}}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {{
  background: {CARD} !important; color: {TEXT} !important;
  border-radius: 12px !important; border: 1px solid {BORDER} !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
}}
</style>
"""'''

text = css_pattern.sub(new_css, text)

# Rewrite Top Headers for specific tabs to match "clean bold left-aligned title"
# Example Executive Summary has: <div class="gradient-header"><h1>...</h1><p>...</p></div>
# We kept the 'gradient-header' CSS class but styled it as left-aligned clean text with border-bottom.
# One thing is Interactive Prediction tab had st.header("Interactive Prediction") before, we should replace st.header with gradient-header formatted markdown for consistency.
text = re.sub(r'st\.header\("Interactive Prediction"\)', '''st.markdown('<div class="gradient-header"><h1>🎯 Interactive Prediction</h1><p>Generate predictions on custom prompts and responses using the selected model.</p></div>', unsafe_allow_html=True)''', text)

text = re.sub(r'st\.header\("EDA Explorer"\)', '''st.markdown('<div class="gradient-header"><h1>🔍 EDA Explorer</h1><p>Visualise feature distributions and correlations across the dataset.</p></div>', unsafe_allow_html=True)''', text)

text = re.sub(r'st\.header\("Model Performance"\)', '''st.markdown('<div class="gradient-header"><h1>🏆 Model Performance</h1><p>Compare F1, Accuracy, and Log Loss scores across trained cross-validation models.</p></div>', unsafe_allow_html=True)''', text)

text = re.sub(r'st\.header\("SHAP Explainability"\)', '''st.markdown('<div class="gradient-header"><h1>🧠 SHAP Explainability</h1><p>Understand which features drive the LightGBM human preference predictions.</p></div>', unsafe_allow_html=True)''', text)

text = re.sub(r'st\.header\("What-If Simulator"\)', '''st.markdown('<div class="gradient-header"><h1>🕹️ What-If Simulator</h1><p>Adjust variables in real-time to see how they affect confidence scores.</p></div>', unsafe_allow_html=True)''', text)

text = re.sub(r'st\.header\("Model Agreement Analyzer"\)', '''st.markdown('<div class="gradient-header"><h1>🤝 Model Agreement Analyzer</h1><p>Simulate dual-LLM document QA audits testing for judge alignment.</p></div>', unsafe_allow_html=True)''', text)

text = re.sub(r'st\.header\("Head-to-Head Arena"\)', '''st.markdown('<div class="gradient-header"><h1>⚔️ Head-to-Head Arena</h1><p>Compare the historical win rates of specific AI foundation models directly.</p></div>', unsafe_allow_html=True)''', text)

text = re.sub(r'st\.header\("Dataset Explorer"\)', '''st.markdown('<div class="gradient-header"><h1>📂 Dataset Explorer</h1><p>Filter, explore, and export the raw engineered feature combinations.</p></div>', unsafe_allow_html=True)''', text)

# Remove the duplicated gradient header in tab1 that was there originally
text = text.replace(
'''    st.markdown(
        '<div class="gradient-header">'
        "<h1>Interactive Prediction</h1>"
        "<p>Enter a prompt, set response lengths, choose models, then click Predict.</p>"
        "</div>",
        unsafe_allow_html=True,
    )''',
'''    pass'''
)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Refactor complete! Replaced CSS, State, layout rules, and cards with icons.")
