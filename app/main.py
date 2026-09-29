"""
app/main.py
────────────
Streamlit entry point for AutoReview.

Run:
    streamlit run app/main.py
    # or simply:
    python run.py
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path regardless of how Streamlit is invoked
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ── Page config (MUST be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="AutoReview — Vehicle Review Sentiment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "# AutoReview\nAI-Based Automotive Review & Customer Sentiment Analytics",
    },
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Dark background ── */
.stApp {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #0a1628 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1e 0%, #0f172a 60%, #1a2540 100%) !important;
    border-right: 1px solid #1e2d45 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #e2e8f0 !important;
}

/* ── Text ── */
h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
p, li, span { color: #cbd5e1; }
label { color: #94a3b8 !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b, #162033);
    border: 1px solid #2d3f5e;
    border-radius: 14px;
    padding: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
}
div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.8rem !important; }
div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(124, 58, 237, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(124,58,237,0.5) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}
.stButton > button:disabled {
    background: #1e293b !important;
    color: #475569 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0f766e, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(8, 145, 178, 0.4) !important;
}

/* ── Text inputs / textareas ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #131f35 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3f5e !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    background: #131f35 !important;
    border-color: #2d3f5e !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
.stSelectbox [data-baseweb="popover"] {
    background: #1e293b !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { padding: 10px 0; }
.stSlider .thumb { background: #3b82f6 !important; }
.stSlider .track { background: #334155 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #131f35;
    border-radius: 12px;
    gap: 4px;
    padding: 5px;
    border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #94a3b8 !important;
    font-weight: 500;
    padding: 8px 16px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: #1e293b;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    font-weight: 600 !important;
}

/* ── DataFrames ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #1e2d45 !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1d4ed8, #7c3aed, #06b6d4) !important;
    border-radius: 10px !important;
}
.stProgress > div > div {
    background: #131f35 !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
details {
    background: #131f35 !important;
    border-radius: 12px !important;
    border: 1px solid #1e2d45 !important;
}
details summary {
    color: #94a3b8 !important;
    padding: 12px 16px !important;
    cursor: pointer;
}
details[open] summary {
    border-bottom: 1px solid #1e2d45;
    color: #e2e8f0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #131f35;
    border: 2px dashed #2d3f5e;
    border-radius: 14px;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6;
}

/* ── Checkboxes ── */
.stCheckbox > label { color: #94a3b8 !important; }
.stCheckbox input:checked + div { background: #3b82f6 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a1020; }
::-webkit-scrollbar-thumb { background: #2d3f5e; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 14px; overflow: hidden; }

/* ── Alert / info boxes ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Nav item active state in sidebar ── */
.nav-item-active {
    background: linear-gradient(135deg, #1d4ed820, #7c3aed20) !important;
    border-left: 3px solid #7c3aed !important;
    border-radius: 8px;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Code blocks ── */
code {
    background: #0f172a !important;
    color: #38bdf8 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
}

/* ── Number input ── */
.stNumberInput > div > div > input {
    background: #131f35 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3f5e !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "engine": None,
        "model_loaded": False,
        "current_model_key": None,
        "current_checkpoint": None,
        "single_result": None,
        "active_page": "🏠 Single Review",
        "batch_df_raw": None,
        "batch_results_df": None,
        "train_df": None,
        "eval_df": None,
        "eval_results": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Lazy engine loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine():
    from src.inference.inference_engine import InferenceEngine
    return InferenceEngine()


# ── Page definitions ──────────────────────────────────────────────────────────
PAGES = {
    "🏠 Single Review": ("app.pages.page_home", "render"),
    "📊 Batch Analytics": ("app.pages.page_batch", "render"),
    "🎓 Model Training": ("app.pages.page_training", "render"),
    "📈 Model Evaluation": ("app.pages.page_evaluation", "render"),
    "ℹ️ About": ("app.pages.page_about", "render"),
}

PAGE_COLORS = {
    "🏠 Single Review": "#38bdf8",
    "📊 Batch Analytics": "#22c55e",
    "🎓 Model Training": "#a855f7",
    "📈 Model Evaluation": "#f59e0b",
    "ℹ️ About": "#64748b",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="text-align:center;padding:24px 0 16px;">
        <div style="font-size:3.2rem;filter:drop-shadow(0 4px 12px rgba(56,189,248,0.4));">🚗</div>
        <h2 style="background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;
                   -webkit-text-fill-color:transparent;margin:8px 0 2px;font-size:1.5rem;font-weight:800;">
            AutoReview
        </h2>
        <p style="color:#475569;font-size:0.72rem;margin:0;line-height:1.5;">
            Aspect-Based Sentiment Analysis<br>
            <span style="color:#334155;">for Automotive Reviews</span>
        </p>
    </div>
    <hr style="border:0;border-top:1px solid #1e2d45;margin:0 0 16px 0;">
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────
    st.markdown('<p style="color:#475569;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;margin-bottom:8px;">NAVIGATION</p>', unsafe_allow_html=True)

    for page_name in PAGES:
        color = PAGE_COLORS.get(page_name, "#38bdf8")
        is_active = st.session_state.get("active_page") == page_name
        active_style = (
            f"background:linear-gradient(135deg,{color}15,{color}08);"
            f"border-left:3px solid {color};"
            f"color:{color} !important;"
        ) if is_active else ""
        btn_label = page_name

        if st.button(
            btn_label,
            key=f"nav_{page_name}",
            use_container_width=True,
        ):
            st.session_state["active_page"] = page_name
            st.rerun()

    st.markdown('<hr style="border:0;border-top:1px solid #1e2d45;margin:16px 0;">', unsafe_allow_html=True)

    # ── Model loader ──────────────────────────────────────────────────────
    st.markdown('<p style="color:#475569;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;margin-bottom:8px;">ACTIVE MODEL</p>', unsafe_allow_html=True)

    from src.config_loader import load_config
    cfg = load_config()
    model_options = {v["display_name"]: k for k, v in cfg["models"].items()}

    selected_display = st.selectbox(
        "Model",
        list(model_options.keys()),
        label_visibility="collapsed",
    )
    selected_model_key = model_options[selected_display]

    if st.button("⚡ Load Model", use_container_width=True):
        engine = get_engine()
        with st.spinner(f"Loading {selected_display}…"):
            try:
                engine.load_model(selected_model_key)
                ckpt = engine._current_checkpoint
                st.session_state["model_loaded"] = True
                st.session_state["current_model_key"] = selected_model_key
                st.session_state["current_checkpoint"] = ckpt
                st.session_state["engine"] = engine
                if ckpt:
                    ckpt_name = os.path.basename(ckpt)
                    st.success(f"✅ {selected_display} ready!\n\nCheckpoint: `{ckpt_name}`")
                else:
                    st.warning(f"⚠️ Loaded pretrained weights — fine-tune the model for best results.")
            except Exception as exc:
                st.error(f"Failed to load: {exc}")

    # Status indicator
    st.markdown('<div style="margin-top:10px;">', unsafe_allow_html=True)
    if st.session_state.get("model_loaded"):
        mk = st.session_state.get("current_model_key", "")
        display = cfg["models"].get(mk, {}).get("display_name", mk)
        ckpt = st.session_state.get("current_checkpoint")
        ckpt_label = (
            f"<br><span style='color:#4ade80;font-size:0.65rem;'>📁 {os.path.basename(ckpt)}</span>"
            if ckpt else ""
        )
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#052e16,#0a3d20);'
            f'border:1px solid #22c55e40;border-radius:10px;padding:12px;text-align:center;">'
            f'<span style="color:#22c55e;font-size:0.82rem;font-weight:600;">✅ {display}</span>'
            f'{ckpt_label}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#1c0a0a,#2d0f0f);'
            'border:1px solid #ef444440;border-radius:10px;padding:12px;text-align:center;">'
            '<span style="color:#ef4444;font-size:0.82rem;">⚠️ No model loaded</span><br>'
            '<span style="color:#7f1d1d;font-size:0.68rem;">Select a model and click Load</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:0;border-top:1px solid #1e2d45;margin:16px 0 8px;">', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#1e2d45;font-size:0.65rem;text-align:center;margin:0;">'
        'v1.0.0 · ABSA System · AutoReview</p>',
        unsafe_allow_html=True,
    )


# ── Render active page ────────────────────────────────────────────────────────
active_page = st.session_state.get("active_page", "🏠 Single Review")
module_path, func_name = PAGES.get(active_page, ("app.pages.page_home", "render"))

import importlib
try:
    module = importlib.import_module(module_path)
    render_fn = getattr(module, func_name)
    render_fn()
except Exception as exc:
    st.error(f"Failed to load page '{active_page}': {exc}")
    st.exception(exc)
