"""
theme.py — Design tokens and CSS injection for the Placement Intelligence
Platform's "Dark Luxury / Glassmorphism / Enterprise" theme.

Palette, type scale and effects follow the brief:
  bg #09090B · card rgba(255,255,255,.05) · border rgba(255,255,255,.08)
  purple #6E44FF · purple-2 #9B7BD4 · gold #D7B15A
  text #FFFFFF · text-2 #B5B7C5
  success #32D583 · warning #F79009 · danger #F04438
  Space Grotesk (display) · Inter (body) · IBM Plex Mono (labels/mono)
"""
import streamlit as st

COLORS = {
    "bg": "#09090B",
    "card": "rgba(255,255,255,0.05)",
    "border": "rgba(255,255,255,0.08)",
    "purple": "#6E44FF",
    "purple2": "#9B7BD4",
    "gold": "#D7B15A",
    "text": "#FFFFFF",
    "text2": "#B5B7C5",
    "success": "#32D583",
    "warning": "#F79009",
    "danger": "#F04438",
}

# Plotly-friendly hex list used by charts.py so on-screen charts match the
# same palette as the rest of the UI.
CHART_SEQUENCE = ["#6E44FF", "#D7B15A", "#9B7BD4", "#32D583", "#F79009", "#F04438"]


def _css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {{
    --bg: {COLORS['bg']};
    --card: {COLORS['card']};
    --border: {COLORS['border']};
    --purple: {COLORS['purple']};
    --purple2: {COLORS['purple2']};
    --gold: {COLORS['gold']};
    --text: {COLORS['text']};
    --text2: {COLORS['text2']};
    --success: {COLORS['success']};
    --warning: {COLORS['warning']};
    --danger: {COLORS['danger']};
    --radius: 16px;
    --radius-sm: 10px;
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3, .aip-display {{ font-family: 'Space Grotesk', sans-serif; }}
.aip-mono {{ font-family: 'IBM Plex Mono', monospace; }}

/* ---------------------------------------------------------------------
   Background — animated blurred gradient glow + subtle noise
--------------------------------------------------------------------- */
.stApp {{
    background: var(--bg);
    position: relative;
}}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.1rem; max-width: 1180px; }}
p, span, label, .stMarkdown {{ color: var(--text); }}
.stCaption, [data-testid="stCaptionContainer"] p {{ color: var(--text2) !important; }}

.aip-bg-fx {{
    position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events: none;
    background: var(--bg);
}}
.aip-bg-fx::before, .aip-bg-fx::after {{
    content: ""; position: absolute; border-radius: 50%; filter: blur(120px);
    opacity: 0.35; animation: aip-float 18s ease-in-out infinite alternate;
}}
.aip-bg-fx::before {{
    width: 560px; height: 560px; top: -160px; left: -120px;
    background: radial-gradient(circle, var(--purple) 0%, transparent 70%);
}}
.aip-bg-fx::after {{
    width: 620px; height: 620px; bottom: -220px; right: -160px;
    background: radial-gradient(circle, var(--gold) 0%, transparent 70%);
    animation-delay: 3s;
}}
.aip-bg-noise {{
    position: fixed; inset: 0; z-index: -1; pointer-events: none; opacity: 0.035;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}}
@keyframes aip-float {{
    from {{ transform: translate(0,0) scale(1); }}
    to   {{ transform: translate(30px,40px) scale(1.08); }}
}}
@keyframes aip-fade-in {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.aip-fade-in {{ animation: aip-fade-in 0.5s ease both; }}

/* ---------------------------------------------------------------------
   Header
--------------------------------------------------------------------- */
.aip-header {{
    background: var(--card);
    border: 1px solid var(--border);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: var(--radius);
    padding: 18px 26px;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
    margin-bottom: 16px;
}}
.aip-header-left {{ display: flex; align-items: center; gap: 14px; }}
.aip-brand-name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem; color: var(--text); line-height: 1.15; }}
.aip-brand-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--gold); margin-top: 3px; }}
.aip-brand-sub2 {{ font-size: 0.72rem; color: var(--text2); margin-top: 2px; }}
.aip-badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.aip-badge {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600;
    padding: 5px 12px; border-radius: 999px; border: 1px solid var(--border);
    background: rgba(255,255,255,0.04); color: var(--text2); display: flex; align-items: center; gap: 6px;
}}
.aip-badge-live {{ color: var(--success); border-color: rgba(50,213,131,0.35); background: rgba(50,213,131,0.08); }}
.aip-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--success); display: inline-block; animation: aip-pulse 1.6s ease-in-out infinite; }}
@keyframes aip-pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}

/* ---------------------------------------------------------------------
   Nav pills
--------------------------------------------------------------------- */
div[data-testid="stHorizontalBlock"] div.stButton > button {{
    border-radius: 999px !important;
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.86rem;
    padding: 0.6rem 0.6rem;
    transition: all 0.18s ease;
}}
div.stButton > button[kind="secondary"] {{
    background: var(--card);
    color: var(--text2);
    border: 1px solid var(--border);
}}
div.stButton > button[kind="secondary"]:hover {{
    border-color: rgba(215,177,90,0.45); color: var(--text); transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(110,68,255,0.18);
}}
div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--purple) 0%, var(--purple2) 100%);
    color: #fff; border: none; box-shadow: 0 4px 18px rgba(110,68,255,0.35);
}}

/* ---------------------------------------------------------------------
   Glass card base (Streamlit bordered containers)
--------------------------------------------------------------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}}
.aip-card-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: var(--text); font-size: 1.08rem; margin-bottom: 3px; }}
.aip-card-caption {{ color: var(--text2); font-size: 0.85rem; margin-bottom: 14px; }}
.aip-eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-weight: 500; font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--gold); }}

/* ---------------------------------------------------------------------
   Hero
--------------------------------------------------------------------- */
.aip-hero {{
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    padding: 42px 38px; margin-top: 6px; position: relative; overflow: hidden;
}}
.aip-hero h1 {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 2.3rem; color: #fff;
    margin: 14px 0; line-height: 1.18; max-width: 560px;
}}
.aip-hero h1 em {{ font-style: normal; background: linear-gradient(135deg, var(--gold), var(--purple2)); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.aip-hero p {{ color: var(--text2); font-size: 1rem; max-width: 480px; }}
.aip-proof-strip {{ display: flex; gap: 26px; flex-wrap: wrap; margin-top: 26px; padding-top: 20px; border-top: 1px solid var(--border); }}
.aip-proof-chip {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: var(--text2); }}
.aip-proof-chip b {{ display: block; font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; color: var(--text); font-weight: 600; margin-top: 2px; }}

/* ---------------------------------------------------------------------
   KPI cards
--------------------------------------------------------------------- */
.aip-kpi {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 16px; text-align: center; backdrop-filter: blur(14px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.aip-kpi:hover {{ transform: translateY(-3px); box-shadow: 0 10px 30px rgba(110,68,255,0.18); }}
.aip-kpi-value {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.65rem; background: linear-gradient(135deg, var(--gold), var(--purple2)); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.aip-kpi-label {{ font-family: 'IBM Plex Mono', monospace; color: var(--text2); font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }}

/* ---------------------------------------------------------------------
   Inputs on dark glass surface
--------------------------------------------------------------------- */
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {{
    background: rgba(255,255,255,0.04) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
}}
div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.04) !important; border-color: var(--border) !important; border-radius: var(--radius-sm) !important;
}}
div[data-testid="stRadio"] > div {{ gap: 8px; }}
div[data-testid="stRadio"] label {{
    background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 999px; padding: 4px 14px !important; margin: 2px 4px 2px 0 !important;
}}
div[data-testid="stCheckbox"] label p {{ color: var(--text2); }}

/* ---------------------------------------------------------------------
   Buttons — rounded, gradient, hover glow
--------------------------------------------------------------------- */
div[data-testid="stFormSubmitButton"] > button, div[data-testid="stBaseButton-primaryFormSubmit"] {{
    background: linear-gradient(135deg, var(--purple) 0%, var(--purple2) 55%, var(--gold) 130%);
    color: #fff; border: none; border-radius: 999px;
    font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: 0.01em;
    padding: 0.9rem 1.2rem; font-size: 1rem;
    box-shadow: 0 6px 24px rgba(110,68,255,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-2px); box-shadow: 0 10px 32px rgba(215,177,90,0.35);
}}
div[data-testid="stDownloadButton"] > button {{
    border-radius: 999px !important; border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.04) !important; color: var(--text) !important;
}}
div[data-testid="stDownloadButton"] > button:hover {{ border-color: var(--gold) !important; color: var(--gold) !important; }}

/* ---------------------------------------------------------------------
   Expander / tabs / dataframe / uploader
--------------------------------------------------------------------- */
div[data-testid="stExpander"] {{
    border: 1px solid var(--border) !important; border-radius: 14px !important; background: var(--card);
    backdrop-filter: blur(14px);
}}
button[data-testid="stTab"] {{ color: var(--text2); font-family: 'Inter', sans-serif; font-weight: 600; }}
button[data-testid="stTab"][aria-selected="true"] {{ color: var(--gold) !important; }}
div[data-baseweb="tab-highlight"] {{ background-color: var(--gold) !important; }}
div[data-baseweb="tab-border"] {{ background-color: var(--border) !important; }}

div[data-testid="stFileUploaderDropzone"] {{
    background: rgba(255,255,255,0.03) !important; border: 1.5px dashed var(--border) !important; border-radius: 14px !important;
    transition: border-color 0.2s ease;
}}
div[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--gold) !important; }}
section[data-testid="stFileUploaderDropzoneInstructions"] span, section[data-testid="stFileUploaderDropzoneInstructions"] small {{ color: var(--text2) !important; }}

div[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}

/* ---------------------------------------------------------------------
   Result / badge / gauge components
--------------------------------------------------------------------- */
.aip-result-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 24px 26px; height: 100%;
    backdrop-filter: blur(14px);
}}
.aip-eyebrow-sm {{ font-family: 'IBM Plex Mono', monospace; color: var(--gold); font-weight: 500; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }}
.aip-result-name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.6rem; color: #fff; margin: 2px 0 10px 0; }}
.aip-pill {{ display: inline-block; border-radius: 999px; padding: 6px 18px; font-weight: 700; font-size: 0.85rem; margin-bottom: 12px; }}
.aip-pill-win {{ background: rgba(50,213,131,0.12); color: var(--success); border: 1px solid rgba(50,213,131,0.35); }}
.aip-pill-risk {{ background: rgba(240,68,56,0.12); color: var(--danger); border: 1px solid rgba(240,68,56,0.35); }}
.aip-result-body {{ color: var(--text2); font-size: 0.95rem; line-height: 1.55; }}

.aip-badge2 {{ display: inline-block; border-radius: 999px; padding: 4px 13px; font-weight: 700; font-size: 0.78rem; }}
.aip-badge-high {{ background: rgba(215,177,90,0.15); color: var(--gold); border: 1px solid rgba(215,177,90,0.35); }}
.aip-badge-mid {{ background: rgba(155,123,212,0.15); color: var(--purple2); border: 1px solid rgba(155,123,212,0.35); }}
.aip-badge-low {{ background: rgba(240,68,56,0.13); color: var(--danger); border: 1px solid rgba(240,68,56,0.35); }}

.aip-reason-chip {{
    display: inline-flex; flex-direction: column; gap: 2px; background: rgba(255,255,255,0.04);
    border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; margin: 0 8px 8px 0;
    min-width: 190px;
}}
.aip-reason-label {{ font-weight: 700; font-size: 0.86rem; color: var(--text); }}
.aip-reason-detail {{ font-size: 0.76rem; color: var(--text2); }}

.aip-metric-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 20px 14px; text-align: center;
    backdrop-filter: blur(14px);
}}
.aip-metric-value {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.8rem; color: var(--gold); }}
.aip-metric-label {{ font-family: 'IBM Plex Mono', monospace; color: var(--text2); font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }}

/* ---------------------------------------------------------------------
   Stepper (form progress)
--------------------------------------------------------------------- */
.aip-stepper {{ display: flex; gap: 6px; margin: 4px 0 18px 0; flex-wrap: wrap; }}
.aip-step {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.04em;
    padding: 6px 12px; border-radius: 999px; border: 1px solid var(--border);
    background: rgba(255,255,255,0.03); color: var(--text2);
}}
.aip-step b {{ color: var(--gold); margin-right: 4px; }}

/* ---------------------------------------------------------------------
   Sidebar
--------------------------------------------------------------------- */
section[data-testid="stSidebar"] {{ background: #0C0C10 !important; border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] hr {{ border-color: var(--border); }}
section[data-testid="stSidebar"] a {{ color: var(--gold) !important; }}

/* ---------------------------------------------------------------------
   Decision-threshold slider — colour-coded risk zones
--------------------------------------------------------------------- */
div[data-testid="stSlider"] [data-orientation="horizontal"] [data-orientation="horizontal"] > div:first-child {{
    background: linear-gradient(90deg,
        var(--danger) 0%, var(--danger) 40%,
        var(--purple2) 40%, var(--purple2) 70%,
        var(--gold) 70%, var(--gold) 100%) !important;
    height: 5px !important; opacity: 0.9;
}}
div[data-testid="stSliderThumbValue"] {{
    background: rgba(255,255,255,0.06) !important; border: 1px solid var(--border) !important; color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}
div[data-testid="stSliderTickBar"] p {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--text2); }}

.aip-footer {{ text-align: center; color: var(--text2); font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.02em; margin-top: 34px; padding: 18px 0 10px 0; border-top: 1px solid var(--border); }}

button:focus-visible, input:focus-visible, a:focus-visible {{ outline: 2px solid var(--gold) !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}

@media (max-width: 900px) {{
    .aip-hero {{ padding: 28px 22px; }}
    .aip-hero h1 {{ font-size: 1.7rem; }}
    .aip-header {{ padding: 14px 18px; }}
}}
</style>
"""


def inject_theme():
    """Injects the full CSS theme plus the animated background layer.
    Call once, right after st.set_page_config()."""
    st.markdown(_css(), unsafe_allow_html=True)
    st.markdown('<div class="aip-bg-fx"></div><div class="aip-bg-noise"></div>', unsafe_allow_html=True)
