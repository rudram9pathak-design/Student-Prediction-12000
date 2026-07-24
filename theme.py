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

html, body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; }}
.aip-mono {{ font-family: 'IBM Plex Mono', monospace; }}

/* Background glow + noise */
.stApp {{
    background: var(--bg);
    position: relative;
}}
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
</style>
"""

def inject_theme():
    """Injects the CSS theme plus animated background layer."""
    st.markdown(_css(), unsafe_allow_html=True)
    st.markdown('<div class="aip-bg-fx"></div><div class="aip-bg-noise"></div>', unsafe_allow_html=True)
