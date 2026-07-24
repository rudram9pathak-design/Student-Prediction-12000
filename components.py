"""
components.py — Reusable premium UI building blocks for the Placement
Intelligence Platform. Pure presentation layer: nothing here touches the
model, the schema, or prediction logic.
"""
import datetime as _dt
import streamlit as st
import streamlit.components.v1 as components

from utils import risk_label


def header(test_metrics: dict):
    """Enterprise header: wordmark + live status badges."""
    today = _dt.date.today().strftime("%d %b %Y")
    acc = test_metrics.get("Accuracy", 0) * 100
    st.markdown(f"""
    <div class="aip-header aip-fade-in">
      <div class="aip-header-left">
        <svg width="42" height="42" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="aipGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#D7B15A"/>
              <stop offset="100%" stop-color="#6E44FF"/>
            </linearGradient>
          </defs>
          <circle cx="100" cy="128" r="77" fill="none" stroke="url(#aipGrad)" stroke-width="24"/>
          <circle cx="156" cy="128" r="77" fill="none" stroke="url(#aipGrad)" stroke-width="24" opacity="0.85"/>
        </svg>
        <div>
          <div class="aip-brand-name">Grant Thornton Bharat</div>
          <div class="aip-brand-sub">AI Placement Intelligence Platform</div>
          <div class="aip-brand-sub2">Enterprise Machine Learning Decision System</div>
        </div>
      </div>
      <div class="aip-badges">
        <span class="aip-badge aip-badge-live"><span class="aip-dot"></span>LIVE</span>
        <span class="aip-badge">Random Forest</span>
        <span class="aip-badge">Version 2.1</span>
        <span class="aip-badge">{acc:.1f}% Accuracy</span>
        <span class="aip-badge">{today}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def nav_pills(current_page: str, on_navigate):
    """Top navigation — five pill buttons, active page highlighted."""
    items = [
        ("home", "🏠  Dashboard"),
        ("predict", "🔮  Predict"),
        ("batch", "📁  Batch Prediction"),
        ("performance", "📈  Analytics"),
        ("about", "ℹ️  About"),
    ]
    cols = st.columns(len(items))
    for col, (key, label) in zip(cols, items):
        with col:
            st.button(
                label, use_container_width=True, key=f"nav_{key}",
                type="primary" if current_page == key else "secondary",
                on_click=on_navigate, args=(key,),
            )


def kpi_row(kpis: list):
    """kpis: list of (value_str, label_str). Renders animated-counter KPI cards.
    Numeric values (e.g. '98.5%', '12,482') count up on load via a tiny inline
    script; non-numeric values render statically."""
    cols = st.columns(len(kpis))
    for i, (col, (value, label)) in enumerate(zip(cols, kpis)):
        with col:
            st.markdown(f"""
            <div class="aip-kpi aip-fade-in" style="animation-delay:{i*0.05}s;">
              <div class="aip-kpi-value" id="aip-kpi-{i}">{value}</div>
              <div class="aip-kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def animated_counters(values: list):
    """Injects a small JS snippet that counts numeric KPI values up from 0.
    values: list of (numeric_target, suffix, elem_id) e.g. (98.5, '%', 'aip-kpi-0')."""
    js_items = ",".join(
        f'{{id:"{eid}", target:{target}, suffix:"{suffix}"}}'
        for target, suffix, eid in values
    )
    components.html(f"""
    <script>
    const items = [{js_items}];
    items.forEach(item => {{
        const el = window.parent.document.getElementById(item.id);
        if (!el) return;
        let start = 0;
        const dur = 900;
        const t0 = performance.now();
        function step(t) {{
            const p = Math.min((t - t0) / dur, 1);
            const val = (item.target * p).toFixed(item.target % 1 === 0 ? 0 : 1);
            el.innerText = val + item.suffix;
            if (p < 1) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }});
    </script>
    """, height=0, width=0)


def hero_section(test_metrics: dict, on_predict):
    st.markdown(f"""
    <div class="aip-hero aip-fade-in">
      <div class="aip-eyebrow">Placement Cell &middot; Decision Support</div>
      <h1>See the next step in <em>every</em> student's placement journey.</h1>
      <p>A tuned Random Forest model reads a student's academic record, skills and
      experience, and returns a clear placement-readiness read — with the reasoning
      behind it, not just a number.</p>
      <div class="aip-proof-strip">
        <div class="aip-proof-chip">Hold-out accuracy<b>{test_metrics['Accuracy']*100:.1f}%</b></div>
        <div class="aip-proof-chip">ROC-AUC<b>{test_metrics['ROC-AUC']*100:.1f}%</b></div>
        <div class="aip-proof-chip">Recall<b>{test_metrics['Recall']*100:.1f}%</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔮 Predict Placement", type="primary", on_click=on_predict, args=("predict",))


def card_header(title, caption):
    st.markdown(f'<div class="aip-card-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="aip-card-caption">{caption}</div>', unsafe_allow_html=True)


def risk_badge_html(proba_pct):
    label, tier = risk_label(proba_pct)
    return f'<span class="aip-badge2 aip-badge-{tier}">{label}</span>'


def stepper(labels: list, active_index: int = None):
    """Visual progress stepper shown above the prediction form's sections."""
    chips = "".join(
        f'<span class="aip-step"><b>{i+1}</b>{lab}</span>'
        for i, lab in enumerate(labels)
    )
    st.markdown(f'<div class="aip-stepper">{chips}</div>', unsafe_allow_html=True)


def reason_chips(reasons: list):
    chips = "".join(
        f'<div class="aip-reason-chip"><span class="aip-reason-label">{r["label"]}</span>'
        f'<span class="aip-reason-detail">{r["detail"]}</span></div>'
        for r in reasons
    )
    st.markdown(f'<div style="display:flex; flex-wrap:wrap;">{chips}</div>', unsafe_allow_html=True)


def recommendation_cards(placed: bool):
    """AI Recommendations — Resume / Interview / Skills / Career advice cards."""
    if placed:
        cards = [
            ("📄 Resume", "Keep it tight and quantified — highlight measurable outcomes from projects and internships."),
            ("🎤 Interview", "Do 2–3 more mock interviews focused on behavioural + role-specific technical rounds."),
            ("🛠️ Skills", "Deepen one specialisation rather than spreading thin — pick a track and go deep."),
            ("🧭 Career Advice", "Start shortlisting target companies now and tailor applications to each one."),
        ]
    else:
        cards = [
            ("📄 Resume", "Add measurable outcomes to every project/internship line — numbers beat adjectives."),
            ("🎤 Interview", "Book mock interviews weekly; interview reps move this score faster than anything else."),
            ("🛠️ Skills", "Close active backlogs first, then invest in the technical-skills gap the model flagged."),
            ("🧭 Career Advice", "Talk to the placement cell about a structured 6-week readiness plan."),
        ]
    cols = st.columns(4)
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="aip-result-card" style="padding:18px 16px;">
              <div class="aip-eyebrow-sm">{title}</div>
              <div class="aip-result-body" style="font-size:0.86rem;">{body}</div>
            </div>
            """, unsafe_allow_html=True)
