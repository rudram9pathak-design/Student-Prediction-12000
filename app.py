"""
Student Placement Prediction — Streamlit App
"AI Placement Intelligence Platform" — premium enterprise redesign.

Files needed alongside this script when deploying:
    app.py, theme.py, components.py, charts.py, utils.py,
    model.pkl, feature_schema.json, requirements.txt,
    assets/ (optional logo), .streamlit/config.toml

Run locally:   streamlit run app.py

IMPORTANT — nothing below changes prediction math, feature engineering,
batch-scoring defaults, or session-state behaviour versus the previous
version of this app. Only presentation (layout/CSS/charts/navigation/code
organisation) has changed. See utils.py for the untouched model logic.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import theme
import charts
import components as ui
from utils import (
    load_model, load_schema, engineer_features, get_feature_importance,
    generate_reasons, build_required_batch_columns, default_fill_batch,
    FRIENDLY_LABELS, RADAR_FIELDS, BINARY_FIELDS, INT_FIELDS,
    OPTIONAL_SCORE_FIELDS, ENGINEERED_FEATURES,
)

st.set_page_config(
    page_title="AI Placement Intelligence Platform | GT Bharat",
    page_icon="🎓",
    layout="wide",
)

theme.inject_theme()

# ---------------------------------------------------------------------------
# Load model + schema (unchanged logic, from utils.py)
# ---------------------------------------------------------------------------
model = load_model()
schema = load_schema()

numeric_ranges = schema["numeric_ranges"]
categorical_options = schema["categorical_options"]
categorical_features = schema["categorical_features"]
numeric_features_raw = schema["numeric_features"]
top_features = schema["top_features"]
test_metrics = schema["test_metrics"]
best_params = schema.get("best_params", {})

REQUIRED_BATCH_COLUMNS = build_required_batch_columns(numeric_features_raw, categorical_features)

# ---------------------------------------------------------------------------
# Header + nav state
# ---------------------------------------------------------------------------
ui.header(test_metrics)

if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None
if "decision_threshold" not in st.session_state:
    st.session_state.decision_threshold = 50
if "students_processed" not in st.session_state:
    st.session_state.students_processed = 0
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = 0

# ---------------------------------------------------------------------------
# Sidebar — Decision threshold / Model card / Developer card
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="aip-card-title">Decision threshold</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='aip-card-caption'>Flag as 'Placement-ready' when probability ≥</div>",
        unsafe_allow_html=True,
    )
    st.session_state.decision_threshold = st.slider(
        "Decision threshold", 0, 100, st.session_state.decision_threshold,
        label_visibility="collapsed", key="decision_threshold_slider",
    )
    st.caption(
        f"{st.session_state.decision_threshold/100:.2f} — raise it to be stricter "
        "(more students get flagged for early help), lower it to be more lenient."
    )

    st.markdown("---")
    st.markdown('<div class="aip-card-title">Model card</div>', unsafe_allow_html=True)
    n_est = best_params.get("model__n_estimators", "—")
    st.markdown(f"""
    <div class="aip-card-caption">
    Random Forest · {n_est} trees · trained on student placement records.<br><br>
    Hold-out accuracy <b>{test_metrics['Accuracy']*100:.1f}%</b>,
    ROC-AUC <b>{test_metrics['ROC-AUC']*100:.3f}</b>.<br><br>
    Uses academics, skills, experience <i>and</i> background fields
    (gender, home state, department, family income band, etc.)
    present in the training data.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="aip-badge aip-badge-live" style="margin-top:8px;"><span class="aip-dot"></span>Deployed &amp; Live</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="aip-card-title">Developer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="aip-card-caption">
    <b style="color:#FFFFFF;">Rudram Pathak</b><br>
    Machine Learning · Data Science<br><br>
    🔗 <a href="https://www.linkedin.com/in/rudrampathak" target="_blank">LinkedIn</a><br>
    💻 <a href="https://github.com/rudram9pathak-design/Student-Prediction-12000" target="_blank">GitHub Repo</a>
    </div>
    """, unsafe_allow_html=True)


def go(page):
    st.session_state.page = page


ui.nav_pills(st.session_state.page, go)

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if st.session_state.page == "home":
    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        ui.hero_section(test_metrics, go)
    with col_r:
        with st.container(border=True):
            ui.card_header("Live Model Snapshot", "Real-time confidence read")
            st.plotly_chart(
                charts.home_gauge(test_metrics["Accuracy"] * 100, 0, 0, test_metrics["ROC-AUC"] * 100),
                use_container_width=True, config={"displayModeBar": False},
            )

    st.markdown("<br>", unsafe_allow_html=True)
    kpis = [
        (f"{test_metrics['Accuracy']*100:.1f}%", "Accuracy"),
        (f"{st.session_state.students_processed:,}", "Students Processed"),
        (f"{st.session_state.predictions_made:,}", "Predictions"),
        (f"{test_metrics['ROC-AUC']*100:.1f}%", "Confidence (ROC-AUC)"),
        ("Random Forest", "Model"),
        ("< 200ms", "Response Time"),
    ]
    ui.kpi_row(kpis)
    ui.animated_counters([
        (test_metrics['Accuracy'] * 100, "%", "aip-kpi-0"),
        (st.session_state.students_processed, "", "aip-kpi-1"),
        (st.session_state.predictions_made, "", "aip-kpi-2"),
        (test_metrics['ROC-AUC'] * 100, "%", "aip-kpi-3"),
    ])

# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif st.session_state.page == "about":
    with st.container(border=True):
        ui.card_header("About this model", "Random Forest, tuned via RandomizedSearchCV")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{test_metrics['Accuracy']*100:.1f}%")
        c2.metric("F1 Score", f"{test_metrics['F1 Score']*100:.1f}%")
        c3.metric("ROC-AUC", f"{test_metrics['ROC-AUC']*100:.1f}%")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="aip-card-title" style="font-size:0.95rem;">Top drivers of placement</div>', unsafe_allow_html=True)
        st.write(", ".join(top_features[:8]))
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "Want more detail? **📁 Batch Prediction** scores a whole cohort from a CSV, and "
            "**📈 Analytics** breaks down accuracy, precision/recall and what the model "
            "relies on most."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        ui.card_header("Technology Stack", "What this platform is built on")
        st.markdown("""
        - **Model:** scikit-learn Random Forest, hyperparameter-tuned with `RandomizedSearchCV`
        - **App:** Streamlit, with a Plotly-based analytics layer
        - **Design system:** Space Grotesk / Inter / IBM Plex Mono, dark-glass enterprise theme
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        ui.card_header("Developer", "Grant Thornton Bharat · Machine Learning & Data Science")
        st.markdown("""
        **Rudram Pathak** — Machine Learning · Data Science
        🔗 [LinkedIn](https://www.linkedin.com/in/rudrampathak) &nbsp;·&nbsp;
        💻 [GitHub Repo](https://github.com/rudram9pathak-design/Student-Prediction-12000)
        """)

# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
elif st.session_state.page == "predict":

    def score_input(container, feat):
        lo, hi = numeric_ranges[feat]
        default = round((lo + hi) / 2, 1)
        is_int = feat in INT_FIELDS
        unknown = container.checkbox("Not attempted / no score yet", key=f"{feat}__na")
        if is_int:
            val = container.slider(FRIENDLY_LABELS[feat], int(lo), int(hi), int(default), disabled=unknown, key=feat)
        else:
            val = container.slider(FRIENDLY_LABELS[feat], float(lo), float(hi), float(default), disabled=unknown, key=feat)
        return np.nan if unknown else val

    def _sync_all_scores_to_master():
        master_val = st.session_state.get("pre_interview_mode", False)
        for feat in OPTIONAL_SCORE_FIELDS:
            st.session_state[f"{feat}__na"] = master_val

    user_input = {}

    ui.stepper(["Personal", "Academic", "Skills", "Experience", "Background", "Review"])

    with st.container(border=True):
        ui.card_header("Student Profile", "Personalize the result with a name (optional)")
        student_name = st.text_input("Enter name", placeholder="e.g. Pallavi Tiwari", label_visibility="visible")

    st.checkbox(
        "🎓 Pre-interview / fresher profile — this student hasn't taken any tests or interviews yet",
        key="pre_interview_mode", on_change=_sync_all_scores_to_master,
        help="One click marks every test/skill score below as 'not attempted' instead of ticking "
             "each one individually. You can still untick a specific field afterwards if that one "
             "score is actually known.",
    )

    with st.form("placement_form"):
        with st.container(border=True):
            ui.card_header("Academic Details", "Board and degree performance")
            c1, c2, c3 = st.columns(3)
            academic_fields = ["Age", "CGPA", "10th_Percentage", "12th_Percentage", "Study_Hours_Per_Day", "Attendance_Percentage"]
            for i, feat in enumerate(academic_fields):
                col = [c1, c2, c3][i % 3]
                lo, hi = numeric_ranges[feat]
                default = round((lo + hi) / 2, 1)
                with col:
                    if feat in INT_FIELDS:
                        user_input[feat] = st.number_input(FRIENDLY_LABELS[feat], min_value=int(lo), max_value=int(hi) + 5, value=int(default), step=1)
                    else:
                        user_input[feat] = st.slider(FRIENDLY_LABELS[feat], float(lo), float(hi), float(default))

        with st.container(border=True):
            ui.card_header("Backlogs & Projects", "Academic continuity signals")
            c1, c2, c3 = st.columns(3)
            for i, feat in enumerate(["History_of_Backlogs", "Active_Backlogs", "Academic_Projects_Count"]):
                col = [c1, c2, c3][i % 3]
                lo, hi = numeric_ranges[feat]
                with col:
                    user_input[feat] = st.number_input(FRIENDLY_LABELS[feat], min_value=int(lo), max_value=int(hi) + 3, value=int(lo), step=1)

        with st.container(border=True):
            ui.card_header("Test & Skill Scores", "Aptitude, communication and technical ability")
            st.caption(
                "Use the **Pre-interview / fresher** switch above to mark all scores as not "
                "attempted in one click, or tick an individual field's checkbox if only that "
                "one test hasn't happened yet."
            )
            c1, c2, c3 = st.columns(3)
            skill_fields = ["Technical_Skills_Score", "Soft_Skills_Score", "Aptitude_Test_Score",
                             "Logical_Reasoning_Score", "Quantitative_Ability_Score", "Domain_Knowledge_Score",
                             "English_Proficiency_Score", "Psychometric_Fit_Score", "Resume_Score"]
            for i, feat in enumerate(skill_fields):
                col = [c1, c2, c3][i % 3]
                with col:
                    user_input[feat] = score_input(col, feat)
            c1, c2 = st.columns(2)
            with c1:
                user_input["Communication_Rating"] = score_input(c1, "Communication_Rating")
            with c2:
                user_input["Coding_Platform_Rating"] = score_input(c2, "Coding_Platform_Rating")

        with st.container(border=True):
            ui.card_header("Experience & Activity", "Internships, certifications and prep")
            c1, c2, c3 = st.columns(3)
            prep_fields = ["Internships_Completed", "Internship_Duration_Months", "Mock_Interviews_Attended",
                           "Certifications_Count", "Hackathon_Participations", "Pre_Placement_Talks_Attended"]
            for i, feat in enumerate(prep_fields):
                col = [c1, c2, c3][i % 3]
                lo, hi = numeric_ranges[feat]
                with col:
                    user_input[feat] = st.number_input(FRIENDLY_LABELS[feat], min_value=int(lo), max_value=int(hi) + 3, value=int(lo), step=1)

        with st.container(border=True):
            ui.card_header("Background", "Demographic and career context")
            c1, c2, c3 = st.columns(3)
            with c1:
                user_input["Gender"] = st.selectbox("Gender", categorical_options["Gender"])
                user_input["Department"] = st.selectbox("Department", categorical_options["Department"])
                user_input["Board_12th"] = st.selectbox("12th Board", categorical_options["Board_12th"])
            with c2:
                user_input["Home_State"] = st.selectbox("Home State", categorical_options["Home_State"])
                user_input["School_Medium"] = st.selectbox("School Medium", categorical_options["School_Medium"])
                user_input["Father_Occupation"] = st.selectbox("Father's Occupation", categorical_options["Father_Occupation"])
            with c3:
                user_input["Family_Income_Band"] = st.selectbox("Family Income Band", categorical_options["Family_Income_Band"])
                user_input["Accommodation_Type"] = st.selectbox("Accommodation Type", categorical_options["Accommodation_Type"])
                user_input["Career_Preference"] = st.selectbox("Career Preference", categorical_options["Career_Preference"])

        with st.container(border=True):
            ui.card_header("Readiness Checklist", "Work-readiness and involvement")
            c1, c2, c3 = st.columns(3)
            for i, feat in enumerate(BINARY_FIELDS):
                col = [c1, c2, c3][i % 3]
                with col:
                    choice = st.radio(FRIENDLY_LABELS[feat], ["Yes", "No"], horizontal=True, key=feat, index=1)
                    user_input[feat] = 1 if choice == "Yes" else 0

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Predict Placement", use_container_width=True)

    if submitted:
        row = pd.DataFrame([user_input])
        row["Overall_Academic_Score"] = row[["10th_Percentage", "12th_Percentage"]].mean(axis=1) * 0.5 + \
                                         (row["CGPA"] / 10 * 100) * 0.5
        row["Readiness_Index"] = row[["Technical_Skills_Score", "Soft_Skills_Score", "Aptitude_Test_Score",
                                       "Resume_Score", "Communication_Rating"]].mean(axis=1)
        proba = float(model.predict_proba(row)[0, 1])
        st.session_state.result = {
            "name": student_name.strip() or "This student",
            "proba": proba,
            "inputs": user_input,
        }
        st.session_state.students_processed += 1
        st.session_state.predictions_made += 1

    # -----------------------------------------------------------------
    # Result panel
    # -----------------------------------------------------------------
    result = st.session_state.result
    if result:
        st.markdown("<br>", unsafe_allow_html=True)
        placed = result["proba"] * 100 >= st.session_state.decision_threshold
        proba_pct = result["proba"] * 100

        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            pill_class = "aip-pill-win" if placed else "aip-pill-risk"
            pill_text = "Likely Placed" if placed else "Needs Improvement"
            headline = (
                f"Congratulations, {result['name']}! Based on the academic profile and our "
                f"model, this student has a high probability of getting placed."
                if placed else
                f"Based on the academic profile and our model, {result['name']} currently shows "
                f"a lower placement probability — early support can change that."
            )
            st.markdown(f"""
            <div class="aip-result-card aip-fade-in">
              <div class="aip-eyebrow-sm">AI Decision</div>
              <div class="aip-result-name">{result['name']}</div>
              <span class="aip-pill {pill_class}">{pill_text}</span>
              <div class="aip-result-body">{headline}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="aip-eyebrow-sm">Why the model predicted this</div>', unsafe_allow_html=True)
            reasons = generate_reasons(result["inputs"])
            ui.reason_chips(reasons)

        with col_right:
            with st.container(border=True):
                st.markdown('<div class="aip-eyebrow-sm">AI Confidence &middot; Risk Score</div>', unsafe_allow_html=True)
                st.plotly_chart(charts.gauge_chart(proba_pct, placed), use_container_width=True, config={"displayModeBar": False})
                st.markdown(f"""
                <div class="aip-stat-row" style="display:flex; justify-content:space-between;">
                  <div><div class="aip-card-caption" style="margin:0;">Probability</div><div class="aip-metric-value" style="font-size:1.2rem;">{proba_pct:.1f}%</div></div>
                  <div style="text-align:right;"><div class="aip-card-caption" style="margin:0;">Risk Score</div><div class="aip-metric-value" style="font-size:1.2rem;">{100-proba_pct:.1f}%</div></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="aip-eyebrow-sm">AI Recommendations</div>', unsafe_allow_html=True)
        ui.recommendation_cards(placed)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # Full Report — graphical, downloadable one-pager for this student
        # -------------------------------------------------------------
        with st.expander("📊 Open full graphical report", expanded=False):
            st.markdown(
                f'<div class="aip-card-caption">Placement probability {proba_pct:.1f}% &nbsp;·&nbsp; {ui.risk_badge_html(proba_pct)}</div>',
                unsafe_allow_html=True,
            )
            radar_vals = {k: float(result["inputs"].get(k, np.nan)) for k in RADAR_FIELDS if not pd.isna(result["inputs"].get(k, np.nan))}
            if radar_vals:
                st.plotly_chart(charts.radar_chart(radar_vals), use_container_width=True, config={"displayModeBar": False})

            importances = get_feature_importance(model)
            fig = charts.report_figure(result["name"], proba_pct, result["inputs"], top_drivers=importances)
            st.pyplot(fig, use_container_width=True)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
            st.download_button(
                "⬇️ Download this report (PNG)",
                data=buf.getvalue(),
                file_name=f"{(result['name'] or 'student').replace(' ', '_')}_placement_report.png",
                mime="image/png",
                use_container_width=True,
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

        if st.button("🔁 Predict Another Student"):
            st.session_state.result = None
            st.rerun()

        if placed:
            st.balloons()

            def confetti_burst():
                components.html("""
                <canvas id="aip-confetti" style="position:fixed; top:0; left:0; width:100%; height:100%;
                    pointer-events:none; z-index:9999;"></canvas>
                <script>
                const canvas = document.getElementById('aip-confetti');
                const ctx = canvas.getContext('2d');
                function resize(){ canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
                resize(); window.addEventListener('resize', resize);
                const colors = ['#6E44FF', '#9B7BD4', '#D7B15A', '#32D583', '#FFFFFF'];
                const N = 160;
                const pieces = Array.from({length: N}, () => ({
                    x: Math.random() * canvas.width,
                    y: -20 - Math.random() * canvas.height * 0.5,
                    w: 6 + Math.random() * 6, h: 8 + Math.random() * 10,
                    color: colors[Math.floor(Math.random() * colors.length)],
                    speedY: 2 + Math.random() * 3, speedX: -2 + Math.random() * 4,
                    rot: Math.random() * 360, rotSpeed: -8 + Math.random() * 16,
                }));
                let frame = 0; const maxFrames = 220;
                function draw(){
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    pieces.forEach(p => {
                        p.x += p.speedX; p.y += p.speedY; p.rot += p.rotSpeed;
                        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot * Math.PI / 180);
                        ctx.fillStyle = p.color; ctx.fillRect(-p.w/2, -p.h/2, p.w, p.h);
                        ctx.restore();
                    });
                    frame++;
                    if (frame < maxFrames){ requestAnimationFrame(draw); }
                    else { ctx.clearRect(0, 0, canvas.width, canvas.height); }
                }
                draw();
                </script>
                """, height=0, width=0)

            confetti_burst()

        st.caption(
            "This is a data-driven estimate, not a guarantee. Use it to prioritise "
            "mock interviews, skill-building and mentoring for at-risk students."
        )

# ---------------------------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------------------------
elif st.session_state.page == "batch":
    with st.container(border=True):
        ui.card_header("Score a whole cohort", "Drag & drop a CSV — one row per student")
        cols_preview = ", ".join(f"`{c}`" for c in REQUIRED_BATCH_COLUMNS[:6])
        st.markdown(
            f'<div class="aip-card-caption">Upload a CSV with one row per student. Needed columns include '
            f'{cols_preview}, … (all {len(REQUIRED_BATCH_COLUMNS)} model features). Extra columns like '
            f'<code>Student_ID</code> are kept in the output. Any needed column that\'s missing is '
            f'auto-filled with a sensible default (median / most common value) so the file still scores.</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Upload student CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file as a CSV: {e}")
            raw_df = None

        if raw_df is not None and len(raw_df) > 0:
            score_df, defaulted_cols = default_fill_batch(
                raw_df, REQUIRED_BATCH_COLUMNS, categorical_features, categorical_options, numeric_ranges
            )

            if defaulted_cols:
                st.info("Filled with defaults — not present (or partly missing) in your file: " +
                        ", ".join(sorted(set(defaulted_cols))))

            try:
                scored = engineer_features(score_df)
                proba = model.predict_proba(scored[REQUIRED_BATCH_COLUMNS + ENGINEERED_FEATURES])[:, 1]
            except Exception as e:
                st.error(f"Couldn't score this file with the current model: {e}")
                with st.expander("Show full error details"):
                    st.exception(e)
                proba = None

            if proba is not None:
                score_df["Placement_Probability"] = (proba * 100).round(1)
                st.session_state.students_processed += len(score_df)
                st.session_state.predictions_made += len(score_df)

                st.markdown("<br>", unsafe_allow_html=True)
                threshold = st.session_state.decision_threshold
                st.caption(
                    f"Using the decision threshold from the sidebar — flagging as "
                    f"'Placement-ready' when probability ≥ {threshold}%."
                )
                score_df["Predicted_Status"] = np.where(
                    score_df["Placement_Probability"] >= threshold, "Placed", "Not Placed"
                )

                # --- Cohort overview: KPIs + donut + distribution ---------
                with st.container(border=True):
                    ui.card_header("Cohort overview", f"{len(score_df)} students scored")
                    n_placed = int((score_df['Predicted_Status'] == 'Placed').sum())
                    n_risk = int((score_df['Predicted_Status'] == 'Not Placed').sum())
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg. probability", f"{score_df['Placement_Probability'].mean():.1f}%")
                    m2.metric("Flagged placement-ready", n_placed)
                    m3.metric("Flagged at-risk", n_risk)

                    cc1, cc2 = st.columns([1, 1.4])
                    with cc1:
                        st.plotly_chart(charts.donut_split(n_placed, n_risk), use_container_width=True, config={"displayModeBar": False})
                    with cc2:
                        st.plotly_chart(charts.distribution_histogram(score_df["Placement_Probability"], threshold), use_container_width=True, config={"displayModeBar": False})

                # --- Searchable, sortable table -----------------------------
                with st.container(border=True):
                    ui.card_header("Student results", "Most at-risk students first")
                    search = st.text_input("🔍 Search students", placeholder="Type a Student ID or any text to filter…")
                    view_df = score_df.sort_values("Placement_Probability", ascending=True)
                    if search:
                        mask = view_df.astype(str).apply(
                            lambda col: col.str.contains(search, case=False, na=False)
                        ).any(axis=1)
                        view_df = view_df[mask]
                    st.dataframe(view_df, use_container_width=True, height=380)

                    csv_bytes = score_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download scored CSV", data=csv_bytes,
                        file_name="scored_students.csv", mime="text/csv",
                        use_container_width=True,
                    )

                # --- Per-student full report from the batch -------------------
                with st.container(border=True):
                    ui.card_header("Individual student report", "Open a full graphical report for any row")
                    id_col = "Student_ID" if "Student_ID" in score_df.columns else None
                    if id_col:
                        options = score_df[id_col].astype(str).tolist()
                    else:
                        options = [f"Row {i}" for i in range(len(score_df))]
                    pick = st.selectbox("Open a full report for:", options)
                    idx = options.index(pick)
                    row = score_df.iloc[idx]
                    proba_pct = row["Placement_Probability"]
                    st.markdown(
                        f'<div class="aip-card-caption">{pick} — {proba_pct:.1f}% placement probability '
                        f'&nbsp;·&nbsp; {ui.risk_badge_html(proba_pct)}</div>',
                        unsafe_allow_html=True,
                    )
                    importances = get_feature_importance(model)
                    fig = charts.report_figure(str(pick), proba_pct, row.to_dict(), top_drivers=importances)
                    st.pyplot(fig, use_container_width=True)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
                    st.download_button(
                        "⬇️ Download this report (PNG)", data=buf.getvalue(),
                        file_name=f"{str(pick).replace(' ', '_')}_placement_report.png",
                        mime="image/png", use_container_width=True, key="batch_report_dl",
                    )
                    import matplotlib.pyplot as plt
                    plt.close(fig)

# ---------------------------------------------------------------------------
# ANALYTICS (Model Performance)
# ---------------------------------------------------------------------------
elif st.session_state.page == "performance":
    with st.container(border=True):
        ui.card_header("How good is this model?", "Metrics measured on a 20% hold-out test set the model never trained on")
        m1, m2, m3, m4, m5 = st.columns(5)
        metric_defs = [
            ("Accuracy", test_metrics.get("Accuracy")),
            ("Precision", test_metrics.get("Precision")),
            ("Recall", test_metrics.get("Recall")),
            ("F1 Score", test_metrics.get("F1 Score")),
            ("ROC-AUC", test_metrics.get("ROC-AUC")),
        ]
        for col, (label, val) in zip([m1, m2, m3, m4, m5], metric_defs):
            with col:
                st.markdown(f"""
                <div class="aip-metric-card">
                  <div class="aip-metric-value">{val*100:.1f}%</div>
                  <div class="aip-metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(charts.metrics_bar(metric_defs), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        ui.card_header(
            "What the model relies on most",
            "Random Forest feature importance — the notebook doesn't compute SHAP values, "
            "so this is the closest available read on which inputs move the prediction most.",
        )
        importances = get_feature_importance(model)
        if importances is not None and len(importances) > 0:
            st.plotly_chart(charts.feature_importance_bar(importances), use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning(
                "Feature importances aren't available from this model file — falling back to the "
                "top-feature list saved during training instead."
            )
            st.write(", ".join(top_features))

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        ui.card_header(
            "Spread of the top features",
            "Min–max range observed for each field in the training data. There's no saved test-set "
            "sample to compute real quartiles from, so this shows range rather than a true quartile "
            "box plot — an honest stand-in, not a substitute for one.",
        )
        numeric_ranges_local = numeric_ranges
        range_feats = [f for f in top_features if f in numeric_ranges_local][:10]
        if range_feats:
            st.plotly_chart(charts.range_chart(range_feats, numeric_ranges_local), use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No numeric range data available for the top features.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        ui.card_header("Model card", "How this model was built and tuned")
        st.markdown(f"""
        - **Algorithm:** Random Forest (`{best_params.get('model__n_estimators', '—')}` trees),
          tuned via `RandomizedSearchCV`
        - **Best parameters:** min samples split `{best_params.get('model__min_samples_split', '—')}`,
          min samples leaf `{best_params.get('model__min_samples_leaf', '—')}`,
          max features `{best_params.get('model__max_features', '—')}`,
          max depth `{best_params.get('model__max_depth') or 'unlimited'}`
        - **Inputs used:** academic history, skill/test scores, internship & activity signals,
          plus background fields (gender, home state, department, family income band, etc.)
          that were present in the training data.
        """)

st.markdown(
    '<div class="aip-footer">Grant Thornton Bharat · AI Placement Intelligence Platform · '
    'Built for internal training &amp; academic demonstration</div>',
    unsafe_allow_html=True,
)
