"""
Student Placement Prediction — Streamlit App (GT Bharat dashboard theme)
Loads the trained pipeline (model.pkl) produced by Student_Placement_Prediction.ipynb
and serves an interactive, dashboard-style prediction experience
(Home / Predict / About) with a live placement-probability gauge.

Files needed alongside this script when deploying:
    app.py, model.pkl, feature_schema.json, requirements.txt, logo.png,
    .streamlit/config.toml

Run locally:   streamlit run app.py
Deploy:        push this folder to a GitHub repo, then deploy on
                https://share.streamlit.io pointing at app.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Student Placement Predictor | GT Bharat",
    page_icon="logo.png",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Theme — GT Bharat purple / gold, dark dashboard surface
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root{
    --gt-purple-deep:   #2E1854;
    --gt-purple:        #4F2D7F;
    --gt-purple-light:  #9B7BD4;
    --gt-purple-mist:   #241A38;
    --gt-gold:          #C9A227;
    --gt-gold-light:    #E9CE6D;
    --gt-red:           #E24C63;
    --gt-ink:           #F3EFFB;
    --gt-muted:         #A79CC0;
    --gt-card-bg:       #17102A;
    --gt-card-bg-2:     #1E1633;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3, .gt-display { font-family: 'Poppins', sans-serif; }

.stApp{ background: radial-gradient(ellipse at top, #201335 0%, #0F0A1C 55%, #0B0715 100%); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; max-width: 900px; }
p, span, label, .stMarkdown, div[data-testid="stCaptionContainer"] { color: var(--gt-ink); }
.stCaption, [data-testid="stCaptionContainer"] p { color: var(--gt-muted) !important; }

/* ---------- Banner ---------- */
.gt-banner{
    background: linear-gradient(120deg, var(--gt-purple-deep) 0%, var(--gt-purple) 100%);
    border-radius: 16px;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    margin-bottom: 14px;
}
.gt-banner-left{ display:flex; align-items:center; gap:12px; }
.gt-wordmark{ display:flex; flex-direction:column; line-height:1.05; }
.gt-wordmark .gt-name{ color:#FFFFFF; font-family:'Poppins',sans-serif; font-weight:700; font-size:1rem; letter-spacing:0.05em; text-transform:uppercase; }
.gt-wordmark .gt-sub{ color: var(--gt-gold-light); font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px; }

/* ---------- Nav buttons ---------- */
div[data-testid="column"] div.stButton > button{
    border-radius: 999px !important;
    font-family:'Poppins',sans-serif; font-weight:600; font-size:0.85rem;
    padding: 0.5rem 0.4rem;
}
div.stButton > button[kind="secondary"]{
    background: var(--gt-card-bg-2);
    color: var(--gt-ink);
    border: 1px solid rgba(155,123,212,0.25);
}
div.stButton > button[kind="primary"]{
    background: linear-gradient(120deg, var(--gt-purple) 0%, var(--gt-purple-light) 100%);
    color: #fff; border: none;
    box-shadow: 0 6px 18px rgba(79,45,127,0.5);
}

/* ---------- Hero card ---------- */
.gt-hero{
    background: linear-gradient(135deg, var(--gt-purple-deep) 0%, #1B1030 100%);
    border: 1px solid rgba(155,123,212,0.2);
    border-radius: 20px;
    padding: 40px 36px;
    margin-top: 10px;
}
.gt-hero .gt-eyebrow-text{ color: var(--gt-gold-light); font-weight:700; font-size:0.9rem; margin-bottom:10px; }
.gt-hero h1{ font-family:'Poppins',sans-serif; font-weight:800; font-size:2rem; color:#fff; margin: 0 0 12px 0; line-height:1.2; }
.gt-hero p{ color: var(--gt-muted); font-size:1rem; max-width: 520px; }

/* ---------- Section eyebrow / card ---------- */
.gt-card-title{ font-family:'Poppins',sans-serif; font-weight:700; color: var(--gt-gold-light); font-size:1.02rem; margin-bottom:2px; }
.gt-card-caption{ color: var(--gt-muted); font-size:0.85rem; margin-bottom: 14px; }

div[data-testid="stVerticalBlockBorderWrapper"]{
    background: var(--gt-card-bg) !important;
    border: 1px solid rgba(155,123,212,0.18) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}

/* ---------- Inputs on dark surface ---------- */
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input{
    background: var(--gt-card-bg-2) !important;
    color: var(--gt-ink) !important;
    border: 1px solid rgba(155,123,212,0.25) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div{
    background: var(--gt-card-bg-2) !important;
    border-color: rgba(155,123,212,0.25) !important;
    border-radius: 10px !important;
}

/* ---------- Radio pills (Yes/No, checklist) ---------- */
div[data-testid="stRadio"] > div{ gap: 8px; }
div[data-testid="stRadio"] label{
    background: var(--gt-card-bg-2);
    border: 1px solid rgba(155,123,212,0.25);
    border-radius: 999px;
    padding: 4px 14px !important;
    margin: 2px 4px 2px 0 !important;
}

/* ---------- Buttons (form submit) ---------- */
div[data-testid="stFormSubmitButton"] > button{
    background: linear-gradient(120deg, var(--gt-purple) 0%, var(--gt-gold) 160%);
    color: #fff; border: none; border-radius: 12px;
    font-family:'Poppins',sans-serif; font-weight:700; letter-spacing:0.02em;
    padding: 0.85rem 1.2rem; font-size: 1rem;
    box-shadow: 0 8px 22px rgba(79,45,127,0.45);
}

/* ---------- Expander ---------- */
div[data-testid="stExpander"]{
    border: 1px solid rgba(155,123,212,0.2) !important;
    border-radius: 14px !important;
    background: var(--gt-card-bg);
}

/* ---------- Result cards ---------- */
.gt-result-card{
    background: var(--gt-card-bg);
    border: 1px solid rgba(155,123,212,0.2);
    border-radius: 18px;
    padding: 22px 24px;
    height: 100%;
}
.gt-eyebrow-small{ color: var(--gt-gold-light); font-weight:700; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px; }
.gt-result-name{ font-family:'Poppins',sans-serif; font-weight:800; font-size:1.5rem; color:#fff; margin: 2px 0 10px 0; }
.gt-pill{ display:inline-block; border-radius:999px; padding: 5px 16px; font-weight:700; font-size:0.85rem; margin-bottom: 12px; }
.gt-pill-win{ background: rgba(201,162,39,0.18); color: var(--gt-gold-light); border: 1px solid rgba(201,162,39,0.4); }
.gt-pill-risk{ background: rgba(226,76,99,0.15); color: var(--gt-red); border: 1px solid rgba(226,76,99,0.4); }
.gt-result-body{ color: var(--gt-muted); font-size: 0.95rem; line-height:1.5; }

/* ---------- Gauge ---------- */
.gt-gauge-wrap{ display:flex; flex-direction:column; align-items:center; }
.gt-gauge{
    width:168px; height:168px; border-radius:50%;
    background: conic-gradient(var(--gauge-color) calc(var(--pct)*1%), rgba(255,255,255,0.07) 0);
    display:flex; align-items:center; justify-content:center;
    margin: 6px auto 4px auto;
}
.gt-gauge-inner{
    width:130px; height:130px; border-radius:50%;
    background: var(--gt-card-bg);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.gt-gauge-value{ font-family:'Poppins',sans-serif; font-size:1.55rem; font-weight:800; color:#fff; }
.gt-gauge-label{ font-size:0.62rem; letter-spacing:0.1em; color: var(--gt-muted); margin-top:4px; text-transform:uppercase; }

.gt-track{
    width:100%; height:8px; border-radius:999px; margin: 14px 0 10px 0;
    background: rgba(255,255,255,0.08); overflow:hidden;
}
.gt-track-fill{ height:100%; border-radius:999px; background: linear-gradient(90deg, var(--gt-purple-light), var(--gt-gold)); }

.gt-stat-row{ display:flex; justify-content:space-between; margin-top: 6px; }
.gt-stat-label{ color: var(--gt-muted); font-size:0.8rem; }
.gt-stat-value{ font-family:'Poppins',sans-serif; font-weight:800; font-size:1.15rem; color: var(--gt-gold-light); }

.gt-footer{ text-align:center; color: var(--gt-muted); font-size:0.78rem; margin-top: 30px; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="gt-banner">
  <div class="gt-banner-left">
    <svg width="38" height="38" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="gtGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#E9CE6D"/>
          <stop offset="100%" stop-color="#9B7BD4"/>
        </linearGradient>
      </defs>
      <circle cx="100" cy="128" r="77" fill="none" stroke="url(#gtGrad)" stroke-width="30"/>
      <circle cx="156" cy="128" r="77" fill="none" stroke="url(#gtGrad)" stroke-width="30" opacity="0.85"/>
    </svg>
    <div class="gt-wordmark">
      <span class="gt-name">Grant Thornton Bharat</span>
      <span class="gt-sub">Student Placement Predictor</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model + schema
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_schema():
    with open("feature_schema.json") as f:
        return json.load(f)

model = load_model()
schema = load_schema()

numeric_ranges = schema["numeric_ranges"]
categorical_options = schema["categorical_options"]
top_features = schema["top_features"]
test_metrics = schema["test_metrics"]

# ---------------------------------------------------------------------------
# Nav state
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None

def go(page):
    st.session_state.page = page

nav1, nav2, nav3 = st.columns(3)
with nav1:
    st.button("🏠 Home", use_container_width=True, key="nav_home",
              type="primary" if st.session_state.page == "home" else "secondary",
              on_click=go, args=("home",))
with nav2:
    st.button("🔮 Predict", use_container_width=True, key="nav_predict",
              type="primary" if st.session_state.page == "predict" else "secondary",
              on_click=go, args=("predict",))
with nav3:
    st.button("ℹ️ About", use_container_width=True, key="nav_about",
              type="primary" if st.session_state.page == "about" else "secondary",
              on_click=go, args=("about",))

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if st.session_state.page == "home":
    st.markdown("""
    <div class="gt-hero">
      <div class="gt-eyebrow-text">Welcome!</div>
      <h1>See the next step in every student's placement journey.</h1>
      <p>Use a tuned placement model and a clear academic profile to understand
      placement likelihood with confidence.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔮 Predict Placement", type="primary", on_click=go, args=("predict",))

# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif st.session_state.page == "about":
    with st.container(border=True):
        st.markdown('<div class="gt-card-title">About this model</div>', unsafe_allow_html=True)
        st.markdown('<div class="gt-card-caption">Random Forest, tuned via RandomizedSearchCV</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{test_metrics['Accuracy']*100:.1f}%")
        c2.metric("F1 Score", f"{test_metrics['F1 Score']*100:.1f}%")
        c3.metric("ROC-AUC", f"{test_metrics['ROC-AUC']*100:.1f}%")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="gt-card-title" style="font-size:0.95rem;">Top drivers of placement</div>', unsafe_allow_html=True)
        st.write(", ".join(top_features[:8]))

# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
elif st.session_state.page == "predict":

    FRIENDLY_LABELS = {
        "Age": "Age", "Study_Hours_Per_Day": "Study Hours per Day", "CGPA": "CGPA (out of 10)",
        "History_of_Backlogs": "History of Backlogs (count)", "Active_Backlogs": "Active Backlogs (count)",
        "Technical_Skills_Score": "Technical Skills Score", "Soft_Skills_Score": "Soft Skills Score",
        "Aptitude_Test_Score": "Aptitude Test Score", "Internships_Completed": "Internships Completed",
        "Internship_Duration_Months": "Internship Duration (months)",
        "Internship_PPO_Offered": "Pre-Placement Offer (PPO) from Internship?",
        "Academic_Projects_Count": "Academic Projects Count", "Certifications_Count": "Certifications Count",
        "Hackathon_Participations": "Hackathon Participations",
        "Coding_Platform_Rating": "Coding Platform Rating (e.g. Codeforces/LeetCode)",
        "Attendance_Percentage": "Attendance Percentage", "Mock_Interviews_Attended": "Mock Interviews Attended",
        "10th_Percentage": "10th Percentage", "12th_Percentage": "12th Percentage",
        "Is_First_Generation_Graduate": "First-Generation Graduate?", "Scholarship_Received": "Scholarship Received?",
        "Extra_Curricular_Activities": "Extra-Curricular Activities?", "Leadership_Role_Held": "Leadership Role Held?",
        "LinkedIn_Profile_Updated": "LinkedIn Profile Updated?", "Resume_Score": "Resume Score",
        "Communication_Rating": "Communication Rating (1-5)", "Logical_Reasoning_Score": "Logical Reasoning Score",
        "Quantitative_Ability_Score": "Quantitative Ability Score", "Domain_Knowledge_Score": "Domain Knowledge Score",
        "English_Proficiency_Score": "English Proficiency Score", "Psychometric_Fit_Score": "Psychometric Fit Score",
        "Travel_Ready": "Travel Ready?", "Relocation_Ready": "Relocation Ready?", "Prefers_WFH": "Prefers Work From Home?",
        "Overtime_Ready": "Overtime Ready?", "Tech_Club_Member": "Tech Club Member?",
        "Sports_Club_Member": "Sports Club Member?", "Cultural_Club_Member": "Cultural Club Member?",
        "Pre_Placement_Talks_Attended": "Pre-Placement Talks Attended (count)",
        "Employment_Bond_Acceptable": "Employment Bond Acceptable?",
        "Had_Named_Internship": "Completed a Named-Company Internship?",
    }
    BINARY_FIELDS = [
        "Internship_PPO_Offered", "Is_First_Generation_Graduate", "Scholarship_Received",
        "Extra_Curricular_Activities", "Leadership_Role_Held", "LinkedIn_Profile_Updated",
        "Travel_Ready", "Relocation_Ready", "Prefers_WFH", "Overtime_Ready",
        "Tech_Club_Member", "Sports_Club_Member", "Cultural_Club_Member",
        "Employment_Bond_Acceptable", "Had_Named_Internship",
    ]
    INT_FIELDS = {
        "History_of_Backlogs", "Active_Backlogs", "Internships_Completed",
        "Internship_Duration_Months", "Academic_Projects_Count", "Certifications_Count",
        "Hackathon_Participations", "Mock_Interviews_Attended", "Communication_Rating",
        "Pre_Placement_Talks_Attended", "Age",
    }
    OPTIONAL_SCORE_FIELDS = {
        "Aptitude_Test_Score", "Logical_Reasoning_Score", "Quantitative_Ability_Score",
        "Domain_Knowledge_Score", "English_Proficiency_Score", "Psychometric_Fit_Score",
        "Coding_Platform_Rating", "Technical_Skills_Score", "Soft_Skills_Score",
        "Resume_Score", "Communication_Rating",
    }

    def card_header(title, caption):
        st.markdown(f'<div class="gt-card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gt-card-caption">{caption}</div>', unsafe_allow_html=True)

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

    with st.container(border=True):
        card_header("Student Profile", "Personalize the result with a name (optional)")
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
            card_header("Academic Details", "Board and degree performance")
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
            card_header("Backlogs & Projects", "Academic continuity signals")
            c1, c2, c3 = st.columns(3)
            for i, feat in enumerate(["History_of_Backlogs", "Active_Backlogs", "Academic_Projects_Count"]):
                col = [c1, c2, c3][i % 3]
                lo, hi = numeric_ranges[feat]
                with col:
                    user_input[feat] = st.number_input(FRIENDLY_LABELS[feat], min_value=int(lo), max_value=int(hi) + 3, value=int(lo), step=1)

        with st.container(border=True):
            card_header("Test & Skill Scores", "Aptitude, communication and technical ability")
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
            card_header("Experience & Activity", "Internships, certifications and prep")
            c1, c2, c3 = st.columns(3)
            prep_fields = ["Internships_Completed", "Internship_Duration_Months", "Mock_Interviews_Attended",
                           "Certifications_Count", "Hackathon_Participations", "Pre_Placement_Talks_Attended"]
            for i, feat in enumerate(prep_fields):
                col = [c1, c2, c3][i % 3]
                lo, hi = numeric_ranges[feat]
                with col:
                    user_input[feat] = st.number_input(FRIENDLY_LABELS[feat], min_value=int(lo), max_value=int(hi) + 3, value=int(lo), step=1)

        with st.container(border=True):
            card_header("Background", "Demographic and career context")
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
            card_header("Readiness Checklist", "Work-readiness and involvement")
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
        pred = int(model.predict(row)[0])
        st.session_state.result = {"name": student_name.strip() or "This student", "proba": proba, "pred": pred}

    # ---------------------------------------------------------------------
    # Result panel
    # ---------------------------------------------------------------------
    result = st.session_state.result
    if result:
        st.markdown("<br>", unsafe_allow_html=True)
        placed = result["pred"] == 1
        proba_pct = result["proba"] * 100

        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            pill_class = "gt-pill-win" if placed else "gt-pill-risk"
            pill_text = "Likely Placed" if placed else "At Risk"
            headline = (
                f"Congratulations, {result['name']}! Based on the academic profile and our "
                f"model, this student has a high probability of getting placed."
                if placed else
                f"Based on the academic profile and our model, {result['name']} currently shows "
                f"a lower placement probability — early support can change that."
            )
            st.markdown(f"""
            <div class="gt-result-card">
              <div class="gt-eyebrow-small">Placement Result</div>
              <div class="gt-result-name">{result['name']}</div>
              <span class="gt-pill {pill_class}">{pill_text}</span>
              <div class="gt-result-body">{headline}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            tips = (
                "Keep building interview readiness, tailor the resume to target roles, and "
                "continue showcasing projects and certifications."
                if placed else
                "Prioritise mock interviews, close any active backlogs, and strengthen resume "
                "and communication scores with focused mentoring."
            )
            st.markdown(f"""
            <div class="gt-result-card">
              <div class="gt-eyebrow-small">Personalized Next Steps</div>
              <div class="gt-result-body">{tips}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            gauge_color = "var(--gt-gold)" if placed else "var(--gt-red)"
            st.markdown(f"""
            <div class="gt-result-card">
              <div class="gt-eyebrow-small">Model Confidence</div>
              <div class="gt-card-caption" style="margin-bottom:10px;">Probability from the current prediction</div>
              <div class="gt-gauge-wrap">
                <div class="gt-gauge" style="--pct:{proba_pct}; --gauge-color:{gauge_color};">
                  <div class="gt-gauge-inner">
                    <div class="gt-gauge-value">{proba_pct:.1f}%</div>
                    <div class="gt-gauge-label">Placement Chance</div>
                  </div>
                </div>
                <div class="gt-track" style="width:100%;">
                  <div class="gt-track-fill" style="width:{proba_pct:.1f}%;"></div>
                </div>
                <div class="gt-stat-row" style="width:100%;">
                  <div><div class="gt-stat-label">Probability</div><div class="gt-stat-value">{proba_pct:.1f}%</div></div>
                  <div style="text-align:right;"><div class="gt-stat-label">Confidence</div><div class="gt-stat-value">{proba_pct:.1f}%</div></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔁 Predict Another Student"):
            st.session_state.result = None
            st.rerun()

        if placed:
            st.balloons()

            def confetti_burst():
                components.html("""
                <canvas id="gt-confetti" style="position:fixed; top:0; left:0; width:100%; height:100%;
                    pointer-events:none; z-index:9999;"></canvas>
                <script>
                const canvas = document.getElementById('gt-confetti');
                const ctx = canvas.getContext('2d');
                function resize(){ canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
                resize(); window.addEventListener('resize', resize);
                const colors = ['#9B7BD4', '#4F2D7F', '#C9A227', '#E9CE6D', '#FFFFFF'];
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

st.markdown(
    '<div class="gt-footer">Grant Thornton Bharat · Student Placement Prediction System · '
    'Built for internal training &amp; academic demonstration</div>',
    unsafe_allow_html=True,
)
