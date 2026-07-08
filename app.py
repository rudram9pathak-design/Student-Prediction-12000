"""
Student Placement Prediction — Streamlit App
Loads the trained pipeline (model.pkl) produced by Student_Placement_Prediction.ipynb
and serves an interactive prediction form.

Run locally:   streamlit run app.py
Deploy:        push this folder (app.py, model.pkl, feature_schema.json,
                requirements.txt) to a GitHub repo, then deploy on
                https://share.streamlit.io pointing at app.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="centered")

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
# Header
# ---------------------------------------------------------------------------
st.title("🎓 Student Placement Prediction System")
st.write(
    "Fill in the student's academic, skill and internship details below to get a "
    "placement prediction with a probability score. Model: **Random Forest** "
    "(tuned via RandomizedSearchCV)."
)

with st.expander("ℹ️ About this model"):
    st.write(
        f"- Test Accuracy: **{test_metrics['Accuracy']*100:.1f}%**\n"
        f"- Test F1 Score: **{test_metrics['F1 Score']*100:.1f}%**\n"
        f"- Test ROC-AUC: **{test_metrics['ROC-AUC']*100:.1f}%**\n\n"
        f"Top drivers of placement identified by the model: "
        f"{', '.join(top_features[:6])}."
    )

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
FRIENDLY_LABELS = {
    "Age": "Age",
    "Study_Hours_Per_Day": "Study Hours per Day",
    "CGPA": "CGPA (out of 10)",
    "History_of_Backlogs": "History of Backlogs (count)",
    "Active_Backlogs": "Active Backlogs (count)",
    "Technical_Skills_Score": "Technical Skills Score",
    "Soft_Skills_Score": "Soft Skills Score",
    "Aptitude_Test_Score": "Aptitude Test Score",
    "Internships_Completed": "Internships Completed",
    "Internship_Duration_Months": "Internship Duration (months)",
    "Internship_PPO_Offered": "Pre-Placement Offer (PPO) from Internship?",
    "Academic_Projects_Count": "Academic Projects Count",
    "Certifications_Count": "Certifications Count",
    "Hackathon_Participations": "Hackathon Participations",
    "Coding_Platform_Rating": "Coding Platform Rating (e.g. Codeforces/LeetCode)",
    "Attendance_Percentage": "Attendance Percentage",
    "Mock_Interviews_Attended": "Mock Interviews Attended",
    "10th_Percentage": "10th Percentage",
    "12th_Percentage": "12th Percentage",
    "Is_First_Generation_Graduate": "First-Generation Graduate?",
    "Scholarship_Received": "Scholarship Received?",
    "Extra_Curricular_Activities": "Extra-Curricular Activities?",
    "Leadership_Role_Held": "Leadership Role Held?",
    "LinkedIn_Profile_Updated": "LinkedIn Profile Updated?",
    "Resume_Score": "Resume Score",
    "Communication_Rating": "Communication Rating (1-5)",
    "Logical_Reasoning_Score": "Logical Reasoning Score",
    "Quantitative_Ability_Score": "Quantitative Ability Score",
    "Domain_Knowledge_Score": "Domain Knowledge Score",
    "English_Proficiency_Score": "English Proficiency Score",
    "Psychometric_Fit_Score": "Psychometric Fit Score",
    "Travel_Ready": "Travel Ready?",
    "Relocation_Ready": "Relocation Ready?",
    "Prefers_WFH": "Prefers Work From Home?",
    "Overtime_Ready": "Overtime Ready?",
    "Tech_Club_Member": "Tech Club Member?",
    "Sports_Club_Member": "Sports Club Member?",
    "Cultural_Club_Member": "Cultural Club Member?",
    "Pre_Placement_Talks_Attended": "Pre-Placement Talks Attended (count)",
    "Employment_Bond_Acceptable": "Employment Bond Acceptable?",
    "Had_Named_Internship": "Completed a Named-Company Internship?",
}
BINARY_FIELDS = {
    "Internship_PPO_Offered", "Is_First_Generation_Graduate", "Scholarship_Received",
    "Extra_Curricular_Activities", "Leadership_Role_Held", "LinkedIn_Profile_Updated",
    "Travel_Ready", "Relocation_Ready", "Prefers_WFH", "Overtime_Ready",
    "Tech_Club_Member", "Sports_Club_Member", "Cultural_Club_Member",
    "Employment_Bond_Acceptable", "Had_Named_Internship",
}
INT_FIELDS = {
    "History_of_Backlogs", "Active_Backlogs", "Internships_Completed",
    "Internship_Duration_Months", "Academic_Projects_Count", "Certifications_Count",
    "Hackathon_Participations", "Mock_Interviews_Attended", "Communication_Rating",
    "Pre_Placement_Talks_Attended", "Age",
}

user_input = {}

# Score-type fields that a student may not have attempted yet (pre-interview,
# pre-aptitude-test, etc.). Ticking "Not attempted yet" sends a missing value
# instead of a guessed number; the trained pipeline's imputer (median, learned
# from the training data) fills it in — the same way it handles real missing
# scores, so no retraining or extra logic is needed.
OPTIONAL_SCORE_FIELDS = {
    "Aptitude_Test_Score", "Logical_Reasoning_Score", "Quantitative_Ability_Score",
    "Domain_Knowledge_Score", "English_Proficiency_Score", "Psychometric_Fit_Score",
    "Coding_Platform_Rating", "Technical_Skills_Score", "Soft_Skills_Score",
    "Resume_Score", "Communication_Rating",
}

def score_input(container, feat):
    """Render a slider with a 'not attempted yet' checkbox. Returns NaN if checked."""
    lo, hi = numeric_ranges[feat]
    default = round((lo + hi) / 2, 1)
    is_int = feat in INT_FIELDS
    unknown = container.checkbox("Not attempted / no score yet", key=f"{feat}__na")
    if is_int:
        val = container.slider(FRIENDLY_LABELS[feat], int(lo), int(hi), int(default),
                                disabled=unknown, key=feat)
    else:
        val = container.slider(FRIENDLY_LABELS[feat], float(lo), float(hi), float(default),
                                disabled=unknown, key=feat)
    return np.nan if unknown else val

# ---------------------------------------------------------------------------
# Master "pre-interview / fresher" toggle — one click instead of 11.
# Lives OUTSIDE the form so it reruns the page immediately (form widgets only
# update on submit). Its on_change callback writes into every individual
# "<field>__na" checkbox's session_state *before* those checkboxes are drawn,
# so they all flip together. Anyone can still untick an individual box
# afterwards if a couple of scores actually are known.
# ---------------------------------------------------------------------------
def _sync_all_scores_to_master():
    master_val = st.session_state.get("pre_interview_mode", False)
    for feat in OPTIONAL_SCORE_FIELDS:
        st.session_state[f"{feat}__na"] = master_val

st.checkbox(
    "🎓 Pre-interview / fresher profile — this student hasn't taken any tests or interviews yet",
    key="pre_interview_mode",
    on_change=_sync_all_scores_to_master,
    help="One click marks every test/skill score below as 'not attempted' instead of ticking "
         "each one individually. You can still untick a specific field afterwards if that one "
         "score is actually known.",
)

with st.form("placement_form"):
    st.subheader("Academic Details")
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

    st.subheader("Backlogs & Academics Continued")
    c1, c2, c3 = st.columns(3)
    for i, feat in enumerate(["History_of_Backlogs", "Active_Backlogs", "Academic_Projects_Count"]):
        col = [c1, c2, c3][i % 3]
        lo, hi = numeric_ranges[feat]
        with col:
            user_input[feat] = st.number_input(FRIENDLY_LABELS[feat], min_value=int(lo), max_value=int(hi) + 3, value=int(lo), step=1)

    st.subheader("Skills & Test Scores")
    st.caption(
        "Use the **Pre-interview / fresher** switch above to mark all scores as not "
        "attempted in one click, or tick **'Not attempted / no score yet'** under an "
        "individual field if only that one test hasn't happened yet."
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

    st.subheader("Internships & Preparation")
    c1, c2, c3 = st.columns(3)
    prep_fields = ["Internships_Completed", "Internship_Duration_Months", "Mock_Interviews_Attended",
                   "Certifications_Count", "Hackathon_Participations", "Pre_Placement_Talks_Attended"]
    for i, feat in enumerate(prep_fields):
        col = [c1, c2, c3][i % 3]
        lo, hi = numeric_ranges[feat]
        with col:
            user_input[feat] = st.number_input(FRIENDLY_LABELS[feat], min_value=int(lo), max_value=int(hi) + 3, value=int(lo), step=1)

    st.subheader("Background")
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

    st.subheader("Readiness Checklist")
    c1, c2, c3 = st.columns(3)
    checklist = ["Internship_PPO_Offered", "Had_Named_Internship", "Is_First_Generation_Graduate",
                 "Scholarship_Received", "Extra_Curricular_Activities", "Leadership_Role_Held",
                 "LinkedIn_Profile_Updated", "Travel_Ready", "Relocation_Ready",
                 "Prefers_WFH", "Overtime_Ready", "Tech_Club_Member",
                 "Sports_Club_Member", "Cultural_Club_Member", "Employment_Bond_Acceptable"]
    for i, feat in enumerate(checklist):
        col = [c1, c2, c3][i % 3]
        with col:
            user_input[feat] = 1 if st.checkbox(FRIENDLY_LABELS[feat], key=feat) else 0

    submitted = st.form_submit_button("🔮 Predict Placement", use_container_width=True)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if submitted:
    row = pd.DataFrame([user_input])
    # engineered features (must match notebook's add_engineered_features)
    row["Overall_Academic_Score"] = row[["10th_Percentage", "12th_Percentage"]].mean(axis=1) * 0.5 + \
                                     (row["CGPA"] / 10 * 100) * 0.5
    row["Readiness_Index"] = row[["Technical_Skills_Score", "Soft_Skills_Score", "Aptitude_Test_Score",
                                   "Resume_Score", "Communication_Rating"]].mean(axis=1)

    proba = model.predict_proba(row)[0, 1]
    pred = model.predict(row)[0]

    st.divider()
    if pred == 1:
        st.success(f"✅ Prediction: **Likely to be Placed** — probability {proba*100:.1f}%")
    else:
        st.error(f"⚠️ Prediction: **At risk of Not Being Placed** — placement probability {proba*100:.1f}%")

    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(
        "This is a data-driven estimate, not a guarantee. Use it to prioritise "
        "mock interviews, skill-building and mentoring for at-risk students."
    )
