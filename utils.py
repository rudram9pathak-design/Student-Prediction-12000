"""
utils.py — Backend logic for the Placement Intelligence Platform.

Everything in this file is the ORIGINAL model/data logic from the previous
app.py, copied verbatim. No prediction math, feature engineering, schema
handling, or batch-scoring behaviour has been changed — only reorganised
into functions/constants that app.py can import. This keeps the redesign
100% behaviour-preserving.
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Model + schema loading (unchanged)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_schema():
    with open("feature_schema.json") as f:
        return json.load(f)


ENGINEERED_FEATURES = ["Overall_Academic_Score", "Readiness_Index"]

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

# The skill dimensions used for the radar chart on the full report. All are
# 0-100 (or rescaled to 0-100) so they can share one axis.
RADAR_FIELDS = [
    "Technical_Skills_Score", "Soft_Skills_Score", "Aptitude_Test_Score",
    "Domain_Knowledge_Score", "English_Proficiency_Score", "Resume_Score",
]

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


def engineer_features(row_df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the two engineered columns the notebook adds before training,
    so single-student and batch predictions both feed the model identically."""
    f = row_df.copy()
    f["Overall_Academic_Score"] = f[["10th_Percentage", "12th_Percentage"]].mean(axis=1) * 0.5 + \
                                   (f["CGPA"] / 10 * 100) * 0.5
    f["Readiness_Index"] = f[["Technical_Skills_Score", "Soft_Skills_Score", "Aptitude_Test_Score",
                               "Resume_Score", "Communication_Rating"]].mean(axis=1)
    return f


@st.cache_data
def get_feature_importance(_model):
    """Pull feature importances straight out of the trained pipeline.
    The notebook doesn't compute SHAP values, so this native
    Random-Forest importance is the best available alternative — it comes
    from the same model that makes every prediction, no extra dependency
    needed."""
    try:
        pre = _model.named_steps["preprocessor"]
        clf = _model.named_steps["model"]
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]
        cat_encoder = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        names = list(num_cols) + cat_names
        importances = pd.Series(clf.feature_importances_, index=names)
        # Collapse one-hot columns back to their parent categorical feature
        # so the chart reads "Department" instead of six dummy columns.
        grouped = {}
        for name, val in importances.items():
            parent = name
            for cat in cat_cols:
                if name.startswith(cat + "_"):
                    parent = cat
                    break
            grouped[parent] = grouped.get(parent, 0.0) + val
        return pd.Series(grouped).sort_values(ascending=False)
    except Exception:
        return None


def risk_label(proba_pct):
    """Returns (label, tier) where tier is 'high' | 'mid' | 'low' — unchanged thresholds."""
    if proba_pct >= 70:
        return "Strong Fit", "high"
    elif proba_pct >= 40:
        return "Borderline", "mid"
    return "At Risk", "low"


def build_required_batch_columns(numeric_features_raw, categorical_features):
    return [c for c in numeric_features_raw if c not in ENGINEERED_FEATURES] + categorical_features


def default_fill_batch(raw_df, required_batch_columns, categorical_features, categorical_options, numeric_ranges):
    """Same missing/partial-column defaulting behaviour as the original app —
    unchanged logic, just extracted into a function."""
    score_df = raw_df.copy()
    defaulted_cols = []

    for col in required_batch_columns:
        if col not in score_df.columns:
            defaulted_cols.append(col)
            if col in categorical_features:
                score_df[col] = categorical_options[col][0]
            else:
                lo, hi = numeric_ranges[col]
                score_df[col] = round((lo + hi) / 2, 1)
        else:
            if col in numeric_ranges:
                score_df[col] = pd.to_numeric(score_df[col], errors="coerce")
                lo, hi = numeric_ranges[col]
                fill_val = round((lo + hi) / 2, 1)
                if score_df[col].isna().any():
                    defaulted_cols.append(f"{col} (some rows)")
                score_df[col] = score_df[col].fillna(fill_val)
            else:
                fill_val = categorical_options.get(col, [None])[0]
                if score_df[col].isna().any():
                    defaulted_cols.append(f"{col} (some rows)")
                score_df[col] = score_df[col].fillna(fill_val)

    return score_df, defaulted_cols


def generate_reasons(user_input: dict, importances=None, top_n=5):
    """NEW helper (UI-only, no effect on the prediction): turns the raw input
    values into short human-readable 'why' chips for the result page, using
    simple, transparent thresholds on the same fields the model sees. Purely
    descriptive — it does not feed back into the model.
    """
    reasons = []

    def get(k):
        v = user_input.get(k, np.nan)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    checks = [
        ("CGPA", lambda v: v >= 8.0, "Strong CGPA", lambda v: f"CGPA of {v:.1f}/10 is above the placement-ready band"),
        ("CGPA", lambda v: v < 6.0, "Low CGPA", lambda v: f"CGPA of {v:.1f}/10 is below the typical cutoff"),
        ("Technical_Skills_Score", lambda v: v >= 75, "Excellent Technical Skills", lambda v: f"Technical skills score of {v:.0f} stands out"),
        ("Technical_Skills_Score", lambda v: v < 45, "Weak Technical Skills", lambda v: f"Technical skills score of {v:.0f} needs work"),
        ("Internships_Completed", lambda v: v >= 1, "Internship Experience", lambda v: f"{int(v)} internship(s) completed"),
        ("Internships_Completed", lambda v: v == 0, "No Internship Yet", lambda v: "No internships completed so far"),
        ("Resume_Score", lambda v: v >= 75, "Strong Resume Score", lambda v: f"Resume score of {v:.0f} is well above average"),
        ("Resume_Score", lambda v: v < 45, "Resume Needs Work", lambda v: f"Resume score of {v:.0f} is holding the profile back"),
        ("Attendance_Percentage", lambda v: v >= 85, "Excellent Attendance", lambda v: f"Attendance at {v:.0f}%"),
        ("Attendance_Percentage", lambda v: v < 65, "Low Attendance", lambda v: f"Attendance at {v:.0f}% is a risk flag"),
        ("Active_Backlogs", lambda v: v == 0, "No Active Backlogs", lambda v: "No active backlogs"),
        ("Active_Backlogs", lambda v: v >= 2, "Active Backlogs", lambda v: f"{int(v)} active backlog(s) on record"),
        ("Certifications_Count", lambda v: v >= 3, "Well Certified", lambda v: f"{int(v)} certification(s) completed"),
        ("Mock_Interviews_Attended", lambda v: v >= 3, "Interview Practiced", lambda v: f"{int(v)} mock interview(s) attended"),
        ("Communication_Rating", lambda v: v >= 4, "Strong Communication", lambda v: f"Communication rated {v:.0f}/5"),
    ]

    for feat, cond, label, detail in checks:
        v = get(feat)
        if not np.isnan(v) and cond(v):
            reasons.append({"label": label, "detail": detail(v)})

    return reasons[:top_n] if reasons else [
        {"label": "Profile Recorded", "detail": "Prediction is based on the full submitted profile"}
    ]
