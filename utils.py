"""
utils.py — Backend logic for the Placement Intelligence Platform.

Behaviour is unchanged from the original app.py — only reorganised into
functions/constants for import. Prediction math, feature engineering,
schema handling, and batch-scoring are preserved.
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Model + schema loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model pipeline (RandomForest + preprocessing)."""
    return joblib.load("model.pkl")


@st.cache_data
def load_schema():
    """Load the feature schema JSON."""
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
    """Recreate engineered columns for consistency with training."""
    f = row_df.copy()
    f["Overall_Academic_Score"] = (
        f[["10th_Percentage", "12th_Percentage"]].mean(axis=1) * 0.5
        + (f["CGPA"] / 10 * 100) * 0.5
    )
    f["Readiness_Index"] = f[
        ["Technical_Skills_Score", "Soft_Skills_Score", "Aptitude_Test_Score",
         "Resume_Score", "Communication_Rating"]
    ].mean(axis=1)
    return f


@st.cache_data
def get_feature_importance(_model):
    """Return RandomForest feature importances collapsed to parent features."""
    try:
        pre = _model.named_steps["preprocessor"]
        clf = _model.named_steps["model"]
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]
        cat_encoder = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        names = list(num_cols) + cat_names
        importances = pd.Series(clf.feature_importances_, index=names)

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


def risk_label(proba_pct: float):
    """Map probability % to risk tier and label."""
    if proba_pct >= 70:
        return "Strong Fit", "high"
    elif proba_pct >= 40:
        return "Borderline", "mid"
    return "At Risk", "low"


def build_required_batch_columns(numeric_features_raw, categorical_features):
    """Return required batch columns excluding engineered ones."""
    return [c for c in numeric_features_raw if c not in ENGINEERED_FEATURES] + categorical_features


def default_fill_batch(raw_df, required_batch_columns, categorical_features, categorical_options, numeric_ranges):
    """Fill missing values with defaults (midpoint for numeric, first option for categorical)."""
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
                score_df[col] =
