"""
Student Placement Prediction — Streamlit App
Loads the trained pipeline (model.pkl) and predicts placement status +
probability score from user-entered student details, then renders a full
visual report (gauge, radar chart, benchmark comparisons, key drivers) and
lets the user download a PDF version of the report.
"""

import io
from datetime import datetime

import joblib

import numpy as np
import pandas as pd

import streamlit as stfrom fpdf import FPDF

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    meta = joblib.load("feature_meta.pkl")
    return model, meta


model, meta = load_artifacts()

st.title("🎓 Student Placement Prediction")
st.write(
    "Enter a student's academic and skill profile below to predict whether "
    "they are likely to be **Placed** or **Not Placed**, along with a full "
    "visual report."
)

STATES = ["Delhi", "Gujarat", "Rajasthan", "Madhya Pradesh", "Uttar Pradesh",
          "Karnataka", "Tamil Nadu", "Maharashtra"]
CITIES = ["Jaipur", "Bhopal", "Bengaluru", "Delhi", "Chennai", "Indore",
          "Mumbai", "Lucknow", "Ahmedabad", "Pune"]
COLLEGE_TYPES = ["Private", "Autonomous", "Government"]
DEGREE_STREAMS = ["BCA", "B.Sc", "MCA", "BBA", "B.Com", "MBA", "B.Tech"]

# Typical "placed student" reference profile used only to give the report
# something to benchmark against on the charts. These are placeholder
# figures — swap in real averages computed from your training data
# (e.g. df[df.Placed == 1].mean()) for a more accurate comparison.
BENCHMARK = {
    "10th_Percentage": 82, "12th_Percentage": 80, "Graduation_Percentage": 75,
    "CGPA": 8.0, "Aptitude_Score": 78, "Coding_Score": 78, "Technical_Score": 78,
    "Communication_Score": 75, "Mock_Interview_Score": 75, "Resume_Score": 78,
}

with st.form("student_form"):
    st.subheader("Profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=17, max_value=30, value=21)
        state = st.selectbox("State", STATES)
        city = st.selectbox("City", CITIES)
    with col2:
        college_type = st.selectbox("College Type", COLLEGE_TYPES)
        degree_stream = st.selectbox("Degree Stream", DEGREE_STREAMS)
        internship = st.selectbox("Internship Experience", ["Yes", "No"])
        internship_months = st.slider("Internship Months", 0, 12, 0)

    st.subheader("Academics")
    col3, col4, col5 = st.columns(3)
    with col3:
        ssc = st.slider("10th Percentage", 0, 100, 75)
        hsc = st.slider("12th Percentage", 0, 100, 75)
    with col4:
        grad = st.slider("Graduation Percentage", 0, 100, 70)
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.01)
    with col5:
        backlogs = st.number_input("Backlogs", min_value=0, max_value=15, value=0)
        attendance = st.slider("Attendance (%)", 0, 100, 80)

    st.subheader("Skills & Activities")
    col6, col7, col8 = st.columns(3)
    with col6:
        projects = st.number_input("Projects", min_value=0, max_value=15, value=3)
        certifications = st.number_input("Certifications", min_value=0, max_value=15, value=2)
    with col7:
        aptitude = st.slider("Aptitude Score", 0, 100, 70)
        coding = st.slider("Coding Score", 0, 100, 70)
        technical = st.slider("Technical Score", 0, 100, 70)
    with col8:
        communication = st.slider("Communication Score", 0, 100, 70)
        mock_interview = st.slider("Mock Interview Score", 0, 100, 70)
        resume = st.slider("Resume Score", 0, 100, 70)

    submitted = st.form_submit_button("Predict Placement", use_container_width=True)


# --------------------------------------------------------------------------
# Chart builders
# --------------------------------------------------------------------------

def build_gauge(probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": "Placement Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 40], "color": "#f8d7da"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def build_radar(student_scores: dict, benchmark_scores: dict) -> go.Figure:
    categories = list(student_scores.keys())
    student_vals = list(student_scores.values())
    bench_vals = [benchmark_scores[c] for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=student_vals + [student_vals[0]],
        theta=categories + [categories[0]],
        fill="toself", name="This Student", line_color="#1f77b4",
    ))
    fig.add_trace(go.Scatterpolar(
        r=bench_vals + [bench_vals[0]],
        theta=categories + [categories[0]],
        fill="toself", name="Typical Placed Student", line_color="#ff7f0e",
        opacity=0.5,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, height=420, margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig


def build_benchmark_bar(student_scores: dict, benchmark_scores: dict, title: str) -> go.Figure:
    categories = list(student_scores.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="This Student", x=categories, y=list(student_scores.values()), marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Typical Placed Student", x=categories, y=[benchmark_scores[c] for c in categories], marker_color="#ff7f0e"))
    fig.update_layout(barmode="group", title=title, height=350, margin=dict(l=20, r=20, t=50, b=20), yaxis_range=[0, 100])
    return fig


def get_feature_importance(pipeline_model, columns) -> pd.Series | None:
    """Best-effort extraction of feature importance/coefficients from a fitted pipeline."""
    try:
        clf = pipeline_model
        if hasattr(pipeline_model, "named_steps"):
            clf = list(pipeline_model.named_steps.values())[-1]

        feature_names = columns
        if hasattr(pipeline_model, "named_steps"):
            for step in pipeline_model.named_steps.values():
                if hasattr(step, "get_feature_names_out"):
                    try:
                        feature_names = step.get_feature_names_out()
                    except Exception:
                        pass

        if hasattr(clf, "coef_"):
            values = np.abs(clf.coef_).ravel()
        elif hasattr(clf, "feature_importances_"):
            values = clf.feature_importances_
        else:
            return None

        if len(values) != len(feature_names):
            return None

        return pd.Series(values, index=feature_names).sort_values(ascending=False).head(10)
    except Exception:
        return None


# --------------------------------------------------------------------------
# PDF report generation (static matplotlib versions of the same charts)
# --------------------------------------------------------------------------

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_mpl_gauge(probability: float) -> bytes:
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.bar(x=np.pi / 2, height=1, width=np.pi, color="#eee", bottom=0)
    ax.bar(x=(1 - probability) * np.pi / 2 + probability * (np.pi / 2), height=1,
           width=probability * np.pi, color="#1f77b4", bottom=0, align="edge")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title(f"Placement Probability: {probability * 100:.1f}%", pad=20)
    return fig_to_png_bytes(fig)


def make_mpl_radar(student_scores: dict, benchmark_scores: dict) -> bytes:
    categories = list(student_scores.keys())
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    student_vals = list(student_scores.values()) + [list(student_scores.values())[0]]
    bench_vals = [benchmark_scores[c] for c in categories] + [benchmark_scores[categories[0]]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.plot(angles, student_vals, color="#1f77b4", label="This Student")
    ax.fill(angles, student_vals, color="#1f77b4", alpha=0.25)
    ax.plot(angles, bench_vals, color="#ff7f0e", label="Typical Placed Student")
    ax.fill(angles, bench_vals, color="#ff7f0e", alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    return fig_to_png_bytes(fig)


def make_mpl_bar(student_scores: dict, benchmark_scores: dict, title: str) -> bytes:
    categories = list(student_scores.keys())
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x - width / 2, list(student_scores.values()), width, label="This Student", color="#1f77b4")
    ax.bar(x + width / 2, [benchmark_scores[c] for c in categories], width, label="Typical Placed Student", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def build_pdf_report(student_name_note, prediction, probability, academic_scores,
                      test_soft_scores, drivers_text) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Student Placement Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 14)
    result_text = "Predicted: PLACED" if prediction == 1 else "Predicted: NOT PLACED"
    pdf.cell(0, 10, f"{result_text}   |   Probability: {probability * 100:.1f}%", ln=True)
    pdf.ln(2)

    gauge_png = make_mpl_gauge(probability)
    gauge_path = "/home/claude/_gauge.png"
    with open(gauge_path, "wb") as f:
        f.write(gauge_png)
    pdf.image(gauge_path, x=55, w=100)
    pdf.ln(4)

    radar_png = make_mpl_radar(test_soft_scores, {**BENCHMARK})
    radar_path = "/home/claude/_radar.png"
    with open(radar_path, "wb") as f:
        f.write(radar_png)
    pdf.image(radar_path, x=45, w=120)
    pdf.ln(4)

    academic_bar_png = make_mpl_bar(academic_scores, BENCHMARK, "Academic Scores vs Typical Placed Student")
    academic_path = "/home/claude/_academic.png"
    with open(academic_path, "wb") as f:
        f.write(academic_bar_png)
    pdf.image(academic_path, x=25, w=160)
    pdf.ln(4)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "What Drove This Prediction", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, drivers_text)

    return bytes(pdf.output())


# --------------------------------------------------------------------------
# Prediction + report rendering
# --------------------------------------------------------------------------

if submitted:
    row = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "State": state,
        "City": city,
        "College_Type": college_type,
        "Degree_Stream": degree_stream,
        "10th_Percentage": ssc,
        "12th_Percentage": hsc,
        "Graduation_Percentage": grad,
        "CGPA": cgpa,
        "Backlogs": backlogs,
        "Attendance": attendance,
        "Internship": internship,
        "Internship_Months": internship_months,
        "Projects": projects,
        "Certifications": certifications,
        "Aptitude_Score": aptitude,
        "Coding_Score": coding,
        "Communication_Score": communication,
        "Technical_Score": technical,
        "Mock_Interview_Score": mock_interview,
        "Resume_Score": resume,
    }])

    # Engineered features — must match training-time feature engineering
    row["Avg_Academic_Score"] = row[["10th_Percentage", "12th_Percentage", "Graduation_Percentage"]].mean(axis=1)
    row["Avg_Test_Score"] = row[["Aptitude_Score", "Coding_Score", "Technical_Score"]].mean(axis=1)
    row["Soft_Skill_Score"] = row[["Communication_Score", "Mock_Interview_Score", "Resume_Score"]].mean(axis=1)

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]

    st.divider()
    if prediction == 1:
        st.success("### ✅ Predicted: Placed")
    else:
        st.error("### ❌ Predicted: Not Placed")

    gauge_col, metric_col = st.columns([2, 1])
    with gauge_col:
        st.plotly_chart(build_gauge(probability), use_container_width=True)
    with metric_col:
        st.metric("Placement Probability", f"{probability * 100:.1f}%")
        st.metric("CGPA", f"{cgpa:.2f}")
        st.metric("Backlogs", backlogs)

    # Score groupings used across the report
    academic_scores = {
        "10th_Percentage": ssc, "12th_Percentage": hsc,
        "Graduation_Percentage": grad, "CGPA": round(cgpa * 10, 1),
    }
    skill_scores = {
        "Aptitude_Score": aptitude, "Coding_Score": coding, "Technical_Score": technical,
        "Communication_Score": communication, "Mock_Interview_Score": mock_interview,
        "Resume_Score": resume,
    }
    radar_benchmark = {**BENCHMARK, "CGPA": BENCHMARK["CGPA"] * 10}

    st.subheader("📊 Score Breakdown")
    tab1, tab2, tab3 = st.tabs(["Skill Radar", "Academic Comparison", "Skill Comparison"])
    with tab1:
        radar_scores = {**skill_scores, "10th_Percentage": ssc, "12th_Percentage": hsc,
                         "Graduation_Percentage": grad, "CGPA": round(cgpa * 10, 1)}
        radar_bench = {**BENCHMARK, "CGPA": BENCHMARK["CGPA"] * 10}
        st.plotly_chart(build_radar(radar_scores, radar_bench), use_container_width=True)
    with tab2:
        st.plotly_chart(
            build_benchmark_bar(academic_scores, radar_benchmark, "Academic Scores vs Typical Placed Student"),
            use_container_width=True,
        )
    with tab3:
        st.plotly_chart(
            build_benchmark_bar(skill_scores, BENCHMARK, "Test & Soft Skills vs Typical Placed Student"),
            use_container_width=True,
        )

    st.subheader("🔑 Key Drivers")
    importance = get_feature_importance(model, row.columns)
    drivers_text = (
        "This model weighs academic scores (CGPA, percentages), test/coding "
        "performance, backlogs, and soft-skill scores most heavily. Higher "
        "scores and fewer backlogs push the prediction toward Placed."
    )
    if importance is not None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        importance.sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
        ax.set_title("Top Features Influencing This Model")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        with st.expander("What drove this prediction?"):
            st.write(drivers_text)

    st.divider()
    st.subheader("📄 Download Full Report")
    pdf_bytes = build_pdf_report(
        student_name_note=None,
        prediction=prediction,
        probability=probability,
        academic_scores=academic_scores,
        test_soft_scores=skill_scores,
        drivers_text=drivers_text,
    )
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name=f"placement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Model: Logistic Regression (tuned via GridSearchCV) trained on 10,000 "
    "student records. See the accompanying Jupyter notebook for full EDA, "
    "model comparison, and evaluation metrics."
)
