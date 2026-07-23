# Student Placement Prediction System

End-to-end ML project: EDA → preprocessing → feature engineering → model
building → hyperparameter tuning → Streamlit deployment.

## Files in this submission

| File | Purpose |
|---|---|
| `Student_Placement_Prediction.ipynb` | Full, already-executed pipeline notebook (Phases 2–6): EDA, preprocessing, feature engineering, model building, evaluation, optimization. Open it in Jupyter/VS Code/Colab — every cell already has its output and charts saved in it. |
| `Phase1_Business_Understanding.docx` | Standalone 2-page Phase 1 deliverable (problem, objective, business impact). |
| `Project_Report.docx` | Final 5–6 page project report (problem, approach, insights, model comparison, conclusion) with all charts embedded. |
| `app.py` | Streamlit app — input form + prediction + probability score. |
| `model.pkl` | Trained & tuned scikit-learn Pipeline (preprocessing + Random Forest), produced by the notebook. |
| `feature_schema.json` | Feature names/ranges/categories used by `app.py` to build the form and by the notebook to document the final feature set. |
| `requirements.txt` | Python dependencies for Streamlit Cloud. |
| `indian_student_placements_v2.csv` | Source dataset (12,000 students, 59 columns). |

## How to deploy (Phase 7 — mandatory steps)

1. **Create a GitHub repository** (e.g. `student-placement-prediction`) and push these files to it:
   - `app.py`
   - `model.pkl`
   - `feature_schema.json`
   - `requirements.txt`
   - (optionally the notebook and dataset, for transparency)

   ```bash
   git init
   git add app.py model.pkl feature_schema.json requirements.txt
   git commit -m "Student placement prediction app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/student-placement-prediction.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud:**
   - Go to https://share.streamlit.io and sign in with GitHub.
   - Click **New app**, select your repository, branch `main`, and main file `app.py`.
   - Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically and gives you a live URL (e.g. `https://<app-name>.streamlit.app`).

3. **Submit:** GitHub repo link + live Streamlit URL alongside the notebook and report, per the assignment's Final Deliverables list.

## Run locally (optional, to test before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key results

- Best model: **Random Forest** (tuned via `RandomizedSearchCV`, 5-fold CV, F1-scoring)
- Test set: Accuracy 96.2% · Precision 97.1% · Recall 98.7% · F1 97.9% · ROC-AUC 98.7%
- Top predictors: Internship PPO offer, Technical Skills Score, CGPA/Academic Score, Resume Score, Aptitude Test Score, Mock Interviews Attended
- Class imbalance (~90% Placed / 10% Not Placed) handled via `class_weight="balanced"` (SMOTE's `imbalanced-learn` package was unavailable offline; `class_weight` is the scikit-learn-native equivalent named in the assignment brief).
