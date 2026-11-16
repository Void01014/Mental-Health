import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / "models"

st.set_page_config(page_title="Student Mental Health Risk", layout="wide")
st.title("Student Mental Health Risk")
st.caption("Educational demo trained on simulated data. Use fictional profiles only.")

with (models_dir / "evaluation.json").open(encoding="utf-8") as file:
    evaluation = json.load(file)

st.header("Test results")
st.caption(f'{evaluation["classifier"]} evaluated on {evaluation["class_report"]["macro avg"]["support"]:.0f} reserved examples.')

test_scores = evaluation["test_scores"]
macro_f1 = test_scores["macro_f1"]
test_accuracy = test_scores["accuracy"]
test_precision = test_scores["macro_precision"]
test_recall = test_scores["macro_recall"]
test_roc_auc = test_scores["macro_roc_auc"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{test_accuracy:.3f}")
col2.metric("Macro F1", f"{macro_f1:.3f}")
col3.metric("Macro precision", f"{test_precision:.3f}")
col4.metric("Macro recall", f"{test_recall:.3f}")
col5.metric("Macro ROC AUC", f"{test_roc_auc:.3f}")

matrix = evaluation["confusion_matrix"]
labels = evaluation["labels"]
matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

st.subheader("Confusion matrix")
st.caption("Rows are actual labels. Columns are predicted labels.")
st.dataframe(matrix_df)

report = evaluation["class_report"]
recall_df = pd.DataFrame(
    {"Recall": [report[label]["recall"] for label in labels]},
    index=labels,
)
st.subheader("Recall by risk level")
st.bar_chart(recall_df.reset_index(names="Risk level"), x="Risk level", y="Recall", sort=False)

st.header("Try a fictional student profile")
with st.form("student_profile"):
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", 17, 30, 21)
        gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
        year = st.number_input("Year of study", 1, 5, 2)
        study_hours = st.number_input("Study hours per week", 0.0, 168.0, 20.0)
        gpa = st.number_input("GPA (out of 4)", 0.0, 4.0, 2.6)
        attendance = st.number_input("Attendance (%)", 0.0, 100.0, 85.0)
        job = st.selectbox("Part-time job", ["No", "Yes"])

    with right:
        sleep = st.number_input("Sleep hours per night", 0.0, 24.0, 7.0)
        social_media = st.number_input("Social media hours per day", 0.0, 24.0, 6.0)
        screen_time = st.number_input("Total screen time per day", 0.0, 24.0, 8.5)
        activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"], index=1)
        diet = st.selectbox("Diet quality", ["Poor", "Average", "Good"], index=1)
        stress = st.number_input("Financial stress (1–10)", 1.0, 10.0, 6.0)
        support = st.number_input("Family support (1–10)", 1.0, 10.0, 7.0)
        history = st.selectbox("Previous mental health history", ["No", "Yes"])

    submitted = st.form_submit_button("Predict simulated risk")

if submitted:
    if social_media > screen_time:
        st.error("Social media hours must be within total screen time.")
    else:
        profile = {
            "Age": age,
            "Gender": gender,
            "Year_of_Study": year,
            "Sleep_Hours": sleep,
            "Study_Hours_Per_Week": study_hours,
            "GPA": gpa,
            "Social_Media_Hours": social_media,
            "Screen_Time_Hours": screen_time,
            "Physical_Activity": activity,
            "Financial_Stress": stress,
            "Family_Support": support,
            "Attendance_Rate": attendance,
            "Part_Time_Job": job,
            "Diet_Quality": diet,
            "Previous_Mental_Health_History": history,
        }
        pipeline = joblib.load(models_dir / "mental_health_pipeline.joblib")
        prediction = pipeline.predict(pd.DataFrame([profile]))[0]
        st.success(f"Predicted risk for this fictional profile is {prediction}.")
