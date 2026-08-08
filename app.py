import streamlit as st

from src.predict import predict_student
from src.dashboard import show_dashboard


st.set_page_config(
    page_title="AI Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Student Placement Prediction and Career Recommendation System")

st.markdown("Predict the placement chances of a student and receive AI-powered career recommendations.")

st.divider()

st.header("Student Details")

col1, col2 = st.columns(2)

with col1:

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=10,
        value=1
    )

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=20,
        value=2
    )

    workshops = st.number_input(
        "Workshops / Certifications",
        min_value=0,
        max_value=20,
        value=2
    )

    aptitude = st.slider(
        "Aptitude Test Score",
        0,
        100,
        75
    )

    coding = st.slider(
        "Coding Score",
        0,
        100,
        70
    )

with col2:

    communication = st.slider(
        "Communication Score",
        0,
        100,
        75
    )

    softskills = st.slider(
        "Soft Skills Rating",
        1,
        10,
        8
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    placement_training = st.selectbox(
        "Placement Training",
        ["Yes", "No"]
    )

    ssc = st.slider(
        "SSC Marks",
        0,
        100,
        85
    )

    hsc = st.slider(
        "HSC Marks",
        0,
        100,
        82
    )

st.divider()

if st.button("🚀 Predict Placement", use_container_width=True):

    student = {

        "CGPA": cgpa,

        "Internships": internships,

        "Projects": projects,

        "Workshops/Certifications": workshops,

        "AptitudeTestScore": aptitude,

        "SoftSkillsRating": softskills,

        "ExtracurricularActivities": extracurricular,

        "PlacementTraining": placement_training,

        "SSC_Marks": ssc,

        "HSC_Marks": hsc,

        "CodingScore": coding,

        "CommunicationScore": communication

    }

    result = predict_student(student)

    st.success("Prediction Completed Successfully!")

    show_dashboard(student, result)