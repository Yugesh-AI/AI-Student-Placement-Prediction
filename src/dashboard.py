import streamlit as st

# ==========================================
# Career Readiness Score
# ==========================================

def calculate_readiness(student):

    score = 0

    score += (student["CGPA"] / 10) * 20
    score += (student["CodingScore"] / 100) * 20
    score += (student["CommunicationScore"] / 100) * 15
    score += (student["AptitudeTestScore"] / 100) * 15
    score += (student["SoftSkillsRating"] / 10) * 10
    score += min(student["Internships"], 3) * 5
    score += min(student["Projects"], 5) * 3
    score += min(student["Workshops/Certifications"], 5) * 2

    return round(min(score, 100), 2)


# ==========================================
# Strong Areas
# ==========================================

def get_strengths(student):

    strengths = []

    if student["CGPA"] >= 8:
        strengths.append("CGPA")

    if student["CodingScore"] >= 80:
        strengths.append("Coding")

    if student["CommunicationScore"] >= 80:
        strengths.append("Communication")

    if student["AptitudeTestScore"] >= 80:
        strengths.append("Aptitude")

    if student["SoftSkillsRating"] >= 8:
        strengths.append("Soft Skills")

    if student["Internships"] >= 1:
        strengths.append("Internship")

    return strengths


# ==========================================
# Weak Areas
# ==========================================

def get_weaknesses(student):

    weaknesses = []

    if student["CodingScore"] < 70:
        weaknesses.append("Coding")

    if student["CommunicationScore"] < 70:
        weaknesses.append("Communication")

    if student["AptitudeTestScore"] < 70:
        weaknesses.append("Aptitude")

    if student["Projects"] < 2:
        weaknesses.append("Projects")

    if student["Internships"] == 0:
        weaknesses.append("Internship")

    return weaknesses


# ==========================================
# Career Suggestions
# ==========================================

def career_roles(student):

    roles = []

    if student["CodingScore"] >= 80:
        roles.append("AI Engineer")
        roles.append("Python Developer")

    if student["CommunicationScore"] >= 80:
        roles.append("Business Analyst")

    if student["CGPA"] >= 8:
        roles.append("Data Analyst")

    if len(roles) == 0:
        roles.append("Software Developer")

    return roles


# ==========================================
# Dashboard
# ==========================================

def show_dashboard(student, result):

    readiness = calculate_readiness(student)

    strengths = get_strengths(student)

    weaknesses = get_weaknesses(student)

    roles = career_roles(student)

    st.title("🎓 AI Student Placement Analytics Dashboard")

    st.divider()

    st.subheader("Prediction")

    st.success(f"Prediction : {result['Prediction']}")

    st.metric(
        "Placement Probability",
        f"{result['PlacementProbability']}%"
    )

    st.metric(
        "Confidence Score",
        f"{result['ConfidenceScore']}%"
    )

    st.metric(
        "Career Readiness",
        f"{readiness}/100"
    )

    st.divider()

    st.subheader("Academic Performance")

    st.write(f"CGPA : {student['CGPA']}")

    st.write(f"SSC Marks : {student['SSC_Marks']}")

    st.write(f"HSC Marks : {student['HSC_Marks']}")

    st.write(f"Internships : {student['Internships']}")

    st.write(f"Projects : {student['Projects']}")

    st.divider()

    st.subheader("Skill Comparison")

    st.write("Coding")

    st.progress(student["CodingScore"]/100)

    st.write("Communication")

    st.progress(student["CommunicationScore"]/100)

    st.write("Aptitude")

    st.progress(student["AptitudeTestScore"]/100)

    st.write("Soft Skills")

    st.progress(student["SoftSkillsRating"]/10)

    st.divider()

    st.subheader("Strong Areas")

    for s in strengths:

        st.success(s)

    st.divider()

    st.subheader("Weak Areas")

    for w in weaknesses:

        st.warning(w)

    st.divider()

    st.subheader("Recommended Career Roles")

    for r in roles:

        st.info(r)