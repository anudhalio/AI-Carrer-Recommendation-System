import sys
import os
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================
PROJECT_ROOT = r"C:\Users\shiva\Desktop\AI_Career_Recommendation_System"
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

try:
    from recommendation_engine import recommend_careers
except Exception as import_error:
    recommend_careers = None
    RECOMMENDATION_IMPORT_ERROR = import_error
else:
    RECOMMENDATION_IMPORT_ERROR = None

# ============================================================
# PAGE CONFIGURATION
# Note: Streamlit's colors are set in .streamlit/config.toml.
# ============================================================
st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎓",
    layout="wide",
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def safe_value(value, default=""):
    if value is None:
        return default
    if isinstance(value, (list, tuple, dict, set)):
        return value
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def normalize_recommendations(data):
    if data is None:
        return []
    if isinstance(data, pd.DataFrame):
        return [] if data.empty else data.to_dict(orient="records")
    if isinstance(data, pd.Series):
        return [data.to_dict()]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def get_score(item):
    if not isinstance(item, dict):
        return 0.0
    value = item.get("final_score", item.get("score", 0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_career_name(item):
    if not isinstance(item, dict):
        return "Unknown Career"
    return str(item.get("career", item.get("career_name", "Unknown Career")))


def get_list(item, key):
    if not isinstance(item, dict):
        return []
    value = item.get(key, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return list(value)
    except TypeError:
        return []


def get_salary(item):
    value = item.get("salary", {}) if isinstance(item, dict) else {}
    return value if isinstance(value, dict) else {}


def get_explanation(item):
    return str(item.get("explanation", "")) if isinstance(item, dict) else ""


def show_score(item, key, label):
    try:
        score = float(item.get(key, 0) or 0)
    except (TypeError, ValueError):
        score = 0.0
    st.metric(label, f"{score:.2f}%")


# ============================================================
# SIDEBAR / NAVIGATION
# ============================================================
st.sidebar.title("AI Career Advisor")
st.sidebar.caption("Personalized career guidance")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Career Analysis", "About Project", "About Me"],
)
st.sidebar.markdown("---")
st.sidebar.write(
    "AI-powered career recommendations using skill matching, "
    "aptitude analysis, and NLP similarity."
)

# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.title("AI Career Recommendation System")
    st.subheader("Intelligent Career Guidance Based on Your Profile")
    st.write(
        "This system analyzes your education, branch, interests, technical "
        "skills, aptitude, preferred industry, and experience to recommend "
        "suitable career paths."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Recommendation Type", "Hybrid AI")
    col2.metric("NLP Technique", "TF-IDF")
    col3.metric("Output", "Career Match")

    st.divider()
    st.subheader("How the System Works")
    columns = st.columns(4)
    steps = [
        ("Step 1 — Profile", "Enter your education, interests, skills, and experience."),
        ("Step 2 — Analysis", "The system analyzes your profile."),
        ("Step 3 — AI Matching", "Multiple recommendation factors are combined."),
        ("Step 4 — Recommendation", "Review career matches and a roadmap."),
    ]
    for col, (heading, description) in zip(columns, steps):
        with col:
            st.markdown(f"**{heading}**")
            st.write(description)

    st.divider()
    st.subheader("Recommendation Logic")
    st.write("The final career compatibility score uses these factors:")
    st.markdown(
        """
        - Technical Skill Matching — **30%**
        - Interest Matching — **20%**
        - Education and Branch — **15%**
        - Aptitude — **15%**
        - Industry Preference — **10%**
        - NLP Similarity — **5%**
        - Experience — **5%**
        """
    )
    st.info("The Match Score represents career compatibility, not classification accuracy.")


# ============================================================
# CAREER ANALYSIS PAGE
# ============================================================
def career_analysis_page():
    st.title("Career Analysis")
    st.write("Enter your profile information to receive personalized career recommendations.")

    if RECOMMENDATION_IMPORT_ERROR is not None:
        st.error("The recommendation engine could not be imported.")
        st.code(str(RECOMMENDATION_IMPORT_ERROR))
        st.caption(
            "Check that src\\recommendation_engine.py exists in your project folder."
        )
        return

    if st.button("Refresh / New Prediction"):
        st.session_state.pop("career_analysis_result", None)
        st.rerun()

    st.divider()
    st.subheader("Education and Background")
    col1, col2 = st.columns(2)
    with col1:
        education = st.selectbox("Education", ["B.Tech", "M.Tech", "B.Sc", "MCA", "MBA", "Other"])
    with col2:
        branch = st.selectbox(
            "Branch",
            [
                "Artificial Intelligence & Machine Learning",
                "Computer Science",
                "Information Technology",
                "Electronics & Communication",
                "Electrical Engineering",
                "Mechanical Engineering",
                "Civil Engineering",
                "Commerce",
                "General",
            ],
        )

    st.subheader("Interests and Career Preference")
    col1, col2 = st.columns(2)
    with col1:
        interest = st.multiselect(
            "Your Interests",
            [
                "Artificial Intelligence", "Machine Learning", "Data Science",
                "Programming", "Web Development", "Cloud Computing",
                "Cybersecurity", "Finance", "Business", "Marketing",
                "Research", "Teaching",
            ],
            default=["Artificial Intelligence"],
        )
    with col2:
        industry = st.selectbox(
            "Preferred Industry",
            ["Software", "Finance", "Healthcare", "Marketing", "Education", "Retail", "Manufacturing"],
        )

    experience = st.selectbox(
        "Experience Level",
        ["Student", "Fresher", "0-1 Years", "1-3 Years", "3-5 Years", "5+ Years"],
    )

    st.subheader("Technical Skills")
    skills = st.multiselect(
        "Select your technical skills",
        [
            "Python", "SQL", "Machine Learning", "AI", "Data Science",
            "Data Analytics", "Java", "C++", "JavaScript", "React", "AWS",
            "Google Cloud", "Excel", "Financial Modeling", "Market Research",
            "SEO", "Google Ads", "Social Media", "Content Writing", "Research",
            "Sales", "Customer Service", "Risk Analysis", "Supply Chain",
            "Quality Control", "Teaching", "Medical Research", "Nursing", "Patient Care",
        ],
        default=["Python"],
    )

    st.subheader("Aptitude and Personal Skills")
    col1, col2, col3 = st.columns(3)
    with col1:
        programming = st.select_slider("Programming", options=["Low", "Medium", "High"], value="High")
        mathematics = st.select_slider("Mathematics", options=["Low", "Medium", "High"], value="High")
    with col2:
        communication = st.select_slider("Communication", options=["Low", "Medium", "High"], value="Medium")
        problem_solving = st.select_slider("Problem Solving", options=["Low", "Medium", "High"], value="High")
    with col3:
        creativity = st.select_slider("Creativity", options=["Low", "Medium", "High"], value="Medium")
        leadership = st.select_slider("Leadership", options=["Low", "Medium", "High"], value="Medium")

    st.divider()
    analyze = st.button("Analyze My Career", type="primary", use_container_width=True)

    if analyze:
        if not interest:
            st.warning("Please select at least one interest.")
            return
        if not skills:
            st.warning("Please select at least one technical skill.")
            return

        try:
            with st.spinner("Analyzing your profile..."):
                recommendations_raw = recommend_careers(
                    education,
                    branch,
                    interest,
                    skills,
                    programming,
                    mathematics,
                    communication,
                    problem_solving,
                    creativity,
                    leadership,
                    industry,
                    experience,
                )
            recommendations = normalize_recommendations(recommendations_raw)
            if not recommendations:
                st.warning("No career recommendations were generated.")
                return
            st.session_state["career_analysis_result"] = recommendations
        except Exception as error:
            st.error("An error occurred while analyzing your profile.")
            st.exception(error)
            return

    recommendations = st.session_state.get("career_analysis_result", [])
    if not recommendations:
        return

    top = recommendations[0]
    st.divider()
    st.success("Career analysis completed successfully.")
    st.header("Your Best Career Match")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(get_career_name(top))
        explanation = get_explanation(top)
        if explanation:
            st.write(explanation)
    with col2:
        st.metric("Match Score", f"{get_score(top):.2f}%")

    st.divider()
    st.subheader("AI Analysis Breakdown")
    score_fields = [
        ("skill_score", "Technical Skills"),
        ("interest_score", "Interests"),
        ("education_score", "Education"),
        ("aptitude_score", "Aptitude"),
        ("industry_score", "Industry"),
        ("nlp_similarity", "NLP Similarity"),
        ("experience_score", "Experience"),
    ]
    score_cols = st.columns(4)
    for index, (field, label) in enumerate(score_fields):
        with score_cols[index % 4]:
            show_score(top, field, label)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matching Skills")
        matching = get_list(top, "matching_skills")
        if matching:
            for skill in matching:
                st.write(f"✓ {str(skill).title()}")
        else:
            st.write("No direct technical skill matches found.")
    with col2:
        st.subheader("Skills to Improve")
        missing = get_list(top, "missing_skills")
        if missing:
            for skill in missing:
                st.write(f"• {str(skill).title()}")
        else:
            st.write("No major technical skill gaps found.")

    salary = get_salary(top)
    if salary:
        st.divider()
        st.subheader("Estimated Salary Information")
        salary_cols = st.columns(3)
        for col, key, label in zip(
            salary_cols,
            ["average", "minimum", "maximum"],
            ["Average Salary", "Minimum Salary", "Maximum Salary"],
        ):
            value = salary.get(key, 0)
            try:
                value_text = f"${float(value):,.0f}"
            except (TypeError, ValueError):
                value_text = "Not available"
            col.metric(label, value_text)

    roadmap = get_list(top, "roadmap")
    if roadmap:
        st.divider()
        st.subheader("Recommended Career Roadmap")
        for number, step in enumerate(roadmap, start=1):
            st.write(f"**Step {number}:** {step}")

    st.divider()
    st.subheader("Top Career Matches")
    display_data = [
        {"Career": get_career_name(item), "Match Score": f"{get_score(item):.2f}%"}
        for item in recommendations[:5]
    ]
    st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================
def about_project_page():
    st.title("About Project")
    st.subheader("AI Career Recommendation System")
    st.write(
        "The AI Career Recommendation System is an intelligent career guidance "
        "application designed to recommend suitable career paths based on an individual's profile."
    )
    st.divider()
    st.subheader("Project Objective")
    st.write(
        "The objective is to help students and job seekers identify suitable career options "
        "using artificial intelligence and data-driven recommendation techniques."
    )
    st.subheader("Technologies Used")
    st.markdown(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - TF-IDF
        - Cosine Similarity
        - Streamlit
        - Machine Learning
        """
    )
    st.subheader("AI Recommendation Method")
    st.write(
        "The system uses a hybrid recommendation approach. Different aspects of the user's "
        "profile are analyzed and combined into a final compatibility score."
    )
    st.markdown(
        """
        **Technical Skills — 30%**

        **Interests — 20%**

        **Education and Branch — 15%**

        **Aptitude — 15%**

        **Industry Preference — 10%**

        **NLP Similarity — 5%**

        **Experience — 5%**
        """
    )
    st.info("The final Match Score represents career compatibility, not model classification accuracy.")


# ============================================================
# ABOUT ME PAGE
# ============================================================
def about_me_page():
    st.title("About Me")
    st.subheader("Anureet Kaur")
    st.write("Bachelor of Technology in Artificial Intelligence and Machine Learning")
    st.write("Jan Nayak Chaudhary Devi Lal Memorial College of Engineering, Sirsa")
    st.write("Academic Session: 2023–2027")
    st.divider()
    st.subheader("My Interest")
    st.write("I am interested in Artificial Intelligence, Machine Learning, Data Science, and intelligent software systems.")
    st.subheader("Project Goal")
    st.write(
        "This project demonstrates how artificial intelligence and machine learning techniques "
        "can provide personalized career recommendations."
    )


# ============================================================
# PAGE ROUTING
# ============================================================
if page == "Home":
    home_page()
elif page == "Career Analysis":
    career_analysis_page()
elif page == "About Project":
    about_project_page()
elif page == "About Me":
    about_me_page()
