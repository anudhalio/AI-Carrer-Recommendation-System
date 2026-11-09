import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\shiva\Desktop\AI_Career_Recommendation_System"

CAREER_KB_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "career_knowledge_base.csv"
)

JOBS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed_jobs.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

career_df = pd.read_csv(CAREER_KB_PATH)
jobs_df = pd.read_csv(JOBS_PATH)


# ============================================================
# CAREER ROADMAPS
# ============================================================

ROADMAPS = {

    "AI / Machine Learning Engineer": [
        "Learn Python programming",
        "Strengthen Mathematics and Statistics",
        "Learn Machine Learning algorithms",
        "Learn Deep Learning",
        "Practice TensorFlow or PyTorch",
        "Build AI / ML projects",
        "Learn model deployment and APIs",
        "Complete internship or real-world project",
        "Apply for AI / ML Engineer roles"
    ],

    "Data Scientist": [
        "Learn Python and SQL",
        "Strengthen Statistics and Mathematics",
        "Learn Data Analysis",
        "Learn Machine Learning",
        "Practice Data Visualization",
        "Build Data Science projects",
        "Learn advanced ML techniques",
        "Create a strong portfolio",
        "Apply for Data Scientist roles"
    ],

    "Data Analyst": [
        "Learn Excel",
        "Learn SQL",
        "Learn Statistics",
        "Learn Python for Data Analysis",
        "Learn Power BI or visualization tools",
        "Practice data cleaning",
        "Build analytics projects",
        "Create a portfolio",
        "Apply for Data Analyst roles"
    ],

    "Software Developer": [
        "Strengthen programming fundamentals",
        "Learn Data Structures and Algorithms",
        "Learn Git and GitHub",
        "Learn software development practices",
        "Build real-world applications",
        "Learn APIs and databases",
        "Practice problem solving",
        "Complete internships or projects",
        "Apply for Software Developer roles"
    ],

    "Web Developer": [
        "Learn HTML and CSS",
        "Learn JavaScript",
        "Learn React",
        "Learn backend development",
        "Learn databases",
        "Build responsive websites",
        "Build full-stack projects",
        "Deploy projects online",
        "Apply for Web Developer roles"
    ],

    "Cloud / DevOps Engineer": [
        "Learn Linux fundamentals",
        "Learn networking basics",
        "Learn Git and GitHub",
        "Learn AWS or another cloud platform",
        "Learn Docker",
        "Learn CI/CD",
        "Learn Kubernetes basics",
        "Build cloud deployment projects",
        "Apply for Cloud / DevOps roles"
    ],

    "Cybersecurity Analyst": [
        "Learn computer networking",
        "Learn Linux",
        "Understand cybersecurity fundamentals",
        "Learn security tools",
        "Practice ethical hacking concepts",
        "Learn vulnerability assessment",
        "Study security monitoring",
        "Complete cybersecurity projects",
        "Apply for Cybersecurity Analyst roles"
    ],

    "Financial Analyst": [
        "Strengthen Mathematics",
        "Learn Excel",
        "Learn Financial Modeling",
        "Understand financial statements",
        "Learn statistics",
        "Practice financial analysis",
        "Learn data visualization",
        "Build finance-related projects",
        "Apply for Financial Analyst roles"
    ],

    "Digital Marketing Specialist": [
        "Learn digital marketing fundamentals",
        "Learn SEO",
        "Learn Google Ads",
        "Learn social media marketing",
        "Learn content marketing",
        "Practice market research",
        "Analyze campaign performance",
        "Build marketing campaigns",
        "Apply for Digital Marketing roles"
    ],

    "Business Analyst": [
        "Learn business fundamentals",
        "Improve communication skills",
        "Learn Excel and SQL",
        "Learn data analysis",
        "Develop problem-solving skills",
        "Learn business process analysis",
        "Practice requirement gathering",
        "Build business case studies",
        "Apply for Business Analyst roles"
    ],

    "Research Scientist": [
        "Strengthen Mathematics and Statistics",
        "Learn Python",
        "Develop research methodology",
        "Learn Data Analysis",
        "Study Machine Learning",
        "Read research papers",
        "Conduct independent research",
        "Publish or present research work",
        "Apply for Research roles"
    ]
}


# ============================================================
# EXPERIENCE SCORES
# ============================================================

EXPERIENCE_SCORE = {
    "Student": 100,
    "Fresher": 100,
    "0-1 Years": 90,
    "1-3 Years": 80,
    "3-5 Years": 70,
    "5+ Years": 60
}


# ============================================================
# LEVEL SCORES
# ============================================================

LEVEL_SCORE = {
    "Low": 40,
    "Medium": 70,
    "High": 100
}


# ============================================================
# SKILL NORMALIZATION
# ============================================================

SKILL_MAPPING = {

    "ml": "machine learning",
    "machinelearning": "machine learning",

    "ai": "artificial intelligence",
    "artificialintelligence": "artificial intelligence",

    "ds": "data science",
    "datascience": "data science",

    "da": "data analytics",
    "dataanalysis": "data analytics",

    "js": "javascript",

    "reactjs": "react",

    "powerbi": "power bi",

    "gcp": "google cloud"
}


def normalize_skill(skill):
    """
    Converts different forms of the same skill
    into one standard format.
    """

    if pd.isna(skill):
        return ""

    skill = str(skill).strip().lower()

    skill = re.sub(r"[_\-]+", " ", skill)
    skill = re.sub(r"\s+", " ", skill)

    compact = skill.replace(" ", "")

    if compact in SKILL_MAPPING:
        return SKILL_MAPPING[compact]

    if skill in SKILL_MAPPING:
        return SKILL_MAPPING[skill]

    return skill


def normalize_skills(skills):

    if skills is None:
        return []

    if isinstance(skills, str):

        # Handle strings such as:
        # Python, SQL, Machine Learning

        if "," in skills:
            skills = skills.split(",")

        elif "|" in skills:
            skills = skills.split("|")

        else:
            skills = [skills]

    normalized = []

    for skill in skills:

        value = normalize_skill(skill)

        if value:
            normalized.append(value)

    return list(dict.fromkeys(normalized))


# ============================================================
# GET CAREER INFORMATION
# ============================================================

def get_career_info(career):

    result = career_df[
        career_df["career"].str.lower() == career.lower()
    ]

    if result.empty:
        return None

    return result.iloc[0]


# ============================================================
# TECHNICAL SKILL MATCHING
# ============================================================

def calculate_skill_score(user_skills, career_skills):
    """
    Calculates technical skill compatibility.

    IMPORTANT:
    Mathematics, communication, programming, creativity,
    leadership and problem solving are NOT treated as
    technical skills here.

    They are handled separately by the aptitude module.
    """

    aptitude_only = {
        "mathematics",
        "communication",
        "programming",
        "problem solving",
        "creativity",
        "leadership"
    }

    user_set = set(
        normalize_skills(user_skills)
    )

    career_skill_list = normalize_skills(
        career_skills
    )

    # Remove aptitude-related items
    career_skill_list = [
        skill
        for skill in career_skill_list
        if skill not in aptitude_only
    ]

    career_set = set(career_skill_list)

    if not career_set:
        return 0.0, [], []

    matching = sorted(
        user_set.intersection(career_set)
    )

    missing = sorted(
        career_set - user_set
    )

    score = (
        len(matching) /
        len(career_set)
    ) * 100

    return round(score, 2), matching, missing


# ============================================================
# INTEREST MATCHING
# ============================================================

def calculate_interest_score(
    user_interests,
    career_interests
):

    user_interests = normalize_skills(
        user_interests
    )

    career_interests = normalize_skills(
        career_interests
    )

    if not career_interests:
        return 0.0

    user_set = set(user_interests)
    career_set = set(career_interests)

    matching = user_set.intersection(
        career_set
    )

    score = (
        len(matching) /
        len(career_set)
    ) * 100

    return round(score, 2)


# ============================================================
# EDUCATION / BRANCH MATCHING
# ============================================================

def calculate_education_score(
    education,
    branch,
    career_education
):

    education_text = str(
        career_education
    ).lower()

    education = str(
        education
    ).lower()

    branch = str(
        branch
    ).lower()

    branch_mapping = {

        "aiml": [
            "aiml",
            "cse",
            "it"
        ],

        "cse": [
            "cse",
            "it",
            "aiml"
        ],

        "it": [
            "it",
            "cse",
            "aiml"
        ],

        "ece": [
            "ece",
            "cse",
            "it"
        ],

        "mechanical": [
            "mechanical"
        ],

        "civil": [
            "civil"
        ],

        "electrical": [
            "electrical"
        ],

        "commerce": [
            "commerce"
        ],

        "general": []
    }

    # First check branch compatibility
    compatible_branches = branch_mapping.get(
        branch,
        []
    )

    branch_match = any(
        item in education_text
        for item in compatible_branches
    )

    # Check education degree
    degree_match = False

    education_keywords = [
        education,
        "b.tech",
        "btech",
        "m.tech",
        "mtech",
        "b.sc",
        "bsc",
        "mca",
        "mba"
    ]

    for keyword in education_keywords:

        if keyword and keyword in education_text:
            degree_match = True
            break

    if branch_match and degree_match:
        return 100.0

    if branch_match:
        return 90.0

    if degree_match:
        return 70.0

    return 40.0


# ============================================================
# APTITUDE MATCHING
# ============================================================

def calculate_aptitude_score(
    programming,
    mathematics,
    communication,
    problem_solving,
    creativity,
    leadership,
    career_info
):

    user_levels = {

        "programming": LEVEL_SCORE.get(
            programming,
            70
        ),

        "mathematics": LEVEL_SCORE.get(
            mathematics,
            70
        ),

        "communication": LEVEL_SCORE.get(
            communication,
            70
        ),

        "problem solving": LEVEL_SCORE.get(
            problem_solving,
            70
        ),

        "creativity": LEVEL_SCORE.get(
            creativity,
            70
        ),

        "leadership": LEVEL_SCORE.get(
            leadership,
            70
        )
    }

    career_levels = {

        "programming": str(
            career_info.get(
                "programming_level",
                "Medium"
            )
        ),

        "mathematics": str(
            career_info.get(
                "mathematics_level",
                "Medium"
            )
        ),

        "communication": str(
            career_info.get(
                "communication_level",
                "Medium"
            )
        ),

        "problem solving": str(
            career_info.get(
                "problem_solving_level",
                "Medium"
            )
        ),

        "creativity": str(
            career_info.get(
                "creativity_level",
                "Medium"
            )
        ),

        "leadership": str(
            career_info.get(
                "leadership_level",
                "Medium"
            )
        )
    }

    scores = []

    for skill in user_levels:

        user_score = user_levels[skill]

        required_score = LEVEL_SCORE.get(
            career_levels[skill],
            70
        )

        # Full score if user meets requirement
        if user_score >= required_score:
            score = 100

        else:

            score = (
                user_score /
                required_score
            ) * 100

        scores.append(score)

    if not scores:
        return 0.0

    return round(
        np.mean(scores),
        2
    )


# ============================================================
# INDUSTRY MATCHING
# ============================================================

def calculate_industry_score(
    preferred_industry,
    career_industries
):

    if not preferred_industry:
        return 50.0

    preferred = str(
        preferred_industry
    ).lower()

    industries = normalize_skills(
        career_industries
    )

    if not industries:
        return 50.0

    for industry in industries:

        if preferred in industry or industry in preferred:
            return 100.0

    return 0.0


# ============================================================
# EXPERIENCE MATCHING
# ============================================================

def calculate_experience_score(
    experience
):

    return EXPERIENCE_SCORE.get(
        experience,
        70
    )


# ============================================================
# CREATE CAREER DOCUMENTS FOR NLP
# ============================================================

def create_career_document(row):

    parts = [

        str(row.get("career", "")),

        str(row.get("category", "")),

        str(row.get("industries", "")),

        str(row.get("skills", "")),

        str(row.get("interests", "")),

        str(row.get("education", "")),

        str(row.get("programming_level", "")),

        str(row.get("mathematics_level", "")),

        str(row.get("communication_level", "")),

        str(row.get("problem_solving_level", "")),

        str(row.get("creativity_level", "")),

        str(row.get("leadership_level", ""))
    ]

    return " ".join(parts)


career_documents = career_df.apply(
    create_career_document,
    axis=1
)


# ============================================================
# TF-IDF MODEL
# ============================================================

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

career_matrix = tfidf_vectorizer.fit_transform(
    career_documents
)


# ============================================================
# NLP SIMILARITY
# ============================================================

def calculate_nlp_similarity(
    education,
    branch,
    interests,
    skills,
    programming,
    mathematics,
    communication,
    problem_solving,
    creativity,
    leadership,
    preferred_industry,
    experience
):

    user_profile = " ".join([

        str(education),

        str(branch),

        " ".join(
            normalize_skills(interests)
        ),

        " ".join(
            normalize_skills(skills)
        ),

        str(programming),

        str(mathematics),

        str(communication),

        str(problem_solving),

        str(creativity),

        str(leadership),

        str(preferred_industry),

        str(experience)
    ])

    user_vector = tfidf_vectorizer.transform(
        [user_profile]
    )

    similarities = cosine_similarity(
        user_vector,
        career_matrix
    )[0]

    return similarities * 100


# ============================================================
# SALARY INFORMATION
# ============================================================

def get_salary_info(career):

    if jobs_df.empty:
        return {
            "average": 0,
            "minimum": 0,
            "maximum": 0
        }

    career_words = [
        word.lower()
        for word in re.findall(
            r"[a-zA-Z]+",
            career
        )
        if len(word) > 2
    ]

    if not career_words:
        return {
            "average": float(
                jobs_df["salary"].mean()
            ),
            "minimum": float(
                jobs_df["salary"].min()
            ),
            "maximum": float(
                jobs_df["salary"].max()
            )
        }

    job_titles = jobs_df[
        "job_title"
    ].astype(str).str.lower()

    mask = pd.Series(
        False,
        index=jobs_df.index
    )

    for word in career_words:

        mask = (
            mask |
            job_titles.str.contains(
                word,
                regex=False,
                na=False
            )
        )

    matched_jobs = jobs_df[
        mask
    ]

    if matched_jobs.empty:
        matched_jobs = jobs_df

    return {
        "average": float(
            matched_jobs["salary"].mean()
        ),

        "minimum": float(
            matched_jobs["salary"].min()
        ),

        "maximum": float(
            matched_jobs["salary"].max()
        )
    }


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(
    row,
    matching_skills
):

    reasons = []

    if row["skill_score"] >= 70:
        reasons.append(
            "your technical skills strongly match this career"
        )

    elif row["skill_score"] >= 40:
        reasons.append(
            "your technical skills have a reasonable match with this career"
        )

    if row["interest_score"] >= 60:
        reasons.append(
            "your interests are well aligned with this career"
        )

    if row["education_score"] >= 80:
        reasons.append(
            "your education and branch are suitable for this career"
        )

    if row["aptitude_score"] >= 80:
        reasons.append(
            "your aptitude profile is highly suitable"
        )

    if row["industry_score"] == 100:
        reasons.append(
            "your preferred industry matches this career"
        )

    if not reasons:
        reasons.append(
            "this career has the strongest overall compatibility with your profile"
        )

    explanation = (
        "This career is recommended because "
        + "; ".join(reasons)
        + "."
    )

    return explanation


# ============================================================
# RECOMMEND CAREERS
# ============================================================

def recommend_careers(
    education,
    branch,
    interests,
    skills,
    programming,
    mathematics,
    communication,
    problem_solving,
    creativity,
    leadership,
    preferred_industry,
    experience
):

    nlp_scores = calculate_nlp_similarity(

        education,

        branch,

        interests,

        skills,

        programming,

        mathematics,

        communication,

        problem_solving,

        creativity,

        leadership,

        preferred_industry,

        experience
    )

    results = []

    for index, career_row in career_df.iterrows():

        career = career_row["career"]

        # ----------------------------------------------------
        # SKILLS
        # ----------------------------------------------------

        skill_score, matching_skills, missing_skills = (
            calculate_skill_score(
                skills,
                career_row.get(
                    "skills",
                    ""
                )
            )
        )

        # ----------------------------------------------------
        # INTERESTS
        # ----------------------------------------------------

        interest_score = calculate_interest_score(

            interests,

            career_row.get(
                "interests",
                ""
            )
        )

        # ----------------------------------------------------
        # EDUCATION
        # ----------------------------------------------------

        education_score = calculate_education_score(

            education,

            branch,

            career_row.get(
                "education",
                ""
            )
        )

        # ----------------------------------------------------
        # APTITUDE
        # ----------------------------------------------------

        aptitude_score = calculate_aptitude_score(

            programming,

            mathematics,

            communication,

            problem_solving,

            creativity,

            leadership,

            career_row
        )

        # ----------------------------------------------------
        # INDUSTRY
        # ----------------------------------------------------

        industry_score = calculate_industry_score(

            preferred_industry,

            career_row.get(
                "industries",
                ""
            )
        )

        # ----------------------------------------------------
        # NLP
        # ----------------------------------------------------

        nlp_score = float(
            nlp_scores[index]
        )

        # ----------------------------------------------------
        # EXPERIENCE
        # ----------------------------------------------------

        experience_score = calculate_experience_score(
            experience
        )

        # ----------------------------------------------------
        # FINAL WEIGHTED SCORE
        # ----------------------------------------------------

        final_score = (

            skill_score * 0.30 +

            interest_score * 0.20 +

            education_score * 0.15 +

            aptitude_score * 0.15 +

            industry_score * 0.10 +

            nlp_score * 0.05 +

            experience_score * 0.05
        )

        result = {

            "career": career,

            "category": career_row.get(
                "category",
                ""
            ),

            "final_score": round(
                final_score,
                2
            ),

            "skill_score": round(
                skill_score,
                2
            ),

            "interest_score": round(
                interest_score,
                2
            ),

            "education_score": round(
                education_score,
                2
            ),

            "aptitude_score": round(
                aptitude_score,
                2
            ),

            "industry_score": round(
                industry_score,
                2
            ),

            "nlp_similarity": round(
                nlp_score,
                2
            ),

            "experience_score": round(
                experience_score,
                2
            ),

            "matching_skills": matching_skills,

            "missing_skills": missing_skills,

            "salary": get_salary_info(
                career
            ),

            "roadmap": ROADMAPS.get(
                career,
                []
            )
        }

        results.append(result)

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # ADD EXPLANATIONS
    # --------------------------------------------------------

    result_df["explanation"] = result_df.apply(

        lambda row: generate_explanation(
            row,
            row["matching_skills"]
        ),

        axis=1
    )

    # --------------------------------------------------------
    # SORT BY FINAL SCORE
    # --------------------------------------------------------

    result_df = result_df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# GET TOP RECOMMENDATION
# ============================================================

def get_top_recommendation(
    education,
    branch,
    interests,
    skills,
    programming,
    mathematics,
    communication,
    problem_solving,
    creativity,
    leadership,
    preferred_industry,
    experience
):

    results = recommend_careers(

        education=education,

        branch=branch,

        interests=interests,

        skills=skills,

        programming=programming,

        mathematics=mathematics,

        communication=communication,

        problem_solving=problem_solving,

        creativity=creativity,

        leadership=leadership,

        preferred_industry=preferred_industry,

        experience=experience
    )

    if results.empty:
        return None

    return results.iloc[0].to_dict()


# ============================================================
# BACKEND TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)

    print(
        "AI CAREER RECOMMENDATION ENGINE - BACKEND TEST"
    )

    print("=" * 80)

    test_result = recommend_careers(

        education="B.Tech",

        branch="AIML",

        interests=[
            "Artificial Intelligence",
            "Machine Learning",
            "Data Science"
        ],

        skills=[
            "Python",
            "SQL",
            "Machine Learning",
            "Research"
        ],

        programming="High",

        mathematics="High",

        communication="Medium",

        problem_solving="High",

        creativity="Medium",

        leadership="Medium",

        preferred_industry="Software",

        experience="Student"
    )

    print(
        test_result[
            [
                "career",
                "final_score",
                "skill_score",
                "interest_score",
                "education_score",
                "aptitude_score",
                "industry_score",
                "nlp_similarity"
            ]
        ].head(5)
    )

    print()

    top = get_top_recommendation(

        education="B.Tech",

        branch="AIML",

        interests=[
            "Artificial Intelligence",
            "Machine Learning",
            "Data Science"
        ],

        skills=[
            "Python",
            "SQL",
            "Machine Learning",
            "Research"
        ],

        programming="High",

        mathematics="High",

        communication="Medium",

        problem_solving="High",

        creativity="Medium",

        leadership="Medium",

        preferred_industry="Software",

        experience="Student"
    )

    print(
        "TOP CAREER:",
        top["career"]
    )

    print(
        "MATCH SCORE:",
        top["final_score"]
    )

    print(
        "MATCHING TECHNICAL SKILLS:",
        top["matching_skills"]
    )

    print(
        "TECHNICAL SKILLS TO IMPROVE:",
        top["missing_skills"]
    )

    print(
        "SALARY:",
        top["salary"]
    )

    print(
        "ROADMAP:"
    )

    for step in top["roadmap"]:
        print(
            " -",
            step
        )

    print("=" * 80)