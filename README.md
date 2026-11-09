# AI Career Recommendation System

## Project Overview
The AI Career Recommendation System is a machine learning-based web application developed using Python and Streamlit. It helps students and job seekers discover suitable career paths based on their skills, interests, educational background, and aptitude.

The system uses Natural Language Processing (NLP), TF-IDF Vectorization, Cosine Similarity, and a Hybrid Recommendation Approach to generate personalized career recommendations with match scores.

## Objectives
- Provide personalized career recommendations.
- Analyze users' skills, interests, and educational qualifications.
- Apply machine learning and NLP techniques for career matching.
- Help students make informed career decisions.
- Calculate career match scores using multiple factors.

## Key Features
- User-friendly Streamlit interface.
- Skill-based career recommendations.
- Interest and education analysis.
- Aptitude-based matching.
- TF-IDF text vectorization.
- Cosine Similarity calculation.
- Hybrid recommendation scoring.
- Career match percentages.
- Estimated salary information.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Streamlit | Web Application |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| TF-IDF | Text Vectorization |
| Cosine Similarity | Text Similarity |
| NLP | Text Analysis |
| Matplotlib & Seaborn | Data Visualization |

## How It Works

1. User enters skills, interests, education, and aptitude information.
2. The system processes the user's profile.
3. TF-IDF converts textual information into numerical vectors.
4. Cosine Similarity measures similarity between user profiles and career descriptions.
5. The Hybrid Recommendation Engine combines multiple factors.
6. The system calculates final match scores.
7. Suitable careers are displayed in ranked order.

## Recommendation Methodology

The project uses a hybrid approach combining:

- Skills Matching
- Interest Matching
- Education Matching
- Aptitude Matching
- Industry Preference
- NLP-based Similarity

These factors contribute to the final career recommendation score.

## Project Structure

```text
AI_Career_Recommendation_System/
│
├── app/
├── data/
├── models/
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

## Installation and Setup


**Step 1: Navigate to the project folder**

```bash
cd AI_Career_Recommendation_System
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Run the Streamlit application**

```bash
streamlit run app/app.py
```

## Project Applications
- Student career guidance
- Career planning and exploration
- Skill-based job recommendations
- Educational and professional development

## Future Enhancements
- Integration with real-time job market data.
- Resume analysis and skill extraction.
- More career categories and job roles.
- Personalized learning and skill development paths.
- Integration with job portals.

## License
This project is developed for academic and educational purposes.
