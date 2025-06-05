import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("jobs.csv")

st.title("Career Advisor")
skills_input = st.text_input("Enter your skills (comma-separated):")

if skills_input:
    # Convert job skills and user skills to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    job_matrix = vectorizer.fit_transform(df['skills'])
    user_vector = vectorizer.transform([skills_input])
    # Compute cosine similarities
    cosine_sim = cosine_similarity(user_vector, job_matrix).flatten()
    # Find top 3 matching jobs
    top_indices = cosine_sim.argsort()[::-1][:3]
    top_matches = df.iloc[top_indices]

    st.subheader("Top Job Matches")
    st.table(top_matches[['job_title', 'salary', 'demand']])

    action_plans = {
        "Data Scientist": "Learn more Python and practice data projects.",
        "Web Developer": "Build small websites and learn JavaScript."
    }
    st.subheader("Action Plan")
    for job in top_matches['job_title']:
        if job in action_plans:
            st.write(f"**For {job}:** {action_plans[job]}")

