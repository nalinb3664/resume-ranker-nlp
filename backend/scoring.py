# backend/scoring.py
import numpy as np
import re

# ---- WEIGHT CONFIG ----
WEIGHTS = {
    "university": 0.10,
    "skills": 0.25,
    "experience": 0.30,
    "projects": 0.20,
    "degree": 0.15
}

SIMILARITY_WEIGHT = 0.4
STRUCTURED_WEIGHT = 0.4
KEYWORD_WEIGHT = 0.2

TOP_UNIVERSITIES = [
    "iit", "nit", "mit", "stanford", "oxford",
    "cambridge", "harvard", "imperial college"
]

ADVANCED_DEGREES = [
    "master", "m.tech", "ms", "mba", "phd", "doctorate"
]

TRENDING_TECH = [
    "ai", "machine learning", "deep learning",
    "blockchain", "robotics", "cloud", "cybersecurity"
]

# -----------------------------
# Section scoring helpers
# -----------------------------

def score_university(text):
    return 1 if any(u in text for u in TOP_UNIVERSITIES) else 0

def score_degree(text):
    return 1 if any(d in text for d in ADVANCED_DEGREES) else 0.5

def score_experience(text):
    years = re.findall(r"(\d+)\s+year", text)
    if years:
        years = int(years[0])
        return min(years / 10, 1)
    return 0.2

def score_projects(text):
    count = sum(1 for tech in TRENDING_TECH if tech in text)
    return min(count / 5, 1)

def score_skills(text, job_desc):
    job_keywords = set(job_desc.split())
    resume_keywords = set(text.split())
    match = job_keywords.intersection(resume_keywords)
    if not job_keywords:
        return 0
    return len(match) / len(job_keywords)

# -----------------------------
# Enterprise Keyword Boost
# -----------------------------

def keyword_boost(text, job_desc):
    keywords = job_desc.lower().split()

    if not keywords:
        return 0

    match_count = sum(1 for word in keywords if word in text)
    return match_count / len(keywords)

# -----------------------------
# Must-Have Filter (Enterprise Feature)
# -----------------------------

def must_have_filter(text, job_desc):
    """
    If job description contains words like:
    must:java spring
    required:python
    Those become strict mandatory filters.
    """
    text = text.lower()
    job_desc = job_desc.lower()

    required_patterns = re.findall(r"(?:must|required):([\w\s,]+)", job_desc)

    if not required_patterns:
        return 1  # no strict requirement

    for pattern in required_patterns:
        required_words = [w.strip() for w in pattern.split(",")]
        for word in required_words:
            if word not in text:
                return 0  # reject resume

    return 1
# -----------------------------
# Skill Relevance Guard
# -----------------------------
def is_skill_related(job_desc, resume_texts):
    job_words = job_desc.lower().split()

    if not job_words:
        return False

    # Check if at least one resume contains at least one keyword
    for text in resume_texts:
        text_lower = text.lower()
        for word in job_words:
            if word in text_lower:
                return True

    return False
# -----------------------------
# Hybrid Scoring System
# -----------------------------

def hybrid_score(tfidf_scores, sbert_scores, resume_texts, job_desc):
    
    if not is_skill_related(job_desc, resume_texts):
        # Return zero scores (No valid matching resumes)
        return np.zeros(len(resume_texts))
    final_scores = []
    section_scores = []

    for i, text in enumerate(resume_texts):
        text_lower = text.lower()

        # Structured scores
        university_score = score_university(text_lower)
        degree_score = score_degree(text_lower)
        experience_score = score_experience(text_lower)
        project_score = score_projects(text_lower)
        skill_score = score_skills(text_lower, job_desc.lower())

        total_structured = (
            university_score * WEIGHTS["university"] +
            degree_score * WEIGHTS["degree"] +
            experience_score * WEIGHTS["experience"] +
            project_score * WEIGHTS["projects"] +
            skill_score * WEIGHTS["skills"]
        )

        # Similarity score
        similarity_score = (
            0.4 * tfidf_scores[i] +
            0.6 * sbert_scores[i]
        )

        # Keyword boost
        keyword_score = keyword_boost(text_lower, job_desc)

        # Must-have filtering
        mandatory_pass = must_have_filter(text_lower, job_desc)

        if mandatory_pass == 0:
            final_score = 0  # hard rejection
        else:
            final_score = (
                SIMILARITY_WEIGHT * similarity_score +
                STRUCTURED_WEIGHT * total_structured +
                KEYWORD_WEIGHT * keyword_score
            )

        final_scores.append(final_score)

        section_scores.append({
            "university": round(university_score, 2),
            "degree": round(degree_score, 2),
            "experience": round(experience_score, 2),
            "projects": round(project_score, 2),
            "skills": round(skill_score, 2),
            "keyword": round(keyword_score, 2),
            "similarity": round(similarity_score, 2)
        })

    final_scores = np.array(final_scores)

    # Normalize between 1–100
    min_score = final_scores.min()
    max_score = final_scores.max()

    if max_score - min_score == 0:
        scaled_scores = np.full_like(final_scores, 50)
    else:
        scaled_scores = 1 + (
            (final_scores - min_score) * 99 /
            (max_score - min_score)
        )

    return np.round(scaled_scores, 2)