# backend/tfidf_ranker.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_ranking(resumes, job_desc):
    documents = resumes + [job_desc]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    job_vector = tfidf_matrix[-1]
    resume_vectors = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(resume_vectors, job_vector)
    
    return similarities.flatten()
