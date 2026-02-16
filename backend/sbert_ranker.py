# backend/sbert_ranker.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def sbert_ranking(resumes, job_desc):
    resume_embeddings = model.encode(resumes)
    job_embedding = model.encode([job_desc])
    
    similarities = cosine_similarity(resume_embeddings, job_embedding)
    return similarities.flatten()
