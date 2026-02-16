# backend/explain.py

def explain_match(resume_text, job_desc):
    matched_words = []
    job_words = set(job_desc.split())
    
    for word in resume_text.split():
        if word in job_words:
            matched_words.append(word)
    
    return list(set(matched_words))[:10]
