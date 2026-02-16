from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from sbert_ranker import model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
import numpy as np
import pdfplumber
import io
from scoring import hybrid_score

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Backend Running ✅"}


# -----------------------------
# FEEDBACK MODEL
# -----------------------------
class Feedback(BaseModel):
    candidate: str
    decision: str


@app.post("/feedback/")
def save_feedback(data: Feedback):
    print("Feedback received:", data)
    return {"message": "Feedback stored successfully ✅"}


# -----------------------------
# RANK ENDPOINT
# -----------------------------
@app.post("/rank/")
async def rank_resumes(
    job_desc: str = Form(...),
    use_faiss: bool = Form(False),
    use_rerank: bool = Form(False),
    use_fairness: bool = Form(False),
    use_shap: bool = Form(False),
    use_rag: bool = Form(False),
    weight_tfidf: float = Form(0.4),
    weight_sbert: float = Form(0.6),
    files: List[UploadFile] = File(...)
):

    if weight_tfidf + weight_sbert == 0:
        raise HTTPException(status_code=400, detail="Weights cannot both be zero")

    resume_texts = []

    # -----------------------
    # PDF TEXT EXTRACTION
    # -----------------------
    for file in files:
        content = await file.read()
        text = ""

        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception:
            text = ""

        resume_texts.append(text)

    # -----------------------
    # SBERT Similarity
    # -----------------------
    job_embedding = model.encode([job_desc])
    resume_embeddings = model.encode(resume_texts)

    sbert_scores = cosine_similarity(job_embedding, resume_embeddings)[0]

    # -----------------------
    # TF-IDF Similarity
    # -----------------------
    tfidf_scores = np.zeros(len(resume_texts))

    if weight_tfidf > 0:
        vectorizer = TfidfVectorizer(stop_words="english")
        all_docs = [job_desc] + resume_texts
        tfidf_matrix = vectorizer.fit_transform(all_docs)

        job_vec = tfidf_matrix[0:1]
        resume_vecs = tfidf_matrix[1:]

        tfidf_scores = cosine_similarity(job_vec, resume_vecs)[0]

    # -----------------------
    # COMBINED SCORE
    # -----------------------
    final_scores = hybrid_score(
        tfidf_scores,
        sbert_scores,
        resume_texts,
        job_desc
    )
    

    ranked_indices = np.argsort(final_scores)[::-1]

    ranked_results = []
    score_list = []

    for rank, idx in enumerate(ranked_indices, start=1):
        ranked_results.append({
            "rank": rank,
            "filename": files[idx].filename,
            "score": float(final_scores[idx])
        })
        score_list.append(float(final_scores[idx]))

    return {
        "results": ranked_results,
        "scores": score_list,  # ✅ Added for frontend compatibility
        "section_scores": [],
        "explanations": [],
        "shap_values": {},
        "fairness": {}
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

