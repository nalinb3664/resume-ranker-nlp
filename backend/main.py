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
import json
import os
from datetime import datetime
from fastapi.responses import HTMLResponse
import sqlite3
from faiss_store import load_index, save_index, create_index
import faiss
import pandas as pd
import shap
from xgboost import XGBRegressor

# -----------------------------
# DATABASE
# -----------------------------
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_description TEXT,
    filename TEXT,
    score REAL,
    rank INTEGER,
    timestamp TEXT
)
""")
conn.commit()

# -----------------------------
# MODELS (FIXED POSITION)
# -----------------------------
class SelectedProfile(BaseModel):
    filename: str
    score: float
    rank: int


class SelectionRequest(BaseModel):
    job_description: str
    selected_profiles: List[SelectedProfile]


class Feedback(BaseModel):
    candidate: str
    decision: str
    score: float


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend Running ✅"}

# -----------------------------
# SAVE SELECTION (ONLY ONE)
# -----------------------------
@app.post("/save-selection/")
def save_selection(data: SelectionRequest):

    if not data.selected_profiles:
        return {"error": "No candidates selected ❌"}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for profile in data.selected_profiles:
        cursor.execute("""
            INSERT INTO selections (job_description, 
            filename, 
            score,
                        rank, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data.job_description,
            profile.filename,
            profile.score,
            profile.rank,
            timestamp
        ))

    conn.commit()

    return {
        "message": "Saved to database ✅",
        "count": len(data.selected_profiles)
    }

# -----------------------------
# GET SELECTIONS
# -----------------------------
@app.get("/get-selections/")
def get_selections():

    cursor.execute("""
        SELECT job_description, 
        filename, 
        score, 
        rank, timestamp
        FROM selections
        ORDER BY score DESC
    """)

    rows = cursor.fetchall()

    data = []
    for row in rows:
        data.append({
            "job_description": row[0],
            "filename": row[1],
            "score": row[2],
            "rank": row[3],
            "timestamp": row[4]
        })

    return {"data": data}

# -----------------------------
# FILTER BY JOB
# -----------------------------
@app.get("/filter-job/")
def filter_job(job: str):

    cursor.execute("""
        SELECT filename, score, rank
        FROM selections
        WHERE job_description = ?
        ORDER BY score DESC
    """, (job,))

    rows = cursor.fetchall()

    return [
        {"filename": r[0], "score": r[1], "rank": r[2]}
        for r in rows
    ]

# -----------------------------
# FEEDBACK
# -----------------------------
@app.post("/feedback/")
def save_feedback(data: Feedback):
    file_path = "feedback_data.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
            except:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(data.dict())

    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    return {"message": "Feedback stored successfully ✅"}

# -----------------------------
# RANK ENDPOINT
# -----------------------------
def extract_features(job_desc, resume_text, tfidf_score, sbert_score):
    keywords = job_desc.lower().split()

    overlap = sum(1 for word in keywords if word in resume_text.lower())

    return {
        "tfidf_score": float(tfidf_score),
        "sbert_score": float(sbert_score),
        "keyword_overlap": overlap,
        "resume_length": len(resume_text.split())
    }

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

    # -----------------------------
    # EMBEDDINGS
    # -----------------------------
    job_embedding = model.encode([job_desc]).astype("float32")
    resume_embeddings = model.encode(resume_texts).astype("float32")

    # Normalize for cosine similarity in FAISS
    faiss.normalize_L2(job_embedding)
    faiss.normalize_L2(resume_embeddings)

    # -----------------------------
    # FAISS OR COSINE SEARCH
    # -----------------------------
    if use_faiss:
        index, metadata = load_index()

        # Create fresh index for current batch
        index = create_index()
        index.add(resume_embeddings)

        # Save current batch index
        save_index(index, [{"filename": f.filename} for f in files])

        distances, indices = index.search(job_embedding, len(resume_texts))

        sbert_scores = np.zeros(len(resume_texts))

        for score, idx in zip(distances[0], indices[0]):
            sbert_scores[idx] = score

    else:
        sbert_scores = cosine_similarity(job_embedding, resume_embeddings)[0]

    print("FAISS Enabled:", use_faiss)

    # -----------------------------
    # TF-IDF
    # -----------------------------
    tfidf_scores = np.zeros(len(resume_texts))

    if weight_tfidf > 0:
        vectorizer = TfidfVectorizer(stop_words="english")
        all_docs = [job_desc] + resume_texts
        tfidf_matrix = vectorizer.fit_transform(all_docs)

        job_vec = tfidf_matrix[0:1]
        resume_vecs = tfidf_matrix[1:]

        tfidf_scores = cosine_similarity(job_vec, resume_vecs)[0]

    # -----------------------------
    # FINAL SCORING
    # -----------------------------
    if use_faiss:
        final_scores = sbert_scores
    else:
        final_scores = hybrid_score(
            tfidf_scores,
            sbert_scores,
            resume_texts,
            job_desc
        )
    # -----------------------------
    # SHAP EXPLAINABILITY
    # -----------------------------
    shap_output = []

    if use_shap:
        feature_list = []

        for i in range(len(resume_texts)):
            features = extract_features(
                job_desc,
                resume_texts[i],
                tfidf_scores[i],
                sbert_scores[i]
            )
            feature_list.append(features)

        X = pd.DataFrame(feature_list)
        y = final_scores

        explain_model = XGBRegressor()
        explain_model.fit(X, y)

        explainer = shap.Explainer(explain_model)
        shap_values = explainer(X)

        for i in range(len(X)):
            shap_output.append({
                "candidate": files[i].filename,
                "features": list(X.columns),
                "values": shap_values.values[i].tolist()
            })

    # -----------------------------
    # RANK RESULTS
    # -----------------------------
    ranked_indices = np.argsort(final_scores)[::-1]

    ranked_results = []
    score_list = []

    for rank, idx in enumerate(ranked_indices, start=1):
        ranked_results.append({
            "rank": rank,
            "filename": files[idx].filename,
            "score": float(final_scores[idx]),
            "content": resume_texts[idx]
        })
        score_list.append(float(final_scores[idx]))

    return {
        "results": ranked_results,
        "scores": score_list,
        "section_scores": [],
        "explanations": [],
        "shap_values": shap_output,
        "fairness": {}
    }
if __name__ == "__main__":
 import uvicorn
 uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():

    cursor.execute("""
        SELECT job_description, filename, score, rank, timestamp
        FROM selections
        ORDER BY score DESC
    """)
    rows = cursor.fetchall()

    flat_data = []
    for row in rows:
        flat_data.append({
            "job": row[0],
            "filename": row[1],
            "score": row[2],
            "rank": row[3],
            "time": row[4]
        })

    json_data = json.dumps(flat_data)

    return f"""
    <html>
    <head>
        <title>Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>

    <h2>📊 Candidate Dashboard</h2>

    <table border="1">
        <tr>
            <th>Job</th>
            <th>Filename</th>
            <th>Score</th>
            <th>Rank</th>
            <th>Time</th>
        </tr>
        {''.join([f"<tr><td>{d['job']}</td><td>{d['filename']}</td><td>{d['score']}</td><td>{d['rank']}</td><td>{d['time']}</td></tr>" for d in flat_data])}
    </table>

    <canvas id="chart"></canvas>

    <script>
    let data = {json_data};
    let labels = data.map(d => d.filename);
    let scores = data.map(d => d.score);

    new Chart(document.getElementById("chart"), {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [{{
                label: 'Score',
                data: scores
            }}]
        }}
    }});
    </script>

    </body>
    </html>
    """
@app.delete("/delete/")
def delete_candidate(filename: str):

    cursor.execute("DELETE FROM selections WHERE filename = ?", (filename,))
    conn.commit()

    return {"message": "Deleted from database ✅"}