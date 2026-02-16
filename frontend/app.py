import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="NLP Ranker",
    page_icon="🚀",
    layout="wide"
)

# ------------------ STYLING ------------------
st.markdown("""
<style>
.big-font {
    font-size:28px !important;
    font-weight:600;
}
.stButton>button {
    background-color:#2563EB;
    color:white;
    border-radius:10px;
    padding:10px 20px;
}
.metric-box {
    padding:20px;
    border-radius:12px;
    background-color:#111827;
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">NLP Ranker — Enterprise Version</p>', unsafe_allow_html=True)
st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Ranking Configuration")

use_faiss = st.sidebar.toggle("Enable FAISS Vector Search", value=True)
use_rerank = st.sidebar.toggle("Enable Cross-Encoder Re-ranking", value=True)
use_fairness = st.sidebar.toggle("Run Fairness Audit (Fairlearn)", value=False)
use_shap = st.sidebar.toggle("Show SHAP Explainability", value=True)
use_rag = st.sidebar.toggle("Enable GPT RAG Explanation", value=True)

weight_tfidf = st.sidebar.slider("TF-IDF Weight", 0.0, 1.0, 0.4)
weight_sbert = st.sidebar.slider("SBERT Weight", 0.0, 1.0, 0.6)

st.sidebar.divider()
st.sidebar.info("Backend must support these features.")

# ------------------ INPUT SECTION ------------------
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area("📄 Paste Job Description", height=100)

with col2:
    uploaded_files = st.file_uploader(
        "📂 Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

# Resume counter
if uploaded_files:
    st.info(f"📊 {len(uploaded_files)} resumes uploaded.")
    if len(uploaded_files) > 200:
        st.warning("Large batch detected. Processing may take longer.")

st.divider()

# ------------------ RANK BUTTON ------------------
if st.button("🔎 Rank Candidates", use_container_width=True):

    if not uploaded_files or not job_desc:
        st.warning("Please upload resumes and enter job description.")
        st.stop()

    progress_bar = st.progress(0)

    payload = {
        "job_desc": job_desc,
        "use_faiss": use_faiss,
        "use_rerank": use_rerank,
        "use_fairness": use_fairness,
        "use_shap": use_shap,
        "use_rag": use_rag,
        "weight_tfidf": weight_tfidf,
        "weight_sbert": weight_sbert
    }

    batch_size = 50
    all_results = []

    progress_step = 100 / (len(uploaded_files) // batch_size + 1)
    progress = 0

    with st.spinner("Ranking in progress... 🤖"):

        for i in range(0, len(uploaded_files), batch_size):
            batch = uploaded_files[i:i+batch_size]
            files = [("files", file) for file in batch]

            response = requests.post(
                "http://127.0.0.1:8000/rank/",
                data=payload,
                files=files
            )

            batch_results = response.json().get("results", [])
            all_results.extend(batch_results)

            progress += progress_step
            progress_bar.progress(min(int(progress), 100))

    progress_bar.progress(100)
    st.success("Ranking Completed ✅")

    if not all_results:
        st.error("No results returned from backend.")
        st.stop()

    # Final ranking
    df = pd.DataFrame(all_results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # ------------------ FILTER ------------------
    st.subheader("🎯 Filter Candidates")
    min_score = st.slider("Minimum Score Filter", 0.0, 1.0, 0.0)
    filtered_df = df[df["score"] >= min_score]

    # ------------------ RESULTS ------------------
    st.subheader("🏆 Ranked Candidates")
    st.dataframe(filtered_df, use_container_width=True)

    # ------------------ TOP 10 ------------------
    st.subheader("🔥 Top 10 Candidates")
    top10 = df.head(10)
    st.dataframe(top10, use_container_width=True)

    # ------------------ SCORE CHART ------------------
    st.subheader("📊 Score Distribution")

    fig, ax = plt.subplots()
    ax.bar(top10["filename"], top10["score"])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ------------------ EXPORT CSV ------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Rankings as CSV",
        csv,
        "ranked_candidates.csv",
        "text/csv"
    )

    # ------------------ FEEDBACK ------------------
    st.subheader("📝 Recruiter Feedback")

    selected_candidate = st.selectbox(
        "Select Candidate",
        df["filename"]
    )

    feedback = st.radio(
        "Decision",
        ["Shortlist", "Reject"]
    )

    if st.button("Submit Feedback"):
        feedback_payload = {
            "candidate": selected_candidate,
            "decision": feedback
        }

        requests.post(
            "http://127.0.0.1:8000/feedback/",
            json=feedback_payload
        )

        st.success("Feedback Saved Successfully ✅")

    st.divider()
    st.info("System uses Hybrid Ranking + Semantic Search.")