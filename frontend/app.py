import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import re
from cities_db import CITIES

# ------------------ citieslocation ------------------
@st.cache_data
def extract_location(resume_text):
    resume_text_lower = resume_text.lower()

    for city in CITIES:
        if city.lower() in resume_text_lower:
            return city
    return "Unknown"

def extract_location_advanced(text):
    if not text:
        return "Unknown"

    text_lower = text.lower()

    # Direct city search anywhere in text
    for city in CITIES:
        if city.lower() in text_lower:
            return city

    # Regex search for location/address lines
    patterns = [
        r"location[:\-]?\s*([A-Za-z,\s]+)",
        r"address[:\-]?\s*([A-Za-z,\s]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found = match.group(1).strip()

            for city in CITIES:
                if city.lower() in found.lower():
                    return city

    return "Unknown"
# ------------------ PAGE CONFIG (MUST BE FIRST) ------------------
st.set_page_config(
    page_title="NLP Ranker",
    page_icon="🚀",
    layout="wide"
)

# ------------------ STYLE FUNCTION ------------------
def color_score(val):
    if val > 80:
        return 'color: #22c55e'
    elif val > 60:
        return 'color: #facc15'
    else:
        return 'color: #ef4444'

# ------------------ HEADER ------------------
st.markdown("""
<h1 style='text-align:center;color:#60A5FA;'>
🚀 LeL NLP Resume Ranker 
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    background: rgba(0,255,17,0.05);
    padding: 40px;
    border-radius: 30px;
    backdrop-filter: blur(10px);
}
.big-font {
    font-size:28px !important;
    font-weight:600;
}
.stButton>button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #white, #1E40AF);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Ai Resume analyze</p>', unsafe_allow_html=True)
st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Ranking Configuration")

use_faiss = st.sidebar.toggle("Enable FAISS Vector Search", value=True)
use_rerank = st.sidebar.toggle("Enable Cross-Encoder Re-ranking", value=True)
use_fairness = st.sidebar.toggle("Run Fairness Audit", value=False)
use_shap = st.sidebar.toggle("Show SHAP Explainability", value=True)
use_rag = st.sidebar.toggle("Enable GPT RAG", value=True)

weight_tfidf = st.sidebar.slider("TF-IDF Weight", 0.0, 1.0, 0.4)
weight_sbert = st.sidebar.slider("SBERT Weight", 0.0, 1.0, 0.6)

# ------------------ INPUT ------------------
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area("📄 Paste Job Description", height=100)

with col2:
    uploaded_files = st.file_uploader(
        "📂 Upload Resumes",
        type=["pdf"],
        accept_multiple_files=True
    )

if uploaded_files:
    st.info(f"{len(uploaded_files)} resumes uploaded")

st.divider()

# ------------------ SESSION ------------------

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

if "edited_df" not in st.session_state:
    st.session_state["edited_df"] = None
    
if "shap_data" not in st.session_state:
    st.session_state["shap_data"] = []

df = st.session_state["results_df"]

# ------------------ RANK ------------------
if st.button("🔎 Rank Candidates", use_container_width=True):

    if not uploaded_files or not job_desc:
        st.warning("Upload resumes + enter job description")
        st.stop()

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

    files = [("files", f) for f in uploaded_files]
    st.session_state["shap_data"] = []
    with st.spinner("Ranking..."):
        res = requests.post(
            "http://127.0.0.1:8000/rank/",
            data=payload,
            files=files
        )

    results = res.json().get("results", [])
    st.session_state["shap_data"] = res.json().get("shap_values", [])

    if not results:
        st.error("No results")
        st.stop()

    for item in results:
        resume_text = item.get("content", "") or ""
        item["location"] = extract_location_advanced(resume_text)

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    st.session_state["results_df"] = df
    st.session_state["edited_df"] = None
    st.rerun()
    st.success("Ranking Done")
 
# ----show results-------------------------
df = st.session_state["results_df"]
# -----------------------------
# SHAP DISPLAY (SHORT VERSION)
# -----------------------------
if use_shap:
    shap_data = st.session_state.get("shap_data", [])

    if shap_data:
        with st.expander("🧠 SHAP Explainability Summary", expanded=False):

            for item in shap_data[:3]:   # only top 3 candidates
                st.markdown(f"### {item['candidate']}")

                df_shap = pd.DataFrame({
                    "Feature": item["features"],
                    "Impact": item["values"]
                })

                # sort by absolute importance
                df_shap["abs_impact"] = df_shap["Impact"].abs()
                df_shap = df_shap.sort_values("abs_impact", ascending=False)

                # keep only top 5 features
                top_features = df_shap.head(5)

                st.bar_chart(top_features.set_index("Feature")["Impact"])

                # short text explanation
                best_feature = top_features.iloc[0]["Feature"]
                best_value = top_features.iloc[0]["Impact"]

                st.success(
                    f"Main factor: **{best_feature}** "
                    f"(impact: {best_value:.2f})"
                )
            


if df is not None:

    if "location" not in df.columns:
        df["location"] = "Unknown"

    # LOCATION FILTER
    selected_city = st.selectbox(
        "📍 Filter candidates by location",
        ["All"] + CITIES,
        key="location_filter"
    )

    if selected_city != "All":
        filtered_by_city = df[
            df["location"].str.lower() == selected_city.lower()
        ]
    else:
        filtered_by_city = df.copy()

    # SCORE FILTER
    min_score = st.slider(
        "Min Score",
        0.0,
        100.0,
        0.0,
        key="min_score_slider"
    )

    filtered_df = filtered_by_city[
        filtered_by_city["score"] >= min_score
    ].copy()

    # METRICS
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Candidates", len(filtered_df))
    col2.metric("Top Score", round(filtered_df["score"].max(), 2) if not filtered_df.empty else 0)
    col3.metric("Average Score", round(filtered_df["score"].mean(), 2) if not filtered_df.empty else 0)

    # TABLE
    st.subheader("🏆 Ranked Candidates")

    if st.session_state["edited_df"] is None or len(st.session_state["edited_df"]) != len(filtered_df):
        temp_df = filtered_df.copy()
        temp_df["Select"] = False
        st.session_state["edited_df"] = temp_df

    edited_df = st.data_editor(
        st.session_state["edited_df"],
        use_container_width=True,
        key="candidate_editor"
    )

    st.session_state["edited_df"] = edited_df
    selected_rows = edited_df[edited_df["Select"] == True]


    # SAVE BUTTON
    if st.button("💾 Save Selected Candidates"):

        if selected_rows.empty:
            st.warning("No candidates selected!")
        else:
            selected_profiles = []

            for _, row in selected_rows.iterrows():
                selected_profiles.append({
                    "filename": row["filename"],
                    "score": float(row["score"]),
                    "rank": int(row["rank"])
                })

            payload = {
                "job_description": job_desc,
                "selected_profiles": selected_profiles
            }

            res = requests.post(
                "http://127.0.0.1:8000/save-selection/",
                json=payload
            )

            st.success("Saved Successfully")
            st.json(res.json())

    # TOP 10 #
    st.subheader("🔥 Top 10")
    st.dataframe(
        df.head(10).style
        .background_gradient(cmap="Blues", subset=["score"])
        .format({"score": "{:.2f}"})
    )

    # COLOR TABLE (FIXED POSITION)
    st.dataframe(df.style.applymap(color_score, subset=["score"]))

    # CHARTS
    st.subheader("📊 Charts")
    plt.style.use("dark_background")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.barh(filtered_df.head(10)["filename"], filtered_df.head(10)["score"])
        ax.invert_yaxis()
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.hist(df["score"], bins=10)
        st.pyplot(fig2)

    # PIE CHART
    st.subheader("📊 Selection Distribution")

    fig3, ax3 = plt.subplots()
    sizes = [
        len(df[df["score"] > 80]),
        len(df[(df["score"] > 60) & (df["score"] <= 80)]),
        len(df[df["score"] <= 60])
    ]

    ax3.pie(sizes, labels=["Strong", "Medium", "Weak"], autopct='%1.1f%%')
    st.pyplot(fig3)

# ------------------ LOAD SAVED ------------------
st.subheader("📂 Top saved Profile Re-Ranking")

if st.button("Load Candidate Database"):

    res = requests.get("http://127.0.0.1:8000/get-selections/")
    data = res.json().get("data", [])

    saved_df = pd.DataFrame(data)

    if not saved_df.empty:
        st.dataframe(saved_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.barh(saved_df["filename"], saved_df["score"])
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info("No saved data")