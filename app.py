# app.py (robust loader + recommender)
import streamlit as st
import pickle
import requests
from pathlib import Path
import numpy as np
import traceback

# ========== CONFIG ==========
# Must be the RAW / resolve URL (NOT blob). Example:
# https://huggingface.co/Goyam02/movies_similarity/resolve/main/similarity.pkl
HF_RAW_URL = "https://huggingface.co/Goyam02/movies_similarity/resolve/main/similarity.pkl"
# ============================

# Normalize to Path objects
SIM_LOCAL = Path("similarity.pkl")
MOVIES_LOCAL = Path("movies.pkl")

def download_with_progress(url: str, target: Path):
    """Download URL -> target with Streamlit progress bar."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk_size = 8192
        downloaded = 0
        with open(target, "wb") as f:
            with st.spinner("Downloading similarity matrix (this may take a while)..."):
                progress = st.progress(0)
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.progress(min(100, int(downloaded * 100 / total)))
    return target

@st.cache_data(show_spinner=False)
def load_similarity_from_local_or_hf():
    """Load similarity matrix from local file or download from HF_RAW_URL."""
    try:
        # check local file first
        if SIM_LOCAL.exists():
            with open(SIM_LOCAL, "rb") as f:
                sim = pickle.load(f)
            st.info("Loaded similarity from local similarity.pkl")
            return sim

        # else download from HF_RAW_URL
        if HF_RAW_URL and HF_RAW_URL.startswith("http"):
            st.info("similarity.pkl not found locally — attempting download from Hugging Face")
            download_with_progress(HF_RAW_URL, SIM_LOCAL)
            with open(SIM_LOCAL, "rb") as f:
                sim = pickle.load(f)
            st.success("Downloaded and loaded similarity.pkl from Hugging Face")
            return sim

        raise FileNotFoundError("No local similarity.pkl and HF_RAW_URL is not set")
    except Exception as e:
        # show full traceback in Streamlit so you can debug
        st.error(f"Error loading similarity: {e}")
        st.text(traceback.format_exc())
        raise

def load_movies_local():
    """Load movies.pkl from repo root (no HF)."""
    try:
        if not MOVIES_LOCAL.exists():
            raise FileNotFoundError("movies.pkl not found in repository root.")
        with open(MOVIES_LOCAL, "rb") as f:
            movies = pickle.load(f)
        return movies
    except Exception as e:
        st.error(f"Error loading movies.pkl: {e}")
        st.text(traceback.format_exc())
        raise

# ---------- MAIN ----------
try:
    movies = load_movies_local()
except Exception:
    st.stop()

try:
    similarity = load_similarity_from_local_or_hf()
except Exception:
    st.stop()

# ensure similarity is a numpy array for indexing
sim_arr = np.array(similarity)

movies_list = movies['title'].values

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
    except Exception:
        return []
    distances = sim_arr[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for idx, score in movie_list:
        recommended_movies.append(movies.iloc[idx].title)
    return recommended_movies

# UI
st.title("Movie Recommendation System")
selected_movie = st.selectbox("Choose a movie", movies_list)

if st.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        recs = recommend(selected_movie)
    if not recs:
        st.write("No recommendations found.")
    else:
        for r in recs:
            st.write(r)
