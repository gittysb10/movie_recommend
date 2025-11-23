# 🎬 Movie Recommendation System (Content-Based)

[![Streamlit](https://img.shields.io/badge/Platform-Streamlit-FF4B4B?logo=streamlit)](#)
[![HuggingFace](https://img.shields.io/badge/Model%20Storage-HuggingFace-yellow?logo=huggingface)](#)
[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

A lightweight **content-based movie recommendation system** built using TMDB metadata.
It extracts key movie features, processes them into vectors, computes similarity scores, and serves the recommendations through an interactive **Streamlit UI**.

---

## 🚀 Features

* Content-based filtering using **cosine similarity**
* Metadata extraction:

  * Genres
  * Keywords
  * Top 3 cast members
  * Director
  * Overview
* Stemming of normalized text (NLTK)
* Vectorization via **CountVectorizer**
* Fast, precomputed similarity matrix
* Supports downloading large models from **Hugging Face**
* Clean, interactive Streamlit interface

---

## 📁 Project Structure

```
.
├── app.py                   # Streamlit UI application
├── main.ipynb               # Data preprocessing + similarity computation
├── movies.pkl               # Cleaned movie metadata
├── requirements.txt         # Dependencies for app & notebook
├── runtime.txt              # Python version (for Streamlit Cloud)
├── README.md                # Documentation
└── data/
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

---

## 🧠 System Architecture

```
 TMDB CSVs
    │
    ▼
 Data Cleaning & Feature Extraction (main.ipynb)
    │
    ├── create movies.pkl
    └── compute similarity matrix → similarity.pkl
    │
    ▼
 Streamlit App (app.py)
    │
    ├── load movies.pkl
    ├── load similarity from local OR Hugging Face
    └── recommend top similar movies
```

---

## 🛠️ Local Setup

### 1️⃣ Create a virtual Environment 

```bash
python -m venv venv
```

### 2️⃣ Activate the Environment

```bash
source venv/Scripts/activate  # Windows
source venv/bin/activate      # Macos 
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Your app will open at:

👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Regenerating Artifacts (optional)

Open and run:

```
main.ipynb
```

This notebook:

* Cleans the TMDB dataset
* Creates a `tags` column
* Computes cosine similarity
* Saves artifacts:

  * `movies.pkl`
  * `similarity.pkl` **or** `similarity.npz`
    (depending on your notebook code)

#

## ☁️ Deploying on Streamlit Cloud

### 1. Push the repo to GitHub

(NO large model files like similarity.pkl — use HF for that)

### 2. Add these two files:

**runtime.txt**

```
python-3.10.12
```

**requirements.txt (example)**

```
streamlit
numpy==1.25.3
pandas
scikit-learn
joblib
requests
```

### 3. Ensure `app.py` contains your HF URL:

```python
HF_RAW_URL = "https://huggingface.co/<username>/<repo>/resolve/main/similarity.pkl"
```

### 4. Deploy

Go to: **[https://share.streamlit.io](https://share.streamlit.io)**

* New App
* Choose your GitHub repo
* Branch: `main`
* Entry point: `app.py`
* Deploy 🎉

Streamlit will download the model from Hugging Face on first run.

---

## 🌐 Hosting the similarity model on Hugging Face

Upload `similarity.pkl` to your HF repo.

Use the **raw URL**:

```
https://huggingface.co/<username>/<repo>/resolve/main/similarity.pkl
```

⚠️ *Do NOT use the blob link — it won’t work.*
Use `/resolve/main/` or `/raw/main/`.

---

## 🧠 Recommendation Logic

```python
movie_index = movies[movies['title'] == movie].index[0]
distances = similarity[movie_index]
movie_list = sorted(
    list(enumerate(distances)),
    reverse=True,
    key=lambda x: x[1]
)[1:6]
return [movies.iloc[i[0]].title for i in movie_list]
```

---

## ❗ Common Issues & Fixes

### 🔥 GitHub rejecting large files?

GitHub doesn’t allow >100MB.
Solution:

* Upload large similarity file to Hugging Face
* Let `app.py` download it at runtime

### 🔥 Streamlit build failing on numpy?

Use a wheel-friendly version:

```
numpy==1.25.3
```

Add runtime.txt:

```
python-3.10.12
```

### 🔥 Using private Hugging Face files?

Add a token to Streamlit Secrets.

---

## 🌟 Future Enhancements

* TMDB poster integration
* Movie detail pages
* Hybrid recommender (Content + Collaborative Filtering)
* Semantic similarity with Sentence Transformers
* Compressed sparse similarity matrix

---

## ❤️ Acknowledgements

* TMDB for the dataset
* Streamlit for the UI
* Hugging Face for large file hosting
* Scikit-learn & Pandas for preprocessing


