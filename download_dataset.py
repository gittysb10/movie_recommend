import os

# Force KaggleHub to download into the project root directory
os.environ["KAGGLEHUB_CACHE"] = "./data"

import kagglehub

path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

print("Dataset downloaded to:", path)
