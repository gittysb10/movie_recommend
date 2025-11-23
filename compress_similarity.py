import pickle
import numpy as np

# Load original similarity
similarity = pickle.load(open("similarity.pkl", "rb"))

# Convert to float16 (halves size)
similarity = similarity.astype('float16')

# Save using numpy compressed format
np.savez_compressed("similarity.npz", similarity=similarity)

print("Done! File saved as similarity.npz")
