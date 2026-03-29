import os
import pandas as pd
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer  # Local embeddings

# =========================
# LOAD CLEANED DATASET
# =========================
file_path = "cleaned_symptom_dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("❌ cleaned_symptom_dataset.csv not found! Run preprocess.py first.")

print("📂 Loading cleaned dataset...")
df = pd.read_csv(file_path)

# Ensure required column exists
if "combined_symptoms" not in df.columns:
    raise ValueError("❌ 'combined_symptoms' column not found in dataset!")

texts = df["combined_symptoms"].astype(str).tolist()

print(f"✅ Dataset Loaded Successfully!")
print(f"📊 Total Records: {len(texts)}")

# =========================
# LOAD LOCAL EMBEDDING MODEL
# =========================
print("\n🧠 Generating Embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, small, good for cosine similarity

# Generate embeddings
embedding_matrix = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

print("✅ Embeddings Generated!")
print("📐 Embedding Dimension:", embedding_matrix.shape[1])

# =========================
# CREATE FAISS INDEX (COSINE SIMILARITY)
# =========================
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)  # Cosine similarity via inner product
index.add(embedding_matrix)

print("\n⚡ FAISS Vector Database Created Successfully!")
print("🔢 Total Vectors Stored:", index.ntotal)

# =========================
# SAVE VECTOR DB + METADATA
# =========================
faiss.write_index(index, "symptom_disease.index")

with open("metadata.pkl", "wb") as f:
    pickle.dump(df, f)

print("\n💾 Files Saved Successfully:")
print("1️⃣ symptom_disease.index  → Vector Database (FAISS)")
print("2️⃣ metadata.pkl          → Disease + Symptoms Mapping")
print("\n🎉 Vector DB Ready for Cosine Similarity Search!")
