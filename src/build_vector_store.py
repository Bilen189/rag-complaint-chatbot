import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

CHUNKS_FILE = "data/processed/complaint_chunks.csv"
VECTOR_STORE_DIR = "vector_store/chroma_db"

print("Loading chunks...")
df = pd.read_csv(CHUNKS_FILE)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(
    df["chunk_text"].tolist(),
    show_progress_bar=True
)

print("Creating ChromaDB...")
client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

collection = client.get_or_create_collection(
    name="complaints"
)

batch_size = 1000

for start in range(0, len(df), batch_size):

    end = min(start + batch_size, len(df))

    batch_df = df.iloc[start:end]

    collection.add(
        ids=[f"chunk_{i}" for i in batch_df.index],
        embeddings=embeddings[start:end].tolist(),
        documents=batch_df["chunk_text"].tolist(),
        metadatas=[
            {
                "complaint_id": str(row["complaint_id"]),
                "product_category": str(row["product_category"])
            }
            for _, row in batch_df.iterrows()
        ]
    )

print("Vector store created successfully.")
print("Location:", VECTOR_STORE_DIR)