import shutil
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/complaint_chunks.csv")
VECTOR_STORE_DIR = Path("vector_store/chroma_db")
COLLECTION_NAME = "complaints"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000


def validate_chunks_file(df):
    required_cols = [
        "complaint_id",
        "product_category",
        "product",
        "issue",
        "company",
        "chunk_text"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in chunks file: {missing_cols}")

    if df["chunk_text"].isna().any():
        raise ValueError("Some chunks have missing text.")

    if df["product_category"].isna().any():
        print("Warning: Some product_category values are missing.")


def clean_metadata_value(value):
    if pd.isna(value):
        return "Unknown"
    return str(value)


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}. "
            "Run src/prepare_chunks.py first."
        )

    print("Loading chunks...")
    df = pd.read_csv(CHUNKS_FILE)
    validate_chunks_file(df)

    print("Chunk product distribution:")
    print(df["product_category"].value_counts(dropna=False))

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(
        df["chunk_text"].tolist(),
        show_progress_bar=True
    )

    if VECTOR_STORE_DIR.exists():
        print("Removing existing vector store...")
        shutil.rmtree(VECTOR_STORE_DIR)

    print("Creating ChromaDB...")
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        batch_df = df.iloc[start:end]

        metadatas = [
            {
                "complaint_id": clean_metadata_value(row["complaint_id"]),
                "product_category": clean_metadata_value(row["product_category"]),
                "product": clean_metadata_value(row["product"]),
                "issue": clean_metadata_value(row["issue"]),
                "company": clean_metadata_value(row["company"]),
            }
            for _, row in batch_df.iterrows()
        ]

        collection.add(
            ids=[f"chunk_{i}" for i in batch_df.index],
            embeddings=embeddings[start:end].tolist(),
            documents=batch_df["chunk_text"].tolist(),
            metadatas=metadatas
        )

    print("Vector store created successfully.")
    print("Location:", VECTOR_STORE_DIR)
    print("Total chunks stored:", collection.count())


if __name__ == "__main__":
    main()