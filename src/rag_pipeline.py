from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


VECTOR_STORE_DIR = Path("vector_store/chroma_db")
COLLECTION_NAME = "complaints"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class ComplaintRAGPipeline:
    """
    RAG pipeline for retrieving complaint chunks from ChromaDB.
    """

    def __init__(self, vector_store_dir=VECTOR_STORE_DIR):
        if not vector_store_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {vector_store_dir}. "
                "Run src/build_vector_store.py first."
            )

        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=str(vector_store_dir))
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Embed a user question and retrieve top-k relevant complaint chunks.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question_embedding = self.model.encode(question).tolist()

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved_chunks.append({
                "text": document,
                "metadata": metadata,
                "distance": distance
            })

        return retrieved_chunks


if __name__ == "__main__":
    rag = ComplaintRAGPipeline()

    question = "What are common issues customers have with credit cards?"
    results = rag.retrieve(question)

    print("Question:", question)
    print("\nTop Retrieved Chunks:\n")

    for i, item in enumerate(results, start=1):
        print(f"Result {i}")
        print("Product:", item["metadata"].get("product_category"))
        print("Distance:", item["distance"])
        print("Text:", item["text"][:500])
        print("-" * 80)