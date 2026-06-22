from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


VECTOR_STORE_DIR = Path("vector_store/chroma_db")
COLLECTION_NAME = "complaints"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.

Use only the retrieved complaint excerpts below.
If the excerpts do not contain enough information, say that there is not enough information.

Context:
{context}

Question:
{question}

Answer:
"""


class ComplaintRAGPipeline:
    """RAG pipeline for retrieving complaint chunks and generating grounded answers."""

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
        """Embed a user question and retrieve top-k relevant complaint chunks."""
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

    def build_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format retrieved chunks into a context block for the prompt."""
        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk["metadata"]

            context_parts.append(
                f"Source {i}\n"
                f"Product: {metadata.get('product_category', 'Unknown')}\n"
                f"Company: {metadata.get('company', 'Unknown')}\n"
                f"Issue: {metadata.get('issue', 'Unknown')}\n"
                f"Complaint excerpt: {chunk['text']}\n"
            )

        return "\n".join(context_parts)

    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """Insert question and retrieved context into the prompt template."""
        context = self.build_context(retrieved_chunks)

        return PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate a grounded answer using retrieved complaint excerpts.

        This lightweight generator summarizes recurring issues from retrieved sources.
        It is designed as a safe baseline before integrating a larger LLM.
        """
        if not retrieved_chunks:
            return "I do not have enough information from the retrieved complaints to answer this question."

        issues = []
        companies = set()
        products = set()

        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]
            text = chunk["text"].lower()

            products.add(metadata.get("product_category", "Unknown"))
            companies.add(metadata.get("company", "Unknown"))

            if "late fee" in text or "fees" in text:
                issues.append("fees or unexpected charges")
            if "payment" in text or "pay" in text:
                issues.append("payment processing problems")
            if "credit score" in text or "credit report" in text:
                issues.append("credit score or reporting impacts")
            if "customer service" in text or "calling" in text:
                issues.append("customer service difficulties")
            if "cancel" in text or "closed" in text:
                issues.append("account closure or cancellation issues")
            if "fraud" in text or "unauthorized" in text:
                issues.append("fraud or unauthorized activity concerns")

        unique_issues = list(dict.fromkeys(issues))

        if not unique_issues:
            unique_issues = ["general dissatisfaction or unresolved account issues"]

        answer = (
            f"Based on the retrieved complaint excerpts, customers mainly report "
            f"{', '.join(unique_issues[:4])}. "
            f"The retrieved sources are mostly related to {', '.join(products)} complaints. "
            f"These complaints suggest that users are experiencing difficulties that may require "
            f"closer review by product, support, and compliance teams."
        )

        return answer

    def answer_question(self, question: str, top_k: int = 5) -> Dict:
        """Run retrieval, prompt creation, and answer generation."""
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, retrieved_chunks)
        answer = self.generate_answer(question, retrieved_chunks)

        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "sources": retrieved_chunks
        }


if __name__ == "__main__":
    rag = ComplaintRAGPipeline()

    question = "What are common issues customers have with credit cards?"
    result = rag.answer_question(question)

    print("Question:", result["question"])
    print("\nGenerated Answer:\n")
    print(result["answer"])

    print("\nRetrieved Sources:\n")
    for i, source in enumerate(result["sources"], start=1):
        print(f"Source {i}")
        print("Product:", source["metadata"].get("product_category"))
        print("Company:", source["metadata"].get("company"))
        print("Issue:", source["metadata"].get("issue"))
        print("Text:", source["text"][:400])
        print("-" * 80)