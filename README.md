# RAG Complaint Chatbot

## Project Overview

This project builds a Retrieval-Augmented Generation (RAG) system for analyzing CFPB financial complaints. The system enables users to ask natural language questions about customer complaints and receive evidence-based responses generated from retrieved complaint narratives.

---

## Task 1: EDA and Preprocessing

The CFPB complaint dataset was explored and preprocessed to prepare it for semantic retrieval.

### Steps Performed

* Loaded the CFPB complaint dataset.
* Removed complaints with missing narratives.
* Filtered complaints to:

  * Credit Card
  * Savings Account
  * Money Transfer
  * Personal Loan
* Cleaned and normalized complaint text.
* Saved the cleaned dataset to:

```text
data/filtered_complaints.csv
```

---

## Task 2: Chunking, Embeddings, and Vector Store

### Stratified Sampling

A proportional stratified sample of 12,000 complaints was created to preserve the original product distribution while satisfying the required 10,000–15,000 sample range.

Sample distribution:

| Product Category | Sample Size |
| ---------------- | ----------: |
| Credit Card      |       5,116 |
| Savings Account  |       3,792 |
| Money Transfer   |       2,626 |
| Personal Loan    |         466 |

### Chunking Strategy

LangChain's `RecursiveCharacterTextSplitter` was used.

* Chunk Size: 500 characters
* Chunk Overlap: 50 characters

This configuration preserves contextual information while improving retrieval quality.

### Embedding Model

The following embedding model was used:

```text
all-MiniLM-L6-v2
```

The model generates dense vector representations that capture semantic meaning and support similarity-based retrieval.

### Vector Database

ChromaDB was used as the vector database.

Generated complaint chunks were embedded and stored with metadata for efficient semantic search.

---

## Running the Project

```bash
python src/prepare_chunks.py
python src/build_vector_store.py
```

---

## Repository Notes

Large generated datasets and vector store files are excluded from version control using `.gitignore` to keep the repository lightweight and compliant with GitHub file size limits.

These artifacts can be regenerated using the provided scripts.
