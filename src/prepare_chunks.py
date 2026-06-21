import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_PATH = Path("data/filtered_complaints.csv")
OUTPUT_PATH = Path("data/processed/complaint_chunks.csv")

SAMPLE_SIZE = 12000
RANDOM_STATE = 42

def create_stratified_sample(df, category_col="product_category"):
    return (
        df.groupby(category_col, group_keys=False)
        .apply(lambda x: x.sample(
            n=max(1, round(len(x) / len(df) * SAMPLE_SIZE)),
            random_state=RANDOM_STATE
        ))
        .reset_index(drop=True)
    )

def main():
    df = pd.read_csv(INPUT_PATH)

    sample_df = create_stratified_sample(df)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    rows = []

    for _, row in sample_df.iterrows():
        text = str(row["clean_narrative"])
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            rows.append({
                "complaint_id": row.get("Complaint ID", ""),
                "product_category": row.get("product_category", ""),
                "product": row.get("Product", ""),
                "issue": row.get("Issue", ""),
                "company": row.get("Company", ""),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_text": chunk
            })

    chunks_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(OUTPUT_PATH, index=False)

    print("Sample complaints:", sample_df.shape)
    print("Total chunks:", chunks_df.shape)
    print("Saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()