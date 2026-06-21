import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_PATH = Path("data/filtered_complaints.csv")
OUTPUT_PATH = Path("data/processed/complaint_chunks.csv")

SAMPLE_SIZE = 12000  # Required range: 10,000–15,000 complaints
RANDOM_STATE = 42

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def validate_sample_size(sample_size):
    """Ensure the sample size follows the assignment requirement."""
    if sample_size < 10000 or sample_size > 15000:
        raise ValueError("Sample size must be between 10,000 and 15,000.")


def validate_input_file(input_path):
    """Ensure the cleaned complaint dataset exists before processing."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")


def validate_required_columns(df):
    """Ensure required columns exist in the cleaned dataset."""
    required_cols = ["clean_narrative", "product_category"]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def create_stratified_sample(df, category_col="product_category"):
    """
    Create a proportional stratified sample across product categories.

    This keeps the sample representative of the original cleaned dataset
    while staying within the required 10,000–15,000 complaint range.
    """
    sampled_parts = []

    for category, group in df.groupby(category_col):
        category_sample_size = max(
            1,
            round(len(group) / len(df) * SAMPLE_SIZE)
        )

        sampled_group = group.sample(
            n=category_sample_size,
            random_state=RANDOM_STATE
        )

        sampled_parts.append(sampled_group)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    return sampled_df


def chunk_complaints(sample_df):
    """Split complaint narratives into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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

    return pd.DataFrame(rows)


def main():
    validate_sample_size(SAMPLE_SIZE)
    validate_input_file(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    validate_required_columns(df)

    sample_df = create_stratified_sample(df)

    chunks_df = chunk_complaints(sample_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(OUTPUT_PATH, index=False)

    print("Sample complaints:", sample_df.shape)
    print("Sample distribution:")
    print(sample_df["product_category"].value_counts())
    print("Total chunks:", chunks_df.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()