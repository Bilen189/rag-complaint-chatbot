from rag_pipeline import ComplaintRAGPipeline

TEST_QUESTIONS = [
    "What are common issues customers have with credit cards?",
    "Why are customers unhappy with money transfers?",
    "What complaints are common in savings accounts?",
    "What issues do people report about personal loans?",
    "Are customers reporting fraud-related concerns?",
    "What problems affect customer credit scores?",
    "What payment issues are customers experiencing?",
    "What customer service complaints appear most often?"
]


def main():
    rag = ComplaintRAGPipeline()

    print("=" * 100)
    print("RAG EVALUATION")
    print("=" * 100)

    for i, question in enumerate(TEST_QUESTIONS, start=1):

        result = rag.answer_question(question)

        print(f"\nQuestion {i}")
        print("-" * 100)
        print(question)

        print("\nGenerated Answer:")
        print(result["answer"])

        print("\nTop Sources:")

        for source in result["sources"][:2]:
            print(
                f"- {source['metadata'].get('product_category')} | "
                f"{source['metadata'].get('issue')}"
            )

        print("\n")
        

if __name__ == "__main__":
    main()