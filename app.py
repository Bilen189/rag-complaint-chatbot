import streamlit as st
from src.rag_pipeline import ComplaintRAGPipeline

st.set_page_config(
    page_title="CrediTrust Complaint Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 CrediTrust Complaint Chatbot")
st.markdown(
    "Ask questions about customer complaints and receive evidence-based answers."
)

@st.cache_resource
def load_rag():
    return ComplaintRAGPipeline()

rag = load_rag()

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "sources" not in st.session_state:
    st.session_state.sources = []

question = st.text_input(
    "Enter your question:",
    placeholder="What are common issues customers have with credit cards?"
)

col1, col2 = st.columns(2)

with col1:
    ask = st.button("Ask")

with col2:
    clear = st.button("Clear")

if clear:
    st.session_state.answer = ""
    st.session_state.sources = []
    st.rerun()

if ask:

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching complaint database..."):

            result = rag.answer_question(question)

            st.session_state.answer = result["answer"]
            st.session_state.sources = result["sources"]

if st.session_state.answer:

    st.subheader("Generated Answer")
    st.success(st.session_state.answer)

    st.subheader("Retrieved Sources")

    for i, source in enumerate(st.session_state.sources, start=1):

        metadata = source["metadata"]

        with st.expander(f"Source {i}"):

            st.write(
                f"**Product Category:** "
                f"{metadata.get('product_category', 'Unknown')}"
            )

            st.write(
                f"**Company:** "
                f"{metadata.get('company', 'Unknown')}"
            )

            st.write(
                f"**Issue:** "
                f"{metadata.get('issue', 'Unknown')}"
            )

            st.write(
                f"**Similarity Distance:** "
                f"{source['distance']:.4f}"
            )

            st.write("**Complaint Excerpt:**")

            st.write(source["text"])