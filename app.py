# app.py
import streamlit as st
from rag import HealthcareRAG

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Healthcare Assistant RAG",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Healthcare Assistant RAG")
st.markdown("""
This system analyzes **medical reports and scanned PDFs** (like X-rays or MRI summaries) 
and retrieves relevant medical guidelines or knowledge from embedded data to assist diagnosis.
""")

# -----------------------------
# Initialize RAG System (cached)
# -----------------------------
@st.cache_resource
def init_rag():
    return HealthcareRAG()

rag = init_rag()

# -----------------------------
# Sidebar: PDF Upload
# -----------------------------
st.sidebar.header("📄 Upload Patient Reports (PDF only)")
uploaded_files = st.sidebar.file_uploader(
    "Select one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("📚 Processing uploaded reports..."):
        for file in uploaded_files:
            pdf_bytes = file.read()
            rag.ingest_pdf_bytes(pdf_bytes, source_name=file.name)
    st.sidebar.success("✅ All reports have been successfully processed.")

# -----------------------------
# Query Section
# -----------------------------
st.markdown("### 💬 Ask a Clinical Question")
query_text = st.text_area(
    "Enter your question about the patient's condition, diagnosis, or treatment:",
    placeholder="Example: What are the recommended next steps for a patient showing early signs of pneumonia?",
    height=100
)

if st.button("🔍 Get Diagnostic Suggestion"):
    if not query_text.strip():
        st.warning("Please enter a clinical question first.")
    else:
        with st.spinner("🧠 Generating diagnostic suggestion..."):
            result = rag.query(query_text)
        st.success("✅ Diagnostic Suggestion Generated")
        st.markdown("### 🩺 Suggested Diagnosis / Insights")
        st.write(result["text"])

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
#### ℹ️ About
This demo uses:
- **BioBERT** for text embeddings (medical understanding)
- **OCR** (Tesseract) for scanned PDFs
- **FAISS** for similarity search
- **LangGraph + GPT-4.1** for reasoning and summarization

> Developed as a Multimodal RAG demonstration using **LangGraph**.
""")
