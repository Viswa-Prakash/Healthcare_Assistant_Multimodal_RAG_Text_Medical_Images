# rag.py
import os
import io
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import pytesseract

# Transformers
from transformers import AutoTokenizer, AutoModel

# LangChain / LangGraph
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain.schema.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# FAISS
import faiss


# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model Initialization
# -----------------------------
# BioBERT for text embeddings
tokenizer_text = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_text = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
model_text.eval()

BaseModel.model_config = {"arbitrary_types_allowed": True}


# -----------------------------
# Embedding Helper
# -----------------------------
@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    """Return L2-normalized BioBERT embedding as float32 numpy array."""
    if not text:
        text = ""
    enc = tokenizer_text(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    out = model_text(**enc)
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state.mean(dim=1)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    return pooled.cpu().numpy()[0].astype(np.float32)


# -----------------------------
# FAISS VectorStore
# -----------------------------
class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.documents = []
        self.metadatas = []
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity via normalized vectors

    def add(self, emb: np.ndarray, doc: Document):
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self.index.add(emb)
        self.documents.append(doc)
        self.metadatas.append(doc.metadata or {})

    def similarity_search(self, query_emb: np.ndarray, k: int = 5):
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(q, min(k, len(self.documents)))
        return [self.documents[i] for i in I[0] if i != -1]


# -----------------------------
# Healthcare RAG
# -----------------------------
class HealthcareRAG:
    def __init__(self, embed_dim: int = 768):
        self.vstore = VectorStore(dimension=embed_dim)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        self.llm = init_chat_model("openai:gpt-4.1")
        self.checkpointer = MemorySaver()
        self.graph = self._create_graph()

    # -------------------------
    # PDF Ingestion (text + OCR)
    # -------------------------
    def ingest_pdf_bytes(self, pdf_bytes: bytes, source_name: str):
        """Extract text from PDF or OCR scanned pages if needed."""
        docs = []

        # Try extracting text directly from PDF
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": source_name, "page": i}))
        except Exception as e:
            print(f"Error reading {source_name} via PyPDF2: {e}")

        # If no text found, fallback to OCR
        if not docs:
            try:
                images = convert_from_bytes(pdf_bytes)
                for i, img in enumerate(images):
                    text = pytesseract.image_to_string(img)
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": source_name, "page": i, "type": "ocr"}))
            except Exception as e:
                print(f"OCR failed for {source_name}: {e}")

        # Add all extracted text chunks to FAISS
        if docs:
            for doc in docs:
                for chunk in self.splitter.split_documents([doc]):
                    emb = embed_text(chunk.page_content or "")
                    self.vstore.add(emb, chunk)

    # -------------------------
    # LangGraph Workflow
    # -------------------------
    def _create_graph(self):
        workflow = StateGraph(dict)

        def retrieve_text(state):
            query_text = state.get("query_text", "")
            if not query_text.strip():
                return {"query_text": query_text, "text_hits": []}
            q_emb = embed_text(query_text)
            hits = self.vstore.similarity_search(q_emb, k=10)
            return {"query_text": query_text, "text_hits": hits}

        def summarize(state):
            query_text = state.get("query_text", "")
            text_hits = state.get("text_hits", [])

            context = "\n\n".join(
                [f"[Source: {h.metadata.get('source','')}] {h.page_content}" for h in text_hits[:6]]
            )

            message = HumanMessage(
                content=f"Clinical question: {query_text}\n\nRelevant context:\n{context}\n\n"
                        f"Generate a diagnostic suggestion based on the above clinical data."
            )

            try:
                response = self.llm.invoke([message])
                answer_text = getattr(response, "content", str(response))
            except Exception as e:
                answer_text = f"LLM error: {e}"

            return {"answer": {"text": answer_text}}

        workflow.add_node("retrieve_text", retrieve_text)
        workflow.add_node("summarize", summarize)
        workflow.set_entry_point("retrieve_text")
        workflow.add_edge("retrieve_text", "summarize")
        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # -------------------------
    # Query Interface
    # -------------------------
    def query(self, query_text: str):
        if not query_text.strip():
            return {"text": "Please provide a clinical query."}
        state = {"query_text": query_text}
        res = self.graph.invoke(state, config={"configurable": {"thread_id": "streamlit-session"}})
        ans = res.get("answer") if isinstance(res, dict) else res
        return {"text": ans.get("text", "") if ans else ""}
