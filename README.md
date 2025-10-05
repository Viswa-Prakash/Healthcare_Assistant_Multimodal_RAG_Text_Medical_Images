# Healthcare Assistant – Multimodal RAG (Text + X-Rays + MRI Scans)

A multimodal Retrieval-Augmented Generation (RAG) system for healthcare, enabling doctors and researchers to:

- Upload patient reports (PDFs, scanned documents).

- Upload medical images (X-rays, MRI scans).

- Retrieve relevant medical guidelines + annotated images.

- Generate diagnostic suggestions using LLMs.

This project uses LangGraph, FAISS, BioBERT, and CLIP to handle both text and image modalities.

## Features

- OCR with Tesseract for scanned medical reports.

- BioBERT embeddings for medical text.

- CLIP embeddings for X-rays / MRI scans.

- Separate FAISS indexes for text and images for efficient retrieval.

- LangGraph workflow:

 - retrieve_text → find relevant guidelines.

 - retrieve_images → fetch similar medical scans.

 - summarize → LLM generates a combined diagnostic suggestion.

- Streamlit UI to upload reports & scans, and query the system.


## Tech Stack

- LLMs & Embeddings:

 - BioBERT (medical text)

 - OpenAI CLIP (images + text)

- Vector Store: FAISS

- Frameworks: LangGraph, LangChain

- OCR: Tesseract OCR

- Frontend: Streamlit


## Installation
```
1. Clone Repo
git clone https://github.com/yourusername/healthcare-assistant-rag.git
cd healthcare-assistant-rag

2. Install Python Dependencies
pip install -r requirements.txt


3. Start Streamlit App
streamlit run app.py

4. Use the UI
Upload PDF reports and X-ray/MRI scans.

Ask queries like:

“What are the recommended treatments for this lung X-ray?”

“Summarize this MRI scan with reference to guidelines.”

“Compare this report to known case studies.”
```