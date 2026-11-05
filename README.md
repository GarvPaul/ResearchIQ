# ResearchIQ – AI-Powered Research Paper Assistant

ResearchIQ is an AI-powered research assistant that helps users upload PDF research papers, search documents, and ask questions to retrieve relevant information.

## Features
- Upload and process PDF research papers.
- Extract and split document text into smaller chunks.
- Generate semantic embeddings using Sentence Transformers.
- Perform similarity-based search using FAISS.
- Generate context-aware answers using Llama 3.
- Interactive user interface built with Streamlit.
- Store document and chat information using SQLite.

## Tech Stack
- Python
- Streamlit
- LangChain
- Sentence Transformers
- FAISS
- Llama 3 (Ollama)
- PyMuPDF
- SQLite

## How It Works
1. User uploads a PDF document.
2. The application extracts and chunks the text.
3. Sentence Transformers generate embeddings.
4. FAISS retrieves relevant document sections based on the user's question.
5. Llama 3 generates an answer using the retrieved context.

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/GarvPaul/ResearchIQ.git
cd ResearchIQ
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and the Llama 3 model is available:

```bash
ollama pull llama3.2:3b
```

Run the application:

```bash
streamlit run app.py
```

## Author
Garv Paul

GitHub: https://github.com/GarvPaul
