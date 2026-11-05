
import os
import requests
import streamlit as st
import pymupdf
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import json
from database import (
    initialize_database,
    save_document,
    save_chat,
    get_documents,
    get_chat_history
)

initialize_database()
# --------------------------------------------------
# ResearchIQ - AI Research Assistant
# --------------------------------------------------

st.set_page_config(
    page_title="ResearchIQ | AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.markdown(
    """
    <style>
    /* Fix uploaded PDF section visibility */
[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
}

[data-testid="stFileUploader"] button {
    background-color: #e8efff !important;
    color: #14264b !important;
    border: 1px solid #aabce0 !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
    color: #14264b !important;
    font-weight: 600 !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_pages(uploaded_files):
    """Extract text from uploaded PDFs while retaining page numbers."""
    documents = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    documents.append(
                        {
                            "filename": uploaded_file.name,
                            "page": page_number,
                            "text": text,
                        }
                    )

    return documents


def make_chunks(pages, chunk_size=1000, overlap=150):
    """Create overlapping text chunks with source metadata."""
    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]

        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()

            if len(piece) >= 40:
                chunks.append(
                    {
                        "filename": page["filename"],
                        "page": page["page"],
                        "text": piece,
                    }
                )

    return chunks


def build_search_index(chunks, model):
    """Encode chunks and build a FAISS similarity-search index."""
    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, vectors


def retrieve_chunks(question, chunks, index, model, top_k=5):
    """Return the most relevant chunks for a question."""
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    count = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, count)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        result = dict(chunks[int(idx)])
        result["score"] = float(score)
        results.append(result)

    return results


def generate_answer(question, sources):
    """Generate a concise answer grounded in research-paper evidence."""

    context_parts = []

    for i, source in enumerate(sources, start=1):
        context_parts.append(
            f"[Source {i}]\n"
            f"Document: {source['filename']}\n"
            f"Page: {source['page']}\n"
            f"Excerpt:\n{source['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are ResearchIQ, an academic research assistant.

Answer the user's question using ONLY the evidence provided below.

RULES:
1. Start with a direct answer in 1-2 sentences.
2. Use 3-5 concise bullet points when useful.
3. Cite factual claims with exact labels such as [Source 1].
4. Only use source numbers provided in the evidence.
5. Never invent findings, quotations, titles, or page numbers.
6. If the evidence is insufficient, clearly say so.
7. Avoid repetition and unnecessary introductions.
8. Separate evidence from interpretation.

RETRIEVED EVIDENCE:
{context}

USER QUESTION:
{question}

Write the answer now.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful research assistant. "
                        "Prioritize factual accuracy and evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()["message"]["content"]



# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 📚 ResearchIQ")
    st.caption("Your AI Research Assistant")
    st.divider()

    page = st.radio(
        "Workspace",
        ["Ask Papers", "Library", "About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Local AI • Evidence-based answers")
    st.caption(f"LLM: {OLLAMA_MODEL}")


# --------------------------------------------------
# Main workspace
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Research Workspace</h1>
        <p>Explore your research papers with AI-powered, evidence-based answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "pages" not in st.session_state:
    st.session_state.pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.subheader("Upload Papers")

    uploaded_files = st.file_uploader(
        "Choose research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based research PDFs.",
    )

   
    if st.button("Process Papers", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            try:
                with st.spinner("Extracting text from your papers..."):
                    pages = extract_pdf_pages(uploaded_files)

                if not pages:
                    st.error(
                        "No selectable text was found. "
                        "The PDFs may be scanned images."
                    )
                else:
                    chunks = make_chunks(pages)

                    if not chunks:
                        st.error("Not enough text was found to build an index.")
                    else:
                        with st.spinner("Loading the embedding model..."):
                            embedding_model = load_embedding_model()

                        with st.spinner("Building the search index..."):
                            index, _ = build_search_index(
                                chunks, embedding_model
                            )

                        st.session_state.pages = pages
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.processed_files = [
                            f.name for f in uploaded_files
                        ]

                        # Save document metadata in SQLite
                        unique_filenames = sorted({
                            p["filename"] for p in pages
                        })

                        for filename in unique_filenames:
                            file_pages = [
                                p for p in pages
                                if p["filename"] == filename
                            ]

                            file_chunks = [
                                c for c in chunks
                                if c["filename"] == filename
                            ]

                            save_document(
                                filename=filename,
                                page_count=len({
                                    p["page"] for p in file_pages
                                }),
                                chunk_count=len(file_chunks)
                            )

                        st.success(
                            f"Processed {len(uploaded_files)} PDF(s), "
                            f"{len(pages)} pages and {len(chunks)} chunks."
                        )

            except Exception as exc:
                st.error(f"Could not process the PDFs: {exc}")



if page == "Ask Papers":
    total_files = len(st.session_state.get("processed_files", []))
    total_pages = len(st.session_state.pages)
    total_chunks = len(st.session_state.chunks)

    col1, col2, col3 = st.columns(3)

    col1.metric("Papers processed", total_files)
    col2.metric("Pages with text", total_pages)
    col3.metric("Searchable chunks", total_chunks)

    st.subheader("Ask your research papers")

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What are the main findings and limitations "
            "of these research papers?"
        ),
        height=100,
    )

    top_k = st.slider(
        "Evidence passages to retrieve",
        min_value=2,
        max_value=8,
        value=5,
    )

    if st.button("Ask ResearchIQ", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif st.session_state.index is None:
            st.warning("Upload and process your PDFs first.")
        else:
            try:
                with st.spinner("Searching papers for relevant evidence..."):
                    embedding_model = load_embedding_model()
                    sources = retrieve_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.index,
                        embedding_model,
                        top_k=top_k,
                    )

                with st.spinner("Generating an evidence-based answer..."):
                    answer = generate_answer(question, sources)

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    }
                )
                save_chat(
                    question,
                    answer,
                    json.dumps(sources, default=str),
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to Ollama. Check that Ollama is running "
                    "and that its local API is available."
                )
            except requests.exceptions.HTTPError as exc:
                st.error(
                    f"Ollama returned an HTTP error: {exc}. "
                    f"Check that {OLLAMA_MODEL} is available with "
                    "`ollama list`."
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

    
    if st.session_state.history:
        result = st.session_state.history[-1]

        st.markdown("### Answer")
        st.markdown(result["answer"])

        # Generate downloadable research report
        report_lines = [
            "# ResearchIQ - Research Report",
            "",
            "## Research Question",
            result["question"],
            "",
            "## AI-Generated Answer",
            result["answer"],
            "",
            "## Retrieved Evidence",
        ]

        for i, source in enumerate(result["sources"], start=1):
            report_lines.extend([
                "",
                f"### Source {i}",
                f"Filename: {source['filename']}",
                f"Page: {source['page']}",
                f"Similarity Score: {source['score']:.3f}",
                "",
                source["text"],
            ])

        report_content = "\n".join(report_lines)

        st.download_button(
            label="Download Research Report",
            data=report_content,
            file_name="ResearchIQ_Report.md",
            mime="text/markdown",
            type="primary",
        )

        st.markdown("### Retrieved Evidence")

        for i, source in enumerate(result["sources"], start=1):
            with st.expander(
                f"[Source {i}] {source['filename']} — "
                f"page {source['page']} "
                f"(similarity: {source['score']:.3f})"
            ):
                st.write(source["text"])



elif page == "Library":
    st.subheader("Your Paper Library")

    documents = get_documents()
    chat_history = get_chat_history()

    st.markdown("### Saved Research Papers")

    if not documents:
        st.info("No saved papers yet.")
    else:
        for doc in documents:
            with st.expander(f"📄 {doc['filename']}"):
                st.write(f"Pages: {doc['page_count']}")
                st.write(f"Chunks: {doc['chunk_count']}")
                st.caption(f"Processed: {doc['processed_at']}")

    st.divider()

    st.markdown("### Previous Conversations")

    if not chat_history:
        st.info("No previous conversations yet.")
    else:
        for chat in chat_history:
            with st.expander(f"Q: {chat['question']}"):
                st.write(chat["answer"])
                st.caption(f"Saved: {chat['created_at']}")


elif page == "About":
    st.subheader("About ResearchIQ")

    st.write(
        """
        ResearchIQ is a local-first research assistant that helps users
        explore multiple academic PDFs.

        **Current capabilities**
        - PDF text extraction with page metadata
        - Semantic search using Sentence Transformers
        - FAISS similarity retrieval
        - Local answer generation through Ollama
        - Source excerpts and page references
        - SQLite document and chat history
        - Downloadable research reports

        **Future improvements**
        - Hybrid keyword and semantic retrieval
        - Cross-encoder reranking
        - Cross-paper comparison
        - Literature review generation
        """
    )


st.divider()
st.caption(
    "ResearchIQ • Local prototype. Verify important claims against "
    "the original research papers."
)
