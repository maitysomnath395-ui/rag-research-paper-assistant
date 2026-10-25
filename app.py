import hashlib
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configure the Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Paper Assistant")
st.caption(
    "Ask questions about research papers using RAG, "
    "FAISS, SentenceTransformer and Ollama."
)


# --------------------------------------------------
# 2. Load the embedding model only once
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. Extract text and create page-aware chunks
# --------------------------------------------------

def extract_chunks(document_data, chunk_size=1000, overlap=200):
    chunks = []
    metadata = []

    for filename, pdf_bytes in document_data:
        try:
            with pymupdf.open(
                stream=pdf_bytes,
                filetype="pdf"
            ) as document:

                for page_number, page in enumerate(document, start=1):
                    page_text = page.get_text().strip()

                    if not page_text:
                        continue

                    start = 0

                    while start < len(page_text):
                        end = start + chunk_size
                        chunk = page_text[start:end]

                        if chunk.strip():
                            chunks.append(chunk)
                            metadata.append({
                                "source": filename,
                                "page": page_number
                            })

                        start += chunk_size - overlap

        except Exception as error:
            raise ValueError(
                f"Could not read '{filename}': {error}"
            ) from error

    if not chunks:
        raise ValueError(
            "No readable text was found in the PDFs. "
            "Scanned PDFs may require OCR."
        )

    return chunks, metadata


# --------------------------------------------------
# 4. Create the FAISS search index
# --------------------------------------------------

@st.cache_resource
def build_search_engine(document_data):
    chunks, metadata = extract_chunks(document_data)

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Normalize vectors for cosine-similarity search
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, metadata, index


# --------------------------------------------------
# 5. Retrieve relevant passages and generate an answer
# --------------------------------------------------

def answer_question(question, chunks, metadata, index):
    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)

    number_of_results = min(4, len(chunks))

    scores, indices = index.search(
        query_embedding,
        number_of_results
    )

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        retrieved.append({
            "text": chunks[idx],
            "source": metadata[idx]["source"],
            "page": metadata[idx]["page"],
            "score": float(score)
        })

    context_parts = []

    for i, item in enumerate(retrieved, start=1):
        citation = (
            f"[Source {i}: {item['source']}, "
            f"page {item['page']}]"
        )

        context_parts.append(
            f"{citation}\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a research paper assistant.

Answer the user's question using the supplied research
paper passages.

Instructions:
- Be clear and accurate.
- Use only information supported by the supplied passages.
- Cite factual claims using the source labels provided,
  such as [Source 1].
- If the passages do not contain enough information,
  clearly say so.
- Do not invent facts or references.

Research paper passages:
{context}

Question:
{question}

Answer:
"""

    # Connect to the hosted Ollama API
    try:
        api_key = st.secrets["OLLAMA_API_KEY"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "OLLAMA_API_KEY is missing. Add it under "
            "Streamlit Cloud Settings > Secrets."
        )

    client = ollama.Client(
        host="https://ollama.com",
        headers={
            "Authorization": f"Bearer {api_key}"
        }
    )

    response = client.chat(
        model="gemma4:31b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return answer, retrieved


# --------------------------------------------------
# 6. Select PDFs
# --------------------------------------------------

st.sidebar.header("📄 Research papers")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more text-based research papers."
)

# Use uploaded PDFs when provided.
# Otherwise, fall back to the existing default PDF.
if uploaded_files:
    document_data = tuple(
        (file.name, file.getvalue())
        for file in uploaded_files
    )

    document_label = ", ".join(
        file.name for file in uploaded_files
    )

else:
    default_path = Path("data/research_paper.pdf")

    if not default_path.exists():
        st.warning(
            "Upload a PDF using the sidebar to get started."
        )
        st.stop()

    document_data = (
        (default_path.name, default_path.read_bytes()),
    )

    document_label = default_path.name


# Fingerprint the selected PDFs so chat history can be
# reset when the document collection changes.
fingerprint_source = hashlib.sha256()

for filename, pdf_bytes in document_data:
    fingerprint_source.update(filename.encode("utf-8"))
    fingerprint_source.update(pdf_bytes)

document_fingerprint = fingerprint_source.hexdigest()


# --------------------------------------------------
# 7. Build the index for the selected PDFs
# --------------------------------------------------

try:
    with st.spinner("Preparing your research papers..."):
        chunks, metadata, index = build_search_engine(
            document_data
        )

    st.success(
        f"Ready! Indexed {len(chunks)} text chunks "
        f"from the selected PDF collection."
    )

except Exception as error:
    st.error(f"Could not prepare the PDFs: {error}")
    st.stop()


# --------------------------------------------------
# 8. Maintain chat history for the current session
# --------------------------------------------------

if (
    "document_fingerprint" not in st.session_state
    or st.session_state.document_fingerprint
    != document_fingerprint
):
    st.session_state.document_fingerprint = (
        document_fingerprint
    )
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []


if st.sidebar.button("Clear chat history"):
    st.session_state.messages = []
    st.rerun()


# --------------------------------------------------
# 9. Display selected documents and chat history
# --------------------------------------------------

st.caption(f"Current documents: {document_label}")

st.subheader("💬 Ask your research assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("View cited source passages"):
                    for i, item in enumerate(sources, start=1):
                        st.markdown(
                            f"**Source {i}: "
                            f"{item['source']} — "
                            f"Page {item['page']}**"
                        )
                        st.write(item["text"])


# --------------------------------------------------
# 10. Accept questions and generate answers
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your research paper..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching the papers and generating an answer..."
            ):
                answer, retrieved = answer_question(
                    question,
                    chunks,
                    metadata,
                    index
                )

            st.markdown(answer)

            if retrieved:
                with st.expander("View cited source passages"):
                    for i, item in enumerate(retrieved, start=1):
                        st.markdown(
                            f"**Source {i}: "
                            f"{item['source']} — "
                            f"Page {item['page']}**"
                        )
                        st.write(item["text"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved
            })

        except Exception as error:
            st.error(
                f"Could not generate an answer: {error}"
            )

            st.info(
                "Check your Ollama API key, hosted model "
                "availability, and Streamlit Cloud logs."
            )

            # Remove the unanswered user message so it
            # can be submitted again after fixing the error.
            st.session_state.messages.pop()
