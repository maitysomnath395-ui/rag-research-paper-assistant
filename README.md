# RAG-Based AI Research Paper Assistant

An AI-powered research assistant that answers questions about research papers using Retrieval-Augmented Generation (RAG).

## Features

- Upload custom research papers in PDF format
- Extract and split PDF text into chunks
- Generate text embeddings using SentenceTransformers
- Retrieve relevant passages using FAISS
- Generate answers using a locally running Ollama language model
- Display source filenames and page numbers
- Maintain chat history during the session

## Technologies Used

- Python
- Streamlit
- FAISS
- SentenceTransformers
- PyMuPDF
- Ollama

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Install Ollama from https://ollama.com/download/windows.
5. Download the language model:

   `ollama pull llama3.2:3b`

6. Run the application:

   `streamlit run app.py`

## Usage

Open the local Streamlit URL in your browser, upload a research paper, and ask questions about its contents.

## Notes

Ollama must be installed and running locally. Research PDFs are excluded from the repository by default. Scanned PDFs may require OCR before their text can be retrieved.
