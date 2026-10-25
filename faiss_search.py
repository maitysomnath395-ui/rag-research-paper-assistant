import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Read the research paper
pdf_path = "data/research_paper.pdf"
document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

# Step 2: Split the text into chunks
chunk_size = 1000
overlap = 200
chunks = []
start = 0

while start < len(full_text):
    chunk = full_text[start:start + chunk_size]

    if chunk.strip():
        chunks.append(chunk)

    start += chunk_size - overlap

# Step 3: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Step 4: Create the FAISS index
embeddings = np.asarray(embeddings, dtype="float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("FAISS index created successfully!")
print("Total indexed chunks:", index.ntotal)

# Step 5: Ask a question
question = input("\nAsk a question about the research paper: ")

# Convert the question into an embedding
question_embedding = model.encode([question])
question_embedding = np.asarray(
    question_embedding, dtype="float32"
)

# Step 6: Find the 3 closest chunks
distances, indices = index.search(question_embedding, 3)

print("\nMost relevant passages:\n")

for rank, chunk_index in enumerate(indices[0], start=1):
    print(f"\n--- Result {rank} ---")
    print(chunks[chunk_index])
    print("Distance:", distances[0][rank - 1])