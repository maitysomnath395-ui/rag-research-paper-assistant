from sentence_transformers import SentenceTransformer
from text_chunker import get_text_chunks

# Load the pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Convert a list of text strings into numerical embeddings.
    """
    embeddings = model.encode(texts)
    return embeddings


# Test the embedding generator when run directly
if __name__ == "__main__":
    chunks = get_text_chunks()

    embeddings = generate_embeddings(chunks)

    print("Total chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("First embedding (first 10 numbers):")
    print(embeddings[0][:10])