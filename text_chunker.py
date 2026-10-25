import pymupdf


def get_text_chunks():
    # Read the research paper
    pdf_path = "data/research_paper.pdf"
    document = pymupdf.open(pdf_path)

    try:
        # Extract text from all pages
        full_text = ""

        for page in document:
            full_text += page.get_text() + "\n"

    finally:
        document.close()

    # Split text into chunks
    chunk_size = 1000
    overlap = 200

    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    print("Total characters:", len(full_text))
    print("Total chunks:", len(chunks))

    return chunks


# Test the function when running this file directly
if __name__ == "__main__":
    chunks = get_text_chunks()

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
