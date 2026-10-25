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

    client = ollama.Client(
        host="https://ollama.com",
        headers={
            "Authorization": (
                f"Bearer {st.secrets['OLLAMA_API_KEY']}"
            )
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
