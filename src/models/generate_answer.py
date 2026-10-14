import os
import sys
from groq import Groq
from src.config import CHROMA_DB_DIR, COLLECTIONS, GROQ_API_KEY, LLM_MODEL
from src.models.advanced_retriever import ParameterizedRetriever


def generate_research_answer(
    query: str,
    collection_name: str = COLLECTIONS["semantic"],
    strategy: str = "hybrid_rerank",
    top_k: int = 5,
) -> dict:
    retriever = ParameterizedRetriever(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DB_DIR),
    )

    if strategy == "dense":
        retrieval_out = retriever.search_dense(query, top_k=top_k)
    elif strategy == "hybrid":
        retrieval_out = retriever.search_hybrid(query, top_k=top_k)
    else:
        retrieval_out = retriever.search_hybrid_rerank(query, top_k=top_k)

    if isinstance(retrieval_out, tuple) and len(retrieval_out) == 2:
        results, timings = retrieval_out
    elif isinstance(retrieval_out, dict):
        results = retrieval_out.get("results", [])
        timings = retrieval_out.get("timings", {})
    else:
        results = retrieval_out if isinstance(retrieval_out, list) else []
        timings = {}

    context_blocks = []
    for idx, item in enumerate(results):
        meta = item.get("metadata", {})
        paper_id = meta.get("paper_id", "Unknown")
        context_blocks.append(f"[{idx + 1}] (Paper ID: {paper_id})\n{item.get('document', '')}")

    context_str = "\n\n".join(context_blocks)[:8000]

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=GROQ_API_KEY)
    prompt = (
        "You are an academic AI research assistant. Answer the query strictly using the provided context. "
        "Cite sources inline using bracketed numbers like [1], [2]. If the context is insufficient, state so clearly.\n\n"
        f"Context:\n{context_str}\n\nQuery: {query}"
    )

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return {
        "query": query,
        "answer": completion.choices[0].message.content,
        "results": results,
        "timings": timings,
    }


if __name__ == "__main__":
    test_q = sys.argv[1] if len(sys.argv) > 1 else "What are the main challenges in LLM unlearning?"
    output = generate_research_answer(test_q)
    print("\n--- Answer ---\n", output["answer"])