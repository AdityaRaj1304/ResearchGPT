import json
import logging
from groq import Groq
from tenacity import retry, wait_exponential, stop_after_attempt
from src.config import CHROMA_DB_DIR, COLLECTIONS, DATA_DIR, EVALUATION_DIR, GROQ_API_KEY, LLM_MODEL
from src.models.advanced_retriever import ParameterizedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=3, max=30))
def judge_relevance(client: Groq, query: str, chunk_text: str) -> bool:
    prompt = (
        "Determine if the following document excerpt contains specific technical information "
        "relevant to answering the research query. Respond ONLY with a JSON object: "
        '{"relevant": true} or {"relevant": false}.\n\n'
        f"Query: {query}\nExcerpt: {chunk_text[:2500]}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    payload = json.loads(resp.choices[0].message.content or "{}")
    return bool(payload.get("relevant", False))


def run_silver_labeling():
    raw_queries_path = EVALUATION_DIR / "benchmark_queries.json"
    output_path = DATA_DIR / "benchmark_queries.json"

    if not raw_queries_path.exists():
        logger.error(f"Source queries not found at {raw_queries_path}")
        return

    with open(raw_queries_path, "r", encoding="utf-8") as f:
        candidate_queries = json.load(f)

    retriever = ParameterizedRetriever(
        collection_name=COLLECTIONS["fixed"],
        persist_directory=str(CHROMA_DB_DIR),
    )
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    labeled_queries = []
    for idx, item in enumerate(candidate_queries):
        q_id = item.get("query_id") or item.get("id") or f"q_{idx + 1:02d}"
        q_text = item.get("query", "")
        existing_rel = item.get("relevant_paper_ids", [])

        if existing_rel:
            labeled_queries.append({
                "query_id": q_id,
                "query": q_text,
                "relevant_paper_ids": existing_rel,
            })
            continue

        res = retriever.search_hybrid_rerank(q_text, top_k=5)
        if isinstance(res, tuple) and len(res) == 2:
            chunks, _ = res
        elif isinstance(res, dict):
            chunks = res.get("results", [])
        else:
            chunks = res if isinstance(res, list) else []

        relevant_papers = set()
        for chunk in chunks:
            paper_id = chunk.get("metadata", {}).get("paper_id")
            if not paper_id:
                continue
            if client:
                try:
                    if judge_relevance(client, q_text, chunk.get("document", "")):
                        relevant_papers.add(paper_id)
                except Exception as e:
                    logger.warning(f"LLM labeling fallback for {q_id}: {e}")
            else:
                # Top-1 fallback if no API key provided
                relevant_papers.add(paper_id)
                break

        if not relevant_papers and chunks:
            top_pid = chunks[0].get("metadata", {}).get("paper_id")
            if top_pid:
                relevant_papers.add(top_pid)

        labeled_queries.append({
            "query_id": q_id,
            "query": q_text,
            "relevant_paper_ids": sorted(list(relevant_papers)),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_queries, f, indent=2)
    logger.info(f"Saved {len(labeled_queries)} labeled benchmark queries to {output_path}")


if __name__ == "__main__":
    run_silver_labeling()