import os
import sys
import json

# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "..", "models")
sys.path.append(models_dir)

from advanced_retriever import AdvancedResearchRetriever


class SilverStandardLabeler:
    """
    Generates a Silver-Standard benchmark for retrieval evaluation.

    Workflow:
        Query
            ↓
        Hybrid Retrieval
            ↓
        Cross-Encoder Reranking
            ↓
        Top-N Candidate Papers
            ↓
        Manual Verification
            ↓
        Gold Standard
    """

    def __init__(
        self,
        benchmark_path="data/evaluation/benchmark_queries.json"
    ):

        print("=" * 70)
        print("Initializing Silver Standard Label Generator")
        print("=" * 70)

        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(
                f"Benchmark file not found:\n{benchmark_path}"
            )

        self.benchmark_path = benchmark_path

        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        self.retriever = AdvancedResearchRetriever()

        print(f"Loaded {len(self.queries)} benchmark queries.\n")

    def generate_silver_standard(
        self,
        candidate_pool=50,
        top_candidates=5,
        output_path="data/evaluation/benchmark_queries_silver.json",
    ):

        print("Generating candidate relevance labels...\n")

        silver_queries = []

        total_candidates = 0

        for query in self.queries:

            query_text = query["query"]

            # ----------------------------------------------------------
            # Retrieve a large candidate pool and rerank
            # ----------------------------------------------------------

            candidates = self.retriever.retrieve_and_rerank(
                query=query_text,
                final_top_k=candidate_pool,
            )

            # ----------------------------------------------------------
            # Keep only the highest ranked candidates
            # ----------------------------------------------------------

            top_docs = candidates[:top_candidates]

            candidate_metadata = []

            relevant_ids = []

            for rank, doc in enumerate(top_docs, start=1):

                candidate_metadata.append(
                    {
                        "rank": rank,
                        "paper_id": str(doc["id"]),
                        "title": doc.get("title", ""),
                        "rerank_score": round(
                            float(doc.get("rerank_score", 0.0)),
                            4,
                        ),
                    }
                )

                relevant_ids.append(str(doc["id"]))

            total_candidates += len(candidate_metadata)

            silver_queries.append(
                {
                    "id": query.get("id", ""),
                    "query": query_text,
                    "expected_keywords": query.get(
                        "expected_keywords",
                        [],
                    ),

                    # Automatically suggested IDs
                    "relevant_paper_ids": relevant_ids,

                    # Metadata for human verification
                    "candidate_papers": candidate_metadata,
                }
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                silver_queries,
                f,
                indent=4,
                ensure_ascii=False,
            )

        print("=" * 70)
        print("Silver Standard Benchmark Generated")
        print("=" * 70)
        print(f"Queries Processed      : {len(silver_queries)}")
        print(f"Candidate Papers Saved : {total_candidates}")
        print(
            f"Average Candidates     : "
            f"{round(total_candidates / len(silver_queries), 2)}"
        )
        print(f"Output File            : {output_path}")
        print("=" * 70)

if __name__ == "__main__":

    labeler = SilverStandardLabeler()

    labeler.generate_silver_standard(
        candidate_pool=50,
        top_candidates=5,
    )