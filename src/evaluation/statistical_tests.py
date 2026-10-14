import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

try:
    from src.config import PROCESSED_DIR
except ImportError:
    from pathlib import Path
    PROCESSED_DIR = Path("data/processed")


def compute_paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0 or np.isnan(std_diff):
        return 0.0
    return float(np.mean(diff) / std_diff)


def run_tests(results_path=None):
    if results_path is None:
        results_path = PROCESSED_DIR / "benchmark_results.csv"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing {results_path}")
    df = pd.read_csv(results_path)

    # Normalize column names for robust access
    col_cut = "Cutoff" if "Cutoff" in df.columns else "cutoff"
    col_strat = "Strategy" if "Strategy" in df.columns else "strategy"
    col_coll = "Collection" if "Collection" in df.columns else "collection"
    col_ndcg = "nDCG" if "nDCG" in df.columns else ("ndcg_at_k" if "ndcg_at_k" in df.columns else "nDCG@K")
    col_qid = "QueryID" if "QueryID" in df.columns else ("query_id" if "query_id" in df.columns else "id")

    # Analyze on Cutoff=5 for nDCG
    df_k5 = df[df[col_cut] == 5] if col_cut in df.columns else df

    collections = df_k5[col_coll].unique()
    strategies = df_k5[col_strat].unique()

    test_results = []

    # 1. Compare chunking strategies
    for strat in strategies:
        df_strat = df_k5[df_k5[col_strat] == strat]
        for c1, c2 in itertools.combinations(collections, 2):
            scores1 = df_strat[df_strat[col_coll] == c1].sort_values(col_qid)[col_ndcg].values
            scores2 = df_strat[df_strat[col_coll] == c2].sort_values(col_qid)[col_ndcg].values

            if len(scores1) > 1 and len(scores1) == len(scores2):
                diffs = scores1 - scores2
                cohens_d = compute_paired_cohens_d(scores1, scores2)
                if np.all(diffs == 0):
                    p_ttest, p_wilcoxon = 1.0, 1.0
                else:
                    try:
                        _, p_ttest = ttest_rel(scores1, scores2)
                    except ValueError:
                        p_ttest = 1.0
                    try:
                        _, p_wilcoxon = wilcoxon(scores1, scores2)
                    except ValueError:
                        p_wilcoxon = 1.0

                test_results.append({
                    "Comparison Type": "Chunking Strategy",
                    "Fixed Factor": f"Strategy: {strat}",
                    "Group A": c1,
                    "Group B": c2,
                    "Metric": "nDCG@5",
                    "p_ttest": p_ttest,
                    "p_wilcoxon": p_wilcoxon,
                    "cohens_d": cohens_d,
                    "Significant (p<0.05)": bool(p_ttest < 0.05 or p_wilcoxon < 0.05)
                })

    # 2. Compare retrieval architectures
    for coll in collections:
        df_coll = df_k5[df_k5[col_coll] == coll]
        for s1, s2 in itertools.combinations(strategies, 2):
            scores1 = df_coll[df_coll[col_strat] == s1].sort_values(col_qid)[col_ndcg].values
            scores2 = df_coll[df_coll[col_strat] == s2].sort_values(col_qid)[col_ndcg].values

            if len(scores1) > 1 and len(scores1) == len(scores2):
                diffs = scores1 - scores2
                cohens_d = compute_paired_cohens_d(scores1, scores2)
                if np.all(diffs == 0):
                    p_ttest, p_wilcoxon = 1.0, 1.0
                else:
                    try:
                        _, p_ttest = ttest_rel(scores1, scores2)
                    except ValueError:
                        p_ttest = 1.0
                    try:
                        _, p_wilcoxon = wilcoxon(scores1, scores2)
                    except ValueError:
                        p_wilcoxon = 1.0

                test_results.append({
                    "Comparison Type": "Retrieval Architecture",
                    "Fixed Factor": f"Collection: {coll}",
                    "Group A": s1,
                    "Group B": s2,
                    "Metric": "nDCG@5",
                    "p_ttest": p_ttest,
                    "p_wilcoxon": p_wilcoxon,
                    "cohens_d": cohens_d,
                    "Significant (p<0.05)": bool(p_ttest < 0.05 or p_wilcoxon < 0.05)
                })

    res_df = pd.DataFrame(test_results)
    out_dir = PROCESSED_DIR if hasattr(PROCESSED_DIR, "mkdir") else Path(PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "statistical_significance.csv"
    res_df.to_csv(out_path, index=False)
    print(f"Statistical tests complete. Saved to {out_path}")


if __name__ == "__main__":
    run_tests()
