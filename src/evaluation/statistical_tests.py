import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import itertools
import os

def run_tests(results_path="data/processed/benchmark_results.csv"):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing {results_path}")
    df = pd.read_csv(results_path)
    
    # Analyze on Cutoff=5 for nDCG
    df_k5 = df[df["Cutoff"] == 5]
    
    collections = df_k5["Collection"].unique()
    strategies = df_k5["Strategy"].unique()
    
    test_results = []
    
    # 1. Compare chunking strategies
    for strat in strategies:
        df_strat = df_k5[df_k5["Strategy"] == strat]
        for c1, c2 in itertools.combinations(collections, 2):
            scores1 = df_strat[df_strat["Collection"] == c1].sort_values("QueryID")["nDCG"].values
            scores2 = df_strat[df_strat["Collection"] == c2].sort_values("QueryID")["nDCG"].values
            
            if len(scores1) > 1 and len(scores1) == len(scores2):
                diffs = scores1 - scores2
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
                    "Significant (p<0.05)": bool(p_ttest < 0.05 or p_wilcoxon < 0.05)
                })
                
    # 2. Compare retrieval architectures
    for coll in collections:
        df_coll = df_k5[df_k5["Collection"] == coll]
        for s1, s2 in itertools.combinations(strategies, 2):
            scores1 = df_coll[df_coll["Strategy"] == s1].sort_values("QueryID")["nDCG"].values
            scores2 = df_coll[df_coll["Strategy"] == s2].sort_values("QueryID")["nDCG"].values
            
            if len(scores1) > 1 and len(scores1) == len(scores2):
                diffs = scores1 - scores2
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
                    "Significant (p<0.05)": bool(p_ttest < 0.05 or p_wilcoxon < 0.05)
                })
                
    res_df = pd.DataFrame(test_results)
    os.makedirs("data/processed", exist_ok=True)
    res_df.to_csv("data/processed/statistical_significance.csv", index=False)
    print("Statistical tests complete. Saved to data/processed/statistical_significance.csv")

if __name__ == "__main__":
    run_tests()
