import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(results_path="data/processed/benchmark_results.csv", out_dir="data/processed/plots"):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing {results_path}")
        
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(results_path)
    
    df_k5 = df[df["Cutoff"] == 5]
    
    # 1. nDCG and Recall Comparison
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df_k5, x="Collection", y="nDCG", hue="Strategy", errorbar="se")
    plt.title("nDCG@5 across Collections & Strategies")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df_k5, x="Collection", y="Recall", hue="Strategy", errorbar="se")
    plt.title("Recall@5 across Collections & Strategies")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ndcg_recall_comparison.png"))
    plt.close()
    
    # 2. Latency vs Precision
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_k5, x="TotalRetrievalMS", y="Precision", hue="Strategy", style="Collection", s=100, alpha=0.7)
    plt.xscale("log")
    plt.title("Latency vs Precision@5 (Efficiency Frontier)")
    plt.xlabel("Total Retrieval Latency (ms) [Log Scale]")
    plt.ylabel("Precision@5")
    plt.grid(True, alpha=0.3)
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_vs_precision.png"))
    plt.close()
    
    # 3. Subgroup Performance by Difficulty
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_k5, x="Difficulty", y="nDCG", hue="Strategy")
    plt.title("nDCG@5 by Query Difficulty")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "subgroup_performance_by_difficulty.png"))
    plt.close()
    
    # 4. Generation Fidelity Matrix
    heatmap_data_cit = df_k5.pivot_table(index="Collection", columns="Strategy", values="CitationSupportRate", aggfunc="mean")
    heatmap_data_uns = df_k5.pivot_table(index="Collection", columns="Strategy", values="UnsupportedClaimRate", aggfunc="mean")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(heatmap_data_cit, annot=True, cmap="YlGnBu", ax=axes[0], vmin=0, vmax=100)
    axes[0].set_title("Mean Citation Support Rate (%)")
    
    sns.heatmap(heatmap_data_uns, annot=True, cmap="OrRd", ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title("Mean Unsupported Claim Rate (%)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "generation_fidelity_matrix.png"))
    plt.close()
    print(f"Plots generated and saved to {out_dir}")

if __name__ == "__main__":
    plot_results()
