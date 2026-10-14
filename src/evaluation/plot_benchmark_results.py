import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import PROCESSED_DIR

PLOTS_DIR = PROCESSED_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_benchmark_df() -> pd.DataFrame:
    csv_path = PROCESSED_DIR / "benchmark_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot generate plots; {csv_path} does not exist.")
    return pd.read_csv(csv_path)


def plot_ndcg_by_strategy(df: pd.DataFrame):
    col_metric = "ndcg_at_k" if "ndcg_at_k" in df.columns else ("nDCG" if "nDCG" in df.columns else "nDCG@K")
    col_strat = "strategy" if "strategy" in df.columns else "Strategy"
    col_chunk = "collection" if "collection" in df.columns else "Collection"

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=col_chunk, y=col_metric, hue=col_strat, errorbar=("ci", 95))
    plt.title("nDCG@K Across Chunking Collections & Retrieval Strategies")
    plt.ylabel("Mean nDCG@K")
    plt.xlabel("Collection")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ndcg_by_strategy.png", dpi=300)
    plt.close()


def plot_latency_breakdown(df: pd.DataFrame):
    col_strat = "strategy" if "strategy" in df.columns else "Strategy"
    col_lat = (
        "total_latency_ms"
        if "total_latency_ms" in df.columns
        else ("TotalRetrievalMS" if "TotalRetrievalMS" in df.columns else "TotalLatency_ms")
    )

    if col_lat not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=col_strat, y=col_lat)
    plt.yscale("log")
    plt.title("Retrieval Latency Distribution by Strategy (Log Scale)")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Retrieval Strategy")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_breakdown.png", dpi=300)
    plt.close()


def plot_precision_recall_tradeoff(df: pd.DataFrame):
    col_p = (
        "precision_at_k"
        if "precision_at_k" in df.columns
        else ("Precision" if "Precision" in df.columns else "Precision@K")
    )
    col_r = (
        "recall_at_k"
        if "recall_at_k" in df.columns
        else ("Recall" if "Recall" in df.columns else "Recall@K")
    )
    col_strat = "strategy" if "strategy" in df.columns else "Strategy"
    col_chunk = "collection" if "collection" in df.columns else "Collection"

    agg = df.groupby([col_chunk, col_strat])[[col_p, col_r]].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=agg, x=col_r, y=col_p, hue=col_chunk, style=col_strat, s=140)
    plt.title("Precision@K vs. Recall@K Across Configurations")
    plt.xlabel("Mean Recall@K")
    plt.ylabel("Mean Precision@K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_tradeoff.png", dpi=300)
    plt.close()


def main():
    df = load_benchmark_df()
    plot_ndcg_by_strategy(df)
    plot_latency_breakdown(df)
    plot_precision_recall_tradeoff(df)
    print(f"Saved data-driven benchmark plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
