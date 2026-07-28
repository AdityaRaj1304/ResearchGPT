import msvc_runtime
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    strategies = ["whole_doc", "fixed", "overlap", "semantic"]
    data = {}
    for strategy in strategies:
        filepath = f"data/processed/chunks_{strategy}.parquet"
        if os.path.exists(filepath):
            data[strategy] = pd.read_parquet(filepath)
    return data

def plot_chunk_length_histograms(data, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    strategies = ["whole_doc", "fixed", "overlap", "semantic"]
    
    for i, strategy in enumerate(strategies):
        if strategy not in data:
            continue
        ax = axes[i // 2, i % 2]
        lengths = data[strategy]['token_length']
        sns.histplot(lengths, bins=50, ax=ax, color='skyblue')
        ax.set_title(f"Token Lengths: {strategy}")
        ax.set_xlabel("Token Length")
        ax.set_ylabel("Frequency")
        ax.axvline(x=512, color='red', linestyle='--', label='512 Token Cap')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chunk_length_histograms.png"), dpi=300)
    plt.close()

def plot_chunk_size_boxplot(data, output_dir):
    strategies = ["fixed", "overlap", "semantic"]
    combined_data = []
    
    for strategy in strategies:
        if strategy in data:
            df = data[strategy]
            for length in df['token_length']:
                combined_data.append({"Strategy": strategy, "Token Length": length})
                
    if not combined_data:
        return
        
    df_plot = pd.DataFrame(combined_data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Strategy", y="Token Length", data=df_plot, palette="Set2")
    plt.title("Chunk Token Length Distribution (Excluding whole_doc)")
    plt.axhline(y=512, color='red', linestyle='--', label='512 Token Cap')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chunk_size_boxplot.png"), dpi=300)
    plt.close()

def plot_chunks_per_paper_dist(data, output_dir):
    combined_data = []
    
    for strategy, df in data.items():
        chunks_per_paper = df.groupby('paper_id').size().reset_index(name='chunks')
        for count in chunks_per_paper['chunks']:
            combined_data.append({"Strategy": strategy, "Chunks per Paper": count})
            
    if not combined_data:
        return
        
    df_plot = pd.DataFrame(combined_data)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Strategy", y="Chunks per Paper", data=df_plot, palette="muted", inner="quartile")
    plt.title("Distribution of Chunks per Paper by Strategy")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chunks_per_paper_dist.png"), dpi=300)
    plt.close()

def main():
    data = load_data()
    output_dir = "data/processed/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating chunk_length_histograms.png...")
    plot_chunk_length_histograms(data, output_dir)
    
    print("Generating chunk_size_boxplot.png...")
    plot_chunk_size_boxplot(data, output_dir)
    
    print("Generating chunks_per_paper_dist.png...")
    plot_chunks_per_paper_dist(data, output_dir)
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    main()
