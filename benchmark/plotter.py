import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import re
import numpy as np

def clean_and_average(val):
    """Parses a string list '[0.9, 1.0]' and returns its simple average (float)."""
    if isinstance(val, (int, float)):
        return val
    try:
        # Clean numpy tags like np.float64()
        cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', str(val))
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            # Return the mean of the list
            return np.mean(parsed)
        return float(parsed)
    except:
        return 0.0

def plot_benchmark_results(csv_file: str, output_dir: str) -> None:
    if not os.path.exists(csv_file):
        print(f"❌ Error: Results file {csv_file} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(csv_file)
    
    # 2. Calculate Averages Explicitly
    # This collapses the lists "[0.9, 0.8]" into single numbers "0.85"
    for col in ['faithfulness', 'answer_relevancy']:
        if col in df.columns:
            df[col] = df[col].apply(clean_and_average)

    os.makedirs(output_dir, exist_ok=True)

    # ---------------- PLOT 1: QUALITY (Final Fix) ----------------
    quality_cols = [c for c in ['faithfulness', 'answer_relevancy'] if c in df.columns]
    
    if quality_cols:
        plt.figure(figsize=(10, 6))
        
        # Melt for plotting
        df_melted = df.melt(id_vars=['strategy', 'embedding_size'], 
                            value_vars=quality_cols, 
                            var_name='Metric', value_name='Score')
        
        # Plot
        ax = sns.barplot(data=df_melted, x='strategy', y='Score', hue='embedding_size', palette='viridis', errorbar=None)
        
        # --- SMART ZOOM CALCULATION ---
        # Find the lowest BAR height (not the lowest data point)
        min_bar_height = df_melted['Score'].min()
        
        # Set bottom limit 0.05 below the lowest bar
        # If the lowest bar is 0.60, chart starts at 0.55
        bottom_limit = max(0, min_bar_height - 0.05)
        
        plt.ylim(bottom_limit, 1.0) 
        plt.title(f"RAG Quality Scores (Zoomed: {bottom_limit:.2f} - 1.0)")
        plt.grid(axis='y', alpha=0.3)
        plt.ylabel("Average Score (0-1)")

        # Add labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10, fontweight='bold')

        plt.savefig(os.path.join(output_dir, "benchmark_quality.png"))
        plt.close()
        print(f"✅ Saved Quality Plot to {output_dir}/benchmark_quality.png")

    # ---------------- PLOT 2: LATENCY ----------------
    latency_cols = [c for c in ['avg_retrieval_sec', 'avg_gen_sec'] if c in df.columns]
    
    if latency_cols:
        plt.figure(figsize=(10, 6))
        df_melted = df.melt(id_vars=['strategy', 'embedding_size'], 
                            value_vars=latency_cols, 
                            var_name='Metric', value_name='Seconds')
        
        # Plot
        ax = sns.barplot(data=df_melted, x='strategy', y='Seconds', hue='embedding_size', palette='magma', errorbar=None)
        
        plt.title("System Latency (Lower is Better)")
        plt.yscale('log')
        plt.ylabel("Seconds (Log Scale)")
        plt.grid(axis='y', alpha=0.3, which="both")
        
        # Add labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

        plt.savefig(os.path.join(output_dir, "benchmark_latency.png"))
        plt.close()
        print(f"✅ Saved Latency Plot to {output_dir}/benchmark_latency.png")

if __name__ == "__main__":
    RESULTS_CSV = "benchmark/results/final_benchmark_results_part1.csv"
    OUTPUT_DIR = "benchmark/results"
    plot_benchmark_results(RESULTS_CSV, OUTPUT_DIR)