#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

'''
Creation of dataset which contain just the main measp features and the best configuration for each instance,
with a filtering step to keep only the "significant" instances 
(where the best solver is at least 20% faster than the second best or the second best is a timeout)
'''

# --- GLOBAL CONSTANTS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMEOUT_VALUE = 1200.0

# --- FILTERING PARAMETERS ---
MIN_GAP_RATIO = 1.20  
MIN_TIME_RELEVANCE = 0.500

# Task definition: (Times File, Features File, Output Prefix)
TASKS = [
    (
        "../../extract_time_training.csv", 
        "../../feature_measp/features_all_general_training.csv", 
        "training"
    ),
    (
        "../../extract_time_ijcai.csv", 
        "../../feature_measp/features_all_general_ijcai.csv", 
        "ijcai"
    )
]

def estrai_dominio(instance_path):
    if "GraphColouring" in instance_path:
        return "GraphColouring"
    elif "GroupAssignment" in instance_path:
        return "GroupAssignment"
    elif "Knapsack" in instance_path:
        return "Knapsack"
    return "Unknown"

def semplifica_config(cmd):
    cmd_str = str(cmd).lower()
    if 'encoding-amosum-eo' in cmd_str:
        if 'l=true' in cmd_str and 'm=minfly' in cmd_str:
            return 'EOCLINGO-L'
        elif 'l=false' in cmd_str:
            return 'EOCLINGO-INF-MR'
    elif 'amoclingo' in cmd_str:
        if 'l=false' in cmd_str and 'm=minfly' in cmd_str:
            return 'AMOCLINGO-INF-MR'
        elif 'l=true' in cmd_str and 'm=minfly' in cmd_str:
            return 'AMOCLINGO-L'
    if 'wasp' in cmd_str:
        return 'WASP-AMO'
    elif 'clingo' in cmd_str and 'amoclingo' not in cmd_str:
        return 'CLINGO-AMO'
    return cmd_str

def process_dataset(df_times, valid_solvers, df_features, name_suffix, output_path, plots_dir):
    print(f"\n--- Processing Dataset: {name_suffix.upper()} ---")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Filter columns keeping only the valid solvers for this dataset
    present_solvers = [c for c in valid_solvers if c in df_times.columns]
    
    if len(present_solvers) < 2:
        print(f"Not enough valid solvers found in {name_suffix}. Found: {present_solvers}")
        return
        
    df_subset = df_times[['domain', 'instance_name'] + present_solvers].copy()
    
    # Calculate the two best times for each instance
    def get_best_and_gap(row):
        times = row[present_solvers].astype(float).sort_values()
        best_time = times.iloc[0]
        second_best_time = times.iloc[1] if len(times) > 1 else TIMEOUT_VALUE
        
        # If the best is a timeout or missing, the instance is unsolvable
        if pd.isna(best_time) or best_time >= TIMEOUT_VALUE:
            return pd.Series(['TIMEOUT', best_time, 1.0, False])
        
        # Calculate ratio (e.g., 2.0 means the second solver is twice as slow)
        ratio = second_best_time / max(best_time, 0.0001)
        
        # Significance condition
        only_one_solved = (second_best_time >= TIMEOUT_VALUE)
        is_significant = only_one_solved or ((best_time > MIN_TIME_RELEVANCE) and (ratio >= MIN_GAP_RATIO))
        
        return pd.Series([times.index[0], best_time, ratio, is_significant])

    res = df_subset.apply(get_best_and_gap, axis=1)
    df_subset[['best_config', 'min_time', 'performance_ratio', 'is_significant']] = res

    # Filter the dataset: keep only "significant" instances and exclude TIMEOUTs
    df_filtered = df_subset[(df_subset['is_significant'] == True) & (df_subset['best_config'] != 'TIMEOUT')]

    print(f"   -> Total evaluated instances: {len(df_subset)}")
    print(f"   -> Discarded instances (low significance or too fast): {len(df_subset) - len(df_filtered)}")
    print(f"   -> Kept instances: {len(df_filtered)}")

    # Specific plots generation
    sns.set_theme(style="whitegrid")
    
    # Performance Ratio Plot
    plt.figure(figsize=(10, 6))
    valid_ratios = df_subset[df_subset['best_config'] != 'TIMEOUT']['performance_ratio']
    if not valid_ratios.empty:
        sns.histplot(valid_ratios, bins=50, kde=True)
    plt.axvline(MIN_GAP_RATIO, color='red', linestyle='--', label=f'Threshold ({MIN_GAP_RATIO}x)')
    plt.title(f"Performance Gap Distribution ({name_suffix.upper()})")
    plt.xlabel("Ratio (How much the second is slower than the first)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"performance_gap_dist_{name_suffix}.png"))
    plt.close()

    # Winners Count Plot
    plt.figure(figsize=(10, 6))
    if not df_filtered.empty:
        order = df_filtered['best_config'].value_counts().index
        sns.countplot(data=df_filtered, y='best_config', order=order, palette='viridis')
    plt.title(f"Winners in filtered dataset ({name_suffix.upper()})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"filtered_winners_count_{name_suffix}.png"))
    plt.close()

    # Merge with features
    df_labels_only = df_filtered[['domain', 'instance_name', 'best_config', 'min_time', 'performance_ratio']]
    df_final = pd.merge(df_features, df_labels_only, on=['domain', 'instance_name'], how='inner')

    df_final.to_csv(output_path, index=False)
    print(f"   -> Dataset successfully saved: {output_path}")

def run_pipeline(times_rel_path, features_rel_path, label):
    """Executes the entire analysis pipeline for a specific input dataset."""
    print(f"\n🚀 STARTING PIPELINE FOR: {label.upper()}")
    
    times_path = os.path.join(BASE_DIR, times_rel_path)
    features_path = os.path.join(BASE_DIR, features_rel_path)
    
    if not os.path.exists(times_path) or not os.path.exists(features_path):
        print(f"⚠️ Skipping {label}: Files not found.\n   Check: {times_path}\n   Check: {features_path}")
        return

    # Data loading
    df_raw = pd.read_csv(times_path)
    df_features = pd.read_csv(features_path)
    
    # Base data preprocessing
    df_raw['domain'] = df_raw['instance'].apply(estrai_dominio)
    df_raw['instance_name'] = df_raw['instance'].apply(lambda x: os.path.basename(str(x)))
    df_raw['config'] = df_raw['config'].apply(semplifica_config)
    df_raw['cpu_time'] = pd.to_numeric(df_raw['cpu_time'], errors='coerce')

    # Pivot: columns = configurations (solvers)
    df_times = df_raw.pivot_table(index=['domain', 'instance_name'], columns='config', values='cpu_time').reset_index()
    
    # --- Solver groups definition ---
    base_solvers = ['CLINGO-AMO', 'WASP-AMO']
    amo_solvers = ['AMOCLINGO-INF-MR', 'AMOCLINGO-L']
    eo_solvers = ['EOCLINGO-INF-MR', 'EOCLINGO-L']
    
    # Dynamic output paths
    plots_dir = os.path.join(BASE_DIR, f"../plots/preprocessed_{label}")
    out_amo = os.path.join(BASE_DIR, f"{label}_dataset_measp_amo.csv")
    out_eo = os.path.join(BASE_DIR, f"{label}_dataset_measp_eo.csv")

    # Process and generate dataset for AMOCLINGO (Base + AMOCLINGO)
    process_dataset(df_times, base_solvers + amo_solvers, df_features, f"{label}_amo", out_amo, plots_dir)
    
    # Process and generate dataset for EOCLINGO (Base + EOCLINGO)
    process_dataset(df_times, base_solvers + eo_solvers, df_features, f"{label}_eo", out_eo, plots_dir)

def main():
    print("Starting execution...")
    # Loop through each task defined in the TASKS list
    for times_file, feat_file, label in TASKS:
        run_pipeline(times_file, feat_file, label)
        
    print("\n✅ All tasks completed successfully.")

if __name__ == "__main__":
    main()