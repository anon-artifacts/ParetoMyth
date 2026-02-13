#!/usr/bin/env python3
"""
Big Valley Analysis - Save-as-you-go Version
Implements RQ1 methodology: Spearman correlation between fitness and d2h.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- MOOT ENVIRONMENT SETUP ---
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.configurations.model_config_static import ModelConfigurationStatic
from models.model_wrapper_static import ModelWrapperStatic
from utils.data_loader_templated import load_data
from utils.EncodingUtils import EncodingUtils
from models.Data import Data

BASE_PATH = "../moot/optimize"
OUTPUT_FILE = "big_valley_fdc_incremental.csv"

def hill_climb_to_optimum(start_idx, data_nn, model_wrapper, columns, k):
    """
    Finds the nearest local optimum (peak) as per paper methodology.
    """
    current_idx = start_idx
    visited = {current_idx}
    
    while True:
        current_row = data_nn.rows[current_idx]
        current_dict = {c: current_row[i] for i, c in enumerate(columns)}
        current_fitness = model_wrapper.run_model(current_dict)
        
        neighbor_indices = data_nn.k_nearest_indices(current_idx, k=k)
        best_neighbor_idx = current_idx
        best_neighbor_fitness = current_fitness
        
        for n_idx in neighbor_indices:
            if n_idx in visited:
                continue
            
            n_row = data_nn.rows[n_idx]
            n_dict = {c: n_row[i] for i, c in enumerate(columns)}
            n_fitness = model_wrapper.run_model(n_dict)
            
            if n_fitness > best_neighbor_fitness:
                best_neighbor_fitness = n_fitness
                best_neighbor_idx = n_idx
        
        if best_neighbor_idx == current_idx:
            break
            
        current_idx = best_neighbor_idx
        visited.add(current_idx)
        
    return current_idx

def run_big_valley_rq1(csv_file):
    """
    Executes Big Valley test: Spearman correlation (fitness vs distance to hill)[cite: 219].
    """
    try:
        X_raw, Y_raw = load_data(csv_file)
        col_types = {col: EncodingUtils.infer_column_type(X_raw[col].tolist())[0] 
                     for col in X_raw.columns}
        X_enc = EncodingUtils.encode_dataframe(X_raw, col_types)
        cols = list(X_enc.columns)
        
        data_nn = Data(X_enc.values.tolist(), col_types)
        model_config = ModelConfigurationStatic(
            {col: X_enc[col].tolist() for col in cols}, csv_file, 1, column_types=col_types
        )
        model_wrapper = ModelWrapperStatic(X_enc, Y_raw, model_config)
        
        n_rows = len(X_enc)
        sample_size = min(n_rows, 1000) 
        np.random.seed(42)
        start_indices = np.random.choice(n_rows, sample_size, replace=False)
        
        k_neighbors = max(5, int(n_rows * 0.01))
        fitness_values = []
        d2h_distances = []
        
        for idx in start_indices:
            row = data_nn.rows[idx]
            row_dict = {c: row[i] for i, c in enumerate(cols)}
            fitness = model_wrapper.run_model(row_dict)
            
            peak_idx = hill_climb_to_optimum(idx, data_nn, model_wrapper, cols, k_neighbors)
            d2h = data_nn.xdist(data_nn.rows[idx], data_nn.rows[peak_idx])
            
            fitness_values.append(fitness)
            d2h_distances.append(d2h)
        
        rho, pval = spearmanr(fitness_values, d2h_distances)
        # Structure holds when correlation < -0.5[cite: 220].
        is_big_valley = "Yes" if rho < -0.5 else "No" 
        
        return [os.path.basename(csv_file), n_rows, round(rho, 4), pval, is_big_valley]
    except Exception as e:
        return None

def main():
    csv_files = []
    for root, _, files in os.walk(BASE_PATH):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    
    # Initialize CSV header
    header = ["Dataset", "Rows", "SpearmanRho", "PValue", "BigValley"]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print(f"Analyzing {len(csv_files)} datasets. Saving results to {OUTPUT_FILE}...")
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_big_valley_rq1, csv_path): csv_path for csv_path in csv_files}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                # Append each result immediately to the CSV
                with open(OUTPUT_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(result)

    # Final summary display
    final_df = pd.DataFrame(pd.read_csv(OUTPUT_FILE))
    bv_percent = (final_df['BigValley'] == 'Yes').mean() * 100
    print(f"\nAnalysis complete. Big Valley prevalence: {bv_percent:.1f}%[cite: 11, 223].")

if __name__ == "__main__":
    main()