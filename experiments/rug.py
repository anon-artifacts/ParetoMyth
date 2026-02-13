#!/usr/bin/env python3
"""
Ruggedness Analysis (Fitness Autocorrelation)
Measures: Spearman correlation between point fitness and neighbor fitness.
High Rho (> 0.5) = Smooth Landscape
Low Rho (< 0.5) = Rugged Landscape
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
OUTPUT_FILE = "landscape_ruggedness_results.csv"

def run_ruggedness_test(csv_file):
    """
    Measures local ruggedness via Fitness Autocorrelation.
    """
    try:
        # Load and Encode Data
        X_raw, Y_raw = load_data(csv_file)
        col_types = {col: EncodingUtils.infer_column_type(X_raw[col].tolist())[0] 
                     for col in X_raw.columns}
        X_enc = EncodingUtils.encode_dataframe(X_raw, col_types)
        cols = list(X_enc.columns)
        
        # Initialize internal MOOT tools [cite: 162, 733]
        data_nn = Data(X_enc.values.tolist(), col_types)
        model_config = ModelConfigurationStatic(
            {col: X_enc[col].tolist() for col in cols}, csv_file, 1, column_types=col_types
        )
        model_wrapper = ModelWrapperStatic(X_enc, Y_raw, model_config)
        
        # Sampling budget based on NEO problem types [cite: 661, 662]
        n_rows = len(X_enc)
        sample_size = min(n_rows, 1000) 
        np.random.seed(42)
        start_indices = np.random.choice(n_rows, sample_size, replace=False)
        
        # Define k-neighborhood (1% of data or min 1)
        k_neighbors = 1 
        
        point_fitness = []
        neighbor_fitness = []
        
        for idx in start_indices:
            # Current Point Fitness
            row = data_nn.rows[idx]
            row_dict = {c: row[i] for i, c in enumerate(cols)}
            fitness = model_wrapper.run_model(row_dict)
            
            # Nearest Neighbor Fitness (Autocorrelation) [cite: 160]
            nn_idx = data_nn.k_nearest_indices(idx, k=k_neighbors)[0]
            nn_row = data_nn.rows[nn_idx]
            nn_dict = {c: nn_row[i] for i, c in enumerate(cols)}
            nn_fitness = model_wrapper.run_model(nn_dict)
            
            point_fitness.append(fitness)
            neighbor_fitness.append(nn_fitness)
        
        # Compute Spearman Correlation (Autocorrelation)
        rho, pval = spearmanr(point_fitness, neighbor_fitness)
        
        # Interpretation: Higher Rho = Smoother, Lower Rho = More Rugged
        # Ruggedness is often defined as (1 - Rho)
        is_rugged = "Yes" if rho < 0.5 else "No"
        
        return [os.path.basename(csv_file), n_rows, round(rho, 4), round(1-rho, 4), is_rugged]
    except Exception as e:
        print(f"Error in {csv_file}: {e}")
        return None

def main():
    # Collect files from MOOT benchmark [cite: 7, 163]
    csv_files = []
    for root, _, files in os.walk(BASE_PATH):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    
    # Initialize output CSV with headers
    header = ["Dataset", "Rows", "Autocorrelation_Rho", "Ruggedness_Score", "Is_Rugged"]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print(f"Analyzing {len(csv_files)} datasets for Ruggedness. Saving to {OUTPUT_FILE}...")
    
    # Run in parallel to handle large row counts [cite: 165]
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_ruggedness_test, cp): cp for cp in csv_files}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                with open(OUTPUT_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(result)

    print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()