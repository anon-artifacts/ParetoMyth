#!/usr/bin/env python3
"""
CORRECT Big Valley Test - Exactly as Paper Describes

From paper (page 12):
"For each dataset, we run 1000 evaluations with random sampling
and measure d2h (distance to nearest local optimum, identified via hill-climbing
from each sampled point). We compute the Spearman correlation between fitness
and d2h. Big Valley structure holds when correlation < −0.5"

KEY: 
- "fitness" = d2h value at starting point (quality metric)
- "d2h" in correlation = SPATIAL distance to the peak in configuration space
- Both in normalized [0,1] space
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.configurations.model_config_static import ModelConfigurationStatic
from models.model_wrapper_static import ModelWrapperStatic
from utils.data_loader_templated import load_data
from utils.EncodingUtils import EncodingUtils
from models.Data import Data

BASE_PATH = "../moot/optimize"

def enhanced_tabular_hill_climb(start_idx, data_nn, model_wrapper, columns, k, max_iterations=100):
    """Hill climb to find nearest local optimum"""
    current_idx = start_idx
    visited = {current_idx}
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        current_row = data_nn.rows[current_idx]
        current_dict = {c: current_row[i] for i, c in enumerate(columns)}
        current_quality = model_wrapper.run_model(current_dict)
        
        neighbor_indices = data_nn.k_nearest_indices(current_idx, k=k)
        
        best_n = current_idx
        best_quality = current_quality
        
        for n_idx in neighbor_indices:
            if n_idx in visited: 
                continue
            
            n_row = data_nn.rows[n_idx]
            n_dict = {c: n_row[i] for i, c in enumerate(columns)}
            n_quality = model_wrapper.run_model(n_dict)
            
            if n_quality > best_quality:
                best_quality = n_quality
                best_n = n_idx
                improved = True
        
        if improved:
            current_idx = best_n
            visited.add(current_idx)
    
    return current_idx


def run_big_valley_correct(csv_file):
    """
    Implement Big Valley test EXACTLY as paper describes:
    
    1. Sample 1000 random starting points
    2. For each, compute FITNESS = run_model() output (higher is better)
    3. Hill climb to find nearest local optimum (peak)
    4. Measure D2H = spatial distance in config space to that peak
    5. Correlate: fitness vs d2h
    6. Expected: NEGATIVE correlation (better fitness → closer to peaks)
    
    Big Valley interpretation:
    - fitness = quality metric (higher = better)
    - d2h = "distance to hill" = spatial distance to nearest local optimum
    - Negative correlation: high-quality points are spatially close to peaks
    """
    try:
        X_raw, Y_raw = load_data(csv_file)
        col_types = {col: EncodingUtils.infer_column_type(X_raw[col].tolist())[0] 
                     for col in X_raw.columns}
        X_enc = EncodingUtils.encode_dataframe(X_raw, col_types)
        cols = list(X_enc.columns)
        
        # Data class normalizes X internally for distance computation
        data_nn = Data(X_enc.values.tolist(), col_types)
        
        hyperparameter_configs = {col: X_enc[col].tolist() for col in X_enc.columns}
        model_config = ModelConfigurationStatic(
            hyperparameter_configs, csv_file, 1, column_types=col_types
        )
        # ModelWrapper normalizes Y internally
        model_wrapper = ModelWrapperStatic(X_enc, Y_raw, model_config)
        
        # Sample 1000 evaluations (or less if dataset is smaller)
        n_rows = len(X_enc)
        sample_size = min(n_rows, 1000)
        np.random.seed(42)  # For reproducibility
        start_indices = np.random.choice(n_rows, sample_size, replace=False)
        
        k_neighbors = max(5, int(n_rows * 0.01))
        
        fitness_values = []  # fitness at starting point (higher = better)
        d2h_distances = []  # spatial distance to nearest local optimum (d2h = distance to hill)
        
        for idx in start_indices:
            start_row = data_nn.rows[idx]
            start_dict = {c: start_row[i] for i, c in enumerate(cols)}
            
            # FITNESS = run_model() output (higher = better quality)
            fitness = model_wrapper.run_model(start_dict)
            
            # Find nearest local optimum via hill climbing
            peak_idx = enhanced_tabular_hill_climb(
                idx, data_nn, model_wrapper, cols, k_neighbors
            )
            
            # D2H = spatial distance to that peak in normalized configuration space
            # This is "distance to hill" - the nearest local optimum
            d2h = data_nn.xdist(data_nn.rows[idx], data_nn.rows[peak_idx])
            
            fitness_values.append(fitness)
            d2h_distances.append(d2h)
        
        # Correlation: fitness vs d2h (spatial distance to nearest peak)
        # Expected for Big Valley: NEGATIVE correlation
        # Higher fitness (better quality) → smaller d2h (closer to peak)
        rho, pval = spearmanr(fitness_values, d2h_distances)
        
        # Statistics
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        mean_d2h = np.mean(d2h_distances)
        std_d2h = np.std(d2h_distances)
        
        result = {
            "Dataset": os.path.basename(csv_file),
            "Rows": n_rows,
            "SpearmanRho": rho,
            "PValue": pval,
            "BigValley": "Yes" if rho < -0.5 else "No",
            "MeanFitness": mean_fitness,
            "StdFitness": std_fitness,
            "MeanD2H": mean_d2h,
            "StdD2H": std_d2h,
            "SampleSize": len(fitness_values),
        }
        
        print(f"{os.path.basename(csv_file):30s} rho={rho:7.3f}  BigValley={'YES' if rho < -0.5 else 'NO '}")
        
        return result
        
    except Exception as e:
        print(f"ERROR {os.path.basename(csv_file)}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    results = []
    csv_files = []
    
    print("Scanning for multi-objective datasets...")
    for root, _, files in os.walk(BASE_PATH):
        for f in files:
            if f.endswith(".csv"):
                csv_path = os.path.join(root, f)
                try:
                    _, Y_raw = load_data(csv_path)
                    if Y_raw.shape[1] == 1:  # Multi-objective only
                        csv_files.append(csv_path)
                except:
                    pass
    
    print(f"Found {len(csv_files)} multi-objective datasets\n")
    print(f"{'Dataset':<30s} {'Rho':>7s}  BigValley")
    print("=" * 50)
    
    # Parallel processing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_big_valley_correct, csv): csv for csv in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Fatal error: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('SpearmanRho')
    results_df.to_csv("big_valley_results_final.csv", index=False)
    
    # Summary statistics
    print("\n" + "="*80)
    print("BIG VALLEY ANALYSIS - FINAL RESULTS")
    print("="*80)
    
    total = len(results_df)
    big_valley_count = (results_df['SpearmanRho'] < -0.5).sum()
    
    print(f"\nTotal datasets analyzed: {total}")
    print(f"Big Valley (rho < -0.5): {big_valley_count} ({100*big_valley_count/total:.1f}%)")
    print(f"Non-Big Valley: {total - big_valley_count} ({100*(total-big_valley_count)/total:.1f}%)")
    
    print(f"\nCorrelation statistics:")
    print(f"  Median rho: {results_df['SpearmanRho'].median():.3f}")
    print(f"  Mean rho: {results_df['SpearmanRho'].mean():.3f}")
    print(f"  Min rho: {results_df['SpearmanRho'].min():.3f}")
    print(f"  Max rho: {results_df['SpearmanRho'].max():.3f}")
    
    # Distribution by correlation strength
    strong_neg = (results_df['SpearmanRho'] < -0.5).sum()
    weak_neg = ((results_df['SpearmanRho'] >= -0.5) & (results_df['SpearmanRho'] < -0.2)).sum()
    near_zero = ((results_df['SpearmanRho'] >= -0.2) & (results_df['SpearmanRho'] <= 0.2)).sum()
    weak_pos = ((results_df['SpearmanRho'] > 0.2) & (results_df['SpearmanRho'] <= 0.5)).sum()
    strong_pos = (results_df['SpearmanRho'] > 0.5).sum()
    
    print(f"\nCorrelation distribution:")
    print(f"  Strong negative (< -0.5):     {strong_neg:3d} ({100*strong_neg/total:5.1f}%) ← Big Valley")
    print(f"  Weak negative (-0.5 to -0.2): {weak_neg:3d} ({100*weak_neg/total:5.1f}%)")
    print(f"  Near zero (-0.2 to 0.2):      {near_zero:3d} ({100*near_zero/total:5.1f}%)")
    print(f"  Weak positive (0.2 to 0.5):   {weak_pos:3d} ({100*weak_pos/total:5.1f}%)")
    print(f"  Strong positive (> 0.5):      {strong_pos:3d} ({100*strong_pos/total:5.1f}%) ← Anti-Big Valley")
    
    if big_valley_count / total >= 0.85:
        print(f"\n✅ MATCHES PAPER: ~{100*big_valley_count/total:.0f}% have Big Valley structure")
    elif strong_pos / total >= 0.5:
        print(f"\n⚠️  ANTI-BIG VALLEY DETECTED: {100*strong_pos/total:.0f}% show POSITIVE correlation")
        print(f"    This contradicts the paper's findings!")
    else:
        print(f"\n⚠️  WEAK STRUCTURE: Most datasets show weak/near-zero correlations")
        print(f"    This suggests spatial distance is not meaningful (curse of dimensionality)")
    
    print(f"\nTop 10 strongest Big Valley datasets (most negative rho):")
    top_bv = results_df.nsmallest(10, 'SpearmanRho')
    for _, row in top_bv.iterrows():
        print(f"  {row['Dataset']:30s} rho={row['SpearmanRho']:7.3f}")
    
    print(f"\nTop 10 strongest Anti-Big Valley datasets (most positive rho):")
    bottom_bv = results_df.nlargest(10, 'SpearmanRho')
    for _, row in bottom_bv.iterrows():
        print(f"  {row['Dataset']:30s} rho={row['SpearmanRho']:7.3f}")
    
    print(f"\nResults saved to: big_valley_results_final.csv")


if __name__ == "__main__":
    main()