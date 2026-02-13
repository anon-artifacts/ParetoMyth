#!/usr/bin/env python3
"""
CORRECT Big Valley Test: Local Fitness Smoothness

Big Valley in SE means:
"Nearby points in configuration space have similar fitness"

This is what enables:
- Hill climbing to work (can follow gradients)
- Basins to exist (local structure)
- Regional zoom to be effective (local exploitation)

Test:
1. For each point, measure fitness
2. Measure fitness of k-nearest neighbors
3. Compute correlation between point fitness and neighbor fitness
4. Big Valley: Strong positive correlation (rho > 0.5)

This is the REAL Big Valley test that matters for your argument!
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.configurations.model_config_static import ModelConfigurationStatic
from models.model_wrapper_static import ModelWrapperStatic
from utils.data_loader_templated import load_data
from utils.EncodingUtils import EncodingUtils
from models.Data import Data
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_PATH = "../moot/optimize"


def test_local_smoothness(csv_file):
    """
    Test Big Valley as local fitness smoothness:
    "Nearby points have similar fitness"
    
    This is what actually matters for:
    - Hill climbing effectiveness
    - Basin structure
    - Regional zoom justification
    """
    try:
        X_raw, Y_raw = load_data(csv_file)
        col_types = {col: EncodingUtils.infer_column_type(X_raw[col].tolist())[0] 
                     for col in X_raw.columns}
        X_enc = EncodingUtils.encode_dataframe(X_raw, col_types)
        cols = list(X_enc.columns)
        
        data_nn = Data(X_enc.values.tolist(), col_types)
        hyperparameter_configs = {col: X_enc[col].tolist() for col in X_enc.columns}
        model_config = ModelConfigurationStatic(
            hyperparameter_configs, csv_file, 1, column_types=col_types
        )
        model_wrapper = ModelWrapperStatic(X_enc, Y_raw, model_config)
        
        n_rows = len(X_enc)
        sample_size = min(n_rows, 1000)
        np.random.seed(42)
        start_indices = np.random.choice(n_rows, sample_size, replace=False)
        k_neighbors = max(5, int(n_rows * 0.01))
        
        # Test 1: Point fitness vs average neighbor fitness
        point_fitness = []
        avg_neighbor_fitness = []
        
        # Test 2: Point fitness vs best neighbor fitness  
        best_neighbor_fitness = []
        
        # Test 3: Fitness variance within neighborhoods
        neighborhood_variances = []
        
        for idx in start_indices:
            # Fitness at this point
            start_row = data_nn.rows[idx]
            start_dict = {c: start_row[i] for i, c in enumerate(cols)}
            fitness = model_wrapper.run_model(start_dict)
            
            # Fitness of k neighbors
            neighbor_indices = data_nn.k_nearest_indices(idx, k=k_neighbors)
            neighbor_fitnesses = []
            for n_idx in neighbor_indices:
                n_row = data_nn.rows[n_idx]
                n_dict = {c: n_row[i] for i, c in enumerate(cols)}
                n_fitness = model_wrapper.run_model(n_dict)
                neighbor_fitnesses.append(n_fitness)
            
            point_fitness.append(fitness)
            avg_neighbor_fitness.append(np.mean(neighbor_fitnesses))
            best_neighbor_fitness.append(np.max(neighbor_fitnesses))
            neighborhood_variances.append(np.var(neighbor_fitnesses))
        
        # Correlation 1: Point vs average neighbor
        rho_avg, p_avg = spearmanr(point_fitness, avg_neighbor_fitness)
        
        # Correlation 2: Point vs best neighbor
        rho_best, p_best = spearmanr(point_fitness, best_neighbor_fitness)
        
        # Smoothness metric: Low variance = smooth
        mean_variance = np.mean(neighborhood_variances)
        global_variance = np.var(point_fitness)
        smoothness_ratio = 1 - (mean_variance / global_variance) if global_variance > 0 else 0
        
        # Hill climbing success rate (does best neighbor improve?)
        improvements = [best - point for point, best in zip(point_fitness, best_neighbor_fitness)]
        improvement_rate = np.mean([1 if imp > 0.001 else 0 for imp in improvements])
        avg_improvement = np.mean([imp for imp in improvements if imp > 0])
        
        # Decision criteria for Big Valley
        # Strong: rho > 0.7, Moderate: rho > 0.5
        big_valley_strong = rho_avg > 0.7
        big_valley_moderate = rho_avg > 0.5
        big_valley_smooth = smoothness_ratio > 0.3
        big_valley_hillclimb = improvement_rate > 0.7
        
        # Overall: Need high correlation OR (smooth + hillclimb works)
        big_valley = big_valley_moderate or (big_valley_smooth and big_valley_hillclimb)
        
        result = {
            "Dataset": os.path.basename(csv_file),
            "Rows": n_rows,
            "Rho_AvgNeighbor": rho_avg,
            "Rho_BestNeighbor": rho_best,
            "SmoothnessRatio": smoothness_ratio,
            "ImprovementRate": improvement_rate,
            "AvgImprovement": avg_improvement if improvement_rate > 0 else 0,
            "BigValley": "Yes" if big_valley else "No",
            "BigValleyStrength": "Strong" if big_valley_strong else "Moderate" if big_valley_moderate else "Weak",
            "SampleSize": len(point_fitness),
        }
        
        status = "✓✓" if big_valley_strong else "✓ " if big_valley else "✗ "
        print(f"{status} {os.path.basename(csv_file):30s} rho={rho_avg:6.3f} smooth={smoothness_ratio:5.3f} climb={improvement_rate:5.3f}")
        
        return result
        
    except Exception as e:
        print(f"ERROR {os.path.basename(csv_file)}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    results = []
    csv_files = []
    
    print("Scanning datasets...")
    for root, _, files in os.walk(BASE_PATH):
        for f in files:
            if f.endswith(".csv"):
                csv_path = os.path.join(root, f)
                try:
                    _, Y_raw = load_data(csv_path)
                    if Y_raw.shape[1] > 1:
                        csv_files.append(csv_path)
                except:
                    pass
    
    print(f"Found {len(csv_files)} datasets\n")
    print(f"Big Valley Test: Local Fitness Smoothness")
    print(f"Criteria: Nearby points have similar fitness (enables hill climbing)")
    print(f"\n{'St'} {'Dataset':<30s} {'Rho':>6s} {'Smooth':>6s} {'Climb':>6s}")
    print("=" * 60)
    
    # Parallel processing
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(test_local_smoothness, csv): csv for csv in csv_files}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Fatal error: {e}")
    
    # Save and analyze
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Rho_AvgNeighbor', ascending=False)
    results_df.to_csv("big_valley_smoothness_test.csv", index=False)
    
    print("\n" + "="*70)
    print("BIG VALLEY ANALYSIS - LOCAL SMOOTHNESS")
    print("="*70)
    
    total = len(results_df)
    big_valley = (results_df['BigValley'] == 'Yes').sum()
    strong = (results_df['BigValleyStrength'] == 'Strong').sum()
    moderate = (results_df['BigValleyStrength'] == 'Moderate').sum()
    
    print(f"\nTotal datasets: {total}")
    print(f"Big Valley structure: {big_valley} ({100*big_valley/total:.1f}%)")
    print(f"  Strong (rho > 0.7):   {strong} ({100*strong/total:.1f}%)")
    print(f"  Moderate (rho > 0.5): {moderate} ({100*moderate/total:.1f}%)")
    
    print(f"\nCorrelation statistics:")
    print(f"  Median rho: {results_df['Rho_AvgNeighbor'].median():.3f}")
    print(f"  Mean rho: {results_df['Rho_AvgNeighbor'].mean():.3f}")
    
    print(f"\nHill climbing metrics:")
    print(f"  Median improvement rate: {results_df['ImprovementRate'].median():.1%}")
    print(f"  Mean improvement rate: {results_df['ImprovementRate'].mean():.1%}")
    print(f"  Datasets with >70% climb success: {(results_df['ImprovementRate'] > 0.7).sum()}")
    
    print("\n** INTERPRETATION **")
    print("Big Valley = 'Nearby points have similar fitness'")
    print("This enables:")
    print("  1. Hill climbing (can follow gradients)")
    print("  2. Basin structure (local regions of quality)")
    print("  3. Regional zoom effectiveness (local exploitation)")
    
    if big_valley / total >= 0.9:
        print(f"\n✓✓ STRONG EVIDENCE: {100*big_valley/total:.0f}% show Big Valley")
        print("   → Matches paper's claim")
        print("   → Regional zoom theoretically justified")
    elif big_valley / total >= 0.7:
        print(f"\n✓ GOOD EVIDENCE: {100*big_valley/total:.0f}% show Big Valley")
        print("   → Most datasets support regional zoom")
    else:
        print(f"\n⚠ MIXED EVIDENCE: Only {100*big_valley/total:.0f}% show Big Valley")
        print("   → May need additional explanation")
    
    print(f"\nResults saved to: big_valley_smoothness_test.csv")
    
    # Top examples
    print(f"\nTop 10 strongest Big Valley (highest local smoothness):")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['Dataset']:30s} rho={row['Rho_AvgNeighbor']:.3f} ({row['BigValleyStrength']})")


if __name__ == "__main__":
    main()