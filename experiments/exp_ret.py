#!/usr/bin/env python3
"""
Direct Test of Paper's Core Claim: Diminishing Returns

Instead of testing Big Valley (proxy), test the ACTUAL claim:
"Expected improvement from k additional evaluations decays as O(1/√k)"

This is what justifies regional zoom vs global exploration!

Mathematical Test:
1. Start from a good point
2. Measure improvement after k steps: I(k)
3. Show: I(k) ~ C/√k (diminishing returns)
4. Contrast with random search (constant returns)

If diminishing returns exist → regional zoom is justified
(Whether or not Big Valley spatial clustering exists!)
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.configurations.model_config_static import ModelConfigurationStatic
from models.model_wrapper_static import ModelWrapperStatic
from utils.data_loader_templated import load_data
from utils.EncodingUtils import EncodingUtils
from models.Data import Data

BASE_PATH = "../moot/optimize"

def hill_climb_with_trajectory(start_idx, data_nn, model_wrapper, columns, k, max_steps=50):
    """Hill climb and record fitness at each step"""
    trajectory = []
    current_idx = start_idx
    visited = {current_idx}
    
    for step in range(max_steps):
        current_row = data_nn.rows[current_idx]
        current_dict = {c: current_row[i] for i, c in enumerate(columns)}
        current_fitness = model_wrapper.run_model(current_dict)
        trajectory.append(current_fitness)
        
        neighbor_indices = data_nn.k_nearest_indices(current_idx, k=k)
        best_n = current_idx
        best_fitness = current_fitness
        
        for n_idx in neighbor_indices:
            if n_idx in visited:
                continue
            n_row = data_nn.rows[n_idx]
            n_dict = {c: n_row[i] for i, c in enumerate(columns)}
            n_fitness = model_wrapper.run_model(n_dict)
            if n_fitness > best_fitness:
                best_fitness = n_fitness
                best_n = n_idx
        
        if best_n == current_idx:
            # Converged - fill rest with final value
            trajectory.extend([current_fitness] * (max_steps - step - 1))
            break
        
        current_idx = best_n
        visited.add(current_idx)
    
    return trajectory


def test_diminishing_returns(csv_file):
    """
    Test if improvement decays as O(1/√k)
    
    Returns evidence for/against regional zoom justification
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
        sample_size = min(n_rows, 100)
        np.random.seed(42)
        start_indices = np.random.choice(n_rows, sample_size, replace=False)
        k_neighbors = max(5, int(n_rows * 0.01))
        
        # Collect trajectories
        all_trajectories = []
        for idx in start_indices:
            traj = hill_climb_with_trajectory(idx, data_nn, model_wrapper, cols, k_neighbors)
            all_trajectories.append(traj)
        
        # Average improvement at each step
        avg_trajectory = np.mean(all_trajectories, axis=0)
        improvements = [avg_trajectory[k] - avg_trajectory[0] for k in range(len(avg_trajectory))]
        
        # Test O(1/√k) fit
        steps = np.arange(1, len(improvements))
        imp = np.array(improvements[1:])
        
        # Fit: I(k) = C/√k + offset
        def model(k, C, offset):
            return C / np.sqrt(k) + offset
        
        try:
            popt, _ = curve_fit(model, steps, imp, p0=[0.1, 0.0], maxfev=10000)
            C, offset = popt
            predicted = model(steps, C, offset)
            
            # R² goodness of fit
            ss_res = np.sum((imp - predicted) ** 2)
            ss_tot = np.sum((imp - np.mean(imp)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            fit_quality = "Good" if r_squared > 0.7 else "Moderate" if r_squared > 0.5 else "Poor"
        except:
            r_squared = 0
            fit_quality = "Failed"
        
        # Early vs late improvement comparison
        early_improvement = avg_trajectory[5] - avg_trajectory[0]  # First 5 steps
        late_improvement = avg_trajectory[25] - avg_trajectory[20] if len(avg_trajectory) > 25 else 0  # Steps 20-25
        
        diminishing_ratio = late_improvement / early_improvement if early_improvement > 0 else 1.0
        
        # Convergence rate
        converged_by_step = []
        for traj in all_trajectories:
            for step in range(1, len(traj)):
                if abs(traj[step] - traj[step-1]) < 0.001:
                    converged_by_step.append(step)
                    break
            else:
                converged_by_step.append(len(traj))
        
        median_convergence = np.median(converged_by_step)
        
        result = {
            "Dataset": os.path.basename(csv_file),
            "Rows": n_rows,
            "R2_Fit": r_squared,
            "FitQuality": fit_quality,
            "EarlyImprovement": early_improvement,
            "LateImprovement": late_improvement,
            "DiminishingRatio": diminishing_ratio,
            "MedianConvergence": median_convergence,
            "AvgFinalFitness": np.mean([t[-1] for t in all_trajectories]),
            "DiminishingReturns": "Yes" if diminishing_ratio < 0.3 else "No",
        }
        
        status = "✓" if diminishing_ratio < 0.3 else "✗"
        print(f"{status} {os.path.basename(csv_file):30s} R²={r_squared:.2f} ratio={diminishing_ratio:.3f} conv@{median_convergence:.0f}")
        
        return result
        
    except Exception as e:
        print(f"ERROR {os.path.basename(csv_file)}: {e}")
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
    print(f"Testing Diminishing Returns (O(1/√k) decay)")
    print(f"{'Status'} {'Dataset':<30s} {'R² Fit':<8s} {'Late/Early':<10s} {'Converge'}")
    print("=" * 70)
    
    for csv_file in csv_files[:20]:  # Test first 20
        result = test_diminishing_returns(csv_file)
        if result:
            results.append(result)
    
    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv("diminishing_returns_test.csv", index=False)
    
    print("\n" + "="*70)
    print("SUMMARY: DIMINISHING RETURNS ANALYSIS")
    print("="*70)
    
    total = len(results_df)
    diminishing = (results_df['DiminishingReturns'] == 'Yes').sum()
    
    print(f"\nTotal datasets: {total}")
    print(f"Show diminishing returns: {diminishing} ({100*diminishing/total:.1f}%)")
    print(f"  (Late/Early improvement ratio < 0.3)")
    
    print(f"\nGoodness of fit to O(1/√k):")
    print(f"  Good fit (R² > 0.7):     {(results_df['R2_Fit'] > 0.7).sum()}")
    print(f"  Moderate fit (R² > 0.5): {(results_df['R2_Fit'] > 0.5).sum()}")
    
    print(f"\nMedian convergence step: {results_df['MedianConvergence'].median():.0f}")
    print(f"Mean late/early ratio: {results_df['DiminishingRatio'].mean():.3f}")
    
    print("\n** INTERPRETATION **")
    if diminishing / total > 0.8:
        print("✓ Strong evidence: >80% show diminishing returns")
        print("  → Regional zoom justified (early steps most valuable)")
    elif diminishing / total > 0.5:
        print("~ Moderate evidence: >50% show diminishing returns")
        print("  → Regional zoom justified for most tasks")
    else:
        print("✗ Weak evidence: <50% show diminishing returns")
        print("  → Need alternative explanation for EZR success")
    
    print(f"\nResults saved to: diminishing_returns_test.csv")


if __name__ == "__main__":
    main()