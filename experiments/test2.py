#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
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
OUTPUT_FILE = "big_valley_zone_analysis.csv"

def analyze_zone_structure(csv_file):
    """
    Quantifies Big Valley structure via Zone Fraction and Distance Ratio.
    Zone V = solutions within 10% of the best observed d2h.
    """
    try:
        X_raw, Y_raw = load_data(csv_file)
        col_types = {col: EncodingUtils.infer_column_type(X_raw[col].tolist())[0] for col in X_raw.columns}
        X_enc = EncodingUtils.encode_dataframe(X_raw, col_types)
        cols = list(X_enc.columns)
        
        data_nn = Data(X_enc.values.tolist(), col_types)
        model_wrapper = ModelWrapperStatic(X_enc, Y_raw, ModelConfigurationStatic({c: X_enc[c].tolist() for c in cols}, csv_file, 1, column_types=col_types))
        
        # 1. Sample and calculate fitness for all points to define the "High-Quality Zone"
        all_fitness = []
        for row in data_nn.rows:
            row_dict = {c: row[i] for i, c in enumerate(cols)}
            all_fitness.append(model_wrapper.run_model(row_dict))
        
        all_fitness = np.array(all_fitness)
        best_f = np.max(all_fitness)
        threshold = best_f * 0.9  # Within 10% of the best
        
        # 2. Identify Zone Indices
        zone_indices = np.where(all_fitness >= threshold)[0]
        other_indices = np.where(all_fitness < threshold)[0]
        
        if len(zone_indices) < 2 or len(other_indices) < 2:
            return None

        # 3. Calculate Intra-Zone Distance (Average distance between points in V)
        intra_dists = []
        sample_size = min(len(zone_indices), 100)
        subset_v = np.random.choice(zone_indices, sample_size, replace=False)
        for i in range(len(subset_v)):
            for j in range(i + 1, len(subset_v)):
                intra_dists.append(data_nn.xdist(data_nn.rows[subset_v[i]], data_nn.rows[subset_v[j]]))
        
        mean_intra = np.mean(intra_dists)

        # 4. Calculate Inter-Zone Distance (Average distance from others to Zone V)
        inter_dists = []
        subset_o = np.random.choice(other_indices, min(len(other_indices), 100), replace=False)
        for idx_o in subset_o:
            # Distance to the nearest point in the High-Quality Zone
            dists_to_v = [data_nn.xdist(data_nn.rows[idx_o], data_nn.rows[idx_v]) for idx_v in subset_v]
            inter_dists.append(np.min(dists_to_v))
            
        mean_inter = np.mean(inter_dists)
        
        dist_ratio = mean_inter / mean_intra if mean_intra > 0 else 0
        zone_fraction = len(zone_indices) / len(all_fitness)
        
        # 5. Determine Task Family based on filename patterns 
        fname = os.path.basename(csv_file).lower()
        family = "Other"
        if any(x in fname for x in ["ss-", "config", "linux", "llvm", "postgre"]): family = "Config"
        elif any(x in fname for x in ["tuning", "hyper", "deeparch"]): family = "Tuning"
        elif any(x in fname for x in ["fm-", "ffm-", "feature"]): family = "Feature"
        elif "scrum" in fname: family = "Scrum"
        elif any(x in fname for x in ["pom", "xomo", "nasa"]): family = "POM"

        return [family, os.path.basename(csv_file), round(zone_fraction * 100, 2), round(dist_ratio, 2)]

    except Exception as e:
        return None

def main():
    csv_files = [os.path.join(r, f) for r, _, fs in os.walk(BASE_PATH) for f in fs if f.endswith(".csv")]
    
    # Save as you go
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Family", "Dataset", "ZoneFractionPct", "DistRatio"])

    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(analyze_zone_structure, cp): cp for cp in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                results.append(res)
                with open(OUTPUT_FILE, "a", newline="") as f:
                    csv.writer(f).writerow(res)

    # Generate Final Table Summary
    df = pd.DataFrame(results, columns=["Family", "Dataset", "ZoneFractionPct", "DistRatio"])
    summary = df.groupby("Family").agg(
        Datasets=('Dataset', 'count'),
        Zone_Fraction_Avg=('ZoneFractionPct', 'mean'),
        Dist_Ratio_Avg=('DistRatio', 'mean')
    ).reset_index()
    
    # Simple heuristic for "Confirmed" column
    summary['BV_Confirmed'] = summary.apply(lambda x: "Yes" if x['Dist_Ratio_Avg'] > 4 else "Partial", axis=1)
    
    print("\n--- TABLE 1: BIG VALLEY STRUCTURE ANALYSIS ---")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()