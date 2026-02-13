#!/usr/bin/env python3
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
OUTPUT_FILE = "landscape_ruggedness_incremental.csv"

def run_ruggedness_test(csv_file):
    try:
        X_raw, Y_raw = load_data(csv_file)
        col_types = {col: EncodingUtils.infer_column_type(X_raw[col].tolist())[0] for col in X_raw.columns}
        X_enc = EncodingUtils.encode_dataframe(X_raw, col_types)
        cols = list(X_enc.columns)
        
        data_nn = Data(X_enc.values.tolist(), col_types)
        model_wrapper = ModelWrapperStatic(X_enc, Y_raw, ModelConfigurationStatic({c: X_enc[c].tolist() for c in cols}, csv_file, 1, column_types=col_types))
        
        n_rows = len(X_enc)
        sample_size = min(n_rows, 1000) 
        np.random.seed(42)
        indices = np.random.choice(n_rows, sample_size, replace=False)
        
        point_f, neighbor_f = [], []
        
        for idx in indices:
            # Point fitness
            row_dict = {c: data_nn.rows[idx][i] for i, c in enumerate(cols)}
            f = model_wrapper.run_model(row_dict)
            
            # Nearest neighbor fitness
            nn_idx = data_nn.k_nearest_indices(idx, k=1)[0]
            nn_dict = {c: data_nn.rows[nn_idx][i] for i, c in enumerate(cols)}
            nf = model_wrapper.run_model(nn_dict)
            
            point_f.append(f)
            neighbor_f.append(nf)
        
        rho, _ = spearmanr(point_f, neighbor_f)
        is_rugged = "Yes" if rho < 0.5 else "No"
        
        return [os.path.basename(csv_file), n_rows, round(rho, 4), is_rugged]
    except:
        return None

def main():
    csv_files = [os.path.join(r, f) for r, _, fs in os.walk(BASE_PATH) for f in fs if f.endswith(".csv")]
    
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Rows", "Autocorrelation_Rho", "Is_Rugged"])

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_ruggedness_test, c): c for c in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                with open(OUTPUT_FILE, "a", newline="") as f:
                    csv.writer(f).writerow(res)

if __name__ == "__main__":
    main()