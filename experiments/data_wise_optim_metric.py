import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


RAW_ROOT = "../moot/optimize"
REPORT_CSV = "../results/optimization_performance/report.csv"


###################################################################
# 1. EXACT DRR IMPLEMENTATION (as in the DRR paper)
###################################################################
def compute_DRR_exact(X, energy_thresh=0.95, delta=0.01):
    """
    DRR = 1 - (I_D / D_eff)

    Paper method:
      - I_D: #components reaching 95% cumulative variance
      - D_eff: #components with eigenvalue ≥ 1% of max eigenvalue
    """
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(Xz)

    eigvals = pca.explained_variance_
    if np.any(np.isnan(eigvals)):
        return np.nan

    total_var = eigvals.sum()
    if total_var <= 0:
        return np.nan

    # I_D = components reaching 95% variance
    cum_energy = np.cumsum(eigvals) / total_var
    I_D = np.searchsorted(cum_energy, energy_thresh) + 1

    # D_eff = components above 1% of max eigenvalue
    max_eig = eigvals[0]
    D_eff = np.sum(eigvals >= delta * max_eig)

    if D_eff == 0:
        return np.nan

    return 1 - (I_D / D_eff)


###################################################################
# 2. Process a single dataset file (safe for parallelism)
###################################################################
def process_one_file(path):
    fname = os.path.basename(path)

    try:
        df = pd.read_csv(path)
    except:
        return None

    num = df.select_dtypes(include=[np.number]).dropna()
    # Need at least some rows + features
    if num.shape[0] < 10 or num.shape[1] < 3:
        return None

    X = num.values

    # DRR exact
    DRR = compute_DRR_exact(X)

    # Submetrics for OASX
    mean_vals = num.mean()
    var_vals = num.var()
    VMSX = np.mean(var_vals / (mean_vals.abs() + 1e-6))

    diffs = np.mean(np.abs(np.diff(X, axis=0)))
    global_dev = np.mean(np.abs(num - mean_vals))
    RBRX = diffs / (global_dev + 1e-6)

    SRCX = np.mean(np.abs(np.diff(X, n=2, axis=0)))

    return {
        "File": fname,
        "DRR": DRR,
        "VMSX": VMSX,
        "RBRX": RBRX,
        "SRCX": SRCX,
    }


###################################################################
# 3. Collect all dataset files recursively
###################################################################
all_files = []
for dirpath, _, files in os.walk(RAW_ROOT):
    for fn in files:
        if fn.endswith(".csv"):
            all_files.append(os.path.join(dirpath, fn))

print(f"Found {len(all_files)} dataset files.")


###################################################################
# 4. Parallel processing using joblib
###################################################################
print("Computing intrinsic dataset metrics (parallel)...")
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_one_file)(path)
    for path in tqdm(all_files)
)

intrinsic_df = pd.DataFrame([r for r in results if r is not None])
intrinsic_df["File"] = intrinsic_df["File"].astype(str).str.strip()

print("Processed:", len(intrinsic_df), "datasets.")


###################################################################
# 5. Load report.csv and compute D2H improvement
###################################################################
df = pd.read_csv(REPORT_CSV, engine="python", skipfooter=1, on_bad_lines="skip")

import re
sk = re.compile(r"(\d+)")

opt_cols = []

# Extract numeric d2h values
for col in df.columns:
    if any(k in col for k in ["SMAC", "ACT", "KM++", "RAND"]):
        df[col + "_d2h"] = df[col].astype(str).str.extract(sk).astype(float)
        opt_cols.append(col + "_d2h")

df["BestOpt"] = df[opt_cols].max(axis=1)
df["Improvement"] = df["BestOpt"] - df["B4.mu"]

df["BaseFile"] = df["File"].astype(str).str.strip()


###################################################################
# 6. Merge DRR and submetrics
###################################################################
merged = df.merge(intrinsic_df, left_on="BaseFile", right_on="File", how="left")

def rank_norm(s):
    return s.rank() / s.count()

###################################################################
# Original OASX (your version)
###################################################################
for m in ["VMSX", "RBRX", "SRCX"]:
    merged[m + "_n"] = rank_norm(merged[m])

merged["OASX"] = merged[["VMSX_n", "RBRX_n", "SRCX_n"]].mean(axis=1)


###################################################################
# 7. Improved OAS metrics
###################################################################

# Raw DRR + RBRX
merged["OAS_DR"] = (-merged["DRR"] + merged["RBRX"]) / 2

# Rank-based (direction corrected)
merged["OAS_best"] = (
    rank_norm(-merged["DRR"]) +
    rank_norm(merged["RBRX"])
) / 2

# Full composite with weak metrics (baseline)
merged["OAS_full"] = (
    rank_norm(-merged["DRR"]) +
    rank_norm(merged["RBRX"]) +
    rank_norm(merged["VMSX"]) +
    rank_norm(merged["SRCX"])
) / 4


###################################################################
# 8. Difficulty split into Easy/Medium/Hard
###################################################################
q1, q2 = merged["Improvement"].quantile([0.33, 0.66])

def diff_class(v):
    if pd.isna(v):
        return "Unknown"
    if v <= q1:
        return "Hard"
    elif v <= q2:
        return "Medium"
    return "Easy"

merged["DifficultyClass"] = merged["Improvement"].apply(diff_class)

print("\nDataset counts per difficulty:")
print(merged["DifficultyClass"].value_counts())


###################################################################
# 9. Correlations
###################################################################
compare_metrics = [
    "DRR",
    "RBRX",
    "OAS_DR",
    "OAS_best",
    "OAS_full",
    "OASX",
    "VMSX",
    "SRCX",
]

print("\n=== Correlations with Actual Improvement ===")
for m in compare_metrics:
    valid = merged[[m, "Improvement"]].dropna()
    r, _ = pearsonr(valid[m], valid["Improvement"])
    print(f"{m:10} : Pearson = {r:.4f}")


###################################################################
# 10. PLOTS
###################################################################
sns.set(style="whitegrid")

# --- Plot 1: DRR & OASX vs Difficulty ---
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
sns.boxplot(data=merged, x="DifficultyClass", y="DRR", palette="viridis")
sns.stripplot(data=merged, x="DifficultyClass", y="DRR", color="black", alpha=0.5, size=3)
plt.title("DRR vs Difficulty", fontsize=14)

plt.subplot(1,2,2)
sns.boxplot(data=merged, x="DifficultyClass", y="OASX", palette="magma")
sns.stripplot(data=merged, x="DifficultyClass", y="OASX", color="black", alpha=0.5, size=3)
plt.title("OASX vs Difficulty", fontsize=14)

plt.tight_layout()
plt.show()


# --- Plot 2: Improved OAS metrics (DRR+RBRX) ---
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
sns.boxplot(data=merged, x="DifficultyClass", y="OAS_DR", palette="Set2")
plt.title("OAS_DR vs Difficulty", fontsize=14)

plt.subplot(1,2,2)
sns.boxplot(data=merged, x="DifficultyClass", y="OAS_best", palette="Set3")
plt.title("OAS_best (Rank-Normalized) vs Difficulty", fontsize=14)

plt.tight_layout()
plt.show()


# --- Plot 3: Submetrics VMSX, RBRX, SRCX ---
plt.figure(figsize=(16,5))
for i, m in enumerate(["VMSX", "RBRX", "SRCX"]):
    plt.subplot(1,3,i+1)
    sns.boxplot(data=merged, x="DifficultyClass", y=m, palette="pastel")
    sns.stripplot(data=merged, x="DifficultyClass", y=m, color="black", size=3, alpha=0.5)
    plt.title(f"{m} vs Difficulty", fontsize=14)
plt.tight_layout()
plt.show()


# --- Plot 4: Scatter/LOWESS for DRR & OAS_best ---
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
sns.regplot(x=merged["DRR"], y=merged["Improvement"], lowess=True, scatter_kws={'alpha':0.4})
plt.title("DRR vs Actual Improvement", fontsize=14)

plt.subplot(1,2,2)
sns.regplot(x=merged["OAS_best"], y=merged["Improvement"], lowess=True, scatter_kws={'alpha':0.4})
plt.title("OAS_best vs Actual Improvement", fontsize=14)

plt.tight_layout()
plt.show()
