import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys
import numpy as np

# ===========================================
# File path
# ===========================================
file_path = "../results/optimization_performance/report_tmp.csv"

if not os.path.exists(file_path):
    print(f"Error: File not found: {file_path}")
    sys.exit(1)

if os.stat(file_path).st_size == 0:
    print(f"Error: File is empty: {file_path}")
    sys.exit(1)

try:
    df = pd.read_csv(file_path, engine="python", sep=r"\s*,\s*")
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# ===========================================
# Clean column names
# ===========================================
df.columns = (
    df.columns
      .str.strip()
      .str.replace('"', '', regex=False)
      .str.replace(' ', '', regex=False)
)

if "File" not in df.columns:
    print("Error: 'File' column missing after cleaning. Columns are:")
    print(df.columns.tolist())
    sys.exit(1)

def clean_label(label):
    if not isinstance(label, str):
        return label
    return label.replace(".csv", "")

df["File"] = df["File"].apply(clean_label)

# ===========================================
# Parse columns like FAMILY-BUDGET_sd
# ===========================================
col_pat = re.compile(r"^(.+)-(\d+)_sd$")
records = []

for col in df.columns:
    m = col_pat.match(col)
    if not m:
        continue

    family_raw, budget_s = m.group(1), m.group(2)
    budget = int(budget_s)

    fam_upper = family_raw.upper()
    # renaming rules
    if fam_upper == "ACT":
        family = "LITE"
    elif fam_upper in ["KM++", "KMPLUS", "KM_PP"]:
        family = "LINE"
    elif fam_upper in ["RAND", "RANDOM"]:
        family = "RANDOM"
    elif fam_upper == "NSGAIII":
        continue
    else:
        family = family_raw

    tmp = df[["File", col]].copy()
    tmp = tmp.rename(columns={col: "sd"})
    tmp["family"] = family
    tmp["budget"] = budget
    records.append(tmp)

if not records:
    print("Error: No FAMILY-BUDGET_sd columns detected.")
    sys.exit(1)

long_df = pd.concat(records, ignore_index=True)
long_df = long_df.dropna(subset=["sd"])

# If there are multiple rows per dataset, collapse to one per (File,family,budget)
long_df = (
    long_df
    .groupby(["File", "family", "budget"], as_index=False)["sd"]
    .median()
)

families = sorted(long_df["family"].unique().tolist())
budgets = sorted(long_df["budget"].unique().tolist())

print("Detected families:", families)
print("Detected budgets:", budgets)

# ===========================================
# Output dir
# ===========================================
output_root = "../results/optimization_performances/sd_boxplots_by_budget"
os.makedirs(output_root, exist_ok=True)

# ===========================================
# Plot: one figure per budget
# ===========================================
for b in budgets:
    sub = long_df[long_df["budget"] == b]

    # Compute per-family SD distributions + medians
    fam_data = []
    for fam in families:
        vals = sub[sub["family"] == fam]["sd"].values
        if len(vals) == 0:
            continue
        fam_data.append((fam, vals, np.median(vals)))

    # Sort by median SD (smallest → largest)
    fam_data.sort(key=lambda x: x[2])

    # Unpack in sorted order
    labels = [f for f, _, _ in fam_data]
    data_to_plot = [v for _, v, _ in fam_data]


    if not data_to_plot:
        print(f"Skipping budget {b}: no data")
        continue

    plt.figure(figsize=(12, 7))

    plt.boxplot(
        data_to_plot,
        labels=labels,
        showmeans=True,
        meanline=True
    )

    plt.ylabel("Run-to-run Standard Deviation (across datasets)", fontsize=13)
    plt.xticks(rotation=30, fontsize=11)
    plt.title(f"Run-to-run SD Across Datasets — Budget {b}", fontsize=15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Optional if SD is heavy-tailed:
    # plt.yscale("log")

    save_path = os.path.join(output_root, f"sd_boxplot_budget_{b}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)
