import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from io import StringIO
import seaborn as sns

file_path = "../results/optimization_performance/report.csv"

# ----------------------------------------------------
# FIX MALFORMED LAST ROW
# ----------------------------------------------------
with open(file_path, "r") as f:
    lines = f.readlines()

header = lines[0]
expected_cols = header.count(",") + 1

raw_last = lines[-1].strip().split(",")
fixed_last = ",".join(raw_last[:expected_cols]) + "\n"
clean_lines = lines[:-1] + [fixed_last]

df = pd.read_csv(StringIO("".join(clean_lines)))
df.columns = df.columns.str.strip()

# ----------------------------------------------------
# SUMMARY ROW
# ----------------------------------------------------
summary_row = df.tail(1)

# ----------------------------------------------------
# Extract optimizers and budgets
# ----------------------------------------------------
optimizers = {}
pattern = re.compile(r"([A-Za-z\+\~]+)-(\d+)$")

for col in df.columns:
    m = pattern.match(col)
    if m:
        fam, budget = m.group(1), int(m.group(2))
        optimizers.setdefault(fam, []).append(budget)

for fam in optimizers:
    optimizers[fam] = sorted(optimizers[fam])

samples = sorted({b for v in optimizers.values() for b in v})

# ----------------------------------------------------
# Numeric extractor
# ----------------------------------------------------
def numeric_only(cell):
    if isinstance(cell, str):
        m = re.match(r"\s*([0-9]*\.?[0-9]+)", cell)
        return float(m.group(1)) if m else np.nan
    return float(cell)

# ----------------------------------------------------
# Extract values
# ----------------------------------------------------
def extract_percent_values(family):
    vals = []
    for b in optimizers[family]:
        col = f"{family}-{b}"
        vals.append(
            numeric_only(summary_row[col].values[0])
            if col in summary_row else np.nan
        )
    return vals

plot_data = {"Samples": samples}
for fam in optimizers:
    plot_data[fam] = extract_percent_values(fam)

plot_df = pd.DataFrame(plot_data)

rename_map = {
    "ACT": "LITE",
    "KM++": "LINE",
    "RAND": "RANDOM"
}

# ----------------------------------------------------
# HEATMAP (VARIANCE-AWARE)
# ----------------------------------------------------
heatmap_df = plot_df.set_index("Samples").T
heatmap_df.rename(index=rename_map, inplace=True)

plt.figure(figsize=(8, 4))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".1f",
    cmap="viridis",
    linewidths=0.6,
    cbar_kws={"label": "% Best"}
)

plt.xlabel("Samples", fontsize=14)
plt.ylabel("Optimizer", fontsize=14)
plt.title("Optimizer Performance Across Sampling Budgets", fontsize=15)

plt.tight_layout()

save_path = "../results/optimization_performance/optimization_performance_heatmap.png"
plt.savefig(save_path, dpi=300)
plt.show()

print("Saved heatmap to:", save_path)
