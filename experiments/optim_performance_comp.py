import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys
from io import StringIO

file_path = "../results/optimization_performance/report.csv"
SKIP_OPTIMIZERS = {"NSGAIII"}

# ----------------------------------------------------
# FIX MALFORMED LAST ROW (trim to expected #columns)
# ----------------------------------------------------
with open(file_path, "r") as f:
    lines = f.readlines()

header = lines[0]
expected_cols = header.count(",") + 1

# Fix last line
raw_last = lines[-1].strip().split(",")
fixed_last = ",".join(raw_last[:expected_cols]) + "\n"

# Replace the last line with fixed version
clean_lines = lines[:-1] + [fixed_last]

df = pd.read_csv(StringIO("".join(clean_lines)))
df.columns = df.columns.str.strip()

# ----------------------------------------------------
# SUMMARY ROW = LAST ROW IN FILE
# ----------------------------------------------------
summary_row = df.tail(1)

# ----------------------------------------------------
# Extract optimizers and budgets automatically
# ----------------------------------------------------
optimizers = {}
pattern = re.compile(r"([A-Za-z\+\~]+)-(\d+)$")

for col in df.columns:
    m = pattern.match(col)
    if m:
        family, budget = m.group(1), int(m.group(2))
        if family in SKIP_OPTIMIZERS:
            continue
        optimizers.setdefault(family, []).append(budget)

for fam in optimizers:
    optimizers[fam] = sorted(optimizers[fam])

samples = sorted({b for budgets in optimizers.values() for b in budgets})

# ----------------------------------------------------
# Numeric extractor for cells like "43 a"
# ----------------------------------------------------
def numeric_only(cell):
    if isinstance(cell, str):
        m = re.match(r"\s*([0-9]*\.?[0-9]+)", cell)
        return float(m.group(1)) if m else np.nan
    return float(cell)

# ----------------------------------------------------
# Extract percent-best values from summary row
# ----------------------------------------------------
def extract_percent_values(family):
    vals = []
    for b in optimizers[family]:
        col = f"{family}-{b}"
        if col in summary_row:
            vals.append(numeric_only(summary_row[col].values[0]))
        else:
            vals.append(np.nan)
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
# PLOT
# ----------------------------------------------------
plt.figure(figsize=(7, 5))

markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
colors = plt.cm.tab10.colors
font_size = 14

families = list(optimizers.keys())

for idx, fam in enumerate(families):
    plt.plot(
        plot_df["Samples"],
        plot_df[fam],
        marker=markers[idx % len(markers)],
        linestyle=linestyles[idx % len(linestyles)],
        color=colors[idx % len(colors)],
        label=rename_map.get(fam, fam),   # ← REPLACED
        linewidth=2.5,
        markersize=7
    )
plt.xscale("log")
plt.xlabel("Samples", fontsize=font_size)
plt.ylabel("% Best", fontsize=font_size)

# Explicit x-axis ticks
custom_ticks = [6, 12, 24, 50, 100, 200]
plt.xticks(custom_ticks, [str(t) for t in custom_ticks], fontsize=font_size)

# Y-axis: 0 → 100 with ticks every 10
plt.ylim(0, 100)
plt.yticks(np.arange(0, 101, 10))

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="lower right", fontsize=font_size)
plt.tight_layout()

save_path = "../results/optimization_performance/optimization_performance_comparison.png"
plt.savefig(save_path, dpi=300)
plt.show()

print("Saved plot to:", save_path)
