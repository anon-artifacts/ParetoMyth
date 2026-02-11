import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys

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
    # Read CSV, tolerate spaces around commas
    df = pd.read_csv(file_path, engine="python", sep=r"\s*,\s*")
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Clean column names
df.columns = (
    df.columns
      .str.strip()
      .str.replace('"', '', regex=False)
      .str.replace(' ', '', regex=False)
)

# Ensure File column exists
if "File" not in df.columns:
    print("Error: 'File' column missing after cleaning. Columns are:")
    print(df.columns.tolist())
    sys.exit(1)

# ===========================================
# Identify runtime columns and group by FAMILY
# e.g. DEHB-6_time, DEHB-12_time, ... -> family "DEHB"
# ===========================================
time_cols = [c for c in df.columns if c.endswith("_time")]

family_pattern = re.compile(r"^(.+)-\d+_time$")  # capture family before "-<budget>_time"
family_groups = {}

for col in time_cols:
    m = family_pattern.match(col)
    if not m:
        continue
    family = m.group(1)  # e.g. "DEHB", "NSGAIII", "KM++", "RAND"
    family_groups.setdefault(family, []).append(col)

if not family_groups:
    print("Error: No runtime families detected.")
    sys.exit(1)

print("Detected families:", list(family_groups.keys()))

# ===========================================
# Clean dataset names (optional, just cosmetic)
# ===========================================
def clean_label(label):
    if not isinstance(label, str):
        return label
    return label.replace(".csv", "")

df["File"] = df["File"].apply(clean_label)

# ===========================================
# Compute per-dataset, per-family AVERAGE runtime
# For each dataset:
#   family runtime = mean over all budgets & rows for that family
# ===========================================
combined_rows = []

for file_name, subset in df.groupby("File"):
    entry = {"File": file_name}
    for family, cols in family_groups.items():
        # mean over all rows and all budget columns of this family
        entry[family] = subset[cols].stack().mean()  # stack flattens to 1D
    combined_rows.append(entry)

combined_df = pd.DataFrame(combined_rows)

# ===========================================
# Sort datasets by one reference family (prefer DEHB if present)
# ===========================================
if "DEHB" in family_groups:
    sort_family = "DEHB"
else:
    sort_family = list(family_groups.keys())[0]

combined_df = combined_df.sort_values(by=sort_family).reset_index(drop=True)

# Assign numeric dataset IDs for x-axis
combined_df["DatasetID"] = combined_df.index + 1

# ===========================================
# Plotting: one curve per FAMILY
# ===========================================
output_dir = "../results/runtime_plot"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 7))
x = combined_df["DatasetID"]

colors = plt.cm.tab10.colors
markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']

families = list(family_groups.keys())
styles = {}

for i, fam in enumerate(families):
    styles[fam] = (
        markers[i % len(markers)],
        "-",
        colors[i % len(colors)]
    )

for fam in families:
    if fam in combined_df.columns:
        marker, linestyle, color = styles[fam]
        plt.plot(
            x,
            combined_df[fam],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=fam,
        )

# X-axis as dataset IDs, show every 10th tick if many datasets
n_datasets = len(combined_df)
step = 10 if n_datasets > 20 else 1
ticks = list(range(1, n_datasets + 1, step))

plt.xticks(ticks=ticks, labels=ticks)
plt.xlabel("Dataset ID", fontsize=12)

plt.yscale("log")
plt.ylabel("Avg. Runtime (log scale)", fontsize=12)
plt.yticks(fontsize=11)

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=min(5, len(families)),
    frameon=False,
    fontsize=12
)

plt.tight_layout()

save_path = os.path.join(output_dir, "families_avg_runtime_comparison.png")
plt.savefig(save_path)
plt.close()

print("Plot saved to:", save_path)
