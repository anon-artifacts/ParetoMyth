import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys

# ===========================================
# File path
# ===========================================
file_path = "../results/optimization_performance/report_tmp.csv"
SKIP_FAMILIES = {"NSGAIII"}

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
# ===========================================
time_cols = [c for c in df.columns if c.endswith("_time")]

family_pattern = re.compile(r"^(.+)-\d+_time$")
family_groups = {}

for col in time_cols:
    m = family_pattern.match(col)
    if not m:
        continue
    family = m.group(1)

    # Apply Renaming Rules
    fam_upper = family.upper()
    if fam_upper == "ACT":
        family = "LITE"
    elif fam_upper in ["KM++", "KMPLUS", "KM_PP"]:
        family = "LINE"
    elif fam_upper in ["RAND", "RANDOM"]:
        family = "RANDOM"
    elif fam_upper == "NSGAIII":
        continue   # <<< SKIP HERE

    family_groups.setdefault(family, []).append(col)

if not family_groups:
    print("Error: No runtime families detected.")
    sys.exit(1)

print("Detected families:", list(family_groups.keys()))

# Clean dataset names for readability
def clean_label(label):
    if not isinstance(label, str):
        return label
    return label.replace(".csv", "")

df["File"] = df["File"].apply(clean_label)

# ===========================================
# Compute per-dataset mean runtime per family
# ===========================================
combined_rows = []

for file_name, subset in df.groupby("File"):
    entry = {"File": file_name}
    for family, cols in family_groups.items():
        entry[family] = subset[cols].stack().mean()
    combined_rows.append(entry)

combined_df = pd.DataFrame(combined_rows)

# ===========================================
# Box Plot Data Preparation
# ===========================================
families = list(family_groups.keys())
data_to_plot = [combined_df[fam].dropna().values for fam in families]

# ===========================================
# Plotting: Box Plot per Algorithm Family
# ===========================================
output_dir = "../results/runtime_plot"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 7))

box = plt.boxplot(
    data_to_plot,
    labels=families,
    patch_artist=True,
    showmeans=True,
    meanline=True
)

# Coloring the boxes
colors = plt.cm.tab10.colors
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.yscale("log")
plt.ylabel("Runtime (log scale)", fontsize=14)
plt.xticks(rotation=30, fontsize=12)
plt.title("Runtime Distribution per Algorithm Family", fontsize=16)

plt.grid(axis='y', linestyle='--', alpha=0.5)

save_path = os.path.join(output_dir, "families_runtime_boxplot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("Box plot saved to:", save_path)

