import pandas as pd
import numpy as np
import re

# --------------------------------------------
# Load data
# --------------------------------------------
df = pd.read_csv("../results/optimization_performance/report_tmp.csv")

# --------------------------------------------
# Extract optimizer families + budgets
# Example: DEHB-10, DEHB-20, SMAC-10, ...
# --------------------------------------------
optimizers = {}

for col in df.columns:
    m = re.match(r"([A-Za-z\+\~]+)-(\d+)$", col)   # captures "DEHB" and "10"
    if m:
        family, budget = m.group(1), int(m.group(2))
        optimizers.setdefault(family, []).append(budget)

# Sort budgets inside each family
for fam in optimizers:
    optimizers[fam] = sorted(optimizers[fam])

# --------------------------------------------
# Helpers
# --------------------------------------------

def is_best(cell):
    """Returns True if cell ends in 'a' ignoring spaces."""
    if not isinstance(cell, str):
        return False
    return cell.strip().endswith("a")

def numeric_value(cell):
    """Extract the numeric part from '43 a'."""
    if isinstance(cell, str):
        m = re.match(r"\s*([0-9]*\.?[0-9]+)", cell)
        return float(m.group(1)) if m else np.nan
    return float(cell)

# --------------------------------------------
# Build Table-3 for your optimizers
# --------------------------------------------

table = []

for fam, budgets in optimizers.items():
    for b in budgets:

        col = f"{fam}-{b}"         # e.g., DEHB-10
        tcol = f"{fam}-{b}_time"   # runtime column

        if col not in df.columns:
            continue

        # Column values
        cells = df[col]

        # percent best = % of rows ending with 'a'
        best_mask = cells.apply(is_best)
        percent_best = 100 * best_mask.mean()

        # when best, what was won? (mean of numeric D2H for rows ending in 'a')
        if best_mask.sum() > 0:
            avg_win = cells[best_mask].apply(numeric_value).mean()
        else:
            avg_win = np.nan

        # speed = mean runtime
        avg_speed = df[tcol].astype(float).mean() if tcol in df else np.nan

        table.append([fam, b, percent_best, avg_win, avg_speed])

# store result
table3 = pd.DataFrame(table, columns=[
    "optimizer", "budget", "percent_best", "avg_win", "speed_ms"
])

# Pivot to match the look in your Figure 3
table3_pivot = table3.pivot(index="optimizer",
                            columns="budget",
                            values=["percent_best", "avg_win", "speed_ms"])

# Pretty print
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n=== TABLE 3 (CORRECTED) ===\n")
print(table3_pivot.round(2))

table3_pivot.to_csv("table3_correct.csv")
