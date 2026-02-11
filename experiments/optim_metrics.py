import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import pairwise_distances
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns


###########################################################
# 1. READ CSV + DROP LAST MALFORMED ROW
###########################################################

df = pd.read_csv(
    "../results/optimization_performance/report.csv",
    engine="python",
    skipfooter=1,
    on_bad_lines="skip"
)

###########################################################
# 2. PARSE D2H + SCOTT–KNOTT RANK
###########################################################

sk_pattern = re.compile(r"^\s*([0-9]+)\s*([a-z])?\s*$", re.IGNORECASE)

def parse_d2h_and_rank(value):
    if pd.isna(value):
        return np.nan, None
    value = str(value).strip()
    m = sk_pattern.match(value)
    if m:
        d2h = float(m.group(1))
        r = m.group(2) if m.group(2) else None
        return d2h, r
    try:
        return float(value), None
    except:
        return np.nan, None


optimizer_cols = [
    c for c in df.columns 
    if any(tag in c for tag in ["SMAC", "ACT", "KM++", "RAND"])
]

for col in optimizer_cols:
    d2h = []
    rank = []
    for v in df[col]:
        d, r = parse_d2h_and_rank(v)
        d2h.append(d)
        rank.append(r)
    df[col + "_d2h"] = d2h
    df[col + "_rank"] = rank

d2h_cols = [c for c in df.columns if c.endswith("_d2h")]


###########################################################
# 3. TRUE OPTIMIZATION IMPROVEMENT
###########################################################

df["BestOpt"] = df[d2h_cols].max(axis=1)
df["Improvement"] = df["BestOpt"] - df["B4.mu"]
df["Optimizable"] = (df["Improvement"] > 5).astype(int)


###########################################################
# 4. NEW METRICS + NORMALIZATION
###########################################################

eps = 1e-6

### VMS
df["VMS"] = df["B4.sd"] / (df["B4.mu"] + eps)

### RBR
rand_cols = [c for c in d2h_cols if "RAND" in c]
df["BestRand"] = df[rand_cols].max(axis=1)
df["RBR"] = (df["BestOpt"] + eps) / (df["BestRand"] + eps)

### LS (SMAC minus KM)
km_cols  = [c for c in d2h_cols if "KM++" in c]
smac_cols = [c for c in d2h_cols if "SMAC" in c]
df["KMmean"] = df[km_cols].mean(axis=1)
df["SMACmean"] = df[smac_cols].mean(axis=1)
df["LS"] = (df["SMACmean"] - df["KMmean"]) / (abs(df["KMmean"]) + eps)

### SRC (curvature)
def curvature(row):
    smac_vals = np.array([row[c] for c in smac_cols])
    deltas = np.diff(smac_vals)
    return np.sum(np.clip(-deltas, 0, None))

df["SRC"] = df.apply(curvature, axis=1)


###########################################################
# 5. RANK NORMALIZATION
###########################################################

def rank_norm(series):
    return series.rank(method="average") / len(series)

df["VMS_n"] = rank_norm(df["VMS"])
df["RBR_n"] = rank_norm(df["RBR"])
df["LS_n"]  = rank_norm(df["LS"])
df["SRC_n"] = rank_norm(df["SRC"])


###########################################################
# 6. OAS3 (your composite metric)
###########################################################

df["OAS3"] = (df["VMS_n"] + df["RBR_n"] + df["LS_n"] + df["SRC_n"]) / 4.0


###########################################################
# 7. DRR IMPLEMENTATION (Algorithm 1)
###########################################################

def intrinsic_dimension_fractal(X, steps=20):
    n = len(X)
    D = pairwise_distances(X)
    max_d = np.max(D)

    D = D + np.eye(n) * (max_d + 1)

    Rs_log = np.linspace(np.log(1e-6), np.log(max_d), steps)
    Rs = np.exp(Rs_log)

    CRs = []
    for R in Rs:
        count = np.sum(D < R)
        count -= n
        C_R = (2 * count) / (n * (n - 1))
        CRs.append(C_R)

    CRs = np.array(CRs)
    gradients = np.gradient(CRs, Rs)
    gradients = gaussian_filter1d(gradients, sigma=2)

    I = np.max(gradients)
    return I


# Compute DRR using columns: D, #R, #X, #Y, B4.mu, B4.sd, etc.
feature_cols = ["D", "#R", "#X", "#Y", "B4.mu", "B4.sd"]

X = df[feature_cols].to_numpy()

I = intrinsic_dimension_fractal(X)
R = X.shape[1]

DRR = 1 - (I / R)

df["DRR"] = DRR
df["ID"] = I


###########################################################
# 8. CORRELATIONS
###########################################################

metrics = ["VMS_n", "RBR_n", "LS_n", "SRC_n", "OAS3", "DRR"]

print("\n=== Pearson Correlations ===")
for m in metrics:
    r, _ = pearsonr(df[m], df["Improvement"])
    print(f"{m}: {r:.4f}")

print("\n=== Spearman Correlations ===")
for m in metrics:
    r, _ = spearmanr(df[m], df["Improvement"])
    print(f"{m}: {r:.4f}")


###########################################################
# 9. REGRESSION WITH DRR INCLUDED
###########################################################

X = df[metrics]
y = df["Improvement"]

lin = LinearRegression().fit(X, y)

print("\n=== Linear Regression ===")
print("Coefficients:", dict(zip(metrics, lin.coef_)))
print("R²:", lin.score(X, y))

log = LogisticRegression(max_iter=5000).fit(X, df["Optimizable"])

print("\n=== Logistic Regression ===")
print("Coefficients:", dict(zip(metrics, log.coef_[0])))
print("Accuracy:", log.score(X, df["Optimizable"]))


###########################################################
# 10. PLOTS
###########################################################

plt.figure(figsize=(9,7))
sns.regplot(x=df["OAS3"], y=df["Improvement"], lowess=True,
            scatter_kws={'alpha':0.5, 's':40}, line_kws={'color':"red"})
plt.title("OAS3 vs Improvement")
plt.show()

plt.figure(figsize=(9,7))
sns.regplot(x=df["DRR"], y=df["Improvement"], lowess=True,
            scatter_kws={'alpha':0.5, 's':40}, line_kws={'color':"red"})
plt.title("DRR vs Improvement")
plt.show()
