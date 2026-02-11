# 🔍 Repository for "How Low Can You Go? The Data-Light SE Challenge"
---

## 📄 Summary

This repository contains the code, scripts, and data used to reproduce the results from our paper:

> **"How Low Can You Go? The   Data-Light SE Challenge"**  
> _Submitted to FSE 2026

We present the **BINGO effect**, a prevalent data compression phenomenon in software engineering (SE) optimization. Leveraging this, we show that **simple optimizers**—`RANDOM`, `LITE`, `LINE`—perform on par with the state-of-the-art optimizers, while running up to **10,000× faster**.

---
## 🧪 Experimental Setup

All experiments were run on a 4-core Linux (Ubuntu 24.04) system (1.30GHz, 16GB RAM, no GPU).

### Configuration

- **Datasets**: 39 MOOT tasks in `data/moot/`
- **Repeats**: 20 runs per optimizer
- **Budgets**: {6, 12, 18, 24, 50, 100, 200}
- **Optimizers**: `DEHB`, `SMAC`, `NSGAIII`, `TPE`, `LITE`, `LINE`, `RANDOM`
- **Evaluation**:
  - Effectiveness/ Benefit: distance-to-heaven (multi-objective)
  - Cost: no. of accessed labels, wall-clock time
---
## 📊 Reproducing the Results (Table 4, Figures 4 & 5)

These instructions reproduce all core results from the paper, including **Table 4**, **Figure 4**, and **Figure 5**.

All experiments were run using **Python 3.13**.

---

### ➤ Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ➤ Step 2: Generate Table 4

```bash
cd experiments/LUA_run_all/
make comparez
make report
```

The output will be saved to:

```
results/optimization_performance/report.csv
```

---

### ➤ Step 3: Generate Figure 4 (%Best vs. Label Budget)

```bash
cd experiments/
python3 optim_performance_comp.py
```

---

### ➤ Step 4: Generate Figure 5 (Runtime Comparison)

```bash
cd experiments/
python3 performance_box.py
```

---

### 🧪 Optional: Re-run Optimizers

We include precomputed results for `DEHB`, `SMAC`, `NSGAIII`, `TPE`, and `LITE` to save time. To regenerate:

```bash
# A. Remove existing results
rm -rf results  #removes all results

# B. Generate commands
make generate-commands NAME=Active_Learning   # This is LITE or use NAME=DEHB or SMAC or TPE or NSGAIII

# C. Run the optimizer
cd experiments/
./commands.sh
```

---

## 📦 Repository Structure

```plaintext
.
├── data/                         # Input data directory with MOOT datasets: 127 SE optimization tasks

├── active_learning/              # Active learning source code
│   ├── LICENSE.md                # Original license (MIT)
│   └── src/
│       └── bl.py                 # Contains Bayesian active learner

├── experiments/                  # Scripts for running experiments and generating plots/tables
│   ├── FileResultsReader.py      # Reads optimizer result files
│   ├── LUA_run_all/              # Lua scripts containing LITE and TABLE V generation logic
│   │   ├── Makefile              # Automates command/script generation
│   │   ├── run_all.lua           # Generates TABLE V
│   │   └── stats.lua             # Scott-Knott/effect size stats logic
│   ├── __init__.py
│   ├── experiement_runner_parallel.py  # Runs the optimizers
│   ├── optim_performance_comp.py       # Script to generate Fig. 5
│   └── performance_box.py           # Script to generate Fig. 6

├── models/                       # Manages and evaluates configs
│   ├── __init__.py
│   ├── configurations/
│   │   └── model_config_static.py   # Reads and manages tabular configs from MOOT
|   └── Data.py     # Class for maintaining data, caching and KD-Tree
│   └── model_wrapper_static.py     # Wrapper class for config evaluation

├── optimizers/                        # Optimizers implemented in the paper
│   ├── ActLearnOptimizer.py           # Active learning optimizer (LITE)
│   ├── DEHBOptimizer.py               # DEHB optimizer
│   ├── NSGAIIIOptimizer.py            # Multi-objective evolutionary optimizer (NSGA-III)
│   ├── SMACOptimizer.py               # SMAC (Sequential Model-Based Algorithm Configuration)
│   ├── TPEOptimizer.py                # Tree-structured Parzen Estimator optimizer
│   ├── __init__.py
│   └── base_optimizer.py              # Abstract base class for all optimizers

├── results/                          # Output directories for optimizer runs
│   ├── results_Active_Learning/      # Results from LITE
│   ├── results_DEHB/                 # Results from DEHB
│   ├── results_NSGAIII/              # Results from NSGA-III
│   ├── results_SMAC/                 # Results from SMAC
│   └── results_TPE/                  # Results from TPE


└── utils/                        # Utility scripts and shared functions
    ├── DistanceUtil.py           # Computes "distance to heaven"
    ├── LoggingUtil.py            # Sets up and manages logging
    ├── LoggingUtil.py            # Encodes/Decodes different data types
    ├── __init__.py
    └── data_loader_templated.py  # Loads and parses CSV datasets

├── .gitignore                    # Ignore logs, cache, and other non-reproducible files
├── LICENSE                       # MIT license (temporarily redacted)
├── Makefile                      # Automates command/script generation and execution
├── README.md                     # Artifact overview and reproduction instructions
└── requirements.txt              # Python dependencies for experiments and plotting
```

---

## ⚙️ Optimizers

| Optimizer | Description |
|----------|-------------|
| `RANDOM` | Random sampling of bucketed data |
| `LITE`   | Naive Bayes-based active learner (selects high g/r) |
| `LINE`   | Diversity sampling via KMeans++ |
| `DEHB`   | Differential Evolution + Hyperband |
| `NSGAIII`| Multi-objective evolutionary optimization |
| `SMAC`   | Model-based Bayesian optimization |
| `TPE`    | Parzen estimator-based Bayesian optimization |


---

## 🔐 License

> Includes MIT-licensed components.

---

## 🔗 External Links

> Will be updated upon acceptance:
- 📜 Paper DOI
- 📁 Dataset DOI
- 🧪 Artifact DOI


---




