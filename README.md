# Fairness through Distributional Stability

Code for the paper *A Comprehensive View of Fairness through Distributional Stability* under review at NeurIPS 2026.
## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

## Repository structure

```
stability-fairness/
├── utils.py                        # Core: SOCP mixture, ERM baseline, PostProc,...
├── shifty.py                       # Shifty-ttest integration (Giguere et al., ICLR 2022)
├── shared/
│   ├── evaluation.py               # Shared helpers: eval_stats, make_phi_curve, ...
│   ├── plotting.py                 # Shared helpers: plot_phi_panel, plot_heatmap, ...
│   └── stats_tests.py              # Wilcoxon tests
├── datasets/
│   ├── loaders.py                  # load_adult, load_compas
│   ├── adult/
│   │   └── adult_norm.csv
│   ├── compas/
│   │   └── Processed_Compas.csv
├── run_adult.py                    # Multi-seed eval + figures + stat tests (Adult)
├── run_compas.py                   # Multi-seed eval + figures + stat tests (COMPAS)
├── run_acs.py                      # Geographic robustness + figures (ACS/Folktables)
├── run_synthetic.py                # Phi curves + Pareto frontiers (synthetic data)
├── tuning.py                       # Hyperparameter grid search (all datasets)
├── t-study.py                      # Impact of t for the adult dataset
├── figures/                        # Output figures (created automatically)
├── results/                        # Output results (created automatically)
├── requirements.txt
└── README.md
```

## Reproducing experiments

**Main experiments:**
```bash
python run_adult.py (note that for 20 seeds this can take some time)
python run_compas.py
python run_acs.py (note that for 20 seeds this can take some time)
```

Each script runs the full multi-seed evaluation, saves results to `results/`, generates
figures to `figures/`, and prints statistical significance tests. Results are
checkpointed after every seed — a crash only loses the current seed.
The results present in the `results/` folder are the one used to produce figures and tables in the paper.

The number of seeds and hyperparameters are set at the top of each script.

**Hyperparameter search (delta × pi_bar heatmap):**
```bash
python tuning.py --dataset adult      --constraint DP
python tuning.py --dataset adult      --constraint EO
```

## Shifty setup

Download and extract the Shifty repository next to `shifty.py`:
`https://github.com/sgiguere/Fairness-Guarantees-under-Demographic-Shift`
```
stability-fairness/
└── Fairness-Guarantees-under-Demographic-Shift/
    └── Python/
        ├── utils/
        ├── datasets/
        └── core/
```

Set `RUN_SHIFTY = False` in any `run_*.py` to skip Shifty and save time.

## Utils API
To run stable, we need to execute the following lines.


```python
from utils import SOCPFairnessMixture, ERMBaseline, phi_dp, phi_eo

# DP constraint
socp = SOCPFairnessMixture(K=30, pi_bar=0.05, m=60, delta=5e-5, t=0.5, constraint="DP")
socp.fit(X_train, G_train, y_train, seed=42)

# EO constraint
socp_eo = SOCPFairnessMixture(K=30, pi_bar=0.15, m=60, delta=1e-5, t=0.54, constraint="EO")
socp_eo.fit(X_train, G_train, y_train, seed=42)

# Phi curve
phi = socp.phi_curve(X_test, G_test, pi_grid)               # DP
phi = socp_eo.phi_curve(X_test, G_test, pi_grid, Y=y_test)  # EO

# Unfairness gap
u = socp.unfairness_gap(X_test, G_test)               # DP
u = socp_eo.unfairness_gap(X_test, G_test, Y=y_test)  # EO
```

## Dependencies

See `requirements.txt`.
