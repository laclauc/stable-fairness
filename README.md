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
stable-fairness-public/
├── utils.py                        # Core: SOCP mixture, ERM baseline, importance weights, phi/unfairness-gap functionals
├── postproc.py                     # PostProcessor: post-processing baseline (Chzhen, Hebiri & Taturyan 2025)
├── shifty.py                       # Shifty-ttest integration (Giguere et al., ICLR 2022)
├── shared/
│   ├── evaluation.py               # Shared helpers: eval_stats, make_phi_curve, compute_targets, ...
│   ├── plotting.py                 # Shared helpers: plot_phi_panel, plot_heatmap, ...
│   └── stats_tests.py              # Wilcoxon paired significance tests
├── datasets/
│   ├── loaders.py                  # load_adult, load_compas
│   ├── adult/
│   │   └── adult_norm.csv
│   └── compas/
│       └── Processed_Compas.csv
├── run_adult.py                    # Multi-seed eval + figures + stat tests (Adult)
├── run_compas.py                   # Multi-seed eval + figures + stat tests (COMPAS)
├── run_acs.py                      # Geographic robustness experiment (ACS/Folktables)
├── figure-acs.py                   # Standalone ACS figures from results/acs_geographic_robustness.csv
├── tuning.py                       # Hyperparameter grid search — δ × π̄ heatmap (all datasets)
├── t-study.py                      # Impact of t on SOCP fairness (Adult)
├── figures/                        # Output figures (created automatically)
├── results/                        # Pre-computed results used in the paper
│   ├── adult_multiseed_results.csv
│   ├── compas_multiseed_results.csv
│   ├── acs_geographic_robustness.csv
│   ├── adult_phi_curves_DP.npy
│   ├── adult_phi_curves_EO.npy
│   ├── compas_phi_curves_DP.npy
│   ├── compas_phi_curves_EO.npy
│   └── t_study_adult.csv
├── requirements.txt
└── README.md
```

## Reproducing experiments

**Main experiments:**
```bash
python run_adult.py    # note: 20 seeds can take some time
python run_compas.py
python run_acs.py      # note: 20 seeds can take some time
```

Each script runs the full multi-seed evaluation, saves results to `results/`, generates
figures to `figures/`, and prints statistical significance tests. Results are
checkpointed after every seed — a crash only loses the current seed.
The results present in the `results/` folder are the ones used to produce figures and tables in the paper.

The number of seeds and hyperparameters are set at the top of each script.

**ACS figures only (from pre-computed results):**
```bash
python figure-acs.py
```

**Hyperparameter search (δ × π̄ heatmap):**
```bash
python tuning.py --dataset adult --constraint DP
python tuning.py --dataset adult --constraint EO
```

**Effect of t on fairness (Adult):**
```bash
python t-study.py
```

## Shifty setup

Download and extract the Shifty repository next to `shifty.py`:
`https://github.com/sgiguere/Fairness-Guarantees-under-Demographic-Shift`
```
stable-fairness-public/
└── Fairness-Guarantees-under-Demographic-Shift/
    └── Python/
        ├── utils/
        ├── datasets/
        └── core/
```

Set `RUN_SHIFTY = False` in any `run_*.py` to skip Shifty and save time.

## Utils API

### SOCPFairnessMixture

```python
from utils import SOCPFairnessMixture, ERMBaseline, phi_dp, phi_eo, unfairness_gap_dp, unfairness_gap_eo

# DP constraint
socp = SOCPFairnessMixture(K=30, pi_bar=0.05, m=60, delta=5e-5, t=0.5, constraint="DP")
socp.fit(X_train, G_train, y_train, seed=42)

# EO constraint
socp_eo = SOCPFairnessMixture(K=30, pi_bar=0.15, m=60, delta=1e-5, t=0.54, constraint="EO")
socp_eo.fit(X_train, G_train, y_train, seed=42)

# Phi curve
phi = socp.phi_curve(X_test, G_test, pi_grid)                # DP
phi = socp_eo.phi_curve(X_test, G_test, pi_grid, Y=y_test)  # EO

# Unfairness gap
u = unfairness_gap_dp(socp.predict_proba(X_test), G_test)
u = unfairness_gap_eo(socp_eo.predict_proba(X_test), G_test, y_test)
```

## Dependencies

See `requirements.txt`.
