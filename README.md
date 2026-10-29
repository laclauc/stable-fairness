# STABLE — fairness through distributional stability

Code for the NeurIPS 2026 paper *A Comprehensive View of Fairness through
Distributional Stability*.

A classifier trained under a fairness constraint is fair at the group
proportions of its training data, but its prediction rate can drift once it is
deployed in a population with a different composition. STABLE asks the
prediction rate $\Phi(h;\pi')$ to stay close to a target $t$ for every
deployment proportion $\pi'$ in a range $[\bar\pi, 1-\bar\pi]$, and learns a
convex mixture of logistic regressions by solving a second-order cone program.

```python
from stable import SOCPFairnessMixture

model = SOCPFairnessMixture(pi_bar=0.02, delta=1e-5, t=0.2, constraint="DP")
model.fit(X_train, G_train, y_train)
p = model.predict_proba(X_test)        # P(y = 1 | x)
```

`notebooks/demo.ipynb` walks through a complete example on Adult.

## Installation

Python ≥ 3.9.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Adult and COMPAS are read from `data/` (see `data/README.md`); ACS Income is
downloaded by [folktables](https://github.com/socialfoundations/folktables) on
first use.

## Repository

```
stable/            the method
  core.py            SOCP mixture (Algorithm 1), importance weights, Phi, U
  dro.py             STABLE_DRO ablation (SA-DRO-style objective, same dictionary)
  metrics.py         evaluation: accuracy, U(h), deployment gap, level, Phi(h; 0/1)
  data.py            Adult, COMPAS and ACS loaders
experiments/       each script writes one CSV in results/
plots/             each script reads results/ and writes one figure in figures/
notebooks/demo.ipynb
```

## Reproducing the paper

Experiments write to `results/`, figures and tables to `figures/`. Seeds are
indexed identically across runs, so runs of different methods can be merged
(`--inputs a.csv+b.csv` in the plotting scripts). STABLE and STABLE_DRO use the
$\bar\pi$ and dictionary selected on validation data by `experiments/tuning.py`
(`SELECTED` in `experiments/methods.py`); the fairness weight $\lambda$ of
STABLE_DRO is swept on its own grid per setting (`LAM_GRID`, same file).
Figure and table numbers refer to the camera-ready version.

| Paper | Command |
|---|---|
| Pareto frontiers, Adult / COMPAS (Fig. 1) | `python experiments/pareto.py --dataset {adult,compas} --constraint {DP,EO} --seeds 10`<br>`python plots/pareto_main.py` |
| Prediction rate under demographic shift (Fig. 2) | `python plots/shift_curves.py` (from the same results) |
| Geographic generalization on ACS (Fig. 3 and 7, Table 4) | `python experiments/acs_pareto.py --constraint {DP,EO} --seeds 5`<br>`python plots/acs_geo.py --constraint {DP,EO}`<br>`python plots/tables.py acs` |
| Hyperparameter and dictionary selection (Table 3), sensitivity (Fig. 4) | `python experiments/tuning.py --dataset {adult,compas,acs} --constraint {DP,EO}`<br>`python plots/heatmaps.py --dataset adult --constraint DP` |
| Dictionary ablation (Fig. 5) | `python experiments/pareto.py --dataset {adult,compas} --constraint {DP,EO} --seeds 10 --methods STABLE_CF,STABLE_noCF --tag _cf_ablation`<br>`python plots/pareto_main.py --methods STABLE_CF,STABLE_noCF --out cf_ablation --inputs adult:DP=results/pareto_adult_DP_cf_ablation.csv ...` |
| Effect of the target $t$ (Fig. 8) | `python experiments/t_study.py`<br>`python plots/t_study.py` |
| Running times (Table 2) | `python plots/tables.py runtime` (from the results above) |


### Baselines

ERM + constraint is fairlearn's `ExponentiatedGradient`. The paper also
compares to **PostProc** (Chzhen et al., 2025) and **Shifty** (Giguere et
al., ICLR 2022), whose code is not distributed here. To include them, place
the authors' implementations on the Python path as modules `postproc`
(class `PostProcessor`) and `shifty` (functions `run_shifty_dp`,
`run_shifty_eo`), then add them to `--methods` (`PostProc`, `Shifty`).


## Citation

```bibtex
@inproceedings{stable2026,
  title     = {A Comprehensive View of Fairness through Distributional Stability},
  author    = {Taturyan, Gayane and Laclau, Charlotte and Cl{\'e}men{\c{c}}on, St{\'e}phan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, see `LICENSE`.
