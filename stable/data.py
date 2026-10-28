"""Dataset loaders. Each returns (X, G, y) as numpy arrays, G and y in {0, 1}.

Adult and COMPAS are read from the CSV files in `data/` (see data/README.md);
ACS Income is downloaded on first use through folktables.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Held-out ACS states used for the geographic experiments (train on CA).
ACS_TRAIN_STATE = "CA"
ACS_TEST_STATES = ["ME", "WA", "NY", "FL", "WI", "IL", "PA", "OH", "MI", "GA", "OR"]
ACS_VALIDATION_STATES = ["LA", "TX", "MN"]


def load_adult(path=DATA_DIR / "adult" / "adult_norm.csv"):
    """Adult Income. G = 1 for men, y = 1 for income > 50K; all other
    columns of the (already encoded and normalised) CSV are features."""
    df = pd.read_csv(path)
    g_col, y_col = "gender:is_Male", "income:is_>50K"
    X = df.drop(columns=[g_col, y_col]).values.astype(float)
    return X, df[g_col].values.astype(int), df[y_col].values.astype(int)


def load_compas(path=DATA_DIR / "compas" / "Processed_Compas.csv"):
    """COMPAS, restricted to African-American and Caucasian defendants.
    G = 1 for African-American, y = 1 for recidivism (is_recid)."""
    df = pd.read_csv(path)
    df = df[df["race"].isin(["African-American", "Caucasian"])]
    G = (df["race"] == "African-American").astype(int).values
    y = df["is_recid"].values.astype(int)
    num = ["age", "priors_count", "days_b_screening_arrest", "duration"]
    cat = ["c_charge_degree", "age_cat", "sex"]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = np.hstack([df[num].values.astype(float), enc.fit_transform(df[cat])])
    return X, G, y


def load(dataset: str):
    """Adult or COMPAS, by name."""
    return {"adult": load_adult, "compas": load_compas}[dataset]()


def stratified_subsample(X, G, y, n, seed):
    """Subsample n rows, stratified on (y, G). No-op if n >= len(y)."""
    if n is None or n >= len(y):
        return X, G, y
    _, idx = train_test_split(np.arange(len(y)), test_size=n / len(y),
                              random_state=seed, stratify=np.stack([y, G], axis=1))
    return X[idx], G[idx], y[idx]


def load_acs(state: str, year: str = "2018", max_n: Optional[int] = None, seed: int = 0):
    """ACS Income for one state; G = 1 for White respondents (RAC1P = 1)."""
    from folktables import ACSDataSource, ACSIncome

    src = ACSDataSource(survey_year=year, horizon="1-Year", survey="person",
                        root_dir=str(DATA_DIR / "acs"))
    X, y, G_raw = ACSIncome.df_to_numpy(src.get_data(states=[state], download=True))
    y, G = y.astype(int), (G_raw == 1).astype(int)
    return stratified_subsample(X, G, y, max_n, seed)
