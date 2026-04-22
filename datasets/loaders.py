"""Dataset loaders — returns (X, G, y) numpy arrays."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_adult(data_path):
    df     = pd.read_csv(data_path)
    G_col  = "gender:is_Male"
    Y_col  = "income:is_>50K"
    X_cols = [c for c in df.columns if c not in [G_col, Y_col]]
    return df[X_cols].values.astype(float), df[G_col].values.astype(int), df[Y_col].values.astype(int)


def load_compas(data_path):
    df  = pd.read_csv(data_path)
    df  = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    G   = (df["race"] == "African-American").astype(int).values
    y   = df["is_recid"].values.astype(int)
    num = ["age", "priors_count", "days_b_screening_arrest", "duration"]
    cat = ["c_charge_degree", "age_cat", "sex"]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X   = np.hstack([df[num].values.astype(float), enc.fit_transform(df[cat])])
    return X, G, y


def load_german(data_path):
    df     = pd.read_csv(data_path)
    G      = df["Sex"].values.astype(int)
    y      = df["Type"].values.astype(int)
    X_cols = [c for c in df.columns if c not in ["Sex", "Type"]]
    return df[X_cols].values.astype(float), G, y


def load_communities(data_path):
    df   = pd.read_csv(data_path).fillna(0)
    sens = ["racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"]
    df["race"] = df[sens].idxmax(axis=1)
    df   = df.drop(columns=sens)
    df   = df.drop(df[df["ViolentCrimesPerPop"] == 0].index).copy()
    mapping = {"racePctWhite": 1, "racepctblack": 0, "racePctAsian": 0, "racePctHisp": 0}
    df["race"] = df["race"].map(mapping)
    y      = (df["ViolentCrimesPerPop"] >= 0.2).astype(int).values
    G      = df["race"].values.astype(int)
    X_cols = [c for c in df.columns if c not in ["ViolentCrimesPerPop", "race"]]
    return df[X_cols].values.astype(float), G, y
