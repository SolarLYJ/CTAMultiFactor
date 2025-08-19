from __future__ import annotations
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from typing import List, Tuple
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def build_pipeline(already_scaled: bool, n_comp: int) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if not already_scaled:
        steps.append(("scale", StandardScaler()))
    steps.append(("pca", PCA(n_components=n_comp)))
    return Pipeline(steps)


def reduce_dimension(X: pd.DataFrame,
                      cfg: dict) -> Tuple[pd.DataFrame, Pipeline]:
    dr_cfg = cfg.get("dim_reduction", {"method": "pca", "n_components": 20})
    pipe = build_pipeline(cfg["preprocess"]["cs_zscore"], dr_cfg["n_components"])
    Z = pipe.fit_transform(X)
    cols = [f"PC{i+1:02d}" for i in range(Z.shape[1])]
    pipe.feat_cols = X.columns
    return pd.DataFrame(Z, index=X.index, columns=cols), pipe