from __future__ import annotations
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from typing import List, Tuple
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def _ic(f: pd.Series, y: pd.Series) -> float:
    # 把 f、y 拼在一起，只保留共同索引并去掉 NaN
    df = pd.concat([f, y], axis=1, join="inner").dropna()
    if len(df) < 30:          # 数据点太少就放弃
        return np.nan
    # 直接 Spearman 相关或 rank 后 Pearson 都行
    return df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())


def batch_ic(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    return pd.Series({c: _ic(df[c], y) for c in tqdm(df.columns, desc="Rank-IC (sel)")})


def remove_high_corr(df: pd.DataFrame, th: float) -> List[str]:
    corr = df.corr().abs()
    keep = []
    for c in df.columns:
        if all(corr.loc[c, k] < th for k in keep):
            keep.append(c)
    return keep


def build_pipeline(already_scaled: bool, n_comp: int) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if not already_scaled:
        steps.append(("scale", StandardScaler()))
    steps.append(("pca", PCA(n_components=n_comp, random_state=42)))
    return Pipeline(steps)


def select_and_reduce(X: pd.DataFrame,
                      y: pd.Series,
                      cfg: dict) -> Tuple[pd.DataFrame, Pipeline]:
    ic = batch_ic(X, y).abs()
    top = ic.sort_values(ascending=False).head(cfg["top_k_factor_ic"])
    X = X[top.index]
    X = X[remove_high_corr(X, cfg["corr_threshold"])]

    dr_cfg = cfg.get("dim_reduction", {"method": "pca", "n_components": 20})
    pipe = build_pipeline(cfg["preprocess"]["cs_zscore"], dr_cfg["n_components"])
    Z = pipe.fit_transform(X)
    cols = [f"PC{i+1:02d}" for i in range(Z.shape[1])]
    pipe.feat_cols = X.columns
    return pd.DataFrame(Z, index=X.index, columns=cols), pipe