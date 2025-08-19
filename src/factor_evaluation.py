"""
Rank-IC 计算 & 分层回测
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------- Rank-IC ----------------------------------------------------
def _rank_ic_one_day(f: pd.Series, r: pd.Series, min_obs: int) -> float:
    mask = ~(f.isna() | r.isna())
    if mask.sum() < min_obs:
        return np.nan
    frank = f[mask].rank()
    rrank = r[mask].rank()
    return np.corrcoef(frank, rrank)[0, 1]


def compute_ic_matrix(X: pd.DataFrame,
                      y: pd.Series,
                      min_obs: int = 30) -> pd.DataFrame:
    ic_dict = {}
    for col in tqdm(X.columns, desc="Rank-IC"):
        ic = (pd.concat([X[col], y], axis=1, keys=["f", "r"])
              .groupby(level=0) # 按日期分组
              .apply(lambda sub: _rank_ic_one_day(sub["f"], sub["r"], min_obs)))
        ic_dict[col] = ic
    return pd.DataFrame(ic_dict)


def ic_summary(ic_mat: pd.DataFrame) -> pd.DataFrame:
    m = ic_mat.mean()
    s = ic_mat.std()
    stat = pd.DataFrame({"mean_ic": m,
                         "std_ic": s,
                         "ir": m / s,
                         "pos_ratio": (ic_mat > 0).mean()})
    return stat.sort_values("mean_ic", ascending=False)


# ---------- 分层测试 ---------------------------------------------------
def layer_spread(factor: pd.Series,
                 future_ret: pd.Series,
                 n_layer: int,
                 ic_sign: float) -> pd.Series:
    """
    若 ic_sign<0 → 多第一层、空最后一层
    """
    df = pd.concat([factor, future_ret], axis=1, keys=["f", "r"]).dropna()

    def _one_day(sub): # 单日分层收益
        q = pd.qcut(sub["f"], n_layer, labels=False, duplicates="drop")
        sub = sub.assign(layer=q)
        mean_r = sub.groupby("layer")["r"].mean()
        if ic_sign >= 0:        # 正 IC：多高空低
            return mean_r.iloc[-1] - mean_r.iloc[0]
        else:                   # 负 IC：多低空高
            return mean_r.iloc[0] - mean_r.iloc[-1]

    return df.groupby(level=0).apply(_one_day)

def top_k_uncorrelated(ic_stat: pd.Series,
                       X: pd.DataFrame,
                       k: int,
                       thr: float = 0.9,
                       ic_thr: float = 0.01) -> list[str]:
    """
    根据 |meanIC| 由高到低挑选 <=k 个低相关因子,且meanIC > ic_threshold
    """
    # 计算因子相关矩阵
    corr = X.corr().abs()

    keep = []
    for fac in ic_stat.abs().sort_values(ascending=False).index:
        if len(keep) >= k:
            break
        if ic_stat.abs().loc[fac] <= ic_thr:
            continue
        # 与当前已选因子的相关系数都要 < thr
        if all(corr.loc[fac, j] < thr for j in keep):
            keep.append(fac)
    return keep