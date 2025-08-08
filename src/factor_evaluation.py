"""
Rank-IC 计算 & 分层回测
"""
from __future__ import annotations
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


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
              .groupby(level=0)
              .apply(lambda sub: _rank_ic_one_day(sub["f"], sub["r"], min_obs)))
        ic_dict[col] = ic
    return pd.DataFrame(ic_dict)


def ic_summary(ic_mat: pd.DataFrame) -> pd.DataFrame:
    m = ic_mat.mean(); s = ic_mat.std()
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

    def _one_day(sub):
        q = pd.qcut(sub["f"], n_layer, labels=False, duplicates="drop")
        sub = sub.assign(layer=q)
        mean_r = sub.groupby("layer")["r"].mean()
        if ic_sign >= 0:        # 正 IC：多高空低
            return mean_r.iloc[-1] - mean_r.iloc[0]
        else:                   # 负 IC：多低空高
            return mean_r.iloc[0] - mean_r.iloc[-1]

    return df.groupby(level=0).apply(_one_day)


def plot_nav(series, path: Path | str, title: str):
    """
    保存净值曲线；若父目录不存在则自动创建
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nav = (1 + series).cumprod()
    plt.figure(figsize=(8, 3))
    nav.plot()
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()