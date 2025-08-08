"""
src/preprocess.py
============================================================
原始因子清洗
------------------------------------------------------------
• ±inf → NaN
• 缺失率过滤
• Winsorize 去极值
• 时间序列平滑
• 缺失值填补 :  median_cs | ts_ffill
• 横截面 z-score 标准化（可选）
============================================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize


# -----------------------------------------------------------------
def winsorize_col(s: pd.Series, pct: float) -> pd.Series:
    """对单列做 winsorize，保持 NaN 位置不变"""
    if s.isna().all():
        return s
    arr = winsorize(s.to_numpy(), limits=[pct, pct])
    return pd.Series(arr, index=s.index)


def cs_zscore_transform(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 z-score：每个交易日对所有品种归一化"""
    cs_mean = df.groupby(level=0).transform("mean")
    cs_std  = df.groupby(level=0).transform("std").replace(0, np.nan)
    return (df - cs_mean) / cs_std


def ts_smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """按 symbol × factor 做滚动均值平滑"""
    if window <= 1:
        return df
    tmp = df.unstack()                      # -> date × symbol × factor
    smoothed = tmp.rolling(window, min_periods=1).mean()
    return smoothed.stack().reindex_like(df)


def ts_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """
    时间序列方向填补：
        对每个 symbol 的因子时间序列 forward-fill
    """
    def _ff(sub: pd.DataFrame):
        sub = sub.sort_index(level=0)
        return sub.ffill()
    return df.groupby(level=1, group_keys=False).apply(_ff)


# -----------------------------------------------------------------
def clean_factor_df(
    df: pd.DataFrame,
    nan_threshold: float = 0.2,
    winsor_pct: float = 0.01,
    fill_method: str = "median_cs",   # median_cs | ts_ffill
    smooth_window: int = 1,
    cs_zscore: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    原始因子清洗主入口
    Parameters
    ----------
    df : DataFrame(index=[date,symbol], columns=factors)
    """
    out = df.copy()

    # 1. inf → NaN
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. 缺失率过滤
    nan_ratio = out.isna().mean()
    drop_cols = nan_ratio[nan_ratio > nan_threshold].index
    if len(drop_cols) and verbose:
        print(f"[Pre] drop {len(drop_cols)} cols (nan_ratio > {nan_threshold:.0%})")
    out.drop(columns=drop_cols, inplace=True)

    # 3. Winsorize
    for c in out.columns:
        out[c] = winsorize_col(out[c], winsor_pct)

    # 4. 时间序列平滑
    out = ts_smooth(out, smooth_window)

    # 5. 填补缺失
    if fill_method == "median_cs":
        med = out.groupby(level=0).transform("median")
        out = out.fillna(med)
    elif fill_method == "ts_ffill":
        out = ts_ffill(out)
    else:
        raise ValueError("fill_method must be 'median_cs' or 'ts_ffill'")

    # 6. 横截面 z-score
    if cs_zscore:
        out = cs_zscore_transform(out)

    return out