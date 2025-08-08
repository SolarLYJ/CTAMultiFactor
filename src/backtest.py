"""
长/短多空回测 + 绩效统计 & 净值曲线绘制
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
def long_short_pnl(pred: pd.DataFrame,
                   future_ret: pd.DataFrame,
                   k_long: float,
                   k_short: float,
                   tc: float = 0.0) -> pd.Series:
    """
    生成每日组合收益

    pred       : 行=date, 列=symbol, 预测值越大看多
    future_ret : 行=date, 列=symbol, 下一日实际收益
    """
    rank = pred.rank(axis=1, ascending=False, pct=True)

    long_m  = rank <= k_long
    short_m = rank >= 1 - k_short

    w_long  = long_m.div(long_m.sum(axis=1), axis=0).fillna(0)
    w_short = -short_m.div(short_m.sum(axis=1), axis=0).fillna(0)
    weight  = w_long + w_short

    daily_ret = (weight * future_ret).sum(axis=1)

    turnover = weight.diff().abs().sum(axis=1)
    daily_ret -= tc * turnover            # 交易成本

    return daily_ret      # Series(index=date)


# ------------------------------------------------------------
def perf_stats(daily_ret: pd.Series, freq: int = 252) -> dict:
    """
    计算常用指标：总收益、年化、夏普、卡玛、最大回撤
    """
    nav = (1 + daily_ret).cumprod()

    ann_ret = nav.iloc[-1] ** (freq / len(nav)) - 1
    ann_vol = daily_ret.std() * np.sqrt(freq)
    sharpe  = ann_ret / ann_vol if ann_vol else np.nan

    rolling_max = nav.cummax()
    drawdown    = nav / rolling_max - 1
    max_dd      = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd else np.nan

    return dict(
        total_return = nav.iloc[-1] - 1,
        annual_return = ann_ret,
        annual_vol = ann_vol,
        sharpe = sharpe,
        max_drawdown = max_dd,
        calmar = calmar
    )


# ------------------------------------------------------------
def plot_nav(daily_ret: pd.Series, path: Path | str):
    """
    保存净值曲线 PNG
    """
    nav = (1 + daily_ret).cumprod()
    plt.figure(figsize=(10, 4))
    nav.plot()
    plt.title("Net Asset Value Curve")
    plt.ylabel("NAV")
    plt.grid(True, linestyle="--", alpha=.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()