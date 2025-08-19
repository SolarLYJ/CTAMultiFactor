"""
长/短多空回测 + 绩效统计 & 净值曲线绘制
"""
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
def perf_stats(ret: pd.Series, ann: int = 252):
    r = ret.mean() * ann
    v = ret.std(ddof=0) * (ann**0.5)
    sharpe = r / v if v else np.nan
    mdd = (ret.add(1).cumprod().cummax() - ret.add(1).cumprod()).max()
    return {"AnnRet": r, "Vol": v, "Sharpe": sharpe, "MDD": mdd}

def split_stats(ret: pd.Series, split_date: str) -> pd.DataFrame:

    ins = ret.loc[:split_date]
    oos = ret.loc[split_date:]
    all_data = ret

    stats_is = perf_stats(ins)
    stats_oos = perf_stats(oos)
    stats_all = perf_stats(all_data)

    stats_df = pd.DataFrame({
        "IS": stats_is,    # 样本内指标
        "OOS": stats_oos,  # 样本外指标
        "ALL": stats_all   # 全样本指标
    })

    return stats_df


# ------------------------------------------------------------
def plot_nav(nav: pd.Series, path: Path,
             title: str = "", split_date: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(nav, label="NAV")
    if split_date is not None:
        ax.axvline(pd.to_datetime(split_date), ls="--", c="k", alpha=.5)
        ax.text(pd.to_datetime(split_date), ax.get_ylim()[1],
                "  OOS", va="top", ha="left")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)