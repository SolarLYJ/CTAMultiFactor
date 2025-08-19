import pandas as pd
import numpy as np

def add_ic_decay_factor(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    示例因子：过去 window 日 ret 衰减加权平均
    """
    close = df.xs("Close", level=1, axis=1)
    ret = close.pct_change()
    decay = ret.ewm(span=window).mean()
    decay.columns = pd.MultiIndex.from_product(
        [decay.columns, [f"f_decay_ret_{window}"]]
    )
    return pd.concat([df, decay], axis=1)

def pipeline_feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    """按顺序调用所有衍生函数"""
    df = add_ic_decay_factor(df)
    return df