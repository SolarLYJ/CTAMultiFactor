import pandas as pd
import numpy as np

def add_ic_decay_factor(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    示例因子：过去 window 日 IC 衰减加权平均
    (演示补充因子的写法，真正因子请自行挖掘)
    """
    # TODO: 使用真实因子 or 成交量/持仓等重构
    close = df.xs("Close", level=1, axis=1)
    ret = close.pct_change()
    decay = ret.ewm(span=window).mean()
    decay.columns = pd.MultiIndex.from_product(
        [decay.columns.get_level_values(0), [f"decay_ret_{window}"]]
    )
    return pd.concat([df, decay], axis=1)

def pipeline_feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    """按顺序调用所有衍生函数"""
    df = add_ic_decay_factor(df)
    # TODO: append更多自定义因子
    return df