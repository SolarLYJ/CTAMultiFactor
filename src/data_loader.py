import pickle
from pathlib import Path
import pandas as pd
import yaml

def read_cfg(path: str) -> dict:
    """读取 YAML，并将日期字段全部转成 pd.Timestamp"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    date_keys = ("train_start", "train_end", "test_start", "test_end")
    for k in date_keys:
        if k in cfg and not isinstance(cfg[k], pd.Timestamp):
            cfg[k] = pd.to_datetime(cfg[k])

    return cfg

def load_pkl(path: str | Path):
    """读取原始 pkl"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def concat_symbols(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    把层次结构展开成单张表，方便后续处理。
    index: datetime
    columns: MultiIndex [symbol, field]
    """
    df_list = []
    for sym, df in data.items():
        df_cols = pd.MultiIndex.from_product([[sym], df.columns])
        df.columns = df_cols
        df_list.append(df)
    merged = pd.concat(df_list, axis=1).sort_index()
    return merged

def get_return(df: pd.DataFrame) -> pd.Series:
    """
    计算下一交易日收益率 (可替换为任意定义)
    """
    close = df.xs("Close", level=1, axis=1)
    ret = close.pct_change().shift(-1)      # t 日特征预测 t+1 收益
    return ret

if __name__ == "__main__":
    raw = load_pkl("../data/sample_data.pkl")
    merged = concat_symbols(raw)
    print(merged.shape, merged.head())