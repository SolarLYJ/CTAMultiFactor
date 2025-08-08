import pickle
from pathlib import Path
import pandas as pd

def load_raw(path: str | Path) -> dict[str, pd.DataFrame]:
    """读取原始 pkl，返回 {symbol: DataFrame}"""
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
    raw = load_raw("../data/sample_data.pkl")
    merged = concat_symbols(raw)
    print(merged.shape, merged.head())