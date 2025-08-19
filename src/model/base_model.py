from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

class BaseModel(ABC):
    """所有模型的统一接口"""

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):
        return joblib.load(path)
    
def reg_metric(y_true, y_pred):
    # 将 y_true 和 y_pred 转换为 Series 并对齐索引
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)
    
    # 过滤掉包含 NaN 的样本
    valid_mask = ~y_true.isna() & ~y_pred.isna()
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    # 计算指标
    return {
        "RMSE": mean_squared_error(y_true_clean, y_pred_clean) **0.5,
        "R2": r2_score(y_true_clean, y_pred_clean)
    }