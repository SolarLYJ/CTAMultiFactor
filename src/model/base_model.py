from abc import ABC, abstractmethod
import joblib
from pathlib import Path

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