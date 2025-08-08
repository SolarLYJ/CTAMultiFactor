import lightgbm as lgb
from .base_model import BaseModel

class LGBModel(BaseModel):
    def __init__(self, **params):
        self.params = params
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)