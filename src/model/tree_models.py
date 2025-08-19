import lightgbm as lgb
from .base_model import BaseModel
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

class LGBRegressor(BaseModel):
    def __init__(self, **params):
        default_params = dict(
            boosting_type="gbdt",
            objective="regression",
            metric="rmse",
            learning_rate=0.20,
            n_estimators=15,
            num_leaves=5,
            max_depth=-1,
            min_data_in_leaf=300,
            min_sum_hessian_in_leaf=15.0,
            feature_fraction=0.55,
            bagging_fraction=0.7,
            bagging_freq=1,
            lambda_l1=0.5,
            lambda_l2=4.0,
            max_bin=255
        )
        default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class XGBRegressor(BaseModel):
    def __init__(self, **params):
        self.params = params
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class RFRegressor(BaseModel):
    def __init__(self, **params):
        self.params = params
        self.model = RandomForestRegressor(params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class GBDTRegressor(BaseModel):
    def __init__(self, **params):
        self.params = params
        self.model = GradientBoostingRegressor(params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)