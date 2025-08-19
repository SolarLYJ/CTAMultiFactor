"""
step-2  读取已清洗数据 → 建模 & 组合回测
--------------------------------------
依赖文件（由 factor_analysis.py 生成）:
clean_X.pkl     # 清洗后的 (date,symbol)×factor
ret_y.pkl     #  (date,symbol)×ret
top_factor_list.txt
"""
import sys, pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]   # project_root
sys.path.append(str(ROOT / "src"))

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from feature_selection import reduce_dimension
from backtest import long_short_pnl, plot_nav, split_stats
from model.tree_models import LGBRegressor, XGBRegressor, RFRegressor, GBDTRegressor
from model.base_model import reg_metric
from data_loader import load_pkl, read_cfg

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_REGISTRY = {"lgb": LGBRegressor, "xgb":XGBRegressor, "rf":RFRegressor, "gbdt":GBDTRegressor}

cfg_path="config/config.yaml"
cfg = read_cfg(cfg_path)
out = Path(cfg["result_path"])
out.mkdir(exist_ok=True, parents=True)

# ---------- 1. load cleaned factors ----------
if not (out / "clean_X.pkl").exists() or not (out / "ret_y.pkl").exists():
    raise FileNotFoundError("文件不存在，请先运行 factor_analysis.py")

X_full = load_pkl(out / "clean_X.pkl")
y_full = load_pkl(out / "ret_y.pkl")

# ---------- 2. 训练/测试划分 ----------
idx_x = X_full.index.get_level_values(0)
idx_y = y_full.index.get_level_values(0)

tr_x = (idx_x >= cfg["train_start"]) & (idx_x <= cfg["train_end"])
te_x = (idx_x >= cfg["test_start"])  & (idx_x <= cfg["test_end"])

tr_y = (idx_y >= cfg["train_start"]) & (idx_y <= cfg["train_end"])
te_y = (idx_y >= cfg["test_start"])  & (idx_y <= cfg["test_end"])

X_tr, X_te = X_full.loc[tr_x], X_full.loc[te_x]
y_tr, y_te = y_full.loc[tr_y], y_full.loc[te_y]

# ----------只用 Top-K 因子 ----------
top_file = out / "top_factor_list.txt"
if top_file.exists():
    top_list = top_file.read_text().splitlines()
    X_tr = X_tr[top_list]
    X_te = X_te[top_list]
    print(f"using factor list from {top_file}")

# ---------- 3. 特征筛选 + 降维 ----------
X_tr_sel, pipe = reduce_dimension(X_tr, cfg)
X_te_sel = pd.DataFrame(pipe.transform(X_te[pipe.feat_cols]),
                        index=X_te.index, columns=X_tr_sel.columns)
y_tr = y_tr.loc[X_tr_sel.index]
y_te = y_te.loc[X_te_sel.index]

# ---------- 4. 模型 ----------
model = MODEL_REGISTRY[cfg["model_name"]](**cfg["hyper_params"])
model.fit(X_tr_sel, y_tr)

# ── 训练集 ──
pred_tr = model.predict(X_tr_sel)
m_tr = reg_metric(y_tr, pred_tr)

# ── 测试集 ──
pred_te = model.predict(X_te_sel)
m_te = reg_metric(y_te, pred_te)

# ── 全样本 ──
X_all = pd.concat([X_tr_sel, X_te_sel])
y_all = pd.concat([y_tr, y_te])
pred_all = model.predict(X_all)
m_all = reg_metric(y_all, pred_all)

# 汇总输出
out_df = pd.concat([
    pd.Series(m_tr, name="IS").to_frame().T,
    pd.Series(m_te, name="OOS").to_frame().T,
    pd.Series(m_all, name="ALL").to_frame().T
], ignore_index=False)
out_df.to_csv(out / f"{cfg['model_name']}_model_metrics.csv")
print(out_df)

# ---------- 5. 回测 ----------
# pred_te_ser = pd.Series(pred_te, index=X_te_sel.index)
pred_all_ser = pd.Series(pred_all, index=X_all.index)
ret = long_short_pnl(pred_all_ser.unstack(), y_all.unstack(),
                        cfg["k_long"], cfg["k_short"], cfg["transaction_cost"])
stats = split_stats(ret, cfg["train_end"])

ret.to_csv(out / cfg["model_name"] / f"{cfg['model_name']}_daily_return.csv")
nav = ret.add(1).cumprod()
plot_nav(nav, out / cfg["model_name"] / f"{cfg['model_name']}_total_nav.png",title=f"{cfg['model_name']}_total_nav", split_date=cfg["train_end"])
stats.to_csv(out / cfg["model_name"] / f"{cfg['model_name']}_perf_stats.csv")

print("="*10+"back-test finished"+"="*10)
print(stats)