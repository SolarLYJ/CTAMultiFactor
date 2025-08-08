"""
step-2  读取已清洗数据 → 建模 & 组合回测
--------------------------------------
依赖文件（由 factor_analysis.py 生成）:
clean_X.parquet
top_factor_list.txt   *可选
"""
# --- make ../src importable ---------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # project_root
sys.path.append(str(ROOT / "src"))
# -------------------------------------------------------------

from pathlib import Path
import yaml, pandas as pd
from tqdm import tqdm

from preprocess import clean_factor_df          # 只供补充使用
from feature_selection import select_and_reduce
from backtest import long_short_pnl, perf_stats, plot_nav as plot_nav_port
from model.tree_models import LGBModel

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def read_cfg(path: str) -> dict:
    """读取 YAML，并将日期字段全部转成 pd.Timestamp"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    date_keys = ("train_start", "train_end", "test_start", "test_end")
    for k in date_keys:
        if k in cfg and not isinstance(cfg[k], pd.Timestamp):
            cfg[k] = pd.to_datetime(cfg[k])

    return cfg

MODEL_REGISTRY = {"lgb": LGBModel}

def main(cfg_path="config/config.yaml"):
    cfg = read_cfg(cfg_path)
    out = Path(cfg["result_path"]); out.mkdir(exist_ok=True, parents=True)

    # ---------- 1. load cleaned factors ----------
    clean_path = out / "clean_X.parquet"
    if not clean_path.exists():
        raise FileNotFoundError("clean_X.parquet 不存在，请先运行 factor_analysis.py")

    X_full = pd.read_parquet(clean_path)
    y_full = pd.read_csv(out / "ic_matrix.csv", nrows=0)  # dummy 只取索引格式
    # 直接重新读取 return
    from data_loader import load_raw, concat_symbols, get_return
    raw = concat_symbols(load_raw(cfg["data_path"]))
    y_full = get_return(raw).stack(dropna=False); y_full.index.names = ["date", "symbol"]

    # ---------- 2. 训练/测试划分 ----------
    idx_x = X_full.index.get_level_values(0)
    idx_y = y_full.index.get_level_values(0)

    tr_x = (idx_x >= cfg["train_start"]) & (idx_x <= cfg["train_end"])
    te_x = (idx_x >= cfg["test_start"])  & (idx_x <= cfg["test_end"])

    tr_y = (idx_y >= cfg["train_start"]) & (idx_y <= cfg["train_end"])
    te_y = (idx_y >= cfg["test_start"])  & (idx_y <= cfg["test_end"])

    X_tr, X_te = X_full.loc[tr_x], X_full.loc[te_x]
    y_tr, y_te = y_full.loc[tr_y], y_full.loc[te_y]

    # ---------- 3. 特征筛选 + 降维 ----------
    X_tr_sel, pipe = select_and_reduce(X_tr, y_tr, cfg)
    X_te_sel = pd.DataFrame(pipe.transform(X_te[pipe.feat_cols]),
                            index=X_te.index, columns=X_tr_sel.columns)
    y_tr = y_tr.loc[X_tr_sel.index]
    y_te = y_te.loc[X_te_sel.index]

    # ---------- 4. 模型 ----------
    model = MODEL_REGISTRY[cfg["model_name"]](**cfg["hyper_params"])
    model.fit(X_tr_sel, y_tr)
    pred = pd.Series(model.predict(X_te_sel), index=X_te_sel.index)

    # ---------- 5. 回测 ----------
    nav = long_short_pnl(pred.unstack(), y_te.unstack(),
                         cfg["k_long"], cfg["k_short"], cfg["transaction_cost"])
    stats = perf_stats(nav)

    nav.to_csv(out / "daily_return.csv")
    plot_nav_port(nav, out / "total_nav.png")
    pd.Series(stats).to_csv(out / "perf_stats.csv")

    print("\n🎯  back-test finished")
    for k, v in stats.items():
        print(f"{k:16s}: {v: .4f}")

if __name__ == "__main__":
    main()