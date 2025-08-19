"""
step-1  因子清洗 + Rank-IC + 分层回测
------------------------------------
输出：
clean_X.pkl     # 清洗后的 (date,symbol)×factor
ret_y.pkl     #  (date,symbol)×ret
ic_matrix.csv
ic_stats.csv
top_factor_list.txt # |meanIC| Top-K 名单
layer_test/*.png
"""
# --- make ../src importable ---------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # project_root
sys.path.append(str(ROOT / "src"))
# -------------------------------------------------------------

from pathlib import Path
import yaml, pandas as pd
from tqdm import tqdm

from data_loader import load_pkl, concat_symbols, get_return, read_cfg
from feature_engineering import pipeline_feature_eng
from preprocess import clean_factor_df
from factor_evaluation import compute_ic_matrix, ic_summary, layer_spread, top_k_uncorrelated
from backtest import plot_nav
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

cfg_path="config/config.yaml"
cfg = read_cfg(cfg_path)
out = Path(cfg["result_path"])
out.mkdir(exist_ok=True)
layer_dir = out / "layer_test"

# ---------- 1. load & clean ----------
print("="*10+"loading + cleaning"+"="*10)
raw = concat_symbols(load_pkl(cfg["data_path"]))
raw = pipeline_feature_eng(raw)
mask = raw.columns.get_level_values(1).str.startswith(("f", "ff"))
X = raw.loc[:, mask].stack(level=0)
X.index.names = ["date", "symbol"]
X = clean_factor_df(X, **cfg["preprocess"])
X.to_pickle(out / "clean_X.pkl")

y = get_return(raw).stack(dropna=False)
y.index.names = ["date", "symbol"]
y.to_pickle(out / "ret_y.pkl")

# ---------- 2. Rank-IC ----------
if (out / "ic_matrix.csv").exists():
    print("="*10+"ic_matrix.csv already exists, skip calc"+"="*10)
    ic_mat  = pd.read_csv(out / "ic_matrix.csv", index_col=0, parse_dates=True)
    ic_stat = pd.read_csv(out / "ic_stats.csv",  index_col=0)
else:
    print("="*10+"computing Rank-IC"+"="*10)
    ic_mat  = compute_ic_matrix(X, y, cfg["analysis"]["ic_min_obs"])
    ic_mat.to_csv(out / "ic_matrix.csv")
    ic_stat = ic_summary(ic_mat)
    ic_stat.to_csv(out / "ic_stats.csv")
print("="*10+"IC files ready"+"="*10)

# ---------- 3. layer back-test ----------
top_k = cfg["analysis"]["top_k_factor"]
n_layer = cfg["analysis"]["n_layer"]
best  = top_k_uncorrelated(ic_stat["mean_ic"], X, top_k,
                        cfg["analysis"]["corr_threshold"],cfg['analysis']['ic_threshold'])
(out / "top_factor_list.txt").write_text("\n".join(best))   # 保存名单

print("="*10+"layer test Top-{top_k}"+"="*10)
for fac in tqdm(best[:cfg["analysis"]["top_k_plot"]]):
    mean_ic = ic_stat.loc[fac, "mean_ic"]
    ret     = layer_spread(X[fac], y, n_layer, mean_ic) # 分层回测
    nav = (1 + ret).cumprod()
    plot_nav(nav, layer_dir / f"{fac}.png",
            title=f"{fac} {n_layer}-layer LS (meanIC={mean_ic:.3f})",
            split_date=cfg["train_end"])