"""
step-1  因子清洗 + Rank-IC + 分层回测
------------------------------------
输出：
clean_X.parquet     # 清洗后的 (date,symbol)×factor
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

from data_loader import load_raw, concat_symbols, get_return
from feature_engineering import pipeline_feature_eng
from preprocess import clean_factor_df
from factor_evaluation import compute_ic_matrix, ic_summary, layer_spread, plot_nav
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def read_cfg(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_path="config/config.yaml"):
    cfg = read_cfg(cfg_path)
    out = Path(cfg["result_path"]); out.mkdir(exist_ok=True, parents=True)
    layer_dir = out / "layer_test"

    # ---------- 1. load & clean ----------
    print("📥  loading + cleaning ...")
    raw = concat_symbols(load_raw(cfg["data_path"]))
    raw = pipeline_feature_eng(raw)
    mask = raw.columns.get_level_values(1).str.startswith(("f", "ff"))
    X = raw.loc[:, mask].stack(level=0); X.index.names = ["date", "symbol"]
    X = clean_factor_df(X, **cfg["preprocess"])
    X.to_parquet(out / "clean_X.parquet")       # ⭐ 保存

    y = get_return(raw).stack(dropna=False); y.index.names = ["date", "symbol"]

    # ---------- 2. Rank-IC ----------
    if (out / "ic_matrix.csv").exists():
        print("📝  ic_matrix.csv already exists, skip calc")
        ic_mat  = pd.read_csv(out / "ic_matrix.csv", index_col=0, parse_dates=True)
        ic_stat = pd.read_csv(out / "ic_stats.csv",  index_col=0)
    else:
        print("📊  computing Rank-IC ...")
        ic_mat  = compute_ic_matrix(X, y, cfg["analysis"]["ic_min_obs"])
        ic_mat.to_csv(out / "ic_matrix.csv")
        ic_stat = ic_summary(ic_mat)
        ic_stat.to_csv(out / "ic_stats.csv")
    print("✅  IC files ready")

    # ---------- 3. layer back-test ----------
    top_k = cfg["analysis"]["top_k_plot"]
    best  = ic_stat["mean_ic"].abs().sort_values(ascending=False).head(top_k).index
    (out / "top_factor_list.txt").write_text("\n".join(best))   # ⭐ 保存名单

    print(f"🧪  layer test Top-{top_k}")
    for fac in tqdm(best):
        mean_ic = ic_stat.loc[fac, "mean_ic"]
        spread  = layer_spread(X[fac], y, 5, mean_ic)
        plot_nav(spread, layer_dir / f"{fac}.png",
                 title=f"{fac} 5-layer LS (meanIC={mean_ic:.3f})")

if __name__ == "__main__":
    main()