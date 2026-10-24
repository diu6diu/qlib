"""WOBV（K线强弱）横截面因子化检验，2026-10-23。

把期货时序策略 WOBV 改写成可横截面比较的股票因子：
  每根K线"推进质量" q = |C-O| / (H-L)（长上影下影小实体 → q≈0，乱晃；
  大实体紧凑 → q≈1，真推进），日推进量 = sign(C-O) × V × q。
  因子 F = rolling(25).sum(推进量) / rolling(25).sum(V) ∈ [-1, 1]
  = WOBV 与其 25 日均线的偏离在横截面上的可比形式。
对照：朴素 OBV 版（不加质量权重）、ROC20 动量。
评估：主板股票池，与生产模型同口径的标签（fwd=open[t+6]/open[t+1]-1）
日 rank IC + 20 日均值；与 base/p0 模型得分的截面相关（重合度）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/jerry/workspace/trading_v2/qlib")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "benchmarks_dynamic" / "mainboard_leftside"))

from experiments.rightside_alpha.data import get_tradable_mask, load_qlib_data  # noqa: E402
from experiments.rightside_alpha.factors import cs_zscore  # noqa: E402
from generate_live_picks import H, rank_ic_daily  # noqa: E402
import os  # noqa: E402
import qlib  # noqa: E402

END = "2026-10-21"
OUT = ROOT / "experiments" / "alpha_enhancement" / "results" / "wobv_factor_test.csv"


def main():
    uri = os.path.expanduser("~/.qlib/qlib_data/cn_data_mootdx")
    qlib.init(provider_uri=uri, region="cn")
    data = load_qlib_data("mainboard", "2022-06-01", END, provider_uri=uri)
    close, open_, high, low, volume = (data["close"], data["open"], data["high"], data["low"], data["volume"])
    tradable = get_tradable_mask(close, volume)

    body = (close - open_).abs()
    rng = (high - low).where((high - low) > 0)
    q = (body / rng).clip(0, 1)
    push = np.sign(close - open_) * volume * q
    F = push.rolling(25).sum() / volume.rolling(25).sum()  # WOBV 净推进强度
    F_obv = (np.sign(close - open_) * volume).rolling(25).sum() / volume.rolling(25).sum()
    roc20 = close.pct_change(20)

    fwd = (open_.shift(-(H + 1)) / open_.shift(-1) - 1.0).where(tradable)

    rows = []
    for name, fac in [("WOBV净推进", F), ("朴素OBV", F_obv), ("ROC20动量", roc20)]:
        fac = fac.where(tradable)
        ic = rank_ic_daily(fac, fwd)
        ic20 = ic.rolling(20).mean()
        rows.append(dict(factor=name, ic_mean=ic.mean(), icir=ic.mean() / ic.std(),
                         win=(ic > 0).mean(), ic20_now=ic20.dropna().iloc[-1]))
        print(f"{name}: 日IC均值 {ic.mean():+.4f} | ICIR {ic.mean()/ic.std():+.2f} | "
              f"IC>0胜率 {(ic>0).mean():.0%} | 当前20日IC {ic20.dropna().iloc[-1]:+.4f}")

    # 与生产双模型得分的截面相关（重合度）：用 sig_hist 合并库
    from experiments.alpha_enhancement.ic_streak_history import merge_hist
    zF = cs_zscore(F.where(tradable))
    for mname in ("base", "p0"):
        z_m = cs_zscore(merge_hist(mname)["score"].unstack(level="instrument")
                        .reindex(index=close.index, columns=close.columns).where(tradable))
        common = zF.notna() & z_m.notna()
        corr = zF[common].corrwith(z_m[common], axis=1).mean()
        rows.append(dict(factor=f"corr(WOBV,{mname}模型)", ic_mean=corr, icir=np.nan, win=np.nan, ic20_now=np.nan))
        print(f"WOBV 与 {mname} 模型得分截面相关: {corr:+.3f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
