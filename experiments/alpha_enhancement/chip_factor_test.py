"""筹码因子原型（自算，不花钱）IC 检验，2026-10-23。

通达信式筹码分布的简化重构：以近 120 日、半衰期 20 日的指数衰减权重
近似"筹码留存"（衰减=换手消耗的代理，无流通股本数据下的标准近似），
构造三因子：
  1. winner  获利盘比例 = Σ w·V·1[typ<close] / Σ w·V
  2. costdev 成本偏离   = close / (Σ w·V·typ / Σ w·V) - 1
  3. concen  成本集中度 = 筹码加权 typ 的变异系数（越低越集中）
typ=(H+L+C)/3。卖点是 120 日路径依赖——Alpha158 窗口 ≤60 学不到的结构。
检验同 wobv_factor_test：主板池、同标签、日 rank IC + 与 base/p0 相关。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/jerry/workspace/trading_v2/qlib")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "benchmarks_dynamic" / "mainboard_leftside"))

import qlib  # noqa: E402
from experiments.rightside_alpha.data import get_tradable_mask, load_qlib_data  # noqa: E402
from experiments.rightside_alpha.factors import cs_zscore  # noqa: E402
from generate_live_picks import H, rank_ic_daily  # noqa: E402

WIN, HALF = 120, 20.0
OUT = ROOT / "experiments" / "alpha_enhancement" / "results" / "chip_factor_test.csv"


def build_factors(close, high, low, volume):
    typ = (high + low + close) / 3
    w = {k: 0.5 ** (k / HALF) for k in range(1, WIN + 1)}
    wv_sum = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    wvt_sum = wv_sum.copy()
    win_sum = wv_sum.copy()
    for k, wk in w.items():
        v_k = volume.shift(k)
        t_k = typ.shift(k)
        wv = wk * v_k
        wv_sum += wv
        wvt_sum += wv * t_k
        win_sum += wv * (t_k < close)
    vwap = wvt_sum / wv_sum
    winner = win_sum / wv_sum
    costdev = close / vwap - 1
    # 集中度：加权 std/vwap，二趟
    var_sum = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for k, wk in w.items():
        var_sum += wk * volume.shift(k) * (typ.shift(k) - vwap) ** 2
    concen = (var_sum / wv_sum) ** 0.5 / vwap
    return winner, costdev, concen


def main():
    uri = os.path.expanduser("~/.qlib/qlib_data/cn_data_mootdx")
    qlib.init(provider_uri=uri, region="cn")
    data = load_qlib_data("mainboard", "2022-01-01", "2026-10-21", provider_uri=uri)
    close, open_, high, low, volume = (data["close"], data["open"], data["high"], data["low"], data["volume"])
    tradable = get_tradable_mask(close, volume)

    print("构建筹码因子...")
    winner, costdev, concen = build_factors(close, high, low, volume)
    fwd = (open_.shift(-(H + 1)) / open_.shift(-1) - 1.0).where(tradable)

    rows = []
    for name, fac in [("winner获利盘", winner), ("costdev成本偏离", costdev), ("concen集中度", concen)]:
        fac = fac.where(tradable)
        ic = rank_ic_daily(fac, fwd)
        ic20 = ic.rolling(20).mean()
        rows.append(dict(factor=name, ic_mean=ic.mean(), icir=ic.mean() / ic.std(),
                         win=(ic > 0).mean(), ic20_now=ic20.dropna().iloc[-1]))
        print(f"{name}: 日IC均值 {ic.mean():+.4f} | ICIR {ic.mean()/ic.std():+.2f} | "
              f"胜率 {(ic>0).mean():.0%} | 当前20日IC {ic20.dropna().iloc[-1]:+.4f}")

    from experiments.alpha_enhancement.ic_streak_history import merge_hist
    for name, fac in [("winner", winner), ("costdev", costdev), ("concen", concen)]:
        zF = cs_zscore(fac.where(tradable))
        for mname in ("base", "p0"):
            z_m = cs_zscore(merge_hist(mname)["score"].unstack(level="instrument")
                            .reindex(index=close.index, columns=close.columns).where(tradable))
            common = zF.notna() & z_m.notna()
            corr = zF[common].corrwith(z_m[common], axis=1).mean()
            rows.append(dict(factor=f"corr({name},{mname})", ic_mean=corr, icir=np.nan, win=np.nan, ic20_now=np.nan))
            print(f"{name} × {mname} 模型截面相关: {corr:+.3f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
