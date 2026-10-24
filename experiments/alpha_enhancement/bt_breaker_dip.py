"""实验B′（2026-10-20，预登记于 FINDINGS.md）：深回撤提前回场（buy-the-dip）。

在 bt_self_breaker.py 口径上叠加三态断路器：ON(100%) → ret20≤kill →
HALF(50%) → ret20≤dip → DIP(100%)；DIP 持有至 ret20 > recover 直接回 ON
（不在 dip~recover 区间反复落回半仓——那是纯水平规则的病态：刚开始恢复
就砍仓）。T 收盘判定 T+1 生效（exposure.shift(1)），切换收 0.2% 单边×
切换名义。

主样本：种子信号 2024-07-01~2026-07-03（须精确复现 README 的 V3 数字
53.3%/1.89/−21.5%）。OOS 延伸：sig_hist 合并库延至 2026-10-19（含当前
实盘深回撤段），核对方向一致性。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/jerry/workspace/trading_v2/qlib")
sys.path.insert(0, str(ROOT))

from experiments.alpha_enhancement.bt_self_breaker import SWITCH_COST, apply_breaker, run_engine, stats  # noqa: E402

EXT_END = "2026-10-19"
KILL, RECOVER, HALF = -0.03, -0.01, 0.5


def apply_dip_breaker(
    nav: pd.Series, ret_w: pd.Series, kill: float, recover: float, dip: float, half: float = HALF
) -> tuple[pd.Series, dict]:
    """三态 ON/HALF/DIP。DIP 中持有至 ret_w > recover 才回 ON。"""
    r = nav.pct_change().fillna(0.0)
    state, exposure, switches, dip_days = "ON", [], 0, 0
    for i in range(len(nav)):
        rw = ret_w.iloc[i]
        if pd.notna(rw):
            if state == "ON" and rw <= kill:
                state, switches = "HALF", switches + 1
            elif state == "HALF" and rw <= dip:
                state, switches = "DIP", switches + 1
            elif state == "HALF" and rw > recover:
                state, switches = "ON", switches + 1
            elif state == "DIP" and rw > recover:
                state, switches = "ON", switches + 1
        dip_days += state == "DIP"
        exposure.append({"ON": 1.0, "HALF": half, "DIP": 1.0}[state])
    expo = pd.Series(exposure, index=nav.index).shift(1).fillna(1.0)  # T+1 生效
    d_expo = expo.diff().abs().fillna(0.0)
    r_out = r * expo - SWITCH_COST * d_expo
    return (1 + r_out).cumprod(), dict(switches=switches, expo_mean=float(expo.mean()), dip_days=dip_days)


def run_engine_hist(end: str = EXT_END) -> pd.Series:
    """sig_hist 合并库信号（含实盘段）的同款引擎 NAV，用于 OOS 延伸。"""
    import qlib  # noqa: E402

    from experiments.rightside_alpha.data import get_tradable_mask, load_qlib_data  # noqa: E402
    from experiments.rightside_alpha.factors import cs_zscore  # noqa: E402
    from experiments.rightside_alpha.run_rightside_v2 import run_backtest  # noqa: E402

    uri = os.path.expanduser("~/.qlib/qlib_data/cn_data_mootdx")
    qlib.init(provider_uri=uri, region="cn")
    data = load_qlib_data("mainboard", "2023-01-01", end, provider_uri=uri)
    close, open_, high, low, volume = (data["close"], data["open"], data["high"], data["low"], data["volume"])
    tradable = get_tradable_mask(close, volume)

    hist_dir = ROOT / "examples/benchmarks_dynamic/mainboard_leftside/results/mainboard_leftside/sig_hist"

    def load_wide(name):
        sig = pd.read_pickle(hist_dir / name)
        if isinstance(sig, pd.DataFrame):
            sig = sig[sig.columns[0]]
        return sig.unstack(level="instrument").reindex(index=close.index, columns=close.columns)

    z0 = cs_zscore(load_wide("base.pkl").where(tradable))
    z1 = cs_zscore(load_wide("p0.pkl").where(tradable))
    z_blend50 = cs_zscore((z0 + z1) / 2)
    sig_long = z_blend50.loc["2024-07-01" : end].stack().rename("score").to_frame()

    nav, _, m = run_backtest(
        sig_long,
        open_,
        tradable,
        start_time="2024-07-01",
        end_time=end,
        topk=10,
        n_drop=2,
        selection="official",
        capital=100_000,
        high_=high,
        low_=low,
        close_=close,
        model_limit=True,
        limit_retry_days=0,
        limit_rule="keep",
        gap_skip=True,
    )
    print(f"engine(hist): ann={m['ann_return']:.2%} Sharpe={m['sharpe']:.3f} MDD={m['max_drawdown']:.2%}")
    return nav


def evaluate(nav: pd.Series) -> dict[str, tuple[pd.Series, dict]]:
    daily_ret = nav.pct_change()
    ret_w = daily_ret.rolling(20).sum()
    out = {"V0 基线": (nav, dict(switches=0, expo_mean=1.0, dip_days=0))}
    out["V3 现生产(-3/-1 半仓)"] = apply_breaker(nav, ret_w, KILL, RECOVER, HALF)
    for label, dip in [("V7 dip-6%", -0.06), ("V8 dip-8%", -0.08), ("V9 dip-10%", -0.10)]:
        out[label] = apply_dip_breaker(nav, ret_w, KILL, RECOVER, dip)
    return out


def report(variants: dict, title: str) -> None:
    print(f"\n{'── ' + title + ' ──'}")
    print(f"{'变体':<24s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>8s} {'切换':>4s} {'平均仓位':>8s} {'DIP天数':>7s}")
    for label, (n2, info) in variants.items():
        ann, sharpe, mdd = stats(n2)
        print(
            f"{label:<24s} {ann:8.2%} {sharpe:7.3f} {mdd:8.2%} {info['switches']:4d}"
            f" {info['expo_mean']:8.1%} {info.get('dip_days', 0):>7d}"
        )


def seg_stats(nav: pd.Series, start: str, end: str) -> tuple[float, float, float] | None:
    seg = nav.loc[start:end]
    if len(seg) == 0:
        return None
    seg = seg / seg.iloc[0]
    r = seg.pct_change().dropna()
    ann = float(r.mean() * 252)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    mdd = float((seg / seg.cummax() - 1).min())
    return ann, sharpe, mdd


def main() -> None:
    # ── 主样本：种子信号，复现 V3 基线 ──
    nav = run_engine()
    variants = evaluate(nav)
    report(variants, "主样本 2024-07-01~2026-07-03（种子信号）")

    print("\n--- 2026H1 回撤段（2026-01-01~2026-07-03）---")
    for label, (n2, _) in variants.items():
        s = seg_stats(n2, "2026-01-01", "2026-07-03")
        if s:
            print(f"{label:<24s} ann={s[0]:7.2%} Sharpe={s[1]:6.2f} MDD={s[2]:7.2%}")

    # ── OOS 延伸：hist 合并库信号延至 2026-10-19 ──
    nav_ext = run_engine_hist()
    variants_ext = evaluate(nav_ext)
    report(variants_ext, "延伸样本 2024-07-01~2026-10-19（hist 合并库，含实盘段）")

    print("\n--- OOS 段（2026-07-06~2026-10-19，当前实盘深回撤段）---")
    for label, (n2, _) in variants_ext.items():
        s = seg_stats(n2, "2026-07-06", EXT_END)
        if s:
            print(f"{label:<24s} ann={s[0]:7.2%} Sharpe={s[1]:6.2f} MDD={s[2]:7.2%}")

    # ── 门限裁决（主样本）──
    base = {k: stats(v[0]) for k, v in variants.items()}
    v3 = base["V3 现生产(-3/-1 半仓)"]
    print("\n── 门限裁决（预登记：Sharpe≥V3 且 年化≥V3 且 MDD 恶化<3pp）──")
    for label in ["V7 dip-6%", "V8 dip-8%", "V9 dip-10%"]:
        ann, sharpe, mdd = base[label]
        ok = sharpe >= v3[1] and ann >= v3[0] and mdd >= v3[2] - 0.03
        print(
            f"  {label}: Sharpe {sharpe:.3f} vs {v3[1]:.3f} | ann {ann:.2%} vs {v3[0]:.2%}"
            f" | MDD {mdd:.2%} vs {v3[2]:.2%}  → {'PASS' if ok else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
