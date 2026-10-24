"""实验D（2026-10-23，预登记于 FINDINGS.md）：深跌清仓档（三态断路器）。

V3′ 两态（kill −3%、recover −1.5%、OFF→50%）加深跌档：ret20 ≤ D → 0%，
ret20 > D+2pp → 回 50%。对照 V0 基线 / V3′。口径同实验C：T 收盘判定
T+1 生效，切换收 0.2% 单边 × |Δexposure|。
主样本 = 种子信号 2024-07-01~2026-07-03；OOS 延伸 = sig_hist 合并库
延至 2026-10-19（bt_breaker_dip.run_engine_hist）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/jerry/workspace/trading_v2/qlib")
sys.path.insert(0, str(ROOT))

from experiments.alpha_enhancement.bt_breaker_dip import run_engine_hist, seg_stats  # noqa: E402
from experiments.alpha_enhancement.bt_self_breaker import SWITCH_COST, run_engine, stats  # noqa: E402

KILL, RECOVER, HALF = -0.03, -0.015, 0.5
HYST = 0.02
DEEP_KILLS = [("D-6%", -0.06), ("D-8%", -0.08), ("D-10%", -0.10), ("D-12%", -0.12)]

CACHE = ROOT / "experiments" / "alpha_enhancement" / "results" / "cache"
NAV_MAIN_PKL = CACHE / "nav_main_20240701_20260703.pkl"
NAV_EXT_PKL = CACHE / "nav_ext_20240701_20261019.pkl"


def apply_breaker_3state(
    nav: pd.Series, ret_w: pd.Series, deep: float
) -> tuple[pd.Series, dict]:
    """三态：ON=100% / HALF=50% / FLAT=0%。深跌优先判定（ON 可直跳 FLAT）。"""
    r = nav.pct_change().fillna(0.0)
    state, exposure, switches = "ON", [], 0
    for i in range(len(nav)):
        rw = ret_w.iloc[i]
        if pd.notna(rw):
            if state != "FLAT" and rw <= deep:
                state, switches = "FLAT", switches + 1
            elif state == "FLAT" and rw > deep + HYST:
                state, switches = "HALF", switches + 1
            elif state == "ON" and rw <= KILL:
                state, switches = "HALF", switches + 1
            elif state == "HALF" and rw > RECOVER:
                state, switches = "ON", switches + 1
        exposure.append({"ON": 1.0, "HALF": HALF, "FLAT": 0.0}[state])
    expo = pd.Series(exposure, index=nav.index).shift(1).fillna(1.0)  # T+1 生效
    d_expo = expo.diff().abs().fillna(0.0)
    r_out = r * expo - SWITCH_COST * d_expo
    return (1 + r_out).cumprod(), dict(switches=switches, expo_mean=float(expo.mean()))


def apply_breaker_2state(nav: pd.Series, ret_w: pd.Series) -> tuple[pd.Series, dict]:
    """V3′：两态 kill −3% / recover −1.5% / OFF→50%（生产新口径）。"""
    r = nav.pct_change().fillna(0.0)
    state, exposure, switches = "ON", [], 0
    for i in range(len(nav)):
        rw = ret_w.iloc[i]
        if pd.notna(rw):
            if state == "ON" and rw <= KILL:
                state, switches = "HALF", switches + 1
            elif state == "HALF" and rw > RECOVER:
                state, switches = "ON", switches + 1
        exposure.append(1.0 if state == "ON" else HALF)
    expo = pd.Series(exposure, index=nav.index).shift(1).fillna(1.0)
    d_expo = expo.diff().abs().fillna(0.0)
    r_out = r * expo - SWITCH_COST * d_expo
    return (1 + r_out).cumprod(), dict(switches=switches, expo_mean=float(expo.mean()))


def evaluate(nav: pd.Series) -> dict[str, tuple[pd.Series, dict]]:
    ret_w = nav.pct_change().rolling(20).sum()
    out = {"V0 基线": (nav, dict(switches=0, expo_mean=1.0))}
    out["V3′ 两态"] = apply_breaker_2state(nav, ret_w)
    for label, d in DEEP_KILLS:
        out[label] = apply_breaker_3state(nav, ret_w, d)
    return out


def report(variants: dict, title: str, ret_w: pd.Series) -> None:
    print(f"\n{'── ' + title + ' ──'}")
    print(f"（样本内 ret20 最小值 {ret_w.min():.1%} @ {ret_w.idxmin().date()}）")
    print(f"{'变体':<10s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>8s} {'切换':>4s} {'平均仓位':>8s}")
    for label, (n2, info) in variants.items():
        ann, sharpe, mdd = stats(n2)
        print(f"{label:<10s} {ann:8.2%} {sharpe:7.3f} {mdd:8.2%} {info['switches']:4d} {info['expo_mean']:8.1%}")


def load_or_run(path: Path, runner) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_pickle(path)
    nav = runner()
    nav.to_pickle(path)
    return nav


def main() -> None:
    nav = load_or_run(NAV_MAIN_PKL, run_engine)
    ret_w = nav.pct_change().rolling(20).sum()
    variants = evaluate(nav)
    report(variants, "主样本 2024-07-01~2026-07-03（种子信号）", ret_w)

    print("\n--- 2026H1 回撤段（2026-01-01~2026-07-03）---")
    for label, (n2, _) in variants.items():
        s = seg_stats(n2, "2026-01-01", "2026-07-03")
        if s:
            print(f"{label:<10s} ann={s[0]:7.2%} Sharpe={s[1]:6.2f} MDD={s[2]:7.2%}")

    nav_ext = load_or_run(NAV_EXT_PKL, run_engine_hist)
    ret_w_ext = nav_ext.pct_change().rolling(20).sum()
    variants_ext = evaluate(nav_ext)
    report(variants_ext, "延伸样本 2024-07-01~2026-10-19（hist 合并库，含实盘段）", ret_w_ext)

    print("\n--- OOS 段（2026-07-06~2026-10-19，当前实盘深回撤段）---")
    for label, (n2, _) in variants_ext.items():
        s = seg_stats(n2, "2026-07-06", "2026-10-19")
        if s:
            print(f"{label:<10s} ann={s[0]:7.2%} Sharpe={s[1]:6.2f} MDD={s[2]:7.2%}")


if __name__ == "__main__":
    main()
