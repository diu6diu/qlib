"""实验C（2026-10-20，预登记于 FINDINGS.md）：断路器 recover 回场阈值扫描。

V3 口径（两态 ON/HALF，kill −3%、OFF→50%）固定，扫描 recover ∈
{−0.5%, −1.0%(=V3), −1.5%, −2.0%, −2.5%}。口径与 bt_self_breaker.py /
bt_breaker_dip.py 完全一致：T 收盘判定 T+1 生效，切换收 0.2% 单边。
主样本 = 种子信号 2024-07-01~2026-07-03；OOS 延伸 = sig_hist 合并库
延至 2026-10-19。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd

ROOT = Path("/Users/jerry/workspace/trading_v2/qlib")
sys.path.insert(0, str(ROOT))

from experiments.alpha_enhancement.bt_breaker_dip import run_engine_hist, seg_stats  # noqa: E402
from experiments.alpha_enhancement.bt_self_breaker import apply_breaker, run_engine, stats  # noqa: E402

KILL, HALF = -0.03, 0.5
RECOVERS = [("R-0.5%", -0.005), ("R-1.0%(=V3)", -0.01), ("R-1.5%", -0.015), ("R-2.0%", -0.02), ("R-2.5%", -0.025)]


def evaluate(nav: pd.Series) -> dict[str, tuple[pd.Series, dict]]:
    ret_w = nav.pct_change().rolling(20).sum()
    out = {"V0 基线": (nav, dict(switches=0, expo_mean=1.0))}
    for label, rec in RECOVERS:
        out[label] = apply_breaker(nav, ret_w, KILL, rec, HALF)
    return out


def report(variants: dict, title: str) -> None:
    print(f"\n{'── ' + title + ' ──'}")
    print(f"{'变体':<14s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>8s} {'切换':>4s} {'平均仓位':>8s}")
    for label, (n2, info) in variants.items():
        ann, sharpe, mdd = stats(n2)
        print(f"{label:<14s} {ann:8.2%} {sharpe:7.3f} {mdd:8.2%} {info['switches']:4d} {info['expo_mean']:8.1%}")


def main() -> None:
    nav = run_engine()
    variants = evaluate(nav)
    report(variants, "主样本 2024-07-01~2026-07-03（种子信号）")

    print("\n--- 2026H1 回撤段 ---")
    for label, (n2, _) in variants.items():
        s = seg_stats(n2, "2026-01-01", "2026-07-03")
        if s:
            print(f"{label:<14s} ann={s[0]:7.2%} Sharpe={s[1]:6.2f} MDD={s[2]:7.2%}")

    nav_ext = run_engine_hist()
    variants_ext = evaluate(nav_ext)
    report(variants_ext, "延伸样本 2024-07-01~2026-10-19（hist 合并库，含实盘段）")

    print("\n--- OOS 段（2026-07-06~2026-10-19）---")
    for label, (n2, _) in variants_ext.items():
        s = seg_stats(n2, "2026-07-06", "2026-10-19")
        if s:
            print(f"{label:<14s} ann={s[0]:7.2%} Sharpe={s[1]:6.2f} MDD={s[2]:7.2%}")

    # ── 门限裁决（主样本 vs V3，MDD 容差收紧至 1pp）──
    base = {k: stats(v[0]) for k, v in variants.items()}
    base_ext = {k: stats(v[0]) for k, v in variants_ext.items()}
    v3 = base["R-1.0%(=V3)"]
    print("\n── 门限裁决（Sharpe≥V3 且 年化≥V3 且 MDD 恶化<1pp 且 OOS 段方向不反转）──")
    for label, _ in RECOVERS[1:]:  # 跳过 R-1.0% 自身
        ann, sharpe, mdd = base[label]
        oos_v3 = seg_stats(variants_ext["R-1.0%(=V3)"][0], "2026-07-06", "2026-10-19")
        oos_x = seg_stats(variants_ext[label][0], "2026-07-06", "2026-10-19")
        oos_ok = oos_x[0] >= oos_v3[0] if (oos_x and oos_v3) else False
        ok = sharpe >= v3[1] and ann >= v3[0] and mdd >= v3[2] - 0.01 and oos_ok
        print(
            f"  {label}: Sharpe {sharpe:.3f} vs {v3[1]:.3f} | ann {ann:.2%} vs {v3[0]:.2%}"
            f" | MDD {mdd:.2%} vs {v3[2]:.2%} | OOS ann {oos_x[0]:.2%} vs {oos_v3[0]:.2%}"
            f"  → {'PASS' if ok else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
