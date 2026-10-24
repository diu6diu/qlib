"""实验E（2026-10-23，预登记于 FINDINGS.md）：顶部过热减仓 top-breaker。

ret20 上穿 T 后延迟 D 个交易日降半仓（动量段跑完再躲衰竭段），ret20
回落至 R 以下回场。与底部 V3′（kill −3%/recover −1.5%/50%）独立共存，
仓位 = 底部状态 × 顶部状态。T 收盘判定 T+1 生效，切换收 0.2% 单边。
用缓存 NAV（bt_deep_kill.py 落盘）：主样本 + 延伸样本。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/jerry/workspace/trading_v2/qlib")
sys.path.insert(0, str(ROOT))

from experiments.alpha_enhancement.bt_breaker_dip import seg_stats  # noqa: E402
from experiments.alpha_enhancement.bt_self_breaker import SWITCH_COST, stats  # noqa: E402

CACHE = ROOT / "experiments" / "alpha_enhancement" / "results" / "cache"
KILL, RECOVER = -0.03, -0.015  # 底部 V3′（生产口径）
R_TOP = 0.03  # 顶部回场线
TRIGGERS = [0.08, 0.12]
DELAYS = [0, 10, 20]


def _expo_bottom(ret20: pd.Series) -> list[float]:
    state, out = "ON", []
    for v in ret20:
        if pd.notna(v):
            if state == "ON" and v <= KILL:
                state = "OFF"
            elif state == "OFF" and v > RECOVER:
                state = "ON"
        out.append(1.0 if state == "ON" else 0.5)
    return out


def _expo_top(ret20: pd.Series, trig: float, delay: int) -> list[float]:
    """上穿 trig（自下而上）→ 倒计时 delay 天 → 半仓；ret20 < R_TOP 回场
    （倒计时内跌破 R_TOP 则取消）。再次上穿重新计时。"""
    state, out, countdown = "ON", [], None
    prev = np.nan
    for v in ret20:
        if pd.notna(v):
            if state == "ON":
                if pd.notna(prev) and prev < trig <= v and countdown is None:
                    countdown = delay  # 上穿，开始倒计时
                if countdown is not None:
                    if v < R_TOP:
                        countdown = None  # 自燃回落，取消
                    else:
                        countdown -= 1
                        if countdown <= 0:
                            state, countdown = "OFF", None
            elif state == "OFF" and v < R_TOP:
                state = "ON"
        out.append(1.0 if state == "ON" else 0.5)
        prev = v
    return out


def run(nav: pd.Series, trig: float | None, delay: int) -> tuple[pd.Series, int]:
    r = nav.pct_change().fillna(0.0)
    ret20 = nav.pct_change(20)
    eb = _expo_bottom(ret20)
    et = _expo_top(ret20, trig, delay) if trig is not None else [1.0] * len(nav)
    e = pd.Series([a * b for a, b in zip(eb, et)], index=nav.index).shift(1).fillna(1.0)
    d = e.diff().abs().fillna(0.0)
    n2 = (1 + r * e - SWITCH_COST * d).cumprod()
    return n2, int((d > 0).sum())


def main() -> None:
    for navfile, seg in [
        ("nav_main_20240701_20260703.pkl", "主样本 2024-07-01~2026-07-03"),
        ("nav_ext_20240701_20261019.pkl", "延伸样本 2024-07-01~2026-10-19"),
    ]:
        nav = pd.read_pickle(CACHE / navfile)
        print(f"\n{'── ' + seg + ' ──'}")
        print(f"{'变体':<22s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>8s} {'切换':>4s}")
        n2, sw = run(nav, None, 0)
        print(f"{'V3′ 纯底部（对照）':<22s} {stats(n2)[0]:8.2%} {stats(n2)[1]:7.3f} {stats(n2)[2]:8.2%} {sw:4d}")
        for t in TRIGGERS:
            for d in DELAYS:
                n2, sw = run(nav, t, d)
                a, s, m = stats(n2)
                print(f"T+{t:.0%} D{d:<2d}             {a:8.2%} {s:7.3f} {m:8.2%} {sw:4d}")
        if "ext" in navfile:
            print("\n--- OOS 段（2026-07-06~2026-10-19）---")
            for t, d in [(None, 0)] + [(t, d) for t in TRIGGERS for d in DELAYS]:
                n2, _ = run(nav, t, d or 0)
                s = seg_stats(n2, "2026-07-06", "2026-10-19")
                label = "V3′ 纯底部" if t is None else f"T+{t:.0%} D{d}"
                print(f"{label:<22s} ann={s[0]:7.2%} Sharpe={s[1]:6.2f} MDD={s[2]:7.2%}")


if __name__ == "__main__":
    main()
