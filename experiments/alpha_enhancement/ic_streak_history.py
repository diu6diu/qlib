"""历史 IC 连负（Phase 19 de-risk）触发情况回溯。

口径与 generate_live_picks.py 完全一致：
  fwd = open[t+6]/open[t+1] - 1；每日截面 rank IC；rolling(20).mean().shift(6)；
  触发 = base 与 p0 双 IC < 0 连续 >= 10 个交易日（每日重判定，非 latch）。

用法：.venv/bin/python experiments/alpha_enhancement/ic_streak_history.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "benchmarks_dynamic" / "mainboard_leftside"))

from generate_live_picks import H, IC_WIN, DERISK_DAYS, rank_ic_daily  # noqa: E402
from common import init_qlib  # noqa: E402
from qlib.data import D  # noqa: E402

SIG_HIST = REPO / "examples/benchmarks_dynamic/mainboard_leftside/results/mainboard_leftside/sig_hist"
SEED_DIR = REPO / "experiments/alpha_enhancement/signals"


def merge_hist(name: str) -> pd.DataFrame:
    """合并 seed + sig_hist（同 merge_hist_store 的语义：seed 打底、hist 追加）。"""
    frames = []
    seed = SEED_DIR / f"sig_w5{'_p0' if name == 'p0' else ''}.pkl"
    if seed.exists():
        frames.append(pd.read_pickle(seed))
    live = SIG_HIST / f"{name}.pkl"
    if live.exists():
        frames.append(pd.read_pickle(live))
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def main():
    init_qlib()
    hist_base, hist_p0 = merge_hist("base"), merge_hist("p0")
    print(f"信号历史: base {hist_base.index.get_level_values(0).min().date()} ~ "
          f"{hist_base.index.get_level_values(0).max().date()}")

    z0 = hist_base["score"].unstack(level="instrument")
    z1 = hist_p0["score"].unstack(level="instrument")
    start = pd.Timestamp("2023-01-01")
    instruments = sorted(set(z0.columns) | set(z1.columns))
    px = D.features(instruments, ["$open", "$close", "$volume"], start_time=str(start.date()))
    open_ = px["$open"].unstack(level="instrument")
    close_ = px["$close"].unstack(level="instrument")
    vol_ = px["$volume"].unstack(level="instrument")
    valid = close_.notna() & (vol_ > 0)
    tradable = valid & (close_.notna().cumsum() >= 60)

    z0 = z0.reindex(index=open_.index, columns=open_.columns).where(tradable)
    z1 = z1.reindex(index=open_.index, columns=open_.columns).where(tradable)
    fwd = (open_.shift(-(H + 1)) / open_.shift(-1) - 1.0).where(tradable)

    ic0 = rank_ic_daily(z0, fwd).rolling(IC_WIN).mean().shift(H + 1)
    ic1 = rank_ic_daily(z1, fwd).rolling(IC_WIN).mean().shift(H + 1)

    # 双 IC 为负的 mask（两者都非 NaN 且 < 0）
    both_neg = (ic0 < 0) & (ic1 < 0) & ic0.notna() & ic1.notna()

    # 连负 streak 分段
    streak_id = (both_neg != both_neg.shift()).cumsum()
    segs = both_neg.groupby(streak_id).agg(["sum", "first"])
    runs = segs[segs["first"]]
    print(f"\n===== 双 IC < 0 连负段（共 {len(runs)} 段，streak >= {DERISK_DAYS} 天会触发半仓）=====")
    out_rows = []
    for sid, row in runs.iterrows():
        days = both_neg[streak_id == sid]
        idx = days.index
        i_end = ic0.index.get_loc(idx[-1])
        ic0_seg = ic0.loc[idx[0] : idx[-1]]
        ic1_seg = ic1.loc[idx[0] : idx[-1]]
        out_rows.append(
            {
                "start": idx[0].date(),
                "end": idx[-1].date(),
                "days": int(row["sum"]),
                "triggered": row["sum"] >= DERISK_DAYS,
                "ic0_worst": round(float(ic0_seg.min()), 4),
                "ic1_worst": round(float(ic1_seg.min()), 4),
            }
        )
    for r in out_rows:
        flag = " *** 触发半仓" if r["triggered"] else ""
        print(f"{r['start']} ~ {r['end']}  连负 {r['days']:>2} 天{flag}  (ic0最低 {r['ic0_worst']:+.3f}, ic1最低 {r['ic1_worst']:+.3f})")

    # 当前状态
    cur = both_neg.iloc[::-1]
    streak_now = 0
    for v in cur:
        if v:
            streak_now += 1
        else:
            break
    last = ic0.dropna().index.max()
    print(f"\n当前（截至 {last.date()}）：双 IC 连负 streak = {streak_now} 天（触发线 {DERISK_DAYS}）")
    print(f"  ic0={ic0.loc[last]:+.4f}  ic1={ic1.loc[last]:+.4f}")

    # 保存 IC 序列备查（ic0/ic1 = 20 日均值；ic0_d/ic1_d = 日值，供日报
    # 计算滚动 ICIR 与胜率——机构健康标准三件套：均值/ICIR/胜率）
    ic0_d = rank_ic_daily(z0, fwd)
    ic1_d = rank_ic_daily(z1, fwd)
    out = pd.DataFrame({"ic0": ic0, "ic1": ic1, "both_neg": both_neg, "ic0_d": ic0_d, "ic1_d": ic1_d})
    out_path = REPO / "experiments/alpha_enhancement/results/ic_history.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
