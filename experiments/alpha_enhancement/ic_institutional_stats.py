"""双模型日频 IC 的机构口径统计（均值/ICIR/胜率），2026-10-23。

口径与 ic_streak_history.py 完全一致（日截面 rank IC，fwd=open[t+6]/open[t+1]-1）。
"""

import sys
from pathlib import Path

REPO = Path("/Users/jerry/workspace/trading_v2/qlib")
sys.path.insert(0, str(REPO / "examples/benchmarks_dynamic/mainboard_leftside"))
sys.path.insert(0, str(REPO / "experiments/alpha_enhancement"))

from generate_live_picks import H, rank_ic_daily  # noqa: E402
from ic_streak_history import merge_hist  # noqa: E402
from common import init_qlib  # noqa: E402
from qlib.data import D  # noqa: E402


def main():
    init_qlib()
    hist_base, hist_p0 = merge_hist("base"), merge_hist("p0")
    z0 = hist_base["score"].unstack(level="instrument")
    z1 = hist_p0["score"].unstack(level="instrument")
    instruments = sorted(set(z0.columns) | set(z1.columns))
    px = D.features(instruments, ["$open", "$close", "$volume"], start_time="2023-01-01")
    open_ = px["$open"].unstack(level="instrument")
    close_ = px["$close"].unstack(level="instrument")
    vol_ = px["$volume"].unstack(level="instrument")
    valid = close_.notna() & (vol_ > 0)
    tradable = valid & (close_.notna().cumsum() >= 60)
    z0 = z0.reindex(index=open_.index, columns=open_.columns).where(tradable)
    z1 = z1.reindex(index=open_.index, columns=open_.columns).where(tradable)
    fwd = (open_.shift(-(H + 1)) / open_.shift(-1) - 1.0).where(tradable)

    for name, z in (("base", z0), ("p0", z1)):
        daily = rank_ic_daily(z, fwd).dropna()
        mean, std = daily.mean(), daily.std()
        icir_d = mean / std
        print(
            f"{name}: 日IC {len(daily)} 天 | 均值 {mean:.3f} | 标准差 {std:.3f} | "
            f"ICIR(日) {icir_d:.2f} | ICIR(年化) {icir_d * (252 ** 0.5):.1f} | "
            f"IC>0 胜率 {(daily > 0).mean():.0%} | |IC|>0.03 占比 {(daily.abs() > 0.03).mean():.0%}"
        )


if __name__ == "__main__":
    main()
