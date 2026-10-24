"""实验A（2026-10-20，预登记于 experiments/alpha_enhancement/FINDINGS.md）：
断路器压力期"全仓替换"增益检验。

问题：Phase 21 断路器 OFF（ret20 ≤ −3% → 半仓）期间，若取消 n_drop 限制、
全仓替换为当日新鲜 top10，恢复是否更快？该方案成立的核心假设是：压力期
"当日新鲜 top10"相对"现有持仓"有显著为正的前向收益差。本脚本在纯信号层
（无摩擦、无成本）检验该假设，并给出成本调整后的结论。

口径（与 bt_self_breaker.py 一致）：
- 信号：sig_hist 合并库 base/p0（2024-07-01 ~ 2026-10-19，含实盘段），
  blend50 = 逐日截面 z-score 等权平均再 z-score（Phase 15-A 配方）。
- 持仓：official TopkDropout tk10/nd2 无摩擦逐日模拟（select_official
  语义逐字复制自 generate_live_picks.py）。
- 持仓链日收益：H(t) 赚 close(t)→close(t+1) 等权（忽略漂移，仅用于
  识别压力段）；ret20 = 20 日滚动求和；压力事件 = ret20 ≤ −3%。
- 前向收益：T+1 开盘 → T+h 收盘，h ∈ {1,5,10}，等权。
- 对照：全部日期 / 非压力日期的同一差值（检验压力期特异性）。
- 成本：另报增量成本 ≈ 0.4%×(1−overlap/10) − 0.08%（全换双边 vs 正常
  nd2 日换手），净差值 = 毛差值 − 增量成本。

样本说明："全部触发日"样本的 5d/10d 前向窗口互相重叠，t 值仅供参考；
主样本为"事件段首日"（连续触发日合并为一段，取首日）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/jerry/workspace/trading_v2/qlib")
sys.path.insert(0, str(ROOT))

START, END = "2024-07-01", "2026-10-19"
TOPK, N_DROP = 10, 2
KILL = -0.03
HORIZONS = (1, 5, 10)
COST_FULL_SWAP = 0.004  # 全换：卖 100% + 买 100%，单边 0.2%（bt_self_breaker SWITCH_COST 同量级）
COST_ND2_DAY = 0.0008  # 正常 nd2 日：卖 20% + 买 20%


def select_official(scores: pd.Series, holdings: set, topk: int, n_drop: int) -> list:
    """逐字复制自 generate_live_picks.py（official TopkDropout 语义）。"""
    scores = scores.dropna()
    if not holdings:
        return list(scores.sort_index().nlargest(topk).index)
    held_scores = scores.reindex(list(holdings)).dropna()
    to_drop = sorted(held_scores.index, key=lambda s: (held_scores[s], s))[:n_drop]
    cand = scores.drop(index=list(holdings), errors="ignore")
    can_add = sorted(cand.index, key=lambda s: (cand[s], s), reverse=True)[:n_drop]
    selected = sorted(set(holdings) - set(to_drop)) + can_add
    if len(selected) < topk:
        rest = [c for c in sorted(cand.index, key=lambda s: (cand[s], s), reverse=True) if c not in selected]
        selected += rest[: topk - len(selected)]
    return selected


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return np.nan
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def summarize(diff: pd.Series, label: str) -> None:
    d = diff.dropna()
    if len(d) == 0:
        print(f"  {label:<22s} n=0")
        return
    print(
        f"  {label:<22s} n={len(d):3d}  mean={d.mean():+7.2%}  median={d.median():+7.2%}"
        f"  t={t_stat(d):+5.2f}  win%={100 * (d > 0).mean():5.1f}"
    )


def main() -> None:
    import qlib  # noqa: E402

    from experiments.rightside_alpha.data import get_tradable_mask, load_qlib_data  # noqa: E402
    from experiments.rightside_alpha.factors import cs_zscore  # noqa: E402

    uri = os.path.expanduser("~/.qlib/qlib_data/cn_data_mootdx")
    qlib.init(provider_uri=uri, region="cn")
    data = load_qlib_data(
        "mainboard", "2023-01-01", END, fields=("$open", "$close", "$volume"), provider_uri=uri
    )
    open_, close, volume = data["open"], data["close"], data["volume"]
    tradable = get_tradable_mask(close, volume)

    hist_dir = ROOT / "examples/benchmarks_dynamic/mainboard_leftside/results/mainboard_leftside/sig_hist"

    def load_wide(name: str) -> pd.DataFrame:
        sig = pd.read_pickle(hist_dir / name)
        if isinstance(sig, pd.DataFrame):
            sig = sig[sig.columns[0]]
        return sig.unstack(level="instrument").reindex(index=close.index, columns=close.columns)

    z0 = cs_zscore(load_wide("base.pkl").where(tradable))
    z1 = cs_zscore(load_wide("p0.pkl").where(tradable))
    z = cs_zscore((z0 + z1) / 2)  # blend50，Phase 15-A 配方
    print("signals blended.")

    # ── 1. 无摩擦 official TopkDropout 逐日模拟 ──
    dates = list(close.index[(close.index >= pd.Timestamp(START)) & (close.index <= pd.Timestamp(END))])
    holdings: set = set()
    books: dict[pd.Timestamp, set] = {}
    for t in dates:
        holdings = set(select_official(z.loc[t], holdings, TOPK, N_DROP))
        books[t] = holdings

    # 持仓链日收益（等权，close→close，忽略漂移）
    port_ret = pd.Series(index=dates, dtype=float)
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        h = sorted(books[t])
        port_ret[t1] = (close.loc[t1, h] / close.loc[t, h] - 1).mean()
    ret20 = port_ret.rolling(20).sum()

    stress = ret20[ret20 <= KILL].index
    # 连续触发日合并为事件段
    episodes: list[list[pd.Timestamp]] = []
    for t in stress:
        if episodes and (dates.index(t) - dates.index(episodes[-1][-1])) == 1:
            episodes[-1].append(t)
        else:
            episodes.append([t])
    print(f"\n样本: {dates[0].date()} ~ {dates[-1].date()} 共 {len(dates)} 个交易日")
    print(f"压力触发日(ret20<={KILL:.0%}): {len(stress)} 天, 合并为 {len(episodes)} 个事件段")

    # ── 2. 前向收益：fresh top10 vs held ──
    cal = list(close.index)
    cal_idx = {d: i for i, d in enumerate(cal)}

    def fwd(names: list, T: pd.Timestamp, h: int) -> float:
        i = cal_idx[T]
        if i + h >= len(cal):
            return np.nan
        o = open_.loc[cal[i + 1], names]
        c = close.loc[cal[i + h], names]
        r = (c / o - 1).dropna()
        return float(r.mean()) if len(r) else np.nan

    rows = []
    for t in dates:
        s = z.loc[t].dropna()
        fresh = list(s.sort_index().nlargest(TOPK).index)
        held = sorted(books[t])
        row = {"date": t, "stress": t in set(stress), "overlap": len(set(fresh) & set(held))}
        for h in HORIZONS:
            f_fresh, f_held = fwd(fresh, t, h), fwd(held, t, h)
            row[f"diff{h}"] = f_fresh - f_held if pd.notna(f_fresh) and pd.notna(f_held) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date")

    ep_first = [e[0] for e in episodes]
    samples = {
        "压力-事件段首日": df.loc[ep_first],
        "压力-全部触发日": df[df["stress"]],
        "非压力日": df[~df["stress"]],
        "全部日期": df,
    }

    for h in HORIZONS:
        print(f"\n── 前向 {h}d 差值（fresh top10 − held），毛收益 ──")
        for label, sub in samples.items():
            summarize(sub[f"diff{h}"], label)

    print("\n── overlap（|fresh ∩ held|，满分 10）──")
    for label in ["压力-事件段首日", "压力-全部触发日", "非压力日"]:
        sub = samples[label]["overlap"]
        print(f"  {label:<22s} n={len(sub):3d}  mean={sub.mean():.1f}  median={sub.median():.0f}")

    # ── 3. 成本调整（事件段首日，5d 主口径）──
    ep = df.loc[ep_first].copy()
    ep["inc_cost"] = COST_FULL_SWAP * (1 - ep["overlap"] / TOPK) - COST_ND2_DAY
    ep["net5"] = ep["diff5"] - ep["inc_cost"]
    print("\n── 事件段首日 5d：成本调整后 ──")
    print(f"  平均增量成本 = {ep['inc_cost'].mean():.2%}（全换相对正常 nd2 日）")
    summarize(ep["net5"], "净差值（毛−增量成本）")

    # ── 4. 事件段明细 ──
    print("\n── 事件段明细 ──")
    print(f"{'段首日':<12s}{'段末日':<12s}{'天数':>4s}{'min_ret20':>10s}{'overlap':>8s}{'diff5d(首日)':>13s}")
    for e in episodes:
        d0 = e[0]
        print(
            f"{d0.date()!s:<12s}{e[-1].date()!s:<12s}{len(e):>4d}{ret20.loc[e].min():>10.1%}"
            f"{df.loc[d0, 'overlap']:>8.0f}{df.loc[d0, 'diff5']:>+13.2%}"
        )

    # ── 5. 预登记门限裁决 ──
    gate_mean = ep["diff5"].mean()
    # 按段胜率：段内所有触发日的 5d 差值均值 > 0 的段占比
    ep_win = np.mean([df.loc[e]["diff5"].mean() > 0 for e in episodes]) if episodes else np.nan
    print(f"\n── 门限裁决（预登记）──")
    print(f"  事件段首日 5d 毛差值均值 = {gate_mean:+.2%}（门限 ≥ +0.5%）")
    print(f"  按段胜率 = {ep_win:.0%}（门限 ≥ 60%）")
    verdict = "PASS → 进入实验B（V7 回测变体）" if (gate_mean >= 0.005 and ep_win >= 0.6) else "FAIL → 否决全仓替换"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
