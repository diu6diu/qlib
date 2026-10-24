"""Generate live picks for the blend50 dual-model W5 strategy.

Pipeline position: after ``run_roll.py`` rolling retrains (base + P0, horizon=5)
and ``extract_signal.py`` dumps both signals.

Chain: dual signals -> cross-sectional z-score each, equal-weight average,
re-zscore (blend50, Phase 15-A, same recipe as bt_blend.py) -> ST removal ->
official TopkDropout (tk10, nd2) target portfolio -> rebalance instructions.

Rolling 20d rank-ICs of both signals no longer arbitrate between models
(Phase 15-A: blend50 replaced IC-switch — statistically equal returns, higher
ICIR, better roll20_min, smaller MDD, one mechanism less). They now drive the
exposure-level de-risk rule (Phase 19): when BOTH rolling ICs stay negative
for DERISK_DAYS consecutive trading days, the target portfolio is halved
(topk 10 -> 5) until the alarm clears. Model untouched, position cut —
per the Phase 15 verdict that model surgery is useless and exposure gating
is the adopted response.

Info columns (close / EPS / niyoy / sentiment) are attached for manual review
only — they do NOT filter anything (Phase 9~11 rejected all soft filters).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import common
from qlib.data import D

# Rolling-IC monitoring parameters — same calibration as the backtests (W5).
H = 5  # signal horizon (W5)
IC_WIN = 20  # rolling IC window (trading days)
DERISK_DAYS = 10  # both rolling ICs < 0 for this many consecutive days => half position

SENTIMENT_FIELDS = [
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
    "attention_score",
]


# ── Rolling IC (monitoring only, no longer used for model arbitration) ──


def rank_ic_daily(z: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Per-day spearman IC; z and fwd are (date x instrument) wide frames."""
    return z.corrwith(fwd, axis=1, method="spearman")


# ── Official TopkDropout semantics (matches run_rightside_substrategy.py) ──


def select_official(
    scores: pd.Series,
    holdings: set[str],
    topk: int,
    n_drop: int,
) -> list[str]:
    """Constant-topk portfolio: drop the n_drop worst-scored holdings, add the
    n_drop best-scored non-held candidates. If holdings fell below topk (e.g.
    ST force-sells), top back up to topk like qlib's TopkDropoutStrategy
    (signal_strategy.py: n_buy = n_drop + topk - len(last))."""
    scores = scores.dropna()
    if not holdings:
        # sort_index first: deterministic tie-break for heavily-tied scores.
        return list(scores.sort_index().nlargest(topk).index)

    held_scores = scores.reindex(list(holdings)).dropna()
    to_drop = sorted(held_scores.index, key=lambda s: (held_scores[s], s))[:n_drop]
    cand = scores.drop(index=list(holdings), errors="ignore")
    can_add = sorted(cand.index, key=lambda s: (cand[s], s), reverse=True)[:n_drop]
    selected = sorted(set(holdings) - set(to_drop)) + can_add
    if len(selected) < topk:  # e.g. ST force-sell shrank holdings: top back up
        rest = [c for c in sorted(cand.index, key=lambda s: (cand[s], s), reverse=True) if c not in selected]
        selected += rest[: topk - len(selected)]
    return selected


# ── Signal history store ──


def load_signal(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if isinstance(df, pd.Series):
        df = df.to_frame("score")
    if "score" not in df.columns:
        df = df.rename(columns={df.columns[0]: "score"})
    return df[["score"]]


def merge_hist_store(
    hist_dir: Path,
    seed_dir: Path,
    name: str,
    seed_name: str,
    new_sig: pd.DataFrame,
) -> pd.DataFrame:
    """Merge today's signal into the history store (in memory) and return it.

    First run seeds the store from the experiment signals (sig_w5*.pkl).
    Overlapping dates are overwritten by the newer signal. The store is NOT
    persisted here — call persist_hist_store after the picks CSV is written,
    so a crash mid-run cannot leave "signal stored but no holdings CSV".
    """
    path = hist_dir / f"{name}.pkl"
    if path.exists():
        old = load_signal(str(path))
    else:
        seed = seed_dir / seed_name
        if not seed.exists():
            sys.exit(f"Seed signal missing: {seed} — refusing to start IC history from scratch")
        old = load_signal(str(seed))

    new_dates = new_sig.index.get_level_values("datetime").unique()
    old = old[~old.index.get_level_values("datetime").isin(new_dates)]
    return pd.concat([old, new_sig]).sort_index()


def persist_hist_store(hist_dir: Path, name: str, merged: pd.DataFrame) -> None:
    hist_dir.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(hist_dir / f"{name}.pkl")


def compute_rolling_ics(
    hist_base: pd.DataFrame,
    hist_p0: pd.DataFrame,
    target_date: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (rolling_ic_base series, rolling_ic_p0 series, tradable mask row).

    The IC series drive the Phase 19 exposure de-risk rule (both ICs negative
    for DERISK_DAYS consecutive days => half-position target); the tradable
    mask (same definition as the backtest, data.py:get_tradable_mask) defines
    the universe over which the blend50 z-scores are computed.
    """
    z0 = hist_base["score"].unstack(level="instrument")
    z1 = hist_p0["score"].unstack(level="instrument")

    # Pin the OHLCV load start to the backtest's mask origin (2023-01-01) so
    # the >=60-day-history count matches get_tradable_mask unconditionally —
    # even if the hist store is ever re-seeded with a later start date.
    start = pd.Timestamp("2023-01-01")
    instruments = sorted(set(z0.columns) | set(z1.columns))
    print(f"Loading OHLCV for {len(instruments)} instruments since {start.strftime('%Y-%m-%d')}...")
    px = D.features(
        instruments,
        ["$open", "$close", "$volume"],
        start_time=start.strftime("%Y-%m-%d"),
        end_time=target_date.strftime("%Y-%m-%d"),
        freq="day",
    )
    open_ = px["$open"].unstack(level="instrument")
    close_ = px["$close"].unstack(level="instrument")
    vol_ = px["$volume"].unstack(level="instrument")
    # Same tradable mask as the backtest (data.py:get_tradable_mask):
    # non-NaN close, positive volume, >=60 days of history.
    valid = close_.notna() & (vol_ > 0)
    tradable = valid & (close_.notna().cumsum() >= 60)

    z0 = z0.reindex(index=open_.index, columns=open_.columns).where(tradable)
    z1 = z1.reindex(index=open_.index, columns=open_.columns).where(tradable)
    fwd = (open_.shift(-(H + 1)) / open_.shift(-1) - 1.0).where(tradable)

    ic0 = rank_ic_daily(z0, fwd).rolling(IC_WIN).mean().shift(H + 1)
    ic1 = rank_ic_daily(z1, fwd).rolling(IC_WIN).mean().shift(H + 1)

    # No silent asof fallback: the target date must exist in the IC calendar.
    if target_date not in ic0.index:
        sys.exit(f"{target_date.strftime('%Y-%m-%d')} not in IC calendar (data not patched to target?)")
    return ic0, ic1, tradable.loc[target_date]


# ── ST filter (Tencent names) ──


def fetch_stocks_from_tencent(
    codes: list[str], batch_size: int = 400
) -> tuple[dict[str, str], dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Fetch ({qlib_code: name}, price, open, prev_close, high, low) from Tencent.

    The qlib mootdx dump stores normalized prices (~1.0 scale), so the real
    quote price is taken from here for the review columns. After the close,
    ``price`` is the day's real close and ``open`` the day's real open — used
    for the manual broker cross-check (close / buy-day open / pct change) and
    for the limit-up seal detection (open==high==low & gap>9.5% buys).
    """
    names: dict[str, str] = {}
    prices: dict[str, float] = {}
    opens: dict[str, float] = {}
    prev_closes: dict[str, float] = {}
    highs: dict[str, float] = {}
    lows: dict[str, float] = {}
    for i in range(0, len(codes), batch_size):
        batch = codes[i : i + batch_size]
        tencent_codes = ",".join(common.tencent_code(c) for c in batch)
        url = f"http://qt.gtimg.cn/q={tencent_codes}"
        try:
            r = requests.get(url, timeout=20)
            r.encoding = "gbk"
            for line in r.text.strip().split(";"):
                line = line.strip()
                if not line.startswith("v_"):
                    continue
                parts = line.split('="')
                if len(parts) != 2:
                    continue
                head, body = parts
                code = head.split("_")[-1]
                fields = body.split("~")
                if len(fields) < 2:
                    continue
                qcode = common.qlib_code(code)
                names[qcode] = fields[1]

                def _f(idx: int) -> float | None:
                    try:
                        return float(fields[idx])
                    except (IndexError, ValueError):
                        return None

                if (v := _f(3)) is not None:
                    prices[qcode] = v  # 现价（收盘后=当日收盘）
                if (v := _f(4)) is not None:
                    prev_closes[qcode] = v  # 昨收
                if (v := _f(5)) is not None:
                    opens[qcode] = v  # 今开
                if (v := _f(33)) is not None:
                    highs[qcode] = v  # 最高
                if (v := _f(34)) is not None:
                    lows[qcode] = v  # 最低
        except Exception as e:
            print(f"Tencent batch {i//batch_size+1} failed: {e}", file=sys.stderr)
    return names, prices, opens, prev_closes, highs, lows


def identify_st_names(names: dict[str, str]) -> set[str]:
    """Return Qlib codes whose names indicate ST / delisting risk."""
    st_codes = set()
    st_re = re.compile(r"(?<![A-Za-z0-9])[\*＊]?ST(?![A-Za-z0-9])|退")
    for code, name in names.items():
        if name and st_re.search(name):
            st_codes.add(code)
    return st_codes


# ── Info columns (manual review only, never filters) ──


def latest_completed_quarter(dt: pd.Timestamp) -> int:
    year, month = dt.year, dt.month
    if month in (1, 2, 3, 4):
        return (year - 1) * 100 + 4
    elif month in (5, 6, 7, 8):
        return year * 100 + 1
    elif month in (9, 10):
        return year * 100 + 2
    else:
        return year * 100 + 3


def load_info_columns(instruments: list[str], target_date: pd.Timestamp) -> pd.DataFrame:
    """Load EPS / niyoy / sentiment_mean for review. Tolerant to missing data.

    Real prices come from the Tencent quote fetch (qlib stores normalized prices).
    """
    info = pd.DataFrame(index=pd.Index(instruments, name="instrument"))
    date_str = target_date.strftime("%Y-%m-%d")

    try:
        yoy_period = latest_completed_quarter(target_date) - 100
        fund = D.features(
            instruments,
            ["P($$eps_q_q)", "P($$ni_q_q)", f"PRef($$ni_q_q, {yoy_period})"],
            start_time=date_str,
            end_time=date_str,
            freq="day",
        )
        if not fund.empty:
            fund = fund.reset_index()
            fund["instrument"] = fund["instrument"].astype(str).str.upper()
            fund = fund.set_index("instrument")
            info["eps_q"] = fund["P($$eps_q_q)"]
            info["niyoy_q"] = fund["P($$ni_q_q)"] / fund[f"PRef($$ni_q_q, {yoy_period})"] - 1.0
    except Exception as e:
        print(f"fundamentals load failed: {e}", file=sys.stderr)

    try:
        calendar = D.calendar(start_time=None, end_time=target_date, freq="day")
        if len(calendar) >= 2:
            prev_str = calendar[-2].strftime("%Y-%m-%d")
            sent = D.features(instruments, ["$sentiment_mean"], start_time=prev_str, end_time=prev_str, freq="day")
            if not sent.empty:
                info["sentiment_mean"] = sent["$sentiment_mean"].groupby(level="instrument").last()
    except Exception as e:
        print(f"sentiment load failed: {e}", file=sys.stderr)

    return info


# ── Previous holdings ──


def find_prev_picks_file(out_dir: Path, target_date: pd.Timestamp) -> tuple[Path, pd.Timestamp] | None:
    """Latest picks_live_*.csv strictly before target_date -> (file, date)."""
    best_date, best_file = None, None
    for f in out_dir.glob("picks_live_*.csv"):
        m = re.search(r"picks_live_(\d{8})\.csv$", f.name)
        if not m:
            continue
        d = pd.Timestamp(m.group(1))
        if d < target_date and (best_date is None or d > best_date):
            best_date, best_file = d, f
    return (best_file, best_date) if best_file is not None else None


def find_prev_holdings(out_dir: Path, target_date: pd.Timestamp, max_rows: int = 20) -> set[str] | None:
    """Latest picks_live_*.csv strictly before target_date -> held instruments."""
    prev = find_prev_picks_file(out_dir, target_date)
    if prev is None:
        return None
    best_file, best_date = prev
    df = pd.read_csv(best_file)
    if len(df) > max_rows:
        sys.exit(
            f"{best_file.name} has {len(df)} rows (>{max_rows}); looks like a legacy "
            "top-100 file — remove/migrate it before running the tk10 pipeline"
        )
    print(f"Previous holdings from {best_file.name} ({best_date.strftime('%Y-%m-%d')})")
    return set(df["instrument"].astype(str).str.strip().str.upper())


# ── Limit-up seal rule state (Phase 20, 2026-10-16) ──
# Mirrors the engine's limit_rule="keep" (bt_limit_rules.py): a scheduled BUY
# that turns out one-word-sealed/suspended at the execution open is abandoned
# (never chase the board open) and the whole swap is cancelled — its paired
# sell (best-scored of the day's drops) is kept. The sealed name stays in the
# exclusion set while it remains in the elite zone (score rank <= topk) and is
# released only after decaying out (a fresh signal). State file lives next to
# the picks CSV; entries are recorded automatically from the exec cross-check
# (yesterday's BUY vs today's open/high/low), assuming the user followed the
# printed paired-sell instruction.

LIMIT_SKIP_FILE = "limit_skip.json"

# ── Phase 21 自回撤断路器（V3 半仓版，bt_self_breaker.py 验证）──
# 跟踪纸面链自身收益：最近 20 个交易日（≈1 个月）的累计收益 ≤ -3% → 半仓
# 执行（正常换仓照旧、每笔按半仓金额）；回升 > -1.5% → 回场全仓。迟滞带防抖。
# 与 TF 腿快闸（run_paper_t3a.py §R8）同尺度同迟滞结构，只是 OFF 动作为
# 半仓而非清仓（V3 年化几乎不亏、Sharpe 最高）。T 收盘判定、T+1 执行：
# 本脚本在 T 日晚运行，用当日收盘数据更新状态，写入 T 的 picks（T+1 开盘
# 执行），与回测 exposure.shift(1) 口径一致。
# recover -1% → -1.5%（2026-10-23，实验C bt_breaker_recover_scan 边际 PASS：
# 年化 +0.48pp / Sharpe +0.015 / MDD 与切换次数不变，OOS 不劣；回场更早，
# 迟滞带 2pp→1.5pp 仍在稳定区内，-2.5% 起洗仓）。
BREAKER_FILE = "breaker.json"
BREAKER_WINDOW = 20
BREAKER_KILL = -0.03
BREAKER_RECOVER = -0.015
BREAKER_HIST_KEEP = 60  # 多存些，便于人工复查


def load_limit_skip(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            print(f"WARNING: {path} unreadable ({e}); starting fresh", file=sys.stderr)
    return {}


def save_limit_skip(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True))


def detect_sealed_buys(
    prev_df: pd.DataFrame,
    prev_date: pd.Timestamp,
    prev_sells: list,
    opens: dict,
    prev_closes: dict,
    names: dict,
    limit_skip: dict,
) -> list:
    """Phase 20 封板判定（实盘窗口内、选股前调用）：昨日计划买入、今日执行
    开盘一字封板/停牌/缺口 >+9.5% 的名字记入冷静名单 limit_skip（原地修改），
    返回新记录的名字列表。

    逐名 sanity：prev_close 必须≈昨日信号收盘（偏离 <2%），明显偏离说明拿到
    的是过期/错配行情（如周末回溯运行），该条只打印、不记录——2026-10-17
    周六彩排事故的教训。配对保留者按昨日 picks 得分排序一一对应还原（卖出者
    不在昨日 picks 里、无得分，按代码序兜底）。只在实盘窗口（上一信号执行日
    == 实时行情日）由调用方保证后调用。
    """
    prev_buys = prev_df[prev_df["action"] == "BUY"] if "action" in prev_df.columns else prev_df.iloc[0:0]
    if not len(prev_buys):
        return []
    prev_scores = (
        dict(zip(prev_df["instrument"].astype(str).str.strip().str.upper(), prev_df["score"]))
        if "score" in prev_df.columns
        else {}
    )
    prev_buys_bf = sorted(
        (str(c).strip().upper() for c in prev_buys["instrument"]),
        key=lambda c: (-prev_scores.get(c, -np.inf), c),
    )
    prev_sells_bf = sorted(prev_sells, key=lambda c: (-prev_scores.get(c, -np.inf), c))
    newly = []
    for _, r in prev_buys.iterrows():
        c = str(r["instrument"]).strip().upper()
        sig_close = float(r["price"]) if pd.notna(r.get("price")) else np.nan
        o, pc = opens.get(c), prev_closes.get(c)
        if o is None or pc is None:
            sealed, reason = True, "停牌"  # 停牌/无行情：按买不进处理（与引擎 NaN open 一致）
        else:
            # Phase 20+：一字封板 或 开盘缺口>+9.5% 都视为买不进——实盘隔夜限价单
            # =收盘×1.095，>+9.5% 的开盘本就不会成交，且历史有毒
            # （diagnose_signalday_limit 2026-10-16：>+9.5%缺口成交持有期
            # -2.1%/胜率34% vs ≤+9.5% 的 +11.2%/胜率90%）。
            sealed = (o / pc - 1) > 0.095
            reason = f"开盘缺口 {o / pc - 1:+.1%}"
            if sealed and (not pd.notna(sig_close) or sig_close <= 0 or abs(pc / sig_close - 1) >= 0.02):
                print(
                    f"[state] 跳过 limit_skip 记录 {c}：prev_close {pc} 与信号日收盘 "
                    f"{sig_close} 无法对齐（偏离需 <2%），行情疑似错配"
                )
                sealed = False
        if sealed and c not in limit_skip:
            i = prev_buys_bf.index(c)
            kept = prev_sells_bf[i] if i < len(prev_sells_bf) else None
            limit_skip[c] = {"kept": kept, "since": prev_date.strftime("%Y-%m-%d")}
            print(
                f"Limit-skip recorded: {c} {names.get(c, '')} {reason} at "
                f"{prev_date.strftime('%Y-%m-%d')} exec open — 放弃买入，"
                f"保留 {kept or '(无配对卖出，空坑)'}"
            )
            newly.append(c)
    return newly


def apply_limit_skip_correction(prev_holdings: set, limit_skip: dict, newly_sealed: list) -> set:
    """Phase 20 持仓还原（选股前）：封板名从未成交 → 移出持仓；配对卖出者按
    规则保留，仅对当日新发现的条目一次性加回（与引擎 exec 时 restore 口径
    一致）。存量条目不重复加回 kept——kept 若已在后续选股中被正常轮出，重复
    加回 = 幽灵复活（2026-10-17 晋拓股份事故同类）。返回修正后的持仓集合。"""
    corrected = set(prev_holdings)
    for sealed, info in limit_skip.items():
        corrected.discard(sealed)
        if sealed in newly_sealed and info.get("kept"):
            corrected.add(info["kept"])
            print(f"台账修正: {sealed} 封板未成交移出持仓；{info['kept']} 配对轮出保留，回列持仓")
    return corrected


def load_breaker(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            print(f"WARNING: {path} unreadable ({e}); starting fresh", file=sys.stderr)
    return {"state": "ON", "hist": []}  # hist: [{"date": "YYYY-MM-DD", "r": float}]


def save_breaker(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1))


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="Generate live picks: blend50 dual-model W5, top10, ST-only filter")
    parser.add_argument("--sig-base", required=True, help="Path to today's base (pure W5) signal pickle")
    parser.add_argument("--sig-p0", required=True, help="Path to today's P0 (W5+P0) signal pickle")
    parser.add_argument(
        "--hist-dir",
        default=str(Path(__file__).resolve().parent / "results" / "mainboard_leftside" / "sig_hist"),
        help="Persistent signal-history store directory",
    )
    parser.add_argument(
        "--seed-dir",
        default=str(ROOT / "experiments" / "alpha_enhancement" / "signals"),
        help="Directory with seed signals (sig_w5.pkl / sig_w5_p0.pkl) for first run",
    )
    parser.add_argument(
        "--out-path",
        default=str(Path(__file__).resolve().parent / "results" / "mainboard_leftside" / "picks_live.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--topk", type=int, default=10, help="Portfolio size")
    parser.add_argument("--n-drop", type=int, default=2, help="Max names swapped per day (official semantics)")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD). Defaults to latest signal date.")
    parser.add_argument("--provider-uri", default=common.PROVIDER_URI, help="Qlib provider URI")
    args = parser.parse_args()

    common.init_qlib(provider_uri=args.provider_uri)

    sig_base = load_signal(args.sig_base)
    sig_p0 = load_signal(args.sig_p0)

    if args.date:
        target_date = pd.Timestamp(args.date)
        common.validate_trading_day(target_date, common.read_calendar(Path(args.provider_uri).expanduser()))
    else:
        target_date = min(
            sig_base.index.get_level_values("datetime").max(),
            sig_p0.index.get_level_values("datetime").max(),
        )
    print(f"Target signal date: {target_date.strftime('%Y-%m-%d')}")

    # Fail fast before the expensive OHLCV load if either signal lacks the date.
    for name, sig in (("base", sig_base), ("p0", sig_p0)):
        if target_date not in sig.index.get_level_values("datetime"):
            sys.exit(f"{name} signal has no rows on {target_date.strftime('%Y-%m-%d')}")

    # 1. Merge today's signals into the history stores (in memory; persisted
    #    only after the picks CSV is successfully written).
    hist_dir = Path(args.hist_dir)
    seed_dir = Path(args.seed_dir)
    hist_base = merge_hist_store(hist_dir, seed_dir, "base", "sig_w5.pkl", sig_base)
    hist_p0 = merge_hist_store(hist_dir, seed_dir, "p0", "sig_w5_p0.pkl", sig_p0)

    # 2. Rolling ICs: no longer model arbitration — they drive the Phase 19
    #    exposure de-risk rule (both ICs < 0 for DERISK_DAYS consecutive days
    #    => halve target portfolio). Also today's tradable mask (backtest
    #    definition: valid close, volume>0, >=60 days of history) which sets
    #    the z-score universe below.
    ic0, ic1, trad_today = compute_rolling_ics(hist_base, hist_p0, target_date)
    ic_b, ic_p = float(ic0.loc[target_date]), float(ic1.loc[target_date])
    tail0 = ic0.loc[:target_date].dropna().tail(DERISK_DAYS)
    tail1 = ic1.loc[:target_date].dropna().tail(DERISK_DAYS)
    derisk = (
        len(tail0) == DERISK_DAYS and len(tail1) == DERISK_DAYS and bool((tail0 < 0).all()) and bool((tail1 < 0).all())
    )
    print(f"Rolling 20d IC: ic_base={ic_b:.4f} ic_p0={ic_p:.4f}")
    if np.isnan(ic_b) or np.isnan(ic_p):
        print("NOTE: rolling IC is NaN (insufficient history) — de-risk rule inactive")
    elif derisk:
        print(
            f"DE-RISK ACTIVE: both rolling ICs < 0 for {DERISK_DAYS} consecutive days "
            f"— target portfolio halved ({args.topk} -> {args.topk // 2})"
        )
    elif ic_b < 0 and ic_p < 0:
        streak = 0
        for a, b in zip(tail0[::-1], tail1[::-1]):
            if a < 0 and b < 0:
                streak += 1
            else:
                break
        print(
            f"NOTE: both rolling ICs < 0 (streak {streak}/{DERISK_DAYS} days " f"— de-risk triggers at {DERISK_DAYS})"
        )

    # 3. blend50 (Phase 15-A recipe, same as bt_blend.py): cross-sectional
    #    z-score each signal over today's tradable universe, equal-weight
    #    average, re-zscore. Union alignment: a name missing from either
    #    signal yields NaN and is dropped, matching the backtest.
    day0 = sig_base.xs(target_date, level="datetime")["score"]
    day1 = sig_p0.xs(target_date, level="datetime")["score"]
    trad_codes = set(trad_today[trad_today].index)
    s0 = day0[day0.index.isin(trad_codes)]
    s1 = day1[day1.index.isin(trad_codes)]
    if s0.empty or s1.empty:
        sys.exit(f"no tradable signal rows on {target_date.strftime('%Y-%m-%d')}")

    def _z(s: pd.Series) -> pd.Series:
        return ((s - s.mean()) / s.std()).replace([np.inf, -np.inf], np.nan)

    day_sig = _z((_z(s0) + _z(s1)) / 2).dropna().sort_values(ascending=False)
    if day_sig.empty:
        sys.exit(
            f"blend50 cross-section is empty on {target_date.strftime('%Y-%m-%d')} " "— refusing to write empty picks"
        )
    n_all = len(set(day0.index) | set(day1.index))
    print(
        f"Candidates on {target_date.strftime('%Y-%m-%d')}: {len(day_sig)} "
        f"(of {n_all} signal rows; {n_all - len(day_sig)} non-tradable/incomplete dropped)"
    )

    # 4. Hard filter: ST names. Blend scores are still heavily tied (dozens of
    #    names share the top score), so the official tie-break can select adds
    #    from anywhere in the ranked list — the ST check must cover the whole
    #    cross-section, not just the top-N. Tradability (suspension) is already
    #    applied above via the tradable mask.
    out = Path(args.out_path)
    prev_holdings = find_prev_holdings(out.parent, target_date, max_rows=args.topk * 2) or set()

    # Phase 20 封板跳过规则：还原真实持仓。冷静名单中的名字是"曾计划买入
    # 但执行日封板而放弃"的（未成交），其配对轮出者按规则被保留。
    #
    # 实盘窗口判定：冷静名单的状态读写只在"上一信号日的执行日 == 实时行情
    # 所属交易日"时合法。周末/回溯运行时腾讯返回的是最近交易日的行情，与历
    # 史换仓错配——2026-10-17 周六回溯运行把 10-16 的 +9.7% 开盘缺口误记到
    # 10-14→10-15 换仓上（星帅尔实际 10-15 低开 -6% 已成交），幽灵复活已卖
    # 出的晋拓股份，导致 10-19 生产换仓凭空多卖一只、pick 变 11 只。窗口外：
    # 状态只读、按空名单处理、禁止写盘（成交核对处另有逐名 sanity：prev_close
    # 必须≈信号日收盘）。
    _cal = common.read_calendar(Path(args.provider_uri).expanduser())
    _prev_info = find_prev_picks_file(out.parent, target_date)
    _quote_day = max((d for d in _cal if d <= pd.Timestamp.now().normalize()), default=None)
    _exec_day = next((d for d in _cal if _prev_info is not None and d > _prev_info[1]), None)
    state_live = _quote_day is not None and _exec_day == _quote_day
    limit_skip_path = out.parent / LIMIT_SKIP_FILE
    if not state_live:
        print(
            f"[state] 非实盘执行窗口（上一信号执行日={_exec_day}, 实时行情日={_quote_day}），"
            "limit_skip 只读并按空名单处理"
        )
    limit_skip = load_limit_skip(limit_skip_path) if state_live else {}

    # Phase 20+ 封板判定前置到选股之前（2026-10-23 修复）：原先记录在 picks
    # 写盘之后的成交核对段，封板发现日的选股仍用未修正持仓——2026-10-21
    # 三峡新材封板未成交，当日却按"已持有"选股，错卖中钨高新（东山精密
    # 当日得分全场最低 1.34，本应被正常轮出）。现在选股前完成判定+还原，
    # 与引擎 keep 规则 exec 当日修正的口径一致。
    _prev_df, _prev_sells = None, []
    if state_live and _prev_info is not None:
        _prev_df = pd.read_csv(_prev_info[0])
        _prev_target = set(_prev_df["instrument"].astype(str).str.strip().str.upper())
        _prev_prev = find_prev_holdings(out.parent, _prev_info[1], max_rows=args.topk * 2) or set()
        # 配对保留者（昨日卖出名单）可能不在候选/持仓里，并入行情抓取，
        # 保证它被还原回持仓后当日价格可用。
        _prev_sells = sorted(_prev_prev - _prev_target)

    name_probe = list(day_sig.index)
    for c in prev_holdings | set(_prev_sells):
        if c not in name_probe:
            name_probe.append(c)
    print(f"Fetching names for {len(name_probe)} codes (ST check)...")
    names, prices, opens, prev_closes, highs, lows = fetch_stocks_from_tencent(name_probe)
    coverage = len(names) / max(1, len(name_probe))
    if coverage < 0.95:
        sys.exit(f"Tencent name coverage {coverage:.0%} < 95%; aborting rather than bypassing the ST filter")

    newly_sealed: list = []
    if _prev_df is not None:
        newly_sealed = detect_sealed_buys(_prev_df, _prev_info[1], _prev_sells, opens, prev_closes, names, limit_skip)
    if limit_skip:
        prev_holdings = apply_limit_skip_correction(prev_holdings, limit_skip, newly_sealed)
        print(
            f"Limit-skip cooling names: {sorted(limit_skip)} "
            f"(kept incumbents: {[i.get('kept') for i in limit_skip.values()]})"
        )

    st_codes = identify_st_names(names)
    if st_codes:
        print(f"ST / delisting names removed: {sorted(st_codes)}")

    scores = day_sig[~day_sig.index.isin(st_codes)]

    # Phase 20：冷静名单解禁（得分排名跌出 topk 精英区 = 作为全新信号重新
    # 允许入选），仍在精英区的继续排除（不追开板，与回测引擎口径一致）。
    if limit_skip:
        ranks = scores.rank(ascending=False, method="min")
        for c in list(limit_skip):
            if c not in ranks.index or ranks[c] > args.topk:
                print(f"Limit-skip released (out of elite zone): {c}")
                del limit_skip[c]
        if limit_skip:
            scores = scores.drop(index=[c for c in limit_skip if c in scores.index])
            print(f"Limit-skip excluded from candidates: {sorted(limit_skip)}")

    # Held stocks that turned ST are force-sold (hard risk rule, beyond n_drop).
    forced_sell = prev_holdings & st_codes
    holdings_for_select = prev_holdings - forced_sell

    # Phase 21：自回撤断路器状态更新（V3 半仓）。当日（target_date）组合
    # 收益 = 昨日 picks 十只的 pct_chg 均值（等权、收盘对收盘的纸面近似）。
    # 单调日期保护：回溯日期（--date 早于最后记录）的运行只读状态、不写
    # 历史，彩排不会污染断路器。
    breaker_path = out.parent / BREAKER_FILE
    breaker = load_breaker(breaker_path)
    ret20 = np.nan
    breaker_flipped = None  # "OFF"=本次触发, "ON"=本次回场, None=状态延续
    _prev_picks = find_prev_picks_file(out.parent, target_date)
    if breaker["hist"] and target_date.strftime("%Y-%m-%d") <= breaker["hist"][-1]["date"]:
        print(f"断路器: 回溯运行（最后记录 {breaker['hist'][-1]['date']}），状态只读")
    elif _prev_picks is not None:
        # 当日（target_date）组合收益 = 昨日 picks 十只在"今日真实收盘 vs 今日
        # 昨收"上的等权收益（这些名字今日持仓：8 只延续 + 2 只今日开盘新进；
        # 开盘卖出的 2 只已离场，不计）。用本次抓取的实时行情计算，而不是
        # prev 文件里的 pct_chg 列——那一列是信号日（T-1）的涨跌幅，差一天。
        _pdf = pd.read_csv(_prev_picks[0])
        _rs = []
        for _c in _pdf["instrument"].astype(str).str.strip().str.upper():
            _p, _pc = prices.get(_c), prev_closes.get(_c)
            if _p is not None and _pc is not None and _pc > 0:
                _rs.append(_p / _pc - 1.0)
        if _rs:
            _r = float(np.mean(_rs))
            breaker["hist"].append({"date": target_date.strftime("%Y-%m-%d"), "r": round(_r, 6)})
            breaker["hist"] = breaker["hist"][-BREAKER_HIST_KEEP:]
    if len(breaker["hist"]) >= BREAKER_WINDOW:
        ret20 = float(sum(h["r"] for h in breaker["hist"][-BREAKER_WINDOW:]))
        if breaker["state"] == "ON" and ret20 <= BREAKER_KILL:
            breaker["state"] = "OFF"
            breaker_flipped = "OFF"
            print(f"BREAKER 触发: 20日滚动收益 {ret20:+.1%} ≤ {BREAKER_KILL:.0%} → 明日半仓执行")
        elif breaker["state"] == "OFF" and ret20 > BREAKER_RECOVER:
            breaker["state"] = "ON"
            breaker_flipped = "ON"
            print(f"BREAKER 回场: 20日滚动收益 {ret20:+.1%} > {BREAKER_RECOVER:.0%} → 明日恢复全仓")
    breaker_off = breaker["state"] == "OFF"
    if breaker != load_breaker(breaker_path):
        save_breaker(breaker_path, breaker)

    # 5. Official topk-dropout target portfolio. Under the de-risk rule the
    #    effective topk is halved AND no new names are added (risk-off day:
    #    sells only, zero extra turnover) — target = best-scored eff_topk of
    #    current holdings; the excess become sells, i.e. the position cut.
    eff_topk = args.topk // 2 if derisk else args.topk
    if derisk:
        held_scores = scores.reindex(list(holdings_for_select)).dropna()
        target = sorted(held_scores.index, key=lambda c: (held_scores[c], c), reverse=True)[:eff_topk]
        if len(target) < eff_topk:  # edge: holdings below eff_topk (e.g. ST force-sells)
            rest = [c for c in sorted(scores.index, key=lambda c: (scores[c], c), reverse=True) if c not in target]
            target += rest[: eff_topk - len(target)]
        target = sorted(target)
    else:
        target = select_official(scores, holdings_for_select, eff_topk, args.n_drop)
    buys = sorted((c for c in target if c not in prev_holdings), key=lambda c: (-scores.get(c, np.nan), c))
    sells = sorted(prev_holdings - set(target), key=lambda c: (-scores.get(c, -np.inf), c))
    holds = [c for c in target if c in prev_holdings]

    # 6. Output table with review-only info columns.
    info = load_info_columns(target, target_date)
    df = pd.DataFrame({"instrument": target})
    df["score"] = df["instrument"].map(scores)
    df["model_used"] = "blend50"
    df["risk_state"] = "HALF(de-risk)" if derisk else ("HALF(breaker)" if breaker_off else "FULL")
    df["action"] = ["HOLD" if c in prev_holdings else "BUY" for c in target]
    df["price"] = [prices.get(c, np.nan) for c in df["instrument"]]  # real quote, not normalized
    # 信号日真实涨跌幅（人工核对列；收盘后 price=当日收盘，prev_close=昨收）
    df["pct_chg"] = [
        (
            round(prices[c] / prev_closes[c] - 1.0, 4)
            if c in prices and c in prev_closes and prev_closes[c] > 0
            else np.nan
        )
        for c in df["instrument"]
    ]
    for col in ["eps_q", "niyoy_q", "sentiment_mean"]:
        if col in info.columns:
            df[col] = df["instrument"].map(info[col])
    df["name"] = [names.get(c, "") for c in df["instrument"]]
    df.insert(0, "datetime", target_date.strftime("%Y-%m-%d"))

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Persist signal history stores only after the picks CSV is safely written.
    persist_hist_store(hist_dir, "base", hist_base)
    persist_hist_store(hist_dir, "p0", hist_p0)

    # 7. Rebalance instructions (execute at next trading day's open).
    print(f"\n=== 换仓指令 ({target_date.strftime('%Y-%m-%d')} 信号, 次日开盘执行) ===")
    print(f"模型: blend50  (监控 ic_base={ic_b:.4f}, ic_p0={ic_p:.4f})")
    if derisk:
        print(
            f"*** 风控降仓中: 双信号 roll20 IC 连续 {DERISK_DAYS} 日为负, "
            f"目标仓位 50% ({eff_topk}/{args.topk} 只) — 建议等比例减半持仓 ***"
        )
        if breaker_off:
            print(f"*** 断路器同为半仓（20日收益 {ret20:+.1%}），维持上述降仓不动 ***")
    elif breaker_off:
        if breaker_flipped == "OFF":
            print(
                f"*** 断路器刚触发: 20日滚动收益 {ret20:+.1%} ≤ {BREAKER_KILL:.0%} — "
                f"明日开盘把全部持仓减半（现有持仓与当日换仓一并减半）***"
            )
            print(f"*** 之后换仓照常、新交易按半仓金额；回升 > {BREAKER_RECOVER:.0%} 自动回场 ***")
        else:
            print(
                f"*** 断路器半仓中（存量已减半）: 换仓照常、新交易按半仓金额；"
                f"20日滚动 {ret20:+.1%}，回升 > {BREAKER_RECOVER:.0%} 自动回场 ***"
            )
    elif breaker_flipped == "ON":
        print(
            f"*** 断路器回场: 20日滚动收益 {ret20:+.1%} > {BREAKER_RECOVER:.0%} — "
            f"明日恢复全仓（把减半部分加回），当日换仓按全额执行 ***"
        )
    else:
        if not np.isnan(ret20):
            print(
                f"仓位状态: 满仓 ({eff_topk} 只) | 断路器 ON（20日滚动 {ret20:+.1%}，触发 {BREAKER_KILL:.0%} / 回场 {BREAKER_RECOVER:.0%}）"
            )
        else:
            print(
                f"仓位状态: 满仓 ({eff_topk} 只) | 断路器 ON（数据积累 {len(breaker['hist'])}/{BREAKER_WINDOW}，暂不触发）"
            )
    if not prev_holdings:
        print(f"初始建仓 {len(target)} 只:")
        for c in target:
            print(f"  买入 {c} {names.get(c, '')} @ {prices.get(c, float('nan')):.2f}")
    else:
        if sells:
            print("卖出:")
            for c in sells:
                tag = " (ST强平)" if c in forced_sell else ""
                print(f"  {c} {names.get(c, '')} @ {prices.get(c, float('nan')):.2f}{tag}")
        if buys:
            print("买入:")
            for c in buys:
                print(f"  {c} {names.get(c, '')} @ {prices.get(c, float('nan')):.2f}")
        print(f"持有: {len(holds)} 只")
        if not sells and not buys:
            print("无需交易。")
        if buys and sells:
            print(
                "\n封板/高开跳过规则：若上面'买入'第 N 只开盘一字封板、停牌，或开盘"
                "涨幅超过 +9.5%（实盘限价单=收盘×1.095，本就不会成交），该笔放弃不追；"
                "同时'卖出'第 N 只保留不卖（买入/卖出两组均按得分从高到低排列，按序号"
                "一一对应）。当晚选股即自动排除该名字并还原持仓（保留的卖出者回列）。"
            )
    print(f"\nSaved picks to {out}")

    # 8. Execution cross-check (manual broker verification): previous picks'
    #    BUY names vs today's REAL open — 计划买入价 = 买入日开盘价. Quotes were
    #    already fetched for name_probe (prev holdings are included there).
    #    Phase 20 封板判定与 limit_skip 记录已前置到选股前（detect_sealed_buys），
    #    本段只输出核对表，不再写入状态。
    prev = find_prev_picks_file(out.parent, target_date)
    if prev is not None:
        prev_file, prev_date = prev
        prev_df = pd.read_csv(prev_file)
        prev_buys = prev_df[prev_df["action"] == "BUY"] if "action" in prev_df.columns else prev_df.iloc[0:0]
        if len(prev_buys):
            ec_rows = []
            for _, r in prev_buys.iterrows():
                c = str(r["instrument"]).strip().upper()
                sig_close = float(r["price"]) if pd.notna(r.get("price")) else np.nan
                o, p, pc = opens.get(c), prices.get(c), prev_closes.get(c)
                ec_rows.append(
                    dict(
                        instrument=c,
                        name=names.get(c, ""),
                        signal_date=prev_date.strftime("%Y-%m-%d"),
                        signal_close=sig_close,
                        exec_date=target_date.strftime("%Y-%m-%d"),
                        open=o,  # 计划买入价 = 买入日开盘价
                        open_gap=(
                            round(o / sig_close - 1.0, 4) if o and pd.notna(sig_close) and sig_close > 0 else np.nan
                        ),
                        close=p,
                        pct_chg=round(p / pc - 1.0, 4) if p and pc else np.nan,
                    )
                )
            ec = pd.DataFrame(ec_rows)
            ec_path = out.parent / f"exec_check_{target_date.strftime('%Y%m%d')}.csv"
            ec.to_csv(ec_path, index=False)
            print(f"\n=== 成交核对（{prev_date.strftime('%Y-%m-%d')} 信号买入 → 今日实际开盘）===")
            print(ec.to_string(index=False))
            print(f"Saved exec check to {ec_path}")

    if state_live and limit_skip != load_limit_skip(limit_skip_path):
        save_limit_skip(limit_skip_path, limit_skip)


if __name__ == "__main__":
    main()
