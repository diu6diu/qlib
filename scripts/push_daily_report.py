#!/usr/bin/env python3
"""Push the daily 3-strategy summary to WeChat via WxPusher.

Usage:
    push_daily_report.py [--date YYYY-MM-DD] [--status N]
        Build the daily report from that day's artifacts and push it.
    push_daily_report.py --msg "text"
        Push raw text (used by run_daily_all.sh for holiday/failure notices).

Credentials are read from ~/.config/qlib/wxpusher.env (WXPUSHER_APP_TOKEN,
WXPUSHER_UID). Missing artifacts degrade to a "not found" line per strategy,
so a half-failed pipeline still produces a useful message.

Exit code: 0 on successful push, 1 otherwise (so the caller can log it).
"""

import argparse
import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path.home() / ".config" / "qlib" / "wxpusher.env"
API_URL = "https://wxpusher.zjiecode.com/api/send/message"

WEEKDAYS_CN = "一二三四五六日"


def load_env() -> dict:
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def push(content: str, summary: str) -> bool:
    env = load_env()
    payload = {
        "appToken": env["WXPUSHER_APP_TOKEN"],
        "uids": [env["WXPUSHER_UID"]],
        "contentType": 1,  # plain text
        "summary": summary[:100],
        "content": content,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    ok = body.get("code") == 1000
    if not ok:
        print(f"wxpusher error: {body}", file=sys.stderr)
    return ok


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _pct(x: float) -> str:
    return f"{x:+.1%}"


# ---------------------------------------------------------------- production


def production_section(date: str) -> list:
    compact = date.replace("-", "")
    base = ROOT / "examples/benchmarks_dynamic/mainboard_leftside/results/mainboard_leftside"
    lines = ["【生产 · 主板左腿】"]

    picks_path = base / f"picks_live_{compact}.csv"
    if not picks_path.exists():
        return lines + ["当日 picks 未生成"]

    rows = list(csv.DictReader(picks_path.open()))
    buys = [r.get("name") or r["instrument"] for r in rows if r.get("action", "").upper() == "BUY"]
    lines.append(f"picks {len(rows)} 只" + (f" | 买入 {len(buys)}: {', '.join(buys)}" if buys else ""))

    # 仓位状态（Phase 19 IC 降仓 + Phase 21 自回撤断路器）：picks 的
    # risk_state 列是当日执行口径，breaker.json 补 20 日滚动读数。
    risk_states = {r.get("risk_state", "") for r in rows}
    breaker_txt = None
    try:
        state = json.loads((base / "breaker.json").read_text())
        hist = state.get("hist", [])[-20:]
        ret20 = sum(h["r"] for h in hist) if len(hist) == 20 else None
        ret20_txt = f"{ret20:+.1%}" if ret20 is not None else "积累中"
    except Exception:  # noqa: BLE001 — informational only
        ret20, ret20_txt = None, None
    if any("breaker" in s for s in risk_states):
        try:
            _kill, _recover = _breaker_thresholds()
        except Exception:  # noqa: BLE001
            _kill, _recover = -0.03, -0.015
        breaker_txt = f"⚠️ 断路器半仓执行中（20日滚动 {ret20_txt} ≤ {_kill:.0%} 触发，回升 > {_recover:.1%} 自动回场）"
    elif any("de-risk" in s for s in risk_states):
        breaker_txt = f"⚠️ IC 连续为负降仓半仓（断路器 20日滚动 {ret20_txt}）"
    elif ret20 is not None:
        breaker_txt = f"断路器全仓（20日滚动 {ret20_txt}，触发线 -3%）"
    if breaker_txt:
        lines.append(breaker_txt)

    # The picks CSV only lists the target portfolio (BUY/HOLD); sells are
    # implicit — yesterday's picks that dropped out of today's target
    # (generate_live_picks.py prints them to the log but never writes them).
    # Match the exact dated pattern: string-sorting the raw glob would rank
    # ad-hoc backups like picks_live_20261015.pregtja120.csv AFTER the real
    # picks_live_20261015.csv and diff against the wrong baseline
    # (2026-10-17 日报 showed 卖出4 instead of 卖出2 because of this).
    prev_files = sorted(
        p
        for p in base.glob("picks_live_*.csv")
        if re.fullmatch(r"picks_live_\d{8}\.csv", p.name) and p.name < picks_path.name
    )
    if prev_files:
        prev_rows = list(csv.DictReader(prev_files[-1].open()))
        held_today = {r["instrument"] for r in rows}
        sells = [r.get("name") or r["instrument"] for r in prev_rows if r["instrument"] not in held_today]
        if sells:
            lines.append(f"卖出 {len(sells)}: {', '.join(sells)}")
        if not buys and not sells:
            lines.append("无需交易（持仓不变）")

    moves = []
    for r in rows:
        try:
            moves.append((float(r["pct_chg"]), r.get("name") or r["instrument"]))
        except (KeyError, ValueError):
            pass
    if moves:
        moves.sort()
        lines.append("信号日最强: " + " / ".join(f"{n} {_pct(p)}" for p, n in moves[-3:][::-1]))
        lines.append("信号日最弱: " + " / ".join(f"{n} {_pct(p)}" for p, n in moves[:3]))

    # yesterday's signals -> today's execution quality
    exec_path = base / f"exec_check_{compact}.csv"
    if exec_path.exists():
        ex = list(csv.DictReader(exec_path.open()))
        if ex:
            lines.append(f"昨日信号今日执行 {len(ex)} 只:")
            for r in ex:
                try:
                    gap, chg = float(r["open_gap"]), float(r["pct_chg"])
                    lines.append(f"  {r['name']} 开{_pct(gap)} → 收{_pct(chg)}")
                except (KeyError, ValueError):
                    pass
    return lines


# ------------------------------------------------------------- G right-side


def g_section(date: str) -> list:
    text = _read(ROOT / f"experiments/rightside_trend/results/paper/report_{date}.md")
    lines = ["【G右侧 · 模拟盘】"]
    if not text:
        return lines + ["当日日报未生成"]

    m = re.search(r"^净值 .*$", text, re.M)
    if m:
        lines.append(m.group(0))

    trades = re.findall(r"^- (买 .+|卖 .+)$", text.split("## 可成交性核对")[0], re.M)
    if trades:
        lines.append("今日成交:")
        lines += ["  " + t for t in trades]

    # holdings table -> best/worst unrealized
    m = re.search(r"## 当前持仓\n\S.*\n((?:\s+\S.*\n?)+)", text)
    if m:
        holds = []
        for ln in m.group(1).splitlines():
            f = ln.split()
            if len(f) >= 9:
                try:
                    holds.append((float(f[-2]), f[1], int(f[-1])))
                except ValueError:
                    pass
        if holds:
            holds.sort()
            lines.append("持仓浮盈: " + " / ".join(f"{n} {_pct(u)}({d}天)" for u, n, d in holds[-3:][::-1]))
            lines.append("持仓浮亏: " + " / ".join(f"{n} {_pct(u)}({d}天)" for u, n, d in holds[:3]))

    m = re.search(r"## 明日待执行订单（T\+1 开盘）\n((?:- .+\n?)+)", text)
    if m:
        orders = [ln[2:] for ln in m.group(1).splitlines() if ln.startswith("- ")]
        lines.append("明日订单: " + ("; ".join(orders) if orders else "无"))
    return lines


# --------------------------------------------------------- strategy health

IC_HISTORY = ROOT / "experiments/alpha_enhancement" / "results" / "ic_history.csv"
LIVE_PICKS_SRC = ROOT / "examples/benchmarks_dynamic" / "mainboard_leftside" / "generate_live_picks.py"


def _breaker_thresholds() -> tuple[float, float]:
    """从 generate_live_picks.py 源码读 BREAKER_KILL / BREAKER_RECOVER，
    避免日报口径与生产脚本漂移。"""
    src = LIVE_PICKS_SRC.read_text()
    kill = float(re.search(r"BREAKER_KILL\s*=\s*(-?[\d.]+)", src).group(1))
    recover = float(re.search(r"BREAKER_RECOVER\s*=\s*(-?[\d.]+)", src).group(1))
    return kill, recover


def _ic_band(v: float, hist) -> str:
    """IC 健康分档（2026-10-23 定稿，机构线 + 自身分位双锚）：
    机构线（券商金工通行标准，调仓周期口径 rank IC；本策略为 5 日前瞻
    日 IC 的 20 日均值，均值不变故门槛可直接套用）——≥0.05 较强、
    0.03~0.05 有效、0~0.03 偏弱（低于"有效因子"线）、<0 失效。
    自身线——历史分位（2023 至今 535 个有效读数，P25≈0.03 与机构
    有效线巧合互证）。"""
    pct = (hist.dropna() < v).mean()
    if v < 0:
        band = "失效"
    elif v < 0.03:
        band = "偏弱"
    elif v < 0.05:
        band = "有效"
    else:
        band = "较强"
    return f"{v:+.3f} {band}（历史 {pct:.0%} 分位）"


def health_section(date: str) -> list:
    """策略健康度：双模型 IC（含历史分位分档）、IC 双负连击、断路器
    回场倒计时（机械口径：假设未来走平，旧大跌日掉出 20 日窗口的日数；
    实际反弹会提前）。数据来自 step 1.7 刷新的 ic_history.csv 与
    breaker.json。纯信息区块，任何异常都不阻断推送。"""
    import pandas as pd

    lines = ["【策略健康度】"]
    try:
        ic = pd.read_csv(IC_HISTORY, parse_dates=["datetime"])
        last = ic.iloc[-1]
        lines.append(
            f"模型IC(20日): base {_ic_band(last['ic0'], ic['ic0'])} | "
            f"p0 {_ic_band(last['ic1'], ic['ic1'])}（截至 {last['datetime'].date()}）"
        )
        # 机构三件套之稳定性两项：近 20 日 ICIR 与胜率（日 IC 列 ic0_d/ic1_d）。
        # 机构线：ICIR(日) >0.5 合格 / >1.0 优秀；胜率 >55% 较好 / >60% 优秀。
        if "ic0_d" in ic.columns:
            w0, w1 = ic["ic0_d"].dropna().iloc[-20:], ic["ic1_d"].dropna().iloc[-20:]

            def _icir(s):
                return s.mean() / s.std() if len(s) > 5 and s.std() > 0 else float("nan")

            lines.append(
                f"近20日稳定性: base ICIR {_icir(w0):.2f}/胜率 {(w0 > 0).mean():.0%} | "
                f"p0 ICIR {_icir(w1):.2f}/胜率 {(w1 > 0).mean():.0%}（ICIR>0.5 合格，胜率>55% 较好）"
            )
        streak = 0
        for v in ic["both_neg"].iloc[::-1]:
            if not v:
                break
            streak += 1
        lines.append(f"IC 双负连击: {streak}/10 天（满 10 天自动降半仓）")
    except Exception as e:  # noqa: BLE001 — informational only
        lines.append(f"IC 数据不可用: {type(e).__name__}")

    try:
        kill, recover = _breaker_thresholds()
        base = ROOT / "examples/benchmarks_dynamic/mainboard_leftside/results/mainboard_leftside"
        state = json.loads((base / "breaker.json").read_text())
        hist = state.get("hist", [])[-20:]
        if len(hist) < 20:
            lines.append("断路器: 历史积累中（不足 20 天）")
        else:
            ret20 = sum(h["r"] for h in hist)
            if state.get("state") == "OFF":
                n, v = 0, ret20
                while v <= recover and n < 20:  # 走平情形：逐日掉出旧窗口
                    v -= hist[n]["r"]
                    n += 1
                eta = f"即使完全走平，第 {n} 个交易日自然回场（反弹会提前，新增大跌会延后）" if n < 20 else "走平情形下 20 日内无法回场"
                lines.append(f"断路器半仓中: ret20 {ret20:+.1%}，回场线 >{recover:.1%}；{eta}")
            else:
                lines.append(f"断路器全仓: ret20 {ret20:+.1%}（触发线 {kill:.0%}）")
    except Exception as e:  # noqa: BLE001 — informational only
        lines.append(f"断路器数据不可用: {type(e).__name__}")
    return lines


# ------------------------------------------------------------ market regime

STYLE_CACHE = ROOT / "experiments" / "rightside_trend" / "results" / "cache" / "style_indices.pkl"
MARKET_PAIRS = [("SH000905", "中证500"), ("SZ399303", "国证2000")]


def market_section(date: str) -> list:
    """大盘位置（指数层择时观察，不驱动任何交易）：ret20 的 z 分数测抄底
    （月度尺度均值回归：z≤-1.5 后 20 日 中证500 +6.1%/95%、国证2000
    +9.5%/100%，2026-10-23 窗口扫描验证，W=10~40 为平坦高原）；价格相对
    年线的偏离 z 测偏热（z≥+1.5 后 20 日三指数全负，但触发样本少，仅
    提示）。抄底区分绿灯/黄灯（2026-10-24 全历史复核，FINDINGS.md
    "大盘 ret20 抄底信号的全历史复核"）：信号日价在年线下方=绿灯
    （2007 起 15 次，fwd60 均值+10%/胜率80%/最差-10%），年线上方=
    黄灯（4 次，胜率25%，含 2008-02/2015-06 高位首崩，z 修复是减仓窗
    而非见底）。上下不对称——ret20 过热不是顶部信号（动量延续），勿镜像。
    z 用缓存全历史（2019-12 起）的均值/标准差。数据 = style_indices.pkl，
    step 3 每日刷新（mootdx 死区时回退缓存，截至日期随之滞后）。
    选国证2000 而非沪深300：策略真实贝塔在小盘（ret20 相关 0.76 vs
    0.57），且小盘均值回归最强。纯信息区块，任何异常不阻断推送。"""
    lines = ["【大盘位置】"]
    try:
        import pandas as pd

        cache = pd.read_pickle(STYLE_CACHE)
        shown, last_dt = 0, None
        for code, name in MARKET_PAIRS:
            s = cache[code]["close"].dropna()
            r20 = s.pct_change(20)
            z = (r20 - r20.mean()) / r20.std()
            dev = s / s.rolling(250).mean() - 1
            zd = (dev - dev.mean()) / dev.std()
            lz, ld = z.iloc[-1], zd.iloc[-1]
            last_dt = s.index[-1]
            if lz <= -1.5:
                vs_ma = s.iloc[-1] / s.rolling(250).mean().iloc[-1] - 1
                if vs_ma < 0:
                    light = f" 🟢绿灯抄底区(年线下{vs_ma:+.0%}，全历史80%胜率)"
                else:
                    light = f" 🟡黄灯(年线上{vs_ma:+.0%}，防高位首崩，z修复即减仓窗)"
            elif ld >= 1.5:
                light = " 🔴偏热提示"
            else:
                light = ""
            lines.append(f"{name}: ret20 z={lz:+.1f} | 年线偏离 z={ld:+.1f}{light}")
            shown += 1
        if not shown:
            lines.append("指数数据为空")
        else:
            lines.append(f"（抄底=ret20 z≤-1.5，绿灯=价在年线下方/黄灯=年线上方慎入，偏热=年线偏离 z≥+1.5 仅提示；截至 {last_dt.date()}）")
            lines += _dip_event_tracking(cache)
    except Exception as e:  # noqa: BLE001 — informational section must never break the push
        lines.append(f"指数数据不可用: {type(e).__name__}")
    return lines


def _dip_event_tracking(cache) -> list:
    """中证500 下穿事件跟踪：距下穿天数 / z 当前值 vs 绿灯组同日中位 /
    信号日买入浮盈 / 红线（z<-2.7 破 2022-04 极值；浮亏<-10% 破绿灯组
    历史最差）。事件进行中（最近一次下穿 120 个交易日内且 z 尚未修复到
    -0.5 上方）才显示。依据：FINDINGS.md "大盘 ret20 抄底信号的全历史
    复核"（2026-10-24）。绿灯组中位路径用缓存（2019-12 起）内可判定
    年线的历史下穿事件动态计算。"""
    import numpy as np
    import pandas as pd  # noqa: F401

    s = cache["SH000905"]["close"].dropna()
    r20 = s.pct_change(20)
    z = (r20 - r20.mean()) / r20.std()
    ma = s.rolling(250).mean()
    cross = (z <= -1.5) & (z.shift(1) > -1.5)
    evs = list(z.index[cross.fillna(False)])
    if not evs or z.iloc[-1] > -0.5:
        return []
    t0 = evs[-1]
    i0 = s.index.get_loc(t0)
    n = len(s) - 1 - i0
    if n > 120:
        return []
    # 绿灯组（下穿日价在年线下方）历史事件的 z 路径，用于同日中位对比
    paths: dict = {}
    for t in evs[:-1]:
        i = s.index.get_loc(t)
        if np.isnan(ma.iloc[i]) or s.iloc[i] >= ma.iloc[i]:
            continue
        for d, v in enumerate(z.iloc[i : i + 61].values):
            if not np.isnan(v):
                paths.setdefault(d, []).append(v)
    med_z = float(np.median(paths[n])) if n in paths else np.nan
    pnl = s.iloc[-1] / s.iloc[i0] - 1
    red = []
    if z.iloc[-1] < -2.7:
        red.append("z破-2.7")
    if pnl < -0.10:
        red.append("浮亏破-10%")
    red_txt = "⚠️" + "、".join(red) if red else "未触及"
    med_txt = f"{med_z:+.2f}" if not np.isnan(med_z) else "无参照"
    return [
        f"事件跟踪: 下穿 {t0.date()} 起第 {n} 个交易日，z={z.iloc[-1]:+.2f}"
        f"（绿灯组同日中位 {med_txt}），信号日买入浮盈 {pnl:+.1%}；红线: {red_txt}"
    ]


# ------------------------------------------------------------- futures basis

BASIS_SRC = Path.home() / ".qlib" / "qlib_data" / "source"
BASIS_PAIRS = [("IC00", "SH000905"), ("IM00", "SH000852")]


def basis_section(date: str) -> list:
    """CFFEX basis watch (IC/IM dominant vs spot). Informational only — the
    L3 gate (IC deep-and-deepening discount -> half exposure) is NOT wired
    into any trading logic; this section just reports its state."""
    import pandas as pd

    lines = ["【基差 · 聪明钱(仅观察)】"]
    try:
        shown = 0
        for fut, spot in BASIS_PAIRS:
            f = pd.read_csv(BASIS_SRC / "cffex_sina_normalize" / f"{fut}.csv", parse_dates=["date"])
            s = pd.read_csv(BASIS_SRC / "mootdx_indices_normalize" / f"{spot}.csv", parse_dates=["date"])
            df = f.merge(s, on="date", suffixes=("_f", "_s"))
            if df.empty:
                continue
            df["b"] = df["close_f"] / df["close_s"] - 1.0
            df["z20"] = (df["b"] - df["b"].rolling(20).mean()) / df["b"].rolling(20).std()
            df["d5"] = df["b"] - df["b"].shift(5)
            last = df.dropna(subset=["z20"]).iloc[-1]
            deep = bool(last["z20"] < -1.5 and last["d5"] < 0)
            state = "深贴水加深(L3 若启用将降半仓)" if deep else "正常区间"
            lines.append(
                f"{fut[:2]}: 基差 {last['b']:+.2%} | z20 {last['z20']:+.2f} | "
                f"5日变动 {last['d5']:+.2%} → {state}（截至 {last['date'].date()}）"
            )
            shown += 1
        if not shown:
            lines.append("基差数据为空")
    except Exception as e:  # noqa: BLE001 — informational section must never break the push
        lines.append(f"基差数据不可用: {type(e).__name__}")
    return lines


# ---------------------------------------------------------------------- t3a


def t3a_section(date: str) -> list:
    """Only the TF (index-ETF) leg: left leg = production, right leg = G,
    both already have their own sections."""
    text = _read(ROOT / f"experiments/style_router/results/paper_t3a/report_{date}.md")
    lines = ["【t3a · TF指数腿】"]
    if not text:
        return lines + ["当日日报未生成"]

    m = re.search(r"^- 状态：\*\*(ON|OFF)\*\*(.*)$", text, re.M)
    if not m:
        return lines + ["日报格式异常"]
    lines.append(f"状态: {m.group(1)}")
    mf = re.search(r"快闸（[^）]*）：([+-][\d.]+%)（([^）]*)）", text)
    if mf:
        lines.append(f"快闸: {mf.group(1)}（{mf.group(2)}）")
    ms = re.search(r"慢闸诊断（[^）]*）：([+-][\d.]+)", text)
    if ms:
        lines.append(f"慢闸(仅监测): {ms.group(1)}")

    # target weights, only when the leg is on (skip the all-zero table)
    if m.group(1) == "ON":
        mw = re.search(r"## TF 腿次日目标权重[^\n]*\n\S.*\n((?:\S.*\n?)+?)\n", text)
        if mw:
            active = [
                ln for ln in mw.group(1).splitlines() if ln.strip() and not ln.rstrip().endswith(("0.0", "0.0    0.0"))
            ]
            if active:
                lines.append("次日目标权重:")
                lines += ["  " + ln.strip() for ln in active]
    return lines


# ------------------------------------------------------- off-market fund DCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import dca_dividend_signal

    def dca_section(_date: str) -> list:
        """场外定投信号 (红利低波/银行 股息率分位 -> 正常/加倍/暂停)."""
        try:
            return dca_dividend_signal.build_section()
        except Exception as e:  # noqa: BLE001 — informational section must never break the push
            return ["【定投信号 · 红利低波/银行】", f"股息率数据不可用: {type(e).__name__}"]

except ImportError:  # pragma: no cover - script missing; degrade gracefully

    def dca_section(_date: str) -> list:
        return ["【定投信号 · 红利低波/银行】", "信号模块未安装 (dca_dividend_signal.py)"]


def build_report(date: str, status: int) -> str:
    d = dt.date.fromisoformat(date)
    head = f"📊 量化日报 {d.month}/{d.day} 周{WEEKDAYS_CN[d.weekday()]}"
    parts = [head]
    if status != 0:
        parts.append(f"⚠️ 流水线状态异常(status={status})，详见 results/daily_logs/daily_{date}.log")
    for section in (production_section, health_section, market_section, g_section, t3a_section, basis_section, dca_section):
        parts.append("")
        parts += section(date)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--status", type=int, default=0, help="pipeline exit status")
    ap.add_argument("--msg", help="send raw text instead of the daily report")
    args = ap.parse_args()

    if args.msg:
        content, summary = args.msg, args.msg.splitlines()[0][:100]
    else:
        content = build_report(args.date, args.status)
        summary = f"量化日报 {args.date}" + (" ⚠️" if args.status != 0 else "")

    if push(content, summary):
        print(f"pushed daily report for {args.date}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
