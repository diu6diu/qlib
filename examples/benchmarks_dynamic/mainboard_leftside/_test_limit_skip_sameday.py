"""Phase 20+ 封板判定前置修复（2026-10-23）的合成测试。

覆盖：
1. 开盘缺口 >+9.5% → 记录冷静名单，配对保留者按得分序正确还原；
2. 正常开盘 → 不记录；
3. 停牌（无开盘价）→ 记录；
4. 逐名 sanity：prev_close 与信号日收盘偏离 ≥2%（错配行情）→ 拒绝记录；
5. 幂等：已在名单里的名字不重复记录；
6. 持仓还原：新条目一次性加回 kept；存量条目不重复加回（防幽灵复活，
   2026-10-17 晋拓股份事故同类）。

运行: .venv/bin/python _test_limit_skip_sameday.py   （在本目录下）
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_live_picks import apply_limit_skip_correction, detect_sealed_buys  # noqa: E402

PREV_DATE = pd.Timestamp("2026-10-21")


def _prev_df():
    # 昨日 picks：8 HOLD + 2 BUY（三峡新材高分、京基智农低分）
    return pd.DataFrame(
        {
            "instrument": ["SH600001", "SH600002", "SH600003", "SH600004", "SH600005",
                           "SH600006", "SH600007", "SH600008", "SH600293", "SZ000048"],
            "action": ["HOLD"] * 8 + ["BUY", "BUY"],
            "score": [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.2, 2.1, 13.69, 2.27],
            "price": [10.0] * 8 + [4.32, 14.85],
        }
    )


PREV_SELLS = ["SZ002384", "SZ003036"]  # 昨日卖出名单（泰坦股份、东山精密）
NAMES = {"SH600293": "三峡新材", "SZ000048": "京基智农", "SH600000": "浦发银行"}


def t1_gap_sealed_recorded_with_pairing():
    limit_skip = {}
    opens = {"SH600293": 4.75, "SZ000048": 14.95}  # 三峡 +10.0% 封板；京基 +0.7% 正常
    prev_closes = {"SH600293": 4.32, "SZ000048": 14.85}  # = 昨日信号收盘
    newly = detect_sealed_buys(_prev_df(), PREV_DATE, PREV_SELLS, opens, prev_closes, NAMES, limit_skip)
    assert newly == ["SH600293"], newly
    assert limit_skip == {"SH600293": {"kept": "SZ002384", "since": "2026-10-21"}}, limit_skip
    print("PASS t1 缺口封板记录 + 配对还原")


def t2_normal_open_not_recorded():
    limit_skip = {}
    opens = {"SH600293": 4.36, "SZ000048": 14.95}  # +0.9% 正常
    prev_closes = {"SH600293": 4.32, "SZ000048": 14.85}
    newly = detect_sealed_buys(_prev_df(), PREV_DATE, PREV_SELLS, opens, prev_closes, NAMES, limit_skip)
    assert newly == [] and limit_skip == {}, (newly, limit_skip)
    print("PASS t2 正常开盘不记录")


def t3_suspended_recorded():
    limit_skip = {}
    opens = {"SH600293": None, "SZ000048": 14.95}  # 停牌无开盘价
    prev_closes = {"SH600293": None, "SZ000048": 14.85}
    newly = detect_sealed_buys(_prev_df(), PREV_DATE, PREV_SELLS, opens, prev_closes, NAMES, limit_skip)
    assert newly == ["SH600293"] and limit_skip["SH600293"]["kept"] == "SZ002384", (newly, limit_skip)
    print("PASS t3 停牌按买不进记录")


def t4_sanity_refuses_mismatched_quotes():
    limit_skip = {}
    opens = {"SH600293": 4.75, "SZ000048": 14.95}
    prev_closes = {"SH600293": 4.10, "SZ000048": 14.85}  # 与信号收盘 4.32 偏离 5%：错配行情
    newly = detect_sealed_buys(_prev_df(), PREV_DATE, PREV_SELLS, opens, prev_closes, NAMES, limit_skip)
    assert newly == [] and limit_skip == {}, (newly, limit_skip)
    print("PASS t4 错配行情拒绝记录（sanity）")


def t5_idempotent_no_rerecord():
    limit_skip = {"SH600293": {"kept": "SZ002384", "since": "2026-10-20"}}
    opens = {"SH600293": 4.75, "SZ000048": 14.95}
    prev_closes = {"SH600293": 4.32, "SZ000048": 14.85}
    newly = detect_sealed_buys(_prev_df(), PREV_DATE, PREV_SELLS, opens, prev_closes, NAMES, limit_skip)
    assert newly == [] and limit_skip["SH600293"]["since"] == "2026-10-20", (newly, limit_skip)
    print("PASS t5 已记录名字幂等不重复")


def t6_correction_new_entry_restores_kept():
    holdings = {"SH600001", "SH600002", "SH600003", "SH600004", "SH600005",
                "SH600006", "SH600007", "SH600008", "SH600293", "SZ000048"}
    limit_skip = {"SH600293": {"kept": "SZ002384", "since": "2026-10-21"}}
    out = apply_limit_skip_correction(holdings, limit_skip, ["SH600293"])
    assert "SH600293" not in out and "SZ002384" in out and len(out) == 10, out
    print("PASS t6 新条目：封板名移出 + 配对保留者回列")


def t7_correction_old_entry_no_ghost_resurrection():
    # 存量条目：kept（东山）已在后续选股中被正常轮出 → 不得重复加回
    holdings = {"SH600001", "SH600002", "SH600003", "SH600004", "SH600005",
                "SH600006", "SH600007", "SH600008", "SZ000048", "SZ000049"}
    limit_skip = {"SH600293": {"kept": "SZ002384", "since": "2026-10-21"}}
    out = apply_limit_skip_correction(holdings, limit_skip, [])
    assert "SZ002384" not in out and "SH600293" not in out and len(out) == 10, out
    print("PASS t7 存量条目：不幽灵复活已轮出的 kept")


def t8_kept_none_not_added():
    holdings = {"SH600293", "SZ000048"}
    limit_skip = {"SH600293": {"kept": None, "since": "2026-10-21"}}
    out = apply_limit_skip_correction(holdings, limit_skip, ["SH600293"])
    assert out == {"SZ000048"}, out
    print("PASS t8 空坑（无配对卖出）不加回")


if __name__ == "__main__":
    t1_gap_sealed_recorded_with_pairing()
    t2_normal_open_not_recorded()
    t3_suspended_recorded()
    t4_sanity_refuses_mismatched_quotes()
    t5_idempotent_no_rerecord()
    t6_correction_new_entry_restores_kept()
    t7_correction_old_entry_no_ghost_resurrection()
    t8_kept_none_not_added()
    print("\nALL 8 PASS")
