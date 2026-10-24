# -*- coding: utf-8 -*-
"""中证500 (sh000905) 全历史 ret20 z<=-1.5 事件分析（腾讯通道）。

回答两个问题:
1) 信号后第几个交易日入场收益最大? (入场偏移 d=0..15 的 fwd20/40/60)
2) 股灾/大熊市 (2008, 2011, 2015-16, 2018) 里这个规律是否成立?
"""
import sys
import time

import numpy as np
import pandas as pd
import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
_BASES = [
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get",
]


def fetch_tencent_index(tcode: str = "sh000905") -> pd.Series:
    """向后翻页拉全历史: 每次取 end 之前最多 500 根日线。"""
    frames = []
    end = "2026-10-24"
    for _ in range(30):
        got = None
        for base in _BASES:
            url = f"{base}?param={tcode},day,2005-01-01,{end},500,"
            try:
                r = requests.get(url, timeout=15, headers=_HEADERS)
                klines = r.json().get("data", {}).get(tcode, {}).get("day")
                if klines:
                    got = pd.DataFrame([p[:3] for p in klines], columns=["date", "open", "close"])
                    break
            except Exception:
                continue
        if got is None or got.empty:
            break
        frames.append(got)
        earliest = got["date"].min()
        if earliest <= "2007-01-15":
            break
        end = (pd.Timestamp(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.3)
    df = pd.concat(frames).drop_duplicates("date").sort_values("date")
    s = pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["date"]))
    return s


idx = fetch_tencent_index()
print(f"数据 {idx.index[0].date()} ~ {idx.index[-1].date()}, {len(idx)} 天\n")

r20 = idx.pct_change(20)
z = (r20 - r20.mean()) / r20.std()

cross = (z <= -1.5) & (z.shift(1) > -1.5)
events = list(z.index[cross.fillna(False)])
print(f"下穿事件共 {len(events)} 次\n")

OFFSETS = [0, 1, 2, 3, 5, 7, 10, 15]

rows = []
for t0 in events:
    i0 = idx.index.get_loc(t0)
    row = {"date": t0.date(), "year": t0.year}
    for d in OFFSETS:
        i = i0 + d
        row[f"e{d}_fwd20"] = (idx.iloc[i + 20] / idx.iloc[i] - 1) if i + 20 < len(idx) else np.nan
        row[f"e{d}_fwd60"] = (idx.iloc[i + 60] / idx.iloc[i] - 1) if i + 60 < len(idx) else np.nan
    # 后视镜最优: 信号后 15 个交易日内最低收盘价买入
    win = idx.iloc[i0 : i0 + 16]
    row["best60"] = (idx.iloc[i0 + 60] / win.min() - 1) if i0 + 60 < len(idx) else np.nan
    # 信号日入场 60 日路径最大浮亏
    path = idx.iloc[i0 : i0 + 61] / idx.iloc[i0] - 1
    row["mdd60"] = path.min() if i0 + 60 < len(idx) else np.nan
    rows.append(row)

df = pd.DataFrame(rows).set_index("date")

print("=== 各入场偏移的 fwd60 汇总 (均值/中位/胜率) ===")
for d in OFFSETS:
    c = df[f"e{d}_fwd60"].dropna()
    print(f"信号+{d:>2}日: 均值 {c.mean():+.2%} 中位 {c.median():+.2%} 胜率 {(c>0).mean():.0%} (n={len(c)})")
b = df["best60"].dropna()
print(f"后视镜最低点: 均值 {b.mean():+.2%} 中位 {b.median():+.2%}")

print("\n=== 股灾/熊市年份事件明细 (信号日入场) ===")
bear = df[df["year"].isin([2008, 2010, 2011, 2012, 2015, 2016, 2018, 2022, 2024])]
cols = ["e0_fwd20", "e0_fwd60", "e5_fwd60", "e10_fwd60", "mdd60", "best60"]
print(bear[cols].round(3).to_string())

print("\n=== 全部事件明细 ===")
print(df[["year", "e0_fwd60", "e5_fwd60", "e10_fwd60", "mdd60"]].round(3).to_string())
