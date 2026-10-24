# -*- coding: utf-8 -*-
"""中证500 ret20 低位(z<=-1.5)事件的均值回归时长 + 一次性 vs 分批建仓对比。

样本: 2019-12 ~ 2026-10, 数据来自 experiments/rightside_trend/results/cache/style_indices.pkl
z = (r20 - mean(r20)) / std(r20) 全历史口径
事件: z 下穿 -1.5 的首日
统计: 回归到 z>=-0.5 的交易日数; fwd 5/10/20/40/60 收益; 信号后60日最大浮亏;
      一次性买入 vs 分4批(信号日/+5/+10/+15交易日各1/4)的60日收益对比
"""
import numpy as np
import pandas as pd

CACHE = "experiments/rightside_trend/results/cache/style_indices.pkl"

idx = pd.read_pickle(CACHE)["SH000905"]["close"].dropna()
r20 = idx.pct_change(20)
z = (r20 - r20.mean()) / r20.std()

# 事件: z 下穿 -1.5 首日
cross = (z <= -1.5) & (z.shift(1) > -1.5)
events = z.index[cross.fillna(False)]
print(f"样本 {idx.index[0].date()} ~ {idx.index[-1].date()}, 下穿事件 {len(events)} 次\n")

rows = []
for t0 in events:
    i0 = idx.index.get_loc(t0)
    # 回归到 z >= -0.5
    zi = z.iloc[i0:]
    rec = zi[zi >= -0.5]
    days_rec = idx.index.get_loc(rec.index[0]) - i0 if len(rec) else None
    # 前瞻收益
    fw = {}
    for h in (5, 10, 20, 40, 60):
        fw[h] = (idx.iloc[i0 + h] / idx.iloc[i0] - 1) if i0 + h < len(idx) else np.nan
    # 信号后 60 日最大浮亏 (相对信号日收盘)
    path = idx.iloc[i0 : i0 + 61] / idx.iloc[i0] - 1
    mdd60 = path.min() if i0 + 60 < len(idx) else np.nan
    # 分 4 批: 信号日/+5/+10/+15 交易日各 1/4, 持有到 i0+60
    if i0 + 60 < len(idx):
        buys = [idx.iloc[i0 + d] for d in (0, 5, 10, 15)]
        dca60 = idx.iloc[i0 + 60] / np.mean(buys) - 1
    else:
        dca60 = np.nan
    rows.append(
        dict(date=t0.date(), days_rec=days_rec, mdd60=mdd60, dca60=dca60, **{f"fwd{h}": fw[h] for h in fw})
    )

df = pd.DataFrame(rows).set_index("date")
pd.set_option("display.width", 200)
print(df.round(4).to_string())

print("\n=== 回归时长(交易日, 到 z>=-0.5) ===")
rec = df["days_rec"].dropna()
print(f"中位数 {rec.median():.0f} | 均值 {rec.mean():.1f} | p25 {rec.quantile(.25):.0f} | p75 {rec.quantile(.75):.0f} | 最大 {rec.max():.0f} | 未回归事件 {df['days_rec'].isna().sum()}")

print("\n=== 前瞻收益汇总 ===")
print(df[[f"fwd{h}" for h in (5, 10, 20, 40, 60)]].agg(["mean", "median"]).round(4).to_string())
win = (df[[f"fwd{h}" for h in (5, 10, 20, 40, 60)]] > 0).mean().round(2)
print("胜率:", win.to_dict())

print("\n=== 信号后60日最大浮亏 ===")
print(f"均值 {df['mdd60'].mean():.2%} | 中位 {df['mdd60'].median():.2%} | 最差 {df['mdd60'].min():.2%}")

print("\n=== 一次性 vs 分4批 (60日收益) ===")
cmp_ = df[["fwd60", "dca60"]].dropna()
cmp_.columns = ["一次性", "分4批"]
print(cmp_.round(4).to_string())
print(f"\n一次性: 均值 {cmp_['一次性'].mean():.2%} 中位 {cmp_['一次性'].median():.2%}")
print(f"分4批 : 均值 {cmp_['分4批'].mean():.2%} 中位 {cmp_['分4批'].median():.2%}")
print(f"分批胜出次数: {(cmp_['分4批'] > cmp_['一次性']).sum()}/{len(cmp_)}")
