"""风格/宽基指数日线加载（D3：右侧载体从个股上移到风格/指数层）。

背景：D1/D2 已把「主板个股追涨突破」归档为负 alpha；外部评审指出 2024/2025
右侧可能上移到风格/指数/行业主题载体。本模块提供风格指数数据底座。

来源：mootdx index_bars 分页（每页 800 根，实测 start 分页可用），覆盖 2019-12 起。
TDX 无中证2000(932000)/中证红利(000922)/中证全指(000985)——
小盘用国证2000(399303)代理，市场代理用沪深300或主板等权。
mootdx 指数 close 与腾讯逐日核对（SH000300 已于生产 A6 验证；
本模块加载时对 000015/399303 做腾讯抽验，见 validate_vs_tencent）。

缓存：results/cache/style_indices.pkl（dict[code] -> DataFrame[open/high/low/close/volume]）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_DIR = _REPO_ROOT / "examples" / "benchmarks_dynamic" / "mainboard_leftside"
if str(_LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_LIVE_DIR))

CACHE_DIR = Path(__file__).resolve().parent / "results" / "cache"
CACHE_FILE = CACHE_DIR / "style_indices.pkl"

# code -> (mootdx symbol, 名称)。mootdx index_bars 用裸 6 位代码。
INDICES: dict[str, tuple[str, str]] = {
    "SH000300": ("000300", "沪深300"),
    "SH000905": ("000905", "中证500"),
    "SH000852": ("000852", "中证1000"),
    "SH000016": ("000016", "上证50"),
    "SZ399303": ("399303", "国证2000"),  # 小盘代理（中证2000/微盘 TDX 无）
    "SH000015": ("000015", "上证红利"),
    "SZ399006": ("399006", "创业板指"),
    "SH000688": ("000688", "科创50"),
    "SZ399370": ("399370", "国证成长"),
    "SZ399371": ("399371", "国证价值"),
}

EARLIEST = pd.Timestamp("2019-12-01")  # 覆盖 2020 起 MA250 burn-in
_PAGE = 800


def _fetch_index(sym6: str, max_retries: int = 5) -> pd.DataFrame:
    """分页拉取单只指数全部日线（EARLIEST 至今）。"""
    from common import mootdx_client

    pages: list[pd.DataFrame] = []
    start = 0
    client = None
    while True:
        for attempt in range(max_retries):
            try:
                if client is None or attempt > 0:
                    client = mootdx_client(refresh=attempt > 0)  # 数据探活重选服务器
                df = client.index_bars(symbol=sym6, frequency=9, start=start, offset=_PAGE)
                if df is not None and not df.empty:
                    break
                # Empty page at start>0 = history fully fetched (total bars an
                # exact multiple of _PAGE) — end of data, not an error. Only an
                # empty FIRST page is a real failure (seen 2026-08-07: 000688
                # hit exactly 1600 bars and step 3 of the daily pipeline died).
                if start == 0 and attempt == max_retries - 1:
                    raise ValueError(f"{sym6} empty at start={start}")
            except Exception:
                if attempt == max_retries - 1:
                    raise
                client = None
                time.sleep(0.5 * (attempt + 1))
        if df is None or df.empty:
            break
        pages.append(df)
        if df.index[0].date() <= EARLIEST.date() or len(df) < _PAGE:
            break
        start += _PAGE
        time.sleep(0.2)

    full = pd.concat(pages).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full = full[full.index >= EARLIEST]
    out = full[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


def validate_vs_tencent(indices: dict[str, pd.DataFrame], n_dates: int = 5) -> list[str]:
    """对非 SH000300 指数做腾讯 fqkline 抽验（生产定案：腾讯为真值源）。"""
    from common import fetch_tencent_kline  # examples/benchmarks_dynamic/mainboard_leftside

    logs = []
    for code in ("SH000015", "SZ399303", "SZ399370"):
        df = indices[code]
        dates = df.index[-60:].to_series().sample(n_dates, random_state=7).sort_values()
        ref = fetch_tencent_kline(code, dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"))
        if ref is None:
            logs.append(f"{code}: 腾讯拉取失败，跳过抽验（WARNING）")
            continue
        ref = ref.set_index("date")["close"]
        worst = 0.0
        for dt in dates:
            d = dt.strftime("%Y-%m-%d")
            if d in ref.index:
                diff = abs(df.loc[dt, "close"] / ref.loc[d] - 1)
                worst = max(worst, diff)
        status = "OK" if worst < 0.002 else "MISMATCH(WARNING)"
        logs.append(f"{code}: 腾讯抽验 {n_dates} 日最大偏差 {worst:.4%} → {status}")
    return logs


def _load_cache() -> dict[str, pd.DataFrame]:
    import pickle

    with open(CACHE_FILE, "rb") as fh:
        return pickle.load(fh)


def _write_cache(indices: dict[str, pd.DataFrame]) -> None:
    """原子覆写缓存，避免中途失败留下半个 pickle。"""
    import pickle

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(CACHE_FILE.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(indices, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE_FILE)


def _patch_cache_with_tencent(
    cached: dict[str, pd.DataFrame], end_date: str | None = None, max_retries: int = 3
) -> dict[str, pd.DataFrame]:
    """用腾讯近 45 日行情补齐缓存；全部指数通过重叠校验后才返回新字典。

    腾讯指数成交量为 mootdx 的 100 倍（2026-10-24 十指数重叠实测），合并前
    统一除以 100。重叠收盘价最大相对误差须不超过 0.2%，换算后成交量中位
    比值须在 [0.5, 2.0]，防止把错误代码或错误单位混入历史缓存。
    """
    from common import fetch_tencent_kline

    end = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.now().normalize()
    patched: dict[str, pd.DataFrame] = {}
    for code in INDICES:
        if code not in cached or cached[code].empty:
            raise ValueError(f"{code}: 缓存缺失或为空，拒绝腾讯增量补齐")
        old = cached[code].copy()
        old.index = pd.DatetimeIndex(old.index).normalize()
        start = old.index[-1] - pd.Timedelta(days=45)
        ref = fetch_tencent_kline(
            code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), adjust="", max_retries=max_retries
        )
        if ref is None or ref.empty:
            raise ValueError(f"{code}: 腾讯增量为空")
        new = ref.copy()
        new["date"] = pd.to_datetime(new["date"], errors="raise").dt.normalize()
        new = new.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        new = new[~new.index.duplicated(keep="last")].sort_index()
        new["volume"] /= 100.0

        overlap = old.index.intersection(new.index)
        if overlap.empty:
            raise ValueError(f"{code}: 腾讯与缓存无重叠日期，拒绝合并")
        old_close = old.loc[overlap, "close"].astype(float)
        new_close = new.loc[overlap, "close"].astype(float)
        close_diff = (new_close / old_close - 1).abs()
        if not close_diff.notna().all() or close_diff.max() > 0.002:
            raise ValueError(f"{code}: 重叠收盘价最大偏差 {close_diff.max():.4%} 超过 0.2%")
        volume_pair = pd.concat(
            [old.loc[overlap, "volume"].rename("old"), new.loc[overlap, "volume"].rename("new")], axis=1
        ).replace([float("inf"), float("-inf")], pd.NA)
        volume_pair = volume_pair.dropna()
        volume_pair = volume_pair[(volume_pair["old"] > 0) & (volume_pair["new"] > 0)]
        if volume_pair.empty:
            raise ValueError(f"{code}: 重叠成交量无正值，无法校验单位")
        volume_ratio = float((volume_pair["new"] / volume_pair["old"]).median())
        if not 0.5 <= volume_ratio <= 2.0:
            raise ValueError(f"{code}: 换算后成交量中位比值 {volume_ratio:.4f} 超出 [0.5, 2.0]")
        if new.index[-1] < old.index[-1]:
            raise ValueError(f"{code}: 腾讯末日 {new.index[-1].date()} 早于缓存末日 {old.index[-1].date()}")

        combined = old.combine_first(new)  # 先保留旧行形状，下面显式让腾讯覆盖重叠行
        combined.update(new)
        combined = pd.concat([combined, new.loc[~new.index.isin(combined.index)]]).sort_index()
        combined = combined[["open", "high", "low", "close", "volume"]].astype(float)
        if len(combined) < len(old) or combined.index[-1] < old.index[-1] or combined.isna().any().any():
            raise ValueError(f"{code}: 合并后历史缩水、末日回退或出现空值，拒绝写缓存")
        patched[code] = combined
        print(
            f"{code}: 腾讯补齐至 {combined.index[-1].date()}，重叠 {len(overlap)} 日，"
            f"收盘最大偏差 {close_diff.max():.4%}，成交量比 {volume_ratio:.4f}"
        )
    return patched


def load_style_indices(
    use_cache: bool = True, validate: bool = True, cache_fallback: bool = True
) -> dict[str, pd.DataFrame]:
    """加载全部风格/宽基指数日线，key 为 Qlib 风格代码（SH000300 等）。

    cache_fallback：全量刷新失败（典型：TDX 服务器夜间维护死区，2026-10-22
    管线 step3 在 00:32~00:36 五次探活全部 "no HQ server answered"）且本地
    缓存存在时，先用腾讯近 45 日行情校验并补齐缓存；腾讯也失败才回退旧缓存，
    并打印数据截止日期。无缓存或显式禁用时不吞掉原始异常。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and CACHE_FILE.exists():
        return _load_cache()

    indices: dict[str, pd.DataFrame] = {}
    try:
        for code, (sym6, name) in INDICES.items():
            df = _fetch_index(sym6)
            indices[code] = df
            print(f"{code} {name}: {len(df)} 行 {df.index[0].date()} ~ {df.index[-1].date()}")
    except Exception as e:
        if cache_fallback and CACHE_FILE.exists():
            cached = _load_cache()
            try:
                indices = _patch_cache_with_tencent(cached)
            except Exception as fallback_error:
                latest = max(df.index[-1] for df in cached.values())
                print(
                    f"WARNING: mootdx 全量刷新失败（{type(e).__name__}: {e}），"
                    f"腾讯增量补齐也失败（{type(fallback_error).__name__}: {fallback_error}），"
                    f"回退缓存 style_indices.pkl（数据至 {latest.date()}）"
                )
                return cached
            _write_cache(indices)
            latest = min(df.index[-1] for df in indices.values())
            print(f"WARNING: mootdx 全量刷新失败，已用腾讯增量补齐缓存（全指数至少至 {latest.date()}）")
            return indices
        raise

    if validate:
        for line in validate_vs_tencent(indices):
            print("抽验:", line)

    _write_cache(indices)
    return indices


if __name__ == "__main__":
    load_style_indices(use_cache=False)
