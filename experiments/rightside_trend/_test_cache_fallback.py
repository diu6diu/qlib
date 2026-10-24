"""index_data 缓存兜底与腾讯增量补齐离线测试（2026-10-24）。"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rightside_trend import index_data


def _fake_frame(last_day: str) -> pd.DataFrame:
    idx = pd.date_range("2026-10-19", last_day, freq="B")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)


def _fake_cached() -> dict[str, pd.DataFrame]:
    return {code: _fake_frame("2026-10-21") for code in index_data.INDICES}


def _tencent_frame(df: pd.DataFrame, close_scale: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    out["close"] *= close_scale
    out["volume"] *= 100.0
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    return out.reset_index(drop=True)


def t1_refresh_failure_falls_back_to_cache():
    """mootdx 与腾讯都失败 → 原样返回旧缓存，且不覆写缓存文件。"""
    real_fetch, real_patch = index_data._fetch_index, index_data._patch_cache_with_tencent
    before = index_data.CACHE_FILE.read_bytes()
    calls = {"n": 0}

    def boom(sym6, max_retries=5):
        calls["n"] += 1
        raise ConnectionError("no HQ server answered the data probe")

    try:
        index_data._fetch_index = boom
        index_data._patch_cache_with_tencent = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("dead"))
        got = index_data.load_style_indices(use_cache=False, validate=False)
        assert set(got) == set(index_data.INDICES), "缓存应含全部指数"
        assert calls["n"] == 1, "第一个指数失败即回退，不应继续重试其余指数"
        assert index_data.CACHE_FILE.read_bytes() == before, "双源失败不得改写缓存"
        latest = max(df.index[-1] for df in got.values())
        print(f"PASS t1 双源失败回退旧缓存且不覆写（缓存数据至 {latest.date()}）")
    finally:
        index_data._fetch_index = real_fetch
        index_data._patch_cache_with_tencent = real_patch


def t2_no_cache_reraises():
    """缓存不存在 → 原异常抛出（不产生静默空数据）。"""
    real_fetch, real_cache = index_data._fetch_index, index_data.CACHE_FILE
    try:
        index_data.CACHE_FILE = real_cache.parent / "_nonexistent_cache.pkl"
        index_data._fetch_index = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("dead"))
        try:
            index_data.load_style_indices(use_cache=False, validate=False)
        except ConnectionError:
            print("PASS t2 无缓存时原异常抛出")
        else:
            raise AssertionError("无缓存时应抛出 ConnectionError")
    finally:
        index_data._fetch_index = real_fetch
        index_data.CACHE_FILE = real_cache


def t3_fallback_disabled_reraises():
    """cache_fallback=False → 即使有缓存也抛出（保留显式禁用通道）。"""
    real_fetch = index_data._fetch_index
    try:
        index_data._fetch_index = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("dead"))
        try:
            index_data.load_style_indices(use_cache=False, validate=False, cache_fallback=False)
        except ConnectionError:
            print("PASS t3 cache_fallback=False 时抛出")
        else:
            raise AssertionError("cache_fallback=False 时应抛出")
    finally:
        index_data._fetch_index = real_fetch


def t4_success_path_writes_cache():
    """mootdx 刷新成功 → 返回新数据并覆写临时缓存。"""
    real_fetch = index_data._fetch_index
    try:
        index_data._fetch_index = lambda sym6, max_retries=5: _fake_frame("2026-10-23")
        got = index_data.load_style_indices(use_cache=False, validate=False)
        assert all(df.index[-1] == pd.Timestamp("2026-10-23") for df in got.values())
        print("PASS t4 刷新成功返回新数据并覆写临时缓存")
    finally:
        index_data._fetch_index = real_fetch


def t5_tencent_patch_merges_and_converts_volume():
    """腾讯增量覆盖重叠、追加新日，成交量由 100 倍换算回 mootdx 单位。"""
    import common

    cached = _fake_cached()
    original = cached["SH000300"].copy()
    ref = _tencent_frame(_fake_frame("2026-10-23"))
    real_fetch = common.fetch_tencent_kline
    try:
        common.fetch_tencent_kline = lambda *a, **k: ref.copy()
        got = index_data._patch_cache_with_tencent(cached, end_date="2026-10-23")
        df = got["SH000300"]
        assert len(df) == 5 and df.index[-1] == pd.Timestamp("2026-10-23")
        assert df.loc[pd.Timestamp("2026-10-23"), "volume"] == 1.0
        pd.testing.assert_frame_equal(cached["SH000300"], original)
        print("PASS t5 腾讯合并、重叠覆盖、100 倍成交量换算且不修改入参")
    finally:
        common.fetch_tencent_kline = real_fetch


def t6_tencent_patch_rejects_close_mismatch():
    """重叠收盘价偏差超过 0.2% → 拒绝合并。"""
    import common

    cached = _fake_cached()
    ref = _tencent_frame(_fake_frame("2026-10-23"), close_scale=1.01)
    real_fetch = common.fetch_tencent_kline
    try:
        common.fetch_tencent_kline = lambda *a, **k: ref.copy()
        try:
            index_data._patch_cache_with_tencent(cached, end_date="2026-10-23")
        except ValueError as e:
            assert "收盘价" in str(e)
            print("PASS t6 重叠收盘价异常拒绝合并")
        else:
            raise AssertionError("重叠收盘价异常时必须抛出 ValueError")
    finally:
        common.fetch_tencent_kline = real_fetch


def t7_tencent_patch_rejects_stale_data():
    """腾讯数据末日早于缓存 → 不允许历史回退。"""
    import common

    cached = _fake_cached()
    ref = _tencent_frame(_fake_frame("2026-10-20"))
    real_fetch = common.fetch_tencent_kline
    try:
        common.fetch_tencent_kline = lambda *a, **k: ref.copy()
        try:
            index_data._patch_cache_with_tencent(cached, end_date="2026-10-23")
        except ValueError as e:
            assert "早于缓存末日" in str(e)
            print("PASS t7 腾讯数据滞后时拒绝回退缓存末日")
        else:
            raise AssertionError("腾讯数据滞后时必须抛出 ValueError")
    finally:
        common.fetch_tencent_kline = real_fetch


if __name__ == "__main__":
    real_cache_dir, real_cache_file = index_data.CACHE_DIR, index_data.CACHE_FILE
    with tempfile.TemporaryDirectory(prefix="test_style_cache_", dir=Path(__file__).parent) as tmp:
        index_data.CACHE_DIR = Path(tmp)
        index_data.CACHE_FILE = index_data.CACHE_DIR / "style_indices.pkl"
        index_data._write_cache(_fake_cached())
        try:
            t1_refresh_failure_falls_back_to_cache()
            t2_no_cache_reraises()
            t3_fallback_disabled_reraises()
            t4_success_path_writes_cache()
            t5_tencent_patch_merges_and_converts_volume()
            t6_tencent_patch_rejects_close_mismatch()
            t7_tencent_patch_rejects_stale_data()
        finally:
            index_data.CACHE_DIR, index_data.CACHE_FILE = real_cache_dir, real_cache_file
    print("\nALL 7 PASS（全程使用临时缓存，未触碰真实缓存）")
