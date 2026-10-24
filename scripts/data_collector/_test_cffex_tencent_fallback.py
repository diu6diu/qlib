"""Regression tests for the cffex mootdx -> Tencent index fallback."""

import shutil
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data_collector import cffex_collector as cffex

TEST_PARENT = ROOT / "results" / "step15_repair"


def _workspace_tmp() -> Path:
    path = TEST_PARENT / f"test_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _old_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-10-19", "2026-10-21", freq="B"),
            "open": [99.0, 100.0, 101.0],
            "high": [101.0, 102.0, 103.0],
            "low": [98.0, 99.0, 100.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )


def _new_frame(close_multiplier: float = 1.0) -> pd.DataFrame:
    old = _old_frame()
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-10-19", "2026-10-23", freq="B"),
            "open": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": old["close"].tolist() + [103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [98.0, 99.0, 100.0, 101.0, 102.0],
            "volume": [110000.0, 120000.0, 130000.0, 140000.0, 150000.0],
        }
    ).assign(close=lambda df: df["close"] * close_multiplier)


def _patch_tencent(frame):
    return patch.object(cffex, "_shared_tencent_kline", return_value=frame)


def t1_merge_and_scale_volume():
    tmp = _workspace_tmp()
    try:
        raw_path = tmp / "SH000905.csv"
        _old_frame().to_csv(raw_path, index=False)
        with _patch_tencent(_new_frame()):
            got = cffex._fetch_tencent_index_update("SH000905", raw_path, end_date="2026-10-23")
        assert len(got) == 5
        assert got["date"].iloc[-1] == pd.Timestamp("2026-10-23")
        assert got["volume"].tolist() == [1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
        assert got["close"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]
        print("PASS t1 腾讯增量合并、去重及成交量 100 倍换算")
    finally:
        shutil.rmtree(tmp)


def t2_overlap_mismatch_rejected():
    tmp = _workspace_tmp()
    try:
        raw_path = tmp / "SH000905.csv"
        _old_frame().to_csv(raw_path, index=False)
        with _patch_tencent(_new_frame(close_multiplier=1.01)):
            try:
                cffex._fetch_tencent_index_update("SH000905", raw_path, end_date="2026-10-23")
            except ValueError as exc:
                assert "close mismatch" in str(exc)
            else:
                raise AssertionError("重叠价格不一致时必须拒绝合并")
        print("PASS t2 重叠价格不一致时拒绝合并")
    finally:
        shutil.rmtree(tmp)


def _collector_fixture(tmp: Path):
    qlib_dir = tmp / "qlib"
    source_base = tmp / "source"
    (qlib_dir / "calendars").mkdir(parents=True)
    (qlib_dir / "instruments").mkdir(parents=True)
    pd.Series(pd.date_range("2026-10-19", "2026-10-23", freq="B")).dt.strftime("%Y-%m-%d").to_csv(
        qlib_dir / "calendars" / "day.txt", index=False, header=False
    )
    (qlib_dir / "instruments" / "index.txt").write_text("SH000300\t2020-01-02\t2026-10-21\n")
    raw_dir = source_base / "mootdx_indices"
    raw_dir.mkdir(parents=True)
    for symbol in cffex.INDEX_MAP:
        _old_frame().to_csv(raw_dir / f"{symbol}.csv", index=False)
    return qlib_dir, source_base


class _FakeDump:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def dump(self):
        self.calls.append(self.kwargs)


def t3_fetch_indices_fallback_and_instrument_update():
    tmp = _workspace_tmp()
    try:
        qlib_dir, source_base = _collector_fixture(tmp)
        _FakeDump.calls = []
        with (
            patch.object(cffex, "fetch_mootdx_index", side_effect=ConnectionError("dead")),
            _patch_tencent(_new_frame()),
            patch.object(cffex, "DumpFeaturesOnly", _FakeDump),
            patch.object(cffex.time, "sleep", return_value=None),
        ):
            cffex.CffexCollector(qlib_dir=qlib_dir, source_base=source_base, max_retries=1).fetch_indices()
        assert len(_FakeDump.calls) == 1
        for symbol in cffex.INDEX_MAP:
            raw = pd.read_csv(source_base / "mootdx_indices" / f"{symbol}.csv")
            normalized = pd.read_csv(source_base / "mootdx_indices_normalize" / f"{symbol}.csv")
            assert raw["date"].iloc[-1] == "2026-10-23"
            assert len(raw) == 5
            assert normalized["volume"].iloc[-2] == 1400.0
        instruments = (qlib_dir / "instruments" / "index.txt").read_text()
        assert "SH000300\t2020-01-02\t2026-10-21" in instruments
        assert instruments.count("2026-10-23") == 3
        print("PASS t3 步骤 1.5 兜底写盘并更新 index.txt，保留其他指数")
    finally:
        shutil.rmtree(tmp)


def t4_both_sources_failed_no_dump_or_overwrite():
    tmp = _workspace_tmp()
    try:
        qlib_dir, source_base = _collector_fixture(tmp)
        raw_path = source_base / "mootdx_indices" / "SH000905.csv"
        before = raw_path.read_bytes()
        _FakeDump.calls = []
        with (
            patch.object(cffex, "fetch_mootdx_index", side_effect=ConnectionError("dead")),
            _patch_tencent(None),
            patch.object(cffex, "DumpFeaturesOnly", _FakeDump),
            patch.object(cffex.time, "sleep", return_value=None),
        ):
            try:
                cffex.CffexCollector(qlib_dir=qlib_dir, source_base=source_base, max_retries=1).fetch_indices()
            except ValueError as exc:
                assert "Tencent fallback returned no data" in str(exc)
            else:
                raise AssertionError("两个数据源都失败时必须抛出异常")
        assert _FakeDump.calls == []
        assert raw_path.read_bytes() == before
        print("PASS t4 双源失败时抛出异常，不写 bin、不覆盖历史")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    TEST_PARENT.mkdir(parents=True, exist_ok=True)
    t1_merge_and_scale_volume()
    t2_overlap_mismatch_rejected()
    t3_fetch_indices_fallback_and_instrument_update()
    t4_both_sources_failed_no_dump_or_overwrite()
    print("\nALL 4 PASS")
