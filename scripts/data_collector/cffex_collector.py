# -*- coding: utf-8 -*-
"""Collect CFFEX index-futures daily bars (Sina) and missing spot indices (mootdx)
into the qlib binary store (~/.qlib/qlib_data/cn_data_mootdx).

- fetch_futures: IF0/IC0/IH0/IM0 dominant continuous contracts from Sina
  InnerFuturesNewService.getDailyKLine, dumped as IF00/IC00/IH00/IM00 with a new
  instruments/index_futures.txt.
- fetch_indices: SH000905/SH000852/SH000016 via mootdx index_bars, appended to
  instruments/index.txt (existing lines untouched).

Design: mootdx full refresh is primary. If it fails, cached raw index history
is patched with recent Tencent bars after overlap and volume-unit checks.
Each run re-dumps bins for these 7 symbols only. Calendars/day.txt and
instruments/all.txt are never touched. If a fetch comes back empty or with
fewer bars than the previous raw csv, the run aborts before dumping anything
(never shrink existing data).

Usage:
    python scripts/data_collector/cffex_collector.py fetch_futures
    python scripts/data_collector/cffex_collector.py fetch_indices
    python scripts/data_collector/cffex_collector.py fetch_all
"""

import json
import re
import sys
import time
from pathlib import Path

import fire
import pandas as pd
import requests
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from scripts.dump_bin import DumpDataBase

DEFAULT_QLIB_DIR = "~/.qlib/qlib_data/cn_data_mootdx"
DEFAULT_SOURCE_BASE = "~/.qlib/qlib_data/source"

# qlib symbol -> Sina continuous-contract symbol
FUTURES_MAP = {
    "IF00": "IF0",
    "IC00": "IC0",
    "IH00": "IH0",
    "IM00": "IM0",
}

# qlib symbol -> mootdx 6-digit index symbol (SH market)
INDEX_MAP = {
    "SH000905": "000905",
    "SH000852": "000852",
    "SH000016": "000016",
}

DUMP_FIELDS = ["open", "high", "low", "close", "volume"]
SINA_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_x=/"
    "InnerFuturesNewService.getDailyKLine?symbol={symbol}"
)
SINA_HEADERS = {"User-Agent": "Mozilla/5.0"}
SINA_PATTERN = re.compile(r"=\((\[.*?\])\)", re.DOTALL)

MOOTDX_PAGE = 800


class DumpFeaturesOnly(DumpDataBase):
    """Dump csv files to bins against the EXISTING calendar only.

    Unlike DumpDataAll this never rewrites calendars/day.txt or
    instruments/all.txt; bins are fully rewritten (ALL_MODE) per symbol.
    Serial on purpose: 7 symbols is trivial and avoids ProcessPoolExecutor.
    """

    def dump(self):
        calendars = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        for file_path in self.df_files:
            self._dump_bin(file_path, calendars)


def _request_with_retry(url, max_retries: int = 5, base_sleep: float = 2.0) -> str:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=SINA_HEADERS, timeout=15)
            resp.raise_for_status()
            m = SINA_PATTERN.search(resp.text)
            if m:
                return m.group(1)
            raise ValueError(f"unexpected Sina response: {resp.text[:200]!r}")
        except Exception:
            if attempt == max_retries - 1:
                raise
            sleep_s = base_sleep * (attempt + 1)
            logger.warning(f"request failed (attempt {attempt + 1}/{max_retries}), sleep {sleep_s}s")
            time.sleep(sleep_s)
    raise RuntimeError("unreachable")


def fetch_sina_futures(symbol: str, max_retries: int = 5) -> pd.DataFrame:
    """Fetch full daily history of one Sina dominant continuous contract (e.g. IF0)."""
    payload = _request_with_retry(SINA_URL.format(symbol=symbol), max_retries=max_retries)
    records = json.loads(payload)
    if not records:
        raise ValueError(f"{symbol}: Sina returned 0 records")
    df = pd.DataFrame(records)
    df = df.rename(columns={"d": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "p": "oi"})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "oi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", "open", "high", "low", "close", "volume", "oi"]].sort_values("date").reset_index(drop=True)
    return df


def _import_mootdx_quotes():
    """Import the real mootdx package, not scripts/data_collector/mootdx/.

    When run as a script, sys.path[0] is scripts/data_collector/ whose mootdx/
    subpackage would shadow the site-packages mootdx.
    """
    script_dir = str(Path(__file__).resolve().parent)
    popped = sys.path.pop(0) if sys.path and Path(sys.path[0]).resolve() == Path(script_dir) else None
    try:
        for name in list(sys.modules):
            mod = sys.modules.get(name)
            if name.split(".")[0] == "mootdx" and mod is not None and script_dir in str(getattr(mod, "__file__", "")):
                del sys.modules[name]
        from mootdx.quotes import Quotes
    finally:
        if popped is not None:
            sys.path.insert(0, popped)
    return Quotes


def _shared_mootdx_client(refresh: bool = False):
    """Data-verified mootdx client from mainboard_leftside/common.py.

    _import_mootdx_quotes() runs first so the real mootdx is already in
    sys.modules when common.py imports it (bypassing the shadow above).
    """
    _import_mootdx_quotes()
    live_dir = str(Path(__file__).resolve().parents[2] / "examples" / "benchmarks_dynamic" / "mainboard_leftside")
    if live_dir not in sys.path:
        sys.path.append(live_dir)
    from common import mootdx_client

    return mootdx_client(refresh=refresh)


def _shared_tencent_kline(code: str, start: str, end: str, max_retries: int = 3):
    """Fetch raw index bars through the live pipeline's verified Tencent channel."""
    live_dir = str(Path(__file__).resolve().parents[2] / "examples" / "benchmarks_dynamic" / "mainboard_leftside")
    if live_dir not in sys.path:
        sys.path.append(live_dir)
    from common import fetch_tencent_kline

    return fetch_tencent_kline(code, start=start, end=end, adjust="", max_retries=max_retries)


def _fetch_tencent_index_update(
    qlib_sym: str, raw_path: Path, end_date: str | None = None, max_retries: int = 3
) -> pd.DataFrame:
    """Merge a recent Tencent patch into the cached full mootdx history.

    Tencent index volume is 100 times the mootdx ``index_bars`` unit. Overlap
    checks protect both price compatibility and the volume-unit conversion.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"{qlib_sym}: Tencent fallback requires cached raw history at {raw_path}")
    old = pd.read_csv(raw_path, parse_dates=["date"])
    if old.empty:
        raise ValueError(f"{qlib_sym}: cached raw history is empty")

    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.now().normalize()
    start = max(old["date"].min(), old["date"].max() - pd.Timedelta(days=45))
    new = _shared_tencent_kline(
        qlib_sym,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        max_retries=max_retries,
    )
    if new is None or new.empty:
        raise ValueError(f"{qlib_sym}: Tencent fallback returned no data")
    new = new.copy()
    new["date"] = pd.to_datetime(new["date"])
    new["volume"] = pd.to_numeric(new["volume"], errors="raise") / 100.0
    new = new[["date"] + DUMP_FIELDS]

    overlap = old.merge(new, on="date", suffixes=("_old", "_new"))
    if overlap.empty:
        raise ValueError(f"{qlib_sym}: Tencent fallback has no overlap with cached history")
    close_error = (overlap["close_new"] / overlap["close_old"] - 1).abs().max()
    if close_error > 0.002:
        raise ValueError(f"{qlib_sym}: Tencent/mootdx close mismatch {close_error:.4%}")
    nonzero = overlap["volume_old"] > 0
    volume_ratio = (overlap.loc[nonzero, "volume_new"] / overlap.loc[nonzero, "volume_old"]).median()
    if pd.isna(volume_ratio) or not 0.5 <= volume_ratio <= 2.0:
        raise ValueError(f"{qlib_sym}: Tencent/mootdx volume-unit mismatch (scaled ratio {volume_ratio})")

    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    if len(combined) < len(old) or combined["date"].iloc[-1] < old["date"].iloc[-1]:
        raise ValueError(f"{qlib_sym}: Tencent fallback would shrink or roll back cached history")
    return combined


def fetch_mootdx_index(sym6: str, max_retries: int = 5) -> pd.DataFrame:
    """Fetch full daily history of one index via mootdx index_bars pagination."""
    pages = []
    start = 0
    client = None
    while True:
        df = None
        for attempt in range(max_retries):
            try:
                if client is None or attempt > 0:
                    client = _shared_mootdx_client(refresh=attempt > 0)  # data-probed server
                df = client.index_bars(symbol=sym6, frequency=9, start=start, offset=MOOTDX_PAGE)
                if df is not None and not df.empty:
                    break
                # empty page at start>0 = history fully fetched; empty FIRST page is a failure
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
        if len(df) < MOOTDX_PAGE:
            break
        start += MOOTDX_PAGE
        time.sleep(0.2)

    if not pages:
        raise ValueError(f"{sym6}: no data fetched")
    full = pd.concat(pages).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    out = full[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.DatetimeIndex(out.index).normalize()
    out.index.name = "date"
    return out.reset_index()


def _guard_not_shrinking(df: pd.DataFrame, raw_path: Path, symbol: str):
    """Abort if the new fetch has fewer bars than the previously saved raw csv."""
    if raw_path.exists():
        old_len = sum(1 for _ in raw_path.open()) - 1  # minus header
        if len(df) < old_len:
            raise RuntimeError(
                f"{symbol}: new fetch has {len(df)} bars < previous {old_len} bars; "
                f"refusing to shrink data, aborting before any dump"
            )


def _write_qlib_csv(df: pd.DataFrame, path: Path):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out[["date"] + DUMP_FIELDS].to_csv(path, index=False)


def _read_calendar(qlib_dir: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(qlib_dir / "calendars" / "day.txt", header=None)[0])


def _clipped_range(df: pd.DataFrame, calendar: pd.DatetimeIndex):
    """First/last calendar dates covered by df (bins are aligned to the calendar)."""
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    start = max(dates.min(), calendar.min())
    end = min(dates.max(), calendar.max())
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


class CffexCollector:
    def __init__(
        self,
        qlib_dir: str = DEFAULT_QLIB_DIR,
        source_base: str = DEFAULT_SOURCE_BASE,
        request_sleep: float = 2.5,
        max_retries: int = 5,
    ):
        """

        Parameters
        ----------
        qlib_dir: str
            qlib binary data directory
        source_base: str
            base dir for raw + qlib-format csvs
        request_sleep: float
            seconds to sleep between Sina symbols (be polite)
        max_retries: int
            request retry count per symbol
        """
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.source_base = Path(source_base).expanduser().resolve()
        self.request_sleep = request_sleep
        self.max_retries = max_retries
        if not (self.qlib_dir / "calendars" / "day.txt").exists():
            raise ValueError(f"calendar not found under {self.qlib_dir}")

    # ------------------------------------------------------------------
    def fetch_futures(self):
        raw_dir = self.source_base / "cffex_sina"
        norm_dir = self.source_base / "cffex_sina_normalize"
        raw_dir.mkdir(parents=True, exist_ok=True)
        norm_dir.mkdir(parents=True, exist_ok=True)

        # fetch everything first; only dump after ALL symbols pass the shrink guard
        fetched = {}
        for qlib_sym, sina_sym in FUTURES_MAP.items():
            df = fetch_sina_futures(sina_sym, max_retries=self.max_retries)
            raw_path = raw_dir / f"{qlib_sym}.csv"
            _guard_not_shrinking(df, raw_path, qlib_sym)
            fetched[qlib_sym] = df
            logger.info(
                f"{qlib_sym} ({sina_sym}): {len(df)} bars "
                f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}"
            )
            time.sleep(self.request_sleep)

        for qlib_sym, df in fetched.items():
            df.to_csv(raw_dir / f"{qlib_sym}.csv", index=False)
            _write_qlib_csv(df, norm_dir / f"{qlib_sym}.csv")

        DumpFeaturesOnly(
            data_path=str(norm_dir),
            qlib_dir=str(self.qlib_dir),
            include_fields=",".join(DUMP_FIELDS),
            max_workers=1,
        ).dump()

        calendar = _read_calendar(self.qlib_dir)
        lines = []
        for qlib_sym, df in fetched.items():
            start, end = _clipped_range(df, calendar)
            lines.append(f"{qlib_sym}\t{start}\t{end}")
        inst_path = self.qlib_dir / "instruments" / "index_futures.txt"
        inst_path.write_text("\n".join(lines) + "\n")
        logger.info(f"wrote {inst_path}:\n" + "\n".join(lines))

    # ------------------------------------------------------------------
    def fetch_indices(self):
        raw_dir = self.source_base / "mootdx_indices"
        norm_dir = self.source_base / "mootdx_indices_normalize"
        raw_dir.mkdir(parents=True, exist_ok=True)
        norm_dir.mkdir(parents=True, exist_ok=True)

        fetched = {}
        for qlib_sym, sym6 in INDEX_MAP.items():
            raw_path = raw_dir / f"{qlib_sym}.csv"
            try:
                df = fetch_mootdx_index(sym6, max_retries=self.max_retries)
            except Exception as mootdx_error:
                df = _fetch_tencent_index_update(qlib_sym, raw_path, end_date=None, max_retries=self.max_retries)
                logger.warning(
                    f"mootdx 全量刷新失败（{qlib_sym}: {mootdx_error}），"
                    f"回退腾讯增量（数据至 {df['date'].iloc[-1].date()}）"
                )
            _guard_not_shrinking(df, raw_path, qlib_sym)
            fetched[qlib_sym] = df
            logger.info(
                f"{qlib_sym} ({sym6}): {len(df)} bars {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}"
            )
            time.sleep(0.5)

        for qlib_sym, df in fetched.items():
            df.to_csv(raw_dir / f"{qlib_sym}.csv", index=False)
            _write_qlib_csv(df, norm_dir / f"{qlib_sym}.csv")

        DumpFeaturesOnly(
            data_path=str(norm_dir),
            qlib_dir=str(self.qlib_dir),
            include_fields=",".join(DUMP_FIELDS),
            max_workers=1,
        ).dump()

        calendar = _read_calendar(self.qlib_dir)
        inst_path = self.qlib_dir / "instruments" / "index.txt"
        new_symbols = set(INDEX_MAP)
        kept = []
        if inst_path.exists():
            kept = [
                line.rstrip("\n")
                for line in inst_path.read_text().splitlines()
                if line.strip() and line.split("\t")[0].upper() not in new_symbols
            ]
        for qlib_sym, df in fetched.items():
            start, end = _clipped_range(df, calendar)
            kept.append(f"{qlib_sym}\t{start}\t{end}")
        inst_path.write_text("\n".join(kept) + "\n")
        logger.info(f"wrote {inst_path}:\n" + "\n".join(kept))

    # ------------------------------------------------------------------
    def fetch_all(self):
        self.fetch_futures()
        self.fetch_indices()


if __name__ == "__main__":
    fire.Fire(CffexCollector)
