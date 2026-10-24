#!/usr/bin/env bash
# Daily production + paper-trading pipeline (trading days only).
#
# Chain (order matters, see experiments/style_router/FINDINGS.md §R8):
#   1. examples/benchmarks_dynamic/mainboard_leftside/run_live_pipeline.sh <today>
#      — production mainboard left-side: data patch, dual-model rolling retrain,
#        writes results/mainboard_leftside/picks_live_YYYYMMDD.csv
#   1.5 scripts/data_collector/cffex_collector.py fetch_all
#      — full refresh of CFFEX index-futures (IF/IC/IH/IM dominant, Sina) and
#        style indices (SH000905/SH000852/SH000016, mootdx) bins
#   2. experiments/rightside_trend/run_paper.py
#      — G right-side paper account, writes results/paper/report_YYYY-MM-DD.md
#        and nav_daily.csv (consumed by step 3)
#   3. experiments/style_router/run_paper_t3a.py
#      — t3a three-leg paper account, writes results/paper_t3a/report_YYYY-MM-DD.md
#
# Scheduled via crontab: 0 20 * * 1-5 <this script>
# Log: results/daily_logs/daily_YYYY-MM-DD.log
#
# Non-trading weekdays (holidays) are skipped via a quote-timestamp probe.
# Steps run independently: a failure in one does not abort the rest, and the
# script exits non-zero if any step failed so the log is easy to grep.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TODAY="$(date +%F)"
LOG_DIR="$ROOT/results/daily_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/daily_$TODAY.log"

exec >>"$LOG" 2>&1
echo "===== daily pipeline start $(date '+%F %T') ====="

# Trading-day guard: the Tencent quote for sh000001 carries the exchange's
# latest bar timestamp (YYYYMMDDhhmmss). If it is not today, the market is
# closed (weekend/holiday) -> skip. If the probe itself fails (network or
# rate-limit, seen 2026-07-23), fail open and let the pipeline's own
# fallbacks deal with it.
# Note: the quote payload is GBK-encoded, so grep must run under LC_ALL=C
# (byte mode) — under a UTF-8 locale the invalid bytes stop the match.
QDATE="$(curl -s --max-time 10 'https://qt.gtimg.cn/q=sh000001' \
    | LC_ALL=C grep -ao '20[0-9]\{6\}[0-9]\{6\}' | head -1 | cut -c1-8 || true)"
if [ -n "$QDATE" ] && [ "$QDATE" != "$(date +%Y%m%d)" ]; then
    echo "not a trading day (quote date $QDATE vs today $(date +%Y%m%d)); exit 0"
    "$ROOT/.venv/bin/python" "$ROOT/scripts/push_daily_report.py" \
        --msg "$TODAY 休市，流水线跳过。" || echo "push notice FAILED"
    exit 0
fi
echo "trading-day probe: quote date ${QDATE:-unavailable}, proceeding"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3
STATUS=0

echo "--- step 1/3: production run_live_pipeline.sh $TODAY"
if bash "$ROOT/examples/benchmarks_dynamic/mainboard_leftside/run_live_pipeline.sh" "$TODAY"; then
    echo "step 1 OK"
else
    echo "step 1 FAILED (exit $?)"
    STATUS=1
fi

echo "--- step 1.5/3: index futures + style indices refresh (cffex_collector)"
if "$PY" scripts/data_collector/cffex_collector.py fetch_all; then
    echo "futures refresh OK"
else
    echo "futures refresh FAILED (exit $?)"
    STATUS=1
fi

# Eastmoney news sentiment (FinBERT-Chinese) daily accumulation. The API only
# retains ~6 months of history, so this must run every trading day; the script
# rotates a cursor across the universe (--batch per night) so a full sweep
# takes ~6 nights. Idempotent: 5-day overlap window, atomic file replace,
# resumable via results/sentiment_daily_progress.json.
echo "--- step 1.6/3: news sentiment daily update (sentiment_daily, rotation)"
if "$PY" scripts/data_collector/mootdx/sentiment_daily.py update \
    --batch 300 --workers 4; then
    echo "sentiment update OK"
else
    echo "sentiment update FAILED (exit $?)"
    STATUS=1
fi

echo "--- step 1.7/3: model IC history refresh (ic_streak_history)"
if "$PY" experiments/alpha_enhancement/ic_streak_history.py; then
    echo "ic refresh OK"
else
    echo "ic refresh FAILED (exit $?)"
    STATUS=1
fi

echo "--- step 2/3: paper run_paper.py (G right-side)"
if "$PY" experiments/rightside_trend/run_paper.py; then
    echo "step 2 OK"
else
    echo "step 2 FAILED (exit $?)"
    STATUS=1
fi

echo "--- step 3/3: paper run_paper_t3a.py (t3a three-leg)"
if "$PY" experiments/style_router/run_paper_t3a.py; then
    echo "step 3 OK"
else
    echo "step 3 FAILED (exit $?)"
    STATUS=1
fi

echo "===== daily pipeline end $(date '+%F %T'), status=$STATUS ====="

echo "--- push daily report (wxpusher)"
if "$PY" "$ROOT/scripts/push_daily_report.py" --date "$TODAY" --status "$STATUS"; then
    echo "push OK"
else
    echo "push FAILED (exit $?)"
fi

exit "$STATUS"
