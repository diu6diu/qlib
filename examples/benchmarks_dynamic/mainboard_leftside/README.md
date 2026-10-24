# Main-board left-side quality-reversal strategy

**Production chain (since 2026-07-24): blend50 dual-model W5.** Two rolling-retrain
LightGBM models (pure Alpha158 "base" and Alpha158+P0, both horizon=5) are retrained
daily; their signals are cross-sectionally z-scored and averaged 50/50 (blend50);
picks are the official TopkDropout top10/n_drop=2 portfolio with ST-only hard
filtering. Rolling 20d rank-ICs are still computed for alpha-decay monitoring only.
Rationale and backtests: `experiments/alpha_enhancement/FINDINGS.md` Phase 9–16
(blend50, 2024-07~2026-07, official tk10 nd2 @100k. Phase 15-A headline
ann 161.3%/Sharpe 2.83/MDD −23.4% was measured on pre-repair data and no longer
reproduces; re-run 2026-08-15 after ex-rights repairs: 130.2%/3.59/−25.2%.
With sealed-limit-up/suspended fills modeled as unfilled — the executable
benchmark: **ann 57.8%, Sharpe 1.79, MDD −32.3%**. It replaced the 2026-07-23
IC-switch chain — statistically equal returns, higher ICIR, no arbitration
mechanism).

Legacy note: the old chain (single horizon-1 model + BB20 + sentiment + quality gate,
top-100 picks) was retired after Phase 9–11 showed all soft filters destroy alpha.

## Files

| File | Purpose |
|------|---------|
| `build_univ.py` | Build `mainboard.txt` instrument file (60[0135]/00[0123], listed >1yr, top-1800 by turnover). |
| `roll_base.yaml` | Qlib workflow config for LightGBM + Alpha158. |
| `run_roll.py` | Drive rolling retrain (25 windows, step=20 days, horizon=1). |
| `extract_signal.py` | Pull combined `pred.pkl` from mlflow. |
| `bt_roll.py` | Backtest TopkDropout top100/n_drop=10 and save top-100 picks. |
| `run_pipeline.sh` | End-to-end baseline pipeline. |
| `common.py` | Shared helpers for the live pipeline (Tencent klines, Qlib binary I/O, calendar helpers). |
| `extend_ohlcv.py` | Extend Qlib OHLCV by a few trading days from Tencent klines (`patch` or `dump`). |
| `extend_ohlcv_mootdx.py` | Same patch via mootdx/TDX (fallback when Tencent blocks the IP); repairs calendar gaps. |
| `run_live_pipeline.sh` | One-shot daily pipeline: patch data → dual retrain (base+P0, horizon=5) → extract both signals → generate picks → clean stale handler caches. |
| `generate_live_picks.py` | blend50 signal fusion → ST removal → official tk10/nd2 target portfolio → rebalance instructions (卖X买Y). Phase 19: both roll20 ICs < 0 for 10 consecutive days => de-risk rule halves the target (tk10→tk5, sells only) until the streak breaks; `risk_state` column in picks CSV. Phase 20: buys unfillable at the execution open (sealed one-word board / suspension / open gap >+9.5%) are skipped with the paired sell kept (swap cancelled, portfolio stays at topk) and cooled in `limit_skip.json` until they decay out of the elite zone. Since 2026-10-23 the sealed check runs BEFORE stock selection (detected from yesterday's picks + today's open), so the same night's picks already exclude the sealed name and restore the paired incumbent — previously it was recorded after selection, leaving holdings one day stale. Phase 21 (V3 breaker): the paper chain's own trailing 20-day return ≤ −3% switches execution to half size (normal rotation continues, `risk_state="HALF(breaker)"` in `breaker.json`); recovery above −1.5% restores full size (−1% until 2026-10-23; experiment C scan: ann +0.5pp / Sharpe +0.015, earlier re-entry, hysteresis 2pp→1.5pp still in the stable zone). Backtest: bt_self_breaker.py — ann 53.3%/Sharpe 1.89/MDD −21.5% vs 56.3%/1.77/−28.2% baseline. |
| `time_decay_reweight.py` | Phase 15-C recency-reweighting infra for LGBModel (direction rejected; kept for reference). |
| `roll_base_rw.yaml` | Rolling config using the recency reweighter (Phase 15-C experiment). |
| `alpha158_p1.py` | Phase 18 `Alpha158P1` handler: Alpha158P0 + MAX/MIN/AMIHUD/SKEW anomaly factors (rejected — front-loaded gains, 2026H1 collapse; kept for reference). |
| `roll_p1.yaml` | Rolling config using `Alpha158P1` (Phase 18 experiment). |
| `test_pit.py` | Small sanity check for PIT fundamental access (`$$eps_q_q`, `$$ni_q_q`). |
| `alpha158_sentiment.py` | Custom `Alpha158Sentiment` DataHandler adding Eastmoney news sentiment features. |
| `roll_sentiment.yaml` | Rolling-retrain config using `Alpha158Sentiment`. |
| `run_roll_sentiment.py` | Drive sentiment-enhanced rolling retrain. |
| `sentiment_overlay.py` | Post-hoc sentiment overlays on the existing baseline signal. |
| `collect_historical_sentiment.py` | Wrapper to collect historical sentiment for `mainboard.txt`. |
| `dump_gtja120.py` | Daily incremental update of the `$gtja120` field (GTJA alpha#120, vwap-deviation reversal). Runs in `run_live_pipeline.sh` after the OHLCV patch + ex-day repair, before the retrain; fatal on failure. Live since 2026-10-16 (triple-validated in `experiments/gtja120/FINDINGS.md`; handlers are `Alpha158Gtja120Base`/`Alpha158Gtja120P0` in `experiments/gtja120/alpha158_gtja120.py`, referenced by `roll_base.yaml`/`roll_p0.yaml`). |

## Production status (blend50, first live picks 2026-07-24)

- Universe: main-board ~1800 (`mainboard.txt`), daily signals through 2026-07-24
- Execution: T+1 open, official TopkDropout top10/n_drop=2, 100k account
- Costs: commission 0.015%/side (min ¥5/stock), stamp 0.05% sell-only,
  transfer 0.002%, slippage 0.1%/side (`experiments/rightside_alpha/simulation.py`)
- Backtest 2024-07 ~ 2026-07 (`experiments/alpha_enhancement/FINDINGS.md` Phase 15-A;
  Phase 16 re-run 2026-08-15 on post-repair data):

| metric | executable (sealed/suspended unfilled) | fill-all (original engine) | Phase 15-A (2026-07-24 data, superseded) |
|--------|---------------------------------------:|---------------------------:|-----------------------------------------:|
| annualized_return | 0.578 | 1.302 | 1.613 |
| sharpe | 1.79 | 3.59 | 2.83 |
| max_drawdown | -0.323 | -0.252 | -0.234 |

> Executable = `run_backtest(model_limit=True)`: new buys that are suspended or
> sealed limit-up (open==high==low, gap >9.5%) at the T+1 open are skipped to
> cash — 79 such events in two years, averaging +24.9% over their paper holding
> spans, are unattainable live. The Phase 15-A column predates the 2026-08
> ex-rights data repairs and no longer reproduces. Treat **0.578 as the
> realistic ceiling**; 2026H1 alone: ann 13.7%, Sharpe 0.44.

> Reference only: the legacy single-model top-100 chain (retrained
> 2026-07-02 ~ 2026-07-03: ann 39.7%, IR 1.97, MDD -17.9%, `bt_roll.py`)
> was superseded by the dual-model W5 chain (Phase 9–13), which was then
> replaced by blend50 (Phase 15).

## Live pick pipeline

For a real trading day, the full workflow is wrapped in `run_live_pipeline.sh`:

```bash
./run_live_pipeline.sh 2026-06-30
```

The script requires one argument: the target date (`YYYY-MM-DD`).  It runs these
stages in order (dual retrain takes ~1.5h total; a fresh Alpha158 handler cache,
~3.4GB per config, is built for each target date):

1. **Extend OHLCV data** with `extend_ohlcv.py patch` (Tencent kline).  If that
   fails (IP rate-limit/ban), the script falls back to `extend_ohlcv_mootdx.py`
   (TDX servers via mootdx), which also repairs calendar gaps left behind by
   failed patch runs.  Note `extend_ohlcv_mootdx.py` fills missing **weekdays**
   without holiday detection — verify the window has no market holiday.
2. **Dual rolling retrain** with `run_roll.py --horizon 5`:
   - `--config roll_base.yaml --exp-name mainboard_w5_live_base` (pure W5)
   - `--config roll_p0.yaml --exp-name mainboard_w5_live_p0` (W5+P0)
3. **Extract both signals** with `extract_signal.py` (one per experiment)
4. **Generate picks + rebalance instructions** with `generate_live_picks.py`
5. **Delete stale handler caches** (`Alpha*.pkl` older than a day; they are
   keyed to their target date and never reused, ~3.4GB each)

### Operational notes

- **Hardware**: retrain peaks at ~5-6GB RAM.  On the 8GB production Mac keep
  >=10GB disk free and avoid heavy apps during the run — on 2026-07-24 a
  99%-full disk stopped swap growth and the retrain was SIGKILLed 3 times
  (details in `experiments/alpha_enhancement/FINDINGS.md` ops sections).
- **`PortAnaRecord` is disabled** in both rolling configs: its TopkDropout
  report needs one trading day beyond the backtest end and crashes when the
  target is the last calendar day.  Live picks only need `SignalRecord`
  (combined `pred.pkl` is saved before the record phase, so a record failure
  never invalidates the signal).
- **Manual `run_roll.py` reruns** outside the pipeline must clear
  `mlruns/.trash` first — deleted experiments still hold their names
  (`ExpAlreadyExistError`).  The pipeline handles this by removing `mlruns/`
  at the start of every run.
- **Benchmark index (SH000300) is patched by the pipeline** via
  `instruments/index.txt` right after the stock patch (Tencent first, mootdx
  fallback; failure only warns — picks don't need it, rolling-backtest excess
  returns do).  The index carries only OHLCV bins (factor is treated as 1.0);
  `extend_ohlcv.py` derives its end-date from `close.day.bin` when it is
  absent from `instruments/all.txt`, so collector runs that regenerate
  `all.txt` from stock CSVs cannot silently drop it.  It once stagnated at
  2026-06-30 because nothing patched it.
- **Ex-rights health check runs at the end of every live pipeline**:
  `scripts/check_exrights_health.py --days 45` scans the trailing window for
  |daily ret|>11% and verifies each candidate against Tencent qfq (the sole
  source of truth for corporate actions; cn_data/Yahoo is screening-only).
  Findings land in `~/.qlib/qlib_data/cn_data_mootdx/exrights_health_latest.csv`;
  the pipeline only warns, it never blocks picks.  mootdx qfq lags recent
  corporate actions, so CONFIRMED rows need repair per
  `scripts/repair_exrights_fakes.py`'s scaling formula.

### Data extension: `patch` vs `dump`

`extend_ohlcv.py` fetches raw daily klines from Tencent and scales them into the
Qlib-normalized space.  It has two subcommands:

* `patch` (default in `run_live_pipeline.sh`) — writes new trading days directly
  into the trailing end of each symbol's `.day.bin` files and updates the
  instrument end-dates.  This is fast and avoids regenerating the entire dataset,
  but it mutates the Qlib binary store in place.
* `dump` — exports per-symbol CSVs and invokes `scripts/dump_bin.py dump_update`.
  Use this when you want a safer, reversible update path or when you need to
  re-dump features through the standard Qlib converter.

Both subcommands accept `--target-date`, `--inst-file`, and `--qlib-dir`.  `patch`
also supports `--fail-ratio-threshold` to abort if too many symbols fail to fetch.
By default `--target-date` must already be a trading day in the Qlib calendar;
use `--force-append-calendar` to append a missing weekday after the calendar end
(for example when the downloaded dataset has not yet been updated for the latest
day).  Holidays are not auto-detected, so only force-append dates you have
verified are real trading days.

```bash
# Direct binary patch (fast, in-place)
python extend_ohlcv.py patch --target-date 2026-06-30 \
    --inst-file mainboard.txt --qlib-dir ~/.qlib/qlib_data/cn_data_mootdx \
    --force-append-calendar

# CSV + dump_bin.py path (safer, reproducible)
python extend_ohlcv.py dump --target-date 2026-06-30 \
    --inst-file mainboard.txt --qlib-dir ~/.qlib/qlib_data/cn_data_mootdx \
    --csv-dir /tmp/extend_csv
```

### Rolling retrain CLI

`run_roll.py` renders `roll_base.yaml` for the target date and runs the rolling
retrain.  Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--target-date` | `2026-06-30` | End date for the live rolling retrain. |
| `--config` | `roll_base.yaml` | YAML config template. |
| `--provider-uri` | `~/.qlib/qlib_data/cn_data_mootdx` | Qlib provider URI. |
| `--exp-name` | `mainboard_leftside` | Final combined experiment name. |
| `--rolling-exp` | `mainboard_rolling_models` | Experiment name for individual rolling models. |
| `--horizon` | `1` | Prediction horizon (days). |
| `--step` | `20` | Rolling step (days). |
| `--train-start` | `2020-01-03` | Training start date. |

```bash
python run_roll.py --target-date 2026-06-30 --exp-name mainboard_leftside
```

### Signal extraction CLI

`extract_signal.py` pulls the combined `pred.pkl` from the MLflow experiment
created by `run_roll.py`:

| Option | Default | Description |
|--------|---------|-------------|
| `--exp-name` | `mainboard_leftside` | MLflow experiment name. |
| `--out-path` | `/tmp/mainboard/sig_base.pkl` | Output path for the signal pickle. |
| `--provider-uri` | `~/.qlib/qlib_data/cn_data_mootdx` | Qlib provider URI. |

```bash
python extract_signal.py --exp-name mainboard_leftside \
    --out-path /tmp/mainboard/sig_20260630.pkl
```

### Live pick generation CLI (blend50 dual-model)

`generate_live_picks.py` takes both daily signals and applies:

1. **Signal history store** – today's signals are merged into
   `results/mainboard_leftside/sig_hist/{base,p0}.pkl` (first run seeds from
   `experiments/alpha_enhancement/signals/sig_w5*.pkl`, covering 2024-07 onward).
2. **blend50 fusion** – each signal is cross-sectionally z-scored over today's
   tradable universe (valid close, volume>0, ≥60 days of history), averaged
   50/50, and re-z-scored (same recipe as `experiments/alpha_enhancement/bt_blend.py`).
   Rolling 20-day rank-ICs of both signals (spearman vs the realized 5-day
   open-to-open return, `shift(6)` against look-ahead) are printed for
   alpha-decay monitoring only — they no longer pick a model.
3. **ST / delisting filter (hard, the only filter)** – fetch all candidate names
   from Tencent and drop names containing `ST`, `*ST`, or `退`.  Model scores are
   heavily tied, so the ST check covers the whole cross-section, not just top-N.
   Held stocks that turn ST are force-sold (beyond n_drop).
4. **Official TopkDropout portfolio** – constant topk=10, sell the n_drop=2
   worst-scored holdings, buy the 2 best-scored non-held candidates (same
   semantics as the backtest engine).  Previous holdings are read from the
   latest earlier `picks_live_*.csv` in the output directory.
5. **Rebalance instructions** – prints 卖出/买入/持有 lists with real quote
   prices (Tencent; qlib stores normalized prices) for manual execution at the
   next open.  EPS / niyoy / sentiment columns are review-only, never filters.
   Overnight-order practice (Phase 16 evidence): place a limit order at
   close×1.095 — every open ≤ +9.5% fills at the open price; opens above
   +9.5% that are not sealed boards are historically toxic (5d mean −5.5%,
   win rate 29%), and sealed boards cannot fill anyway.
6. **Sealed/gap-board skip (Phase 20, 2026-10-16)** – in the exec check, a buy
   candidate whose execution open is suspended, a sealed one-word limit-up board,
   or gaps above +9.5% vs the signal-day close is recorded in
   `results/mainboard_leftside/limit_skip.json` (with its paired drop,
   matched by score rank against the previous sells). Rationale: the live buy
   is a limit order at close×1.095, so a >+9.5% gap never fills anyway, and
   such opens are historically toxic (holding-period −2.1% at 34% win vs
   ≤+9.5% gaps +11.2% at 90% win; `experiments/alpha_enhancement/
   diagnose_signalday_limit.py`). On the next run the skipped name is excluded
   from candidates while it stays in the elite zone (score rank ≤ topk;
   released once it decays out — never chase a board open), and the paired
   kept incumbent is restored to the holdings so the swap is cancelled and
   the portfolio stays at topk. Since 2026-10-23 this detection is pre-run
   before stock selection (yesterday's picks vs today's open), so the
   correction takes effect in the same night's picks; only newly recorded
   entries re-add the paired kept name (existing entries never resurrect a
   name that was later rotated out normally). Backtest (bt_swap_rules.py arm G):
   ann 53.2% / Sharpe 1.72 vs 46.9% / 1.51 without the gap skip.

| Option | Default | Description |
|--------|---------|-------------|
| `--sig-base` | (required) | Today's base (pure W5) signal pickle. |
| `--sig-p0` | (required) | Today's P0 (W5+P0) signal pickle. |
| `--hist-dir` | `results/mainboard_leftside/sig_hist` | Persistent signal-history store. |
| `--seed-dir` | `experiments/alpha_enhancement/signals` | Seed signals for the first run. |
| `--out-path` | `results/mainboard_leftside/picks_live.csv` | Output CSV path. |
| `--topk` | `10` | Portfolio size. |
| `--n-drop` | `2` | Max names swapped per day (official semantics). |
| `--date` | latest common signal date | Target date (`YYYY-MM-DD`). |
| `--provider-uri` | `~/.qlib/qlib_data/cn_data_mootdx` | Qlib provider URI. |

```bash
python generate_live_picks.py --date 2026-07-23 \
    --sig-base /tmp/mainboard/sig_w5_base_20260723.pkl \
    --sig-p0 /tmp/mainboard/sig_w5_p0_20260723.pkl \
    --out-path results/mainboard_leftside/picks_live_20260723.csv
```

> Note: the dumped PIT files use field names like `eps_q_q`; the expression must be `P($$eps_q_q)` rather than `P($$eps_q)`.

### Sentiment-data ceiling

Historical Eastmoney individual-stock news sentiment only exists from 2026-01-05
onwards (see the overlay experiment below).  The current production chain uses
sentiment as a review-only info column; it never filters.

## Sentiment overlay experiment (2026-01-05 ~ 2026-06-28)

The Eastmoney individual-stock news API only returns articles from 2026-01-05
onwards, regardless of how far back we request.  Therefore a full rolling retrain
with sentiment features is **not possible** with this data source — the model
would see only NaN sentiment values throughout 2020-2023 training.

As a quick directional test, `sentiment_overlay.py` applies post-hoc sentiment
filters to the existing left-side signal.

| overlay | ann_return | information_ratio | max_drawdown |
|---------|-----------:|------------------:|-------------:|
| baseline_2026 | 0.165 | 0.755 | -0.131 |
| neg_sentiment_lt_-0.1 | 0.182 | **0.818** | -0.131 |
| neg_sentiment_lt_0.0 | 0.182 | **0.818** | -0.131 |
| neg_ratio_gt_0.2 | 0.181 | 0.815 | -0.131 |
| avoid_pos_sentiment_gt_0.7 | 0.180 | 0.807 | -0.135 |
| avoid_pos_sentiment_gt_0.5 | 0.174 | 0.777 | -0.133 |
| has_news_only | 0.130 | 0.691 | -0.107 |
| attention_weighted | 0.141 | 0.648 | -0.131 |
| most_negative_sentiment_quintile | -0.112 | -0.615 | -0.125 |
| avoid_pos_and_require_news | -0.113 | -0.626 | -0.120 |

**Take-away:** mild negative-sentiment filters give a small IR uplift (~+0.06)
on the short 2026 sample, while sparse/extreme filters (quintile, require-news)
destroy performance.  The uplift is marginal and, more importantly, cannot be
validated on a longer history with the current data source.

### Full sentiment-enhanced retrain (blocked by data source)

The integration code (`alpha158_sentiment.py`, `roll_sentiment.yaml`,
`run_roll_sentiment.py`) is ready, but Eastmoney does not provide historical
news before 2026.  To train on sentiment, a different data source with
historical A-share individual-stock news is required.
