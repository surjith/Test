"""5-minute Opening Range Breakout on "Stocks in Play" (Zarattini, Barbon, Aziz 2024).

Rules replicated (as published in the paper's abstract and its public summaries):
  * Universe: US common stocks; open > $5, 14-day avg volume >= 1M shares, 14-day ATR > $0.50.
  * Opening range = first 5-minute bar. Bullish bar -> long only, bearish -> short only, doji -> skip.
  * Entry: stop order at the opening-range high (long) / low (short).
  * Stop loss: 10% of the 14-day ATR away from the entry price. Otherwise exit at the close.
  * Stocks in Play: relative volume = opening-range volume / mean opening-range volume of the
    previous 14 sessions; require >= 100% and trade the 20 highest each day.
  * Sizing: risk 1% of equity per trade, position notional capped at 4x equity / 20 slots.
  * Commission $0.0035 per share.

Usage:
    python backtest.py screen     # write data/screened.json from daily bars
    python backtest.py trades     # simulate every candidate ticker-day on 5-minute bars
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
ATR_N = 14
STOP_ATR = 0.10
MIN_PX, MIN_AVGVOL, MIN_ATR = 5.0, 1_000_000, 0.50
MODES = ("opt", "mid", "pes")


def daily_features(t: str) -> pd.DataFrame | None:
    """Per-day screen inputs, using only information available before that day's open."""
    try:
        d = pd.read_parquet(DATA / "daily" / f"{t}.parquet")
    except FileNotFoundError:
        return None
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(d) < ATR_N + 2:
        return None
    pc = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    f = pd.DataFrame({
        "atr": tr.rolling(ATR_N).mean().shift(1),
        "avgvol": d["Volume"].rolling(ATR_N).mean().shift(1),
        "prev_close": pc,
        "d_open": d["Open"],
    })
    f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
    return f.dropna()


def screen():
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=95)
    keep = []
    for p in sorted((DATA / "daily").glob("*.parquet")):
        f = daily_features(p.stem)
        if f is None:
            continue
        f = f[f.index >= start]
        ok = (f.prev_close > MIN_PX * 0.9) & (f.avgvol >= MIN_AVGVOL) & (f.atr > MIN_ATR)
        if ok.any():
            keep.append(p.stem)
    json.dump(keep, open(DATA / "screened.json", "w"))
    print(len(keep), "tickers pass the liquidity screen on at least one day")


def simulate_day(o, h, l, c, side, level, stop_dist):
    """Simulate one ticker-day after the opening range.

    o/h/l/c are arrays of the bars *after* the opening range. Returns
    (entry_idx, entry_px, {mode: (exit_px, reason)}) or None if the breakout never triggers.

    The three variants differ only on the entry bar, where bar data cannot tell whether the
    adverse extreme happened before or after the fill:
      opt - stopped on the entry bar only if the bar *closes* beyond the stop (upper bound)
      pes - stopped on the entry bar whenever its range touches the stop (lower bound)
      mid - standard OHLC path convention: a bar that closes in the trade's favour is assumed to
            have made its adverse extreme first (-> opt rule), otherwise last (-> pes rule)
    """
    s = side
    trig = np.nonzero(h >= level)[0] if s > 0 else np.nonzero(l <= level)[0]
    if len(trig) == 0:
        return None
    i = trig[0]
    gap_entry = (o[i] >= level) if s > 0 else (o[i] <= level)
    entry = o[i] if gap_entry else level
    stop = entry - s * stop_dist
    adverse = l if s > 0 else h  # the extreme that can hit the stop

    def hit(x):  # price x at or beyond the stop
        return (x <= stop) if s > 0 else (x >= stop)

    res = {}
    favourable_bar = s * (c[i] - o[i]) >= 0
    for mode in MODES:
        exit_px, reason = None, None
        # entry bar
        if gap_entry or mode == "pes" or (mode == "mid" and not favourable_bar):
            if hit(adverse[i]):
                exit_px, reason = stop, "stop"
        else:
            if hit(c[i]):
                exit_px, reason = stop, "stop"
        if exit_px is None:
            for j in range(i + 1, len(o)):
                if hit(o[j]):
                    exit_px, reason = o[j], "stop_gap"
                    break
                if hit(adverse[j]):
                    exit_px, reason = stop, "stop"
                    break
        if exit_px is None:
            exit_px, reason = c[-1], "eod"
        res[mode] = (exit_px, reason)
    return i, entry, res


def load_intraday(t: str, folder="m5") -> pd.DataFrame | None:
    try:
        x = pd.read_parquet(DATA / folder / f"{t}.parquet")
    except FileNotFoundError:
        return None
    x = x[["Open", "High", "Low", "Close", "Volume"]].dropna()
    x.index = pd.to_datetime(x.index).tz_convert("America/New_York").tz_localize(None)
    x = x[(x.index.time >= pd.Timestamp("09:30").time()) & (x.index.time < pd.Timestamp("16:00").time())]
    return x


def trades():
    """Simulate every screened ticker-day; selection of the top 20 happens in analyze.py."""
    OUT.mkdir(exist_ok=True)
    tickers = json.load(open(DATA / "screened.json"))
    rows = []
    for n, t in enumerate(tickers):
        f = daily_features(t)
        x = load_intraday(t)
        if f is None or x is None or x.empty:
            continue
        days = x.index.normalize()
        first = x[x.index.time == pd.Timestamp("09:30").time()]
        orv = pd.Series(first["Volume"].values, index=first.index.normalize())
        # mean opening-range volume of the previous 14 sessions (no look-ahead)
        orv_avg = orv.rolling(ATR_N).mean().shift(1)
        for day, g in x.groupby(days):
            if day not in f.index or day not in orv_avg.index:
                continue
            if g.index[0].time() != pd.Timestamp("09:30").time() or len(g) < 40:
                continue
            atr, avgvol = f.at[day, "atr"], f.at[day, "avgvol"]
            avg_orv = orv_avg.at[day]
            o, h, l, c, v = (g[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
            if not (o[0] > MIN_PX and avgvol >= MIN_AVGVOL and atr > MIN_ATR):
                continue
            if not np.isfinite(avg_orv) or avg_orv <= 0 or v[0] <= 0:
                continue
            side = int(np.sign(c[0] - o[0]))
            row = dict(date=day, ticker=t, side=side, relvol=v[0] / avg_orv, atr=atr, avgvol=avgvol,
                       or_open=o[0], or_high=h[0], or_low=l[0], or_close=c[0], or_vol=v[0],
                       day_close=c[-1], traded=False)
            if side != 0:
                level = h[0] if side > 0 else l[0]
                r = simulate_day(o[1:], h[1:], l[1:], c[1:], side, level, STOP_ATR * atr)
                if r is not None:
                    i, entry, res = r
                    row.update(traded=True, entry_time=g.index[1 + i], entry=entry)
                    for m, (px, why) in res.items():
                        row[f"exit_{m}"], row[f"why_{m}"] = px, why
            rows.append(row)
        if n % 200 == 0:
            print(n, len(tickers), len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "candidates.parquet")
    print(len(df), "candidate ticker-days;", int(df.traded.sum()), "breakouts;",
          df.date.nunique(), "days", df.date.min().date(), "->", df.date.max().date())


if __name__ == "__main__":
    {"screen": screen, "trades": trades}[sys.argv[1]]()
