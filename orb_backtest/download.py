"""Download daily and intraday bars from Yahoo Finance for the ORB backtest.

Usage:
    python download.py symbols    # NASDAQ/NYSE/AMEX listings (snapshot from the rreichel3/US-Stock-Symbols repo)
    python download.py daily      # 1y of daily bars for the pre-filtered universe
    python download.py intraday   # 60d of 5-minute bars for stocks passing the liquidity screen
    python download.py m1         # 1-minute bars (Yahoo keeps ~30 days) for the selected top-20 names
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"


def universe() -> list[str]:
    rows = []
    for ex in ("nasdaq", "nyse", "amex"):
        rows += json.load(open(DATA / f"{ex}_full.json"))
    df = pd.DataFrame(rows)
    df["px"] = pd.to_numeric(df["lastsale"].str.replace("[$,]", "", regex=True), errors="coerce")
    df["vol"] = pd.to_numeric(df["volume"], errors="coerce")
    # common-stock style symbols only (drops preferreds / units / warrants with ^ or /)
    df = df[df["symbol"].str.fullmatch(r"[A-Z]{1,5}")]
    # loose pre-filter; the real screen (price>5, 14d avg vol>=1M, ATR>0.5) is applied per day later
    df = df[(df.px > 4) & (df.vol > 150_000)]
    return sorted(df["symbol"].unique())


def fetch(tickers, out: Path, **kw):
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not (out / f"{t}.parquet").exists()]
    print(f"{len(tickers)} tickers, {len(todo)} to download", flush=True)
    B = 40
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        for attempt in range(4):
            try:
                d = yf.download(batch, group_by="ticker", auto_adjust=False, progress=False,
                                threads=8, **kw)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, e, flush=True)
                time.sleep(20 * (attempt + 1))
        else:
            continue
        ok = 0
        for t in batch:
            try:
                x = d[t].dropna(how="all")
            except KeyError:
                continue
            if len(x):
                x.to_parquet(out / f"{t}.parquet")
                ok += 1
        print(f"{i + len(batch)}/{len(todo)} ok={ok}/{len(batch)}", flush=True)
        if ok < len(batch) * 0.5:
            time.sleep(30)  # likely rate-limited; back off
        else:
            time.sleep(1)


def fetch_m1():
    """1-minute bars for the ticker-days the strategy actually selected (last ~30 days only)."""
    out = DATA / "m1"
    out.mkdir(exist_ok=True)
    sel = pd.read_parquet(Path(__file__).parent / "results" / "selected.parquet")
    today = pd.Timestamp.today().normalize()
    first = today - pd.Timedelta(days=28)
    sel = sel[sel.date >= first]
    windows = [(first + pd.Timedelta(days=7 * k), first + pd.Timedelta(days=7 * (k + 1))) for k in range(5)]
    for k, (a, b) in enumerate(windows):
        tick = sorted(sel[(sel.date >= a) & (sel.date < b)].ticker.unique())
        tick = [t for t in tick if not (out / f"{t}_{k}.parquet").exists()]
        print("window", a.date(), b.date(), len(tick), flush=True)
        for i in range(0, len(tick), 40):
            batch = tick[i:i + 40]
            d = yf.download(batch, group_by="ticker", auto_adjust=False, progress=False, threads=8,
                            start=a.strftime("%Y-%m-%d"), end=min(b, today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            interval="1m")
            for t in batch:
                try:
                    x = d[t].dropna(how="all")
                except KeyError:
                    continue
                if len(x):
                    x.to_parquet(out / f"{t}_{k}.parquet")
            time.sleep(1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "symbols":
        DATA.mkdir(exist_ok=True)
        for ex in ("nasdaq", "nyse", "amex"):
            url = f"https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/{ex}/{ex}_full_tickers.json"
            (DATA / f"{ex}_full.json").write_bytes(requests.get(url, timeout=60).content)
    elif mode == "daily":
        fetch(universe() + ["SPY"], DATA / "daily", period="1y", interval="1d")
    elif mode == "intraday":
        keep = json.load(open(DATA / "screened.json"))
        fetch(keep + ["SPY"], DATA / "m5", period="60d", interval="5m")
    elif mode == "m1":
        fetch_m1()
