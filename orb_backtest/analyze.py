"""Select the daily top-20 Stocks in Play, build the portfolio and compute statistics.

Usage:
    python analyze.py select    # write results/selected.parquet (the daily top-20 by relative volume)
    python analyze.py report    # portfolio stats, sensitivities, results/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import DATA, MODES, OUT, STOP_ATR, load_intraday, simulate_day

TOP_N = 20
RISK = 0.01          # fraction of equity risked per trade
MAX_LEV = 4.0        # portfolio leverage cap -> each of the 20 slots gets at most MAX_LEV/TOP_N of equity
COMM = 0.0035        # $ per share, each side
START_EQUITY = 25_000


def candidates() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "candidates.parquet")
    df["stop_dist"] = STOP_ATR * df.atr
    return df


def select(df: pd.DataFrame, n=TOP_N) -> pd.DataFrame:
    c = df[(df.relvol >= 1.0) & (df.side != 0)].copy()
    c["rank"] = c.groupby("date").relvol.rank(ascending=False, method="first")
    return c[c["rank"] <= n].sort_values(["date", "rank"])


def per_share(t: pd.DataFrame, mode: str, slip_cents=0.0, slip_bps=0.0) -> pd.Series:
    """Net P&L per share. Slippage is charged on the stop-entry and on stop-outs (both are
    market orders fired through a moving price); the closing exit is charged commission only."""
    gross = t.side * (t[f"exit_{mode}"] - t.entry)
    slip = slip_cents / 100 + slip_bps / 1e4 * t.entry
    stopped = t[f"why_{mode}"].str.startswith("stop")
    return gross - 2 * COMM - slip * (1 + stopped.astype(float))


def portfolio(t: pd.DataFrame, mode: str, **slip) -> pd.DataFrame:
    """Daily compounding: every position is sized off the equity at that day's open."""
    t = t[t.traded].copy()
    t["pps"] = per_share(t, mode, **slip)
    eq, out = START_EQUITY, []
    for day, g in t.groupby("date"):
        # stop_dist is infinite in the no-stop variant: then only the leverage cap sizes the trade
        risk_shares = np.where(np.isfinite(g.stop_dist), RISK * eq / g.stop_dist, np.inf)
        shares = np.floor(np.minimum(risk_shares, MAX_LEV / TOP_N * eq / g.entry))
        pnl = float((shares * g.pps).sum())
        out.append(dict(date=day, pnl=pnl, ret=pnl / eq, n=len(g),
                        gross_lev=float((shares * g.entry).sum() / eq)))
        eq += pnl
    return pd.DataFrame(out).set_index("date")


def stats(daily: pd.Series, bench: pd.Series | None = None) -> dict:
    eq = (1 + daily).cumprod()
    n = len(daily)
    s = dict(days=n, total_return=float(eq.iloc[-1] - 1),
             ann_return=float(eq.iloc[-1] ** (252 / n) - 1),
             ann_vol=float(daily.std() * np.sqrt(252)),
             sharpe=float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else float("nan"),
             max_dd=float((eq / eq.cummax().clip(lower=1) - 1).min()),
             win_days=float((daily > 0).mean()),
             t_stat=float(daily.mean() / daily.std() * np.sqrt(n)) if daily.std() > 0 else float("nan"))
    if bench is not None:
        b = bench.reindex(daily.index).fillna(0)
        beta = float(np.cov(daily, b)[0, 1] / b.var())
        s.update(beta=beta, ann_alpha=float((daily.mean() - beta * b.mean()) * 252),
                 corr=float(np.corrcoef(daily, b)[0, 1]))
    return s


def trade_stats(t: pd.DataFrame, mode: str, **slip) -> dict:
    t = t[t.traded]
    pps = per_share(t, mode, **slip)
    r = pps / t.stop_dist
    return dict(trades=int(len(t)), hit_rate=float((pps > 0).mean()), avg_R=float(r.mean()),
                median_R=float(r.median()), avg_win_R=float(r[pps > 0].mean()),
                avg_loss_R=float(r[pps <= 0].mean()),
                profit_factor=float(pps[pps > 0].sum() / -pps[pps <= 0].sum()),
                avg_bps=float((pps / t.entry).mean() * 1e4),
                stopped=float(t[f"why_{mode}"].str.startswith("stop").mean()),
                long_share=float((t.side > 0).mean()))


def spy_daily() -> pd.Series:
    d = pd.read_parquet(DATA / "daily" / "SPY.parquet")["Close"]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d.pct_change()


_cache: dict = {}


def bars(t: str, folder: str) -> pd.DataFrame | None:
    if (t, folder) not in _cache:
        if folder == "m5":
            x = load_intraday(t)
        else:
            parts = [load_intraday(p.stem, "m1") for p in sorted((DATA / "m1").glob(f"{t}_*.parquet"))]
            parts = [p for p in parts if p is not None and len(p)]
            x = pd.concat(parts).sort_index() if parts else None
            if x is not None:
                x = x[~x.index.duplicated()]
        _cache[(t, folder)] = x
    return _cache[(t, folder)]


def resim(sel: pd.DataFrame, folder="m5", stop_mult=STOP_ATR) -> pd.DataFrame:
    """Re-run the selected ticker-days on 5-minute or 1-minute bars with a given stop width.

    Selection (ticker, side, ATR) always comes from the 5-minute run. On 1-minute bars the
    breakout level is rebuilt from the five bars 09:30-09:34 and the trade is walked minute by
    minute (Yahoo keeps only ~30 days of 1-minute history). stop_mult=inf means no stop.
    """
    or_end = pd.Timestamp("09:35").time()
    min_bars = 40 if folder == "m5" else 300
    rows = []
    for t, g in sel.groupby("ticker"):
        x = bars(t, folder)
        if x is None:
            continue
        for _, row in g.iterrows():
            d = x[x.index.normalize() == row.date]
            if len(d) < min_bars or d.index[0].time() != pd.Timestamp("09:30").time():
                continue
            orb, rest = d[d.index.time < or_end], d[d.index.time >= or_end]
            side = int(row.side)
            level = orb.High.max() if side > 0 else orb.Low.min()
            new = row.to_dict()
            new.update(stop_dist=stop_mult * row.atr, level=level)
            r = simulate_day(rest.Open.values, rest.High.values, rest.Low.values, rest.Close.values,
                             side, level, new["stop_dist"])
            new["traded"] = r is not None
            if r is not None:
                i, entry, res = r
                new.update(entry_time=rest.index[i], entry=entry)
                for m, (px, why) in res.items():
                    new[f"exit_{m}"], new[f"why_{m}"] = px, why
            rows.append(new)
    return pd.DataFrame(rows)


STOPS = [0.05, 0.10, 0.25, 0.50, 1.00, np.inf]


def stop_sensitivity(sel: pd.DataFrame, folder: str, spy: pd.Series) -> list[dict]:
    out = []
    for k in STOPS:
        r = resim(sel, folder, k)
        rec = dict(stop="none" if np.isinf(k) else f"{k:.0%} ATR")
        for m in MODES:
            p = portfolio(r, m)
            ts = trade_stats(r, m)
            rec[m] = dict(total_return=float((1 + p.ret).prod() - 1), t_stat=stats(p.ret)["t_stat"],
                          hit_rate=ts["hit_rate"], avg_bps=ts["avg_bps"], stopped=ts["stopped"])
        out.append(rec)
    return out


def main_select():
    sel = select(candidates())
    sel.to_parquet(OUT / "selected.parquet")
    print(len(sel), "selections on", sel.date.nunique(), "days;", int(sel.traded.sum()), "traded")


SLIPS = {
    "paper (commission only)": {},
    "+1c/share slippage": dict(slip_cents=1.0),
    "+2 bps slippage": dict(slip_bps=2.0),
    "+5 bps slippage": dict(slip_bps=5.0),
    "+10 bps slippage": dict(slip_bps=10.0),
}


def main_report():
    df = candidates()
    sel = select(df)
    spy = spy_daily()
    base = "paper (commission only)"
    S = {"period": [str(df.date.min().date()), str(df.date.max().date())],
         "days": int(df.date.nunique()), "universe_tickers": int(df.ticker.nunique()),
         "candidate_ticker_days": int(len(df)), "selections": int(len(sel)),
         "median_stop_bps": float((sel.stop_dist / sel.or_open * 1e4).median()),
         "median_or_range_over_stop": float(((sel.or_high - sel.or_low) / sel.stop_dist).median()),
         "median_relvol": float(sel.relvol.median()), "median_price": float(sel.or_open.median())}

    # 1. headline portfolio under the three intrabar conventions and slippage scenarios
    S["portfolio"], curves = {}, {}
    for mode in MODES:
        S["portfolio"][mode] = {}
        for name, slip in SLIPS.items():
            p = portfolio(sel, mode, **slip)
            S["portfolio"][mode][name] = {**stats(p.ret, spy), **trade_stats(sel, mode, **slip),
                                          "avg_gross_lev": float(p.gross_lev.mean())}
            if name in (base, "+1c/share slippage"):
                curves[f"5m {mode} | {name}"] = (1 + p.ret).cumprod()
    days = sorted(df.date.unique())
    b = spy.reindex(days).fillna(0)
    S["spy"] = stats(b)
    curves["SPY"] = (1 + b).cumprod()

    # 2. does relative volume sort outcomes? every breakout in the screened universe
    allc = df[df.traded & (df.side != 0)].copy()
    bins = [0, 0.5, 1, 1.5, 2, 3, 5, 10, np.inf]
    allc["bucket"] = pd.cut(allc.relvol, bins)
    S["relvol_buckets"] = []
    for k, g in allc.groupby("bucket", observed=True):
        rec = dict(bucket=str(k), n=int(len(g)))
        for m in MODES:
            pps = per_share(g, m)
            rec[f"R_{m}"] = float((pps / g.stop_dist).mean())
            rec[f"bps_{m}"] = float((pps / g.entry).mean() * 1e4)
        S["relvol_buckets"].append(rec)
    S["all_breakouts"] = {m: trade_stats(df[df.side != 0], m) for m in MODES}

    # 3. is the top-20 pick better than 20 random screened names per day? (same rules/sizing)
    rng = np.random.default_rng(0)
    groups = [g for _, g in df[df.side != 0].groupby("date")]
    rand = {m: [] for m in MODES}
    for _ in range(300):
        pick = pd.concat([g.sample(min(TOP_N, len(g)), random_state=int(rng.integers(1 << 31))) for g in groups])
        for m in MODES:
            rand[m].append(float((1 + portfolio(pick, m).ret).prod() - 1))
    S["random20"] = {m: dict(mean=float(np.mean(v)), p05=float(np.percentile(v, 5)),
                             p95=float(np.percentile(v, 95)), top20=S["portfolio"][m][base]["total_return"],
                             pct_beaten=float(np.mean(np.array(v) < S["portfolio"][m][base]["total_return"])))
                     for m, v in rand.items()}

    S["by_side"] = {m: {("long" if s > 0 else "short"): trade_stats(sel[sel.side == s], m) for s in (1, -1)}
                    for m in MODES}
    S["stop_sensitivity_5m"] = stop_sensitivity(sel, "m5", spy)

    # 4. 1-minute rerun for the recent window
    if (DATA / "m1").exists():
        m1 = resim(sel, "m1")
        m1.to_parquet(OUT / "selected_1m.parquet")
        same = sel.merge(m1[["date", "ticker"]], on=["date", "ticker"])
        S["m1"] = {"period": [str(m1.date.min().date()), str(m1.date.max().date())],
                   "days": int(m1.date.nunique()), "selections": int(len(m1)), "portfolio": {},
                   "same_days_5m": {}}
        lvl5 = np.where(m1.side > 0, m1.or_high, m1.or_low)
        S["m1"]["level_matches_5m"] = float((np.abs(m1.level - lvl5) <= 0.011).mean())
        t1 = m1[m1.traded]
        S["m1"]["entry_bar_touches_stop"] = float((t1.why_pes.str.startswith("stop") &
                                                   (t1.exit_pes == t1.entry - t1.side * t1.stop_dist)).mean())
        for mode in MODES:
            S["m1"]["portfolio"][mode] = {}
            for name, slip in SLIPS.items():
                p = portfolio(m1, mode, **slip)
                S["m1"]["portfolio"][mode][name] = {**stats(p.ret, spy), **trade_stats(m1, mode, **slip)}
                if name in (base, "+1c/share slippage"):
                    curves[f"1m {mode} | {name}"] = (1 + p.ret).cumprod()
            p = portfolio(same, mode)
            S["m1"]["same_days_5m"][mode] = {**stats(p.ret, spy), **trade_stats(same, mode)}
        S["stop_sensitivity_1m"] = stop_sensitivity(sel.merge(m1[["date", "ticker"]], on=["date", "ticker"]), "m1", spy)
        m1[["date", "ticker", "rank", "relvol", "side", "atr", "traded", "entry_time", "entry"] +
           [f"{a}_{m}" for m in MODES for a in ("exit", "why")]].to_csv(OUT / "trades_top20_1min.csv", index=False)

    pd.DataFrame(curves).to_csv(OUT / "equity_curves.csv")
    cols = ["date", "ticker", "rank", "relvol", "side", "atr", "or_high", "or_low", "traded", "entry_time",
            "entry"] + [f"{a}_{m}" for m in MODES for a in ("exit", "why")]
    sel[cols].to_csv(OUT / "trades_top20.csv", index=False)
    json.dump(S, open(OUT / "summary.json", "w"), indent=1)

    show = ["total_return", "sharpe", "max_dd", "t_stat", "beta", "trades", "hit_rate", "avg_R", "profit_factor",
            "avg_bps", "stopped"]

    def table(d):
        return pd.DataFrame({(m, n): v for m, x in d.items() for n, v in x.items()}).T[show].round(3).to_string()

    print({k: v for k, v in S.items() if not isinstance(v, (dict, list)) or k == "period"})
    print(table(S["portfolio"]))
    print("SPY", {k: round(v, 3) for k, v in S["spy"].items()})
    print(pd.DataFrame(S["relvol_buckets"]).round(2).to_string())
    print("all breakouts", pd.DataFrame(S["all_breakouts"]).T.round(3).to_string())
    print("random20", pd.DataFrame(S["random20"]).T.round(3).to_string())
    for m in MODES:
        print("side", m, pd.DataFrame(S["by_side"][m]).T[["trades", "hit_rate", "avg_R", "avg_bps"]].round(3).to_string())

    def sens(rows):
        return pd.DataFrame({(r["stop"], m): r[m] for r in rows for m in MODES}).T.round(3).to_string()

    print("stop sensitivity, 5m bars\n", sens(S["stop_sensitivity_5m"]))
    if "m1" in S:
        print("1-minute rerun", S["m1"]["period"], S["m1"]["days"], "days; entry bar touches stop:",
              round(S["m1"]["entry_bar_touches_stop"], 3), "; breakout level equals 5m level:",
              round(S["m1"]["level_matches_5m"], 3))
        print(table(S["m1"]["portfolio"]))
        print("same days, 5m bars\n", pd.DataFrame(S["m1"]["same_days_5m"]).T[
            ["total_return", "trades", "hit_rate", "avg_R", "avg_bps"]].round(3).to_string())
        print("stop sensitivity, 1m bars\n", sens(S["stop_sensitivity_1m"]))


if __name__ == "__main__":
    {"select": main_select, "report": main_report}[sys.argv[1]]()
