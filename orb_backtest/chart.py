"""Render results/equity.png from results/equity_curves.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).parent / "results"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

c = pd.read_csv(OUT / "equity_curves.csv", index_col=0, parse_dates=True)
base = "paper (commission only)"
panels = [("Full window, 5-minute bars", f"5m opt | {base}", f"5m pes | {base}"),
          ("Last 21 sessions, 1-minute bars", f"1m opt | {base}", f"1m pes | {base}")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE,
                         gridspec_kw={"width_ratios": [46, 21], "wspace": 0.36})
for ax, (title, hi, lo) in zip(axes, panels):
    d = c[[hi, lo, "SPY"]].dropna(subset=[hi])
    spy_ret = c["SPY"].pct_change().fillna(c["SPY"].iloc[0] - 1).reindex(d.index)
    d = d.assign(SPY=(1 + spy_ret).cumprod())
    start = d.index[0] - pd.tseries.offsets.BDay(1)
    d = pd.concat([pd.DataFrame(1.0, index=[start], columns=d.columns), d]) * 100
    ax.set_facecolor(SURFACE)
    ax.fill_between(d.index, d[lo], d[hi], color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(d.index, d[hi], color=BLUE, lw=2)
    ax.plot(d.index, d[lo], color=BLUE, lw=2, ls=(0, (4, 2)))
    ax.plot(d.index, d["SPY"], color=ORANGE, lw=2)
    ax.axhline(100, color=AXIS, lw=1, zorder=0)
    # end labels: value only (the legend carries identity), pushed apart so they never overlap
    ends = sorted(((d[col].iloc[-1], name) for col, name in ((hi, "Best case"), (lo, "Worst case"), ("SPY", "SPY"))),
                  reverse=True)
    ys = [e[0] for e in ends]
    for k in range(1, len(ys)):
        ys[k] = min(ys[k], ys[k - 1] - 2.8)
    for (val, name), y in zip(ends, ys):
        ax.annotate(f"{name} {val - 100:+.1f}%", (d.index[-1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2, annotation_clip=False)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
axes[0].set_ylabel("Equity, start = 100", color=INK2, fontsize=9)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="ORB top-20, best-case fills"),
                    Line2D([], [], color=BLUE, lw=2, ls=(0, (4, 2)), label="ORB top-20, worst-case fills"),
                    Line2D([], [], color=ORANGE, lw=2, label="SPY buy & hold")],
           loc="upper left", bbox_to_anchor=(0.055, 0.925), ncol=3, frameon=False, fontsize=9, labelcolor=INK2,
           handlelength=2.6, columnspacing=1.8)
fig.suptitle("5-minute ORB on the top-20 Stocks in Play vs SPY, Sep 17 – Nov 19, 2026",
             x=0.06, ha="left", fontsize=12.5, color=INK, fontweight="bold")
fig.text(0.06, 0.02, "Commission $0.0035/share, no slippage. Shaded band = range between best- and worst-case "
         "assumptions about fills inside the entry bar. Data: Yahoo Finance.", fontsize=8, color=MUTED)
fig.subplots_adjust(left=0.06, right=0.88, top=0.78, bottom=0.14)
fig.savefig(OUT / "equity.png", dpi=200, facecolor=SURFACE)
