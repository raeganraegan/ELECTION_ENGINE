"""plots.py -- every figure ENKI draws. Reads finished results; never fits or simulates."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from adjustText import adjust_text
except ImportError:                                   # labels just overlap a little more without it
    def adjust_text(*args, **kwargs):
        return None

PARTY_COLORS = {"ALP": "#E13940", "LNP": "#0047AB", "LIB": "#0047AB", "NAT": "#0066CC",
                "GRN": "#104C34", "ONP": "#FF8C00", "OTHER": "#708090"}


def _finish(path, dpi) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close()


def seat_distribution_plot(res, target_party: str, output_path, dpi: int = 300) -> None:
    """Density of every party's seat-wide primary share; worst / average / best markers for `target_party` only."""
    draws = res.draws_df
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in draws.columns:
        is_target = col == target_party
        sns.kdeplot(data=draws[col].dropna(), label=col, color=PARTY_COLORS.get(col.upper(), "#333333"),
                    linewidth=2.8 if is_target else 1.8, ax=ax, fill=True, alpha=0.18 if is_target else 0.05)
    color = PARTY_COLORS.get(target_party.upper(), "#333333")
    p05, p50, p95 = np.percentile(draws[target_party].dropna(), [5, 50, 95])
    top = ax.get_ylim()[1]
    for v, name, y, ha, dx in ((p05, "Worst (0.05)", 0.70, "right", -0.4), (p50, "Average (0.50)", 0.90, "center", 0.0), (p95, "Best (0.95)", 0.70, "left", 0.4)):
        ax.vlines(v, 0, top * (y - 0.03), colors=color, linestyles="dotted", linewidth=2)
        ax.text(v + dx, top * y, f"{target_party} {name}\n{v:.1f}%", ha=ha, va="bottom", fontsize=9, fontweight="bold", color=color)
    ax.set_title("ENKI Multi-Party Primary Vote Outcome Distributions", fontsize=15, fontweight="bold", pad=25)
    ax.set_xlabel("Projected Primary Vote Share (%)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_xlim(0, 70)
    ticks = [t for t in ax.get_xticks() if 0 <= t <= 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}%" for t in ticks])
    ax.yaxis.set_major_formatter("{x:.2f}")
    ax.legend(title="Party", frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    sns.despine(top=True, right=True)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    plt.tight_layout()
    _finish(output_path, dpi)


def booth_ridgeline_plot(res, target_party: str, top_n_booths: int, output_path, max_draws: int = 1500, dpi: int = 300,
                         special_keywords: Sequence[str] = ("PRE-POLL", "PREPOLL", "DECLARATION", "ABSENT", "POSTAL")) -> None:
    """Ridgeline of the model's actual booth-level draws. Pre-poll style booths are always shown; the rest are the largest by volume."""
    draws, fc = res.booth_draws_df, res.booth_forecast
    if draws is None or draws.empty:
        raise ValueError("No booth-level draws on the result.")
    if len(draws) > max_draws:
        draws = draws.sample(max_draws, random_state=0)
    special = [b for b in draws.columns if any(k in str(b).upper() for k in special_keywords)]
    regular = fc[~fc["booth"].isin(special)].sort_values("formal_weight", ascending=False)["booth"].tolist()
    order = sorted(special + regular[: max(5, top_n_booths - len(special))], key=lambda b: draws[b].mean())
    long = draws[order].melt(var_name="booth", value_name="vote_share")
    long["booth"] = pd.Categorical(long["booth"], categories=order, ordered=True)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(long, row="booth", hue="booth", aspect=12, height=0.55, palette=sns.color_palette("plasma", n_colors=len(order)))
    g.map(sns.kdeplot, "vote_share", clip_on=False, fill=True, alpha=0.7, lw=1.5, bw_adjust=0.6)
    g.map(sns.kdeplot, "vote_share", clip_on=False, color="black", lw=1, bw_adjust=0.6)
    g.map(plt.axhline, y=0, lw=1.5, color="black", clip_on=False)
    axes = list(g.axes.flatten())
    g.map(lambda x, color, label: plt.gca().text(-0.03, 0.2, str(order[axes.index(plt.gca())]), fontweight="bold", color="black",
                                                 ha="right", va="center", transform=plt.gca().transAxes, fontsize=8.5), "vote_share")
    g.figure.subplots_adjust(hspace=-0.35)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.axes[-1, 0].set_xlim(max(0.0, long["vote_share"].min() - 1.0), min(100.0, long["vote_share"].max() + 1.0))
    g.axes[-1, 0].set_xlabel(f"Forecasted {target_party} Primary Vote Share (%)", fontsize=11, fontweight="bold", labelpad=10)
    g.figure.suptitle(f"ENKI Monte Carlo {target_party} Vote Share Distribution (model draws)", fontsize=13, fontweight="bold", y=0.98)
    g.figure.subplots_adjust(left=0.32, right=0.95, top=0.92, bottom=0.08)
    _finish(output_path, dpi)


def booth_volume_ridgeline(res, output_path, top_n: int = 15, log_scale: bool = True, dpi: int = 300) -> None:
    """Forecast formal votes per booth (largest first) against the previous election's volume."""
    names, V, hist = res.booth_names, res.volume_matrix, res.prepared_df
    last = hist[hist["year"] == hist["year"].max()].set_index("booth")["n"].to_dict()
    top = np.argsort(V.mean(axis=0))[::-1][:top_n]
    fig, axes = plt.subplots(len(top), 1, figsize=(12, len(top) * 0.45), sharex=True)
    axes = [axes] if len(top) == 1 else list(axes)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top)))
    f = (lambda v: np.log10(v + 1)) if log_scale else (lambda v: v)
    for k, (ax, i) in enumerate(zip(axes, top)):
        sns.kdeplot(f(V[:, i]), ax=ax, fill=True, color=colors[k], alpha=0.6, linewidth=1.5)
        prev = pd.to_numeric(last.get(names[i], np.nan), errors="coerce")
        if not np.isnan(prev) and prev > 0:
            ax.axvline(f(prev), color="red", linestyle="--", linewidth=1.5, ymin=0.1, ymax=0.9, label="Previous Election Volume" if k == 0 else "")
        ax.set_facecolor((0, 0, 0, 0))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.text(-0.03, 0.2, str(names[i]), transform=ax.transAxes, ha="right", va="center", fontsize=9, fontweight="bold", color="black")
        ax.spines["bottom"].set_visible(k == len(top) - 1)
    axes[-1].set_xlabel("Log10(Forecasted Voters + 1)" if log_scale else "Forecasted Number of Voters ($n$)", fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    plt.suptitle("ENKI Monte Carlo Booth Volume Distribution vs. Previous Election" + (" (Log Scale)" if log_scale else ""), y=0.98, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0.22, 0, 0.85, 0.96])
    _finish(output_path, dpi)


def booth_swing_scatter(res, output_path, label_all: bool = True, dpi: int = 300) -> None:
    """Expected booth volume vs projected swing from the booth's own LAST-election share."""
    fc = res.booth_forecast.dropna(subset=["last_share"]).copy()
    fc["swing"] = fc["mean"] - fc["last_share"]
    fc["n_bt"] = fc["formal_weight"].astype(float)
    fc["abs_swing"] = fc["swing"].abs()
    lim = max(5.0, fc["abs_swing"].max() * 1.35)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(data=fc, x="n_bt", y="swing", size="n_bt", sizes=(40, 350), color="black", alpha=0.75, legend=False, ax=ax)
    ax.set_xscale("log")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    shown = fc if label_all else fc.sort_values("abs_swing", ascending=False).head(5)
    texts = [ax.text(r["n_bt"], r["swing"], str(r["booth"]), fontsize=8, fontweight="bold", color="black") for _, r in shown.iterrows()]
    adjust_text(texts, x=shown["n_bt"].values, y=shown["swing"].values, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7))
    ax.set_title("ENKI Expected Booth Size vs. Projected Swing from the Booth's Last Election Share", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Expected Formal Votes at Next Election (Log Scale)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Projected Mean Share minus Last Election Share (points)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylim(-lim, lim)
    ax.yaxis.set_major_formatter("{x:.1f}")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    _finish(output_path, dpi)


_BACKTEST_COLUMNS = {   # column -> header. Model and baselines are compared on the SAME (common) booths.
    "target": "target", "seat_pred_median": "seat pred\nmedian", "seat_actual": "seat\nactual", "seat_error": "MODEL\nseat err (all)",
    "model_seat_err_common": "MODEL\nseat err", "nochange_seat_err": "NO-CHANGE\nseat err", "statewide_swing_seat_err": "STATE-SWING\nseat err",
    "state_mode": "statewide\ninput", "n_polls": "polls\nused", "seat_in_interval": "in 90%\ninterval", "seat_width": "90% band\nwidth", "state_error": "poll err\n(pts)", "model_MAE_common": "MODEL\nbooth MAE", "nochange_MAE": "NO-CHANGE\nbooth MAE",
    "statewide_swing_MAE": "STATE-SWING\nbooth MAE", "booth_CRPS": "MODEL\nbooth CRPS", "booth_cover_50": "50% cover\n(target 50)",
    "booth_cover_90": "90% cover\n(target 90)", "booth_cover_95": "95% cover\n(target 95)"}


def backtest_table_png(summary: pd.DataFrame, output_path, dpi: int = 300) -> None:
    if summary is None or summary.empty:
        return
    cols = [c for c in _BACKTEST_COLUMNS if c in summary.columns]
    d = summary[cols].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(2)
    fig, ax = plt.subplots(figsize=(max(14, len(cols) * 1.6), len(d) * 0.8 + 2.5))
    ax.axis("off")
    table = ax.table(cellText=d.values, colLabels=[_BACKTEST_COLUMNS[c] for c in cols], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.2, 1.8)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if r % 2 == 0 else "white")
    plt.suptitle("ENKI Backtest Error & Calibration Diagnostics", fontsize=14, fontweight="bold", y=0.92)
    plt.tight_layout()
    _finish(output_path, dpi)


def backtest_multiparty_table_png(summary: pd.DataFrame, parties: Sequence[str], output_path, dpi: int = 300) -> None:
    """Render distinct tables (one per party) plus aggregate metrics onto ONE singular PNG image using matplotlib."""
    if summary is None or summary.empty:
        return
    n_parties = len(parties)
    fig, axes = plt.subplots(n_parties, 1, figsize=(14, 3 * n_parties + 2), dpi=dpi)
    if n_parties == 1:
        axes = [axes]

    fig.suptitle("ENKI Multi-Party Backtest Performance & Opponent Breakdown", fontsize=16, fontweight='bold', y=0.98)

    for i, party in enumerate(parties):
        ax = axes[i]
        ax.axis('off')

        rows_data = []
        for _, r in summary.iterrows():
            tgt = int(r["target"])
            pred_col = f"seat_err_{party}"
            err_val = r.get(pred_col, np.nan)
            rows_data.append([
                tgt,
                f"{r.get('state_mode', 'N/A')}",
                f"{r.get('n_booths', 0)}",
                f"{r.get('booth_MAE', 0.0):.2f}",
                f"{err_val:+.2f}" if not np.isnan(err_val) else "N/A",
                f"{r.get('multi_party_MAE', 0.0):.2f}",
                f"{r.get('divergences', 0)}",
                f"{r.get('rhat_max', 1.0):.3f}"
            ])

        col_labels = ["Target Year", "State Mode", "Booths Scored", "Booth MAE", f"{party} Seat Err (pts)", "Aggregate MAE", "MCMC Div", "Max R^hat"]
        
        table = ax.table(cellText=rows_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.5)

        for col_idx in range(len(col_labels)):
            cell = table[(0, col_idx)]
            cell.set_facecolor('#1F4E78')
            cell.set_text_props(color='white', weight='bold')

        ax.set_title(f"Opponent / Party: {party} Breakdown", fontsize=11, fontweight='bold', loc='left', pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _finish(output_path, dpi)


def backtest_distribution_validation_plot(draws_dict: dict, target_party: str, output_path, dpi: int = 300) -> None:
    """Backtest validation density plot with actual election outcome vertical dashed lines."""
    booths = draws_dict["booths"]
    parties = list(draws_dict["parties"])
    draws = draws_dict["booth_party_draws"] # shape (D, B, K)
    actual_votes = draws_dict["actual_party_votes"] # shape (B, K)
    actual_n = draws_dict["actual_n"] # shape (B,)
    
    total_vol = actual_n.sum()
    seat_draws = pd.DataFrame(index=range(draws.shape[0]))
    seat_actuals = {}
    
    for k_idx, party in enumerate(parties):
        seat_draws[party] = (draws[:, :, k_idx] * actual_n[None, :]).sum(axis=1) / total_vol
        seat_actuals[party] = float((actual_votes[:, k_idx].sum() / total_vol) * 100.0)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(11, 6))
    
    for col in seat_draws.columns:
        is_target = col.upper() == target_party.upper()
        color = PARTY_COLORS.get(col.upper(), "#333333")
        sns.kdeplot(data=seat_draws[col].dropna(), label=col, color=color,
                    linewidth=2.8 if is_target else 1.8, ax=ax, fill=True, alpha=0.18 if is_target else 0.05)
        
        actual_val = seat_actuals.get(col, 0.0)
        top = ax.get_ylim()[1]
        ax.vlines(actual_val, 0, top * 0.85, colors=color, linestyles="dashed", linewidth=2)
        ax.text(actual_val, top * 0.88, f"{col} Actual\n{actual_val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color)

    ax.set_title(f"ENKI Backtest Validation Outcome Distributions", fontsize=15, fontweight="bold", pad=25)
    ax.set_xlabel("Projected vs Actual Primary Vote Share (%)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_xlim(0, 70)
    ticks = [t for t in ax.get_xticks() if 0 <= t <= 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}%" for t in ticks])
    ax.yaxis.set_major_formatter("{x:.2f}")
    ax.legend(title="Party", frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    sns.despine(top=True, right=True)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    plt.tight_layout()
    _finish(output_path, dpi)

# ----------------------------------------------------------------------------- OPV step (Step 8) figures
# The same five figures INANNA_V2 drew. Booth figures are for the most frequent finalist pairing (opv.report_pair).
def _pair_cols(o):
    a, b = o.report_pair
    return a, b, PARTY_COLORS.get(a, "#E13940"), PARTY_COLORS.get(b, "#0047AB")


def opv_first_preference_breakdown(o, output_path, dpi: int = 300) -> None:
    """Stacked first-preference shares per booth (mean over all draws), sorted by the reported party's share."""
    a, _, _, _ = _pair_cols(o)
    sns.set_theme(style="whitegrid")
    pref = o.booth_primary_mean.sort_values(a if a in o.booth_primary_mean.columns else o.booth_primary_mean.columns[0], ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(pref))
    left = np.zeros(len(pref))
    for p in pref.columns:
        ax.barh(y, pref[p].values, left=left, color=PARTY_COLORS.get(p, "#333333"), label=f"{p} Primary (%)", edgecolor="white")
        left += pref[p].values
    ax.set_yticks(y)
    ax.set_yticklabels(pref.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of First Preference Formal Votes (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Raw First Preference Vote Shares by Voting Booth (Sorted by {a})", fontsize=13, fontweight="bold", pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    _finish(output_path, dpi)


def opv_2cp_breakdown(o, output_path, dpi: int = 300) -> None:
    """Final two-candidate shares and the exhausted share, as a percentage of ALL formal votes in each booth."""
    a, b, ca, cb = _pair_cols(o)
    sns.set_theme(style="whitegrid")
    d = o.booth_2cp.copy()
    ev = d["expected_formal_votes"].clip(lower=1.0)
    d["pa"], d["pb"], d["pe"] = 100 * d[f"{a}_2cp_votes_mean"] / ev, 100 * d[f"{b}_2cp_votes_mean"] / ev, 100 * d["exhausted_votes_mean"] / ev
    d = d.sort_values("pa", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(d))
    ax.barh(y, d["pa"], color=ca, label=f"{a} Final Share (%)", edgecolor="white")
    ax.barh(y, d["pb"], left=d["pa"], color=cb, label=f"{b} Final Share (%)", edgecolor="white")
    ax.barh(y, d["pe"], left=d["pa"] + d["pb"], color="#708090", label="Exhausted Share (%)", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(d["booth"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of Total Formal Votes (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Final Two-Candidate ({a} vs {b}) & Exhausted Vote Shares by Booth", fontsize=13, fontweight="bold", pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    _finish(output_path, dpi)


def opv_uncertainty_forest(o, output_path, dpi: int = 300) -> None:
    """Per-booth 2CP share of the reported party: mean and 90% interval across draws."""
    a, b, ca, _ = _pair_cols(o)
    sns.set_theme(style="whitegrid")
    d = o.booth_2cp.dropna(subset=[f"{a}_2cp_share_mean"]).sort_values(f"{a}_2cp_share_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(d))
    m = d[f"{a}_2cp_share_mean"]
    ax.errorbar(m, y, xerr=[(m - d[f"{a}_2cp_p05"]).clip(lower=0), (d[f"{a}_2cp_p95"] - m).clip(lower=0)], fmt="o", color=ca, ecolor=ca,
                elinewidth=2, capsize=4, label=f"{a} Mean & 90% CI")
    ax.axvline(50.0, color="black", linestyle="--", linewidth=1.5, label="50% Majority Threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(d["booth"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{a} Final Matchup Vote Share (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Booth-Level {a} vs {b} Forecast Uncertainty (90% CI)", fontsize=13, fontweight="bold", pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    _finish(output_path, dpi)


def opv_exhausted_scatter(o, output_path, dpi: int = 300) -> None:
    """Exhausted share of formal votes against the reported party's 2CP share, one point per booth (size = expected votes)."""
    a, _, ca, _ = _pair_cols(o)
    d = o.booth_2cp.dropna(subset=[f"{a}_2cp_share_mean", "exhausted_share_of_total_mean"])
    sizes = (d["expected_formal_votes"] / d["expected_formal_votes"].max()).to_numpy() * 400 + 30
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(d[f"{a}_2cp_share_mean"], d["exhausted_share_of_total_mean"], s=sizes, alpha=0.65, color=ca)
    if len(d) >= 3 and d[f"{a}_2cp_share_mean"].nunique() > 1:
        k, c = np.polyfit(d[f"{a}_2cp_share_mean"], d["exhausted_share_of_total_mean"], 1)
        xs = np.array([d[f"{a}_2cp_share_mean"].min(), d[f"{a}_2cp_share_mean"].max()])
        ax.plot(xs, k * xs + c, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"{a} Final Vote Share (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exhausted Share of Formal Votes (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Exhausted Ballot Rate vs. {a} Support Across Booths", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    _finish(output_path, dpi)


def _kde_or_line(vals, ax, label, color, lw) -> None:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) > 2 and np.ptp(vals) > 1e-9:
        sns.kdeplot(x=vals, ax=ax, color=color, linewidth=lw, label=label, clip=(0, 100))
    elif len(vals):
        ax.axvline(float(vals.mean()), color=color, linewidth=lw, label=label)


def opv_seat_outcome(o, output_path, dpi: int = 300) -> None:
    """Seat-wide 2CP distributions of the reported pairing and the exhausted share, the 50% line with the reported party's
    win chance over ALL draws (not just this pairing), and the reported party's quartiles within this pairing."""
    a, b, ca, cb = _pair_cols(o)
    sd = o.seat_draws
    sub = sd[sd["pair"].isin([f"{a} v {b}", f"{b} v {a}"])]
    va, vb, ex = sub[f"{a}_2CP_pct"].to_numpy(), sub[f"{b}_2CP_pct"].to_numpy(), sub["exhausted_pct_formal"].to_numpy()
    q25, q50, q75 = np.nanpercentile(va, [25, 50, 75])
    p_win = float(o.win_probs.get(a, 0.0))
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, vals, colour, lw in ((f"{a} Final Share", va, ca, 2.8), (f"{b} Final Share", vb, cb, 2.4), ("Exhausted Share", ex, "#708090", 2.2)):
        _kde_or_line(vals, ax, label, colour, lw)
    top = ax.get_ylim()[1] * 1.30
    ax.set_ylim(0, top)
    ax.vlines(50.0, 0, top * 0.88, colors="black", linestyles=":", linewidth=2.4)
    ax.text(50.0, top * 0.90, f"50% Win Margin\n{a} WIN CHANCE: {100.0 * p_win:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for val, lab, ha, dx, yfrac in ((q25, "p(0.25)", "right", -0.8, 0.50), (q50, "p(0.50)", "center", 0.0, 0.66), (q75, "p(0.75)", "left", 0.8, 0.50)):
        ax.vlines(val, 0, top * (yfrac - 0.02), colors=ca, linestyles="--", linewidth=1.6)
        ax.text(val + dx, top * yfrac, f"{a} {lab}\n{val:.1f}%", ha=ha, va="bottom", fontsize=9, fontweight="bold", color=ca,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlim(0, 100)
    ax.set_title(f"Likely Overall Outcome of the Seat ({a} vs {b}, {100 * o.report_pair_prob:.0f}% of draws)", fontsize=14, fontweight="bold", pad=18)
    ax.set_xlabel("OPV Vote Share (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability", fontsize=11, fontweight="bold", labelpad=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    _finish(output_path, dpi)