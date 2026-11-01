"""ENKI_HELPERS.py -- everything that supports ENKI_V3 but is not the model itself.

Contents (nothing in here fits or simulates the model; ENKI_V3.py imports from this module, never the reverse):
  * input handling      load_x, as_frame, validate_inputs, jsonable
  * data / sampler checks   check_booth_data, diagnose_sampler
  * output folders & files  output_folder, append_history, save_actual_primaries, save_run_meta
  * plotters            generate_seat_distribution_plot, generate_booth_ridgeline_plot,
                        generate_booth_volume_ridgeline, generate_booth_swing_scatter_plot
  * terminal report     print_run_report
  * process-flow bookkeeping   PIPELINE_STEPS, PIPELINE_EDGES, check_trace, save_trace, load_trace
        (the single source of truth for the step list that ENKI_V3 records and ENKI_PFD_PDF draws)
Backtesting is deliberately NOT here; it stays in ENKI_V3.py until the dedicated backtest script exists.
"""
from __future__ import annotations
import difflib
import json
import re
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from ENKI_V3 import ModelConfig, ModelOutput, RunOptions

try:
    from adjustText import adjust_text
except ImportError:                                   # labels just overlap a little more without it
    def adjust_text(*args, **kwargs):
        return None


# ----------------------------------------------------------------------------- helpers moved out of ENKI_V2
def combine_parties(polls: pd.DataFrame, groups: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """e.g. combine_parties(polls, {"OTHER": ["SFF", "ONP", "OTH", "UND"]})"""
    p = polls.copy()
    for new, cols in groups.items():
        p[new] = p[[c for c in cols if c in p.columns]].fillna(0).sum(axis=1)
    return p


def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def check_booth_data(df: pd.DataFrame, cfg: "ModelConfig") -> None:
    """Warn (never raise) when the booth CSV does not reconcile. Rows are compared as formal votes
    (total - informal) vs the sum of every named column (ALP/LNP/GRN/IND/ONP). Only runs when an IND column exists,
    because without one OTHER is the remainder by design. Also flags duplicate booth-years and big year-on-year jumps in the seat total."""
    if not cfg.verbose:
        return
    w = df.copy()
    for c in ("total_votes", "informal_votes", "IND", "ONP", "ALP", "LNP", "GRN"):
        if c in w.columns:
            w[c] = pd.to_numeric(w[c].astype(str).str.replace(",", ""), errors="coerce")
    w["_n"] = w["total_votes"] - w["informal_votes"]
    issues = []
    if "IND" in w.columns:
        named = [c for c in ("ALP", "LNP", "GRN", "IND", "ONP") if c in w.columns]
        w["_gap"] = w["_n"] - w[named].sum(axis=1)
        off = w[w["_gap"] != 0]
        if len(off):
            worst = off.loc[off["_gap"].abs().idxmax()]
            over = int((off["_gap"] < 0).sum())
            issues.append(f"{len(off)} of {len(w)} rows: named parties do not sum to formal votes "
                        f"({over} rows have MORE named votes than formal). Worst: {worst['booth']} {int(worst['year'])}, "
                        f"gap {int(worst['_gap']):+d} votes. The gap is silently dumped into OTHER.")
    dup = int(w.duplicated(["year", "booth"]).sum())
    if dup:
        issues.append(f"{dup} duplicate booth-year row(s); they are summed together, which double counts if they are accidental repeats.")
    tot = w.groupby("year")["_n"].sum().sort_index()
    jump = tot.pct_change().dropna()
    for yr, ch in jump[jump.abs() > cfg.data_check_jump_threshold].items():
        issues.append(f"Seat formal total moves {100 * ch:+.0f}% into {int(yr)} ({int(tot.loc[yr]):,} votes); "
                      "check for missing booths or vote types (postal, iVote, absent).")
    if issues:
        print("\n!! DATA CHECK: booth CSV does not fully reconcile !!")
        for m in issues:
            print("   - " + m)
    else:
        print("-> Data check: booth CSV reconciles (rows sum to formal votes, no duplicates, no big total jumps).")


def diagnose_sampler(idata, summ, label: str, cfg: Optional["ModelConfig"] = None) -> int:
    rh_max = cfg.rhat_threshold if cfg is not None else 1.01
    ess_min = cfg.ess_threshold if cfg is not None else 400.0
    divergences = int(idata.sample_stats["diverging"].sum())
    if divergences > 0 or (summ["r_hat"] > rh_max).any() or (summ["ess_bulk"] < ess_min).any():
        print(f"WARNING [{label}]: sampler diagnostics look poor (divergences={divergences}, "
            f"max r_hat={summ['r_hat'].max():.3f}, min ESS={summ['ess_bulk'].min():.0f}). "
            "Do not trust the forecast until this is resolved.")
    return divergences


def output_folder(base: str, seat_name: str, pred_date: Optional[str] = None) -> Path:
    """<base>/<Seat_Name>/<YYYY-MM-DD>/   (date only, no time)"""
    d = pred_date or date.today().isoformat()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", seat_name).strip("_") or "Seat"
    return Path(base) / slug / d


def append_history(res: ModelOutput, seat_folder: Path, pred_date: str) -> Path:
    row = {
        "date": pred_date, 
        "S_next": res.params.get("S_next", np.nan), 
        "S_next_low": res.params.get("S_next_low", np.nan),
        "S_next_high": res.params.get("S_next_high", np.nan), 
        "beta": res.params.get("beta", res.params.get("p_g", np.nan)),
        "d_bar": res.params.get("d_bar", 0.0)
    }
    
    for _, r in res.seat.iterrows():
        scenario = r["scenario"]
        share_val = r.get("primary_share_pct", r.get("seat_share_pct", r.get("seat_share", np.nan)))
        row[f"seat_{scenario}"] = share_val
        
    path = seat_folder / "history.csv"
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["date"] != pred_date], new], ignore_index=True).sort_values("date")
        
    new.to_csv(path, index=False)
    return path


def generate_booth_ridgeline_plot(res_model_output, target_party="ALP", top_n_booths=15,
                                output_path="booth_distributions.png", max_draws=1500, dpi=300,
                                special_keywords=("PRE-POLL", "PREPOLL", "DECLARATION", "ABSENT", "POSTAL"), **kwargs):
    """Ridgeline of the model's ACTUAL booth-level predictive draws (res.booth_draws_df); booths with
    pre-poll / declaration style names are always included, the rest are the largest by expected volume."""
    output_path = Path(output_path)
    draws = res_model_output.booth_draws_df
    fc = res_model_output.forecast
    if draws is None or draws.empty:
        raise ValueError("No booth-level draws on the result (res.booth_draws_df); re-run predict().")
    if len(draws) > max_draws:
        draws = draws.sample(max_draws, random_state=0)
    special = [b for b in draws.columns if any(k in str(b).upper() for k in special_keywords)]
    regular = fc[~fc["booth"].isin(special)].sort_values("formal_weight", ascending=False)["booth"].tolist()
    chosen = special + regular[: max(5, top_n_booths - len(special))]
    order = sorted(chosen, key=lambda b: draws[b].mean())
    df_long = draws[order].melt(var_name="booth", value_name="vote_share")
    df_long["booth"] = pd.Categorical(df_long["booth"], categories=order, ordered=True)
    min_val = max(0.0, df_long["vote_share"].min() - 1.0)
    max_val = min(100.0, df_long["vote_share"].max() + 1.0)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    palette = sns.color_palette("plasma", n_colors=len(order))
    g = sns.FacetGrid(df_long, row="booth", hue="booth", aspect=12, height=0.55, palette=palette)
    g.map(sns.kdeplot, "vote_share", clip_on=False, fill=True, alpha=0.7, lw=1.5, bw_adjust=0.6)
    g.map(sns.kdeplot, "vote_share", clip_on=False, color="black", lw=1, bw_adjust=0.6)
    g.map(plt.axhline, y=0, lw=1.5, color="black", clip_on=False)

    def annotate_facet(x, color, label):
        ax = plt.gca()
        ax.text(-0.03, 0.2, str(order[list(g.axes.flatten()).index(ax)]), fontweight="bold", color="black",
                ha="right", va="center", transform=ax.transAxes, fontsize=8.5)

    g.map(annotate_facet, "vote_share")
    g.figure.subplots_adjust(hspace=-0.35)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    ax_bottom = g.axes[-1, 0]
    ax_bottom.set_xlim(min_val, max_val)
    ax_bottom.set_xlabel(f"Forecasted {target_party} Primary Vote Share (%)", fontsize=11, fontweight="bold", labelpad=10)
    g.figure.suptitle(f"ENKI Monte Carlo {target_party} Vote Share Distribution (model draws)", fontsize=13, fontweight="bold", y=0.98)
    g.figure.subplots_adjust(left=0.32, right=0.95, top=0.92, bottom=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"-> Ridgeline plot saved to: {output_path}")


def generate_seat_distribution_plot(res_model_output, output_path="seat_distribution.png", dpi=300):
    """
    Plots primary vote probability density curves for ALL parties in forecast_draws.csv.
    Applies Worst/Average/Best reference lines EXCLUSIVELY to ALP.
    """
    if res_model_output.draws_df is not None:
        draws_df = res_model_output.draws_df
    else:
        draws_df = pd.DataFrame()

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(11, 6))

    party_colors = {
        "ALP": "#E13940",   # Labor Red
        "LNP": "#0047AB",   # Coalition Blue
        "LIB": "#0047AB",
        "NAT": "#0066CC",
        "GRN": "#104C34",   # Greens Green
        "ONP": "#FF8C00",   # One Nation Orange
        "OTHER": "#708090"  # Slate Grey
    }

    for col in draws_df.columns:
        if col.upper() in ["SCENARIO", "GAMMA_NEXT"]:
            continue
        vals = draws_df[col].dropna()
        p_name = col.upper()
        color = party_colors.get(p_name, "#333333")
        
        is_alp = (p_name == "ALP")
        lw = 2.8 if is_alp else 1.8
        alpha_fill = 0.18 if is_alp else 0.05

        sns.kdeplot(
            data=vals,
            label=p_name,
            color=color,
            linewidth=lw,
            ax=ax,
            fill=True,
            alpha=alpha_fill
        )

    if "ALP" in draws_df.columns:
        alp_draws = draws_df["ALP"].dropna()
        p05_val, avg_val, p95_val = np.percentile(alp_draws, [5, 50, 95])
    else:
        seat_df = res_model_output.seat
        avg_row = seat_df[seat_df["scenario"] == "average"]
        avg_val = float(avg_row["primary_share_pct"].iloc[0])
        p05_val = float(avg_row["seat_p05"].iloc[0]) if "seat_p05" in avg_row.columns else float(seat_df[seat_df["scenario"] == "worst"]["primary_share_pct"].iloc[0])
        p95_val = float(avg_row["seat_p95"].iloc[0]) if "seat_p95" in avg_row.columns else float(seat_df[seat_df["scenario"] == "best"]["primary_share_pct"].iloc[0])

    y_lims = ax.get_ylim()
    max_y = y_lims[1]
    alp_color = party_colors["ALP"]

    targets = [
        {"val": p05_val, "label": f"ALP Worst (0.05)\n{p05_val:.1f}%", "y_pos": max_y * 0.70, "ha": "right", "x_offset": -0.4},
        {"val": avg_val, "label": f"ALP Average (0.50)\n{avg_val:.1f}%", "y_pos": max_y * 0.90, "ha": "center", "x_offset": 0.0},
        {"val": p95_val, "label": f"ALP Best (0.95)\n{p95_val:.1f}%", "y_pos": max_y * 0.70, "ha": "left", "x_offset": 0.4}
    ]

    for item in targets:
        v = item["val"]
        lbl = item["label"]
        y_pos = item["y_pos"]
        ha = item["ha"]
        x_off = item["x_offset"]

        ax.vlines(x=v, ymin=0, ymax=y_pos - (max_y * 0.03), colors=alp_color, linestyles="dotted", linewidth=2)
        ax.text(
            x=v + x_off, y=y_pos, s=lbl, ha=ha, va="bottom",
            fontsize=9, fontweight="bold", color=alp_color
        )

    ax.set_title("ENKI Multi-Party Primary Vote Outcome Distributions", fontsize=15, fontweight="bold", pad=25)
    ax.set_xlabel("Projected Primary Vote Share (%)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold", labelpad=10)

    ax.set_xlim(0, 70)  # Standard bounds for primary votes
    ticks = [t for t in ax.get_xticks() if 0 <= t <= 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}%" for t in ticks])
    ax.yaxis.set_major_formatter('{x:.2f}')

    ax.legend(title="Party", frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    sns.despine(top=True, right=True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def generate_booth_volume_ridgeline(booths, forecasted_matrix, df_historical, output_path, top_n=15, log_scale=True, dpi=300):
    output_path = Path(output_path)
    
    # 1. Get the latest historical vote counts for each booth
    max_year = df_historical['year'].max()
    vcol = 'n' if 'n' in df_historical.columns else 'total_votes'      # formal votes, to match the forecast
    df_latest = df_historical[df_historical['year'] == max_year].set_index('booth')[vcol].to_dict()
    
    # Calculate mean forecasted volume per booth to sort them descending
    mean_volumes = forecasted_matrix.mean(axis=0)
    sorted_indices = np.argsort(mean_volumes)[::-1] # Descending order
    
    top_indices = sorted_indices[:top_n]
    plot_booths = [booths[i] for i in top_indices]
    plot_matrix = forecasted_matrix[:, top_indices] # Shape: (num_sims, top_n)
    
    # 2. Setup figure with a WIDE left margin so booth names never get cut off
    fig, axes = plt.subplots(len(plot_booths), 1, figsize=(12, len(plot_booths) * 0.45), sharex=True)
    
    if len(plot_booths) == 1:
        axes = [axes]
        
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_booths)))
    
    for idx, (i, booth_name) in enumerate(zip(top_indices, plot_booths)):
        ax = axes[idx]
        sim_counts = plot_matrix[:, idx]
        
        # Optional log-transform for skewed pre-poll data
        if log_scale:
            plot_data = np.log10(sim_counts + 1)
        else:
            plot_data = sim_counts
            
        # Draw the kernel density estimate
        sns.kdeplot(plot_data, ax=ax, fill=True, color=colors[idx], alpha=0.6, linewidth=1.5)
        
        # Get and transform historical baseline marker safely
        raw_hist_val = df_latest.get(booth_name, np.nan)
        hist_val = pd.to_numeric(raw_hist_val, errors='coerce')
        
        if not np.isnan(hist_val) and hist_val > 0:
            plot_hist = np.log10(hist_val + 1) if log_scale else hist_val
            # On the top axis (idx == 0), give it a native label so matplotlib picks it up for the legend automatically
            label_str = 'Previous Election Volume' if idx == 0 else ""
            ax.axvline(x=plot_hist, color='red', linestyle='--', linewidth=1.5, ymin=0.1, ymax=0.9, label=label_str)
        
        # Formatting style
        ax.set_facecolor((0, 0, 0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Place booth name safely in the expanded left margin area
        ax.text(-0.03, 0.2, str(booth_name), transform=ax.transAxes, 
                ha='right', va='center', fontsize=9, fontweight='bold', color='black')
        
        if idx < len(plot_booths) - 1:
            ax.spines['bottom'].set_visible(False)
        else:
            ax.spines['bottom'].set_visible(True)
            xlabel_text = "Log10(Forecasted Voters + 1)" if log_scale else "Forecasted Number of Voters ($n$)"
            ax.set_xlabel(xlabel_text, fontsize=10, fontweight='bold')

    # --- NATIVE LEGEND SETUP (POSITIONED OUT OF THE WAY) ---
    if len(axes) > 0:
        axes[0].legend(
            loc='upper left', 
            bbox_to_anchor=(1.02, 1.0), 
            frameon=True, 
            facecolor='white', 
            edgecolor='none', 
            fontsize=9
        )

    title_suffix = " (Log Scale)" if log_scale else ""
    plt.suptitle(f"ENKI Monte Carlo Booth Volume Distribution vs. Previous Election{title_suffix}", y=0.98, fontsize=12, fontweight='bold')
    
    # Leave extra space on the left (0.22) for booth names and right (0.85) for external legend
    plt.tight_layout(rect=[0.22, 0, 0.85, 0.96])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"-> Booth volume ridgeline plot saved to: {output_path}")


def generate_booth_swing_scatter_plot(res_model_output, booth_input_df=None, output_path="booth_swing_scatter.png", label_all=True, dpi=300):
    """Expected booth volume vs projected swing from the booth's own LAST-election share.
    Both come from the model output (fc.last_share, fc.formal_weight); nothing is invented."""
    fc = res_model_output.forecast.copy()
    need = {"mean", "last_share", "formal_weight"}
    if not need <= set(fc.columns):
        raise ValueError(f"forecast table missing {sorted(need - set(fc.columns))}; re-run predict().")
    fc = fc.dropna(subset=["last_share"])
    fc["swing"] = fc["mean"] - fc["last_share"]
    fc["n_bt"] = fc["formal_weight"].astype(float)
    fc["abs_swing"] = fc["swing"].abs()
    max_swing = max(5.0, fc["abs_swing"].max() * 1.35)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(data=fc, x="n_bt", y="swing", size="n_bt", sizes=(40, 350), color="black", alpha=0.75,
                    legend=False, ax=ax)
    
    # Set x-axis to log scale
    ax.set_xscale("log")
    
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    labelled = fc if label_all else fc.sort_values(by="abs_swing", ascending=False).head(5)
    texts = [ax.text(r["n_bt"], r["swing"], str(r["booth"]), fontsize=8, fontweight="bold", color="black")
            for _, r in labelled.iterrows()]
    adjust_text(texts, x=labelled["n_bt"].values, y=labelled["swing"].values, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7))
    ax.set_title("ENKI Expected Booth Size vs. Projected Swing from the Booth's Last Election Share", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Expected Formal Votes at Next Election (Log Scale)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Projected Mean Share minus Last Election Share (points)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylim(-max_swing, max_swing)
    ax.yaxis.set_major_formatter('{x:.1f}')
    sns.despine(top=True, right=True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_actual_primaries(prepared: pd.DataFrame, parties: Sequence[str], folder) -> None:
    """Seat-wide ACTUAL primary votes per election (all booths): year, formal_total, one column per modelled party."""
    rows = []
    for y, g in prepared.groupby("year"):
        r = {"year": int(y), "formal_total": int(g["n"].sum())}
        r.update({p: int(g[f"y_{p}"].sum()) for p in parties})
        rows.append(r)
    Path(folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(folder) / "actual_primaries.csv", index=False)


def save_run_meta(folder, cfg: ModelConfig, booths_path: str, polls_path: str) -> None:
    """Small provenance record so a downstream stage can see which run it is reading."""
    meta = {"seat_name": cfg.seat_name, "party": cfg.party, "parties": list(cfg.parties), "prediction_date": cfg.prediction_date,
            "election_date": cfg.election_date, "booths_file": str(booths_path), "polls_file": str(polls_path),
            "onp_layer": bool(cfg.onp_layer), "brian_mode": bool(cfg.brian_mode), "saved_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "run_meta.json").write_text(json.dumps(meta, indent=2))


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def load_x(x) -> Dict[str, Any]:
    """dict | path to .json -> one flat dict. Groups ('data','config','run') are merged; flat keys win over grouped ones."""
    if isinstance(x, (str, Path)):
        x = json.loads(Path(x).read_text(encoding="utf-8"))
    if not isinstance(x, dict):
        raise TypeError("ENKI(x): x must be a dict or a path to a .json file.")
    flat: Dict[str, Any] = {}
    for grp in ("data", "config", "run"):
        sub = x.get(grp)
        if sub is None:
            continue
        for k, v in dict(sub).items():
            if k in flat:
                raise ValueError(f"Key {k!r} appears in more than one group of x.")
            flat[k] = v
    flat.update({k: v for k, v in x.items() if k not in ("data", "config", "run")})
    return flat


def as_frame(v, name: str) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if isinstance(v, (str, Path)):
        return pd.read_csv(v)
    if isinstance(v, (list, dict)):
        return pd.DataFrame(v)
    raise TypeError(f"{name}: expected a DataFrame, a CSV path, or records; got {type(v).__name__}.")


def validate_inputs(cfg: "ModelConfig", run: "RunOptions") -> List[str]:
    """Hard errors are raised together; soft problems are returned as warnings."""
    err: List[str] = []
    warn: List[str] = []
    P = list(cfg.parties)
    if len(P) < 2 or len(set(P)) != len(P):
        err.append("parties must have at least two distinct entries.")
    if cfg.party not in P:
        err.append(f"party={cfg.party!r} must be one of parties={P}.")
    if "OTHER" in P and P[-1] != "OTHER":
        warn.append("OTHER is not the last party; the LAST party is the reference category.")
    for n in ("draws", "tune", "chains"):
        if int(getattr(cfg, n)) < 1:
            err.append(f"{n} must be >= 1.")
    for n in ("target_accept", "volume_target_accept"):
        if not 0.0 < float(getattr(cfg, n)) < 1.0:
            err.append(f"{n} must be strictly between 0 and 1.")
    sp = [float(v) for v in cfg.scenario_pct]
    if len(sp) != 3 or not (0.0 < sp[0] < sp[1] < sp[2] < 100.0):
        err.append("scenario_pct must be three increasing percentiles in (0, 100), e.g. (5, 50, 95).")
    for n in ("sigma_alpha_sd", "sigma_eps_sd", "sigma_d_median", "sigma_d_sdlog", "beta_data_sd", "beta_uniform_sd",
              "beta_grn_sd", "beta_other_sd", "m_prior_sd", "rho_beta_a", "rho_beta_b", "vol_sigma_a_sd", "vol_tau_sd",
              "vol_kappa_alpha", "vol_kappa_beta", "poll_half_life_months", "poll_z", "election_cycle_years",
              "poll_default_sample_size", "volume_step_floor"):
        if not float(getattr(cfg, n)) > 0.0:
            err.append(f"{n} must be > 0.")
    for n in ("poll_house_sd", "time_horizon_gamma", "volume_total_sd_floor", "backtest_poll_house_sd"):
        if float(getattr(cfg, n)) < 0.0:
            err.append(f"{n} must be >= 0.")
    if cfg.d_persist is not None and not -1.0 < float(cfg.d_persist) < 1.0:
        err.append("d_persist must be None (estimate it) or a number in (-1, 1).")
    if cfg.beta_preset not in ("data", "uniform"):
        err.append("beta_preset must be 'data' or 'uniform'.")
    if cfg.volume_drift not in ("last", "mean", "none"):
        err.append("volume_drift must be 'last', 'mean' or 'none'.")
    if not 0.0 < float(cfg.expected_turnout_rate) <= 1.0:
        err.append("expected_turnout_rate must be in (0, 1].")
    if cfg.registered_electors is not None and float(cfg.registered_electors) <= 0.0:
        err.append("registered_electors must be positive (or None).")
    for n in ("brian_onp_from_lnp", "brian_onp_from_alp", "onp_max_seat_share"):
        if not 0.0 <= float(getattr(cfg, n)) <= 1.0:
            err.append(f"{n} must be between 0 and 1.")
    if float(cfg.onp_seat_ratio_median) <= 0.0 or float(cfg.onp_seat_ratio_sdlog) < 0.0:
        err.append("onp_seat_ratio_median must be > 0 and onp_seat_ratio_sdlog >= 0.")
    try:
        pred, elec = pd.to_datetime(cfg.prediction_date), pd.to_datetime(cfg.election_date)
        if elec < pred:
            warn.append(f"election_date {cfg.election_date} is before prediction_date {cfg.prediction_date}: "
                        "the time-horizon uncertainty collapses to 0.")
    except Exception:
        err.append("prediction_date / election_date must be parseable dates (YYYY-MM-DD).")
    if cfg.brian_mode and cfg.onp_layer:
        warn.append("brian_mode and onp_layer are both on: BrianMode takes precedence and the statewide-ONP layer is bypassed.")
    if cfg.brian_mode and "ONP" in P:
        err.append("brian_mode builds ONP itself, so 'ONP' must not be in parties.")
    if run.output_root is None and run.make_plots:
        warn.append("make_plots is on but output_root is None, so no plots are drawn.")
    if err:
        raise ValueError("ENKI(x): invalid input:\n  - " + "\n  - ".join(err))
    return warn



def print_run_report(y, cfg, priors) -> None:
    pd.set_option("display.width", 160)
    for w in y.warnings:
        print(f"WARNING: {w}")
    if y.backtest is not None:
        print("\nBACKTEST (share model)")
        print(y.backtest.round(2).T.to_string())
    res = y.forecast
    if res is None:
        return
    print("\nSEAT-WIDE PRIMARY VOTE (joint draws, volume-weighted; sums to 100% in every draw)")
    print(y.seat_summary.round(2).to_string())
    if res.attribution is not None:
        print(f"\nUNCERTAINTY ATTRIBUTION: width of the 90% band on {cfg.party} seat share, one layer off at a time")
        print(res.attribution.round(2).to_string(index=False))
    print(f"\nDiagnostics: max r_hat={res.params['rhat_max']:.3f}, min ESS={res.params['ess_bulk_min']:.0f}, "
        f"divergences={res.params['divergences']} (volume model: {res.params.get('volume_divergences', 'n/a')})")
    print("beta (sensitivity to the party's own statewide log-share swing): posterior vs prior")
    for p in cfg.parties:
        print(f"  {p:6s}: {res.params[f'beta_{p}']:.2f} +/- {res.params[f'beta_{p}_sd']:.2f}   (prior {priors[p][0]:.2f} +/- {priors[p][1]:.2f})")


# ----------------------------------------------------------------------------- process-flow registry
# One list drives three things: what ENKI_V3 must record (trace keys), what ENKI_PFD_PDF draws, and what check_trace verifies.
# Layout coordinates are for the page-1 diagram (a 14 x 7.5 canvas); edges are (from, to, kind).
PIPELINE_STEPS: List[Dict[str, Any]] = [
    dict(id="S0", name="Prep", pos=(2.0, 6.0),
        what="counts -> shares; statewide log-swing",
        eq="x_kt = ln S_kt - mean_t ln S_kt",
        inputs="booth_results, state_results", outputs="x (T x K), S_hist",
        trace=["s0_years", "s0_parties", "s0_state_pct", "s0_x", "s0_x_center"]),
    dict(id="S1", name="Fit", pos=(5.4, 6.0),
        what="posterior of the multinomial-logit model",
        eq="eta = m + alpha_b + beta*x + d_t + eps",
        inputs="Y, x, priors", outputs="theta^(i), i = 1..D",
        trace=["s1_beta", "s1_rho", "s1_sigma_d", "s1_sigma_alpha", "s1_sigma_eps", "s1_m", "s1_prior_beta", "s1_diag"]),
    dict(id="S2", name="Polls", pos=(2.0, 3.4),
        what="weighted poll average + error budget -> statewide share draws",
         eq="S ~ Dir(conc * s_hat);  se^2 = se_poll^2 + house^2 + g^2 tau",
        inputs="polls, dates, config", outputs="S^(i) (D x K)",
        trace=["s2_se_parts", "s2_conc", "s2_central", "s2_central_onp", "s2_S_next", "s2_onp_state"]),
    dict(id="S3", name="Tide", pos=(5.4, 3.4),
        what="statewide swing + AR(1) tide -> party-level shift",
        eq="lin = beta*(ln S - c) + rho*d_T + sqrt(1-rho^2)*sigma_d*z",
        inputs="theta, S", outputs="lin (D x K)",
        trace=["s3_x_next", "s3_bx", "s3_d_next", "s3_lin", "s3_d_hist_mean", "s3_d_hist_p05", "s3_d_hist_p95", "s3_bx_hist"]),
    dict(id="S4", name="Booths", pos=(8.8, 3.4),
        what="booth-level shares (softmax)",
        eq="p_b = softmax(m + alpha_b + lin + eps_b)",
        inputs="lin, alpha, eps", outputs="p_b (D x B x K)",
        trace=["s4_p_pre", "s4_booths", "s4_last_party_share"]),
    dict(id="S5", name="Volumes", pos=(2.0, 0.9),
        what="formal votes per booth",
         eq="V ~ Mult(T', Dir(kappa * softmax(a + tau' early)))",
        inputs="booth history | electors | projections", outputs="V (D x B)",
        trace=["s5_V", "s5_hist_totals", "s5_last_n", "s5_mode"]),
    dict(id="S6", name="ONP", pos=(12.2, 3.4),
        what="One Nation entry / vote transfer (mass-conserving)",
        eq="p'_k = (1-f_k) p_k ;  p'_ONP = sum_k f_k p_k",
        inputs="p_b, f or S_ONP", outputs="p'_b (D x B x K+1)",
        trace=["s6_mode", "s6_retain"]),
    dict(id="S7", name="Seat", pos=(8.8, 0.9),
        what="volume-weighted seat shares and percentiles",
        eq="s_k = sum_b V_b p'_kb / sum_b V_b",
        inputs="V, p'_b", outputs="seat shares (D x K+1)",
        trace=["s7_seat_draws", "s7_out_parties"]),
]
PIPELINE_EDGES = [("S0", "S1", "solid"), ("S1", "S3", "solid"), ("S2", "S3", "solid"), ("S3", "S4", "solid"),
                ("S4", "S6", "solid"), ("S6", "S7", "solid"), ("S5", "S7", "solid"), ("S0", "S5", "dashed")]


def check_trace(trace: Optional[dict]) -> List[str]:
    """Names every step whose recorded signals are incomplete (so the diagram and the code cannot silently drift apart)."""
    if not trace:
        return ["trace is empty: the process report has nothing to draw."]
    out = []
    for st in PIPELINE_STEPS:
        missing = [k for k in st["trace"] if k not in trace]
        if missing:
            out.append(f"trace step {st['id']} ({st['name']}) is missing: {', '.join(missing)}")
    return out


def save_trace(folder, trace: dict) -> None:
    """Arrays -> trace.npz; DataFrames -> trace_<key>.csv; scalars, strings and `meta` -> trace_meta.json."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    scalars: Dict[str, Any] = {}
    tables: List[str] = []
    for k, v in trace.items():
        if k == "meta":
            continue
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"trace_{k}.csv", index=False)
            tables.append(k)
        elif isinstance(v, (str, bool, int, float, np.generic)):
            scalars[k] = v.item() if isinstance(v, np.generic) else v
        else:
            arrays[k] = np.asarray(v)
    np.savez_compressed(out / "trace.npz", **arrays)
    (out / "trace_meta.json").write_text(
        json.dumps({"scalars": scalars, "tables": tables, "meta": trace.get("meta", {})}, indent=2, default=jsonable), encoding="utf-8")


def load_trace(folder) -> dict:
    """Inverse of save_trace."""
    out = Path(folder)
    meta = json.loads((out / "trace_meta.json").read_text(encoding="utf-8"))
    trace: Dict[str, Any] = {}
    with np.load(out / "trace.npz", allow_pickle=False) as z:
        for k in z.files:
            trace[k] = z[k]
    trace.update(meta.get("scalars", {}))
    for k in meta.get("tables", []):
        df = pd.read_csv(out / f"trace_{k}.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        trace[k] = df
    trace["meta"] = meta.get("meta", {})
    return trace