"""ENKI_BACKTEST.py -- Decoupled historical backtesting logic for ENKI V3.

Houses all walk-forward validation, probabilistic scoring, and historical
seat/volume backtests using lazy imports to prevent circular dependencies.
Includes advanced probabilistic calibration diagnostics (evaluating true 
interval coverage vs nominal intervals and skill vs luck metrics).
"""

from __future__ import annotations
import gc
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def compute_calibration_diagnostics(
    draws: np.ndarray,
    actual_shares: np.ndarray,
    intervals: Sequence[int] = (50, 80, 90, 95),
) -> Dict[str, float]:
    """Calculate empirical coverage across multiple nominal credible intervals.

    draws: shape (D, N_booths, N_parties) or (D, N_booths)
    actual_shares: matching shape to draws[0]
    """
    results = {}
    if draws.ndim == 3:
        # Evaluate primary party (or flatten across all parties)
        pass

    for ci in intervals:
        alpha = (100.0 - ci) / 100.0
        lo_pct = (alpha / 2.0) * 100.0
        hi_pct = (1.0 - alpha / 2.0) * 100.0

        lo = np.percentile(draws, lo_pct, axis=0)
        hi = np.percentile(draws, hi_pct, axis=0)

        covered = (actual_shares >= lo) & (actual_shares <= hi)
        empirical_cov = float(np.mean(covered))
        results[f"cover_{ci}"] = empirical_cov
        results[f"bias_{ci}"] = empirical_cov - (ci / 100.0)

    return results


def save_backtest_draws(draws_by_year: dict, folder: Union[str, Path]) -> None:
    """Save one compressed file per backtested election containing predicted booth x party primaries

    plus actual booth votes for downstream consumption (e.g., INANNA).
    """
    out_folder = Path(folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for y, d in draws_by_year.items():
        np.savez_compressed(
            out_folder / f"backtest_draws_{int(y)}.npz",
            booths=np.array(d["booths"]),
            parties=np.array(d["parties"]),
            booth_party_draws=d["booth_party_draws"],
            actual_party_votes=d["actual_party_votes"],
            actual_n=d["actual_n"],
        )


def save_backtest_table_png(summary_df: pd.DataFrame, output_path: Union[str, Path], dpi: int = 300) -> None:
    """Render backtest summary metrics into a clean table PNG for visualization."""
    import matplotlib.pyplot as plt

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_df is None or summary_df.empty:
        print("-> Warning: Backtest summary dataframe is empty. Skipping table PNG generation.")
        return

    display_cols = [
        "target",
        "seat_pred_median",
        "seat_actual",
        "seat_error",
        "nochange_seat_err",
        "statewide_swing_seat_err",
        "seat_in_interval",
        "booth_MAE",
        "nochange_MAE",
        "statewide_swing_MAE",
        "booth_cover_50",
        "booth_cover_90",
        "booth_cover_95",
    ]
    labels = {
        "seat_pred_median": "seat pred\nmedian",
        "seat_actual": "seat\nactual",
        "seat_error": "MODEL\nseat err",
        "nochange_seat_err": "NO-CHANGE\nseat err",
        "statewide_swing_seat_err": "STATE-SWING\nseat err",
        "seat_in_interval": "in 90%\ninterval",
        "booth_MAE": "MODEL\nbooth MAE",
        "nochange_MAE": "NO-CHANGE\nbooth MAE",
        "statewide_swing_MAE": "STATE-SWING\nbooth MAE",
        "booth_cover_50": "50% Cov\n(Target 50%)",
        "booth_cover_90": "90% Cov\n(Target 90%)",
        "booth_cover_95": "95% Cov\n(Target 95%)",
    }
    existing_cols = [c for c in display_cols if c in summary_df.columns]
    if not existing_cols:
        existing_cols = list(summary_df.columns)

    df_plot = summary_df[existing_cols].copy()
    for col in df_plot.columns:
        if df_plot[col].dtype in [float, np.floating]:
            df_plot[col] = df_plot[col].round(2)

    fig, ax = plt.subplots(figsize=(max(14, len(existing_cols) * 1.6), len(df_plot) * 0.8 + 2.5))
    ax.axis("off")

    table = ax.table(
        cellText=df_plot.values,
        colLabels=[labels.get(c, c) for c in df_plot.columns],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.2, 1.8)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if key[0] % 2 == 0 else "white")

    plt.suptitle("ENKI Backtest Error & Calibration Diagnostics", fontsize=14, fontweight="bold", y=0.92)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"-> Backtest summary table PNG saved to: {out_path}")


def backtest_bayes(
    booth_results: pd.DataFrame,
    state_results: Optional[pd.DataFrame] = None,
    config: Optional[object] = None,
    targets: Optional[Sequence[int]] = None,
    return_draws: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """Refit on elections BEFORE each target using lazy imports from ENKI_V3.

    Calculates empirical probabilistic calibration (50%, 80%, 90%, 95% coverage)
    to verify model reliability vs luck.
    """
    from dataclasses import replace as _replace
    from ENKI_V3 import (
        STATE_RESULTS,
        ModelConfig,
        _simulate_next,
        _state_matrix,
        fit_bayesian_model,
        prepare_bayes_data,
    )

    _base = config or ModelConfig()
    cfg = _replace(_base, poll_house_sd=_base.backtest_poll_house_sd, verbose=False)
    state_df = STATE_RESULTS if state_results is None else state_results
    df_all = prepare_bayes_data(booth_results, cfg)
    years = sorted(df_all["year"].unique())
    parties = list(cfg.parties)
    k0 = parties.index(cfg.party)
    rng = np.random.default_rng(cfg.seed)
    lo_p, _, hi_p = (float(x) for x in cfg.scenario_pct)

    rows = []
    draws_out = {}
    for tgt in targets or years[2:]:
        fit = fit_bayesian_model(df_all[df_all["year"] < tgt], cfg, state_df)
        fit["idata"] = None
        D = len(fit["draws"]["beta"])
        S_all = _state_matrix(state_df, parties, years, cfg.onp_state_denominator)
        S_t = S_all.loc[tgt].values
        p_next, _ = _simulate_next(fit, np.tile(S_t, (D, 1)), cfg, rng)

        t = df_all[(df_all["year"] == tgt) & df_all["booth"].isin(fit["booth_names"])]
        col = [fit["booth_names"].index(b) for b in t["booth"]]
        n_t, y_t = t["n"].values, t["y"].values

        if return_draws:
            draws_out[int(tgt)] = {
                "booths": [str(b) for b in t["booth"]],
                "parties": parties,
                "booth_party_draws": 100.0 * p_next[:, col, :],
                "actual_party_votes": np.column_stack([t[f"y_{p}"].values for p in parties]),
                "actual_n": n_t.astype(float),
            }

        P = 100.0 * p_next[:, col, k0]
        med, act = np.median(P, axis=0), 100.0 * y_t / n_t

        # Posterior predictive draws including binomial observation noise
        rep = 100.0 * rng.binomial(n_t[None, :], p_next[:, col, k0]) / n_t[None, :]

        # Calibration Diagnostics across multiple credible intervals
        calib = compute_calibration_diagnostics(rep, act, intervals=(50, 80, 90, 95))

        seat = (P * n_t).sum(axis=1) / n_t.sum()
        seat_act = float((act * n_t).sum() / n_t.sum())
        s_lo, s50, s_hi = np.percentile(seat, [lo_p, 50, hi_p])
        err = med - act

        row = {
            "target": int(tgt),
            "seat_pred_median": s50,
            "seat_actual": seat_act,
            "seat_error": s50 - seat_act,
            "seat_lo": s_lo,
            "seat_hi": s_hi,
            "seat_in_interval": bool(s_lo <= seat_act <= s_hi),
            "booth_MAE": float(np.abs(err).mean()),
            "booth_RMSE": float(np.sqrt((err**2).mean())),
            "booth_cover_50": calib["cover_50"],
            "booth_cover_80": calib["cover_80"],
            "booth_cover_90": calib["cover_90"],
            "booth_cover_95": calib["cover_95"],
            "calib_bias_90": calib["bias_90"],
            "n_booths": len(t),
            "divergences": fit["divergences"],
            "rhat_max": float(fit["diag"]["r_hat"].max()),
        }

        for j, p in enumerate(parties):
            yj = t[f"y_{p}"].values
            sj = (100.0 * p_next[:, col, j] * n_t).sum(axis=1) / n_t.sum()
            row[f"seat_err_{p}"] = float(np.median(sj) - 100.0 * yj.sum() / n_t.sum())

        last_y = max(int(y) for y in years if y < tgt)
        last = df_all[df_all["year"] == last_y].set_index("booth")["share"]
        S_last = float(S_all.loc[last_y, cfg.party])
        mk = np.array([b in last.index for b in t["booth"]])
        row["common_n"] = int(mk.sum())

        if mk.any():
            prev = last.reindex(t["booth"].values[mk]).values
            a_c, n_c, m_c = act[mk], n_t[mk], med[mk]

            def wavg(v):
                return float((v * n_c).sum() / n_c.sum())

            e_md = m_c - a_c
            e_nc = prev - a_c
            e_sw = prev + (float(S_all.loc[tgt, cfg.party]) - S_last) - a_c

            row.update({
                "model_MAE_common": float(np.abs(e_md).mean()),
                "nochange_MAE": float(np.abs(e_nc).mean()),
                "statewide_swing_MAE": float(np.abs(e_sw).mean()),
                "model_seat_err_common": wavg(e_md),
                "nochange_seat_err": wavg(e_nc),
                "statewide_swing_seat_err": wavg(e_sw),
            })

        rows.append(row)
        del fit, p_next, P, rep
        gc.collect()

    return (pd.DataFrame(rows), draws_out) if return_draws else pd.DataFrame(rows)


def backtest_volume(
    booth_results: pd.DataFrame,
    config: Optional[object] = None,
    targets: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Predict each election's booth volumes using lazy imports."""
    from ENKI_V3 import ModelConfig, _fit_volume_posterior, _simulate_volumes, prepare_bayes_data

    cfg = config or ModelConfig()
    df = prepare_bayes_data(booth_results, cfg)
    years = sorted(df["year"].unique())
    rng = np.random.default_rng(cfg.seed)
    rows = []

    for tgt in targets or years[2:]:
        train = df[df["year"] < tgt]
        vfit = _fit_volume_posterior(train, cfg)
        last_y = int(train["year"].max())
        prev_all = train[train["year"] == last_y].set_index("booth")["n"]
        active = np.array([b in prev_all.index for b in vfit["names"]])

        V = _simulate_volumes(vfit, cfg.backtest_volume_sims, cfg, rng, target_year=tgt, active=active)
        t = df[df["year"] == tgt].set_index("booth")["n"]
        idx = [i for i, b in enumerate(vfit["names"]) if active[i] and b in t.index]
        names = [vfit["names"][i] for i in idx]

        act = t.reindex(names).values.astype(float)
        P = V[:, idx]
        med = np.median(P, axis=0)
        lo, hi = np.percentile(P, [5, 95], axis=0)
        prev = prev_all.reindex(names).values.astype(float)
        tot = V.sum(axis=1)

        early = vfit["btype"].astype(bool)
        e_pred = np.median(V[:, early].sum(axis=1) / np.maximum(tot, 1)) if early.any() else np.nan
        e_act = (
            (df[(df["year"] == tgt) & (df["btype"] == 1)]["n"].sum() / df[df["year"] == tgt]["n"].sum())
            if early.any()
            else np.nan
        )

        rows.append({
            "target": int(tgt),
            "n_booths": len(names),
            "booth_MAPE_pct": float(np.mean(np.abs(med - act) / act) * 100),
            "nochange_MAPE_pct": float(np.mean(np.abs(prev - act) / act) * 100),
            "booth_cover_90": float(((act >= lo) & (act <= hi)).mean()),
            "total_pred_median": float(np.median(tot)),
            "total_actual": float(df[df["year"] == tgt]["n"].sum()),
            "total_error_pct": float((np.median(tot) / df[df["year"] == tgt]["n"].sum() - 1) * 100),
            "total_in_90": bool(np.percentile(tot, 5) <= df[df["year"] == tgt]["n"].sum() <= np.percentile(tot, 95)),
            "early_share_pred": float(e_pred),
            "early_share_actual": float(e_act),
            "divergences": vfit["divergences"],
        })

    return pd.DataFrame(rows)