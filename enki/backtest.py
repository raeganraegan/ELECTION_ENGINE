"""backtest.py -- walk-forward validation (both functions take the PREPARED booth table, data.prepare_booths). Refit on the elections BEFORE each target, then predict the target.

What the share backtest conditions on (so read its numbers accordingly):
  * the TRUE statewide result of the target election replaces the poll layer, so it tests the seat-local model
    (booth effects, seat tide, booth noise), not poll error;
  * the TRUE booth volumes of the target election weight the seat total, so it tests shares, not turnout.
Booths that did not exist before the target are not scored. The volume backtest is separate (backtest_volumes).

Metrics per target: seat error; booth MAE / RMSE / CRPS; empirical coverage of the 50/80/90/95% predictive intervals
(predictive draws include binomial voting noise); and, on the booths present in both elections, the same errors for two
naive baselines (no-change and statewide-swing) so the model has something to beat.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import state_matrix
from .share_model import fit_share_model, simulate_shares
from .volume_model import fit_volume_model, simulate_volumes

log = logging.getLogger("enki")
INTERVALS = (50, 80, 90, 95)


@dataclass
class BacktestResult:
    table: pd.DataFrame
    draws: Dict[int, dict]            # per target year: booths, parties, booth_party_draws (D,B,K), actual_party_votes, actual_n


# ----------------------------------------------------------------------------- scoring
def interval_coverage(draws: np.ndarray, actual: np.ndarray, levels: Sequence[int] = INTERVALS) -> Dict[str, float]:
    """Fraction of `actual` (N,) inside the central `level`% interval of `draws` (D, N), and that minus the nominal level."""
    out = {}
    for lv in levels:
        a = (100.0 - lv) / 200.0
        lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)], axis=0)
        cov = float(np.mean((actual >= lo) & (actual <= hi)))
        out[f"cover_{lv}"], out[f"bias_{lv}"] = cov, cov - lv / 100.0
    return out


def crps_ensemble(draws: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score per column: E|X - y| - 0.5 E|X - X'|, from D draws (D, N) and actuals (N,).
    A proper scoring rule that rewards sharp AND calibrated forecasts, in the units of the data; a point forecast scores its MAE."""
    D = draws.shape[0]
    srt = np.sort(draws, axis=0)
    spread = 2.0 / D ** 2 * ((2 * np.arange(1, D + 1) - D - 1)[:, None] * srt).sum(axis=0)     # E|X - X'|
    return np.abs(draws - actual[None, :]).mean(axis=0) - 0.5 * spread


# ----------------------------------------------------------------------------- share backtest
def _baselines(row: dict, df_all: pd.DataFrame, S_all: pd.DataFrame, cfg: ModelConfig, tgt: int, last_y: int,
               t: pd.DataFrame, act: np.ndarray, n_t: np.ndarray, med: np.ndarray, crps: np.ndarray) -> None:
    """Model vs no-change vs statewide-swing, all on the booths that voted in both the last and the target election."""
    last = df_all[df_all["year"] == last_y].set_index("booth")["share"]
    mk = np.array([b in last.index for b in t["booth"]])
    row["common_n"] = int(mk.sum())
    if not mk.any():
        return
    prev = last.reindex(t["booth"].values[mk]).values
    a, n, m = act[mk], n_t[mk], med[mk]
    swing = float(S_all.loc[tgt, cfg.party] - S_all.loc[last_y, cfg.party])
    errs = {"model": m - a, "nochange": prev - a, "statewide_swing": prev + swing - a}
    wavg = lambda v: float((v * n).sum() / n.sum())          # volume-weighted: what the seat total feels
    row.update({"model_MAE_common": float(np.abs(errs["model"]).mean()), "nochange_MAE": float(np.abs(errs["nochange"]).mean()),
                "statewide_swing_MAE": float(np.abs(errs["statewide_swing"]).mean()), "model_CRPS_common": float(crps[mk].mean()),
                "model_seat_err_common": wavg(errs["model"]), "nochange_seat_err": wavg(errs["nochange"]),
                "statewide_swing_seat_err": wavg(errs["statewide_swing"])})


def backtest_shares(df_all: pd.DataFrame, state_df: pd.DataFrame, cfg: ModelConfig,
                    targets: Optional[Sequence[int]] = None, return_draws: bool = False) -> BacktestResult:
    years = sorted(df_all["year"].unique())
    parties = list(cfg.parties)
    k0 = parties.index(cfg.party)
    S_all = state_matrix(state_df, parties, years, cfg.onp_state_denominator)
    lo_p, _, hi_p = (float(x) for x in cfg.scenario_pct)

    rows, draws_out = [], {}
    for tgt in (targets if targets is not None else years[2:]):
        tgt = int(tgt)
        rng = np.random.default_rng(cfg.seed + tgt)           # per-target stream: re-running one target reproduces its row
        fit = fit_share_model(df_all[df_all["year"] < tgt], cfg, state_df)
        D = fit.n_draws
        p = simulate_shares(fit, np.tile(S_all.loc[tgt].values, (D, 1)), cfg, rng).p

        t = df_all[(df_all["year"] == tgt) & df_all["booth"].isin(fit.booth_names)]
        col = [fit.booth_names.index(b) for b in t["booth"]]
        n_t, y_t = t["n"].values, t["y"].values
        if return_draws:
            draws_out[tgt] = {"booths": [str(b) for b in t["booth"]], "parties": parties, "booth_party_draws": 100.0 * p[:, col, :],
                              "actual_party_votes": np.column_stack([t[f"y_{q}"].values for q in parties]), "actual_n": n_t.astype(float)}

        P = 100.0 * p[:, col, k0]
        med, act = np.median(P, axis=0), 100.0 * y_t / n_t
        rep = 100.0 * rng.binomial(n_t[None, :], p[:, col, k0]) / n_t[None, :]       # predictive draws incl. voting noise
        calib = interval_coverage(rep, act)
        crps = crps_ensemble(rep, act)
        seat = (P * n_t).sum(axis=1) / n_t.sum()
        seat_act = float((act * n_t).sum() / n_t.sum())
        s_lo, s50, s_hi = np.percentile(seat, [lo_p, 50, hi_p])
        err = med - act

        row = {"target": tgt, "seat_pred_median": s50, "seat_actual": seat_act, "seat_error": s50 - seat_act,
               "seat_lo": s_lo, "seat_hi": s_hi, "seat_in_interval": bool(s_lo <= seat_act <= s_hi),
               "booth_MAE": float(np.abs(err).mean()), "booth_RMSE": float(np.sqrt((err ** 2).mean())), "booth_CRPS": float(crps.mean()),
               **{f"booth_cover_{lv}": calib[f"cover_{lv}"] for lv in INTERVALS}, "calib_bias_90": calib["bias_90"],
               "n_booths": len(t), "divergences": fit.divergences, "rhat_max": float(fit.diag["r_hat"].max())}
        for j, q in enumerate(parties):
            sj = (100.0 * p[:, col, j] * n_t).sum(axis=1) / n_t.sum()
            row[f"seat_err_{q}"] = float(np.median(sj) - 100.0 * t[f"y_{q}"].sum() / n_t.sum())
        _baselines(row, df_all, S_all, cfg, tgt, max(int(y) for y in years if y < tgt), t, act, n_t, med, crps)
        rows.append(row)
        del fit, p, P, rep
        gc.collect()
    return BacktestResult(pd.DataFrame(rows), draws_out)


# ----------------------------------------------------------------------------- volume backtest
def backtest_volumes(df: pd.DataFrame, cfg: ModelConfig, sims: int = 4000,
                     targets: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Predict each election's booth volumes from the elections before it."""
    years = sorted(df["year"].unique())
    rows = []
    for tgt in (targets if targets is not None else years[2:]):
        tgt = int(tgt)
        rng = np.random.default_rng(cfg.seed + tgt)
        train = df[df["year"] < tgt]
        vfit = fit_volume_model(train, cfg)
        prev_all = train[train["year"] == int(train["year"].max())].set_index("booth")["n"]
        active = np.array([b in prev_all.index for b in vfit.names])
        V = simulate_volumes(vfit, sims, cfg, rng, target_year=tgt, active=active)

        year_df = df[df["year"] == tgt]
        t = year_df.set_index("booth")["n"]
        idx = [i for i, b in enumerate(vfit.names) if active[i] and b in t.index]
        names = [vfit.names[i] for i in idx]
        act = t.reindex(names).values.astype(float)
        P = V[:, idx]
        med = np.median(P, axis=0)
        lo, hi = np.percentile(P, [5, 95], axis=0)
        prev = prev_all.reindex(names).values.astype(float)
        tot, total_act = V.sum(axis=1), float(year_df["n"].sum())
        early = vfit.btype.astype(bool)
        has_early = bool(early.any())
        rows.append({
            "target": tgt, "n_booths": len(names),
            "booth_MAPE_pct": float(np.mean(np.abs(med - act) / act) * 100), "nochange_MAPE_pct": float(np.mean(np.abs(prev - act) / act) * 100),
            "booth_cover_90": float(((act >= lo) & (act <= hi)).mean()),
            "total_pred_median": float(np.median(tot)), "total_actual": total_act,
            "total_error_pct": float((np.median(tot) / total_act - 1) * 100),
            "total_in_90": bool(np.percentile(tot, 5) <= total_act <= np.percentile(tot, 95)),
            "early_share_pred": float(np.median(V[:, early].sum(axis=1) / np.maximum(tot, 1))) if has_early else np.nan,
            "early_share_actual": float(year_df[year_df["btype"] == 1]["n"].sum() / total_act) if has_early else np.nan,
            "divergences": vfit.divergences})
    return pd.DataFrame(rows)
