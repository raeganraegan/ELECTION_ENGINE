"""forecast.py -- runs Steps 0-7 once and assembles the result tables.

    fit (1) -> polls (2) -> shares (3,4) -> ONP (6) -> aggregate with volumes (5,7)

`run_forecast` is the only function here that strings the steps together; each step lives in its own module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .config import LAYERS, SCENARIOS, ModelConfig, PollInput
from .onp import apply_onp
from .pipeline import Stages, build_trace
from .polls import draw_statewide, summarise_polls
from .share_model import fit_share_model, simulate_shares
from .volume_model import resolve_volumes
from .plots import seat_distribution_plot, backtest_distribution_validation_plot

log = logging.getLogger("enki")
ATTRIBUTION_SEED_OFFSET = 101


@dataclass
class Forecast:
    params: Dict[str, float]
    booths: pd.DataFrame                  # per-booth fitted effects (alpha, eps)
    tide: pd.DataFrame                    # per-election statewide swing and seat tide
    booth_forecast: pd.DataFrame          # per-booth forecast of the target party (worst / average / best, mean, p05, p95)
    seat: pd.DataFrame                    # seat-wide target-party scenarios
    draws_df: pd.DataFrame                # seat-wide share of every party, one row per posterior draw
    booth_draws_df: pd.DataFrame          # target party's booth shares, one row per draw
    booth_names: list                     # booths that exist at the forecast
    volume_matrix: np.ndarray             # (D, B) formal votes
    prepared_df: pd.DataFrame
    booth_party_draws: np.ndarray         # (D, B, K[+1]) percent
    party_names: list
    diagnostics: Optional[pd.DataFrame] = None
    attribution: Optional[pd.DataFrame] = None
    trace: Optional[dict] = None
    backtest_result: Optional[object] = None # BacktestResult if run

    def save(self, folder: Union[str, Path], cfg: Optional[ModelConfig] = None) -> None:
        out = Path(folder)
        live_out = out / "live_2027"
        live_out.mkdir(parents=True, exist_ok=True)
        
        pd.Series(self.params, name="value").to_csv(live_out / "params.csv")
        self.booths.to_csv(live_out / "booths.csv", index=False)
        self.tide.to_csv(live_out / "tide.csv", index=False)
        self.booth_forecast.to_csv(live_out / "forecast.csv", index=False)
        self.seat.to_csv(live_out / "seat.csv", index=False)
        if self.diagnostics is not None:
            self.diagnostics.to_csv(live_out / "diagnostics.csv", index=False)
        if self.attribution is not None:
            self.attribution.to_csv(live_out / "uncertainty_attribution.csv", index=False)
        np.savez_compressed(live_out / "volume_draws.npz", names=np.array(self.booth_names), V=self.volume_matrix)
        np.savez_compressed(live_out / "booth_party_draws.npz", booth_party_draws=self.booth_party_draws,
                            booths=np.array(self.booth_names), parties=np.array(self.party_names))
        
        # Save live distribution plot
        seat_distribution_plot(self, cfg.party if cfg else self.party_names[0], live_out / "seat_distribution.png")

        # Save backtests into structured per-year folders if available
        if self.backtest_result is not None:
            bt_root = out / "backtests"
            bt_root.mkdir(parents=True, exist_ok=True)
            tbl = self.backtest_result.table
            
            for tgt_year, draws_dict in self.backtest_result.draws.items():
                year_dir = bt_root / str(tgt_year)
                year_dir.mkdir(parents=True, exist_ok=True)
                
                # Save per-year backtest row summary JSON/CSV
                row_df = tbl[tbl["target"] == tgt_year]
                if not row_df.empty:
                    row_df.to_json(year_dir / "backtest_summary.json", orient="records", indent=2)
                    row_df.to_csv(year_dir / "backtest_summary.csv", index=False)
                
                # Save per-year npz draws
                np.savez_compressed(
                    year_dir / f"backtest_draws_{tgt_year}.npz",
                    booths=np.array(draws_dict["booths"]),
                    parties=np.array(draws_dict["parties"]),
                    booth_party_draws=draws_dict["booth_party_draws"],
                    actual_party_votes=draws_dict["actual_party_votes"],
                    actual_n=draws_dict["actual_n"]
                )
                
                # Generate validation distribution plot with actual outcome dashed lines
                backtest_distribution_validation_plot(draws_dict, cfg.party if cfg else "ALP", year_dir / "seat_distribution_validation.png")


# ----------------------------------------------------------------------------- uncertainty attribution
def uncertainty_attribution(st: Stages, k0: int) -> pd.DataFrame:
    """Which layer makes the seat band wide? Re-run the forecast with one layer frozen at a time (same random stream)."""
    cfg, fit, stw, vol, keep = st.cfg, st.fit, st.stw, st.vol, st.keep
    D = fit.n_draws
    S_fixed = np.tile(np.asarray(stw.central, dtype=float), (D, 1))
    V_fixed = np.tile(np.median(vol.V, axis=0), (D, 1))

    def run(freeze):
        rng = np.random.default_rng(cfg.seed + ATTRIBUTION_SEED_OFFSET)
        p = simulate_shares(fit, S_fixed if "poll" in freeze else stw.S, cfg, rng, freeze).p
        Vm = (V_fixed if "volumes" in freeze else vol.V)[:, keep]
        return 100.0 * np.einsum("db,db->d", Vm, p[:, keep, k0]) / Vm.sum(axis=1) * st.onp.target_scale

    runs = [("all layers on", ())] + [(f"switch off: {l}", (l,)) for l in LAYERS] + [("all layers off (point forecast)", LAYERS)]
    rows, base_w = [], None
    for name, freeze in runs:
        x = run(freeze)
        p05, p50, p95 = np.percentile(x, [5, 50, 95])
        w = float(p95 - p05)
        base_w = w if base_w is None else base_w
        rows.append({"scenario": name, "p05": p05, "p50": p50, "p95": p95, "width_90": w, "sd": float(x.std()),
                     "width_cut_pts": base_w - w, "width_cut_pct": 100.0 * (base_w - w) / base_w if base_w else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tables
def _seat_table(st: Stages, k0: int) -> pd.DataFrame:
    lo, mid, hi = (float(v) for v in st.cfg.scenario_pct)
    Km = len(st.cfg.parties) - 1
    target = st.seat_all[:, k0]
    sq = np.percentile(target, [lo, mid, hi])
    tq = np.percentile(st.sim.gamma[:, k0], [lo, mid, hi]) if k0 < Km else np.full(3, np.nan)
    total = float(np.median(st.vol.V[:, st.keep].sum(axis=1)))
    return pd.DataFrame([{
        "scenario": name, "gamma_next": float(tq[i]), "projected_primary_votes": round(sq[i] / 100.0 * total),
        "total_formal_votes": round(total), "primary_share_pct": float(sq[i]),
        "seat_p05": float(np.percentile(target, 5)), "seat_p95": float(np.percentile(target, 95)),
    } for i, name in enumerate(SCENARIOS)])


def _booth_forecast_table(st: Stages, k0: int) -> pd.DataFrame:
    lo, mid, hi = (float(v) for v in st.cfg.scenario_pct)
    P = 100.0 * st.sim.p[:, st.keep, k0] * st.onp.target_scale[:, None]
    q = np.percentile(P, [5, lo, mid, hi, 95], axis=0)
    last_year = int(st.df["year"].max())
    last_share = st.df[st.df["year"] == last_year].set_index("booth")["share"].reindex(st.kept_names).values
    return pd.DataFrame({"booth": st.kept_names, "worst": q[1], "average": q[2], "best": q[3], "mean": P.mean(axis=0),
                         "p05": q[0], "p95": q[4], "formal_weight": np.median(st.vol.V[:, st.keep], axis=0),
                         "last_formal": st.last_n.reindex(st.kept_names).values, "last_share": last_share})


def _booth_effects_table(st: Stages, k0: int) -> pd.DataFrame:
    dr, parties, Km = st.fit.draws, list(st.cfg.parties), len(st.cfg.parties) - 1
    names, df = st.fit.booth_names, st.df
    tbl = pd.DataFrame({"booth": names, "elections": df.groupby("booth")["year"].nunique().reindex(names).astype(int).values,
                        "formal_votes_total": df.groupby("booth")["n"].sum().reindex(names).astype(int).values})
    for j, p in enumerate(parties[:Km]):
        a = dr["alpha"][:, :, j] - dr["alpha"][:, :, Km]            # log-ratio against the reference party
        tbl[f"alpha_{p}"], tbl[f"alpha_{p}_sd"] = a.mean(0), a.std(0)
    if k0 < Km:
        a = dr["alpha"][:, :, k0] - dr["alpha"][:, :, Km]
        tbl["alpha_b"], tbl["alpha_b_p05"], tbl["alpha_b_p95"] = a.mean(0), np.percentile(a, 5, axis=0), np.percentile(a, 95, axis=0)
        eps = df.assign(eps_logratio=(dr["eps"][:, :, k0] - dr["eps"][:, :, Km]).mean(0)).pivot(index="booth", columns="year", values="eps_logratio")
        eps.columns = [f"eps_{c}" for c in eps.columns]
        tbl = tbl.merge(eps.reset_index(), on="booth", how="left")
    return tbl


def _tide_table(st: Stages) -> pd.DataFrame:
    dr, fit, parties, Km = st.fit.draws, st.fit, list(st.cfg.parties), len(st.cfg.parties) - 1
    tbl = pd.DataFrame({"year": fit.years})
    for j, p in enumerate(parties[:Km]):
        bx = float(dr["beta"][:, j].mean()) * fit.x[:, j] - float(dr["beta"][:, Km].mean()) * fit.x[:, Km]
        dm = (dr["d"][:, :, j] - dr["d"][:, :, Km]).mean(0)
        tbl[f"state_{p}"], tbl[f"beta_x_{p}"], tbl[f"d_{p}"], tbl[f"gamma_{p}"] = fit.S_pct[p].values, bx, dm, bx + dm
    return tbl


def _params(st: Stages, k0: int) -> Dict[str, float]:
    cfg, dr, parties = st.cfg, st.fit.draws, list(st.cfg.parties)
    lo, mid, hi = (float(v) for v in cfg.scenario_pct)
    s = st.fit.diag.set_index("parameter")
    p = {"S_next": float(st.stw.central[k0]), "S_next_low": float(st.stw.target.low), "S_next_high": float(st.stw.target.high),
         "S_next_se_used": float(st.stw.se_parts[3]), "beta": float(dr["beta"][:, k0].mean()), "beta_sd": float(dr["beta"][:, k0].std()),
         "d_persist": float(dr["rho"].mean()), "d_persist_sd": float(dr["rho"].std()),
         "rhat_max": float(s["r_hat"].max()), "ess_bulk_min": float(s["ess_bulk"].min()),
         "divergences": st.fit.divergences, "posterior_draws": st.fit.n_draws, "booths_in_fit": len(st.fit.booth_names),
         "booths_forecast": len(st.keep), "expected_total_formal_votes": float(np.median(st.vol.V[:, st.keep].sum(axis=1))),
         "scenario_percentiles": f"{lo:g}/{mid:g}/{hi:g}"}
    for j, q in enumerate(parties):
        p.update({f"sigma_alpha_{q}": float(dr["sigma_alpha"][:, j].mean()), f"sigma_eps_{q}": float(dr["sigma_eps"][:, j].mean()),
                  f"sigma_d_{q}": float(dr["sigma_d"][:, j].mean()), f"S_next_{q}": float(st.polls.central[j]),
                  f"beta_{q}": float(dr["beta"][:, j].mean()), f"beta_{q}_sd": float(dr["beta"][:, j].std())})
    if st.stw.onp is not None:
        p["onp_state_share_poll"] = float(np.mean(st.stw.onp))
    if st.onp.mode != "none":
        p["onp_mode"] = st.onp.mode
        p["onp_seat_share_mean"] = float(st.seat_all[:, -1].mean())
    if st.vol.vfit is not None:
        p["volume_divergences"] = st.vol.vfit.divergences
        p["volume_rhat_max"] = float(st.vol.vfit.diag["r_hat"].max())
    return p


# ----------------------------------------------------------------------------- the forecast
def run_forecast(df: pd.DataFrame, state_df: pd.DataFrame, poll: Union[PollInput, pd.DataFrame], cfg: ModelConfig,
                 projected_formal_votes: Optional[pd.Series] = None, target_year: Optional[int] = None,
                 run_attribution: bool = True, record_trace: bool = True) -> Forecast:
    """`df` is the prepared booth table (data.prepare_booths)."""
    parties = list(cfg.parties)
    k0 = parties.index(cfg.party)
    rng = np.random.default_rng(cfg.seed)

    fit = fit_share_model(df, cfg, state_df)                                       # Step 1
    D = fit.n_draws
    polls = summarise_polls(poll, cfg, state_df)                                   # Step 2
    stw = draw_statewide(polls, cfg, D, rng)
    sim = simulate_shares(fit, stw.S, cfg, rng)                                    # Steps 3-4
    onp = apply_onp(sim.p, stw.onp, parties, k0, cfg, rng)                         # Step 6
    if cfg.onp_mode == "brian":
        take = cfg.brian_transfer
        log.info("BrianMode ON: ONP takes %s of each party's vote.", take if take else "the default share")
    vol = resolve_volumes(df, fit.booth_names, cfg, D, rng, projected_formal_votes, target_year)   # Step 5

    keep = np.where(vol.active)[0]
    kept_names = [fit.booth_names[i] for i in keep]
    last_year = int(df["year"].max())
    last_n = df[df["year"] == last_year].set_index("booth")["n"].reindex(fit.booth_names)
    Vk, Pk = vol.V[:, keep], onp.p[:, keep, :]
    seat_all = 100.0 * np.einsum("db,dbk->dk", Vk, Pk) / Vk.sum(axis=1, keepdims=True)         # Step 7
    st = Stages(cfg, df, fit, polls, stw, sim, onp, vol, keep, kept_names, seat_all, last_n)

    bt_result = None        # attached by api.ENKI after its single backtest run (the backtest is a RunOptions concern, not a model one)

    return Forecast(
        params=_params(st, k0), booths=_booth_effects_table(st, k0), tide=_tide_table(st),
        booth_forecast=_booth_forecast_table(st, k0), seat=_seat_table(st, k0),
        draws_df=pd.DataFrame(seat_all, columns=onp.out_parties),
        booth_draws_df=pd.DataFrame(100.0 * sim.p[:, keep, k0] * onp.target_scale[:, None], columns=kept_names),
        booth_names=kept_names, volume_matrix=Vk, prepared_df=df, booth_party_draws=Pk * 100.0, party_names=onp.out_parties,
        diagnostics=fit.diag, attribution=uncertainty_attribution(st, k0) if run_attribution else None,
        trace=build_trace(st) if record_trace else None,
        backtest_result=bt_result)