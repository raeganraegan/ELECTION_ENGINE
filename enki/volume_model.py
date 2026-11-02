"""volume_model.py -- Step 5. How many formal votes each booth gets.

Three ways, in priority order (resolve_volumes):
  projected_formal_votes   you supply the booth vector; it is used as is in every draw
  registered_electors      electors x turnout x (1 - informal rate), split by the last election's booth shares
  (neither)                a Dirichlet-multinomial model of booth shares + a plug-in growth model of the seat total
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import ModelConfig
from .share_model import HAS_PYMC, stack_posterior, diagnose_sampler

if HAS_PYMC:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

log = logging.getLogger("enki")


@dataclass
class VolumeFit:
    names: list
    years: list
    btype: np.ndarray                    # (B,) 1 = early-voting style booth
    t0: int                              # index of the first election with early booths
    totals: np.ndarray                   # (T,) seat formal total per election
    divergences: int
    diag: pd.DataFrame
    draws: dict                          # a (S,B) | tau (S,T) | kappa (S,)


@dataclass
class VolumeResult:
    V: np.ndarray                        # (D, B) formal votes
    mode: str                            # "projected" | "electors" | "modelled"
    active: np.ndarray                   # (B,) bool: booths that exist at the forecast
    vfit: Optional[VolumeFit] = None


def projected_votes_from_electors(df: pd.DataFrame, cfg: ModelConfig) -> pd.Series:
    """Booth-level projected FORMAL votes from enrolment. `df` is the prepared booth table."""
    if cfg.registered_electors is None:
        raise ValueError("registered_electors is required (set cfg.registered_electors).")
    total, informal = df["total_votes"].sum(), df["informal_votes"].sum()
    informal_rate = informal / total if total > 0 else cfg.informal_rate_fallback
    latest = df[df["year"] == df["year"].max()].set_index("booth")["n"]
    return latest / latest.sum() * (cfg.registered_electors * cfg.expected_turnout_rate * (1.0 - informal_rate))


def fit_volume_model(df: pd.DataFrame, cfg: ModelConfig) -> VolumeFit:
    """Booth volume as a SHARE of the seat's formal votes: n_t ~ DirichletMultinomial(T_t, kappa * softmax(a + tau_t * early))."""
    if not HAS_PYMC:
        raise ImportError("PyMC is required to fit the volume model: pip install pymc arviz")
    names = sorted(df["booth"].unique())
    years = sorted(int(y) for y in df["year"].unique())
    B, T = len(names), len(years)
    C = df.pivot(index="year", columns="booth", values="n").reindex(index=years, columns=names).values
    present = ~np.isnan(C)
    btype = df.groupby("booth")["btype"].max().reindex(names).fillna(0).values.astype(float)
    has_early = bool(btype.sum() > 0)
    t0 = next((ti for ti in range(T) if present[ti][btype == 1].any()), T)    # first election with an early booth
    n_free = max(T - 1 - t0, 0)
    has_free = has_early and n_free > 0
    with pm.Model(coords={"booth": names}):
        s_a = pm.HalfNormal("sigma_a", sigma=cfg.vol_sigma_a_sd)
        a = pm.Deterministic("a", pm.Normal("a_z", 0.0, 1.0, dims="booth") * s_a, dims="booth")
        if has_free:
            tau = pm.Deterministic("tau", pt.concatenate([pt.zeros(t0 + 1), pm.Normal("tau_free", 0.0, cfg.vol_tau_sd, shape=n_free)]))
        else:
            tau = pt.zeros(T)
        kappa = pm.Gamma("kappa", alpha=cfg.vol_kappa_alpha, beta=cfg.vol_kappa_beta)
        for ti, yr in enumerate(years):
            idx = np.where(present[ti])[0]
            pm.DirichletMultinomial(f"n_{yr}", n=int(C[ti, idx].sum()), a=kappa * pm.math.softmax(a[idx] + tau[ti] * btype[idx]),
                                    observed=C[ti, idx].astype(int))
        idata = pm.sample(draws=(cfg.volume_draws or cfg.draws), tune=max(cfg.tune, cfg.volume_tune_floor), chains=cfg.chains,
                          target_accept=cfg.volume_target_accept, random_seed=cfg.seed, return_inferencedata=True, progressbar=False)
    summ = az.summary(idata, var_names=["sigma_a", "kappa"] + (["tau"] if has_free else []))
    div = diagnose_sampler(idata, summ, "volume model", cfg)
    S = idata.posterior["kappa"].stack(sample=("chain", "draw")).values.shape[-1]
    return VolumeFit(names, years, btype, t0, np.nansum(C, axis=1), div, summ,
                     {"a": stack_posterior(idata, "a"), "tau": stack_posterior(idata, "tau") if has_free else np.zeros((S, T)), "kappa": stack_posterior(idata, "kappa")})


def simulate_volumes(vfit: VolumeFit, D: int, cfg: ModelConfig, rng: np.random.Generator,
                     target_year: Optional[int] = None, active: Optional[np.ndarray] = None) -> np.ndarray:
    """Simulated formal votes (D x booths). Booths with active=False get 0."""
    dr, names, years = vfit.draws, vfit.names, vfit.years
    B = len(names)
    act = np.ones(B, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    ai = np.where(act)[0]
    idx = rng.integers(0, len(dr["kappa"]), size=D)
    a, tau, kappa = dr["a"][idx][:, ai], dr["tau"][idx], dr["kappa"][idx]
    bt = vfit.btype[ai]
    totals = np.asarray(vfit.totals, dtype=float)
    t_year = target_year if target_year is not None else int(round(years[-1] + cfg.election_cycle_years))
    if t_year in years:
        tau_t = tau[:, years.index(t_year)]
        Tn = np.full(D, totals[years.index(t_year)])
    else:
        k = max(1, int(round((t_year - years[-1]) / cfg.election_cycle_years)))        # election cycles ahead
        steps = np.diff(tau[:, min(vfit.t0, tau.shape[1] - 1):], axis=1)               # early-vote log-odds steps since early voting began
        if cfg.volume_drift == "none" or steps.shape[1] == 0:
            drift = np.zeros(D)
        elif cfg.volume_drift == "last":
            drift = steps[:, -1]
        else:
            drift = steps.mean(axis=1)
        sd = np.maximum(steps.std(axis=1, ddof=1) if steps.shape[1] > 1 else 0.0, cfg.volume_step_floor)
        tau_t = tau[:, -1] + k * drift + np.sqrt(k) * sd * rng.standard_normal(D)
        lg = np.diff(np.log(totals))                                                    # plug-in growth of the seat total, not MCMC
        g_mu = float(lg.mean()) if len(lg) else 0.0
        g_sd = max(float(lg.std(ddof=1)) if len(lg) > 1 else 0.0, cfg.volume_total_sd_floor)
        Tn = totals[-1] * np.exp(k * g_mu + np.sqrt(k) * g_sd * rng.standard_normal(D))
    eta = a + tau_t[:, None] * bt[None, :]
    eta = eta - eta.max(axis=1, keepdims=True)
    pi = np.exp(eta)
    pi = pi / pi.sum(axis=1, keepdims=True)
    gam = rng.gamma(kappa[:, None] * pi)
    counts = rng.multinomial(np.round(Tn).astype(int), gam / gam.sum(axis=1, keepdims=True))
    V = np.zeros((D, B))
    V[:, ai] = counts
    return V


def resolve_volumes(df: pd.DataFrame, names: list, cfg: ModelConfig, D: int, rng: np.random.Generator,
                    projected: Optional[pd.Series] = None, target_year: Optional[int] = None) -> VolumeResult:
    """Formal votes per booth (D, B) for the booths in `names`, by whichever of the three paths applies."""
    last_year = int(df["year"].max())
    last_n = df[df["year"] == last_year].set_index("booth")["n"].reindex(names)
    if projected is not None or cfg.registered_electors is not None:
        fixed = projected if projected is not None else projected_votes_from_electors(df, cfg)
        w = fixed.reindex(names).fillna(last_n).fillna(0.0).values.astype(float)
        return VolumeResult(np.tile(w, (D, 1)), "projected" if projected is not None else "electors", w > 0)
    vfit = fit_volume_model(df, cfg)
    if vfit.names != names:
        raise RuntimeError("Booth order differs between share and volume models.")
    active = last_n.fillna(0.0).values > 0
    return VolumeResult(simulate_volumes(vfit, D, cfg, rng, target_year=target_year, active=active), "modelled", active, vfit)
