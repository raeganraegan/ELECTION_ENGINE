"""share_model.py -- Steps 0, 1, 3, 4. The hierarchical multinomial-logit model of booth vote shares.

    eta_{kbt} = m_k + alpha_{kb} + beta_{kb} * x_{kt} + d_{kt} + eps_{kbt}, with sum-to-zero across all K parties
    y_{bt} ~ Multinomial(n_{bt}, softmax(eta_{bt}))

build_design    Step 0   booth table + statewide history -> index maps and the centred statewide log-swing x
fit_share_model Step 1   MCMC (PyMC with sum-to-zero and soft saturation)                     -> ShareFit
simulate_shares Steps 3+4  posterior + a statewide share vector -> booth shares for the next election  -> ShareSim

Nothing in this file knows about polls, volumes, One Nation, plots or files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LAYERS, ModelConfig, beta_prior_table
from .data import state_matrix

log = logging.getLogger("enki")

try:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    HAS_PYMC = True
except ImportError:                                  # everything except fitting still works without PyMC
    HAS_PYMC = False

# Heuristic constants of the empirical prior scales (see empirical_scales).
_SD_D_FROM_SEAT_STD = 1.225
_SD_EPS_FROM_BOOTH_STD = 0.8


# ----------------------------------------------------------------------------- Step 0: design
@dataclass
class Design:
    parties: list
    years: list
    S_pct: pd.DataFrame                 # statewide shares (%) of the modelled parties, by election
    x: np.ndarray                       # (T, K) centred statewide log-share swing
    x_center: np.ndarray                # (K,) the mean subtracted from ln S (re-used at forecast time)
    booth_names: list
    booth_idx: np.ndarray               # (N,) booth index of each row of the booth table
    year_idx: np.ndarray                # (N,)
    Y: np.ndarray                       # (N, K) party vote counts


def build_design(df: pd.DataFrame, cfg: ModelConfig, state_df: pd.DataFrame) -> Design:
    parties = list(cfg.parties)
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) < 2:
        raise ValueError("Need at least two elections to fit the model.")
    S = state_matrix(state_df, parties, years, cfg.onp_state_denominator)
    logS = np.log(S.values / 100.0)
    x_center = logS.mean(axis=0)
    booth_names = sorted(df["booth"].unique())
    log.debug("statewide log-share swing x (centred), years %s:\n%s", years, np.round(logS - x_center, 3))
    return Design(
        parties=parties, years=years, S_pct=S, x=logS - x_center, x_center=x_center, booth_names=booth_names,
        booth_idx=df["booth"].map({b: i for i, b in enumerate(booth_names)}).to_numpy(),
        year_idx=df["year"].map({y: i for i, y in enumerate(years)}).to_numpy(),
        Y=df[[f"y_{p}" for p in parties]].to_numpy().astype(int),
    )


def empirical_scales(df: pd.DataFrame, cfg: ModelConfig) -> Tuple[float, float]:
    """Prior scales for (sigma_d, sigma_eps) read off the seat's own history; used only for a field left as None."""
    parties = list(cfg.parties)
    yearly = df.groupby("year")[["n"] + [f"y_{p}" for p in parties]].sum()
    shares = np.column_stack([yearly[f"y_{p}"] / yearly["n"] for p in parties])
    seat_std = float(np.nanstd(np.log(np.clip(shares, 1e-3, 1.0)), axis=0).mean())
    sigma_d = float(np.clip(seat_std * _SD_D_FROM_SEAT_STD, 0.05, 0.35))

    booth_stds = []
    for _, g in df.groupby("year"):
        for p in parties:
            s = g[f"y_{p}"] / g["n"]
            if s.mean() > 0 and not np.isnan(s.std()):
                booth_stds.append(float(s.std()))
    sigma_eps = float(np.clip(np.mean(booth_stds) * _SD_EPS_FROM_BOOTH_STD, 0.01, 0.15)) if booth_stds else 0.08
    log.info("Auto prior scales for %s: seat tide std %.3f -> sigma_d %.3f | booth std %.3f -> sigma_eps %.3f",
             cfg.seat_name, seat_std, sigma_d, float(np.mean(booth_stds)) if booth_stds else 0.0, sigma_eps)
    return sigma_d, sigma_eps


def tide_innovation_scale(rho, random_walk: bool):
    return 1.0 if random_walk else np.sqrt(np.maximum(1.0 - np.asarray(rho, dtype=float) ** 2, 0.0))


# ----------------------------------------------------------------------------- Step 1: fit
@dataclass
class ShareFit:
    design: Design
    draws: Dict[str, np.ndarray]
    diag: pd.DataFrame
    divergences: int
    idata: object = None

    parties = property(lambda s: s.design.parties)
    years = property(lambda s: s.design.years)
    booth_names = property(lambda s: s.design.booth_names)
    S_pct = property(lambda s: s.design.S_pct)
    x = property(lambda s: s.design.x)
    x_center = property(lambda s: s.design.x_center)
    n_draws = property(lambda s: len(s.draws["beta"]))


def stack_posterior(idata, name: str) -> np.ndarray:
    return np.moveaxis(idata.posterior[name].stack(sample=("chain", "draw")).values, -1, 0)


def diagnose_sampler(idata, summ: pd.DataFrame, label: str, cfg: ModelConfig) -> int:
    divergences = int(idata.sample_stats["diverging"].sum())
    if divergences > 0 or (summ["r_hat"] > cfg.rhat_threshold).any() or (summ["ess_bulk"] < cfg.ess_threshold).any():
        log.warning("[%s] sampler diagnostics look poor (divergences=%d, max r_hat=%.3f, min ESS=%.0f). "
                    "Do not trust the forecast until this is resolved.",
                    label, divergences, summ["r_hat"].max(), summ["ess_bulk"].min())
    return divergences


def fit_share_model(df: pd.DataFrame, cfg: ModelConfig, state_df: pd.DataFrame) -> ShareFit:
    """MCMC fit with full sum-to-zero party constraints and soft saturation bounding."""
    if not HAS_PYMC:
        raise ImportError("PyMC is required to fit the model: pip install pymc arviz")
    des = build_design(df, cfg, state_df)
    parties = des.parties
    K, T, B, N = len(parties), len(des.years), len(des.booth_names), len(df)
    n_obs = des.Y.sum(axis=1)
    
    # Initial log-ratios centered around mean across all K parties
    log_shares = np.log(np.clip(des.Y / des.Y.sum(axis=1, keepdims=True), 1e-4, 1.0))
    m0 = log_shares.mean(axis=0)
    m0 = m0 - m0.mean()  # enforce sum-to-zero on initial guess

    priors = beta_prior_table(cfg)
    auto_d, auto_eps = empirical_scales(df, cfg) if (cfg.sigma_d_median is None or cfg.sigma_eps_sd is None) else (None, None)
    sigma_d_med = cfg.sigma_d_median if cfg.sigma_d_median is not None else auto_d
    sigma_eps_sc = cfg.sigma_eps_sd if cfg.sigma_eps_sd is not None else auto_eps
    random_walk = cfg.tide_process == "random_walk"
    est_rho = (cfg.d_persist is None) and not random_walk
    coords = {"booth": des.booth_names, "year": des.years, "obs": np.arange(N), "party": parties}

    with pm.Model(coords=coords):
        # STRUCTURAL FIX 1: Sum-to-zero priors for party sensitivities and intercepts across all K parties
        mu_beta_raw = pm.Normal("mu_beta_raw", mu=0.0, sigma=0.5, shape=K)
        mu_beta = pm.Deterministic("mu_beta_vec", mu_beta_raw - mu_beta_raw.mean(), dims="party")
        
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=cfg.sigma_beta_sd, shape=K)
        beta_z = pm.Normal("beta_z", 0.0, 1.0, dims=("booth", "party"))
        beta_matrix = pm.Deterministic(
            "beta_matrix", mu_beta[None, :] + beta_z * sigma_beta[None, :], dims=("booth", "party"))

        m_raw = pm.Normal("m_raw", mu=m0, sigma=3.0, shape=K)
        m = pm.Deterministic("m", m_raw - m_raw.mean(), dims="party")
        
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=cfg.sigma_alpha_sd / np.sqrt(2.0), dims="party")
        sigma_d = pm.LogNormal("sigma_d", mu=float(np.log(sigma_d_med) - 0.5 * np.log(2.0)), sigma=cfg.sigma_d_sdlog, dims="party")
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=sigma_eps_sc / np.sqrt(2.0), dims="party")
        
        alpha_z = pm.Normal("alpha_z_raw", 0.0, 1.0, dims=("booth", "party"))
        alpha = pm.Deterministic("alpha_b", alpha_z * sigma_alpha, dims=("booth", "party"))

        if cfg.use_correlated_errors:
            chol, _, stds = pm.LKJCholeskyCov("chol_cov", eta=cfg.lkj_eta, n=K, sd_dist=pm.HalfNormal.dist(sigma=1.0), compute_corr=True)
            L = pm.Deterministic("corr_chol", chol / stds[:, None])
            d_z = pt.dot(pm.Normal("d_z_raw", 0.0, 1.0, shape=(T, K)), L.T)
            eps_z = pt.dot(pm.Normal("eps_z_raw", 0.0, 1.0, shape=(N, K)), L.T)
        else:
            d_z = pm.Normal("d_z", 0.0, 1.0, dims=("year", "party"))
            eps_z = pm.Normal("eps_z", 0.0, 1.0, dims=("obs", "party"))

        if random_walk:
            rho, innov = 1.0, 1.0
        elif est_rho:
            rho = pm.Beta("rho_d", alpha=cfg.rho_beta_a, beta=cfg.rho_beta_b)
            innov = pt.sqrt(pt.maximum(1.0 - rho ** 2, 1e-6))
        else:
            rho = float(cfg.d_persist)
            innov = float(np.sqrt(max(1.0 - rho ** 2, 1e-6)))
        steps = [sigma_d * d_z[0, :]]
        for t in range(1, T):
            steps.append(rho * steps[-1] + innov * sigma_d * d_z[t, :])
        d_t = pm.Deterministic("d_t", pt.stack(steps, axis=0), dims=("year", "party"))

        eps = pm.Deterministic("eps", eps_z * sigma_eps, dims=("obs", "party"))
        
        raw_eta = (m[None, :] + alpha[des.booth_idx]
                   + beta_matrix[des.booth_idx] * des.x[des.year_idx] + d_t[des.year_idx] + eps)
        
        # STRUCTURAL FIX 2: Soft saturation bounding (Carrying capacity tanh activation wrapper)
        # Prevents extreme log-odds divergence in strongholds while preserving local gradients.
        max_logit_bound = 3.5  # Corresponds to roughly ~97% probability ceiling per channel before softmax normalization
        eta = pm.Deterministic("eta", max_logit_bound * pt.tanh(raw_eta / max_logit_bound))

        pm.Multinomial("y_obs", n=n_obs, p=pm.math.softmax(eta, axis=-1), observed=des.Y)
        idata = pm.sample(draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, target_accept=cfg.target_accept,
                          random_seed=cfg.seed, return_inferencedata=True)

    summ = az.summary(idata, var_names=["mu_beta_raw", "sigma_beta", "m", "sigma_alpha", "sigma_d", "sigma_eps"] + (["rho_d"] if est_rho else []))
    divergences = diagnose_sampler(idata, summ, "share model", cfg)
    
    D = stack_posterior(idata, "sigma_d").shape[0]
    draws = {"beta": stack_posterior(idata, "mu_beta_vec"), "beta_matrix": stack_posterior(idata, "beta_matrix"), "m": stack_posterior(idata, "m"),
             "sigma_d": stack_posterior(idata, "sigma_d"), "sigma_eps": stack_posterior(idata, "sigma_eps"),
             "sigma_alpha": stack_posterior(idata, "sigma_alpha"), "sigma_beta": stack_posterior(idata, "sigma_beta"),
             "alpha": stack_posterior(idata, "alpha_b"), "d": stack_posterior(idata, "d_t"), "eps": stack_posterior(idata, "eps"),
             "rho": stack_posterior(idata, "rho_d") if est_rho else np.full(D, 1.0 if random_walk else float(cfg.d_persist))}
    if cfg.use_correlated_errors:
        draws["corr_chol"] = stack_posterior(idata, "corr_chol")
    return ShareFit(des, draws, summ.reset_index().rename(columns={"index": "parameter"}), divergences, idata)


# ----------------------------------------------------------------------------- Steps 3 + 4: simulate
@dataclass
class ShareSim:
    p: np.ndarray
    gamma: np.ndarray
    x_next: np.ndarray
    bx: np.ndarray
    d_next: np.ndarray
    lin_party: np.ndarray


def _freeze_params(dr: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {k: np.broadcast_to(v.mean(axis=0, keepdims=True), v.shape) for k, v in dr.items()}
    if "corr_chol" in out:
        L = out["corr_chol"]
        out["corr_chol"] = L / np.linalg.norm(L, axis=2, keepdims=True)
    return out


def check_layers(freeze: Iterable[str]) -> FrozenSet[str]:
    freeze = frozenset(freeze)
    bad = freeze - set(LAYERS)
    if bad:
        raise ValueError(f"unknown uncertainty layer(s) {sorted(bad)}; choose from {LAYERS}")
    return freeze


def simulate_shares(fit: ShareFit, S_next_pct: np.ndarray, cfg: ModelConfig, rng: np.random.Generator,
                    freeze: Iterable[str] = ()) -> ShareSim:
    freeze = check_layers(freeze)
    dr = _freeze_params(fit.draws) if "params" in freeze else fit.draws
    D, B, K = dr["alpha"].shape

    S = np.clip(np.asarray(S_next_pct, dtype=float), cfg.share_clip_floor, None)
    S = S / 100.0 if cfg.onp_state_denominator else S / S.sum(axis=1, keepdims=True)
    x_next = np.log(S) - fit.x_center

    z_d, z_eps = rng.standard_normal((D, K)), rng.standard_normal((D, B, K))
    if "corr_chol" in dr:
        z_d = np.einsum("dkj,dj->dk", dr["corr_chol"], z_d)
        z_eps = np.einsum("dkj,dbj->dbk", dr["corr_chol"], z_eps)
    rho = np.asarray(dr["rho"], dtype=float)[:, None]
    innov = tide_innovation_scale(rho, cfg.tide_process == "random_walk")
    d_next = rho * dr["d"][:, -1, :] + innov * dr["sigma_d"] * z_d * (0.0 if "tide" in freeze else 1.0)

    bx = dr["beta"] * x_next
    lin_party = bx + d_next
    lin = dr["beta_matrix"] * x_next[:, None, :] + d_next[:, None, :]
    
    raw_eta = dr["m"][:, None, :] + dr["alpha"] + lin
    if cfg.include_eps:
        raw_eta = raw_eta + dr["sigma_eps"][:, None, :] * z_eps * (0.0 if "booth" in freeze else 1.0)
        
    # Apply matching soft saturation tanh ceiling at simulation time
    max_logit_bound = 3.5
    eta = max_logit_bound * np.tanh(raw_eta / max_logit_bound)
    
    eta = eta - eta.max(axis=2, keepdims=True)
    e = np.exp(eta)
    
    # For gamma compatibility (difference from reference / last party or mean)
    Km = K - 1
    return ShareSim(e / e.sum(axis=2, keepdims=True), lin_party[:, :Km] - lin_party[:, Km:], x_next, bx, d_next, lin_party)