"""polls.py -- Step 2. Poll table -> statewide party-share draws S^(i).

summarise_polls : poll table (or one PollInput) -> weighted average + sampling error   (deterministic)
draw_statewide  : + house effect + time-horizon error -> Dirichlet draws                (random)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import ModelConfig, PollInput
from .data import state_matrix

log = logging.getLogger("enki")


@dataclass
class PollSummary:
    central: np.ndarray                  # (K,) % for the modelled parties; with ONP present, normalised so parties + ONP = 100
    onp_central: Optional[float]         # % or None
    target: PollInput                    # the target party's poll average and margin of error (points)
    table: Optional[pd.DataFrame]        # polls that entered the average, with their weights


@dataclass
class StatewideDraws:
    S: np.ndarray                        # (D, K) %
    onp: Optional[np.ndarray]            # (D,) % or None
    central: np.ndarray
    target: PollInput
    se_parts: np.ndarray                 # [poll sampling se, house sd, time-horizon sd, total se], in points
    conc: float                          # Dirichlet concentration


def _pct_scale(polls: pd.DataFrame, cols: Sequence[str]) -> float:
    """Polls may be stored as fractions (0.305) or percentages (30.5). Decided ONCE for the whole table: deciding per
    column would read an ONP column that never exceeds 1.0 point as fractions and inflate it 100x."""
    v = polls[[c for c in cols if c in polls.columns]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if v.size == 0 or np.isnan(v).all():
        return 1.0
    return 100.0 if np.nanmax(v) <= 1.0 else 1.0


def _reference_date(polls: pd.DataFrame, cfg: ModelConfig) -> pd.Timestamp:
    return pd.to_datetime(cfg.poll_as_of) if cfg.poll_as_of else pd.to_datetime(polls["date"]).max()


def _weights(dates: pd.Series, ref: pd.Timestamp, half_life: float) -> pd.Series:
    months = (ref.year - dates.dt.year) * 12 + (ref.month - dates.dt.month)      # whole calendar months before `ref`
    return 0.5 ** (months / half_life)


def weighted_average(polls: pd.DataFrame, party: str, cfg: ModelConfig, scale: float) -> PollInput:
    """Recency-weighted average of one party's polls (%) with a margin of error (points). Ages are measured from the
    latest poll in the table (or poll_as_of), the same for every party."""
    ref = _reference_date(polls, cfg)
    p = polls.dropna(subset=[party]).copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] <= ref]
    if p.empty:
        raise ValueError(f"No {party} polls on or before {ref.date()}.")
    w = _weights(p["date"], ref, cfg.poll_half_life_months)
    s = p[party].astype(float) * scale
    central = float((s * w).sum() / w.sum())
    if cfg.poll_moe is not None:
        moe = cfg.poll_moe
    else:
        n = p["sample_size"].astype(float) if "sample_size" in p.columns else pd.Series(np.nan, index=p.index)
        n = n.fillna(n.median() if n.notna().any() else cfg.poll_default_sample_size)
        pr = central / 100.0
        moe = cfg.poll_z * float(np.sqrt((w ** 2 * pr * (1 - pr) / n).sum()) / w.sum()) * 100.0
    return PollInput(central, central - moe, central + moe)


def poll_weight_table(polls: pd.DataFrame, cols: Sequence[str], cfg: ModelConfig) -> pd.DataFrame:
    """The polls that enter the average, with the weight each one gets."""
    ref = _reference_date(polls, cfg)
    p = polls.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] <= ref]
    months = (ref.year - p["date"].dt.year) * 12 + (ref.month - p["date"].dt.month)
    out = pd.DataFrame({"date": p["date"].values, "age_months": months.values,
                        "weight": _weights(p["date"], ref, cfg.poll_half_life_months).values})
    scale = _pct_scale(polls, cols)
    for c in cols:
        if c in p.columns:
            out[c] = p[c].astype(float).values * scale
    return out


def time_horizon_sd(prediction_date: Optional[str], election_date: str, gamma: float) -> float:
    """Extra statewide poll error (points) = gamma * sqrt(years from the prediction date to the election)."""
    pred = pd.to_datetime(prediction_date) if prediction_date else pd.Timestamp.today()
    days = max(0, (pd.to_datetime(election_date) - pred).days)
    return float(gamma * np.sqrt(days / 365.0))


def summarise_polls(poll: Union[PollInput, pd.DataFrame], cfg: ModelConfig, state_df: pd.DataFrame) -> PollSummary:
    parties = list(cfg.parties)
    i = parties.index(cfg.party)
    needs_onp = cfg.onp_mode == "entry" or cfg.onp_state_denominator
    why = "onp_mode='entry'" if cfg.onp_mode == "entry" else "onp_state_denominator=True"

    if isinstance(poll, pd.DataFrame):
        miss = [p for p in parties if p not in poll.columns]
        if miss:
            raise ValueError(f"Poll table needs a column for every modelled party; missing {miss}")
        cols = parties + (["ONP"] if "ONP" in poll.columns and "ONP" not in parties else [])
        scale = _pct_scale(poll, cols)
        per = {p: weighted_average(poll, p, cfg, scale) for p in parties}
        central = np.array([per[p].central for p in parties], dtype=float)
        target = per[cfg.party]
        has_onp = "ONP" in cols and "ONP" not in parties
        onp_central: Optional[float] = None
        if needs_onp:
            if not has_onp:
                raise ValueError(f"{why} needs an ONP column in the polls table (and OTHER must exclude ONP).")
            onp_central = weighted_average(poll, "ONP", cfg, scale).central
        elif has_onp:
            try:
                onp_c = weighted_average(poll, "ONP", cfg, scale).central
            except ValueError:
                onp_c = 0.0
            if onp_c > 0.5:
                kept = 100.0 * central[i] / (central.sum() + onp_c)
                dropped = 100.0 * central[i] / central.sum()
                log.warning("ONE NATION IS BEING DROPPED FROM THE STATEWIDE POLLS: polls have ONP at %.1f%% but neither "
                            "onp_mode='entry' nor onp_state_denominator is on, so the other parties are scaled up. "
                            "%s statewide poll share: %.1f%% with ONP kept vs %.1f%% as modelled.",
                            onp_c, cfg.party, kept, dropped)
        table = poll_weight_table(poll, cols, cfg)
    else:
        if needs_onp:
            raise ValueError(f"{why} needs a polls table with an ONP column; a single PollInput has no ONP.")
        target = poll
        last = state_matrix(state_df, parties, [int(state_df["year"].max())]).values[0]
        others = np.delete(last, i)
        central = np.empty(len(parties))
        central[i] = poll.central
        central[np.arange(len(parties)) != i] = (100.0 - poll.central) * others / others.sum()
        onp_central, table = None, None

    if onp_central is not None:
        full = np.append(central, onp_central)
        if full.sum() > cfg.poll_sum_warn:
            log.warning("poll shares incl. ONP sum to %.1f%%. With an ONP column, OTHER must EXCLUDE ONP.", full.sum())
        full = 100.0 * full / full.sum()
        central, onp_central = full[:-1], float(full[-1])
    else:
        central = 100.0 * central / central.sum()
    return PollSummary(central, onp_central, target, table)


def draw_statewide(summ: PollSummary, cfg: ModelConfig, D: int, rng: np.random.Generator) -> StatewideDraws:
    """Error budget: se^2 = (poll sampling)^2 + (house effect)^2 + (time horizon)^2, taken on the TARGET party's share and
    turned into a Dirichlet concentration whose variance for that party matches. Returns D draws of statewide shares (%)."""
    i = list(cfg.parties).index(cfg.party)
    se_poll = (summ.target.high - summ.target.low) / (2 * cfg.poll_z)
    h_sd = time_horizon_sd(cfg.prediction_date, cfg.election_date, cfg.time_horizon_gamma)
    se = float(np.sqrt(se_poll ** 2 + cfg.poll_house_sd ** 2 + h_sd ** 2)) / 100.0
    p_t = summ.central[i] / 100.0
    conc = max(p_t * (1 - p_t) / se ** 2 - 1.0, cfg.poll_dirichlet_conc_floor)
    log.info("Statewide poll error: sampling %.2f | house %.2f | horizon %.2f (election %s) -> total %.2f pts",
             se_poll, cfg.poll_house_sd, h_sd, cfg.election_date, se * 100.0)

    if summ.onp_central is None:
        S = rng.dirichlet(conc * summ.central / 100.0, size=D) * 100.0
        onp = None
    else:
        X = rng.dirichlet(conc * np.append(summ.central, summ.onp_central) / 100.0, size=D) * 100.0
        S, onp = X[:, :-1], X[:, -1]
        if not cfg.onp_state_denominator:
            S = 100.0 * S / S.sum(axis=1, keepdims=True)
    return StatewideDraws(S, onp, summ.central, summ.target, np.array([se_poll, cfg.poll_house_sd, h_sd, se * 100.0]), float(conc))
