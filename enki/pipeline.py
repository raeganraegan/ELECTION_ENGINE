"""pipeline.py -- the process-flow registry. One list drives three things: what the forecast must record (trace keys),
what a process-flow diagram draws, and what check_trace verifies, so the diagram and the code cannot silently drift apart.

Each model step returns a typed result (PollSummary / StatewideDraws, ShareSim, VolumeResult, OnpResult); `build_trace`
flattens those results into the `s<step>_<name>` signals below. Nothing writes into a shared dict while the model runs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ModelConfig, beta_prior_table

if TYPE_CHECKING:
    from .onp import OnpResult
    from .polls import PollSummary, StatewideDraws
    from .share_model import ShareFit, ShareSim
    from .volume_model import VolumeResult


# Layout coordinates are for a 14 x 7.5 canvas; edges are (from, to, kind).
PIPELINE_STEPS: List[Dict[str, Any]] = [
    dict(id="S0", name="Prep", pos=(2.0, 6.0), what="counts -> shares; statewide log-swing",
         eq="x_kt = ln S_kt - mean_t ln S_kt", inputs="booth_results, state_results", outputs="x (T x K), S_hist",
         trace=["s0_years", "s0_parties", "s0_state_pct", "s0_x", "s0_x_center"]),
    dict(id="S1", name="Fit", pos=(5.4, 6.0), what="posterior of the multinomial-logit model",
         eq="eta = m + alpha_b + beta_b*x + d_t + eps", inputs="Y, x, priors", outputs="theta^(i), i = 1..D",
         trace=["s1_beta", "s1_rho", "s1_sigma_d", "s1_sigma_alpha", "s1_sigma_eps", "s1_m", "s1_prior_beta", "s1_diag"]),
    dict(id="S2", name="Polls", pos=(2.0, 3.4), what="weighted poll average + error budget -> statewide share draws",
         eq="S ~ Dir(conc * s_hat);  se^2 = se_poll^2 + house^2 + g^2 tau", inputs="polls, dates, config", outputs="S^(i) (D x K)",
         trace=["s2_se_parts", "s2_conc", "s2_central", "s2_central_onp", "s2_S_next", "s2_onp_state"]),
    dict(id="S3", name="Tide", pos=(5.4, 3.4), what="statewide swing + seat tide -> party-level shift",
         eq="lin = beta*(ln S - c) + rho*d_T + sqrt(1-rho^2)*sigma_d*z", inputs="theta, S", outputs="lin (D x K)",
         trace=["s3_x_next", "s3_bx", "s3_d_next", "s3_lin", "s3_d_hist_mean", "s3_d_hist_p05", "s3_d_hist_p95", "s3_bx_hist"]),
    dict(id="S4", name="Booths", pos=(8.8, 3.4), what="booth-level shares (softmax)",
         eq="p_b = softmax(m + alpha_b + lin + eps_b)", inputs="lin, alpha, eps", outputs="p_b (D x B x K)",
         trace=["s4_p_pre", "s4_booths", "s4_last_party_share"]),
    dict(id="S5", name="Volumes", pos=(2.0, 0.9), what="formal votes per booth",
         eq="V ~ Mult(T', Dir(kappa * softmax(a + tau' early)))", inputs="booth history | electors | projections", outputs="V (D x B)",
         trace=["s5_V", "s5_hist_totals", "s5_last_n", "s5_mode"]),
    dict(id="S6", name="ONP", pos=(12.2, 3.4), what="One Nation entry / vote transfer (mass-conserving)",
         eq="p'_k = (1-f_k) p_k ;  p'_ONP = sum_k f_k p_k", inputs="p_b, f or S_ONP", outputs="p'_b (D x B x K+1)",
         trace=["s6_mode", "s6_retain"]),
    dict(id="S7", name="Seat", pos=(8.8, 0.9), what="volume-weighted seat shares and percentiles",
         eq="s_k = sum_b V_b p'_kb / sum_b V_b", inputs="V, p'_b", outputs="seat shares (D x K+1)",
         trace=["s7_seat_draws", "s7_out_parties"]),
]
PIPELINE_EDGES = [("S0", "S1", "solid"), ("S1", "S3", "solid"), ("S2", "S3", "solid"), ("S3", "S4", "solid"),
                  ("S4", "S6", "solid"), ("S6", "S7", "solid"), ("S5", "S7", "solid"), ("S0", "S5", "dashed")]


@dataclass
class Stages:
    """Everything the steps produced, in one place, so the trace is built after the fact rather than during the run."""
    cfg: ModelConfig
    df: pd.DataFrame
    fit: "ShareFit"
    polls: "PollSummary"
    stw: "StatewideDraws"
    sim: "ShareSim"
    onp: "OnpResult"
    vol: "VolumeResult"
    keep: np.ndarray
    kept_names: list
    seat_all: np.ndarray
    last_n: pd.Series


def build_trace(st: Stages) -> Dict[str, Any]:
    cfg, fit, dr = st.cfg, st.fit, st.fit.draws
    parties = list(cfg.parties)
    pri = beta_prior_table(cfg)
    last = st.df[st.df["year"] == st.df["year"].max()].set_index("booth").reindex(st.kept_names)
    trace: Dict[str, Any] = {
        "s0_years": np.array(fit.years), "s0_parties": np.array(parties), "s0_state_pct": np.asarray(fit.S_pct, dtype=float),
        "s0_x": np.asarray(fit.x, dtype=float), "s0_x_center": np.asarray(fit.x_center, dtype=float),
        "s1_beta": dr["beta"], "s1_rho": np.asarray(dr["rho"], dtype=float), "s1_sigma_d": dr["sigma_d"],
        "s1_sigma_alpha": dr["sigma_alpha"], "s1_sigma_eps": dr["sigma_eps"], "s1_m": dr["m"],
        "s1_prior_beta": np.array([list(pri.get(p, (0.0, 0.0))) for p in parties], dtype=float),     # reference party: fixed 0
        "s1_diag": fit.diag,
        "s2_se_parts": st.stw.se_parts, "s2_conc": st.stw.conc, "s2_central": np.asarray(st.stw.central, dtype=float),
        "s2_central_onp": float(st.polls.onp_central) if st.polls.onp_central is not None else float("nan"),
        "s2_S_next": np.asarray(st.stw.S, dtype=float),
        "s2_onp_state": np.asarray(st.stw.onp, dtype=float) if st.stw.onp is not None else np.zeros(0),
        "s3_x_next": st.sim.x_next, "s3_bx": st.sim.bx, "s3_d_next": st.sim.d_next, "s3_lin": st.sim.lin_party,
        "s3_d_hist_mean": dr["d"].mean(axis=0), "s3_d_hist_p05": np.percentile(dr["d"], 5, axis=0),
        "s3_d_hist_p95": np.percentile(dr["d"], 95, axis=0),
        "s3_bx_hist": dr["beta"].mean(axis=0)[None, :] * np.asarray(fit.x),
        "s4_p_pre": (100.0 * st.sim.p[:, st.keep, :]).astype(np.float32), "s4_booths": np.array(st.kept_names),
        "s4_last_party_share": np.column_stack([100.0 * last[f"y_{p}"].values / last["n"].values for p in parties]),
        "s5_V": st.vol.V[:, st.keep].astype(np.float32), "s5_mode": st.vol.mode,
        "s5_hist_totals": st.df.groupby("year")["n"].sum().reindex(fit.years).values.astype(float),
        "s5_last_n": st.last_n.reindex(st.kept_names).values.astype(float),
        "s6_mode": st.onp.mode, "s6_retain": np.asarray(st.onp.retain, dtype=float),
        "s7_seat_draws": st.seat_all, "s7_out_parties": np.array(st.onp.out_parties),
        "meta": {"out_parties": st.onp.out_parties, **asdict(cfg)},
    }
    if st.polls.table is not None:
        trace["s2_poll_table"] = st.polls.table
    return trace


def check_trace(trace: Optional[dict]) -> List[str]:
    """Names every step whose recorded signals are incomplete."""
    if not trace:
        return ["trace is empty: the process report has nothing to draw."]
    return [f"trace step {s['id']} ({s['name']}) is missing: {', '.join(k for k in s['trace'] if k not in trace)}"
            for s in PIPELINE_STEPS if any(k not in trace for k in s["trace"])]
