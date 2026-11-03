"""onp.py -- Step 6. One Nation. It cannot be inferred from a seat's history, so it enters by an explicit, mass-conserving rule.

  none   nothing happens (ONP, if any, sits inside OTHER)
  brian  a fixed fraction of each named party's vote moves to ONP:      p'_k = (1 - f_k) p_k,   p'_ONP = sum_k f_k p_k
  entry  ONP is an entrant scaled from the statewide ONP poll, distributed via a Dirichlet simplex over party siphoning weights.
In every mode the target party's booth shares are multiplied by a per-draw scalar (`target_scale`), which is what the
uncertainty attribution needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .config import BRIAN_DEFAULT, ModelConfig

FLOW_FLOOR = 1e-3  # Dirichlet safety floor


@dataclass
class OnpResult:
    p: np.ndarray                    # (D, B, K) or (D, B, K+1) with ONP last
    mode: str
    out_parties: List[str]
    target_scale: np.ndarray         # (D,) factor applied to the target party's booth shares
    retain: np.ndarray               # (K,) fraction each party keeps (brian only, else empty)
    seat_share: Optional[np.ndarray] # (D,) ONP's share of every booth (entry only)


def apply_onp(p: np.ndarray, onp_state_pct: Optional[np.ndarray], parties: Sequence[str], k0: int,
              cfg: ModelConfig, rng: np.random.Generator) -> OnpResult:
    parties = list(parties)
    D = p.shape[0]
    if cfg.onp_mode == "none":
        return OnpResult(p, "none", parties, np.ones(D), np.zeros(0), None)

    if cfg.onp_mode == "brian":
        take = cfg.brian_transfer or BRIAN_DEFAULT
        retain = np.array([1.0 - float(take.get(q, 0.0)) for q in parties])
        onp = (p * (1.0 - retain)[None, None, :]).sum(axis=2)
        return OnpResult(np.concatenate([p * retain[None, None, :], onp[:, :, None]], axis=2), "brian",
                         parties + ["ONP"], np.full(D, retain[k0]), retain, None)

    if onp_state_pct is None:                                   # unreachable via ENKI(x): summarise_polls already refuses
        raise ValueError("onp_mode='entry' needs the statewide ONP draws.")
        
    ratio = np.exp(np.log(cfg.onp_seat_ratio_median) + cfg.onp_seat_ratio_sdlog * rng.standard_normal(D))
    s = np.clip(ratio * np.asarray(onp_state_pct, dtype=float) / 100.0, 0.0, cfg.onp_max_seat_share)
    
    # 1. Base siphoning weights for party susceptibility to ONP
    default_entry_weights = {
        "LNP": 0.50, "LIB": 0.50, "NAT": 0.50, 
        "ALP": 0.35, 
        "OTHER": 0.12, "OTH": 0.12, 
        "GRN": 0.03
    }
    weights_dict = getattr(cfg, "onp_entry_weights", None) or default_entry_weights
    w_base = np.array([float(weights_dict.get(q, 0.10)) for q in parties], dtype=float)
    
    # Normalize base weights to sum to 1.0 so they form a valid Dirichlet mean vector
    w_base = w_base / w_base.sum()

    # 2. Dirichlet Distribution for Siphoning Uncertainty 
    # Uses cfg.opv_flow_conc (defaulting to 133.0 if you want to reuse that same concentration parameter)
    conc = getattr(cfg, "opv_flow_conc", 133.0) or 133.0
    alpha = np.maximum(w_base, FLOW_FLOOR) * float(conc)
    
    # Sample a unique randomized siphoning proportion vector (D, K) for every Monte Carlo draw
    w_draws = rng.dirichlet(alpha, size=D)  # shape: (D, K)

    # 3. Proportional booth-level siphoning driven by the draw's Dirichlet weights
    B = p.shape[1]
    # weighted_pool per draw, booth: (D, B, 1)
    weighted_pool = (p * w_draws[:, None, :]).sum(axis=2, keepdims=True)
    
    mean_pool = np.maximum(weighted_pool.mean(axis=1, keepdims=True), 1e-6)
    booth_onp_share = s[:, None, None] * (weighted_pool / mean_pool)
    booth_onp_share = np.clip(booth_onp_share, 0.0, 0.50)  # Safety cap at 50%
    
    # Deduct from each party proportionally using the stochastic w_draws
    deduction = booth_onp_share * (w_draws[:, None, :] * p / np.maximum(weighted_pool, 1e-6))
    p_retained = np.maximum(0.0, p - deduction)
    onp_booth = p.sum(axis=2, keepdims=True) - p_retained.sum(axis=2, keepdims=True)
    
    out = np.concatenate([p_retained, onp_booth], axis=2)
    return OnpResult(out, "entry", parties + ["ONP"], 1.0 - s, np.zeros(0), s)