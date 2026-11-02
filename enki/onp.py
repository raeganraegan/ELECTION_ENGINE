"""onp.py -- Step 6. One Nation. It cannot be inferred from a seat's history, so it enters by an explicit, mass-conserving rule.

  none   nothing happens (ONP, if any, sits inside OTHER)
  brian  a fixed fraction of each named party's vote moves to ONP:      p'_k = (1 - f_k) p_k,   p'_ONP = sum_k f_k p_k
  entry  ONP is an entrant scaled from the statewide ONP poll:          p'_k = (1 - s) p_k,     p'_ONP = s,   s = clip(ratio * S_ONP)
In every mode the target party's booth shares are multiplied by a per-draw scalar (`target_scale`), which is what the
uncertainty attribution needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .config import BRIAN_DEFAULT, ModelConfig


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
    B = p.shape[1]
    out = np.concatenate([p * (1.0 - s)[:, None, None], np.broadcast_to(s[:, None, None], (D, B, 1))], axis=2)
    return OnpResult(out, "entry", parties + ["ONP"], 1.0 - s, np.zeros(0), s)
