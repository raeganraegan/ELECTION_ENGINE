"""opv.py -- Step 8. Optional preferential (OPV) count: finalists, preference flows, two-candidate-preferred (2CP) and exhaustion.

For every posterior draw:
  1. The finalists are the two parties with the highest SEAT-WIDE primary share in that draw (catch-all buckets such as
     OTHER cannot be a finalist). Different draws can have different pairings.
  2. Each other party's primary votes in every booth are split by that pairing's flow rates
     [to finalist 1, to finalist 2, exhausted]. Rates are fixed, or (opv_flow_conc) redrawn per draw from a Dirichlet.
  3. 2CP % = finalist votes / (finalist 1 + finalist 2 votes); exhausted votes leave the count. The seat is won by the
     finalist with more than 50% of the 2CP.

Reads only arrays (volumes V and booth primaries P in percent); it knows nothing about fitting, polls or files.

Limitations (deliberate simplifications, not bugs):
  * Finalists are the top two on primaries. The real count excludes candidates one at a time, and OTHER is one bucket, so
    an independent finishing in the final two is invisible here.
  * Flow rates are seat-wide: they vary between draws (if opv_flow_conc is set) but not between booths.
  * A pairing with no flow table is NOT modelled: its draws get NaN 2CP and are reported as `unmodelled`.
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import NSW_PREF_FLOWS

log = logging.getLogger("enki")
OPV_SEED_OFFSET = 202
FLOW_FLOOR = 1e-3                      # Dirichlet needs positive parameters: a rate of exactly 0 is floored at this
_PAIR_KEY = re.compile(r"^\s*([A-Za-z0-9]+)_vs_([A-Za-z0-9]+)\s*$")


@dataclass
class OpvResult:
    eligible: List[str]                  # parties that can be a finalist
    report_pair: Tuple[str, str]         # (A, B): the most frequent modelled pairing; booth tables and plots are for it
    report_pair_prob: float              # fraction of ALL draws with that pairing
    pair_probs: pd.Series                # "ALP v LNP" -> fraction of draws (includes unmodelled pairings)
    win_probs: pd.Series                 # party -> fraction of ALL draws it wins the seat; with unmodelled_prob these sum to 1
    unmodelled_prob: float               # fraction of draws whose pairing has no flow table
    seat_draws: pd.DataFrame             # one row per draw: pair, <party>_2CP_pct (NaN if not a finalist), exhausted, winner
    booth_2cp: pd.DataFrame              # per booth, over the draws that have the reported pairing
    booth_primary_mean: pd.DataFrame     # booth x party mean primary % over all draws
    flow_conc: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- flow tables
def resolve_flows(cfg: ModelConfig) -> Dict[str, Any]:
    return cfg.opv_flows if cfg.opv_flows is not None else NSW_PREF_FLOWS


def out_party_names(cfg: ModelConfig) -> List[str]:
    """Parties present in the booth tensor the OPV step sees: the modelled parties, plus ONP when the ONP layer adds it."""
    names = [p.upper() for p in cfg.parties]
    return names + (["ONP"] if cfg.onp_mode != "none" and "ONP" not in names else [])


def _parse_flows(flows: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    """{'ALP_vs_LNP': {'GRN': [to ALP, to LNP, exhausted], ...}} -> {('ALP','LNP'): {'GRN': [...]}}. Raises on any malformed entry."""
    out: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    errs: List[str] = []
    for key, rows in (flows or {}).items():
        m = _PAIR_KEY.match(str(key))
        if not m:
            errs.append(f"table key {key!r} must look like 'ALP_vs_LNP'")
            continue
        a, b = m.group(1).upper(), m.group(2).upper()
        if a == b or (a, b) in out or (b, a) in out:
            errs.append(f"table {key!r} is a repeat or pairs a party with itself")
            continue
        parsed: Dict[str, List[float]] = {}
        for party, v in dict(rows).items():
            r = [float(x) for x in v]
            if len(r) != 3 or min(r) < 0 or abs(sum(r) - 1.0) > 1e-3:
                errs.append(f"{key}[{party!r}] = {list(v)} must be 3 non-negative numbers summing to 1 (to {a}, to {b}, exhausted)")
            elif str(party).upper() in (a, b):
                errs.append(f"{key} has a row for {party!r}, which is one of the finalists")
            parsed[str(party).upper()] = r
        out[(a, b)] = parsed
    if errs:
        raise ValueError("opv_flows: " + "; ".join(errs))
    return out


def _lookup(parsed, a: str, b: str) -> Optional[Dict[str, List[float]]]:
    """Rows of the a-v-b table oriented [to a, to b, exhausted]; None when neither a_vs_b nor b_vs_a exists."""
    if (a, b) in parsed:
        return parsed[(a, b)]
    if (b, a) in parsed:
        return {k: [v[1], v[0], v[2]] for k, v in parsed[(b, a)].items()}
    return None


def check_flows(flows: Dict[str, Any], cfg: ModelConfig) -> List[str]:
    """Input check, run before any sampling. Hard problems raise; soft ones come back as warnings."""
    parsed = _parse_flows(flows)
    names = out_party_names(cfg)
    excl = {e.upper() for e in cfg.opv_finalist_exclude}
    eligible = [n for n in names if n not in excl]
    errs, warn = [], []
    usable = 0
    for (a, b), rows in parsed.items():
        bad = [x for x in (a, b) if x not in eligible]
        if bad:
            warn.append(f"OPV flow table {a}_vs_{b} is ignored: {bad} cannot be a finalist (not modelled, or excluded).")
            continue
        usable += 1
        missing = [n for n in names if n not in (a, b) and n not in rows and "OTHER" not in rows]
        if missing:
            errs.append(f"{a}_vs_{b} has no flow row for {missing} and no 'OTHER' row to fall back on")
    if errs:
        raise ValueError("opv_flows: " + "; ".join(errs))
    if not usable:
        warn.append("No OPV flow table matches any possible finalist pair of the modelled parties: the OPV step will have nothing to report.")
    return warn


def _rates(rates: Sequence[float], conc: Optional[float], n: int, rng: np.random.Generator) -> np.ndarray:
    r = np.asarray(rates, dtype=float)
    if conc is None:
        return np.broadcast_to(r, (n, 3))
    return rng.dirichlet(np.maximum(r, FLOW_FLOOR) * float(conc), size=n)


# ----------------------------------------------------------------------------- the step
def compute_opv(V: np.ndarray, P: np.ndarray, parties: Sequence[str], booths: Sequence[str], cfg: ModelConfig,
                flows: Dict[str, Any], rng: np.random.Generator) -> OpvResult:
    """V (D, B) formal votes; P (D, B, K) booth primary shares in PERCENT (rows sum to 100); parties in P's column order."""
    names = [str(p).upper() for p in parties]
    nm = np.array(names)
    parsed = _parse_flows(flows)
    V, P = np.asarray(V, dtype=float), np.asarray(P, dtype=float)
    D, B, K = P.shape
    votes = V[:, :, None] * P / 100.0                                   # (D, B, K) primary votes
    tot = V.sum(axis=1)                                                 # (D,)
    primary = votes.sum(axis=1) / tot[:, None] * 100.0                  # (D, K) seat-wide primary %

    excl = {e.upper() for e in cfg.opv_finalist_exclude}
    el = np.array([j for j, n in enumerate(names) if n not in excl])
    if len(el) < 2:
        raise ValueError(f"Need at least two parties that can be finalists; modelled {names}, excluded {sorted(excl)}.")

    # Finalists per draw: top two seat-wide primaries; the pair is stored in alphabetical order (ia < ib by name)
    order = np.argsort(-primary[:, el], axis=1)[:, :2]
    f1, f2 = el[order[:, 0]], el[order[:, 1]]
    swap = nm[f1] > nm[f2]
    ia, ib = np.where(swap, f2, f1), np.where(swap, f1, f2)
    pair_ids = ia * K + ib

    VA, VB, VE = (np.full((D, B), np.nan) for _ in range(3))             # votes to ia / to ib / exhausted; NaN = unmodelled draw
    modelled = np.zeros(D, dtype=bool)
    for pid in np.unique(pair_ids):
        i, j = divmod(int(pid), K)
        idx = np.flatnonzero(pair_ids == pid)
        rows = _lookup(parsed, names[i], names[j])
        if rows is None:
            continue
        va, vb, ve = votes[idx, :, i].copy(), votes[idx, :, j].copy(), np.zeros((len(idx), B))
        for k in range(K):
            if k in (i, j):
                continue
            rk = rows.get(names[k], rows.get("OTHER"))
            if rk is None:
                raise ValueError(f"No flow row for {names[k]} in {names[i]} v {names[j]} and no 'OTHER' row.")
            r = _rates(rk, cfg.opv_flow_conc, len(idx), rng)             # (n, 3) seat-wide rates for these draws
            vk = votes[idx, :, k]
            va += vk * r[:, 0:1]
            vb += vk * r[:, 1:2]
            ve += vk * r[:, 2:3]
        VA[idx], VB[idx], VE[idx] = va, vb, ve
        modelled[idx] = True

    sa, sb, se = VA.sum(axis=1), VB.sum(axis=1), VE.sum(axis=1)         # NaN on unmodelled draws
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_a = 100.0 * sa / (sa + sb)
    winner = np.where(~modelled, "unmodelled", np.where(sa > sb, nm[ia], nm[ib]))
    pair_label = np.char.add(np.char.add(nm[ia], " v "), nm[ib])

    cols: Dict[str, Any] = {"pair": pair_label}
    for e in el:
        arr = np.full(D, np.nan)
        arr[ia == e] = pct_a[ia == e]
        arr[ib == e] = 100.0 - pct_a[ib == e]
        cols[f"{names[e]}_2CP_pct"] = arr
    cols.update({"exhausted_pct_formal": 100.0 * se / tot, "winner": winner, "total_formal_votes": tot, "exhausted_votes": se})
    seat_draws = pd.DataFrame(cols)

    pair_probs = pd.Series(pair_label).value_counts(normalize=True).rename("probability")
    win_probs = pd.Series({names[e]: float((winner == names[e]).mean()) for e in el}, name="probability")
    unmodelled = float((~modelled).mean())
    if not modelled.any():
        raise ValueError(f"No draw's finalist pair has a flow table (draws had: {sorted(set(pair_label))}; tables: "
                         f"{sorted(f'{a}_vs_{b}' for a, b in parsed)}).")

    # Reported pairing = most frequent modelled one; the preferred party (cfg.party) is listed first when it is in it
    ids, cnt = np.unique(pair_ids[modelled], return_counts=True)
    pid = int(ids[np.argmax(cnt)])
    i, j = divmod(pid, K)
    pref = cfg.party.upper()
    if names[j] == pref:
        A, Bq = j, i
    elif names[i] == pref:
        A, Bq = i, j
    else:
        A, Bq = (i, j) if primary[:, i].mean() >= primary[:, j].mean() else (j, i)
    m = pair_ids == pid
    va, vb = ((VA, VB) if A == i else (VB, VA))[0][m], ((VA, VB) if A == i else (VB, VA))[1][m]     # oriented to (A, B)
    ve, vt = VE[m], V[m]
    den = va + vb
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(den > 0, 100.0 * va / den, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # a booth with no 2CP votes in every draw is all-NaN
        pa, pb = names[A], names[Bq]
        booth = pd.DataFrame({
            "booth": list(booths), "expected_formal_votes": vt.mean(axis=0),
            f"{pa}_2cp_votes_mean": va.mean(axis=0), f"{pb}_2cp_votes_mean": vb.mean(axis=0), "exhausted_votes_mean": ve.mean(axis=0),
            f"{pa}_2cp_share_mean": np.nanmean(share, axis=0), f"{pa}_2cp_p05": np.nanpercentile(share, 5, axis=0),
            f"{pa}_2cp_p95": np.nanpercentile(share, 95, axis=0), f"{pb}_2cp_share_mean": 100.0 - np.nanmean(share, axis=0),
            f"{pa}_booth_win_prob": np.nanmean(np.where(np.isnan(share), np.nan, share > 50.0), axis=0),
            "exhausted_share_of_total_mean": (100.0 * ve / np.maximum(vt, 1.0)).mean(axis=0),
            "draws_in_pair": int(m.sum())})

    w: List[str] = []
    rp = float(pair_probs.get(f"{min(pa, pb)} v {max(pa, pb)}", 0.0))
    if rp < cfg.opv_min_pair_prob:
        w.append(f"OPV: the reported pairing {pa} v {pb} occurs in only {100 * rp:.1f}% of draws; booth tables and plots are conditional on it "
                 f"(pairing probabilities: {', '.join(f'{k} {100 * v:.1f}%' for k, v in pair_probs.head(4).items())}). Win probabilities use all draws.")
    if unmodelled > 0:
        bad = [k for k in pair_probs.index if _lookup(parsed, *k.split(" v ")) is None]
        w.append(f"OPV: {100 * unmodelled:.1f}% of draws have a finalist pairing with no flow table ({', '.join(bad)}); they count towards no one's win probability.")
    return OpvResult(eligible=[names[e] for e in el], report_pair=(pa, pb), report_pair_prob=rp, pair_probs=pair_probs, win_probs=win_probs,
                     unmodelled_prob=unmodelled, seat_draws=seat_draws, booth_2cp=booth,
                     booth_primary_mean=pd.DataFrame(P.mean(axis=0), columns=names, index=list(booths)),
                     flow_conc=cfg.opv_flow_conc, warnings=w)


def opv_trace(o: OpvResult) -> Dict[str, np.ndarray]:
    """The `s8_*` signals recorded in the trace (see pipeline.PIPELINE_STEPS)."""
    sd = o.seat_draws
    return {"s8_eligible": np.array(o.eligible), "s8_pair": sd["pair"].to_numpy(dtype=str),
            "s8_2cp": sd[[f"{e}_2CP_pct" for e in o.eligible]].to_numpy(dtype=float),
            "s8_exhausted": sd["exhausted_pct_formal"].to_numpy(dtype=float), "s8_winner": sd["winner"].to_numpy(dtype=str)}