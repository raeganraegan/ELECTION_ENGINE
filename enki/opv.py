"""opv.py -- Step 8. Optional preferential (OPV) count: finalists, preference flows, two-candidate-preferred (2CP) and exhaustion.

Flows come from an AFFINITY model, so any finalist pairing can be counted, including ones with no flow table:

    a source party s sends its votes to the continuing candidates C with softmax over  {exhaust: 0} U {u_sc : c in C}
    u_sc = log-affinity of s's voters for candidate c, relative to exhausting      (IIA: removing a candidate re-spreads its share)

  * Flow tables (opv_flows) are DATA: a row [to a, to b, exhausted] gives u_sa = ln(p_a/p_exh), u_sb = ln(p_b/p_exh). When the final
    pair has a table, its own rows are used as given; otherwise the affinities pooled over all tables are used.
  * An affinity no table pins down (e.g. anything about a new entrant) gets a prior: mean = the source's average known affinity
    (or the weighted average of its `opv_proxies`), sd = opv_affinity_sd. Parties in `opv_entrants` always get at least that sd,
    even where a table exists, because their voters' behaviour is not known.
  * Each draw samples one affinity matrix (u = mu + sd * z) used throughout that draw, so flow ignorance widens the seat band
    rather than removing draws. Known entries use opv_flow_conc for their noise (None = fixed).

For every posterior draw:
  1. Count-by-count finalists. Buckets in opv_finalist_exclude (default OTHER) are distributed first, then the eligible party with
     the fewest seat-wide votes is excluded and distributed, until two remain. (Flows are seat-wide, so this runs on seat totals.)
  2. Final tallies for the finalist pair (f, g): every other party's ORIGINAL primary votes in every booth are split by that pair's
     [to f, to g, exhausted] rates, so a known pair reproduces its table exactly.
  3. 2CP % = f / (f + g); the winner has more than 50%; exhausted votes leave the count.

Reads only arrays (volumes V and booth primaries P in percent); it knows nothing about fitting, polls or files.

Limitations (deliberate simplifications, not bugs):
  * IIA: a source's relative preference between two candidates does not depend on who else is running (exhaustion does rise as
    candidates drop out). Table rows are net flows to the final pair and are reused as direct affinities in the exclusion steps.
  * Flow rates are seat-wide (shared by all booths); independents inside a bucket such as OTHER cannot be finalists.
  * Noise on affinities is independent per (source, candidate), so it ignores correlation within a table row.
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
FLOW_FLOOR = 1e-3                      # rates of exactly 0 are floored at this before taking logs
SELF_AFFINITY = 0.0                    # log-affinity of a party's voters for a "like" party when a proxy names the party itself
_PAIR_KEY = re.compile(r"^\s*([A-Za-z0-9]+)_vs_([A-Za-z0-9]+)\s*$")


@dataclass
class OpvResult:
    eligible: List[str]                  # parties that can be a finalist
    report_pair: Tuple[str, str]         # (A, B): the most frequent pairing; booth tables and plots are for it
    report_pair_prob: float              # fraction of ALL draws with that pairing
    pair_probs: pd.Series                # "ALP v LNP" -> fraction of draws
    win_probs: pd.Series                 # party -> fraction of draws it wins the seat (sums to 1)
    seat_draws: pd.DataFrame             # one row per draw: pair, <party>_2CP_pct (NaN if not a finalist), exhausted, winner
    booth_2cp: pd.DataFrame              # per booth, over the draws that have the reported pairing
    booth_primary_mean: pd.DataFrame     # booth x party mean primary % over all draws
    affinities: pd.DataFrame             # pooled source -> candidate log-affinities, sd and where each came from
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


def check_flows(flows: Dict[str, Any], cfg: ModelConfig) -> List[str]:
    """Input check, run before any sampling. Malformed tables raise; everything else comes back as warnings.
    Missing rows or missing tables are NOT errors: the affinity model fills them from its wide prior."""
    parsed = _parse_flows(flows)
    names = out_party_names(cfg)
    excl = {e.upper() for e in cfg.opv_finalist_exclude}
    eligible = [n for n in names if n not in excl]
    warn, usable = [], 0
    for (a, b) in parsed:
        bad = [x for x in (a, b) if x not in eligible]
        if bad:
            warn.append(f"OPV flow table {a}_vs_{b} is not used as a pairing: {bad} cannot be a finalist (not modelled, or excluded); "
                        "its rows still inform the affinities of the parties that are modelled.")
        else:
            usable += 1
    if not usable:
        warn.append("No OPV flow table matches a possible finalist pair of the modelled parties: flows come from the affinity prior "
                    f"(sd {cfg.opv_affinity_sd:g}) wherever no table row exists.")
    for e, d in (cfg.opv_proxies or {}).items():
        unknown = [p for p in list(d) + [e] if str(p).upper() not in names]
        if unknown:
            warn.append(f"opv_proxies[{e!r}] names {unknown}, which are not modelled parties; they are ignored.")
    return warn


# ----------------------------------------------------------------------------- the affinity model
def _trigamma(x):
    """psi'(x) for x > 0 (recurrence up to x >= 6, then the asymptotic series); avoids a SciPy dependency."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for _ in range(8):
        small = x < 6.0
        out = out + np.where(small, 1.0 / x ** 2, 0.0)
        x = np.where(small, x + 1.0, x)
    return out + 1.0 / x + 1.0 / (2 * x ** 2) + 1.0 / (6 * x ** 3) - 1.0 / (30 * x ** 5)


def _softmax_exh(U: np.ndarray) -> np.ndarray:
    """(n, m) log-affinities of m candidates relative to exhausting -> (n, m + 1) flow rates; column 0 is exhausted."""
    L = np.concatenate([np.zeros((U.shape[0], 1)), U], axis=1)
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


@dataclass
class AffinityModel:
    names: List[str]
    mu: np.ndarray                       # (K, K) pooled mean log-affinity, row = source, column = candidate
    sd: np.ndarray                       # (K, K) pooled sd
    basis: np.ndarray                    # (K, K) object: where each entry came from
    tables: Dict[Tuple[int, int], Dict[int, Tuple[float, float, float, float]]]   # (lo, hi) -> source -> (mu_lo, sd_lo, mu_hi, sd_hi)


def build_affinity(parsed: Dict[Tuple[str, str], Dict[str, List[float]]], names: Sequence[str], cfg: ModelConfig) -> AffinityModel:
    names = list(names)
    K, ix = len(names), {n: i for i, n in enumerate(names)}
    conc = cfg.opv_flow_conc
    wide = float(cfg.opv_affinity_sd)
    ents = {str(e).upper() for e in cfg.opv_entrants}

    est: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    tables: Dict[Tuple[int, int], Dict[int, Tuple[float, float, float, float]]] = {}
    for (a, b), rows in parsed.items():
        if a not in ix or b not in ix:
            continue
        lo, hi = sorted((a, b))
        il, ih = ix[lo], ix[hi]
        for party, r in rows.items():
            if party not in ix or party in (a, b):
                continue
            k = ix[party]
            r = [r[1], r[0], r[2]] if a > b else list(r)                 # orient to (lo, hi)
            p = np.maximum(np.asarray(r, dtype=float), FLOW_FLOOR)
            u = (float(np.log(p[0] / p[2])), float(np.log(p[1] / p[2])))
            s = (0.0, 0.0) if conc is None else tuple(float(np.sqrt(_trigamma(conc * p[q]) + _trigamma(conc * p[2]))) for q in (0, 1))
            est.setdefault((k, il), []).append((u[0], s[0] ** 2))
            est.setdefault((k, ih), []).append((u[1], s[1] ** 2))
            sl = max(s[0], wide) if (party in ents or lo in ents) else s[0]
            sh = max(s[1], wide) if (party in ents or hi in ents) else s[1]
            tables.setdefault((il, ih), {})[k] = (u[0], sl, u[1], sh)

    mu, sd = np.zeros((K, K)), np.zeros((K, K))
    basis = np.full((K, K), "", dtype=object)
    known = np.zeros((K, K), dtype=bool)
    for (k, c), lst in est.items():                                       # pool every table that speaks about (source, candidate)
        v = np.array([x[0] for x in lst])
        var = float(np.mean([x[1] for x in lst])) + (float(v.var(ddof=1)) if len(v) > 1 else 0.0)
        mu[k, c], sd[k, c], known[k, c], basis[k, c] = float(v.mean()), float(np.sqrt(var)), True, "table"
        if names[k] in ents or names[c] in ents:
            sd[k, c], basis[k, c] = max(sd[k, c], wide), "table (entrant, widened)"

    base = np.array([float(mu[k][known[k]].mean()) if known[k].any() else 0.0 for k in range(K)])
    for k in range(K):                                                    # nothing known about (k, c): the source's average, wide
        for c in range(K):
            if k != c and not known[k, c]:
                mu[k, c], sd[k, c], basis[k, c] = base[k], wide, "default (source average)"

    prox = {str(c).upper(): {str(p).upper(): float(w) for p, w in d.items() if str(p).upper() in ix and float(w) > 0}
            for c, d in (cfg.opv_proxies or {}).items()}
    mu0 = mu.copy()
    m = lambda s_, c_: SELF_AFFINITY if s_ == c_ else mu0[s_, c_]
    for k in range(K):
        for c in range(K):
            if k == c or known[k, c]:
                continue
            pc, pk = prox.get(names[c]) or {}, prox.get(names[k]) or {}
            pc = {p: w for p, w in pc.items() if p != names[c]}
            pk = {p: w for p, w in pk.items() if p != names[k]}
            if pc:                                                        # candidate c "is like" the proxies
                mu[k, c], basis[k, c] = sum(w * m(k, ix[p]) for p, w in pc.items()) / sum(pc.values()), "proxy"
            elif pk:                                                      # source k "behaves like" the proxies
                mu[k, c], basis[k, c] = sum(w * m(ix[p], c) for p, w in pk.items()) / sum(pk.values()), "proxy"
    return AffinityModel(names, mu, sd, basis, tables)


def _finalists(T: np.ndarray, u: np.ndarray, elig: np.ndarray, buckets: Sequence[int], K: int) -> np.ndarray:
    """Count-by-count on seat totals T (D, K) with per-draw affinities u (D, K, K). Returns the two surviving party indices per draw."""
    D = T.shape[0]
    cur = T.copy()
    alive = np.zeros((D, K), dtype=bool)
    alive[:, elig] = True
    if len(elig) > 2:
        for s in buckets:                                                 # buckets (e.g. OTHER) go first, to every eligible candidate
            r = _softmax_exh(u[:, s, :][:, elig])
            cur[:, elig] += cur[:, [s]] * r[:, 1:]
            cur[:, s] = 0.0
        pw = 1 << np.arange(K)
        for _ in range(len(elig) - 2):
            e = np.where(alive, cur, np.inf).argmin(axis=1)               # lowest continuing candidate in each draw
            code = e * (1 << K) + (alive * pw).sum(axis=1)
            for cd in np.unique(code):
                idx = np.flatnonzero(code == cd)
                ei, mask = int(cd >> K), int(cd & ((1 << K) - 1))
                cand = [c for c in range(K) if (mask >> c) & 1 and c != ei]
                r = _softmax_exh(u[idx, ei, :][:, cand])
                cur[np.ix_(idx, cand)] += cur[idx, ei][:, None] * r[:, 1:]
                cur[idx, ei] = 0.0
                alive[idx, ei] = False
    return np.argsort(~alive, axis=1, kind="stable")[:, :2]


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
    T = votes.sum(axis=1)                                               # (D, K) seat-wide primary votes
    primary = T / tot[:, None] * 100.0

    excl = {e.upper() for e in cfg.opv_finalist_exclude}
    el = np.array([j for j, n in enumerate(names) if n not in excl])
    buckets = [j for j, n in enumerate(names) if n in excl]
    if len(el) < 2:
        raise ValueError(f"Need at least two parties that can be finalists; modelled {names}, excluded {sorted(excl)}.")

    model = build_affinity(parsed, names, cfg)
    Z = rng.standard_normal((D, K, K))                                  # one affinity draw per posterior draw, shared by every step
    u = model.mu[None] + model.sd[None] * Z

    fin = _finalists(T, u, el, buckets, K)
    f1, f2 = fin[:, 0], fin[:, 1]
    swap = nm[f1] > nm[f2]                                              # pairs are stored alphabetically: ia < ib by name
    ia, ib = np.where(swap, f2, f1), np.where(swap, f1, f2)
    pair_ids = ia * K + ib

    VA, VB, VE = (np.zeros((D, B)) for _ in range(3))                   # votes to ia / to ib / exhausted
    for pid in np.unique(pair_ids):
        i, j = divmod(int(pid), K)
        idx = np.flatnonzero(pair_ids == pid)
        ctx = model.tables.get((i, j), {})                              # this pair's own table rows, where it has them
        va, vb, ve = votes[idx, :, i].copy(), votes[idx, :, j].copy(), np.zeros((len(idx), B))
        for k in range(K):
            if k in (i, j):
                continue
            mi, si, mj, sj = ctx[k] if k in ctx else (model.mu[k, i], model.sd[k, i], model.mu[k, j], model.sd[k, j])
            r = _softmax_exh(np.column_stack([mi + si * Z[idx, k, i], mj + sj * Z[idx, k, j]]))   # (n, 3): exhaust, to i, to j
            vk = votes[idx, :, k]
            ve += vk * r[:, 0:1]
            va += vk * r[:, 1:2]
            vb += vk * r[:, 2:3]
        VA[idx], VB[idx], VE[idx] = va, vb, ve

    sa, sb, se = VA.sum(axis=1), VB.sum(axis=1), VE.sum(axis=1)
    pct_a = 100.0 * sa / (sa + sb)
    winner = np.where(sa > sb, nm[ia], nm[ib])
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

    # Reported pairing = most frequent one; the preferred party (cfg.party) is listed first when it is in it
    ids, cnt = np.unique(pair_ids, return_counts=True)
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

    aff = pd.DataFrame([{"source": names[k], "candidate": names[c], "log_affinity": model.mu[k, c], "sd": model.sd[k, c],
                         "odds_vs_exhaust": float(np.exp(model.mu[k, c])), "basis": model.basis[k, c]}
                        for k in range(K) for c in el if k != c])

    w: List[str] = []
    rp = float(pair_probs.get(f"{min(pa, pb)} v {max(pa, pb)}", 0.0))
    if rp < cfg.opv_min_pair_prob:
        w.append(f"OPV: the reported pairing {pa} v {pb} occurs in only {100 * rp:.1f}% of draws; booth tables and plots are conditional on it "
                 f"(pairing probabilities: {', '.join(f'{k} {100 * v:.1f}%' for k, v in pair_probs.head(4).items())}). Win probabilities use all draws.")
    filled = aff[aff["basis"].isin(["default (source average)", "proxy"])]
    if len(filled):
        ex = ", ".join(f"{r.source}>{r.candidate}" for r in filled.head(6).itertuples())
        w.append(f"OPV: {len(filled)} of {len(aff)} source>candidate preference affinities have no flow table behind them and use the prior "
                 f"(sd {cfg.opv_affinity_sd:g} log-odds), e.g. {ex}{' ...' if len(filled) > 6 else ''}. See opv/affinities.csv.")
    return OpvResult(eligible=[names[e] for e in el], report_pair=(pa, pb), report_pair_prob=rp, pair_probs=pair_probs, win_probs=win_probs,
                     seat_draws=seat_draws, booth_2cp=booth, booth_primary_mean=pd.DataFrame(P.mean(axis=0), columns=names, index=list(booths)),
                     affinities=aff, flow_conc=cfg.opv_flow_conc, warnings=w)


def opv_trace(o: OpvResult) -> Dict[str, np.ndarray]:
    """The `s8_*` signals recorded in the trace (see pipeline.PIPELINE_STEPS)."""
    sd = o.seat_draws
    return {"s8_eligible": np.array(o.eligible), "s8_pair": sd["pair"].to_numpy(dtype=str),
            "s8_2cp": sd[[f"{e}_2CP_pct" for e in o.eligible]].to_numpy(dtype=float),
            "s8_exhausted": sd["exhausted_pct_formal"].to_numpy(dtype=float), "s8_winner": sd["winner"].to_numpy(dtype=str)}