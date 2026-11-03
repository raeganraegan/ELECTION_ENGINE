"""data.py -- raw tables in, clean tables out. Nothing here fits or simulates anything."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import ModelConfig

log = logging.getLogger("enki")

# NSW statewide primary vote (%), by election. Hard-coded: these results do not change.
NSW_STATE_RESULTS = pd.DataFrame({
    "year":  [1988,  1991,  1995,  1999,  2003,  2007,  2011,  2015,  2019,  2023],
    "ALP":   [38.48, 39.11, 41.27, 42.21, 42.68, 38.87, 25.55, 34.08, 33.31, 36.97],
    "LNP":   [49.54, 41.65, 39.81, 33.69, 34.35, 36.88, 51.15, 45.63, 41.58, 35.37],
    "GRN":   [ 1.50,  2.05,  2.85,  2.80,  4.10,  7.60, 10.28, 10.29,  9.57,  9.70],
    "ONP":   [ 0.00,  0.00,  0.00,  6.40,  0.70,  0.00,  0.00,  0.00,  1.10,  1.80],
    "OTHER": [10.48, 17.19, 16.07, 14.90, 18.17, 16.65, 12.65, 10.00, 14.90, 16.60],
})


# NSW general election dates, used to decide which polls belong to which election cycle (override with the `election_dates` input).
NSW_ELECTION_DATES = {1988: "1988-03-19", 1991: "1991-05-25", 1995: "1995-03-25", 1999: "1999-03-27", 2003: "2003-03-22",
                      2007: "2007-03-24", 2011: "2011-03-26", 2015: "2015-03-28", 2019: "2019-03-23", 2023: "2023-03-25"}


def read_frame(v, name: str) -> pd.DataFrame:
    """DataFrame | CSV path | list/dict of records -> DataFrame (always a copy)."""
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if isinstance(v, (str, Path)):
        return pd.read_csv(v)
    if isinstance(v, (list, dict)):
        return pd.DataFrame(v)
    raise TypeError(f"{name}: expected a DataFrame, a CSV path, or records; got {type(v).__name__}.")


def clean_numeric(s: pd.Series) -> pd.Series:
    """'1,234' -> 1234.0; anything unparseable -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def combine_parties(polls: pd.DataFrame, groups: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """Utility for poll tables, e.g. combine_parties(polls, {"OTHER": ["SFF", "OTH", "UND"]})."""
    p = polls.copy()
    for new, cols in groups.items():
        p[new] = p[[c for c in cols if c in p.columns]].fillna(0).sum(axis=1)
    return p


# ----------------------------------------------------------------------------- booth results
def booth_types(df: pd.DataFrame, cfg: ModelConfig) -> Dict[str, int]:
    """booth -> 1 (early / postal / declaration style) or 0 (ordinary), from `booth_type` if present, else the booth name."""
    kws = tuple(k.upper() for k in cfg.early_keywords)
    text = df["booth_type"].where(df["booth_type"].notna(), df["booth"]) if "booth_type" in df.columns else df["booth"]
    flag = text.astype(str).str.upper().map(lambda s: int(any(k in s for k in kws)))
    return flag.groupby(df["booth"]).max().astype(int).to_dict()


def check_booth_data(df: pd.DataFrame, cfg: ModelConfig) -> List[str]:
    """Reconciliation problems in the booth table (never raises). `df` must already be numeric, one row per CSV row."""
    w = df.copy()
    w["_n"] = w["total_votes"] - w["informal_votes"]
    issues: List[str] = []
    if "IND" in w.columns:           # without an IND column OTHER is the remainder by design, so there is nothing to reconcile
        named = [c for c in ("ALP", "LNP", "GRN", "IND", "ONP") if c in w.columns]
        w["_gap"] = w["_n"] - w[named].sum(axis=1)
        off = w[w["_gap"] != 0]
        if len(off):
            worst = off.loc[off["_gap"].abs().idxmax()]
            issues.append(f"{len(off)} of {len(w)} rows: named parties do not sum to formal votes "
                          f"({int((off['_gap'] < 0).sum())} rows have MORE named votes than formal). "
                          f"Worst: {worst['booth']} {int(worst['year'])}, gap {int(worst['_gap']):+d} votes. "
                          "The gap is silently dumped into OTHER.")
    dup = int(w.duplicated(["year", "booth"]).sum())
    if dup:
        issues.append(f"{dup} duplicate booth-year row(s); they are summed together, which double counts accidental repeats.")
    tot = w.groupby("year")["_n"].sum().sort_index()
    jump = tot.pct_change().dropna()
    for yr, ch in jump[jump.abs() > cfg.data_check_jump_threshold].items():
        issues.append(f"Seat formal total moves {100 * ch:+.0f}% into {int(yr)} ({int(tot.loc[yr]):,} votes); "
                      "check for missing booths or vote types (postal, iVote, absent).")
    return issues


def prepare_booths(booths: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Raw booth CSV -> one clean row per (year, booth) with formal votes `n`, per-party counts `y_<party>`,
    the target party's `y` / `share` (%), and `btype` (1 = early-voting style booth)."""
    parties = list(cfg.parties)
    explicit = [p for p in parties if p != "OTHER"]
    need = {"year", "booth", "total_votes", "informal_votes"} | set(explicit)
    missing = need - set(booths.columns)
    if missing:
        raise ValueError(f"booth_results is missing columns: {sorted(missing)}")
    if cfg.party not in parties:
        raise ValueError(f"cfg.party={cfg.party!r} must be one of cfg.parties={parties}")

    df = booths.dropna(subset=["booth", "year"]).copy()
    df["booth"] = df["booth"].astype(str).str.strip()
    if cfg.booth_merge:
        df["booth"] = df["booth"].replace(cfg.booth_merge)
    df = df[~df["booth"].isin(cfg.exclude_booths)].copy()

    counted = ["total_votes", "informal_votes"] + explicit + [c for c in ("IND", "ONP") if c in df.columns and c not in explicit]
    for c in counted:
        df[c] = clean_numeric(df[c])
    needed_now = ["total_votes", "informal_votes"] + explicit
    bad = df[df[needed_now].isna().any(axis=1)]
    if len(bad):
        raise ValueError(f"Non-numeric / blank vote counts in {len(bad)} row(s), e.g. booth={bad.iloc[0]['booth']!r}, "
                         f"year={bad.iloc[0]['year']}.")
    df["year"] = df["year"].astype(int)

    for msg in check_booth_data(df, cfg):
        log.warning("DATA CHECK: %s", msg)

    type_map = booth_types(df, cfg)
    df = df.groupby(["year", "booth"], as_index=False)[needed_now].sum()
    df["btype"] = df["booth"].map(type_map).fillna(0).astype(int)
    df["n"] = (df["total_votes"] - df["informal_votes"]).round().astype(int)
    dropped = int((df["n"] <= 0).sum())
    if dropped:
        log.warning("dropped %d row(s) with non-positive formal votes.", dropped)
    df = df[df["n"] > 0].copy()
    if df.empty:
        raise ValueError("No valid booths remain after filtering.")

    for p in explicit:
        df[f"y_{p}"] = df[p].round().astype(int)
    if "OTHER" in parties:
        df["y_OTHER"] = df["n"] - sum(df[f"y_{p}"] for p in explicit)
        if (df["y_OTHER"] < 0).any():
            raise ValueError("Explicit party votes exceed formal votes in some rows; check the CSV.")
    df["y"] = df[f"y_{cfg.party}"]
    df["share"] = df["y"] / df["n"] * 100.0
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- statewide results
def state_matrix(state_df: pd.DataFrame, parties: Sequence[str], years: Sequence[int],
                 onp_in_denominator: bool = False) -> pd.DataFrame:
    """Statewide shares (%) of the modelled parties, one row per year, renormalised to sum to 100
    (or, with onp_in_denominator, to 100 minus ONP, so a high ONP does not inflate the others)."""
    full = state_df.set_index("year")
    absent_cols = [p for p in parties if p not in full.columns]
    if absent_cols:
        raise ValueError(f"state_results is missing columns: {absent_cols}")
    absent = [y for y in years if y not in full.index]
    if absent:
        raise ValueError(f"State results missing for years: {absent}")
    S = full.loc[list(years), list(parties)].astype(float)
    if not (1.0 < S.values.max() < 100.0):
        raise ValueError("State shares must be percentages (e.g. 36.97), not fractions.")
    denom = S.sum(axis=1)
    if onp_in_denominator:
        if "ONP" not in full.columns:
            raise ValueError("onp_state_denominator needs an ONP column in state_results.")
        if "ONP" not in parties:
            denom = denom + full.loc[list(years), "ONP"].astype(float)
    return S.div(denom, axis=0) * 100.0
