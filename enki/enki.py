"""enki.py -- ENKI election forecast: all mathematics (S0-S8), config, data prep, backtest.

Non-math (IO, plots, PDF, terminal report) lives in helpers.py.

ENKI(x) -> ENKIOutput. x is ONE flat dict (or path to .json). Keys may sit at the top level
or inside `data`, `config`, `run` groups (merged; a key may appear only once). Unknown keys
raise KeyError.
"""
from __future__ import annotations

import difflib
import gc
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

log = logging.getLogger("enki")

# ----------------------------------------------------------------------------- constants / presets
LAYERS = ("poll", "params", "tide", "booth", "volumes")
SCENARIOS = ("worst", "average", "best")
ONP_MODES = ("none", "brian", "entry")
TIDE_PROCESSES = ("ar1", "random_walk")
VOLUME_DRIFTS = ("last", "mean", "none")
BACKTEST_STATE_MODES = ("auto", "oracle", "polls", "noised")
BACKTEST_VOLUME_MODES = ("actual", "modelled")
BETA_PRESETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "data":    {"default": (0.3, 0.30), "GRN": (0.3, 0.35)},
    "uniform": {"default": (1.0, 0.15)},
}
BRIAN_DEFAULT: Dict[str, float] = {"LNP": 0.20, "ALP": 0.05}
FLOW_FLOOR = 1e-3
SELF_AFFINITY = 0.0
DEFAULT_NOISED_HORIZON_DAYS = 180.0
ATTRIBUTION_SEED_OFFSET = 101
OPV_SEED_OFFSET = 202
INTERVALS = (50, 80, 90, 95)

def _load_default(name: str) -> Any:
    """Load a default data file from data/ (fallback: cwd); None if absent."""
    p = Path("data") / name
    if not p.exists():
        p = Path(name)
    if not p.exists():
        return None
    if name.endswith(".csv"):
        return pd.read_csv(p)
    if name.endswith(".json"):
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown default file type: {name}")


RENAMED: Dict[str, str] = {
    "brian_mode": "onp_mode='brian'",
    "onp_layer": "onp_mode='entry'",
    "brian_onp_from_lnp": "brian_transfer={'LNP': ..., 'ALP': ...}",
    "brian_onp_from_alp": "brian_transfer={'LNP': ..., 'ALP': ...}",
    "local_priors": "beta_priors={'ALP': (mean, sd), ...}",
    "beta_g": "beta_priors (per-party (mean, sd))",
    "beta_prior_sd": "beta_priors (per-party (mean, sd))",
    "beta_data_mu": "beta_preset / beta_priors", "beta_data_sd": "beta_preset / beta_priors",
    "beta_uniform_mu": "beta_preset / beta_priors", "beta_uniform_sd": "beta_preset / beta_priors",
    "beta_grn_sd": "beta_priors={'GRN': (mean, sd)}",
    "beta_other_mu": "nothing: the reference party has no beta", "beta_other_sd": "nothing: the reference party has no beta",
    "backtest_poll_house_sd": "nothing: the backtest uses the true statewide result, not polls",
    "attribution_seed_offset": "nothing: now a constant",
}

# ----------------------------------------------------------------------------- config
@dataclass
class PollInput:
    central: float
    low: float
    high: float


@dataclass
class ModelConfig:
    seat_name: str
    party: str
    parties: Sequence[str]
    election_date: str
    prediction_date: Optional[str]
    scenario_pct: Sequence[float]

    exclude_booths: Sequence[str]
    booth_merge: Optional[Dict[str, str]]
    early_keywords: Sequence[str]
    data_check_jump_threshold: float

    beta_preset: str
    beta_priors: Optional[Dict[str, Tuple[float, float]]]
    beta_presets: Optional[Dict[str, Dict[str, Sequence[float]]]]
    brian_default: Optional[Dict[str, float]]
    sigma_beta_sd: float
    m_prior_sd: float
    sigma_alpha_sd: float
    sigma_eps_sd: Optional[float]
    sigma_d_median: Optional[float]
    sigma_d_sdlog: float
    tide_process: str
    d_persist: Optional[float]
    rho_beta_a: float
    rho_beta_b: float
    use_correlated_errors: bool
    lkj_eta: float
    include_eps: bool

    draws: int
    tune: int
    chains: int
    seed: int
    target_accept: float
    rhat_threshold: float
    ess_threshold: float

    poll_half_life_months: float
    poll_as_of: Optional[str]
    poll_moe: Optional[float]
    poll_house_sd: float
    time_horizon_gamma: float
    poll_z: float
    poll_default_sample_size: float
    poll_dirichlet_conc_floor: float
    poll_sum_warn: float
    share_clip_floor: float

    registered_electors: Optional[float]
    total_formal_votes: Optional[float]
    expected_turnout_rate: float
    informal_rate_fallback: float
    election_cycle_years: float
    volume_draws: Optional[int]
    volume_tune_floor: int
    volume_target_accept: float
    volume_drift: str
    volume_step_floor: float
    volume_total_sd_floor: float
    vol_sigma_a_sd: float
    vol_tau_sd: float
    vol_kappa_alpha: float
    vol_kappa_beta: float

    onp_mode: str
    onp_state_denominator: bool
    brian_transfer: Optional[Dict[str, float]]
    onp_seat_ratio_median: float
    onp_seat_ratio_sdlog: float
    onp_max_seat_share: float

    opv_flows: Optional[Dict[str, Dict[str, Sequence[float]]]]
    opv_flow_conc: Optional[float]
    opv_finalist_exclude: Sequence[str]
    opv_min_pair_prob: float
    opv_affinity_sd: float
    opv_entrants: Sequence[str]
    opv_proxies: Optional[Dict[str, Dict[str, float]]]


@dataclass
class RunOptions:
    run_forecast: bool
    run_backtest: bool
    backtest_targets: Optional[Sequence[int]]
    backtest_state_mode: str
    backtest_min_polls: int
    backtest_volume_mode: str
    backtest_poll_horizon_days: Optional[float]
    run_volume_backtest: bool
    backtest_volume_sims: int
    run_attribution: bool
    run_opv: bool
    make_pdf: bool
    pdf_tier_cuts: Sequence[float]
    output_root: Optional[str]
    make_plots: bool
    print_report: bool
    record_trace: bool
    verbose: bool
    plot_top_n_booths: int
    plot_dpi: int
    plot_volume_log_scale: bool
    plot_label_all_booths: bool
    plot_ridgeline_max_draws: int
    plot_ridgeline_keywords: Sequence[str]


def beta_prior_table(cfg: ModelConfig) -> Dict[str, Tuple[float, float]]:
    presets = cfg.beta_presets if cfg.beta_presets is not None else BETA_PRESETS
    preset = presets[cfg.beta_preset]
    out = {p: preset.get(p, preset["default"]) for p in list(cfg.parties)[:-1]}
    for p, ms in (cfg.beta_priors or {}).items():
        out[p] = (float(ms[0]), float(ms[1]))
    return out


def validate(cfg: ModelConfig, run: RunOptions) -> List[str]:
    err: List[str] = []
    warn: List[str] = []
    P = list(cfg.parties)
    if len(P) < 2 or len(set(P)) != len(P):
        err.append("parties must have at least two distinct entries.")
    if cfg.party not in P:
        err.append(f"party={cfg.party!r} must be one of parties={P}.")
    if "OTHER" in P and P[-1] != "OTHER":
        warn.append("OTHER is not the last party; the LAST party is the reference category (no beta).")

    for n in ("draws", "tune", "chains"):
        if int(getattr(cfg, n)) < 1:
            err.append(f"{n} must be >= 1.")
    for n in ("target_accept", "volume_target_accept"):
        if not 0.0 < float(getattr(cfg, n)) < 1.0:
            err.append(f"{n} must be strictly between 0 and 1.")
    sp = [float(v) for v in cfg.scenario_pct]
    if len(sp) != 3 or not (0.0 < sp[0] < sp[1] < sp[2] < 100.0):
        err.append("scenario_pct must be three increasing percentiles in (0, 100), e.g. (5, 50, 95).")

    positive = ("sigma_beta_sd", "sigma_alpha_sd", "sigma_d_sdlog", "m_prior_sd", "rho_beta_a", "rho_beta_b",
                "lkj_eta", "vol_sigma_a_sd", "vol_tau_sd", "vol_kappa_alpha", "vol_kappa_beta",
                "poll_half_life_months", "poll_z", "election_cycle_years", "poll_default_sample_size",
                "volume_step_floor", "poll_dirichlet_conc_floor", "share_clip_floor")
    for n in positive:
        if not float(getattr(cfg, n)) > 0.0:
            err.append(f"{n} must be > 0.")
    for n in ("sigma_eps_sd", "sigma_d_median"):
        v = getattr(cfg, n)
        if v is not None and not float(v) > 0.0:
            err.append(f"{n} must be > 0 (or None for auto-calibration).")
    for n in ("poll_house_sd", "time_horizon_gamma", "volume_total_sd_floor"):
        if float(getattr(cfg, n)) < 0.0:
            err.append(f"{n} must be >= 0.")

    if cfg.d_persist is not None and not -1.0 < float(cfg.d_persist) < 1.0:
        err.append("d_persist must be None (estimate it) or a number in (-1, 1).")
    presets = cfg.beta_presets if cfg.beta_presets is not None else BETA_PRESETS
    if cfg.beta_preset not in presets:
        err.append(f"beta_preset must be one of {sorted(presets)}.")
    if cfg.tide_process not in TIDE_PROCESSES:
        err.append(f"tide_process must be one of {TIDE_PROCESSES}.")
    if cfg.volume_drift not in VOLUME_DRIFTS:
        err.append(f"volume_drift must be one of {VOLUME_DRIFTS}.")
    if not 0.0 < float(cfg.expected_turnout_rate) <= 1.0:
        err.append("expected_turnout_rate must be in (0, 1].")
    if cfg.registered_electors is not None and float(cfg.registered_electors) <= 0.0:
        err.append("registered_electors must be positive (or None).")

    for p, ms in (cfg.beta_priors or {}).items():
        if p not in P[:-1]:
            err.append(f"beta_priors names {p!r}, which is not a non-reference modelled party {P[:-1]}.")
        elif len(ms) != 2 or not float(ms[1]) > 0.0:
            err.append(f"beta_priors[{p!r}] must be (mean, sd) with sd > 0.")

    if cfg.onp_mode not in ONP_MODES:
        err.append(f"onp_mode must be one of {ONP_MODES}.")
    if cfg.onp_mode in ("brian", "entry") and "ONP" in P:
        err.append(f"onp_mode={cfg.onp_mode!r} builds ONP itself, so 'ONP' must not be in parties.")
    if cfg.onp_mode == "brian":
        tr = cfg.brian_transfer or BRIAN_DEFAULT
        if not set(tr) & set(P):
            err.append(f"brian_transfer {sorted(tr)} names none of the modelled parties {P}.")
        for p, f in tr.items():
            if p not in P:
                warn.append(f"brian_transfer names {p!r}, which is not modelled; it is ignored.")
            if not 0.0 <= float(f) <= 1.0:
                err.append(f"brian_transfer[{p!r}] must be between 0 and 1.")
    elif cfg.brian_transfer is not None:
        warn.append("brian_transfer is set but onp_mode is not 'brian'; it is ignored.")
    if not 0.0 <= float(cfg.onp_max_seat_share) <= 1.0:
        err.append("onp_max_seat_share must be between 0 and 1.")
    if float(cfg.onp_seat_ratio_median) <= 0.0 or float(cfg.onp_seat_ratio_sdlog) < 0.0:
        err.append("onp_seat_ratio_median must be > 0 and onp_seat_ratio_sdlog >= 0.")

    try:
        pred, elec = pd.to_datetime(cfg.prediction_date), pd.to_datetime(cfg.election_date)
        if elec < pred:
            warn.append(f"election_date {cfg.election_date} is before prediction_date {cfg.prediction_date}: "
                        "the time-horizon uncertainty collapses to 0.")
    except Exception:
        err.append("prediction_date / election_date must be parseable dates (YYYY-MM-DD).")

    if cfg.opv_flow_conc is not None and not float(cfg.opv_flow_conc) > 0.0:
        err.append("opv_flow_conc must be > 0 (or None for fixed flows).")
    if not 0.0 <= float(cfg.opv_min_pair_prob) <= 1.0:
        err.append("opv_min_pair_prob must be between 0 and 1.")
    if not (len(run.pdf_tier_cuts) == 2 and 0.0 < float(run.pdf_tier_cuts[0]) < float(run.pdf_tier_cuts[1]) < 1.0):
        err.append("pdf_tier_cuts must be two increasing probabilities between 0 and 1, e.g. (0.40, 0.80).")
    if not float(cfg.opv_affinity_sd) > 0.0:
        err.append("opv_affinity_sd must be > 0.")
    for e, d in (cfg.opv_proxies or {}).items():
        if not isinstance(d, dict) or not d or any(float(w) <= 0.0 for w in d.values()):
            err.append(f"opv_proxies[{e!r}] must be a non-empty {{party: positive weight}} dict.")
    if run.backtest_state_mode not in BACKTEST_STATE_MODES:
        err.append(f"backtest_state_mode must be one of {BACKTEST_STATE_MODES}.")
    if run.backtest_volume_mode not in BACKTEST_VOLUME_MODES:
        err.append(f"backtest_volume_mode must be one of {BACKTEST_VOLUME_MODES}.")
    if int(run.backtest_min_polls) < 1:
        err.append("backtest_min_polls must be >= 1.")
    if run.backtest_poll_horizon_days is not None and float(run.backtest_poll_horizon_days) < 0:
        err.append("backtest_poll_horizon_days must be >= 0.")
    if run.output_root is None and run.make_plots:
        warn.append("make_plots is on but output_root is None, so no plots are drawn.")
    if err:
        raise ValueError("ENKI(x): invalid input:\n  - " + "\n  - ".join(err))
    return warn


# ----------------------------------------------------------------------------- data
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
    p = polls.copy()
    for new, cols in groups.items():
        p[new] = p[[c for c in cols if c in p.columns]].fillna(0).sum(axis=1)
    return p


def booth_types(df: pd.DataFrame, cfg: ModelConfig) -> Dict[str, int]:
    kws = tuple(k.upper() for k in cfg.early_keywords)
    text = df["booth_type"].where(df["booth_type"].notna(), df["booth"]) if "booth_type" in df.columns else df["booth"]
    flag = text.astype(str).str.upper().map(lambda s: int(any(k in s for k in kws)))
    return flag.groupby(df["booth"]).max().astype(int).to_dict()


def check_booth_data(df: pd.DataFrame, cfg: ModelConfig) -> List[str]:
    w = df.copy()
    w["_n"] = w["total_votes"] - w["informal_votes"]
    issues: List[str] = []
    if "IND" in w.columns:
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


def state_matrix(state_df: pd.DataFrame, parties: Sequence[str], years: Sequence[int],
                 onp_in_denominator: bool = False) -> pd.DataFrame:
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


# ----------------------------------------------------------------------------- share model (Steps 0, 1, 3, 4)
_SD_D_FROM_SEAT_STD = 1.225
_SD_EPS_FROM_BOOTH_STD = 0.8


@dataclass
class Design:
    parties: list
    years: list
    S_pct: pd.DataFrame
    x: np.ndarray
    x_center: np.ndarray
    booth_names: list
    booth_idx: np.ndarray
    year_idx: np.ndarray
    Y: np.ndarray


def build_design(df: pd.DataFrame, cfg: ModelConfig, state_df: pd.DataFrame) -> Design:
    parties = list(cfg.parties)
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) < 2:
        raise ValueError("Need at least two elections to fit the model.")
    S = state_matrix(state_df, parties, years, cfg.onp_state_denominator)
    logS = np.log(S.values / 100.0)
    x_center = logS.mean(axis=0)
    booth_names = sorted(df["booth"].unique())
    return Design(
        parties=parties, years=years, S_pct=S, x=logS - x_center, x_center=x_center, booth_names=booth_names,
        booth_idx=df["booth"].map({b: i for i, b in enumerate(booth_names)}).to_numpy(),
        year_idx=df["year"].map({y: i for i, y in enumerate(years)}).to_numpy(),
        Y=df[[f"y_{p}" for p in parties]].to_numpy().astype(int),
    )


def empirical_scales(df: pd.DataFrame, cfg: ModelConfig) -> Tuple[float, float]:
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

    log_shares = np.log(np.clip(des.Y / des.Y.sum(axis=1, keepdims=True), 1e-4, 1.0))
    m0 = log_shares.mean(axis=0)
    m0 = m0 - m0.mean()

    priors = beta_prior_table(cfg)
    auto_d, auto_eps = empirical_scales(df, cfg) if (cfg.sigma_d_median is None or cfg.sigma_eps_sd is None) else (None, None)
    sigma_d_med = cfg.sigma_d_median if cfg.sigma_d_median is not None else auto_d
    sigma_eps_sc = cfg.sigma_eps_sd if cfg.sigma_eps_sd is not None else auto_eps
    random_walk = cfg.tide_process == "random_walk"
    est_rho = (cfg.d_persist is None) and not random_walk
    coords = {"booth": des.booth_names, "year": des.years, "obs": np.arange(N), "party": parties}

    with pm.Model(coords=coords):
        prior_mu = np.array([priors[p][0] for p in parties[:-1]], dtype=float)
        prior_sd = np.array([priors[p][1] for p in parties[:-1]], dtype=float)
        raw_mu = np.append(prior_mu, -prior_mu.sum())
        raw_sd = np.append(prior_sd, prior_sd.mean())
        mu_beta_raw = pm.Normal("mu_beta_raw", mu=raw_mu, sigma=raw_sd, shape=K)
        mu_beta = pm.Deterministic("mu_beta_vec", mu_beta_raw - mu_beta_raw.mean(), dims="party")

        sigma_beta = pm.HalfNormal("sigma_beta", sigma=cfg.sigma_beta_sd, shape=K)
        beta_z = pm.Normal("beta_z", 0.0, 1.0, dims=("booth", "party"))
        beta_matrix = pm.Deterministic(
            "beta_matrix", mu_beta[None, :] + beta_z * sigma_beta[None, :], dims=("booth", "party"))

        m_raw = pm.Normal("m_raw", mu=m0, sigma=cfg.m_prior_sd, shape=K)
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

        max_logit_bound = 3.5
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

    max_logit_bound = 3.5
    eta = max_logit_bound * np.tanh(raw_eta / max_logit_bound)

    eta = eta - eta.max(axis=2, keepdims=True)
    e = np.exp(eta)
    Km = K - 1
    return ShareSim(e / e.sum(axis=2, keepdims=True), lin_party[:, :Km] - lin_party[:, Km:], x_next, bx, d_next, lin_party)


# ----------------------------------------------------------------------------- polls (Step 2)
@dataclass
class PollSummary:
    central: np.ndarray
    onp_central: Optional[float]
    target: PollInput
    table: Optional[pd.DataFrame]


@dataclass
class StatewideDraws:
    S: np.ndarray
    onp: Optional[np.ndarray]
    central: np.ndarray
    target: PollInput
    se_parts: np.ndarray
    conc: float


def _pct_scale(polls: pd.DataFrame, cols: Sequence[str]) -> float:
    v = polls[[c for c in cols if c in polls.columns]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if v.size == 0 or np.isnan(v).all():
        return 1.0
    return 100.0 if np.nanmax(v) <= 1.0 else 1.0


def _reference_date(polls: pd.DataFrame, cfg: ModelConfig) -> pd.Timestamp:
    return pd.to_datetime(cfg.poll_as_of) if cfg.poll_as_of else pd.to_datetime(polls["date"]).max()


def _weights(dates: pd.Series, ref: pd.Timestamp, half_life: float) -> pd.Series:
    months = (ref.year - dates.dt.year) * 12 + (ref.month - dates.dt.month)
    return 0.5 ** (months / half_life)


def weighted_average(polls: pd.DataFrame, party: str, cfg: ModelConfig, scale: float) -> PollInput:
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


# ----------------------------------------------------------------------------- onp (Step 6)
@dataclass
class OnpResult:
    p: np.ndarray
    mode: str
    out_parties: List[str]
    target_scale: np.ndarray
    retain: np.ndarray
    seat_share: Optional[np.ndarray]


def apply_onp(p: np.ndarray, onp_state_pct: Optional[np.ndarray], parties: Sequence[str], k0: int,
              cfg: ModelConfig, rng: np.random.Generator) -> OnpResult:
    parties = list(parties)
    D = p.shape[0]
    if cfg.onp_mode == "none":
        return OnpResult(p, "none", parties, np.ones(D), np.zeros(0), None)

    if cfg.onp_mode == "brian":
        brian_default = cfg.brian_default if cfg.brian_default is not None else BRIAN_DEFAULT
        take = cfg.brian_transfer or brian_default
        retain = np.array([1.0 - float(take.get(q, 0.0)) for q in parties])
        onp = (p * (1.0 - retain)[None, None, :]).sum(axis=2)
        return OnpResult(np.concatenate([p * retain[None, None, :], onp[:, :, None]], axis=2), "brian",
                         parties + ["ONP"], np.full(D, retain[k0]), retain, None)

    if onp_state_pct is None:
        raise ValueError("onp_mode='entry' needs the statewide ONP draws.")

    ratio = np.exp(np.log(cfg.onp_seat_ratio_median) + cfg.onp_seat_ratio_sdlog * rng.standard_normal(D))
    s = np.clip(ratio * np.asarray(onp_state_pct, dtype=float) / 100.0, 0.0, cfg.onp_max_seat_share)

    default_entry_weights = {
        "LNP": 0.50, "LIB": 0.50, "NAT": 0.50,
        "ALP": 0.35,
        "OTHER": 0.12, "OTH": 0.12,
        "GRN": 0.03
    }
    weights_dict = getattr(cfg, "onp_entry_weights", None) or default_entry_weights
    w_base = np.array([float(weights_dict.get(q, 0.10)) for q in parties], dtype=float)
    w_base = w_base / w_base.sum()

    conc = getattr(cfg, "opv_flow_conc", 133.0) or 133.0
    alpha = np.maximum(w_base, FLOW_FLOOR) * float(conc)
    w_draws = rng.dirichlet(alpha, size=D)

    B = p.shape[1]
    weighted_pool = (p * w_draws[:, None, :]).sum(axis=2, keepdims=True)
    mean_pool = np.maximum(weighted_pool.mean(axis=1, keepdims=True), 1e-6)
    booth_onp_share = s[:, None, None] * (weighted_pool / mean_pool)
    booth_onp_share = np.clip(booth_onp_share, 0.0, 0.50)

    deduction = booth_onp_share * (w_draws[:, None, :] * p / np.maximum(weighted_pool, 1e-6))
    p_retained = np.maximum(0.0, p - deduction)
    onp_booth = p.sum(axis=2, keepdims=True) - p_retained.sum(axis=2, keepdims=True)

    out = np.concatenate([p_retained, onp_booth], axis=2)
    return OnpResult(out, "entry", parties + ["ONP"], 1.0 - s, np.zeros(0), s)


# ----------------------------------------------------------------------------- opv (Step 8)
_PAIR_KEY = re.compile(r"^\s*([A-Za-z0-9]+)_vs_([A-Za-z0-9]+)\s*$")


@dataclass
class OpvResult:
    eligible: List[str]
    report_pair: Tuple[str, str]
    report_pair_prob: float
    pair_probs: pd.Series
    win_probs: pd.Series
    seat_draws: pd.DataFrame
    booth_2cp: pd.DataFrame
    booth_primary_mean: pd.DataFrame
    affinities: pd.DataFrame
    flow_conc: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def _load_default_pref_flows() -> Dict[str, Any]:
    """Load OPV preference flows from sample_pref_flows.csv -> {pair: {source: [to_a, to_b, exhausted]}}."""
    df = _load_default("sample_pref_flows.csv")
    if df is None:
        return None
    out: Dict[str, Any] = {}
    for _, r in df.iterrows():
        pair = str(r["pair"])
        src = str(r["source"])
        vals = [float(r["to_first"]), float(r["to_second"]), float(r["exhausted"])]
        out.setdefault(pair, {})[src] = vals
    return out


def resolve_flows(cfg: ModelConfig) -> Dict[str, Any]:
    if cfg.opv_flows is not None:
        return cfg.opv_flows
    flows = _load_default_pref_flows()
    if flows is None:
        raise ValueError("resolve_flows: opv_flows is required (config or sample_pref_flows.csv).")
    return flows


def out_party_names(cfg: ModelConfig) -> List[str]:
    names = [p.upper() for p in cfg.parties]
    return names + (["ONP"] if cfg.onp_mode != "none" and "ONP" not in names else [])


def _parse_flows(flows: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
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


def _trigamma(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for _ in range(8):
        small = x < 6.0
        out = out + np.where(small, 1.0 / x ** 2, 0.0)
        x = np.where(small, x + 1.0, x)
    return out + 1.0 / x + 1.0 / (2 * x ** 2) + 1.0 / (6 * x ** 3) - 1.0 / (30 * x ** 5)


def _softmax_exh(U: np.ndarray) -> np.ndarray:
    L = np.concatenate([np.zeros((U.shape[0], 1)), U], axis=1)
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


@dataclass
class AffinityModel:
    names: List[str]
    mu: np.ndarray
    sd: np.ndarray
    basis: np.ndarray
    tables: Dict[Tuple[int, int], Dict[int, Tuple[float, float, float, float]]]


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
            r = [r[1], r[0], r[2]] if a > b else list(r)
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
    for (k, c), lst in est.items():
        v = np.array([x[0] for x in lst])
        var = float(np.mean([x[1] for x in lst])) + (float(v.var(ddof=1)) if len(v) > 1 else 0.0)
        mu[k, c], sd[k, c], known[k, c], basis[k, c] = float(v.mean()), float(np.sqrt(var)), True, "table"
        if names[k] in ents or names[c] in ents:
            sd[k, c], basis[k, c] = max(sd[k, c], wide), "table (entrant, widened)"

    base = np.array([float(mu[k][known[k]].mean()) if known[k].any() else 0.0 for k in range(K)])
    for k in range(K):
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
            if pc:
                mu[k, c], basis[k, c] = sum(w * m(k, ix[p]) for p, w in pc.items()) / sum(pc.values()), "proxy"
            elif pk:
                mu[k, c], basis[k, c] = sum(w * m(ix[p], c) for p, w in pk.items()) / sum(pk.values()), "proxy"
    return AffinityModel(names, mu, sd, basis, tables)


def _finalists(T: np.ndarray, u: np.ndarray, elig: np.ndarray, buckets: Sequence[int], K: int) -> np.ndarray:
    D = T.shape[0]
    cur = T.copy()
    alive = np.zeros((D, K), dtype=bool)
    alive[:, elig] = True
    if len(elig) > 2:
        for s in buckets:
            r = _softmax_exh(u[:, s, :][:, elig])
            cur[:, elig] += cur[:, [s]] * r[:, 1:]
            cur[:, s] = 0.0
        pw = 1 << np.arange(K)
        for _ in range(len(elig) - 2):
            e = np.where(alive, cur, np.inf).argmin(axis=1)
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


def compute_opv(V: np.ndarray, P: np.ndarray, parties: Sequence[str], booths: Sequence[str], cfg: ModelConfig,
                flows: Dict[str, Any], rng: np.random.Generator) -> OpvResult:
    names = [str(p).upper() for p in parties]
    nm = np.array(names)
    parsed = _parse_flows(flows)
    V, P = np.asarray(V, dtype=float), np.asarray(P, dtype=float)
    D, B, K = P.shape
    votes = V[:, :, None] * P / 100.0
    tot = V.sum(axis=1)
    T = votes.sum(axis=1)
    primary = T / tot[:, None] * 100.0

    excl = {e.upper() for e in cfg.opv_finalist_exclude}
    el = np.array([j for j, n in enumerate(names) if n not in excl])
    buckets = [j for j, n in enumerate(names) if n in excl]
    if len(el) < 2:
        raise ValueError(f"Need at least two parties that can be finalists; modelled {names}, excluded {sorted(excl)}.")

    model = build_affinity(parsed, names, cfg)
    Z = rng.standard_normal((D, K, K))
    u = model.mu[None] + model.sd[None] * Z

    fin = _finalists(T, u, el, buckets, K)
    f1, f2 = fin[:, 0], fin[:, 1]
    swap = nm[f1] > nm[f2]
    ia, ib = np.where(swap, f2, f1), np.where(swap, f1, f2)
    pair_ids = ia * K + ib

    VA, VB, VE = (np.zeros((D, B)) for _ in range(3))
    for pid in np.unique(pair_ids):
        i, j = divmod(int(pid), K)
        idx = np.flatnonzero(pair_ids == pid)
        ctx = model.tables.get((i, j), {})
        va, vb, ve = votes[idx, :, i].copy(), votes[idx, :, j].copy(), np.zeros((len(idx), B))
        for k in range(K):
            if k in (i, j):
                continue
            mi, si, mj, sj = ctx[k] if k in ctx else (model.mu[k, i], model.sd[k, i], model.mu[k, j], model.sd[k, j])
            r = _softmax_exh(np.column_stack([mi + si * Z[idx, k, i], mj + sj * Z[idx, k, j]]))
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
    va, vb = ((VA, VB) if A == i else (VB, VA))[0][m], ((VA, VB) if A == i else (VB, VA))[1][m]
    ve, vt = VE[m], V[m]
    den = va + vb
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(den > 0, 100.0 * va / den, np.nan)
    pa, pb = names[A], names[Bq]
    booth = pd.DataFrame({
        "booth": list(booths), "expected_formal_votes": vt.mean(axis=0),
        f"{pa}_2cp_votes_mean": va.mean(axis=0), f"{pb}_2cp_votes_mean": vb.mean(axis=0), "exhausted_votes_mean": ve.mean(axis=0),
        f"{pa}_primary_votes_mean": votes[m][:, :, A].mean(axis=0), f"{pb}_primary_votes_mean": votes[m][:, :, Bq].mean(axis=0),
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
    sd = o.seat_draws
    return {"s8_eligible": np.array(o.eligible), "s8_pair": sd["pair"].to_numpy(dtype=str),
            "s8_2cp": sd[[f"{e}_2CP_pct" for e in o.eligible]].to_numpy(dtype=float),
            "s8_exhausted": sd["exhausted_pct_formal"].to_numpy(dtype=float), "s8_winner": sd["winner"].to_numpy(dtype=str)}


# ----------------------------------------------------------------------------- volume model (Step 5)
@dataclass
class VolumeFit:
    names: list
    years: list
    btype: np.ndarray
    t0: int
    totals: np.ndarray
    divergences: int
    diag: pd.DataFrame
    draws: dict


@dataclass
class VolumeResult:
    V: np.ndarray
    mode: str
    active: np.ndarray
    vfit: Optional[VolumeFit] = None


def projected_votes_from_electors(df: pd.DataFrame, cfg: ModelConfig) -> pd.Series:
    if cfg.registered_electors is None:
        raise ValueError("registered_electors is required (set cfg.registered_electors).")
    total, informal = df["total_votes"].sum(), df["informal_votes"].sum()
    informal_rate = informal / total if total > 0 else cfg.informal_rate_fallback
    latest = df[df["year"] == df["year"].max()].set_index("booth")["n"]
    return latest / latest.sum() * (cfg.registered_electors * cfg.expected_turnout_rate * (1.0 - informal_rate))


def fit_volume_model(df: pd.DataFrame, cfg: ModelConfig) -> VolumeFit:
    if not HAS_PYMC:
        raise ImportError("PyMC is required to fit the volume model: pip install pymc arviz")
    names = sorted(df["booth"].unique())
    years = sorted(int(y) for y in df["year"].unique())
    B, T = len(names), len(years)
    C = df.pivot(index="year", columns="booth", values="n").reindex(index=years, columns=names).values
    present = ~np.isnan(C)
    btype = df.groupby("booth")["btype"].max().reindex(names).fillna(0).values.astype(float)
    has_early = bool(btype.sum() > 0)
    t0 = next((ti for ti in range(T) if present[ti][btype == 1].any()), T)
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
        k = max(1, int(round((t_year - years[-1]) / cfg.election_cycle_years)))
        steps = np.diff(tau[:, min(vfit.t0, tau.shape[1] - 1):], axis=1)
        if cfg.volume_drift == "none" or steps.shape[1] == 0:
            drift = np.zeros(D)
        elif cfg.volume_drift == "last":
            drift = steps[:, -1]
        else:
            drift = steps.mean(axis=1)
        sd = np.maximum(steps.std(axis=1, ddof=1) if steps.shape[1] > 1 else 0.0, cfg.volume_step_floor)
        tau_t = tau[:, -1] + k * drift + np.sqrt(k) * sd * rng.standard_normal(D)
        lg = np.diff(np.log(totals))
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
    last_year = int(df["year"].max())
    last_n = df[df["year"] == last_year].set_index("booth")["n"].reindex(names)
    active = last_n.fillna(0.0).values > 0

    # Fixed total volume: overall volume V is pre-entered; each booth gets a probabilistic chunk K*V
    # where K ~ Dirichlet(alpha) and alpha is built from historical booth shares.
    if cfg.total_formal_votes is not None:
        total = float(cfg.total_formal_votes)
        shares = (last_n / last_n.sum()).fillna(0.0).values.astype(float)
        act_shares = shares[active]
        act_shares = act_shares / act_shares.sum()
        alpha = act_shares * 100.0                # Dirichlet concentration over active booths only
        K = rng.dirichlet(alpha, size=D)          # (D, n_active) probabilistic chunks summing to 1
        V = np.zeros((D, len(names)))
        V[:, active] = K * total
        return VolumeResult(V, "fixed_total", active, None)

    if projected is not None or cfg.registered_electors is not None:
        fixed = projected if projected is not None else projected_votes_from_electors(df, cfg)
        w = fixed.reindex(names).fillna(last_n).fillna(0.0).values.astype(float)
        return VolumeResult(np.tile(w, (D, 1)), "projected" if projected is not None else "electors", w > 0)

    vfit = fit_volume_model(df, cfg)
    if vfit.names != names:
        raise RuntimeError("Booth order differs between share and volume models.")
    return VolumeResult(simulate_volumes(vfit, D, cfg, rng, target_year=target_year, active=active), "modelled", active, vfit)


# ----------------------------------------------------------------------------- pipeline (S0-S8 registry + trace)
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
    dict(id="S8", name="OPV", pos=(12.2, 0.9), what="finalists + preference flows -> 2CP, exhaustion, P(win)", optional=True,
         eq="2CP_a = (v_a + sum_k f_ka v_k) / (v_a + v_b + sum_k (f_ka + f_kb) v_k)", inputs="V, p'_b, flow tables", outputs="2CP, exhausted, winner",
         trace=["s8_eligible", "s8_pair", "s8_2cp", "s8_exhausted", "s8_winner"]),
]
PIPELINE_EDGES = [("S0", "S1", "solid"), ("S1", "S3", "solid"), ("S2", "S3", "solid"), ("S3", "S4", "solid"),
                  ("S4", "S6", "solid"), ("S6", "S7", "solid"), ("S5", "S7", "solid"), ("S6", "S8", "solid"), ("S5", "S8", "solid"), ("S0", "S5", "dashed")]


@dataclass
class Stages:
    cfg: ModelConfig
    df: pd.DataFrame
    fit: ShareFit
    polls: PollSummary
    stw: StatewideDraws
    sim: ShareSim
    onp: OnpResult
    vol: VolumeResult
    keep: np.ndarray
    kept_names: list
    seat_all: np.ndarray
    last_n: pd.Series
    opv: Optional[OpvResult] = None


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
        "s1_prior_beta": np.array([list(pri.get(p, (0.0, 0.0))) for p in parties], dtype=float),
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
        "meta": {"out_parties": st.onp.out_parties, **{k: v for k, v in asdict(cfg).items()}},
    }
    if st.polls.table is not None:
        trace["s2_poll_table"] = st.polls.table
    if st.opv is not None:
        trace.update(opv_trace(st.opv))
    return trace


def check_trace(trace: Optional[dict]) -> List[str]:
    if not trace:
        return ["trace is empty: the process report has nothing to draw."]
    return [f"trace step {s['id']} ({s['name']}) is missing: {', '.join(k for k in s['trace'] if k not in trace)}"
            for s in PIPELINE_STEPS if any(k not in trace for k in s["trace"]) and not (s.get("optional") and not any(k in trace for k in s["trace"]))]


# ----------------------------------------------------------------------------- forecast (Forecast + run_forecast)
@dataclass
class Forecast:
    params: Dict[str, float]
    booths: pd.DataFrame
    tide: pd.DataFrame
    booth_forecast: pd.DataFrame
    seat: pd.DataFrame
    draws_df: pd.DataFrame
    booth_draws_df: pd.DataFrame
    booth_names: list
    volume_matrix: np.ndarray
    prepared_df: pd.DataFrame
    booth_party_draws: np.ndarray
    party_names: list
    diagnostics: Optional[pd.DataFrame] = None
    attribution: Optional[pd.DataFrame] = None
    trace: Optional[dict] = None
    backtest_result: Optional[object] = None
    opv: Optional[OpvResult] = None
    opv_error: Optional[str] = None


def uncertainty_attribution(st: Stages, k0: int) -> pd.DataFrame:
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
        a = dr["alpha"][:, :, j] - dr["alpha"][:, :, Km]
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


def run_forecast(df: pd.DataFrame, state_df: pd.DataFrame, poll: Union[PollInput, pd.DataFrame], cfg: ModelConfig,
                 projected_formal_votes: Optional[pd.Series] = None, target_year: Optional[int] = None,
                 run_attribution: bool = True, record_trace: bool = True, run_opv: bool = True) -> Forecast:
    parties = list(cfg.parties)
    k0 = parties.index(cfg.party)
    rng = np.random.default_rng(cfg.seed)

    fit = fit_share_model(df, cfg, state_df)
    D = fit.n_draws
    polls = summarise_polls(poll, cfg, state_df)
    stw = draw_statewide(polls, cfg, D, rng)
    sim = simulate_shares(fit, stw.S, cfg, rng)
    onp = apply_onp(sim.p, stw.onp, parties, k0, cfg, rng)
    if cfg.onp_mode == "brian":
        take = cfg.brian_transfer
        log.info("BrianMode ON: ONP takes %s of each party's vote.", take if take else "the default share")
    vol = resolve_volumes(df, fit.booth_names, cfg, D, rng, projected_formal_votes, target_year)

    keep = np.where(vol.active)[0]
    kept_names = [fit.booth_names[i] for i in keep]
    last_year = int(df["year"].max())
    last_n = df[df["year"] == last_year].set_index("booth")["n"].reindex(fit.booth_names)
    Vk, Pk = vol.V[:, keep], onp.p[:, keep, :]
    seat_all = 100.0 * np.einsum("db,dbk->dk", Vk, Pk) / Vk.sum(axis=1, keepdims=True)
    opv, opv_error = None, None
    if run_opv:
        try:
            opv = compute_opv(Vk, Pk * 100.0, onp.out_parties, kept_names, cfg, resolve_flows(cfg),
                              np.random.default_rng(cfg.seed + OPV_SEED_OFFSET))
        except Exception as e:
            opv_error = repr(e)
            log.warning("OPV step failed: %s", opv_error)
    st = Stages(cfg, df, fit, polls, stw, sim, onp, vol, keep, kept_names, seat_all, last_n, opv)

    return Forecast(
        params=_params(st, k0), booths=_booth_effects_table(st, k0), tide=_tide_table(st),
        booth_forecast=_booth_forecast_table(st, k0), seat=_seat_table(st, k0),
        draws_df=pd.DataFrame(seat_all, columns=onp.out_parties),
        booth_draws_df=pd.DataFrame(100.0 * sim.p[:, keep, k0] * onp.target_scale[:, None], columns=kept_names),
        booth_names=kept_names, volume_matrix=Vk, prepared_df=df, booth_party_draws=Pk * 100.0, party_names=onp.out_parties,
        diagnostics=fit.diag, attribution=uncertainty_attribution(st, k0) if run_attribution else None,
        trace=build_trace(st) if record_trace else None,
        backtest_result=None, opv=opv, opv_error=opv_error)


# ----------------------------------------------------------------------------- backtest
@dataclass
class BacktestSpec:
    state_mode: str = "auto"
    volume_mode: str = "actual"
    polls: Optional[pd.DataFrame] = None
    horizon_days: Optional[float] = None
    election_dates: Optional[Dict[int, object]] = None
    min_polls: int = 1

    def __post_init__(self):
        if self.state_mode not in BACKTEST_STATE_MODES:
            raise ValueError(f"state_mode must be one of {BACKTEST_STATE_MODES}")
        if self.volume_mode not in BACKTEST_VOLUME_MODES:
            raise ValueError(f"volume_mode must be one of {BACKTEST_VOLUME_MODES}")
        if self.state_mode == "polls" and self.polls is None:
            raise ValueError("state_mode='polls' needs a poll table (backtest_polls, or polls).")
        if self.election_dates is None:
            raise ValueError("BacktestSpec: election_dates is required (derive from state_results or pass explicitly).")
        self.election_dates = {int(y): pd.Timestamp(d) for y, d in self.election_dates.items()}


@dataclass
class BacktestResult:
    table: pd.DataFrame
    draws: Dict[int, dict]


def interval_coverage(draws: np.ndarray, actual: np.ndarray, levels: Sequence[int] = INTERVALS) -> Dict[str, float]:
    out = {}
    for lv in levels:
        a = (100.0 - lv) / 200.0
        lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)], axis=0)
        cov = float(np.mean((actual >= lo) & (actual <= hi)))
        out[f"cover_{lv}"], out[f"bias_{lv}"] = cov, cov - lv / 100.0
    return out


def crps_ensemble(draws: np.ndarray, actual: np.ndarray) -> np.ndarray:
    D = draws.shape[0]
    srt = np.sort(draws, axis=0)
    spread = 2.0 / D ** 2 * ((2 * np.arange(1, D + 1) - D - 1)[:, None] * srt).sum(axis=0)
    return np.abs(draws - actual[None, :]).mean(axis=0) - 0.5 * spread


def cycle_polls(spec: BacktestSpec, tgt: int):
    pl = spec.polls
    if pl is None:
        return None, None
    if {"election_year", "election_date"} <= set(pl.columns):
        sub = pl[pl["election_year"] == tgt]
        if sub.empty:
            return sub, None
        dates = pd.to_datetime(sub["election_date"]).unique()
        if len(dates) != 1:
            raise ValueError(f"the poll table must give ONE election_date per election_year (year {tgt} has {len(dates)}).")
        elec = pd.Timestamp(dates[0])
        start = pd.Timestamp.min
    else:
        if tgt not in spec.election_dates:
            raise ValueError(f"no election date for {tgt}: add it to the election_dates input (known: {sorted(spec.election_dates)}).")
        elec = spec.election_dates[tgt]
        earlier = [d for y, d in spec.election_dates.items() if y < tgt]
        start = max(earlier) if earlier else pd.Timestamp.min
        when = pd.to_datetime(pl["date"])
        end = elec - pd.Timedelta(days=spec.horizon_days) if spec.horizon_days is not None else elec
        sub = pl[(when > start) & (when <= end)]
    if spec.horizon_days is not None and "election_year" in pl.columns:
        sub = sub[pd.to_datetime(sub["date"]) <= elec - pd.Timedelta(days=spec.horizon_days)]
    return sub, elec


def _statewide_draws(spec: BacktestSpec, tgt: int, S_true: np.ndarray, k0: int, D: int, cfg: ModelConfig,
                     rng: np.random.Generator, state_df: pd.DataFrame):
    mode, note, pl, elec = spec.state_mode, None, None, None
    if mode in ("polls", "auto"):
        pl, elec = cycle_polls(spec, tgt)
        if pl is None or len(pl) < spec.min_polls:
            if mode == "polls":
                raise ValueError(f"No valid pre-election polls found for election cycle ending {tgt} (required under state_mode='polls').")
            else:
                log.warning(f"[Backtest] Target {tgt}: {len(pl) if pl is not None else 0} valid poll(s) found (< {spec.min_polls}). Falling back to 'oracle' ground truth.")
                mode = "oracle"
                note = "no poll table" if pl is None else f"only {len(pl)} poll(s) in cycle ending {elec.date() if elec is not None else tgt}"
        else:
            mode = "polls"
    if mode == "oracle":
        log.info(f"[Backtest] Target {tgt}: using ORACLE state ground truth.")
        return np.tile(S_true, (D, 1)), {"state_mode": "oracle", "n_polls": 0, **({"state_note": note} if note else {})}

    total = float(S_true.sum())
    info = {"state_mode": mode}
    if mode == "noised":
        e0 = pd.Timestamp("2000-01-01")
        horizon = DEFAULT_NOISED_HORIZON_DAYS if spec.horizon_days is None else spec.horizon_days
        cfg_t = replace_cfg(cfg, election_date=str(e0.date()), prediction_date=str((e0 - pd.Timedelta(days=horizon)).date()))
        tp = PollInput(float(S_true[k0]), float(S_true[k0]), float(S_true[k0]))
        summ = PollSummary(100.0 * S_true / total, None, tp, None)
    else:
        asof = None if spec.horizon_days is None else elec - pd.Timedelta(days=spec.horizon_days)
        pred = asof if asof is not None else pd.to_datetime(pl["date"]).max()
        cfg_t = replace_cfg(cfg, election_date=str(elec.date()), prediction_date=str(pred.date()),
                            poll_as_of=str(asof.date()) if asof is not None else None)
        try:
            summ = summarise_polls(pl, cfg_t, state_df)
        except ValueError as e:
            if spec.state_mode == "auto":
                log.warning(f"[Backtest] Target {tgt}: poll summary failed ({e}). Falling back to oracle.")
                return np.tile(S_true, (D, 1)), {"state_mode": "oracle", "n_polls": 0, "state_note": str(e)[:120]}
            raise ValueError(f"backtest polls for {tgt}: {e}") from e
        info["n_polls"] = int(len(pl))
    stw = draw_statewide(summ, cfg_t, D, rng)
    info["state_se_used"] = float(stw.se_parts[3])
    if mode == "polls":
        log.info(f"[Backtest] Target {tgt}: used {info['n_polls']} poll(s). State error: {stw.central[k0] - S_true[k0]:+.2f} pts.")
        info.update(poll_central=float(stw.central[k0]), state_actual=float(S_true[k0]),
                    state_error=float(stw.central[k0] - S_true[k0]),
                    state_z=float((S_true[k0] - stw.central[k0]) / stw.se_parts[3]))
    return stw.S * (total / 100.0), info


def replace_cfg(cfg: ModelConfig, **kw) -> ModelConfig:
    f = {k: getattr(cfg, k) for k in ("seat_name", "party", "parties", "election_date", "prediction_date", "scenario_pct",
                                      "exclude_booths", "booth_merge", "early_keywords", "data_check_jump_threshold",
                                      "beta_preset", "beta_priors", "sigma_beta_sd", "m_prior_sd", "sigma_alpha_sd",
                                      "sigma_eps_sd", "sigma_d_median", "sigma_d_sdlog", "tide_process", "d_persist",
                                      "rho_beta_a", "rho_beta_b", "use_correlated_errors", "lkj_eta", "include_eps",
                                      "draws", "tune", "chains", "seed", "target_accept", "rhat_threshold", "ess_threshold",
                                      "poll_half_life_months", "poll_as_of", "poll_moe", "poll_house_sd",
                                      "time_horizon_gamma", "poll_z", "poll_default_sample_size", "poll_dirichlet_conc_floor",
                                      "poll_sum_warn", "share_clip_floor", "registered_electors", "expected_turnout_rate",
                                      "informal_rate_fallback", "election_cycle_years", "volume_draws", "volume_tune_floor",
                                      "volume_target_accept", "volume_drift", "volume_step_floor", "volume_total_sd_floor",
                                      "vol_sigma_a_sd", "vol_tau_sd", "vol_kappa_alpha", "vol_kappa_beta",
                                      "onp_mode", "onp_state_denominator", "brian_transfer", "onp_seat_ratio_median",
                                      "onp_seat_ratio_sdlog", "onp_max_seat_share", "opv_flows", "opv_flow_conc",
                                      "opv_finalist_exclude", "opv_min_pair_prob", "opv_affinity_sd", "opv_entrants",
                                      "opv_proxies")}
    f.update(kw)
    return ModelConfig(**f)


def _baselines(row: dict, df_all: pd.DataFrame, S_all: pd.DataFrame, cfg: ModelConfig, tgt: int, last_y: int,
               t: pd.DataFrame, act: np.ndarray, n_t: np.ndarray, med: np.ndarray, crps: np.ndarray) -> None:
    last = df_all[df_all["year"] == last_y].set_index("booth")["share"]
    mk = np.array([b in last.index for b in t["booth"]])
    row["common_n"] = int(mk.sum())
    if not mk.any():
        return
    prev = last.reindex(t["booth"].values[mk]).values
    a, n, m = act[mk], n_t[mk], med[mk]
    swing = float(S_all.loc[tgt, cfg.party] - S_all.loc[last_y, cfg.party])
    errs = {"model": m - a, "nochange": prev - a, "statewide_swing": prev + swing - a}
    wavg = lambda v: float((v * n).sum() / n.sum())
    row.update({"model_MAE_common": float(np.abs(errs["model"]).mean()), "nochange_MAE": float(np.abs(errs["nochange"]).mean()),
                "statewide_swing_MAE": float(np.abs(errs["statewide_swing"]).mean()), "model_CRPS_common": float(crps[mk].mean()),
                "model_seat_err_common": wavg(errs["model"]), "nochange_seat_err": wavg(errs["nochange"]),
                "statewide_swing_seat_err": wavg(errs["statewide_swing"])})


def backtest_shares(df_all: pd.DataFrame, state_df: pd.DataFrame, cfg: ModelConfig,
                    targets: Optional[Sequence[int]] = None, return_draws: bool = False,
                    spec: Optional[BacktestSpec] = None) -> BacktestResult:
    spec = spec or BacktestSpec()
    years = sorted(df_all["year"].unique())
    parties = list(cfg.parties)
    k0 = parties.index(cfg.party)
    S_all = state_matrix(state_df, parties, years, cfg.onp_state_denominator)
    lo_p, _, hi_p = (float(x) for x in cfg.scenario_pct)

    log.info(f"[Backtest] Starting multi-party walk-forward validation across targets: {targets if targets is not None else years[2:]}")

    rows, draws_out = [], {}
    for tgt in (targets if targets is not None else years[2:]):
        tgt = int(tgt)
        log.info(f"[Backtest] === Processing Target Election Year: {tgt} ===")
        rng = np.random.default_rng(cfg.seed + tgt)
        train = df_all[df_all["year"] < tgt]

        log.info(f"[Backtest] Fitting share model on historical training data (< {tgt}), train rows: {len(train)}")
        fit = fit_share_model(train, cfg, state_df)
        log.info(f"[Backtest] Fit complete. Divergences: {fit.divergences}, Max R^hat: {float(fit.diag['r_hat'].max()):.3f}")

        D = fit.n_draws
        S_draws, state_info = _statewide_draws(spec, tgt, S_all.loc[tgt].values, k0, D, cfg, rng, state_df)
        p = simulate_shares(fit, S_draws, cfg, rng).p

        t = df_all[(df_all["year"] == tgt) & df_all["booth"].isin(fit.booth_names)]
        col = [fit.booth_names.index(b) for b in t["booth"]]
        n_t, y_t = t["n"].values, t["y"].values
        if return_draws:
            draws_out[tgt] = {"booths": [str(b) for b in t["booth"]], "parties": parties, "booth_party_draws": 100.0 * p[:, col, :],
                              "actual_party_votes": np.column_stack([t[f"y_{q}"].values for q in parties]), "actual_n": n_t.astype(float)}

        P = 100.0 * p[:, col, k0]
        med, act = np.median(P, axis=0), 100.0 * y_t / n_t
        rep = 100.0 * rng.binomial(n_t[None, :], p[:, col, k0]) / n_t[None, :]
        calib = interval_coverage(rep, act)
        crps = crps_ensemble(rep, act)

        multi_party_maes = []
        for j, q in enumerate(parties):
            actual_q_pct = 100.0 * t[f"y_{q}"].values / n_t
            pred_q_pct = np.median(100.0 * p[:, col, j], axis=0)
            multi_party_maes.append(np.abs(pred_q_pct - actual_q_pct).mean())
        multi_party_mae_val = float(np.mean(multi_party_maes))

        if spec.volume_mode == "modelled":
            vfit = fit_volume_model(train, cfg)
            if vfit.names != fit.booth_names:
                raise RuntimeError("Booth order differs between share and volume models.")
            prev = train[train["year"] == int(train["year"].max())].set_index("booth")["n"]
            Vc = simulate_volumes(vfit, D, cfg, rng, target_year=tgt, active=np.array([b in prev.index for b in vfit.names]))[:, col]
            seat = (P * Vc).sum(axis=1) / np.maximum(Vc.sum(axis=1), 1.0)
        else:
            seat = (P * n_t).sum(axis=1) / n_t.sum()
        seat_act = float((act * n_t).sum() / n_t.sum())
        s_lo, s50, s_hi = np.percentile(seat, [lo_p, 50, hi_p])
        err = med - act

        log.info(f"[Backtest] Target {tgt} Result: Pred Median = {s50:.2f}%, Actual = {seat_act:.2f}%, Seat Error = {s50 - seat_act:+.2f} pts, Multi-Party MAE = {multi_party_mae_val:.2f}%")

        row = {"target": tgt, "seat_pred_median": s50, "seat_actual": seat_act, "seat_error": s50 - seat_act,
               "seat_lo": s_lo, "seat_hi": s_hi, "seat_width": s_hi - s_lo, "seat_in_interval": bool(s_lo <= seat_act <= s_hi),
               "state_mode": spec.state_mode, "volume_mode": spec.volume_mode, **state_info,
               "booth_MAE": float(np.abs(err).mean()), "booth_RMSE": float(np.sqrt((err ** 2).mean())), "booth_CRPS": float(crps.mean()),
               "multi_party_MAE": multi_party_mae_val,
               **{f"booth_cover_{lv}": calib[f"cover_{lv}"] for lv in INTERVALS}, "calib_bias_90": calib["bias_90"],
               "n_booths": len(t), "divergences": fit.divergences, "rhat_max": float(fit.diag['r_hat'].max())}
        for j, q in enumerate(parties):
            sj = (100.0 * p[:, col, j] * n_t).sum(axis=1) / n_t.sum()
            row[f"seat_err_{q}"] = float(np.median(sj) - 100.0 * t[f"y_{q}"].sum() / n_t.sum())
        _baselines(row, df_all, S_all, cfg, tgt, max(int(y) for y in years if y < tgt), t, act, n_t, med, crps)
        rows.append(row)
        del fit, p, P, rep
        gc.collect()

    return BacktestResult(pd.DataFrame(rows), draws_out)


def backtest_volumes(df: pd.DataFrame, cfg: ModelConfig, sims: int = 4000,
                     targets: Optional[Sequence[int]] = None) -> pd.DataFrame:
    years = sorted(df["year"].unique())
    rows = []
    for tgt in (targets if targets is not None else years[2:]):
        tgt = int(tgt)
        rng = np.random.default_rng(cfg.seed + tgt)
        train = df[df["year"] < tgt]
        vfit = fit_volume_model(train, cfg)
        prev_all = train[train["year"] == int(train["year"].max())].set_index("booth")["n"]
        active = np.array([b in prev_all.index for b in vfit.names])
        V = simulate_volumes(vfit, sims, cfg, rng, target_year=tgt, active=active)

        year_df = df[df["year"] == tgt]
        t = year_df.set_index("booth")["n"]
        idx = [i for i, b in enumerate(vfit.names) if active[i] and b in t.index]
        names = [vfit.names[i] for i in idx]
        act = t.reindex(names).values.astype(float)
        P = V[:, idx]
        med = np.median(P, axis=0)
        lo, hi = np.percentile(P, [5, 95], axis=0)
        prev = prev_all.reindex(names).values.astype(float)
        tot, total_act = V.sum(axis=1), float(year_df["n"].sum())
        early = vfit.btype.astype(bool)
        has_early = bool(early.any())
        rows.append({
            "target": tgt, "n_booths": len(names),
            "booth_MAPE_pct": float(np.mean(np.abs(med - act) / act) * 100), "nochange_MAPE_pct": float(np.mean(np.abs(prev - act) / act) * 100),
            "booth_cover_90": float(((act >= lo) & (act <= hi)).mean()),
            "total_pred_median": float(np.median(tot)), "total_actual": total_act,
            "total_error_pct": float((np.median(tot) / total_act - 1) * 100),
            "total_in_90": bool(np.percentile(tot, 5) <= total_act <= np.percentile(tot, 95)),
            "early_share_pred": float(np.median(V[:, early].sum(axis=1) / np.maximum(tot, 1))) if has_early else np.nan,
            "early_share_actual": float(year_df[year_df["btype"] == 1]["n"].sum() / total_act) if has_early else np.nan,
            "divergences": vfit.divergences})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- input resolution + blackbox
_DATA_KEYS = ("booth_results", "polls", "state_results", "projected_formal_votes",
              "target_year", "backtest_polls", "election_dates")
_TUPLE_FIELDS = ("parties", "opv_finalist_exclude", "opv_entrants", "pdf_tier_cuts",
                 "early_keywords", "exclude_booths", "scenario_pct", "plot_ridgeline_keywords")


@dataclass
class ENKIOutput:
    forecast: Optional[Forecast] = None
    seat_summary: Optional[pd.DataFrame] = None
    opv: Optional[Any] = None
    backtest: Optional[pd.DataFrame] = None
    backtest_draws: Optional[dict] = None
    volume_backtest: Optional[pd.DataFrame] = None
    config: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    data_sources: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    output_dir: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)
    trace: Optional[dict] = None


@dataclass
class Resolved:
    data: Dict[str, Any]
    cfg: ModelConfig
    run: RunOptions
    warnings: List[str]
    sources: Dict[str, str]


def _strip_comments(text: str) -> str:
    """Remove // comments from JSONC text, respecting string boundaries."""
    out = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        else:
            if c == '"':
                in_str = True
                out.append(c)
                i += 1
            elif c == "/" and i + 1 < n and text[i + 1] == "/":
                while i < n and text[i] != "\n":
                    i += 1
            else:
                out.append(c)
                i += 1
    return "".join(out)


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    return str(o)


def load_x(x) -> Dict[str, Any]:
    """dict | path to .json/.jsonc -> one flat dict. Groups ('data', 'config', 'run') are merged; a key may not appear twice."""
    if isinstance(x, (str, Path)):
        x = json.loads(_strip_comments(Path(x).read_text(encoding="utf-8")))
    if not isinstance(x, dict):
        raise TypeError("ENKI(x): x must be a dict or a path to a .json file.")
    flat: Dict[str, Any] = {}
    for grp in ("data", "config", "run"):
        for k, v in dict(x.get(grp) or {}).items():
            if k in flat:
                raise ValueError(f"Key {k!r} appears in more than one group of x.")
            flat[k] = v
    for k, v in x.items():
        if k in ("data", "config", "run"):
            continue
        if k in flat:
            raise ValueError(f"Key {k!r} appears both at the top level and inside a group of x.")
        flat[k] = v
    return flat


def configure_logging(verbose: bool) -> None:
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("-> %(message)s"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(logging.INFO if verbose else logging.WARNING)


def _source(v) -> str:
    return str(v) if isinstance(v, (str, Path)) else "<in-memory>"


def resolve(x) -> Resolved:
    """x -> typed config, run options, data tables, warnings. Every input problem surfaces here, before any sampling."""
    flat = load_x(x)
    cfg_names, run_names = {f.name for f in fields(ModelConfig)}, {f.name for f in fields(RunOptions)}
    known = cfg_names | run_names | set(_DATA_KEYS)
    unknown = sorted(set(flat) - known)
    if unknown:
        hints = []
        for k in unknown:
            if k in RENAMED:
                hints.append(f"{k!r} was replaced by {RENAMED[k]}")
            else:
                m = difflib.get_close_matches(k, sorted(known), n=1)
                hints.append(f"{k!r}" + (f" (did you mean {m[0]!r}?)" if m else ""))
        raise KeyError("ENKI(x): unknown input key(s): " + "; ".join(hints))

    cfg_kw = {k: v for k, v in flat.items() if k in cfg_names}
    run_kw = {k: v for k, v in flat.items() if k in run_names}
    for kw in (cfg_kw, run_kw):
        for k in _TUPLE_FIELDS:
            if kw.get(k) is not None:
                kw[k] = tuple(kw[k])
    if cfg_kw.get("beta_priors"):
        cfg_kw["beta_priors"] = {p: tuple(v) for p, v in cfg_kw["beta_priors"].items()}
    if cfg_kw.get("beta_presets"):
        cfg_kw["beta_presets"] = {
            preset: {p: tuple(v) for p, v in mp.items()} for preset, mp in cfg_kw["beta_presets"].items()
        }
    if cfg_kw.get("prediction_date") is None:
        cfg_kw["prediction_date"] = date.today().strftime("%Y-%m-%d")
    cfg, run = ModelConfig(**cfg_kw), RunOptions(**run_kw)
    warnings_ = validate(cfg, run)
    if run.run_forecast and run.run_opv:
        warnings_ += check_flows(resolve_flows(cfg), cfg)

    if flat.get("booth_results") is None:
        raise KeyError("ENKI(x): 'booth_results' is required (DataFrame or CSV path).")
    data: Dict[str, Any] = {"booth_results": read_frame(flat["booth_results"], "booth_results")}
    src = {"booth_results": _source(flat["booth_results"])}

    if run.run_forecast:
        pl = flat.get("polls")
        if pl is None:
            raise KeyError("ENKI(x): 'polls' is required to forecast (DataFrame, CSV path, or {'central','low','high'}).")
        if isinstance(pl, PollInput):
            data["polls"] = pl
        elif isinstance(pl, dict) and set(pl) == {"central", "low", "high"}:
            data["polls"] = PollInput(float(pl["central"]), float(pl["low"]), float(pl["high"]))
        else:
            data["polls"] = read_frame(pl, "polls")
        src["polls"] = _source(pl)

    if flat.get("state_results") is None:
        sr = _load_default("sample_state_results.csv")
        if sr is None:
            raise KeyError("ENKI(x): 'state_results' is required (DataFrame, CSV path, or sample_state_results.csv).")
        data["state_results"], src["state_results"] = sr, "<sample_state_results.csv>"
    else:
        data["state_results"] = read_frame(flat["state_results"], "state_results")
        src["state_results"] = _source(flat["state_results"])
    absent = sorted(({"year"} | set(cfg.parties)) - set(data["state_results"].columns))
    if absent:
        raise ValueError(f"state_results is missing columns: {absent}")

    pfv = flat.get("projected_formal_votes")
    if pfv is not None:
        data["projected_formal_votes"] = pfv if isinstance(pfv, pd.Series) else pd.Series(pfv, dtype=float)
        if cfg.registered_electors is not None:
            warnings_.append("projected_formal_votes AND registered_electors are both set: projected_formal_votes wins.")
    data["target_year"] = flat.get("target_year")
    if flat.get("election_dates") is not None:
        data["election_dates"] = dict(flat["election_dates"])
    return Resolved(data, cfg, run, warnings_, src)


def _seat_summary(res: Forecast, cfg: ModelConfig) -> pd.DataFrame:
    lo, mid, hi = (float(v) for v in cfg.scenario_pct)
    q = res.draws_df.quantile([lo / 100, mid / 100, hi / 100]).T
    q.columns = [f"p{lo:g}", f"p{mid:g}", f"p{hi:g}"]
    q.insert(0, "mean", res.draws_df.mean())
    return q


def ENKI(x) -> ENKIOutput:
    """The blackbox. ENKI(x) -> y. Computes only; returns in-memory tables and draws."""
    r = resolve(x)
    cfg, run, data = r.cfg, r.run, r.data
    configure_logging(run.verbose)
    y = ENKIOutput(config=json.loads(json.dumps(asdict(cfg), default=lambda o: str(o))),
                   run=json.loads(json.dumps(asdict(run), default=lambda o: str(o))),
                   data_sources=r.sources, warnings=r.warnings)

    df = prepare_booths(data["booth_results"], cfg)
    state_df = data["state_results"]

    if run.run_backtest:
        log.info("RUNNING HISTORICAL BACKTEST")
        election_dates = data.get("election_dates")
        if election_dates is None and "election_date" in state_df.columns:
            election_dates = {int(y): d for y, d in zip(state_df["year"], state_df["election_date"])}
        spec = BacktestSpec(run.backtest_state_mode, run.backtest_volume_mode, data.get("backtest_polls"),
                            run.backtest_poll_horizon_days, election_dates, run.backtest_min_polls)
        bt = backtest_shares(df, state_df, cfg, run.backtest_targets, return_draws=True, spec=spec)
        y.backtest, y.backtest_draws = bt.table, bt.draws
        fell = bt.table.loc[bt.table["state_mode"] == "oracle", "target"].tolist() if run.backtest_state_mode == "auto" else []
        if fell:
            y.warnings.append(f"Backtest used the TRUE statewide result (no poll error tested) for {fell}: no polls fall in those election "
                              f"cycles. Elections backtested with polls: {sorted(set(bt.table['target']) - set(fell))}.")
    if run.run_volume_backtest:
        y.volume_backtest = backtest_volumes(df, cfg, run.backtest_volume_sims, run.backtest_targets)

    if run.run_forecast:
        log.info("RUNNING FORECAST: %s (target %s)", cfg.seat_name, cfg.election_date)
        res = run_forecast(df, state_df, data["polls"], cfg, data.get("projected_formal_votes"), data.get("target_year"),
                           run.run_attribution, run.record_trace, run.run_opv)
        y.forecast, y.trace = res, res.trace
        res.backtest_result = y.backtest
        y.opv = res.opv
        if res.opv_error:
            y.warnings.append(f"OPV step failed (forecast unaffected): {res.opv_error}")
        if res.opv is not None:
            y.warnings += res.opv.warnings
        if res.trace is not None:
            y.warnings += check_trace(res.trace)
        y.seat_summary = _seat_summary(res, cfg)
    return y


def main(x: Optional[dict] = None, config_path: str = "config.jsonc") -> ENKIOutput:
    if x is None:
        x = load_x(config_path)
    y = ENKI(x)
    output_root = x.get("output_root")
    if output_root:
        from .helpers import write_all
        write_all(y, output_root)
    return y


def cli(argv: Sequence[str]) -> ENKIOutput:
    """python -m enki [x.json] [key=value ...]   (values are parsed as JSON where possible)"""
    path, over = None, {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            try:
                over[k] = json.loads(v)
            except json.JSONDecodeError:
                over[k] = v
        else:
            path = a
    if path is None:
        base = load_x("config.jsonc")
        base.update(over)
        return ENKI(base)
    x = load_x(path)
    x.update(over)
    return ENKI(x)

