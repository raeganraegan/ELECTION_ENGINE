"""config.py -- every setting ENKI understands, plus validation. No modelling logic lives here.

`ENKI(x)` takes ONE flat dict. Each key is either a data key (booth_results, polls, ...), a ModelConfig field
(changes the numbers) or a RunOptions field (changes what is run / written, never the numbers).
Unknown keys are an error, so a typo can never silently fall back to a default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# The five uncertainty layers that can be switched off one at a time (uncertainty attribution, simulate_shares).
LAYERS = ("poll", "params", "tide", "booth", "volumes")
SCENARIOS = ("worst", "average", "best")      # = scenario_pct percentiles (default 5th / 50th / 95th)
ONP_MODES = ("none", "brian", "entry")
TIDE_PROCESSES = ("ar1", "random_walk")
VOLUME_DRIFTS = ("last", "mean", "none")

# Prior (mean, sd) of the party-level sensitivity mu_beta to its own statewide log-share swing.
# "default" applies to every modelled non-reference party that is not named. The reference party has no beta (fixed 0).
BETA_PRESETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "data":    {"default": (0.3, 0.30), "GRN": (0.3, 0.35)},
    "uniform": {"default": (1.0, 0.15)},
}


@dataclass
class PollInput:
    """A single statewide poll number for the target party, for callers with no poll table."""
    central: float
    low: float
    high: float


@dataclass
class ModelConfig:
    # ---- what is being forecast
    seat_name: str = "Seat"
    party: str = "ALP"                                         # party whose booth forecasts / tables are reported
    parties: Sequence[str] = ("ALP", "LNP", "GRN", "OTHER")    # modelled jointly; the LAST one is the reference category
    election_date: str = "2027-03-13"
    prediction_date: Optional[str] = None                      # None -> today
    scenario_pct: Sequence[float] = (5, 50, 95)                # worst / average / best percentiles

    # ---- booth data cleaning
    exclude_booths: Sequence[str] = ("NONE",)
    booth_merge: Optional[Dict[str, str]] = None               # {"old venue": "new venue"} merged BEFORE fitting
    early_keywords: Sequence[str] = ("PRE-POLL", "PREPOLL", "PRE POLL", "EARLY", "POSTAL", "DECLARATION",
                                     "ABSENT", "IVOTE", "SPECIAL", "MOBILE")
    data_check_jump_threshold: float = 0.15                    # warn when the seat's formal total jumps by more than this

    # ---- share-model priors
    beta_preset: str = "data"                                  # key of BETA_PRESETS
    beta_priors: Optional[Dict[str, Tuple[float, float]]] = None   # per-party (mean, sd) overrides on top of the preset
    sigma_beta_sd: float = 0.20                                # HalfNormal scale, spread of booth sensitivities around mu_beta
    m_prior_sd: float = 1.50                                   # sd of the seat-level party intercepts around the empirical log-ratio
    sigma_alpha_sd: float = 0.25                               # HalfNormal scale, booth baseline spread
    sigma_eps_sd: Optional[float] = None                       # HalfNormal scale, booth-election noise. None = auto from history
    sigma_d_median: Optional[float] = None                     # LogNormal median, seat tide sd. None = auto from history
    sigma_d_sdlog: float = 0.7
    tide_process: str = "ar1"                                  # "ar1" or "random_walk"
    d_persist: Optional[float] = None                          # None: rho ~ Beta(a, b) is estimated; a number fixes it
    rho_beta_a: float = 2.0
    rho_beta_b: float = 2.0
    use_correlated_errors: bool = False                        # LKJ correlation between parties' tide and booth shocks
    lkj_eta: float = 2.0
    include_eps: bool = True                                   # booth-election noise in predictive draws

    # ---- MCMC
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    seed: int = 42
    target_accept: float = 0.99
    rhat_threshold: float = 1.01
    ess_threshold: float = 400.0

    # ---- statewide polls
    poll_half_life_months: float = 2.0                         # a poll loses half its weight every this many months
    poll_as_of: Optional[str] = None                           # ignore polls after this date (None: use all)
    poll_moe: Optional[float] = None                           # override the computed poll margin of error (points)
    poll_house_sd: float = 1.50                                # extra poll error in points
    time_horizon_gamma: float = 0.8                            # extra poll sd (points) = gamma * sqrt(years to election)
    poll_z: float = 1.96
    poll_default_sample_size: float = 1000.0                   # used when no poll has a sample_size
    poll_dirichlet_conc_floor: float = 10.0
    poll_sum_warn: float = 102.0                               # warn when poll shares incl. ONP sum to more than this
    share_clip_floor: float = 1e-3                             # statewide shares are clipped to at least this (%)

    # ---- volumes (formal votes per booth). Priority: projected_formal_votes > registered_electors > fitted model
    registered_electors: Optional[float] = None
    expected_turnout_rate: float = 0.90
    informal_rate_fallback: float = 0.04                       # used only when the booth file has no votes to infer a rate from
    election_cycle_years: float = 4.0
    volume_draws: Optional[int] = None                         # None: same as draws
    volume_tune_floor: int = 3000
    volume_target_accept: float = 0.95
    volume_drift: str = "last"                                 # early-vote log-odds step: "last", "mean" or "none"
    volume_step_floor: float = 0.25                            # min sd of the next early-vote log-odds step
    volume_total_sd_floor: float = 0.03                        # min sd of the log growth of the seat's TOTAL formal votes
    vol_sigma_a_sd: float = 1.5
    vol_tau_sd: float = 0.5
    vol_kappa_alpha: float = 2.0
    vol_kappa_beta: float = 0.005

    # ---- One Nation (ONP). It cannot be inferred from a seat's history, so it enters by an explicit rule.
    onp_mode: str = "none"                                     # "none" | "brian" (fixed vote transfer) | "entry" (statewide-ONP-scaled entrant)
    onp_state_denominator: bool = False                        # keep ONP in the statewide denominator (needs an ONP poll column)
    brian_transfer: Optional[Dict[str, float]] = None          # onp_mode="brian": share of each party's vote ONP takes. None -> BRIAN_DEFAULT
    onp_seat_ratio_median: float = 0.65                        # onp_mode="entry": seat ONP share / statewide ONP share
    onp_seat_ratio_sdlog: float = 0.5
    onp_max_seat_share: float = 1.0


BRIAN_DEFAULT: Dict[str, float] = {"LNP": 0.20, "ALP": 0.05}


@dataclass
class RunOptions:
    run_forecast: bool = True
    run_backtest: bool = True                                  # refits once per past election: slow
    backtest_targets: Optional[Sequence[int]] = None           # None: every election with >= 2 earlier ones
    run_volume_backtest: bool = False
    backtest_volume_sims: int = 4000
    run_attribution: bool = True                               # which uncertainty layer drives the seat band
    output_root: Optional[str] = None
    make_plots: bool = True
    print_report: bool = True
    record_trace: bool = True
    verbose: bool = True
    plot_top_n_booths: int = 15
    plot_dpi: int = 300
    plot_volume_log_scale: bool = True
    plot_label_all_booths: bool = True
    plot_ridgeline_max_draws: int = 1500
    plot_ridgeline_keywords: Sequence[str] = ("PRE-POLL", "PREPOLL", "DECLARATION", "ABSENT", "POSTAL")


# Keys that used to exist, and what replaced them. Used for a precise error rather than a vague "unknown key".
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


def beta_prior_table(cfg: ModelConfig) -> Dict[str, Tuple[float, float]]:
    """party -> (mean, sd) of mu_beta, for every modelled party EXCEPT the reference (which has no beta)."""
    preset = BETA_PRESETS[cfg.beta_preset]
    out = {p: preset.get(p, preset["default"]) for p in list(cfg.parties)[:-1]}
    for p, ms in (cfg.beta_priors or {}).items():
        out[p] = (float(ms[0]), float(ms[1]))
    return out


def validate(cfg: ModelConfig, run: RunOptions) -> List[str]:
    """Hard errors are raised together; soft problems are returned as warnings."""
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
    if cfg.beta_preset not in BETA_PRESETS:
        err.append(f"beta_preset must be one of {sorted(BETA_PRESETS)}.")
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
            err.append(f"beta_priors[{p!r}] must be (mean, sd) with sd > 0 (the old third 'lower' element no longer exists).")

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

    if run.output_root is None and run.make_plots:
        warn.append("make_plots is on but output_root is None, so no plots are drawn.")
    if err:
        raise ValueError("ENKI(x): invalid input:\n  - " + "\n  - ".join(err))
    return warn
