'''ENKI_V3.py -- the model (fit, polls, tide, booths, volumes, ONP, seat) and the ENKI(x) -> y blackbox.
Helpers (plots, IO, validation, trace files) live in ENKI_HELPERS.py; backtests live in ENKI_BACKTEST.py.

CONSTRUCTS REALITY, AND THE BASICS. THE WINNING/LOSING AND STRATEGY COME LATER.
ENKI IS THE GOD OF WISDOM WHO CREATES THE FUNDAMENTAL LAWS OF THE UNIVERSE, THE BASELINE REALITY.
IN THIS CONTEXT, THIS SCRIPT CREATES THE BASELINE OF REALITY FOR THIS ENTIRE MODEL'''
from __future__ import annotations
import difflib
import gc
import json
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from ENKI_HELPERS import (append_history, as_frame, check_booth_data, check_trace, combine_parties, diagnose_sampler,
                        generate_booth_ridgeline_plot, generate_booth_swing_scatter_plot, generate_booth_volume_ridgeline,
                        generate_seat_distribution_plot, jsonable, load_x, output_folder, print_run_report,
                        save_actual_primaries, save_run_meta, save_trace, validate_inputs)

from ENKI_BACKTEST import backtest_bayes, backtest_volume, save_backtest_draws, save_backtest_table_png

#Just making sure things are fine boss.
try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

SCENARIOS = ("worst", "average", "best")        # = scenario_pct percentiles (default 5th / 50th / 95th)

# NSW statewide primary vote (%), by election. Hard-coded: these results do not change.
# 2023 OTHER was 15.75, which left the row summing to 97.79; it is set to the remainder (17.96) so every
# row sums to ~100 (the model uses log-ratios between parties, so a wrong OTHER distorts every party).
# Verify against NSWEC. Rows are also re-normalised to 100 inside the model.
STATE_RESULTS = pd.DataFrame({
    "year":  [1988,  1991,  1995,  1999,  2003,  2007,  2011,  2015,  2019,  2023],
    "ALP":   [38.48, 39.11, 41.27, 42.21, 42.68, 38.87, 25.55, 34.08, 33.31, 36.97],
    "LNP":   [49.54, 41.65, 39.81, 33.69, 34.35, 36.88, 51.15, 45.63, 41.58, 35.37],
    "GRN":   [ 1.50,  2.05,  2.85,  2.80,  4.10,  7.60, 10.28, 10.29,  9.57,  9.70],
    "ONP":   [ 0.00,  0.00,  0.00,  6.40,  0.70,  0.00,  0.00,  0.00,  1.10,  1.80],
    "OTHER": [10.48, 17.19, 16.07, 14.90, 18.17, 16.65, 12.65, 10.00, 14.90, 16.60],
})


# ----------------------------------------------------------------------------- inputs
@dataclass
class ModelConfig:
    #NOTE Meta data & model basics.
    party: str = "ALP"                            # party whose booth forecasts / tables are reported
    parties: Sequence[str] = ("ALP", "LNP", "GRN", "OTHER")   # modelled jointly; LAST = reference category
    beta_preset: str = "data"                     # "data": ALP/LNP/GRN ~ N(0.3, 0.3) (GRN sd 0.5), OTHER ~ N(0, 0.3); "uniform": N(1, 0.15)
    beta_g: Optional[float] = None                # override the prior mean (non-OTHER parties)
    beta_prior_sd: Optional[float] = None         # override the prior sd (non-OTHER parties)
    beta_priors: Optional[Dict[str, tuple]] = None   # full control: {"ALP": (mu, sd, lower), ..., "OTHER": (0, 0.3, None)}
    volume_total_sd_floor: float = 0.03           # min sd of the log growth of the TOTAL formal votes
    early_keywords: Sequence[str] = ("PRE-POLL", "PREPOLL", "PRE POLL", "EARLY", "POSTAL", "DECLARATION",
                                    "ABSENT", "IVOTE", "SPECIAL", "MOBILE")   # booths treated as early-voting
    sigma_alpha_sd: float = 0.25                   # HalfNormal scale, booth baseline spread (data: ~0.2-0.36 by party)
    verbose: bool = True                          # print design / collinearity / empirical checks
    volume_target_accept: float = 0.95
    exclude_booths: Sequence[str] = ("NONE",)
    booth_merge: Optional[Dict[str, str]] = None  # {"old venue": "new venue"}: merged BEFORE fitting
    poll_half_life_months: float = 2.0            # Weighting of polls as they get older (after two months, a poll loses half its weight)
    poll_as_of: Optional[str] = None              # Debug term, do not touch.
    poll_moe: Optional[float] = None              # Debug term, do not touch.
    seat_name: str = "Seat"                       # Where is this election happening, boss?
    prediction_date: Optional[str] = None         # When the prediction is made, defaults to today (whenver that is)
    election_date: str = "2027-03-13"             # Hard target election date

    #NOTE: MCMC settings (these genuinely change the answer, and are the crux of the monte carlo sim)
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    seed: int = 42
    target_accept: float = 0.99
    poll_house_sd: float = 1.50                    # extra poll error in points (2023 poll miss: 0.84)
    d_persist: Optional[float] = None             # None (default): rho_d ~ Beta(2,2) is ESTIMATED inside the fit (AR(1) tide, shared by all parties)
    include_eps: bool = True                      # booth-level local-shift noise in predictive draws
    sigma_d_median: float = 0.10                  # LogNormal prior on sigma_d (log-ratio units). With beta pinned near 1,
    sigma_d_sdlog: float = 0.7                    # d_t must absorb the seat's own offset (data: 0.3-1.0), so keep it free
    scenario_pct: Sequence[float] = (5, 50, 95)   # worst / average / best percentiles (matches the PDF)
    volume_drift: str = "last"                    # early-vote log-odds step: "last" (default; the share saturates), "mean" or "none"
    volume_step_floor: float = 0.25             # min sd of the next early-vote log-odds step (steps are large and few)
    
    #NOTE: THis is the one nation layer and is highly experimental. 
    # Uses the model to assess one nation.
    # IT CAN'T DO IT WITH ANY KNOWN ACCURACY SINCE THEIR SPIKE IS TOATLLY UNPRECEDENTED.
    # Hence you need to make your own assumptions, plug them in here.
    onp_layer: bool = True                        # polls table has an ONP column -> model One Nation as a separate ENTRANT
    onp_state_denominator: bool = True            # keep ONP in the statewide denominator (history AND polls) so a 20% ONP poll does not inflate the other parties; ONP still gets no seat share unless onp_layer
    sigma_eps_sd: float = 0.03                     # HalfNormal scale of booth-election noise (was hard-coded); lower it if booth coverage sits well above 90%
    run_attribution: bool = True                  # report which uncertainty layer drives the width of the seat band
    #SET BELOW TO 1.4 FOR REGIONAL/OUTER METRO SEATS, LOWER FOR LOWER ONE NATION SHARE.
    onp_seat_ratio_median: float = 0.65            # seat ONP share / statewide ONP share (East Hills 1999: 10.7 / 6.4)
    onp_seat_ratio_sdlog: float = 0.5
    onp_max_seat_share: float = 1.0             # cap on the seat-wide ONP share

    #NOTE: BrianMode (toggle). Replaces the statewide-ONP x seat-ratio entry with a fixed vote-transfer assumption:
    # ONP takes brian_onp_from_lnp of the LNP vote and brian_onp_from_alp of the ALP vote, booth by booth, AFTER the
    # fitted model has produced its ex-ONP booth shares. GRN / OTHER are untouched, so every booth still sums to 100%.
    # It does not need an ONP poll column. Backtests are unaffected (they never apply the ONP layer).
    brian_mode: bool = True
    brian_onp_from_lnp: float = 0.20              # share of the LNP vote that goes to ONP
    brian_onp_from_alp: float = 0.05              # share of the ALP vote that goes to ONP
    
    #NOTE: Below is to distribute KNOWN number of electors (since now I know!)
    #Note number of electors does not equal the number of voters.
    #Turnout matters.
    registered_electors: Optional[float] = None       # Total registered electors from NSWEC
    expected_turnout_rate: float = 0.90               # Expected voter turnout rate (e.g. 90%)
    informal_rate_fallback: float = 0.04              # informal-vote rate used ONLY if the booth file has no votes to infer one from

    #NOTE: everything below used to be a literal buried inside the model. Defaults reproduce the old behaviour exactly.
    # --- share-model priors
    beta_data_mu: float = 0.3                     # beta_preset="data": ALP/LNP/GRN prior mean
    beta_data_sd: float = 0.30                    # beta_preset="data": prior sd
    beta_uniform_mu: float = 1.0                  # beta_preset="uniform": prior mean
    beta_uniform_sd: float = 0.15                 # beta_preset="uniform": prior sd
    beta_grn_sd: float = 0.50                     # GRN prior sd under the "data" preset (small share, weakly informative)
    beta_other_mu: float = 0.0                    # OTHER is decoupled from the statewide swing
    beta_other_sd: float = 0.30
    m_prior_sd: float = 1.0                       # sd of the prior on the party intercepts m_k around their empirical log-ratio
    rho_beta_a: float = 2.0                       # rho_d ~ Beta(a, b), used when d_persist is None
    rho_beta_b: float = 2.0
    # --- volume-model priors and sampler
    volume_draws: Optional[int] = None            # None: same as draws
    volume_tune_floor: int = 3000                 # the volume model always tunes for at least this many steps
    vol_sigma_a_sd: float = 1.5                   # HalfNormal scale of the booth-size spread
    vol_tau_sd: float = 0.5                       # Normal sd of the early-vote log-share steps
    vol_kappa_alpha: float = 2.0                  # kappa ~ Gamma(alpha, beta) (mean alpha/beta = 400 by default)
    vol_kappa_beta: float = 0.005
    election_cycle_years: float = 4.0             # years between elections (sets how many cycles ahead the target is)
    backtest_volume_sims: int = 4000              # draws used inside backtest_volume
    # --- statewide poll layer
    time_horizon_gamma: float = 0.8               # extra poll sd (pts) = gamma * sqrt(years to election)
    poll_z: float = 1.96                          # z behind the poll margin of error AND the low/high -> se conversion
    poll_default_sample_size: float = 1000.0      # used when a poll has no sample_size and none can be inferred
    poll_dirichlet_conc_floor: float = 10.0       # minimum Dirichlet concentration of the statewide share draws
    poll_sum_warn: float = 102.0                  # warn when poll shares incl. ONP sum to more than this
    share_clip_floor: float = 1e-3                # statewide shares are clipped to at least this (%), to keep logs finite
    # --- diagnostics / bookkeeping
    rhat_threshold: float = 1.01                  # sampler warning thresholds
    ess_threshold: float = 400.0
    data_check_jump_threshold: float = 0.15       # flag a year-on-year change in the seat formal total bigger than this
    attribution_seed_offset: int = 101            # attribution re-simulation seed = seed + this
    backtest_poll_house_sd: float = 0.0           # the backtest uses ACTUAL statewide shares, so house error is off by default
    
    # --- Phase 1 Change Additions ---
    #NOTE: This deals with how correlated errors are across parties (i.e., party shock in booth b effects booth n)
    tide_process: str = "ar1"                       # "ar1" (existing) or "random_walk" (secular drift)
    use_correlated_errors: bool = False             # Use LKJ Cholesky covariance matrix for party-level eps/tide errors
    lkj_eta: float = 2.0                            # LKJ prior shape parameter (higher = more concentrated around independence)

@dataclass
class PollInput:
    central: float
    low: float
    high: float


@dataclass
class ModelOutput:
    params: Dict[str, float]
    booths: pd.DataFrame
    tide: pd.DataFrame
    forecast: pd.DataFrame
    seat: pd.DataFrame
    diagnostics: Optional[pd.DataFrame] = None
    draws_df: Optional[pd.DataFrame] = None            # joint seat-wide share draws, one column per party
    booth_draws_df: Optional[pd.DataFrame] = None      # draws x booths, target party, % (real draws, for plots)
    volume_names: Optional[list] = None                # booth order of volume_matrix
    volume_matrix: Optional[np.ndarray] = None         # simulated next-election formal votes (draws x booths)
    prepared_df: Optional[pd.DataFrame] = None         # cleaned input (booth, year, n, y_<party>)
    booth_party_draws: Optional[np.ndarray] = None     # <-- NEW: shape (draws, booths, parties)
    party_names: Optional[list] = None                 # <-- NEW: list of party columns
    attribution: Optional[pd.DataFrame] = None         # which uncertainty layer drives the seat band width

    def save(self, folder: str) -> None:
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        pd.Series(self.params, name="value").to_csv(out / "params.csv")
        self.booths.to_csv(out / "booths.csv", index=False)
        self.tide.to_csv(out / "tide.csv", index=False)
        self.forecast.to_csv(out / "forecast.csv", index=False)
        self.seat.to_csv(out / "seat.csv", index=False)
        if self.diagnostics is not None:
            self.diagnostics.to_csv(out / "diagnostics.csv", index=False)
        if self.attribution is not None:
            self.attribution.to_csv(out / "uncertainty_attribution.csv", index=False)
        if self.volume_matrix is not None:
            np.savez_compressed(out / "volume_draws.npz", names=np.array(self.volume_names), V=self.volume_matrix)
        if self.booth_party_draws is not None and self.party_names is not None:
            # Explicitly save with shape: (draws, booths, parties)
            np.savez_compressed(
                out / "booth_party_draws.npz",
                booth_party_draws=self.booth_party_draws,
                booths=np.array(self.volume_names),
                parties=np.array(self.party_names)
            )


# ----------------------------------------------------------------------------- helpers
def _to_pct(x: pd.Series) -> pd.Series:
    """Polls may be stored as fractions (0.305) or percentages (30.5)."""
    return x * 100.0 if x.dropna().max() <= 1.0 else x



def projected_votes_from_electors(booth_results: pd.DataFrame, cfg: ModelConfig,
                                registered_electors: Optional[float] = None,
                                expected_turnout_rate: Optional[float] = None) -> pd.Series:
    """Booth-level projected FORMAL votes = registered electors x turnout x (1 - historical informal rate), spread over
    booths by their share of the last election's formal votes. Arguments default to cfg.registered_electors / cfg.expected_turnout_rate."""
    registered_electors = cfg.registered_electors if registered_electors is None else registered_electors
    expected_turnout_rate = cfg.expected_turnout_rate if expected_turnout_rate is None else expected_turnout_rate
    if registered_electors is None:
        raise ValueError("registered_electors is required (set cfg.registered_electors).")
    df = prepare_bayes_data(booth_results, cfg)

    # 1. Infer the historical informal rate directly from past data across the seat
    total_votes_all = pd.to_numeric(booth_results["total_votes"].astype(str).str.replace(",", ""), errors="coerce").sum()
    informal_votes_all = pd.to_numeric(booth_results["informal_votes"].astype(str).str.replace(",", ""), errors="coerce").sum()
    historical_informal_rate = informal_votes_all / total_votes_all if total_votes_all > 0 else cfg.informal_rate_fallback

    # 2. Get the latest election year booth distribution
    latest_year = int(df["year"].max())
    latest_df = df[df["year"] == latest_year].set_index("booth")

    # Each booth's share of formal votes at the last election
    booth_shares = latest_df["n"] / latest_df["n"].sum()

    # 3. Calculate total formal votes using registered electors, turnout, and the inferred informal rate
    total_formal = registered_electors * expected_turnout_rate * (1.0 - historical_informal_rate)

    # 4. Distribute across booths by historical ratio
    return booth_shares * total_formal

def poll_summary(polls: pd.DataFrame, party: str, half_life_months: float = 2.0,
                as_of: Optional[str] = None, moe: Optional[float] = None, z: float = 1.96,
                default_n: float = 1000.0) -> "PollInput":
    """Polls table -> PollInput(central, low, high) for statewide share S_next (%).
    Weights: 0.5 ** (calendar months before the as-of month / half_life) (the workbook rule).
    Margin of error (if moe is None): z * standard error of the weighted mean, from each poll's
    sample_size (missing sizes use the median). It ignores house effects, so treat it as a minimum."""
    p = polls.dropna(subset=[party]).copy()
    p["date"] = pd.to_datetime(p["date"])
    ref = pd.to_datetime(as_of) if as_of is not None else p["date"].max()
    p = p[p["date"] <= ref]
    if p.empty:
        raise ValueError("No polls on or before the as-of date.")
    months = (ref.year - p["date"].dt.year) * 12 + (ref.month - p["date"].dt.month)
    w = 0.5 ** (months / half_life_months)
    s = _to_pct(p[party].astype(float))
    central = float((s * w).sum() / w.sum())
    if moe is None:
        n = (p["sample_size"].astype(float) if "sample_size" in p.columns
            else pd.Series(np.nan, index=p.index))
        n = n.fillna(n.median() if n.notna().any() else default_n)
        pr = central / 100.0
        moe = z * float(np.sqrt((w ** 2 * pr * (1 - pr) / n).sum()) / w.sum()) * 100.0
    return PollInput(central, central - moe, central + moe)

def poll_weight_table(polls: pd.DataFrame, parties: Sequence[str], half_life_months: float = 2.0,
                      as_of: Optional[str] = None) -> pd.DataFrame:
    """The polls that enter the average, with the weight each one gets (same rule as poll_summary). Shares in %."""
    p = polls.copy()
    p["date"] = pd.to_datetime(p["date"])
    ref = pd.to_datetime(as_of) if as_of is not None else p["date"].max()
    p = p[p["date"] <= ref]
    months = (ref.year - p["date"].dt.year) * 12 + (ref.month - p["date"].dt.month)
    out = pd.DataFrame({"date": p["date"].values, "age_months": months.values, "weight": (0.5 ** (months / half_life_months)).values})
    for c in list(parties) + (["ONP"] if "ONP" in p.columns and "ONP" not in parties else []):
        if c in p.columns:
            out[c] = _to_pct(p[c].astype(float)).values
    return out


def get_time_horizon_sd(prediction_date, election_date, gamma=3):
    """Extra statewide poll error (pts) = gamma * sqrt(years to election)."""
    pred = pd.to_datetime(prediction_date) if prediction_date else pd.Timestamp.today()
    days = max(0, (pd.to_datetime(election_date) - pred).days)
    return gamma * np.sqrt(days / 365.0)

    # in _poll_draws



# ----------------------------------------------------------------------------- model

# Joint multinomial-logit model with PARTY-SPECIFIC sensitivities (reference party = LAST in cfg.parties, e.g. OTHER).
# Every booth-election has vote counts y_k out of n formal votes, k = parties.
#   y_bt ~ Multinomial(n_bt, softmax(eta_bt))
#   eta_kbt = m_k + alpha_kb + beta_k * x_kt + d_kt + eps_kbt      (k < K)          eta_Kbt = alpha_Kb + beta_K * x_Kt + d_Kt + eps_Kbt   (reference; m_K = 0)
#   x_kt = log S_kt - mean_t log S_kt      party k's OWN statewide log-share swing, centred over elections
#   alpha_kb ~ N(0, sigma_alpha_k)  booth baseline (shrunk)       d_kt = AR(1) over elections, stationary N(0, sigma_d_k), for ALL K parties (softmax is
#   shift-invariant, so the reference party gets a tide too and the model no longer depends on which party is last). The prior median of
#   sigma_d is divided by sqrt(2) so the shock on a log-ratio (sd = sqrt(sigma_i^2 + sigma_j^2)) keeps the old prior scale.
#   eps_kbt  ~ N(0, sigma_eps_k)    booth-election local shift     beta_k ~ Normal(mu_k, sd_k) (NOT truncated: a seat may move against the state)
# OTHER is decoupled from the statewide swing: the seat's independents do not follow the statewide minor-party vote, and
# tying them together (one shared beta on log-ratios vs OTHER) was the source of the large LNP / OTHER backtest errors.
# beta is an ASSUMED prior here (one seat cannot identify it); see empirical_check() and cfg.beta_preset.
# Booth volumes: total formal votes T (enrolment growth) x booth SHARES (Dirichlet-multinomial, early-vote trend tau_t),
# so a fall in ordinary-booth votes appears as growth in early-vote booths and the total is conserved.
# Forecast per posterior draw: S_next ~ Dirichlet(polls + house effect); d_next; volumes V; seat share_k = sum_b V_b p_kb / sum_b V_b.




def prepare_bayes_data(booths: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Clean vote counts. OTHER = formal votes minus the named parties, so every row sums exactly to n. Merges venues via cfg.booth_merge and aggregates duplicates within a year."""
    parties = list(cfg.parties)
    explicit = [p for p in parties if p != "OTHER"]          # OTHER is always the remainder, wherever it sits in the order
    need = {"year", "booth", "total_votes", "informal_votes"} | set(explicit)
    missing = need - set(booths.columns)
    if missing:
        raise ValueError(f"booth_results is missing columns: {sorted(missing)}")
    df = booths.dropna(subset=["booth", "year"]).copy()           # blank trailing rows
    df["booth"] = df["booth"].astype(str).str.strip()
    if cfg.booth_merge:
        df["booth"] = df["booth"].replace(cfg.booth_merge)
    df = df[~df["booth"].isin(cfg.exclude_booths)].copy()
    num = ["total_votes", "informal_votes"] + explicit
    for c in num:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    bad = df[df[num].isna().any(axis=1)]
    if len(bad):
        raise ValueError(f"Non-numeric / blank vote counts in {len(bad)} row(s), e.g. booth={bad.iloc[0]['booth']!r}, "
                        f"year={bad.iloc[0]['year']}. Fix the CSV (blank cells are NOT treated as 0).")
    df["year"] = df["year"].astype(int)
    check_booth_data(df, cfg)
    type_map = _booth_types(df, cfg)
    df = df.groupby(["year", "booth"], as_index=False)[num].sum()  # merged venues in the same year
    df["btype"] = df["booth"].map(type_map).fillna(0).astype(int)   # 1 = early voting (pre-poll, postal, ...)
    df["n"] = (df["total_votes"] - df["informal_votes"]).round().astype(int)
    dropped = int((df["n"] <= 0).sum())
    if dropped:
        print(f"NOTE: dropped {dropped} row(s) with non-positive formal votes.")
    df = df[df["n"] > 0].copy()
    if df.empty:
        raise ValueError("No valid booths remain after filtering.")
    for p in explicit:
        df[f"y_{p}"] = df[p].round().astype(int)
    if "OTHER" in parties:
        df["y_OTHER"] = df["n"] - sum(df[f"y_{p}"] for p in explicit)
        if (df["y_OTHER"] < 0).any():
            raise ValueError("Explicit party votes exceed formal votes in some rows; check the CSV.")
    if cfg.party not in parties:
        raise ValueError(f"cfg.party={cfg.party!r} must be one of cfg.parties={parties}")
    df["formal"] = df["n"]
    df["y"] = df[f"y_{cfg.party}"]
    df["share"] = df["y"] / df["n"] * 100.0
    return df.reset_index(drop=True)


def _state_matrix(state_df: pd.DataFrame, parties: Sequence[str], years: Sequence[int], onp_in_denominator: bool = False) -> pd.DataFrame:
    """Statewide shares (%) for the modelled parties. Default: each row re-normalised to sum to 100 over the modelled parties.
    onp_in_denominator=True: if the table has an ONP column that is not modelled, ONP stays in the denominator, so the
    modelled parties sum to LESS than 100 (the gap is ONP) and are on the same footing as polls that include ONP."""
    full = state_df.set_index("year")
    S = full[list(parties)].astype(float)
    absent = [y for y in years if y not in S.index]
    if absent:
        raise ValueError(f"State results missing for years: {absent}")
    S = S.loc[list(years)]
    if not (1.0 < S.values.max() < 100.0):
        raise ValueError("State shares must be percentages (e.g. 36.97), not fractions.")
    denom = S.sum(axis=1)
    if onp_in_denominator and "ONP" in full.columns and "ONP" not in parties:
        denom = denom + full.loc[list(years), "ONP"].astype(float)
    return S.div(denom, axis=0) * 100.0


def beta_prior_table(cfg: ModelConfig) -> Dict[str, tuple]:
    """party -> (mean, sd, lower) of the party's sensitivity to ITS OWN statewide log-share swing.
    OTHER is decoupled from the statewide swing (prior near 0): the seat's independents do not follow the
    statewide minor-party vote. preset 'data' centres ALP/LNP/GRN on what this seat's history implies;
    preset 'uniform' is the stated scenario beta ~ 1. cfg.beta_g / cfg.beta_prior_sd override both."""
    if cfg.beta_priors:
        return {p: tuple(cfg.beta_priors[p]) for p in cfg.parties}
    uniform = cfg.beta_preset == "uniform"
    mu, sd = (cfg.beta_uniform_mu, cfg.beta_uniform_sd) if uniform else (cfg.beta_data_mu, cfg.beta_data_sd)
    if cfg.beta_g is not None:
        mu = cfg.beta_g
    if cfg.beta_prior_sd is not None:
        sd = cfg.beta_prior_sd
    table = {p: (mu, sd, 0.0) for p in cfg.parties}
    if "GRN" in table and not uniform and cfg.beta_prior_sd is None:
        table["GRN"] = (mu, cfg.beta_grn_sd, 0.0)               # small share, tiny state movement: weakly informative
    if "OTHER" in table:
        # Increase the prior SD from 0.30 to 1.0 to let the sampler explore real variance
        table["OTHER"] = (cfg.beta_other_mu, 1.0, None)
    if "ONP" in table:
        table["ONP"] = (0.3, 1.0, None)
    return table


def _build_design(df: pd.DataFrame, cfg: ModelConfig, state_df: pd.DataFrame) -> dict:
    """Index maps + each party's centred statewide LOG-share swing x_kt = log S_kt - mean_t(log S_kt)."""
    parties = list(cfg.parties)
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) < 2:
        raise ValueError("Need at least two elections to fit the model.")
    S = _state_matrix(state_df, parties, years, cfg.onp_state_denominator)
    logS = np.log(S.values / 100.0)
    x_center = logS.mean(axis=0)
    booth_names = sorted(df["booth"].unique())
    if cfg.verbose:
        print("\n--- STATE LOG-SHARE SWING x (centred; columns = " + ", ".join(parties) + ") ---")
        print("Years:", years)
        print(np.round(logS - x_center, 3))
    return dict(
        parties=parties, years=years, S_pct=S, x=logS - x_center, x_center=x_center,
        booth_names=booth_names,
        booth_idx=df["booth"].map({b: i for i, b in enumerate(booth_names)}).values,
        year_idx=df["year"].map({y: i for i, y in enumerate(years)}).values,
        Y=df[[f"y_{p}" for p in parties]].values.astype(int),
    )


def _stack(idata, name: str) -> np.ndarray:
    """Posterior -> array with draws on axis 0 (chains x draws flattened)."""
    arr = idata.posterior[name].stack(sample=("chain", "draw")).values
    return np.moveaxis(arr, -1, 0)






def fit_bayesian_model(df: pd.DataFrame, cfg: ModelConfig, state_df: pd.DataFrame) -> dict:
    """Joint multinomial-logit model with PARTY-SPECIFIC sensitivities beta_k to each party's own statewide
    log-share swing; the reference party (last in cfg.parties) carries only its beta term."""
    if not HAS_PYMC:
        raise ImportError("PyMC is required. Install via: pip install pymc arviz")
    des = _build_design(df, cfg, state_df)
    parties, Km = des["parties"], len(des["parties"]) - 1
    Y = des["Y"]
    N = len(df)
    n = Y.sum(axis=1)
    m0 = np.log(Y[:, :Km].sum(axis=0) / Y[:, Km].sum())
    K, T = len(parties), len(des["years"])
    est_rho = cfg.d_persist is None
    if not est_rho and not (-1.0 < float(cfg.d_persist) < 1.0):
        raise ValueError("cfg.d_persist must be None (estimate) or a number in (-1, 1).")
    coords = {"booth": des["booth_names"], "year": des["years"], "obs": np.arange(N), "pm": parties[:Km], "party": parties}
    priors = beta_prior_table(cfg)

    B = len(des["booth_names"])
    with pm.Model(coords=coords):
        # 1. Instantiate the hyper-means vector directly for the Km non-reference categories
        mu_beta_raw = pm.Normal("mu_beta_raw", 
                                mu=np.array([priors[p][0] for p in parties[:Km]]), 
                                sigma=np.array([priors[p][1] for p in parties[:Km]]), 
                                shape=Km)
        
        # 2. Enforce zero-anchoring on the reference category (OTHER)
        mu_beta_vec = pm.Deterministic("mu_beta_vec", pt.concatenate([mu_beta_raw, pt.zeros((1,))]), dims="party")
        
        # 3. Variance scale & localized non-centered random variables
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.20, shape=Km)
        sigma_beta_full = pt.concatenate([sigma_beta, pt.zeros((1,))])
        
        beta_z = pm.Normal("beta_z", mu=0.0, sigma=1.0, dims=("booth", "party"))
        
        # 4. Generate the unidentifiable-free slopes matrix
        beta_matrix = pm.Deterministic("beta_matrix", mu_beta_vec[None, :] + beta_z * sigma_beta_full[None, :], dims=("booth", "party"))


        m = pm.Normal("m", mu=m0, sigma=cfg.m_prior_sd, dims="pm")
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=cfg.sigma_alpha_sd / np.sqrt(2.0), dims="party")   # /sqrt2: log-ratio scale unchanged
        sigma_d = pm.LogNormal("sigma_d", mu=float(np.log(cfg.sigma_d_median) - 0.5 * np.log(2.0)), sigma=cfg.sigma_d_sdlog, dims="party")
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=cfg.sigma_eps_sd / np.sqrt(2.0), dims="party")
        # Use pm.TruncatedNormal to keep continuous gradient lines intact for NUTS
        alpha_z_raw = pm.Normal("alpha_z_raw", 0.0, 1.0, dims=("booth", "party"))
        
        alpha_base = alpha_z_raw * sigma_alpha
        lnp_idx = parties.index("LNP") if "LNP" in parties else 1
        
        # Enforce the demographic baseline floor cleanly on the tensor graph
        alpha_lnp_floored = pt.where(alpha_base[:, lnp_idx] < -1.38, -1.38, alpha_base[:, lnp_idx])
        alpha_b_raw = pt.set_subtensor(alpha_base[:, lnp_idx], alpha_lnp_floored)
        
        alpha_b = pm.Deterministic("alpha_b", alpha_b_raw, dims=("booth", "party"))

        # --- PHASE 1 IMPLEMENTATION: Correlated Errors via LKJ Cholesky ---
        if cfg.use_correlated_errors:
            chol_packed, _, _ = pm.LKJCholeskyCov("chol_cov", eta=cfg.lkj_eta, n=K, sd_dist=pm.HalfNormal.dist(sigma=1.0), compute_corr=True)
            d_z_raw = pm.Normal("d_z_raw", 0.0, 1.0, shape=(T, K))
            eps_z_raw = pm.Normal("eps_z_raw", 0.0, 1.0, shape=(N, K))
            d_z = pt.dot(d_z_raw, chol_packed.T)
            eps_z = pt.dot(eps_z_raw, chol_packed.T)
        else:
            d_z = pm.Normal("d_z", 0.0, 1.0, dims=("year", "party"))
            eps_z = pm.Normal("eps_z", 0.0, 1.0, dims=("obs", "party"))

        # --- PHASE 1 IMPLEMENTATION: Secular Random Walk vs AR(1) Tide ---
        if cfg.tide_process == "random_walk":
            d_steps = [sigma_d * d_z[0, :]]
            for t in range(1, T):
                d_steps.append(d_steps[-1] + sigma_d * d_z[t, :])
            d_t = pm.Deterministic("d_t", pt.stack(d_steps, axis=0), dims=("year", "party"))
            rho = 1.0
        else:
            if est_rho:
                rho = pm.Beta("rho_d", alpha=cfg.rho_beta_a, beta=cfg.rho_beta_b)
                innov = pt.sqrt(pt.maximum(1.0 - rho ** 2, 1e-6))
            else:
                rho = float(cfg.d_persist)
                innov = float(np.sqrt(max(1.0 - rho ** 2, 1e-6)))
            d_steps = [sigma_d * d_z[0, :]]
            for t in range(1, T):
                d_steps.append(rho * d_steps[-1] + innov * sigma_d * d_z[t, :])
            d_t = pm.Deterministic("d_t", pt.stack(d_steps, axis=0), dims=("year", "party"))

        eps = pm.Deterministic("eps", eps_z * sigma_eps, dims=("obs", "party"))
        beta_obs = beta_matrix[des["booth_idx"]]                                
        bx = beta_obs * des["x"][des["year_idx"]]                              
        d_obs = d_t[des["year_idx"]]                                            
        m_full = pt.concatenate([m, pt.zeros((1,))])                            
        eta = m_full[None, :] + alpha_b[des["booth_idx"]] + bx + d_obs + eps    

        pm.Multinomial("y_obs", n=n, p=pm.math.softmax(eta, axis=-1), observed=Y)
        idata = pm.sample(draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, target_accept=cfg.target_accept,
                        random_seed=cfg.seed, return_inferencedata=True)

    summ = az.summary(idata, var_names=["mu_beta_raw", "sigma_beta", "m", "sigma_alpha", "sigma_d", "sigma_eps"]
                    + (["rho_d"] if est_rho and cfg.tide_process != "random_walk" else []))

    divergences = diagnose_sampler(idata, summ, "share model", cfg)
    if divergences > 0:
        dv = idata.sample_stats["diverging"].values.reshape(-1).astype(bool)
        sd = idata.posterior["sigma_d"].values.reshape(dv.size, -1)
        print(f"  mean sigma_d in divergent draws {sd[dv].mean():.3f} vs {sd.mean():.3f} overall "
            "(clustered at small values => tighten sigma_d_median / sigma_d_sdlog).")
    
    rho_draws = (_stack(idata, "rho_d") if est_rho and cfg.tide_process != "random_walk" 
                 else np.full(_stack(idata, "sigma_d").shape[0], 1.0 if cfg.tide_process == "random_walk" else float(cfg.d_persist)))

    des.update(
        draws={"beta": _stack(idata, "mu_beta_vec"),  
               "beta_matrix": _stack(idata, "beta_matrix"),
               "m": _stack(idata, "m"),
            "sigma_d": _stack(idata, "sigma_d"), "sigma_eps": _stack(idata, "sigma_eps"),
            "sigma_alpha": _stack(idata, "sigma_alpha"), "sigma_beta": _stack(idata, "sigma_beta"),
            "alpha": _stack(idata, "alpha_b"), "d": _stack(idata, "d_t"), "eps": _stack(idata, "eps"),
            "rho": rho_draws},
        diag=summ.reset_index().rename(columns={"index": "parameter"}), divergences=divergences, idata=idata)
    return des


def _simulate_next(fit: dict, S_next_pct: np.ndarray, cfg: ModelConfig, rng: np.random.Generator, shut: Sequence[str] = (), trace: Optional[dict] = None):
    """One next-election outcome per posterior draw..."""
    dr = fit["draws"]
    if "params" in shut:                                              # diagnostic: collapse the posterior to its mean (no parameter uncertainty)
        dr = {k: np.broadcast_to(v.mean(axis=0, keepdims=True), v.shape) for k, v in dr.items()}
    D, B, K = dr["alpha"].shape
    Km = K - 1
    S = np.clip(np.asarray(S_next_pct, dtype=float), cfg.share_clip_floor, None)
    S = S / 100.0 if cfg.onp_state_denominator else S / S.sum(axis=1, keepdims=True)   # same footing as _state_matrix
    x_next = np.log(S) - fit["x_center"]                              # (D, K)
    
    # Party-level swing term is always (D, K). Booth-specific slopes (the fit's beta_matrix) are (D, B, K).
    bx_party = dr["beta"] * x_next                                            # (D, K)
    rho = (np.asarray(dr["rho"], dtype=float) if "rho" in dr else np.full(D, float(cfg.d_persist or 0.0)))[:, None]   # (D, 1)
    
    # --- PHASE 1 IMPLEMENTATION: Random Walk vs AR(1) Tide Projection ---
    if cfg.tide_process == "random_walk":
        d_next = dr["d"][:, -1, :] + dr["sigma_d"] * rng.standard_normal((D, K)) * (0.0 if "tide" in shut else 1.0)
    else:
        d_next = rho * dr["d"][:, -1, :] + np.sqrt(np.maximum(1.0 - rho ** 2, 0.0)) * dr["sigma_d"] * rng.standard_normal((D, K)) * (0.0 if "tide" in shut else 1.0)
        
    lin_party = bx_party + d_next                                             # (D, K)
    if "beta_matrix" in dr and tuple(dr["beta_matrix"].shape) == (D, B, K):
        lin = dr["beta_matrix"] * x_next[:, None, :] + d_next[:, None, :]    # (D, B, K): each booth has its own sensitivity
    else:
        lin = lin_party[:, None, :]                                           # (D, 1, K): one shared sensitivity
    if lin.shape not in ((D, 1, K), (D, B, K)):                              # guard: a wrong rank here silently becomes a (D, D, B, K) array
        raise ValueError(f"_simulate_next: lin has shape {lin.shape}, expected {(D, 1, K)} or {(D, B, K)}")

    if trace is not None:                                                     # S3 signals (party level, (D, K))
        trace.update({"s3_x_next": x_next, "s3_bx": bx_party, "s3_d_next": d_next, "s3_lin": lin_party})
    m_full = np.concatenate([dr["m"], np.zeros((D, 1))], axis=1)              # reference party: m_K = 0
    eta = m_full[:, None, :] + dr["alpha"] + lin                              # (D, B, K)
    if cfg.include_eps:
        eta = eta + dr["sigma_eps"][:, None, :] * rng.standard_normal((D, B, K)) * (0.0 if "booth" in shut else 1.0)
    eta = eta - eta.max(axis=2, keepdims=True)
    e = np.exp(eta)
    return e / e.sum(axis=2, keepdims=True), lin_party[:, :Km] - lin_party[:, Km:]


def _poll_draws(poll: Union[PollInput, pd.DataFrame], cfg: ModelConfig, state_df: pd.DataFrame,
                D: int, rng: np.random.Generator, trace: Optional[dict] = None):
    """Returns (S_next, central, target, se_pct, onp_state).
    S_next is EX-ONP (rows sum to 100); onp_state is (D,) statewide ONP % draws, or None without an ONP column."""
    parties = list(cfg.parties)
    i = parties.index(cfg.party)
    onp_central = None

    if isinstance(poll, pd.DataFrame):
        miss = [p for p in parties if p not in poll.columns]
        if miss:
            raise ValueError(f"Poll table needs a column for every modelled party; missing {miss}")
        summ = {p: poll_summary(poll, p, cfg.poll_half_life_months, cfg.poll_as_of, cfg.poll_moe,
                            cfg.poll_z, cfg.poll_default_sample_size) for p in parties}
        central = np.array([summ[p].central for p in parties], dtype=float)
        target = summ[cfg.party]
        if not cfg.onp_layer and not cfg.onp_state_denominator and "ONP" in poll.columns and "ONP" not in parties:
            try:
                onp_c = float(poll_summary(poll, "ONP", cfg.poll_half_life_months, cfg.poll_as_of, cfg.poll_moe,
                            cfg.poll_z, cfg.poll_default_sample_size).central)
            except ValueError:
                onp_c = 0.0
            if onp_c > 0.5:
                with_onp = 100.0 * central[i] / (central.sum() + onp_c)
                without = 100.0 * central[i] / central.sum()
                print(f"\n!! WARNING: ONE NATION IS BEING DROPPED FROM THE STATEWIDE POLLS !!\n"
                    f"   Polls have ONP at {onp_c:.1f}% but cfg.onp_layer is False, so it is removed and the other parties are scaled up.\n"
                    f"   {cfg.party} statewide poll share: {with_onp:.1f}% with ONP kept vs {without:.1f}% as modelled "
                    f"(+{without - with_onp:.1f} pts inflation). Seat forecasts inherit this.")
        if (cfg.onp_layer or cfg.onp_state_denominator) and "ONP" in poll.columns and "ONP" not in parties:
            try:
                onp_central = float(poll_summary(poll, "ONP", cfg.poll_half_life_months, cfg.poll_as_of, cfg.poll_moe,
                            cfg.poll_z, cfg.poll_default_sample_size).central)
            except ValueError:
                onp_central = None
    else:
        target = poll
        last = _state_matrix(state_df, parties, [int(state_df["year"].max())]).values[0]
        others = np.delete(last, i)
        central = np.empty(len(parties))
        central[i] = poll.central
        central[np.arange(len(parties)) != i] = (100.0 - poll.central) * others / others.sum()

    if onp_central is not None:
        full = np.append(central, onp_central)
        if full.sum() > cfg.poll_sum_warn and cfg.verbose:
            print(f"WARNING: poll shares incl. ONP sum to {full.sum():.1f}%. With an ONP column, OTHER must EXCLUDE ONP.")
        full = 100.0 * full / full.sum()
        central, onp_central = full[:-1], float(full[-1])
        if cfg.onp_state_denominator and cfg.verbose:
            print(f"-> ONP kept in the statewide denominator: polls ONP {onp_central:.1f}%, {cfg.party} statewide {central[i]:.1f}% "
                f"(ONP gets no seat share unless cfg.onp_layer is on).")
    else:
        central = 100.0 * central / central.sum()
        if cfg.verbose and isinstance(poll, pd.DataFrame) and "ONP" not in poll.columns and cfg.onp_layer:
            print("NOTE: polls have no ONP column, so One Nation is inside OTHER and is NOT modelled as a separate entrant.")
    se_poll = (target.high - target.low) / (2 * cfg.poll_z)

    h_sd = get_time_horizon_sd(cfg.prediction_date, cfg.election_date, cfg.time_horizon_gamma)
    se = np.sqrt(se_poll ** 2 + cfg.poll_house_sd ** 2 + h_sd ** 2) / 100.0
    p_t = central[i] / 100.0
    conc = max(p_t * (1 - p_t) / se ** 2 - 1.0, cfg.poll_dirichlet_conc_floor)
    if trace is not None:                                                  # S2 signals
        trace["s2_se_parts"] = np.array([se_poll, cfg.poll_house_sd, h_sd, se * 100.0], dtype=float)   # pts: sampling, house, horizon, total
        trace["s2_conc"] = float(conc)
        trace["s2_central"] = np.asarray(central, dtype=float)
        trace["s2_central_onp"] = float(onp_central) if onp_central is not None else float("nan")
        if isinstance(poll, pd.DataFrame):
            trace["s2_poll_table"] = poll_weight_table(poll, parties, cfg.poll_half_life_months, cfg.poll_as_of)

    if cfg.verbose:
        print(f"-> Time-Horizon Uncertainty: election {cfg.election_date}. "
            f"Horizon sd: {h_sd:.2f} pts | poll sampling se: {se_poll:.2f} | house sd: {cfg.poll_house_sd:.2f} "
            f"| total poll se: {se * 100.0:.2f} pts")

    if onp_central is None:
        return rng.dirichlet(conc * central / 100.0, size=D) * 100.0, central, target, se * 100.0, None
    X = rng.dirichlet(conc * np.append(central, onp_central) / 100.0, size=D) * 100.0     # (D, K+1), last = ONP
    S = X[:, :-1]
    return (S if cfg.onp_state_denominator else 100.0 * S / S.sum(axis=1, keepdims=True)), central, target, se * 100.0, X[:, -1]
# ----------------------------------------------------------------------------- booth volume model
def _booth_types(df: pd.DataFrame, cfg: ModelConfig) -> Dict[str, int]:
    """booth -> 1 (early voting: pre-poll, postal, declaration, ...) or 0 (ordinary election-day booth).
    Uses a 'booth_type' column if the CSV has one; otherwise matches cfg.early_keywords in the booth name."""
    kws = tuple(k.upper() for k in cfg.early_keywords)
    out: Dict[str, int] = {}
    for _, r in df.iterrows():
        b = str(r["booth"])
        text = (str(r["booth_type"]) if "booth_type" in df.columns and pd.notna(r.get("booth_type")) else b).upper()
        out[b] = max(out.get(b, 0), int(any(k in text for k in kws)))
    return out


def _fit_volume_posterior(df: pd.DataFrame, cfg: ModelConfig) -> dict:
    """Booth volume as a SHARE of the election's total formal votes (conserves the total; a fall in ordinary
    booths shows up as growth in early-vote booths):
        counts_t ~ DirichletMultinomial(T_t, kappa * softmax(eta_bt)),  eta_bt = a_b + tau_t * early_b
    a_b: booth size (shrunk); tau_t: early-vote log-share trend (tau_first = 0); only booths present in
    election t enter that election's softmax. The TOTAL T_t is forecast separately (see _simulate_volumes)."""
    if not HAS_PYMC:
        raise ImportError("PyMC is required. Install via: pip install pymc arviz")
    names = sorted(df["booth"].unique())
    years = sorted(int(y) for y in df["year"].unique())
    B, T = len(names), len(years)
    C = df.pivot(index="year", columns="booth", values="n").reindex(index=years, columns=names).values
    present = ~np.isnan(C)
    btype = df.groupby("booth")["btype"].max().reindex(names).fillna(0).values.astype(float)
    has_early = bool(btype.sum() > 0)
    early_present = [bool(present[ti][btype == 1].any()) for ti in range(T)]
    t0 = next((ti for ti, e in enumerate(early_present) if e), T)      # first election with an early-voting booth
    n_free = max(T - 1 - t0, 0)
    has_free = has_early and n_free > 0                                # tau_t is only identified from t0 onwards
    with pm.Model(coords={"booth": names}):
        s_a = pm.HalfNormal("sigma_a", sigma=cfg.vol_sigma_a_sd)
        a_z = pm.Normal("a_z", 0.0, 1.0, dims="booth")
        a = pm.Deterministic("a", a_z * s_a, dims="booth")
        if has_free:
            tau_free = pm.Normal("tau_free", 0.0, cfg.vol_tau_sd, shape=n_free)
            tau = pm.Deterministic("tau", pt.concatenate([pt.zeros(t0 + 1), tau_free]))
        else:
            tau = pt.zeros(T)
        kappa = pm.Gamma("kappa", alpha=cfg.vol_kappa_alpha, beta=cfg.vol_kappa_beta)                 # concentration (mean 400)
        for ti, yr in enumerate(years):
            idx = np.where(present[ti])[0]
            pi = pm.math.softmax(a[idx] + tau[ti] * btype[idx])
            pm.DirichletMultinomial(f"n_{yr}", n=int(C[ti, idx].sum()), a=kappa * pi,
                                    observed=C[ti, idx].astype(int))
        idata = pm.sample(draws=(cfg.volume_draws or cfg.draws), tune=max(cfg.tune, cfg.volume_tune_floor), chains=cfg.chains,
                        target_accept=cfg.volume_target_accept, random_seed=cfg.seed,
                        return_inferencedata=True, progressbar=False)
    vars_ = ["sigma_a", "kappa"] + (["tau"] if has_free else [])
    summ = az.summary(idata, var_names=vars_)
    div = diagnose_sampler(idata, summ, "volume model", cfg)
    S = idata.posterior["kappa"].stack(sample=("chain", "draw")).values.shape[-1]
    tau_draws = _stack(idata, "tau") if has_free else np.zeros((S, T))
    return dict(names=names, years=years, btype=btype, t0=t0, totals=np.nansum(C, axis=1), divergences=div, diag=summ,
                draws={"a": _stack(idata, "a"), "tau": tau_draws, "kappa": _stack(idata, "kappa")})


def _simulate_volumes(vfit: dict, D: int, cfg: ModelConfig, rng: np.random.Generator,
                    target_year: Optional[int] = None, active: Optional[np.ndarray] = None) -> np.ndarray:
    """Simulated formal votes (D x booths; 0 for inactive booths).
    total   T_next = T_last * exp(k*g + sqrt(k)*sd*z), g = mean historical log growth of the total (enrolment),
            sd floored at cfg.volume_total_sd_floor, k = elections ahead (4-year cycle);
    shares  tau_next = tau_last + k*drift + sqrt(k)*sd*z with drift = mean / last / 0 of the historical early-vote
            trend steps (cfg.volume_drift), sd floored at cfg.volume_step_floor;
    booths  p ~ Dirichlet(kappa * softmax(a_b + tau_next*early_b)) over ACTIVE booths, counts ~ Multinomial(T_next, p)."""
    dr, names = vfit["draws"], vfit["names"]
    B = len(names)
    act = np.ones(B, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    ai = np.where(act)[0]
    idx = rng.integers(0, len(dr["kappa"]), size=D)
    a, tau, kappa = dr["a"][idx][:, ai], dr["tau"][idx], dr["kappa"][idx]
    bt = vfit["btype"][ai]
    years, totals = vfit["years"], np.asarray(vfit["totals"], dtype=float)
    t_year = target_year if target_year is not None else int(round(years[-1] + cfg.election_cycle_years))
    if t_year in years:                                               # in-sample election
        tau_t = tau[:, years.index(t_year)]
        Tn = np.full(D, totals[years.index(t_year)])
    else:
        k = max(1, int(round((t_year - years[-1]) / cfg.election_cycle_years)))
        steps = np.diff(tau[:, min(vfit.get("t0", 0), tau.shape[1] - 1):], axis=1)   # identified steps only
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
    p = gam / gam.sum(axis=1, keepdims=True)
    counts = rng.multinomial(np.round(Tn).astype(int), p)
    V = np.zeros((D, B))
    V[:, ai] = counts
    return np.maximum(V, 0.0)


def _apply_onp_entry(p_next: np.ndarray, onp_state_pct: np.ndarray, cfg: ModelConfig, rng: np.random.Generator):
    """One Nation as an ENTRANT: seat ONP share s = kappa * statewide ONP share (kappa lognormal), capped; the other
    parties keep their modelled shares of the remaining (1 - s). Returns p_out (D x B x (K+1), last = ONP) and s."""
    D, B, _ = p_next.shape
    kappa = np.exp(np.log(cfg.onp_seat_ratio_median) + cfg.onp_seat_ratio_sdlog * rng.standard_normal(D))
    s = np.clip(kappa * np.asarray(onp_state_pct, dtype=float) / 100.0, 0.0, cfg.onp_max_seat_share)
    p_out = np.concatenate([p_next * (1.0 - s)[:, None, None], np.broadcast_to(s[:, None, None], (D, B, 1))], axis=2)
    return p_out, s

def _apply_brian_mode(p_next: np.ndarray, parties: Sequence[str], cfg: ModelConfig):
    """BrianMode: ONP as a fixed vote-transfer. In every booth and draw, ONP takes cfg.brian_onp_from_lnp of the LNP share and
    cfg.brian_onp_from_alp of the ALP share; the rest of each party's vote stays put. Returns (p_out, retain) where p_out is
    D x B x (K+1) (last = ONP) and retain[k] is the fraction of party k's modelled vote that survives."""
    parties = list(parties)
    if "ONP" in parties:
        raise ValueError("BrianMode builds ONP itself, so 'ONP' must not be in cfg.parties.")
    take = {"LNP": cfg.brian_onp_from_lnp, "ALP": cfg.brian_onp_from_alp}
    for name, frac in take.items():
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"BrianMode fraction for {name} must be between 0 and 1, got {frac}.")
    if not any(name in parties for name in take):
        raise ValueError("BrianMode needs ALP and/or LNP in cfg.parties.")
    retain = np.array([1.0 - take.get(p, 0.0) for p in parties], dtype=float)
    onp = np.zeros(p_next.shape[:2], dtype=float)
    for j, p in enumerate(parties):
        onp += p_next[:, :, j] * (1.0 - retain[j])
    p_out = np.concatenate([p_next * retain[None, None, :], onp[:, :, None]], axis=2)
    return p_out, retain


def uncertainty_attribution(fit: dict, S_next: np.ndarray, central: np.ndarray, V: np.ndarray, keep: np.ndarray, k0: int,
                            cfg: ModelConfig, onp_seat: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Which layer makes the seat band wide? Re-simulates the target party's seat share with ONE layer switched off at a time
    (no refit; same random numbers each time), then with all off. Layers: poll (statewide poll error), params (posterior
    parameter uncertainty), tide (the seat's new election-to-election shock), booth (booth-level noise), volumes (turnout/booth sizes).
    A big cut in width = that layer drives the band. Approximate: layers interact, so cuts do not add up to 100%."""
    D = len(fit["draws"]["beta"])
    S_fixed = np.tile(np.asarray(central, dtype=float), (D, 1))
    V_fixed = np.tile(np.median(V, axis=0), (D, 1))

    def run(shut):
        rng = np.random.default_rng(cfg.seed + cfg.attribution_seed_offset)
        p, _ = _simulate_next(fit, S_fixed if "poll" in shut else S_next, cfg, rng, shut=shut)
        Vm = (V_fixed if "volumes" in shut else V)[:, keep]
        share = 100.0 * np.einsum("db,db->d", Vm, p[:, keep, k0]) / Vm.sum(axis=1)
        return share * (1.0 - onp_seat) if onp_seat is not None else share

    layers = ["poll", "params", "tide", "booth", "volumes"]
    runs = [("all layers on", ())] + [(f"switch off: {l}", (l,)) for l in layers] + [("all layers off (point forecast)", tuple(layers))]
    rows, base_w = [], None
    for name, shut in runs:
        x = run(shut)
        p05, p50, p95 = np.percentile(x, [5, 50, 95])
        w = float(p95 - p05)
        base_w = w if base_w is None else base_w
        rows.append({"scenario": name, "p05": p05, "p50": p50, "p95": p95, "width_90": w, "sd": float(x.std()),
                     "width_cut_pts": base_w - w, "width_cut_pct": 100.0 * (base_w - w) / base_w if base_w else np.nan})
    return pd.DataFrame(rows)


def predict_bayes(booths: pd.DataFrame, state_results: pd.DataFrame,
                poll: Union[PollInput, pd.DataFrame], config: ModelConfig,
                projected_formal_votes: Optional[pd.Series] = None,
                target_year: Optional[int] = None,
                trace: Optional[dict] = None) -> ModelOutput:
    cfg = config or ModelConfig()
    if state_results is None:
        state_results = STATE_RESULTS
    df = prepare_bayes_data(booths, cfg)
    parties = list(cfg.parties)
    k0 = parties.index(cfg.party)
    Km = len(parties) - 1
    lo, mid, hi = (float(x) for x in cfg.scenario_pct)

    fit = fit_bayesian_model(df, cfg, state_results)                  # ONE fit per party set
    dr = fit["draws"]
    D = len(dr["beta"])
    names = fit["booth_names"]
    rng = np.random.default_rng(cfg.seed)

    S_next, central, target_poll, se_used, onp_state = _poll_draws(poll, cfg, state_results, D, rng, trace)
    p_next, tide = _simulate_next(fit, S_next, cfg, rng, trace=trace)               # fitted model, ONP excluded
    out_parties, onp_seat, onp_attr = list(parties), None, None
    p_pre, onp_mode, retain_rec = p_next, "none", np.zeros(0)        # p_pre = booth shares BEFORE any ONP step
    if cfg.brian_mode:                                         # BrianMode: fixed LNP/ALP -> ONP transfer, ignores the ONP poll column
        p_next, retain = _apply_brian_mode(p_next, parties, cfg)
        onp_mode, retain_rec = "brian", retain
        out_parties = parties + ["ONP"]
        onp_attr = np.full(D, 1.0 - retain[k0])               # what the target party loses to ONP, for the attribution table
        if cfg.verbose:
            print(f"-> BrianMode ON: ONP takes {100 * cfg.brian_onp_from_lnp:.0f}% of the LNP vote and "
                  f"{100 * cfg.brian_onp_from_alp:.0f}% of the ALP vote (seat-level; statewide ONP poll/kappa layer bypassed).")
    elif onp_state is not None and cfg.onp_layer:               # ONP only enters the SEAT when explicitly switched on
        p_next, onp_seat = _apply_onp_entry(p_next, onp_state, cfg, rng)
        onp_mode = "entry"
        onp_attr = onp_seat
        out_parties = parties + ["ONP"]

    # booth volumes: modelled (total x booth shares) unless projected votes or registered electors are supplied
    last_year = int(df["year"].max())
    last_n = df[df["year"] == last_year].set_index("booth")["n"].reindex(names)
    vfit, vol_mode = None, "modelled"
    if projected_formal_votes is not None:
        vol_mode = "projected"
        w = projected_formal_votes.reindex(names).fillna(last_n).fillna(0.0).values.astype(float)
        V = np.tile(w, (D, 1))
    elif cfg.registered_electors is not None:
        vol_mode = "electors"
        w = projected_votes_from_electors(booths, cfg, cfg.registered_electors, cfg.expected_turnout_rate)
        w = w.reindex(names).fillna(last_n).fillna(0.0).values.astype(float)
        V = np.tile(w, (D, 1))
    else:
        vfit = _fit_volume_posterior(df, cfg)
        if vfit["names"] != names:
            raise RuntimeError("Booth order differs between share and volume models.")
        
        # Pass target_year directly. If it's None, _simulate_volumes handles live forecast.
        V = _simulate_volumes(vfit, D, cfg, rng, target_year=target_year, active=last_n.fillna(0.0).values > 0)
    
    active = (last_n.fillna(0.0).values > 0) if projected_formal_votes is None else (V[0] > 0)
    keep = np.where(active)[0]                                          # booths that voted last time / have projections
    kept_names = [names[i] for i in keep]
    Vk, Pk = V[:, keep], p_next[:, keep, :]

    attribution = (uncertainty_attribution(fit, S_next, central, V, keep, k0, cfg, onp_attr) if cfg.run_attribution else None)

    # seat-wide shares, jointly across parties
    seat_all = 100.0 * np.einsum("db,dbk->dk", Vk, Pk) / Vk.sum(axis=1, keepdims=True)
    draws_df = pd.DataFrame(seat_all, columns=out_parties)      # was columns=parties
    sq = np.percentile(seat_all[:, k0], [lo, mid, hi])
    tq = np.percentile(tide[:, k0], [lo, mid, hi]) if k0 < Km else np.full(3, np.nan)   # reference party has no tide parameter
    exp_total = float(np.median(Vk.sum(axis=1)))
    seat = pd.DataFrame([{
        "scenario": k, "gamma_next": float(tq[i]), "projected_primary_votes": round(sq[i] / 100.0 * exp_total),
        "total_formal_votes": round(exp_total), "primary_share_pct": float(sq[i]),
        "seat_p05": float(np.percentile(seat_all[:, k0], 5)), "seat_p95": float(np.percentile(seat_all[:, k0], 95)),
    } for i, k in enumerate(SCENARIOS)])

    P = 100.0 * Pk[:, :, k0]
    q = np.percentile(P, [5, lo, mid, hi, 95], axis=0)
    last_share = df[df["year"] == last_year].set_index("booth")["share"].reindex(kept_names).values
    fc = pd.DataFrame({"booth": kept_names, "worst": q[1], "average": q[2], "best": q[3],
                    "mean": P.mean(axis=0), "p05": q[0], "p95": q[4],
                    "formal_weight": np.median(Vk, axis=0), "last_formal": last_n.reindex(kept_names).values,
                    "last_share": last_share})
    booth_draws_df = pd.DataFrame(P, columns=kept_names)

    n_el = df.groupby("booth")["year"].nunique()
    tot = df.groupby("booth")["n"].sum()
    booths_tbl = pd.DataFrame({"booth": names, "elections": [int(n_el[b]) for b in names],
                            "formal_votes_total": [int(tot[b]) for b in names]})
    for j, p in enumerate(parties[:Km]):
        a = dr["alpha"][:, :, j] - dr["alpha"][:, :, Km]                  # vs reference party
        booths_tbl[f"alpha_{p}"] = a.mean(0)
        booths_tbl[f"alpha_{p}_sd"] = a.std(0)
    if k0 < Km:
        a = dr["alpha"][:, :, k0] - dr["alpha"][:, :, Km]
        booths_tbl["alpha_b"] = a.mean(0)
        booths_tbl["alpha_b_p05"] = np.percentile(a, 5, axis=0)
        booths_tbl["alpha_b_p95"] = np.percentile(a, 95, axis=0)
        e = df.assign(eps_logratio=(dr["eps"][:, :, k0] - dr["eps"][:, :, Km]).mean(0)).pivot(index="booth", columns="year", values="eps_logratio")
        e.columns = [f"eps_{c}" for c in e.columns]
        booths_tbl = booths_tbl.merge(e.reset_index(), on="booth", how="left")

    years = fit["years"]
    tide_tbl = pd.DataFrame({"year": years})
    for j, p in enumerate(parties[:Km]):
        bx = float(dr["beta"][:, j].mean()) * fit["x"][:, j] - float(dr["beta"][:, Km].mean()) * fit["x"][:, Km]   # vs reference party
        dm = (dr["d"][:, :, j] - dr["d"][:, :, Km]).mean(0)
        tide_tbl[f"state_{p}"] = fit["S_pct"][p].values
        tide_tbl[f"beta_x_{p}"] = bx
        tide_tbl[f"d_{p}"] = dm
        tide_tbl[f"gamma_{p}"] = bx + dm

    s = fit["diag"].set_index("parameter")
    params = {"S_next": float(central[k0]), "S_next_low": float(target_poll.low), "S_next_high": float(target_poll.high),
            "S_next_se_used": float(se_used), "beta": float(dr["beta"][:, k0].mean()), "beta_sd": float(dr["beta"][:, k0].std()),
            "d_persist": float(dr["rho"].mean()), "d_persist_sd": float(dr["rho"].std()), "rhat_max": float(s["r_hat"].max()), "ess_bulk_min": float(s["ess_bulk"].min()),
            "divergences": fit["divergences"], "posterior_draws": D, "booths_in_fit": len(names),
            "booths_forecast": len(keep), "expected_total_formal_votes": exp_total,
            "scenario_percentiles": f"{lo:g}/{mid:g}/{hi:g}"}
    for j, p in enumerate(parties):
        params[f"sigma_alpha_{p}"] = float(dr["sigma_alpha"][:, j].mean())
        params[f"sigma_eps_{p}"] = float(dr["sigma_eps"][:, j].mean())
        params[f"S_next_{p}"] = float(central[j])
    params[f"S_next_{parties[-1]}"] = float(central[-1])
    for j, p in enumerate(parties):                                       # sigma_d now exists for every party
        params[f"sigma_d_{p}"] = float(dr["sigma_d"][:, j].mean())
    for j, p in enumerate(parties):
        params[f"beta_{p}"] = float(dr["beta"][:, j].mean())
        params[f"beta_{p}_sd"] = float(dr["beta"][:, j].std())
    if onp_state is not None:
        params["onp_state_share_poll"] = float(np.mean(onp_state))
    if onp_seat is not None:                                    # just above `if vfit is not None:`
        params["onp_state_share_poll"] = float(np.mean(onp_state))
        params["onp_seat_share_mean"] = float(100.0 * onp_seat.mean())
        params["onp_seat_ratio_median"] = float(cfg.onp_seat_ratio_median)
        
    if cfg.brian_mode:
        params["brian_mode"] = 1.0
        params["brian_onp_from_lnp"] = float(cfg.brian_onp_from_lnp)
        params["brian_onp_from_alp"] = float(cfg.brian_onp_from_alp)
        params["onp_seat_share_mean"] = float(seat_all[:, -1].mean())
    if vfit is not None:
        params["volume_divergences"] = vfit["divergences"]
        params["volume_rhat_max"] = float(vfit["diag"]["r_hat"].max())

    if trace is not None:                                                  # S0, S1, S2 (draws), S3 (history), S4-S7 signals
        pri = beta_prior_table(cfg)
        last_df = df[df["year"] == last_year].set_index("booth").reindex(kept_names)
        trace.update({
            "s0_years": np.array(fit["years"]), "s0_parties": np.array(parties),
            "s0_state_pct": np.asarray(fit["S_pct"], dtype=float), "s0_x": np.asarray(fit["x"], dtype=float),
            "s0_x_center": np.asarray(fit["x_center"], dtype=float),
            "s1_beta": dr["beta"], "s1_rho": np.asarray(dr["rho"], dtype=float), "s1_sigma_d": dr["sigma_d"],
            "s1_sigma_alpha": dr["sigma_alpha"], "s1_sigma_eps": dr["sigma_eps"], "s1_m": dr["m"],
            "s1_prior_beta": np.array([[pri[p][0], pri[p][1]] for p in parties], dtype=float), "s1_diag": fit["diag"],
            "s2_S_next": np.asarray(S_next, dtype=float),
            "s2_onp_state": np.asarray(onp_state, dtype=float) if onp_state is not None else np.zeros(0),
            "s3_d_hist_mean": dr["d"].mean(axis=0), "s3_d_hist_p05": np.percentile(dr["d"], 5, axis=0),
            "s3_d_hist_p95": np.percentile(dr["d"], 95, axis=0), "s3_bx_hist": dr["beta"].mean(axis=0)[None, :] * np.asarray(fit["x"]),
            "s4_p_pre": (100.0 * p_pre[:, keep, :]).astype(np.float32), "s4_booths": np.array(kept_names),
            "s4_last_party_share": np.column_stack([100.0 * last_df[f"y_{p}"].values / last_df["n"].values for p in parties]),
            "s5_V": Vk.astype(np.float32), "s5_mode": vol_mode,
            "s5_hist_totals": df.groupby("year")["n"].sum().reindex(fit["years"]).values.astype(float),
            "s5_last_n": last_n.reindex(kept_names).values.astype(float),
            "s6_mode": onp_mode, "s6_retain": np.asarray(retain_rec, dtype=float),
            "s7_seat_draws": seat_all, "s7_out_parties": np.array(out_parties),
            "meta": {"seat_name": cfg.seat_name, "party": cfg.party, "parties": parties, "out_parties": out_parties,
                    "election_date": cfg.election_date, "prediction_date": cfg.prediction_date,
                    "scenario_pct": [float(v) for v in cfg.scenario_pct], "poll_half_life_months": cfg.poll_half_life_months,
                    "sigma_d_median": cfg.sigma_d_median, "sigma_d_sdlog": cfg.sigma_d_sdlog, "sigma_alpha_sd": cfg.sigma_alpha_sd,
                    "sigma_eps_sd": cfg.sigma_eps_sd, "rho_beta": [cfg.rho_beta_a, cfg.rho_beta_b], "d_persist": cfg.d_persist,
                    "m_prior_sd": cfg.m_prior_sd, "rhat_threshold": cfg.rhat_threshold, "ess_threshold": cfg.ess_threshold,
                    "onp_layer": bool(cfg.onp_layer), "onp_state_denominator": bool(cfg.onp_state_denominator),
                    "brian_mode": bool(cfg.brian_mode), "brian_onp_from_lnp": cfg.brian_onp_from_lnp,
                    "brian_onp_from_alp": cfg.brian_onp_from_alp}})
    return ModelOutput(
        params, booths_tbl, tide_tbl, fc, seat, 
        diagnostics=fit["diag"], 
        draws_df=draws_df,
        booth_draws_df=booth_draws_df, 
        booth_party_draws=Pk * 100.0,  
        party_names=out_parties,               
        volume_names=kept_names, 
        volume_matrix=Vk, 
        prepared_df=df,
        attribution=attribution
    )

# ----------------------------------------------------------------------------- f(input) = output
def predict(booth_results: pd.DataFrame, state_results: Optional[pd.DataFrame],
            poll: Union[PollInput, pd.DataFrame],
            config: Optional[ModelConfig] = None,
            projected_formal_votes: Optional[pd.Series] = None,
            target_year: Optional[int] = None,
            trace: Optional[dict] = None) -> ModelOutput:
    cfg = config or ModelConfig()
    if state_results is None:
        state_results = STATE_RESULTS
    return predict_bayes(booth_results, state_results, poll, cfg, projected_formal_votes, target_year, trace)

# ============================================================================= BLACKBOX:  ENKI(x) -> y
# x = one dict (or a path to a .json file). EVERY variable in the system is a key in x; anything you leave out takes its
# default. Unknown keys are rejected (a typo must never silently fall back to a default). y = ENKIOutput.
#
#   DATA keys    booth_results, polls, state_results, projected_formal_votes, target_year
#                (DataFrame, or a path to a CSV; polls may instead be {"central":..,"low":..,"high":..})
#   MODEL keys   every field of ModelConfig (seat, parties, priors, MCMC, poll layer, ONP/BrianMode, electors, ...)
#   RUN keys     every field of RunOptions (what to run, where to write, plots)
#
# x may be flat, or grouped as {"data": {...}, "config": {...}, "run": {...}}; a flat key overrides a grouped one.
@dataclass
class RunOptions:
    run_forecast: bool = True                     # fit + forecast the live election
    run_backtest: bool = True                     # share-model backtest across past elections (one MCMC fit per target)
    backtest_targets: Optional[Sequence[int]] = None   # None: every election from the third onward
    run_volume_backtest: bool = False             # booth-volume backtest (slow; was never run by the old main)
    output_root: Optional[str] = None             # None: write NOTHING to disk. Otherwise <root>/<Seat>/<date>/ENKI/
    make_plots: bool = True                       # only if output_root is set
    print_report: bool = True                     # terminal summary at the end
    record_trace: bool = True                     # keep every process-step signal (y.trace, trace.npz) for ENKI_PFD_PDF.py
    plot_top_n_booths: int = 15
    plot_dpi: int = 300
    plot_volume_log_scale: bool = True
    plot_label_all_booths: bool = True
    plot_ridgeline_max_draws: int = 1500
    plot_ridgeline_keywords: Sequence[str] = ("PRE-POLL", "PREPOLL", "DECLARATION", "ABSENT", "POSTAL")


@dataclass
class ENKIOutput:
    forecast: Optional[ModelOutput] = None        # the full predict() result (params, booths, tide, forecast, seat, draws, ...)
    seat_summary: Optional[pd.DataFrame] = None   # seat-wide primary vote per party: mean + the scenario percentiles
    backtest: Optional[pd.DataFrame] = None
    backtest_draws: Optional[dict] = None
    volume_backtest: Optional[pd.DataFrame] = None
    config: Dict[str, Any] = field(default_factory=dict)     # the FULLY RESOLVED ModelConfig that was run (defaults filled in)
    run: Dict[str, Any] = field(default_factory=dict)        # the resolved RunOptions
    data_sources: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    output_dir: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)
    trace: Optional[dict] = None                  # signals at every block boundary (see ENKI_HELPERS.PIPELINE_STEPS)


_DATA_KEYS = ("booth_results", "polls", "state_results", "projected_formal_votes", "target_year")
_SEQ_FIELDS = ("parties", "early_keywords", "exclude_booths", "scenario_pct")


def _resolve(x):
    """x -> (data dict, ModelConfig, RunOptions, warnings, data_sources). Nothing is run."""
    flat = load_x(x)
    cfg_names = {f.name for f in fields(ModelConfig)}
    run_names = {f.name for f in fields(RunOptions)}
    known = cfg_names | run_names | set(_DATA_KEYS)
    unknown = sorted(set(flat) - known)
    if unknown:
        hints = []
        for k in unknown:
            m = difflib.get_close_matches(k, sorted(known), n=1)
            hints.append(f"{k!r}" + (f" (did you mean {m[0]!r}?)" if m else ""))
        raise KeyError("ENKI(x): unknown input key(s): " + ", ".join(hints))

    cfg_kw = {k: v for k, v in flat.items() if k in cfg_names}
    run_kw = {k: v for k, v in flat.items() if k in run_names}
    for k in _SEQ_FIELDS:
        if cfg_kw.get(k) is not None:
            cfg_kw[k] = tuple(cfg_kw[k])
    if cfg_kw.get("beta_priors"):
        cfg_kw["beta_priors"] = {p: tuple(v) for p, v in cfg_kw["beta_priors"].items()}
    if run_kw.get("plot_ridgeline_keywords") is not None:
        run_kw["plot_ridgeline_keywords"] = tuple(run_kw["plot_ridgeline_keywords"])
    cfg = ModelConfig(**cfg_kw)
    run = RunOptions(**run_kw)
    if cfg.prediction_date is None:
        cfg.prediction_date = date.today().strftime("%Y-%m-%d")      # recorded in the output so the run is reproducible
    warnings_ = validate_inputs(cfg, run)

    src: Dict[str, str] = {}
    if flat.get("booth_results") is None:
        raise KeyError("ENKI(x): 'booth_results' is required (DataFrame or CSV path).")
    data: Dict[str, Any] = {"booth_results": as_frame(flat["booth_results"], "booth_results")}
    src["booth_results"] = str(flat["booth_results"]) if isinstance(flat["booth_results"], (str, Path)) else "<in-memory>"

    if run.run_forecast:
        if flat.get("polls") is None:
            raise KeyError("ENKI(x): 'polls' is required to forecast (DataFrame, CSV path, or {'central','low','high'}).")
        pl = flat["polls"]
        if isinstance(pl, PollInput):
            data["polls"] = pl
        elif isinstance(pl, dict) and set(pl) == {"central", "low", "high"}:
            data["polls"] = PollInput(float(pl["central"]), float(pl["low"]), float(pl["high"]))
        else:
            data["polls"] = as_frame(pl, "polls")
        src["polls"] = str(pl) if isinstance(pl, (str, Path)) else "<in-memory>"

    if flat.get("state_results") is None:
        data["state_results"] = STATE_RESULTS.copy()
        src["state_results"] = "<built-in STATE_RESULTS>"
    else:
        data["state_results"] = as_frame(flat["state_results"], "state_results")
        src["state_results"] = str(flat["state_results"]) if isinstance(flat["state_results"], (str, Path)) else "<in-memory>"
    need = {"year"} | set(cfg.parties)
    absent = sorted(need - set(data["state_results"].columns))
    if absent:
        raise ValueError(f"state_results is missing columns: {absent}")

    pfv = flat.get("projected_formal_votes")
    if pfv is not None:
        data["projected_formal_votes"] = pfv if isinstance(pfv, pd.Series) else pd.Series(pfv, dtype=float)
        if cfg.registered_electors is not None:
            warnings_.append("projected_formal_votes AND registered_electors are both set: projected_formal_votes wins.")
    data["target_year"] = flat.get("target_year")
    return data, cfg, run, warnings_, src


def _seat_summary(res: ModelOutput, cfg: ModelConfig) -> pd.DataFrame:
    lo, mid, hi = (float(v) for v in cfg.scenario_pct)
    q = res.draws_df.quantile([lo / 100, mid / 100, hi / 100]).T
    q.columns = [f"p{lo:g}", f"p{mid:g}", f"p{hi:g}"]
    q.insert(0, "mean", res.draws_df.mean())
    return q

def ENKI(x) -> ENKIOutput:
    """The blackbox. ENKI(x) -> y.  See the header above for what x may contain."""
    data, cfg, run, warnings_, sources = _resolve(x)
    y = ENKIOutput(config=json.loads(json.dumps(asdict(cfg), default=jsonable)),
                run=json.loads(json.dumps(asdict(run), default=jsonable)),
                data_sources=sources, warnings=warnings_)
    booth_df, state_df = data["booth_results"], data["state_results"]

    out_dir: Optional[Path] = None
    seat_folder: Optional[Path] = None
    if run.output_root:
        seat_folder = output_folder(run.output_root, cfg.seat_name, cfg.prediction_date)
        out_dir = seat_folder / "ENKI"
        out_dir.mkdir(parents=True, exist_ok=True)
        y.output_dir = str(out_dir)

    def _note(name: str, path: Path) -> None:
        y.files[name] = str(path)

    if run.run_backtest:
        print("==================================================")
        print(" RUNNING HISTORICAL BACKTEST")
        print("==================================================")
        y.backtest, y.backtest_draws = backtest_bayes(booth_df, state_df, cfg, targets=run.backtest_targets, return_draws=True)
        if out_dir is not None:
            y.backtest.to_csv(out_dir / "backtest.csv", index=False)
            _note("backtest", out_dir / "backtest.csv")
            save_backtest_draws(y.backtest_draws, out_dir)                # hand-off to INANNA
            if run.make_plots:
                try:
                    save_backtest_table_png(y.backtest, out_dir / "backtest_summary_table.png", dpi=run.plot_dpi)
                    _note("backtest_table_png", out_dir / "backtest_summary_table.png")
                except Exception as e:
                    y.warnings.append(f"backtest table plot failed: {e!r}")

    if run.run_volume_backtest:
        y.volume_backtest = backtest_volume(booth_df, cfg, targets=run.backtest_targets)
        if out_dir is not None:
            y.volume_backtest.to_csv(out_dir / "backtest_volume.csv", index=False)
            _note("backtest_volume", out_dir / "backtest_volume.csv")

    if run.run_forecast:
        print("\n==================================================")
        print(f" RUNNING FORECAST: {cfg.seat_name}  (target {cfg.election_date})")
        print("==================================================")
        trace = {} if run.record_trace else None
        res = predict(booth_results=booth_df, state_results=state_df, poll=data["polls"], config=cfg,
                      projected_formal_votes=data.get("projected_formal_votes"), target_year=data.get("target_year"), trace=trace)
        y.forecast = res
        y.trace = trace
        if trace is not None:
            y.warnings += check_trace(trace)
        y.seat_summary = _seat_summary(res, cfg)
        if out_dir is not None:
            if run.make_plots:
                plots = {
                    "seat_distribution.png": lambda p: generate_seat_distribution_plot(res, output_path=p, dpi=run.plot_dpi),
                    "booth_swing_scatter.png": lambda p: generate_booth_swing_scatter_plot(res, output_path=p, label_all=run.plot_label_all_booths, dpi=run.plot_dpi),
                    "ridgeline.png": lambda p: generate_booth_ridgeline_plot(res_model_output=res, target_party=cfg.party, top_n_booths=run.plot_top_n_booths,
                                                                              output_path=p, max_draws=run.plot_ridgeline_max_draws, dpi=run.plot_dpi,
                                                                              special_keywords=run.plot_ridgeline_keywords),
                    "booth_volume_ridgeline.png": lambda p: generate_booth_volume_ridgeline(booths=res.volume_names, forecasted_matrix=res.volume_matrix,
                                                                                            df_historical=res.prepared_df, output_path=p, top_n=run.plot_top_n_booths,
                                                                                            log_scale=run.plot_volume_log_scale, dpi=run.plot_dpi),
                }
                for fname, fn in plots.items():
                    try:
                        fn(out_dir / fname)
                        _note(fname, out_dir / fname)
                    except Exception as e:                                # a failed picture must never cost the forecast
                        y.warnings.append(f"plot {fname} failed: {e!r}")
            res.save(str(out_dir))
            if trace:
                save_trace(out_dir, trace)
                _note("trace", out_dir / "trace.npz")
            save_actual_primaries(res.prepared_df, cfg.parties, out_dir)   # hand-off to INANNA
            save_run_meta(out_dir, cfg, sources.get("booth_results", ""), sources.get("polls", ""))
            y.seat_summary.to_csv(out_dir / "seat_primary_summary.csv")
            append_history(res, seat_folder, cfg.prediction_date)
            _note("forecast_dir", out_dir)

    if out_dir is not None:
        (out_dir / "inputs_resolved.json").write_text(
            json.dumps({"config": y.config, "run": y.run, "data_sources": y.data_sources}, indent=2, default=jsonable), encoding="utf-8")
        _note("inputs_resolved", out_dir / "inputs_resolved.json")
    if run.print_report:
        print_run_report(y, cfg, beta_prior_table(cfg))
        if out_dir is not None:
            print(f"\nFinished: seat '{cfg.seat_name}'. All outputs written to: {out_dir}/")
    return y


# ----------------------------------------------------------------------------- CLI
def main(seat_name: str = "sample", booth_data: str = "sample_booth_results.csv", polls_data: str = "sample_polls.csv",
        output_root: str = "forecast_output", statewide_historical: Optional[pd.DataFrame] = None,
         registered_voters: Optional[int] = 62169, expected_turnout: float = 0.91, brian_mode: bool = True, **overrides) -> ENKIOutput:
    """Backward-compatible entry point: the old keyword arguments map onto x, and ANY other key of x can be passed as a keyword."""
    x = dict(seat_name=seat_name, booth_results=booth_data, polls=polls_data, output_root=output_root,
            state_results=statewide_historical, registered_electors=registered_voters,
            expected_turnout_rate=expected_turnout, brian_mode=brian_mode)
    x.update(overrides)
    return ENKI(x)


def _cli(argv: Sequence[str]) -> ENKIOutput:
    """python ENKI_V2.py                       -> legacy defaults (sample files)
    python ENKI_V2.py input.json            -> ENKI(input.json)
    python ENKI_V2.py input.json draws=500 poll_house_sd=2.5 brian_mode=false     (key=value overrides, values parsed as JSON)"""
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
        return main(**over)
    x = load_x(path)
    x.update(over)
    return ENKI(x)


if __name__ == "__main__":
    _cli(sys.argv[1:])