'''CONSTRUCTS REALITY, AND THE BASICS. THE WINNING/LOSING AND STRATEGY COME LATER.
ENKI IS THE GOD OF WISDOM WHO CREATES THE FUNDAMENTAL LAWS OF THE UNIVERSE, THE BASELINE REALITY.
IN THIS CONTEXT, THIS SCRIPT CREATES THE BASELINE OF REALITY FOR THIS ENTIRE MODEL'''
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

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
    sigma_alpha_sd: float = 0.4                   # HalfNormal scale, booth baseline spread (data: ~0.2-0.36 by party)
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
    poll_house_sd: float = 3.25                    # extra poll error in points (2023 poll miss: 0.84)
    d_persist: Optional[float] = None             # None (default): rho_d ~ Beta(2,2) is ESTIMATED inside the fit (AR(1) tide, shared by all parties)
    include_eps: bool = True                      # booth-level local-shift noise in predictive draws
    sigma_d_median: float = 0.30                  # LogNormal prior on sigma_d (log-ratio units). With beta pinned near 1,
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
    sigma_eps_sd: float = 0.3                     # HalfNormal scale of booth-election noise (was hard-coded); lower it if booth coverage sits well above 90%
    run_attribution: bool = True                  # report which uncertainty layer drives the width of the seat band
    #SET BELOW TO 1.4 FOR REGIONAL/OUTER METRO SEATS, LOWER FOR LOWER ONE NATION SHARE.
    onp_seat_ratio_median: float = 0.65            # seat ONP share / statewide ONP share (East Hills 1999: 10.7 / 6.4)
    onp_seat_ratio_sdlog: float = 0.5
    onp_max_seat_share: float = 1.0             # cap on the seat-wide ONP share
    
    #NOTE: Below is to distribute KNOWN number of electors (since now I know!)
    #Note number of electors does not equal the number of voters.
    #Turnout matters.
    registered_electors: Optional[float] = None       # Total registered electors from NSWEC
    expected_turnout_rate: float = 0.90               # Expected voter turnout rate (e.g. 90%)
    
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


def combine_parties(polls: pd.DataFrame, groups: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """e.g. combine_parties(polls, {"OTHER": ["SFF", "ONP", "OTH", "UND"]})"""
    p = polls.copy()
    for new, cols in groups.items():
        p[new] = p[[c for c in cols if c in p.columns]].fillna(0).sum(axis=1)
    return p

def projected_votes_from_electors(booth_results: pd.DataFrame, cfg: ModelConfig, 
                                 registered_electors: float, 
                                 expected_turnout_rate: float = 0.90) -> pd.Series:

    df = prepare_bayes_data(booth_results, cfg)
    
    # 1. Infer the historical informal rate directly from past data across the seat
    total_votes_all = booth_results["total_votes"].sum()
    informal_votes_all = booth_results["informal_votes"].sum()
    historical_informal_rate = informal_votes_all / total_votes_all if total_votes_all > 0 else 0.04
    
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
def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def _check_booth_data(df: pd.DataFrame, cfg: ModelConfig) -> None:
    """Warn (never raise) when the booth CSV does not reconcile. Rows are compared as formal votes
    (total - informal) vs the sum of every named column (ALP/LNP/GRN/IND/ONP). Only runs when an IND column exists,
    because without one OTHER is the remainder by design. Also flags duplicate booth-years and big year-on-year jumps in the seat total."""
    if not cfg.verbose:
        return
    w = df.copy()
    for c in ("total_votes", "informal_votes", "IND", "ONP", "ALP", "LNP", "GRN"):
        if c in w.columns:
            w[c] = pd.to_numeric(w[c].astype(str).str.replace(",", ""), errors="coerce")
    w["_n"] = w["total_votes"] - w["informal_votes"]
    issues = []
    if "IND" in w.columns:
        named = [c for c in ("ALP", "LNP", "GRN", "IND", "ONP") if c in w.columns]
        w["_gap"] = w["_n"] - w[named].sum(axis=1)
        off = w[w["_gap"] != 0]
        if len(off):
            worst = off.loc[off["_gap"].abs().idxmax()]
            over = int((off["_gap"] < 0).sum())
            issues.append(f"{len(off)} of {len(w)} rows: named parties do not sum to formal votes "
                        f"({over} rows have MORE named votes than formal). Worst: {worst['booth']} {int(worst['year'])}, "
                        f"gap {int(worst['_gap']):+d} votes. The gap is silently dumped into OTHER.")
    dup = int(w.duplicated(["year", "booth"]).sum())
    if dup:
        issues.append(f"{dup} duplicate booth-year row(s); they are summed together, which double counts if they are accidental repeats.")
    tot = w.groupby("year")["_n"].sum().sort_index()
    jump = tot.pct_change().dropna()
    for yr, ch in jump[jump.abs() > 0.15].items():
        issues.append(f"Seat formal total moves {100 * ch:+.0f}% into {int(yr)} ({int(tot.loc[yr]):,} votes); "
                      "check for missing booths or vote types (postal, iVote, absent).")
    if issues:
        print("\n!! DATA CHECK: booth CSV does not fully reconcile !!")
        for m in issues:
            print("   - " + m)
    else:
        print("-> Data check: booth CSV reconciles (rows sum to formal votes, no duplicates, no big total jumps).")


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
    _check_booth_data(df, cfg)
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
    mu, sd = (1.0, 0.15) if uniform else (0.3, 0.30)
    if cfg.beta_g is not None:
        mu = cfg.beta_g
    if cfg.beta_prior_sd is not None:
        sd = cfg.beta_prior_sd
    table = {p: (mu, sd, 0.0) for p in cfg.parties}
    if "GRN" in table and not uniform and cfg.beta_prior_sd is None:
        table["GRN"] = (mu, 0.50, 0.0)               # small share, tiny state movement: weakly informative
    if "OTHER" in table:
        table["OTHER"] = (0.0, 0.30, None)
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


def _diagnose(idata, summ, label: str) -> int:
    divergences = int(idata.sample_stats["diverging"].sum())
    if divergences > 0 or (summ["r_hat"] > 1.01).any() or (summ["ess_bulk"] < 400).any():
        print(f"WARNING [{label}]: sampler diagnostics look poor (divergences={divergences}, "
            f"max r_hat={summ['r_hat'].max():.3f}, min ESS={summ['ess_bulk'].min():.0f}). "
            "Do not trust the forecast until this is resolved.")
    return divergences




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

    with pm.Model(coords=coords):
        betas = []
        for p in parties:
            mu_b, sd_b, _lower_ignored = priors[p]                              # plain Normal: no truncation at 0
            betas.append(pm.Normal(f"beta_{p}", mu=mu_b, sigma=sd_b))
        beta_vec = pt.stack(betas)                                              # (K,)
        m = pm.Normal("m", mu=m0, sigma=1.0, dims="pm")
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=cfg.sigma_alpha_sd / np.sqrt(2.0), dims="party")   # /sqrt2: log-ratio scale unchanged
        sigma_d = pm.LogNormal("sigma_d", mu=float(np.log(cfg.sigma_d_median) - 0.5 * np.log(2.0)), sigma=cfg.sigma_d_sdlog, dims="party")
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=cfg.sigma_eps_sd / np.sqrt(2.0), dims="party")
        alpha_z = pm.Normal("alpha_z", 0.0, 1.0, dims=("booth", "party"))          # non-centred
        eps_z = pm.Normal("eps_z", 0.0, 1.0, dims=("obs", "party"))                # non-centred
        alpha_b = pm.Deterministic("alpha_b", alpha_z * sigma_alpha, dims=("booth", "party"))   # (B, K) all parties
        if est_rho:
            rho = pm.Beta("rho_d", alpha=2.0, beta=2.0)                         # shared persistence of the seat tide
            innov = pt.sqrt(pt.maximum(1.0 - rho ** 2, 1e-6))
        else:
            rho = float(cfg.d_persist)
            innov = float(np.sqrt(max(1.0 - rho ** 2, 1e-6)))
        d_z = pm.Normal("d_z", 0.0, 1.0, dims=("year", "party"))                # non-centred AR(1) innovations
        d_steps = [sigma_d * d_z[0]]                                            # stationary start: sd = sigma_d
        for t in range(1, T):
            d_steps.append(rho * d_steps[-1] + innov * sigma_d * d_z[t])
        d_t = pm.Deterministic("d_t", pt.stack(d_steps, axis=0), dims=("year", "party"))   # (T, K), ALL parties
        eps = pm.Deterministic("eps", eps_z * sigma_eps, dims=("obs", "party"))             # (N, K) all parties
        bx = beta_vec[None, :] * des["x"][des["year_idx"]]                      # (N, K) state-swing effect
        d_obs = d_t[des["year_idx"]]                                            # (N, K)
        m_full = pt.concatenate([m, pt.zeros((1,))])                            # reference party: m_K = 0
        eta = m_full[None, :] + alpha_b[des["booth_idx"]] + bx + d_obs + eps    # (N, K): every party treated alike
        pm.Multinomial("y_obs", n=n, p=pm.math.softmax(eta, axis=-1), observed=Y)
        idata = pm.sample(draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, target_accept=cfg.target_accept,
                        random_seed=cfg.seed, return_inferencedata=True)

    summ = az.summary(idata, var_names=[f"beta_{p}" for p in parties] + ["m", "sigma_alpha", "sigma_d", "sigma_eps"]
                    + (["rho_d"] if est_rho else []))
    divergences = _diagnose(idata, summ, "share model")
    if divergences > 0:
        dv = idata.sample_stats["diverging"].values.reshape(-1).astype(bool)
        sd = idata.posterior["sigma_d"].values.reshape(dv.size, -1)
        print(f"  mean sigma_d in divergent draws {sd[dv].mean():.3f} vs {sd.mean():.3f} overall "
            "(clustered at small values => tighten sigma_d_median / sigma_d_sdlog).")
    des.update(
        draws={"beta": np.stack([_stack(idata, f"beta_{p}") for p in parties], axis=1),     # (D, K)
            "m": _stack(idata, "m"), "sigma_alpha": _stack(idata, "sigma_alpha"),
            "sigma_d": _stack(idata, "sigma_d"), "sigma_eps": _stack(idata, "sigma_eps"),
            "alpha": _stack(idata, "alpha_b"), "d": _stack(idata, "d_t"), "eps": _stack(idata, "eps"),
            "rho": (_stack(idata, "rho_d") if est_rho
                    else np.full(_stack(idata, "sigma_d").shape[0], float(cfg.d_persist)))},
        diag=summ.reset_index().rename(columns={"index": "parameter"}), divergences=divergences, idata=idata)
    return des


def _simulate_next(fit: dict, S_next_pct: np.ndarray, cfg: ModelConfig, rng: np.random.Generator, shut: Sequence[str] = ()):
    """One next-election outcome per posterior draw. S_next_pct is (draws x parties), rows ~ sum to 100.
    The seat tide of EVERY party (reference included) follows the fitted AR(1): d_next = rho*d_last + sqrt(1-rho^2)*sigma_d*z, with rho
    the posterior draw (or the fixed cfg.d_persist). Returns p (draws x booths x parties, rows sum to 1) and the tide
    (draws x non-reference parties) = [beta_k x_k + d_k] - [beta_ref x_ref + d_ref], i.e. the log-ratio shift vs the reference party."""
    dr = fit["draws"]
    if "params" in shut:                                              # diagnostic: collapse the posterior to its mean (no parameter uncertainty)
        dr = {k: np.broadcast_to(v.mean(axis=0, keepdims=True), v.shape) for k, v in dr.items()}
    D, B, K = dr["alpha"].shape
    Km = K - 1
    S = np.clip(np.asarray(S_next_pct, dtype=float), 1e-3, None)
    S = S / 100.0 if cfg.onp_state_denominator else S / S.sum(axis=1, keepdims=True)   # same footing as _state_matrix
    x_next = np.log(S) - fit["x_center"]                              # (D, K)
    bx = dr["beta"] * x_next                                          # (D, K)
    rho = (np.asarray(dr["rho"], dtype=float) if "rho" in dr else np.full(D, float(cfg.d_persist or 0.0)))[:, None]   # (D, 1)
    d_next = rho * dr["d"][:, -1, :] + np.sqrt(np.maximum(1.0 - rho ** 2, 0.0)) * dr["sigma_d"] * rng.standard_normal((D, K)) * (0.0 if "tide" in shut else 1.0)
    lin = bx + d_next                                                 # (D, K) party-level shift on the logit scale
    m_full = np.concatenate([dr["m"], np.zeros((D, 1))], axis=1)      # reference party: m_K = 0
    eta = m_full[:, None, :] + dr["alpha"] + lin[:, None, :]
    if cfg.include_eps:
        eta = eta + dr["sigma_eps"][:, None, :] * rng.standard_normal((D, B, K)) * (0.0 if "booth" in shut else 1.0)
    eta = eta - eta.max(axis=2, keepdims=True)
    e = np.exp(eta)
    return e / e.sum(axis=2, keepdims=True), lin[:, :Km] - lin[:, Km:]


def _poll_draws(poll: Union[PollInput, pd.DataFrame], cfg: ModelConfig, state_df: pd.DataFrame,
                D: int, rng: np.random.Generator):
    """Returns (S_next, central, target, se_pct, onp_state).
    S_next is EX-ONP (rows sum to 100); onp_state is (D,) statewide ONP % draws, or None without an ONP column."""
    parties = list(cfg.parties)
    i = parties.index(cfg.party)
    onp_central = None

    if isinstance(poll, pd.DataFrame):
        miss = [p for p in parties if p not in poll.columns]
        if miss:
            raise ValueError(f"Poll table needs a column for every modelled party; missing {miss}")
        summ = {p: poll_summary(poll, p, cfg.poll_half_life_months, cfg.poll_as_of, cfg.poll_moe) for p in parties}
        central = np.array([summ[p].central for p in parties], dtype=float)
        target = summ[cfg.party]
        if not cfg.onp_layer and not cfg.onp_state_denominator and "ONP" in poll.columns and "ONP" not in parties:
            try:
                onp_c = float(poll_summary(poll, "ONP", cfg.poll_half_life_months, cfg.poll_as_of, cfg.poll_moe).central)
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
                onp_central = float(poll_summary(poll, "ONP", cfg.poll_half_life_months, cfg.poll_as_of, cfg.poll_moe).central)
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
        if full.sum() > 102.0 and cfg.verbose:
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
    se_poll = (target.high - target.low) / (2 * 1.96)

    h_sd = get_time_horizon_sd(cfg.prediction_date, cfg.election_date)
    se = np.sqrt(se_poll ** 2 + cfg.poll_house_sd ** 2 + h_sd ** 2) / 100.0
    p_t = central[i] / 100.0
    conc = max(p_t * (1 - p_t) / se ** 2 - 1.0, 10.0)

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
        s_a = pm.HalfNormal("sigma_a", sigma=1.5)
        a_z = pm.Normal("a_z", 0.0, 1.0, dims="booth")
        a = pm.Deterministic("a", a_z * s_a, dims="booth")
        if has_free:
            tau_free = pm.Normal("tau_free", 0.0, 0.5, shape=n_free)
            tau = pm.Deterministic("tau", pt.concatenate([pt.zeros(t0 + 1), tau_free]))
        else:
            tau = pt.zeros(T)
        kappa = pm.Gamma("kappa", alpha=2.0, beta=0.005)                 # concentration (mean 400)
        for ti, yr in enumerate(years):
            idx = np.where(present[ti])[0]
            pi = pm.math.softmax(a[idx] + tau[ti] * btype[idx])
            pm.DirichletMultinomial(f"n_{yr}", n=int(C[ti, idx].sum()), a=kappa * pi,
                                    observed=C[ti, idx].astype(int))
        idata = pm.sample(draws=cfg.draws, tune=max(cfg.tune, 3000), chains=cfg.chains,
                        target_accept=cfg.volume_target_accept, random_seed=cfg.seed,
                        return_inferencedata=True, progressbar=False)
    vars_ = ["sigma_a", "kappa"] + (["tau"] if has_free else [])
    summ = az.summary(idata, var_names=vars_)
    div = _diagnose(idata, summ, "volume model")
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
    t_year = target_year if target_year is not None else years[-1] + 4
    if t_year in years:                                               # in-sample election
        tau_t = tau[:, years.index(t_year)]
        Tn = np.full(D, totals[years.index(t_year)])
    else:
        k = max(1, int(round((t_year - years[-1]) / 4.0)))
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


def backtest_volume(booth_results: pd.DataFrame, config: Optional[ModelConfig] = None,
                    targets: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Predict each election's booth volumes from earlier elections only (slow: one fit per target).
    Reports booth and total errors, vs no-change, and the early-vote share of the total."""
    cfg = config or ModelConfig()
    df = prepare_bayes_data(booth_results, cfg)
    years = sorted(df["year"].unique())
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for tgt in (targets or years[2:]):
        train = df[df["year"] < tgt]
        vfit = _fit_volume_posterior(train, cfg)
        last_y = int(train["year"].max())
        prev_all = train[train["year"] == last_y].set_index("booth")["n"]
        active = np.array([b in prev_all.index for b in vfit["names"]])
        V = _simulate_volumes(vfit, 4000, cfg, rng, target_year=tgt, active=active)
        t = df[df["year"] == tgt].set_index("booth")["n"]
        idx = [i for i, b in enumerate(vfit["names"]) if active[i] and b in t.index]
        names = [vfit["names"][i] for i in idx]
        act = t.reindex(names).values.astype(float)
        P = V[:, idx]
        med = np.median(P, axis=0)
        lo, hi = np.percentile(P, [5, 95], axis=0)
        prev = prev_all.reindex(names).values.astype(float)
        tot = V.sum(axis=1)
        early = vfit["btype"].astype(bool)
        e_pred = np.median(V[:, early].sum(axis=1) / np.maximum(tot, 1)) if early.any() else np.nan
        e_act = (df[(df["year"] == tgt) & (df["btype"] == 1)]["n"].sum() / df[df["year"] == tgt]["n"].sum()) if early.any() else np.nan
        rows.append({"target": int(tgt), "n_booths": len(names),
                    "booth_MAPE_pct": float(np.mean(np.abs(med - act) / act) * 100),
                    "nochange_MAPE_pct": float(np.mean(np.abs(prev - act) / act) * 100),
                    "booth_cover_90": float(((act >= lo) & (act <= hi)).mean()),
                    "total_pred_median": float(np.median(tot)), "total_actual": float(df[df["year"] == tgt]["n"].sum()),
                    "total_error_pct": float((np.median(tot) / df[df["year"] == tgt]["n"].sum() - 1) * 100),
                    "total_in_90": bool(np.percentile(tot, 5) <= df[df["year"] == tgt]["n"].sum() <= np.percentile(tot, 95)),
                    "early_share_pred": float(e_pred), "early_share_actual": float(e_act),
                    "divergences": vfit["divergences"]})
    return pd.DataFrame(rows)

def _apply_onp_entry(p_next: np.ndarray, onp_state_pct: np.ndarray, cfg: ModelConfig, rng: np.random.Generator):
    """One Nation as an ENTRANT: seat ONP share s = kappa * statewide ONP share (kappa lognormal), capped; the other
    parties keep their modelled shares of the remaining (1 - s). Returns p_out (D x B x (K+1), last = ONP) and s."""
    D, B, _ = p_next.shape
    kappa = np.exp(np.log(cfg.onp_seat_ratio_median) + cfg.onp_seat_ratio_sdlog * rng.standard_normal(D))
    s = np.clip(kappa * np.asarray(onp_state_pct, dtype=float) / 100.0, 0.0, cfg.onp_max_seat_share)
    p_out = np.concatenate([p_next * (1.0 - s)[:, None, None], np.broadcast_to(s[:, None, None], (D, B, 1))], axis=2)
    return p_out, s

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
        rng = np.random.default_rng(cfg.seed + 101)
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
                target_year: Optional[int] = None) -> ModelOutput:
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

    S_next, central, target_poll, se_used, onp_state = _poll_draws(poll, cfg, state_results, D, rng)
    p_next, tide = _simulate_next(fit, S_next, cfg, rng)               # fitted model, ONP excluded
    out_parties, onp_seat = list(parties), None
    if onp_state is not None and cfg.onp_layer:               # ONP only enters the SEAT when explicitly switched on
        p_next, onp_seat = _apply_onp_entry(p_next, onp_state, cfg, rng)
        out_parties = parties + ["ONP"]

    # booth volumes: modelled (total x booth shares) unless projected votes or registered electors are supplied
    last_year = int(df["year"].max())
    last_n = df[df["year"] == last_year].set_index("booth")["n"].reindex(names)
    vfit = None
    if projected_formal_votes is not None:
        w = projected_formal_votes.reindex(names).fillna(last_n).fillna(0.0).values.astype(float)
        V = np.tile(w, (D, 1))
    elif cfg.registered_electors is not None:
        w = projected_votes_from_electors(booths, cfg)
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

    attribution = (uncertainty_attribution(fit, S_next, central, V, keep, k0, cfg, onp_seat) if cfg.run_attribution else None)

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
        
    if vfit is not None:
        params["volume_divergences"] = vfit["divergences"]
        params["volume_rhat_max"] = float(vfit["diag"]["r_hat"].max())

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

# ----------------------------------------------------------------------------- backtest (share model)
def backtest_bayes(booth_results: pd.DataFrame, state_results: Optional[pd.DataFrame] = None,
                config: Optional[ModelConfig] = None, targets: Optional[Sequence[int]] = None,
                return_draws: bool = False):
    """Refit on elections BEFORE each target; predict it using only the ACTUAL statewide shares (all parties) and
    the ACTUAL booth volumes, so this isolates the share model. Predictive booth shares add binomial noise.
    Also reports no-change and statewide-swing benchmarks for the target party. One MCMC fit per target.
    return_draws=True returns (metrics_df, draws_by_year): for each target year, the predicted booth x party primary
    draws (in %), the booth names and the ACTUAL votes, so INANNA can push them through the preference flows."""
    from dataclasses import replace as _replace
    cfg = _replace(config or ModelConfig(), poll_house_sd=0.0, verbose=False)
    state_results = STATE_RESULTS if state_results is None else state_results
    df_all = prepare_bayes_data(booth_results, cfg)
    years = sorted(df_all["year"].unique())
    parties = list(cfg.parties)
    k0 = parties.index(cfg.party)
    rng = np.random.default_rng(cfg.seed)
    lo_p, _, hi_p = (float(x) for x in cfg.scenario_pct)
    rows = []
    draws_out = {}
    for tgt in (targets or years[2:]):
        fit = fit_bayesian_model(df_all[df_all["year"] < tgt], cfg, state_results)
        D = len(fit["draws"]["beta"])
        S_all = _state_matrix(state_results, parties, years, cfg.onp_state_denominator)
        S_t = S_all.loc[tgt].values
        p_next, _ = _simulate_next(fit, np.tile(S_t, (D, 1)), cfg, rng)
        t = df_all[(df_all["year"] == tgt) & df_all["booth"].isin(fit["booth_names"])]
        col = [fit["booth_names"].index(b) for b in t["booth"]]
        n_t, y_t = t["n"].values, t["y"].values
        if return_draws:
            draws_out[int(tgt)] = {"booths": [str(b) for b in t["booth"]], "parties": parties,
                                "booth_party_draws": 100.0 * p_next[:, col, :],          # (D, B_t, K) in %, rows sum to 100
                                "actual_party_votes": np.column_stack([t[f"y_{p}"].values for p in parties]),
                                "actual_n": n_t.astype(float)}
        P = 100.0 * p_next[:, col, k0]
        med, act = np.median(P, axis=0), 100.0 * y_t / n_t
        rep = 100.0 * rng.binomial(n_t[None, :], p_next[:, col, k0]) / n_t[None, :]
        lo, hi = np.percentile(rep, [lo_p, hi_p], axis=0)
        seat = (P * n_t).sum(axis=1) / n_t.sum()
        seat_act = float((act * n_t).sum() / n_t.sum())
        s_lo, s50, s_hi = np.percentile(seat, [lo_p, 50, hi_p])
        err = med - act
        row = {"target": int(tgt), "seat_pred_median": s50, "seat_actual": seat_act, "seat_error": s50 - seat_act,
            "seat_lo": s_lo, "seat_hi": s_hi, "seat_in_interval": bool(s_lo <= seat_act <= s_hi),
               "booth_MAE": float(np.abs(err).mean()), "booth_RMSE": float(np.sqrt((err ** 2).mean())),
            "booth_cover": float(((act >= lo) & (act <= hi)).mean()), "n_booths": len(t),
            "divergences": fit["divergences"], "rhat_max": float(fit["diag"]["r_hat"].max())}
        for j, p in enumerate(parties):                                 # every party's seat-wide error
            yj = t[f"y_{p}"].values
            sj = (100.0 * p_next[:, col, j] * n_t).sum(axis=1) / n_t.sum()
            row[f"seat_err_{p}"] = float(np.median(sj) - 100.0 * yj.sum() / n_t.sum())
        last_y = max(int(y) for y in years if y < tgt)
        last = df_all[df_all["year"] == last_y].set_index("booth")["share"]
        S_last = float(S_all.loc[last_y, cfg.party])
        mk = np.array([b in last.index for b in t["booth"]])
        row["common_n"] = int(mk.sum())
        if mk.any():
            prev = last.reindex(t["booth"].values[mk]).values
            a_c, n_c, m_c = act[mk], n_t[mk], med[mk]
            wavg = lambda v: float((v * n_c).sum() / n_c.sum())
            e_md, e_nc, e_sw = m_c - a_c, prev - a_c, prev + (float(S_all.loc[tgt, cfg.party]) - S_last) - a_c
            row.update({"model_MAE_common": float(np.abs(e_md).mean()), "nochange_MAE": float(np.abs(e_nc).mean()),
                        "statewide_swing_MAE": float(np.abs(e_sw).mean()), "model_seat_err_common": wavg(e_md),
                        "nochange_seat_err": wavg(e_nc), "statewide_swing_seat_err": wavg(e_sw)})
        rows.append(row)
    return (pd.DataFrame(rows), draws_out) if return_draws else pd.DataFrame(rows)


# ----------------------------------------------------------------------------- f(input) = output
def predict(booth_results: pd.DataFrame, state_results: Optional[pd.DataFrame],
            poll: Union[PollInput, pd.DataFrame],
            config: Optional[ModelConfig] = None,
            projected_formal_votes: Optional[pd.Series] = None,
            target_year: Optional[int] = None) -> ModelOutput:
    cfg = config or ModelConfig()
    if state_results is None:
        state_results = STATE_RESULTS
    return predict_bayes(booth_results, state_results, poll, cfg, projected_formal_votes, target_year)

# ----------------------------------------------------------------------------- output folders
def output_folder(base: str, seat_name: str, pred_date: Optional[str] = None) -> Path:
    """<base>/<Seat_Name>/<YYYY-MM-DD>/   (date only, no time)"""
    d = pred_date or date.today().isoformat()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", seat_name).strip("_") or "Seat"
    return Path(base) / slug / d


def append_history(res: ModelOutput, seat_folder: Path, pred_date: str) -> Path:
    row = {
        "date": pred_date, 
        "S_next": res.params.get("S_next", np.nan), 
        "S_next_low": res.params.get("S_next_low", np.nan),
        "S_next_high": res.params.get("S_next_high", np.nan), 
        "beta": res.params.get("beta", res.params.get("p_g", np.nan)),
        "d_bar": res.params.get("d_bar", 0.0)
    }
    
    for _, r in res.seat.iterrows():
        scenario = r["scenario"]
        share_val = r.get("primary_share_pct", r.get("seat_share_pct", r.get("seat_share", np.nan)))
        row[f"seat_{scenario}"] = share_val
        
    path = seat_folder / "history.csv"
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["date"] != pred_date], new], ignore_index=True).sort_values("date")
        
    new.to_csv(path, index=False)
    return path


#----------------------------------------------------------------------------- PLOTTERS
def generate_booth_ridgeline_plot(res_model_output, target_party="ALP", top_n_booths=15,
                                  output_path="booth_distributions.png", max_draws=1500, **kwargs):
    """Ridgeline of the model's ACTUAL booth-level predictive draws (res.booth_draws_df); booths with
    pre-poll / declaration style names are always included, the rest are the largest by expected volume."""
    output_path = Path(output_path)
    draws = res_model_output.booth_draws_df
    fc = res_model_output.forecast
    if draws is None or draws.empty:
        raise ValueError("No booth-level draws on the result (res.booth_draws_df); re-run predict().")
    if len(draws) > max_draws:
        draws = draws.sample(max_draws, random_state=0)
    special = [b for b in draws.columns if any(k in str(b).upper() for k in ("PRE-POLL", "PREPOLL", "DECLARATION", "ABSENT", "POSTAL"))]
    regular = fc[~fc["booth"].isin(special)].sort_values("formal_weight", ascending=False)["booth"].tolist()
    chosen = special + regular[: max(5, top_n_booths - len(special))]
    order = sorted(chosen, key=lambda b: draws[b].mean())
    df_long = draws[order].melt(var_name="booth", value_name="vote_share")
    df_long["booth"] = pd.Categorical(df_long["booth"], categories=order, ordered=True)
    min_val = max(0.0, df_long["vote_share"].min() - 1.0)
    max_val = min(100.0, df_long["vote_share"].max() + 1.0)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    palette = sns.color_palette("plasma", n_colors=len(order))
    g = sns.FacetGrid(df_long, row="booth", hue="booth", aspect=12, height=0.55, palette=palette)
    g.map(sns.kdeplot, "vote_share", clip_on=False, fill=True, alpha=0.7, lw=1.5, bw_adjust=0.6)
    g.map(sns.kdeplot, "vote_share", clip_on=False, color="black", lw=1, bw_adjust=0.6)
    g.map(plt.axhline, y=0, lw=1.5, color="black", clip_on=False)

    def annotate_facet(x, color, label):
        ax = plt.gca()
        ax.text(-0.03, 0.2, str(order[list(g.axes.flatten()).index(ax)]), fontweight="bold", color="black",
                ha="right", va="center", transform=ax.transAxes, fontsize=8.5)

    g.map(annotate_facet, "vote_share")
    g.figure.subplots_adjust(hspace=-0.35)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    ax_bottom = g.axes[-1, 0]
    ax_bottom.set_xlim(min_val, max_val)
    ax_bottom.set_xlabel(f"Forecasted {target_party} Primary Vote Share (%)", fontsize=11, fontweight="bold", labelpad=10)
    g.figure.suptitle(f"ENKI Monte Carlo {target_party} Vote Share Distribution (model draws)", fontsize=13, fontweight="bold", y=0.98)
    g.figure.subplots_adjust(left=0.32, right=0.95, top=0.92, bottom=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Ridgeline plot saved to: {output_path}")


def generate_seat_distribution_plot(res_model_output, output_path="seat_distribution.png"):
    """
    Plots primary vote probability density curves for ALL parties in forecast_draws.csv.
    Applies Worst/Average/Best reference lines EXCLUSIVELY to ALP.
    """
    if res_model_output.draws_df is not None:
        draws_df = res_model_output.draws_df
    else:
        draws_df = pd.DataFrame()

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(11, 6))

    party_colors = {
        "ALP": "#E13940",   # Labor Red
        "LNP": "#0047AB",   # Coalition Blue
        "LIB": "#0047AB",
        "NAT": "#0066CC",
        "GRN": "#104C34",   # Greens Green
        "ONP": "#FF8C00",   # One Nation Orange
        "OTHER": "#708090"  # Slate Grey
    }

    for col in draws_df.columns:
        if col.upper() in ["SCENARIO", "GAMMA_NEXT"]:
            continue
        vals = draws_df[col].dropna()
        p_name = col.upper()
        color = party_colors.get(p_name, "#333333")
        
        is_alp = (p_name == "ALP")
        lw = 2.8 if is_alp else 1.8
        alpha_fill = 0.18 if is_alp else 0.05

        sns.kdeplot(
            data=vals,
            label=p_name,
            color=color,
            linewidth=lw,
            ax=ax,
            fill=True,
            alpha=alpha_fill
        )

    if "ALP" in draws_df.columns:
        alp_draws = draws_df["ALP"].dropna()
        p05_val, avg_val, p95_val = np.percentile(alp_draws, [5, 50, 95])
    else:
        seat_df = res_model_output.seat
        avg_row = seat_df[seat_df["scenario"] == "average"]
        avg_val = float(avg_row["primary_share_pct"].iloc[0])
        p05_val = float(avg_row["seat_p05"].iloc[0]) if "seat_p05" in avg_row.columns else float(seat_df[seat_df["scenario"] == "worst"]["primary_share_pct"].iloc[0])
        p95_val = float(avg_row["seat_p95"].iloc[0]) if "seat_p95" in avg_row.columns else float(seat_df[seat_df["scenario"] == "best"]["primary_share_pct"].iloc[0])

    y_lims = ax.get_ylim()
    max_y = y_lims[1]
    alp_color = party_colors["ALP"]

    targets = [
        {"val": p05_val, "label": f"ALP Worst (0.05)\n{p05_val:.1f}%", "y_pos": max_y * 0.70, "ha": "right", "x_offset": -0.4},
        {"val": avg_val, "label": f"ALP Average (0.50)\n{avg_val:.1f}%", "y_pos": max_y * 0.90, "ha": "center", "x_offset": 0.0},
        {"val": p95_val, "label": f"ALP Best (0.95)\n{p95_val:.1f}%", "y_pos": max_y * 0.70, "ha": "left", "x_offset": 0.4}
    ]

    for item in targets:
        v = item["val"]
        lbl = item["label"]
        y_pos = item["y_pos"]
        ha = item["ha"]
        x_off = item["x_offset"]

        ax.vlines(x=v, ymin=0, ymax=y_pos - (max_y * 0.03), colors=alp_color, linestyles="dotted", linewidth=2)
        ax.text(
            x=v + x_off, y=y_pos, s=lbl, ha=ha, va="bottom",
            fontsize=9, fontweight="bold", color=alp_color
        )

    ax.set_title("ENKI Multi-Party Primary Vote Outcome Distributions", fontsize=15, fontweight="bold", pad=25)
    ax.set_xlabel("Projected Primary Vote Share (%)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold", labelpad=10)

    ax.set_xlim(0, 70)  # Standard bounds for primary votes
    ticks = [t for t in ax.get_xticks() if 0 <= t <= 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}%" for t in ticks])
    ax.yaxis.set_major_formatter('{x:.2f}')

    ax.legend(title="Party", frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    sns.despine(top=True, right=True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def generate_booth_volume_ridgeline(booths, forecasted_matrix, df_historical, output_path, top_n=15, log_scale=True):
    output_path = Path(output_path)
    
    # 1. Get the latest historical vote counts for each booth
    max_year = df_historical['year'].max()
    vcol = 'n' if 'n' in df_historical.columns else 'total_votes'      # formal votes, to match the forecast
    df_latest = df_historical[df_historical['year'] == max_year].set_index('booth')[vcol].to_dict()
    
    # Calculate mean forecasted volume per booth to sort them descending
    mean_volumes = forecasted_matrix.mean(axis=0)
    sorted_indices = np.argsort(mean_volumes)[::-1] # Descending order
    
    top_indices = sorted_indices[:top_n]
    plot_booths = [booths[i] for i in top_indices]
    plot_matrix = forecasted_matrix[:, top_indices] # Shape: (num_sims, top_n)
    
    # 2. Setup figure with a WIDE left margin so booth names never get cut off
    fig, axes = plt.subplots(len(plot_booths), 1, figsize=(12, len(plot_booths) * 0.45), sharex=True)
    
    if len(plot_booths) == 1:
        axes = [axes]
        
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_booths)))
    
    for idx, (i, booth_name) in enumerate(zip(top_indices, plot_booths)):
        ax = axes[idx]
        sim_counts = plot_matrix[:, idx]
        
        # Optional log-transform for skewed pre-poll data
        if log_scale:
            plot_data = np.log10(sim_counts + 1)
        else:
            plot_data = sim_counts
            
        # Draw the kernel density estimate
        sns.kdeplot(plot_data, ax=ax, fill=True, color=colors[idx], alpha=0.6, linewidth=1.5)
        
        # Get and transform historical baseline marker safely
        raw_hist_val = df_latest.get(booth_name, np.nan)
        hist_val = pd.to_numeric(raw_hist_val, errors='coerce')
        
        if not np.isnan(hist_val) and hist_val > 0:
            plot_hist = np.log10(hist_val + 1) if log_scale else hist_val
            # On the top axis (idx == 0), give it a native label so matplotlib picks it up for the legend automatically
            label_str = 'Previous Election Volume' if idx == 0 else ""
            ax.axvline(x=plot_hist, color='red', linestyle='--', linewidth=1.5, ymin=0.1, ymax=0.9, label=label_str)
        
        # Formatting style
        ax.set_facecolor((0, 0, 0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Place booth name safely in the expanded left margin area
        ax.text(-0.03, 0.2, str(booth_name), transform=ax.transAxes, 
                ha='right', va='center', fontsize=9, fontweight='bold', color='black')
        
        if idx < len(plot_booths) - 1:
            ax.spines['bottom'].set_visible(False)
        else:
            ax.spines['bottom'].set_visible(True)
            xlabel_text = "Log10(Forecasted Voters + 1)" if log_scale else "Forecasted Number of Voters ($n$)"
            ax.set_xlabel(xlabel_text, fontsize=10, fontweight='bold')

    # --- NATIVE LEGEND SETUP (POSITIONED OUT OF THE WAY) ---
    if len(axes) > 0:
        axes[0].legend(
            loc='upper left', 
            bbox_to_anchor=(1.02, 1.0), 
            frameon=True, 
            facecolor='white', 
            edgecolor='none', 
            fontsize=9
        )

    title_suffix = " (Log Scale)" if log_scale else ""
    plt.suptitle(f"ENKI Monte Carlo Booth Volume Distribution vs. Previous Election{title_suffix}", y=0.98, fontsize=12, fontweight='bold')
    
    # Leave extra space on the left (0.22) for booth names and right (0.85) for external legend
    plt.tight_layout(rect=[0.22, 0, 0.85, 0.96])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Booth volume ridgeline plot saved to: {output_path}")
    
def generate_booth_swing_scatter_plot(res_model_output, booth_input_df=None, output_path="booth_swing_scatter.png", label_all=True):
    """Expected booth volume vs projected swing from the booth's own LAST-election share.
    Both come from the model output (fc.last_share, fc.formal_weight); nothing is invented."""
    fc = res_model_output.forecast.copy()
    need = {"mean", "last_share", "formal_weight"}
    if not need <= set(fc.columns):
        raise ValueError(f"forecast table missing {sorted(need - set(fc.columns))}; re-run predict().")
    fc = fc.dropna(subset=["last_share"])
    fc["swing"] = fc["mean"] - fc["last_share"]
    fc["n_bt"] = fc["formal_weight"].astype(float)
    fc["abs_swing"] = fc["swing"].abs()
    max_swing = max(5.0, fc["abs_swing"].max() * 1.35)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(data=fc, x="n_bt", y="swing", size="n_bt", sizes=(40, 350), color="black", alpha=0.75,
                    legend=False, ax=ax)
    
    # Set x-axis to log scale
    ax.set_xscale("log")
    
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    labelled = fc if label_all else fc.sort_values(by="abs_swing", ascending=False).head(5)
    texts = [ax.text(r["n_bt"], r["swing"], str(r["booth"]), fontsize=8, fontweight="bold", color="black")
            for _, r in labelled.iterrows()]
    adjust_text(texts, x=labelled["n_bt"].values, y=labelled["swing"].values, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7))
    ax.set_title("ENKI Expected Booth Size vs. Projected Swing from the Booth's Last Election Share", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Expected Formal Votes at Next Election (Log Scale)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Projected Mean Share minus Last Election Share (points)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylim(-max_swing, max_swing)
    ax.yaxis.set_major_formatter('{x:.1f}')
    sns.despine(top=True, right=True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def save_backtest_table_png(summary_df: pd.DataFrame, output_path: Path):
    """Renders backtest summary metrics into a clean table PNG for a quick glance."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if summary_df is None or summary_df.empty:
        print("-> Warning: Backtest summary dataframe is empty. Skipping table PNG generation.")
        return

    # Model vs naive benchmarks (signed seat errors and booth MAE on booths common to both elections)
    display_cols = [
        "target", "seat_pred_median", "seat_actual", "seat_error", "nochange_seat_err", "statewide_swing_seat_err",
        "seat_in_interval", "booth_MAE", "nochange_MAE", "statewide_swing_MAE", "booth_RMSE", "booth_cover"
    ]
    labels = {"seat_pred_median": "seat pred\nmedian", "seat_actual": "seat\nactual", "seat_error": "MODEL\nseat err",
              "nochange_seat_err": "NO-CHANGE\nseat err", "statewide_swing_seat_err": "STATE-SWING\nseat err",
              "seat_in_interval": "in 90%\ninterval", "booth_MAE": "MODEL\nbooth MAE", "nochange_MAE": "NO-CHANGE\nbooth MAE",
              "statewide_swing_MAE": "STATE-SWING\nbooth MAE", "booth_RMSE": "booth\nRMSE", "booth_cover": "booth\ncover"}
    existing_cols = [c for c in display_cols if c in summary_df.columns]
    if not existing_cols:
        existing_cols = list(summary_df.columns)

    df_plot = summary_df[existing_cols].copy()

    for col in df_plot.columns:
        if df_plot[col].dtype in [float, np.floating]:
            df_plot[col] = df_plot[col].round(2)

    fig, ax = plt.subplots(figsize=(max(12, len(existing_cols) * 1.5), len(df_plot) * 0.8 + 2.5))
    ax.axis("off")
    
    table = ax.table(
        cellText=df_plot.values,
        colLabels=[labels.get(c, c) for c in df_plot.columns],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.2, 1.8)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if key[0] % 2 == 0 else "white")

    plt.suptitle("ENKI Backtest Error & Validation Summary", fontsize=14, fontweight="bold", y=0.92)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Backtest summary table PNG saved to: {output_path}")

# ----------------------------------------------------------------------------- hand-off files for INANNA
def save_backtest_draws(draws_by_year: dict, folder) -> None:
    """One file per backtested election: ENKI's predicted booth x party primaries (from earlier elections only) plus the
    ACTUAL booth votes, so INANNA can test the preference flows without importing or re-running ENKI."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for y, d in draws_by_year.items():
        np.savez_compressed(folder / f"backtest_draws_{int(y)}.npz", booths=np.array(d["booths"]), parties=np.array(d["parties"]),
                            booth_party_draws=d["booth_party_draws"], actual_party_votes=d["actual_party_votes"],
                            actual_n=d["actual_n"])


def save_actual_primaries(prepared: pd.DataFrame, parties: Sequence[str], folder) -> None:
    """Seat-wide ACTUAL primary votes per election (all booths): year, formal_total, one column per modelled party."""
    rows = []
    for y, g in prepared.groupby("year"):
        r = {"year": int(y), "formal_total": int(g["n"].sum())}
        r.update({p: int(g[f"y_{p}"].sum()) for p in parties})
        rows.append(r)
    Path(folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(folder) / "actual_primaries.csv", index=False)


def save_run_meta(folder, cfg: ModelConfig, booths_path: str, polls_path: str) -> None:
    """Small provenance record so a downstream stage can see which run it is reading."""
    meta = {"seat_name": cfg.seat_name, "party": cfg.party, "parties": list(cfg.parties), "prediction_date": cfg.prediction_date,
            "election_date": cfg.election_date, "booths_file": str(booths_path), "polls_file": str(polls_path),
            "onp_layer": bool(cfg.onp_layer), "saved_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "run_meta.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------- CLI
def main(seat_name: str = "sample", booth_data: str = "sample_booth_results.csv", 
        polls_data: str = "sample_polls.csv", output_root = "forecast_output", statewide_historical : pd.DataFrame = pd.DataFrame({
            "year":  [1988,  1991,  1995,  1999,  2003,  2007,  2011,  2015,  2019,  2023],
            "ALP":   [38.48, 39.11, 41.27, 42.21, 42.68, 38.87, 25.55, 34.08, 33.31, 36.97],
            "LNP":   [49.54, 41.65, 39.81, 33.69, 34.35, 36.88, 51.15, 45.63, 41.58, 35.37],
            "GRN":   [ 1.50,  2.05,  2.85,  2.80,  4.10,  7.60, 10.28, 10.29,  9.57,  9.70],
            "ONP":   [ 0.00,  0.00,  0.00,  6.40,  0.70,  0.00,  0.00,  0.00,  1.10,  1.80],
            "OTHER": [10.48, 17.19, 16.07, 14.90, 18.17, 16.65, 12.65, 10.00, 14.90, 16.60],
        }), registered_voters : int =62169, expected_turnout : float = 0.91):
    
    STATE_RESULTS = statewide_historical
    need_cols = {"year", "ALP", "LNP", "GRN", "OTHER"}
    if not need_cols <= set(STATE_RESULTS.columns):               # fail early, clearly (the old DataFrame == check raised an ambiguity error)
        raise ValueError(f"statewide_historical is missing columns: {sorted(need_cols - set(STATE_RESULTS.columns))}")
    seat_name = seat_name
    booths_path = booth_data
    polls_path = polls_data
    output_base = output_root
    pred_date = date.today().strftime("%Y-%m-%d")
    seat_folder = output_folder(output_base, seat_name, pred_date)
    nested_out_dir = seat_folder / "ENKI"
    nested_out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = ModelConfig(
        registered_electors=registered_voters,
        expected_turnout_rate=expected_turnout
    )

    print("==================================================")
    print(" STEP 1: RUNNING HISTORICAL BACKTESTS & SUMMARY")
    print("==================================================")
    
    # Load historical booth and state data for backtesting primary engine accuracy
    booth_df = pd.read_csv(booths_path)
    poll_df = pd.read_csv(polls_path)
    
    # Configure model for Balmain ALP vs GRN matchup
    cfg = ModelConfig(
        party="ALP",
        parties=("ALP", "LNP", "GRN", "OTHER"),
        beta_preset="data",
        seat_name=seat_name,
        prediction_date=pred_date,
        election_date="2027-03-13",
        draws=2000,
        tune=2000,
        chains=4,
        seed=42,
        target_accept=0.99
    )

    print("Backtesting the share and volume models across historical election cycles...")
    bt, bt_draws = backtest_bayes(booth_df, STATE_RESULTS, cfg, return_draws=True)
    bt_csv_path = nested_out_dir / "backtest.csv"
    bt.to_csv(bt_csv_path, index=False)
    save_backtest_draws(bt_draws, nested_out_dir)          # hand-off to INANNA (no ENKI re-run needed there)
    print(bt.round(2).T.to_string())
    
    # Save backtest results as a clean table PNG for a quick visual glance at error margins
    table_png_path = nested_out_dir / "backtest_summary_table.png"
    save_backtest_table_png(bt, table_png_path)

    print("\n==================================================")
    print(" STEP 2: RUNNING LIVE 2027 FORECAST PIPELINE")
    print("==================================================")
    
    print(f"Loading data for {cfg.party} from '{booths_path}' and '{polls_path}'...")
    res = predict(booth_results=booth_df, state_results=STATE_RESULTS, poll=poll_df, config=cfg)

    print("Generating diagnostic plots (ridgelines, log-scale volume distributions, and swing scatter)...")
    generate_seat_distribution_plot(res, output_path=nested_out_dir / "seat_distribution.png")
    generate_booth_swing_scatter_plot(res, output_path=nested_out_dir / "booth_swing_scatter.png")
    generate_booth_ridgeline_plot(
        res_model_output=res, 
        target_party=cfg.party, 
        top_n_booths=15,
        output_path=nested_out_dir / "ridgeline.png"
    )
    generate_booth_volume_ridgeline(
        booths=res.volume_names, 
        forecasted_matrix=res.volume_matrix,
        df_historical=res.prepared_df,
        output_path=nested_out_dir / "booth_volume_ridgeline.png", 
        top_n=15, 
        log_scale=True
    )
    
    res.save(str(nested_out_dir))
    save_actual_primaries(res.prepared_df, cfg.parties, nested_out_dir)   # hand-off to INANNA
    save_run_meta(nested_out_dir, cfg, booths_path, polls_path)
    append_history(res, seat_folder, pred_date)

    pd.set_option("display.width", 160)
    lo, mid, hi = cfg.scenario_pct
    q = res.draws_df.quantile([lo / 100, mid / 100, hi / 100]).T
    q.columns = [f"p{lo:g}", f"p{mid:g}", f"p{hi:g}"]
    q.insert(0, "mean", res.draws_df.mean())
    
    print("\nSEAT-WIDE PRIMARY VOTE (joint draws, volume-weighted; sums to 100% in every draw)")
    print(q.round(2).to_string())
    if res.attribution is not None:
        print(f"\nUNCERTAINTY ATTRIBUTION: width of the 90% band on {cfg.party} seat share, one layer off at a time")
        print(res.attribution.round(2).to_string(index=False))
    print(f"\nDiagnostics: max r_hat={res.params['rhat_max']:.3f}, min ESS={res.params['ess_bulk_min']:.0f}, "
        f"divergences={res.params['divergences']} (volume model: {res.params.get('volume_divergences', 'n/a')})")
    
    priors = beta_prior_table(cfg)
    print("beta (sensitivity to the party's own statewide log-share swing): posterior vs prior")
    for p in cfg.parties:
        print(f"  {p:6s}: {res.params[f'beta_{p}']:.2f} +/- {res.params[f'beta_{p}_sd']:.2f}   (prior {priors[p][0]:.2f} +/- {priors[p][1]:.2f})")

    print(f"\nFinished: seat '{seat_name}'. All outputs written to: {nested_out_dir}/")


if __name__ == "__main__":
    main()