"""A stand-in for fit_share_model that draws a plausible posterior WITHOUT MCMC, so everything downstream of the sampler can be tested."""
import numpy as np
import pandas as pd

from enki.config import ModelConfig
from enki.data import NSW_STATE_RESULTS, prepare_booths
from enki.share_model import ShareFit, build_design


def make_booths(years=(2007, 2011, 2015, 2019, 2023), n_booths=6, seed=0) -> pd.DataFrame:
    """Synthetic raw booth CSV: ordinary booths every year plus a Pre-poll booth from 2015."""
    rng = np.random.default_rng(seed)
    rows = []
    for yi, y in enumerate(years):
        for b in range(n_booths + 1):
            name = f"Booth {b}" if b < n_booths else "Pre-poll"
            if name == "Pre-poll" and y < 2015:
                continue
            total = int(rng.integers(1500, 4000)) if name != "Pre-poll" else int(rng.integers(3000, 6000))
            base = np.array([0.40 + 0.01 * b, 0.30 - 0.01 * b, 0.12, 0.18]) + 0.01 * yi * np.array([1, -1, 0, 0])
            base = base / base.sum()
            formal = int(total * 0.96)
            c = rng.multinomial(formal, base)
            rows.append({"year": y, "booth": name, "total_votes": f"{total:,}", "informal_votes": total - formal,
                         "ALP": c[0], "LNP": c[1], "GRN": c[2]})
    return pd.DataFrame(rows)


def make_polls(onp: bool = True, onp_fraction: bool = False) -> pd.DataFrame:
    d = {"date": ["2026-06-10", "2026-08-15", "2026-10-20"], "ALP": [36.0, 37.5, 38.0], "LNP": [34.0, 33.0, 32.5],
         "GRN": [10.0, 10.5, 10.0], "OTHER": [12.0, 11.0, 11.5], "sample_size": [1000, 1200, 1100]}
    if onp:
        d["ONP"] = [0.6, 0.8, 0.9] if onp_fraction else [7.0, 8.0, 8.5]
    return pd.DataFrame(d)


def fake_fit_factory(D: int = 200):
    def fake_fit(df: pd.DataFrame, cfg: ModelConfig, state_df: pd.DataFrame) -> ShareFit:
        des = build_design(df, cfg, state_df)
        rng = np.random.default_rng(len(df) * 7 + len(des.years))            # same training data -> same posterior
        K, Km, T, B, N = len(des.parties), len(des.parties) - 1, len(des.years), len(des.booth_names), len(df)
        m0 = np.log(des.Y[:, :Km].sum(0) / des.Y[:, Km].sum())
        beta = np.concatenate([0.3 + 0.1 * rng.standard_normal((D, Km)), np.zeros((D, 1))], axis=1)
        bm = beta[:, None, :] + 0.05 * rng.standard_normal((D, B, K))
        bm[:, :, -1] = 0.0
        draws = {"beta": beta, "beta_matrix": bm, "m": m0 + 0.05 * rng.standard_normal((D, Km)),
                 "alpha": 0.2 * rng.standard_normal((D, B, K)), "d": 0.1 * rng.standard_normal((D, T, K)),
                 "eps": 0.05 * rng.standard_normal((D, N, K)), "sigma_d": 0.1 + 0.02 * rng.random((D, K)),
                 "sigma_eps": 0.05 + 0.01 * rng.random((D, K)), "sigma_alpha": 0.2 + 0.02 * rng.random((D, K)),
                 "sigma_beta": 0.1 * rng.random((D, Km)), "rho": 0.4 + 0.2 * rng.random(D)}
        diag = pd.DataFrame({"parameter": ["m[0]"], "r_hat": [1.0], "ess_bulk": [1000.0]})
        return ShareFit(des, draws, diag, 0)
    return fake_fit


def make_backtest_polls(state_df, cfg, years=(2015, 2019, 2023), alp_offset=1.5) -> pd.DataFrame:
    """Historical poll table: three polls per election at 150/90/30 days out, ALP over-polled by alp_offset points."""
    from enki.data import state_matrix
    S = state_matrix(state_df, cfg.parties, list(years))
    rows = []
    for y in years:
        elec = pd.Timestamp(f"{y}-03-25")
        for d in (150, 90, 30):
            r = {"election_year": y, "election_date": str(elec.date()), "date": str((elec - pd.Timedelta(days=d)).date()), "sample_size": 1000}
            r.update({p: float(S.loc[y, p]) + (alp_offset if p == "ALP" else 0.0) for p in cfg.parties})
            rows.append(r)
    return pd.DataFrame(rows)


def fake_volume_fit(df: pd.DataFrame, cfg: ModelConfig):
    """A stand-in for fit_volume_model: no MCMC, same shape of result."""
    from enki.volume_model import VolumeFit
    names = sorted(df["booth"].unique())
    years = sorted(int(y) for y in df["year"].unique())
    rng = np.random.default_rng(11)
    S, B, T = 150, len(names), len(years)
    btype = df.groupby("booth")["btype"].max().reindex(names).values.astype(float)
    totals = df.groupby("year")["n"].sum().reindex(years).values.astype(float)
    diag = pd.DataFrame({"parameter": ["kappa"], "r_hat": [1.0], "ess_bulk": [1000.0]})
    return VolumeFit(names, years, btype, 0, totals, 0, diag,
                     {"a": 0.1 * rng.standard_normal((S, B)), "tau": np.zeros((S, T)), "kappa": np.full(S, 500.0)})


def make_master_polls(cfg, state_df, cycles=(2019, 2023), alp_offset=1.5, live=True, shift_days=(120, 60, 20)) -> pd.DataFrame:
    """One master poll table in the live format (date, firm, sample_size, parties): no election tags. Polls before the 2019 and 2023
    elections, one poll just AFTER 2023, and (optionally) three live polls in 2026."""
    from enki.data import NSW_ELECTION_DATES, state_matrix
    S = state_matrix(state_df, cfg.parties, list(cycles))
    rows = []
    for y in cycles:
        elec = pd.Timestamp(NSW_ELECTION_DATES[y])
        for d in shift_days:
            r = {"date": str((elec - pd.Timedelta(days=d)).date()), "firm": "Resolve", "sample_size": 1000.0}
            r.update({p: (float(S.loc[y, p]) + (alp_offset if p == "ALP" else 0.0)) / 100.0 for p in cfg.parties})
            rows.append(r)
    after = {"date": "2023-05-01", "firm": "Resolve", "sample_size": 1000.0}
    after.update({p: v / 100.0 for p, v in zip(cfg.parties, (30.0, 25.0, 20.0, 25.0))})      # an obviously different post-2023 poll
    rows.append(after)
    if live:
        for dt in ("2026-08-01", "2026-09-01", "2026-10-01"):
            r = {"date": dt, "firm": "Resolve", "sample_size": 1000.0}
            r.update({p: v / 100.0 for p, v in zip(cfg.parties, (36.0, 33.0, 12.0, 19.0))})
            rows.append(r)
    return pd.DataFrame(rows)
