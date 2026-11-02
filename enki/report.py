"""report.py -- the terminal report printed at the end of a run."""
from __future__ import annotations

import pandas as pd

from .config import ModelConfig, beta_prior_table


def print_run_report(y, cfg: ModelConfig) -> None:
    pd.set_option("display.width", 160)
    for w in y.warnings:
        print(f"WARNING: {w}")
    if y.backtest is not None:
        print("\nBACKTEST (share model; conditional on the true statewide result and true booth volumes)")
        print(y.backtest.round(2).T.to_string())
    res = y.forecast
    if res is None:
        return
    print("\nSEAT-WIDE PRIMARY VOTE (joint draws, volume-weighted; sums to 100% in every draw)")
    print(y.seat_summary.round(2).to_string())
    if res.attribution is not None:
        print(f"\nUNCERTAINTY ATTRIBUTION: width of the 90% band on {cfg.party} seat share, one layer off at a time")
        print(res.attribution.round(2).to_string(index=False))
    p = res.params
    print(f"\nDiagnostics: max r_hat={p['rhat_max']:.3f}, min ESS={p['ess_bulk_min']:.0f}, divergences={p['divergences']} "
          f"(volume model: {p.get('volume_divergences', 'n/a')})")
    print("beta (sensitivity to the party's own statewide log-share swing): posterior vs prior")
    priors = beta_prior_table(cfg)
    for q in cfg.parties:
        prior = f"prior {priors[q][0]:.2f} +/- {priors[q][1]:.2f}" if q in priors else "reference party: fixed at 0"
        print(f"  {q:6s}: {p[f'beta_{q}']:.2f} +/- {p[f'beta_{q}_sd']:.2f}   ({prior})")
