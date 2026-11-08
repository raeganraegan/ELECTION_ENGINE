"""report.py -- the terminal report printed at the end of a run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig, beta_prior_table


def print_run_report(y, cfg: ModelConfig) -> None:
    pd.set_option("display.width", 160)
    for w in y.warnings:
        print(f"WARNING: {w}")
    if y.backtest is not None:
        r = y.run
        print(f"\nBACKTEST (share model; statewide input: {r.get('backtest_state_mode')}, volumes: {r.get('backtest_volume_mode')})")
        print(y.backtest.round(2).T.to_string())
    res = y.forecast
    if res is None:
        return
    print("\nSEAT-WIDE PRIMARY VOTE (joint draws, volume-weighted; sums to 100% in every draw)")
    print(y.seat_summary.round(2).to_string())
    if res.opv is not None:
        o = res.opv
        a, b = o.report_pair
        sd = o.seat_draws
        sub = sd[sd["pair"].isin([f"{a} v {b}", f"{b} v {a}"])]
        print("\nOPV COUNT (count-by-count finalists per draw; flows from the affinity model)")
        print("  P(finalist pairing):  " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in o.pair_probs.items()))
        print("  P(wins the seat):     " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in o.win_probs.items()))
        q = np.nanpercentile(sub[f"{a}_2CP_pct"], [5, 50, 95])
        print(f"  {a} 2CP % given {a} v {b}: mean {sub[f'{a}_2CP_pct'].mean():.2f}, p05 {q[0]:.2f}, p50 {q[1]:.2f}, p95 {q[2]:.2f}"
              f" | exhausted {sub['exhausted_pct_formal'].mean():.2f}% of formal votes"
              + (f" | flow rates redrawn each draw (conc {o.flow_conc:g})" if o.flow_conc else " | flow rates fixed"))
    if res.attribution is not None:
        print(f"\nUNCERTAINTY ATTRIBUTION: width of the 90% band on {cfg.party} seat share, one layer off at a time")
        print(res.attribution.round(2).to_string(index=False))
    p = res.params
    print(f"\nDiagnostics: max r_hat={p['rhat_max']:.3f}, min ESS={p['ess_bulk_min']:.0f}, divergences={p['divergences']} "
          f"(volume model: {p.get('volume_divergences', 'n/a')})")
    print("beta (sum-to-zero across parties; sensitivity to the party's own statewide log-share swing): posterior vs prior (prior is on the pre-centring value)")
    priors = beta_prior_table(cfg)
    for q in cfg.parties:
        prior = f"prior {priors[q][0]:.2f} +/- {priors[q][1]:.2f}" if q in priors else "reference party: implied by sum-to-zero centring"
        print(f"  {q:6s}: {p[f'beta_{q}']:.2f} +/- {p[f'beta_{q}_sd']:.2f}   ({prior})")