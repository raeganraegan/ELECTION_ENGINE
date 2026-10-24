'''TAKES IN INPUT FROM ENKI'S SAVED FILES ONLY. DOES NOT IMPORT OR RE-RUN ENKI.
INANNA IS THE CONQUEROR, THE GODDESS OF WAR - IN THIS CONTEXT SHE 
SIMULATES THE ELECTION (THE BATTLE) TO SEE WHO WINS'''

import json
from collections import Counter
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# INPUT PREFERENCE FLOW TABLES (NSW) BELOW
# BASED ON ANTONY GREENS WORK: https://www.abc.net.au/dat/news/elections/nsw/2023/guide/NSW2019_PrefFlowTables.pdf

STATEWIDE_FLOW_MATRIX = {
    "ALP_vs_LNP": {
        "GRN":   [0.525, 0.078, 0.397],  # How Greens preferences flow in an ALP v LNP race
        "ONP":   [0.109, 0.180, 0.711],
        "OTHER": [0.205, 0.185, 0.610]
    },
    "ALP_vs_GRN": {
        "LNP":   [0.100, 0.100, 0.800],  # How LNP preferences flow in an ALP v GRN race
        "ONP":   [0.100, 0.100, 0.800],
        "OTHER": [0.300, 0.300, 0.400]
    }
}


def _validate_flows(flow_matrix: dict) -> None:
    """Every flow row must be 3 non-negative numbers summing to 1.0."""
    for k, v in flow_matrix.items():
        if len(v) != 3 or min(v) < 0 or abs(sum(v) - 1.0) > 1e-3:
            raise ValueError(f"Flow row {k!r} = {v} must be 3 non-negative numbers summing to 1.0 (got sum={sum(v):.4f}).")

# OPV PIPELINE ENGINE
def run_opv_pipeline(seat_dir: Path, party="ALP", custom_flow_matrix=None, finalists=None):
    seat_dir = Path(seat_dir)
    print_enki_run_meta(seat_dir)
    target_folder = seat_dir / "INANNA"
    target_folder.mkdir(parents=True, exist_ok=True)

    # STEP 1: Load directly from .npz files
    data_contract = load_enki_npz_outputs(seat_dir=seat_dir)

    booth_cols = data_contract["booth_cols"]
    party_cols = data_contract["party_cols"]
    volume_matrix = data_contract["volume_matrix"]
    party_tensor = data_contract["party_tensor"]
    
    assert volume_matrix.shape[0] == party_tensor.shape[0], "Draw mismatch!"
    print("✓ Data contract integrity verified: Tensors are fully synchronized.")
    print(f"✓ INANNA loaded {len(booth_cols)} booths, {len(party_cols)} parties across {data_contract['n_draws']} draws.")

    # Finalists: determined from ENKI's draws unless the caller fixes them explicitly
    if finalists is None:
        finalists, pair_p = determine_finalists(data_contract["seat_all"], party_cols, prefer=party)
    else:
        finalists, pair_p = tuple(f.upper() for f in finalists), None
        print(f"✓ Finalists fixed by caller: {finalists[0]} v {finalists[1]}")
    
    # STEP 2: Generate Sanity Check Plots
    generate_inanna_tensor_sanity_plots(
        data_contract=data_contract,
        output_dir=target_folder
    )

    # STEP 3: Execute Vectorised OPV Flow Simulation
    print("✓ Executing vectorised OPV preference flow engine...")
    opv_results = simulate_opv_preference_flows(
        data_contract=data_contract,
        custom_flow_matrix=custom_flow_matrix,
        finalists=finalists
    )
    
    booth_2cp_df = opv_results["booth_2cp_df"]
    seat_2cp_draws = opv_results["seat_2cp_draws"]
    
    # Export CSV datasets
    booth_2cp_df.to_csv(target_folder / "booth_2cp_exhausted_distribution.csv", index=False)
    seat_2cp_draws.to_csv(target_folder / "seat_2cp_draws.csv", index=False)
    
    # STEP 4: Generate All Sorted Booth Breakdown & Diagnostic Charts
    generate_opv_outcome_plots(booth_2cp_df, data_contract, output_dir=target_folder, finalists=finalists)
    generate_seat_outcome_plot(seat_2cp_draws, output_dir=target_folder, finalists=finalists)    
    # Print Seat-Wide Summary to Terminal
    p_a, p_b = [f.upper() for f in finalists]
    a_mean = seat_2cp_draws[f"{p_a}_2CP_pct"].mean()
    a_p05 = np.percentile(seat_2cp_draws[f"{p_a}_2CP_pct"], 5)
    a_p95 = np.percentile(seat_2cp_draws[f"{p_a}_2CP_pct"], 95)
    exh_mean = seat_2cp_draws["Exhausted_pct_formal"].mean()
    p_win = float((seat_2cp_draws[f"{p_a}_2CP_pct"] > 50.0).mean())
    
    print(f"\n--- INANNA OPV SIMULATION SUMMARY ({p_a} vs {p_b}) ---")
    print(f"  {p_a} Matchup Share: {a_mean:.2f}%  (90% CI: {a_p05:.2f}% - {a_p95:.2f}%)")
    print(f"  {p_b} Matchup Share: {100.0 - a_mean:.2f}%")
    print(f"  P({p_a} wins, >50):  {p_win:.3f}")
    print(f"  Exhausted Vote Rate: {exh_mean:.2f}% of total formal votes")
    if pair_p is not None:
        print(f"  P({p_a} v {p_b} is the final two): {pair_p:.3f}  (P(win) above is conditional on this pairing)")
    print(f"✓ Datasets saved to: {target_folder}/")
    
    opv_results["p_win"] = p_win
    data_contract["opv_results"] = opv_results
    return data_contract


# ----- DYNAMIC FINALISTS ------
def _top_two_pairs(shares, cols_upper, exclude=("OTHER",)) -> dict:
    """shares (D, K) seat-wide primary %. Returns {frozenset({a, b}): fraction of draws in which a and b are the top two},
    ignoring catch-all buckets (OTHER) that cannot be a finalist."""
    shares = np.asarray(shares, dtype=float)
    idx = [j for j, c in enumerate(cols_upper) if c not in exclude]
    if len(idx) < 2:
        raise ValueError("Need at least two eligible parties to determine finalists.")
    top = np.argsort(-shares[:, idx], axis=1)[:, :2]
    names = [cols_upper[j] for j in idx]
    counts = Counter(frozenset((names[a], names[b])) for a, b in top)
    return {k: v / len(top) for k, v in counts.items()}


def determine_finalists(shares, cols, prefer="ALP", exclude=("OTHER",), min_prob=0.9, verbose=True):
    """Seat finalists = the pairing most often in the top two across draws (OTHER excluded). Returns ((p_a, p_b), P(pairing)).
    `prefer` is listed first when it is in the pair (its 2CP% is the one reported), otherwise the higher mean share is.
    LIMITATION: one pairing is used for every draw; draws where another pairing occurs are still pushed through this
    pairing's flows, so a low P(pairing) means the 2CP result should be read with caution."""
    cols_u = [str(c).upper() for c in cols]
    shares = np.asarray(shares, dtype=float)
    probs = _top_two_pairs(shares, cols_u, exclude)
    best = max(probs, key=probs.get)
    mean = dict(zip(cols_u, shares.mean(axis=0)))
    pair = sorted(best, key=lambda c: (c != str(prefer).upper(), -mean[c]))
    if verbose:
        print("✓ Finalists determined from ENKI draws (top two in each draw, OTHER excluded):")
        for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:4]:
            print(f"    {' v '.join(sorted(k))}: {100 * v:.1f}% of draws")
        print(f"✓ Using {pair[0]} v {pair[1]}")
        if probs[best] < min_prob:
            print(f"!! WARNING: only {100 * probs[best]:.1f}% of draws have this pairing. The 2CP result is conditional on it "
                f"and the other pairings above are ignored.")
    return (pair[0], pair[1]), float(probs[best])


def _select_flows(flow_matrix: dict, p_a: str, p_b: str) -> dict:
    """Pick the flow table for this pairing, as [to p_a, to p_b, exhausted] per party.
    PAIR-KEYED tables ({"ALP_vs_GRN": {...}, ...}) are looked up by pairing (a reversed key has its first two entries swapped);
    a missing pairing raises. FLAT tables ({"LNP": [...], ...}) are used as given, but a pure finalist row such as
    "GRN": [0, 1, 0] must match the resolved finalists, otherwise it would silently send that party's votes to the wrong place."""
    if any("_vs_" in str(k) for k in flow_matrix):
        if f"{p_a}_vs_{p_b}" in flow_matrix:
            return flow_matrix[f"{p_a}_vs_{p_b}"]
        if f"{p_b}_vs_{p_a}" in flow_matrix:
            return {k: [v[1], v[0], v[2]] for k, v in flow_matrix[f"{p_b}_vs_{p_a}"].items()}
        raise ValueError(f"No flow table for {p_a} v {p_b}. Tables available: {[k for k in flow_matrix]}")
    for k, v in flow_matrix.items():
        if v[0] + v[1] == 1.0 and v[0] in (0.0, 1.0) and v[2] == 0.0:        # pure finalist row
            expected = p_a if v[0] == 1.0 else p_b
            if str(k).upper() != expected:
                raise ValueError(f"Flow table row {k!r}={v} marks {k} as a finalist, but the finalists are {p_a} v {p_b}. "
                                f"Supply a pair-keyed table, e.g. {{'{p_a}_vs_{p_b}': {{...}}}}.")
    return flow_matrix


# ----- HELPER FUNCTIONS ------
def simulate_opv_preference_flows(data_contract: dict, custom_flow_matrix: dict = None, finalists=("ALP", "LNP")) -> dict:
    """
    Applies OPV preference flow rates [to_A, to_B, exhausted] to minor party primary votes
    across all MCMC draws (D) and booths (B) for any two finalist parties (e.g. ALP vs GREENS).
    """
    p_a, p_b = [f.upper() for f in finalists]
    flow_matrix = _select_flows(custom_flow_matrix or STATEWIDE_FLOW_MATRIX, p_a, p_b)
    _validate_flows(flow_matrix)
    
    p_a, p_b = [f.upper() for f in finalists]
    booth_cols = data_contract["booth_cols"]
    party_cols = data_contract["party_cols"]
    volume_matrix = data_contract["volume_matrix"]  # (D, B)
    party_tensor = data_contract["party_tensor"]    # (D, B, K) in % (0-100)
    
    D, B, K = party_tensor.shape
    
    # 1. Convert Primary % Tensor to Raw Votes Tensor: (D, B, K)
    raw_primary_votes = (party_tensor / 100.0) * volume_matrix[:, :, np.newaxis]
    
    # Initialize 2CP + Exhausted raw vote containers: (D, B)
    votes_a = np.zeros((D, B), dtype=float)
    votes_b = np.zeros((D, B), dtype=float)
    exhausted_votes = np.zeros((D, B), dtype=float)
    
    # 2. Assign Direct Primary Votes for Finalists & Distribute Others
    for k_idx, party_name in enumerate(party_cols):
        p_upper = party_name.upper()
        if p_upper == p_a:
            votes_a += raw_primary_votes[:, :, k_idx]
        elif p_upper == p_b or (p_b in ["LNP", "LIB", "NAT"] and p_upper in ["LNP", "LIB", "NAT"]):
            votes_b += raw_primary_votes[:, :, k_idx]
        else:
            # 3. Distribute Minor Party Votes via Flow Rates
            flows = flow_matrix.get(p_upper, flow_matrix.get("OTHER", [0.20, 0.20, 0.60]))
            to_a_rate, to_b_rate, exh_rate = flows
            
            minor_votes = raw_primary_votes[:, :, k_idx]
            votes_a += minor_votes * to_a_rate
            votes_b += minor_votes * to_b_rate
            exhausted_votes += minor_votes * exh_rate

    # Total Formal Votes per draw x booth
    total_formal = volume_matrix  # (D, B)
    total_formal_safe = np.maximum(total_formal, 1.0)
    
    # Total Formal Votes Excluding Exhausted (Active 2CP Denominator)
    total_2cp_active = np.maximum(votes_a + votes_b, 1.0)

    # 4. Compute Booth-Level Posterior Means and Percentiles across Draws
    booth_summary_rows = []
    for b_idx, booth_name in enumerate(booth_cols):
        va = votes_a[:, b_idx]
        vb = votes_b[:, b_idx]
        exh_v = exhausted_votes[:, b_idx]
        tot_v = total_formal_safe[:, b_idx]
        
        # Share of Total Formal Votes (%)
        pct_formal_a = (va / tot_v) * 100.0
        pct_formal_b = (vb / tot_v) * 100.0
        exh_pct_formal = (exh_v / tot_v) * 100.0
        
        # Share of Effective 2CP Votes (%)
        ok = (va + vb) > 0
        pct_2cp_a = va[ok] / (va + vb)[ok] * 100.0
        pct_2cp_b = 100.0 - pct_2cp_a
        has = bool(ok.any())
        
        booth_summary_rows.append({
            "booth": booth_name,
            "expected_formal_votes": np.mean(tot_v),
            f"{p_a}_2cp_votes_mean": np.mean(va),
            f"{p_b}_2cp_votes_mean": np.mean(vb),
            "exhausted_votes_mean": np.mean(exh_v),
            f"{p_a}_2cp_share_mean": np.mean(pct_2cp_a) if has else np.nan,
            f"{p_a}_2cp_p05": np.percentile(pct_2cp_a, 5) if has else np.nan,
            f"{p_a}_2cp_p95": np.percentile(pct_2cp_a, 95) if has else np.nan,
            f"{p_b}_2cp_share_mean": np.mean(pct_2cp_b) if has else np.nan,
            "draws_with_votes_pct": 100.0 * ok.mean(),
            "exhausted_share_of_total_mean": np.mean(exh_pct_formal)
        })
        
    booth_2cp_df = pd.DataFrame(booth_summary_rows)

    # 5. Compute Seat-Wide Aggregate 2CP Draws
    seat_tot_votes = volume_matrix.sum(axis=1)  # (D,)
    seat_votes_a = votes_a.sum(axis=1)          # (D,)
    seat_votes_b = votes_b.sum(axis=1)          # (D,)
    seat_exh_votes = exhausted_votes.sum(axis=1)  # (D,)
    seat_act_votes = np.maximum(seat_votes_a + seat_votes_b, 1.0)
    
    seat_2cp_draws = pd.DataFrame({
        f"{p_a}_2CP_pct": (seat_votes_a / seat_act_votes) * 100.0,
        f"{p_b}_2CP_pct": (seat_votes_b / seat_act_votes) * 100.0,
        "Exhausted_pct_formal": (seat_exh_votes / np.maximum(seat_tot_votes, 1.0)) * 100.0,
        f"{p_a}_2CP_votes": seat_votes_a,
        f"{p_b}_2CP_votes": seat_votes_b,
        "Exhausted_votes": seat_exh_votes,
        "Total_Formal_votes": seat_tot_votes
    })

    return {
        "booth_2cp_df": booth_2cp_df,
        "seat_2cp_draws": seat_2cp_draws,
        "tensors": {
            "votes_a": votes_a,
            "votes_b": votes_b,
            "exhausted_votes": exhausted_votes
        }
    }


def load_enki_npz_outputs(seat_dir: Path) -> dict:
    seat_dir = Path(seat_dir)
    enki_dir = seat_dir / "ENKI"
    if not enki_dir.exists():
        subfolders = list(seat_dir.glob("*/ENKI"))
        enki_dir = subfolders[0] if subfolders else seat_dir

    vol_data = np.load(enki_dir / "volume_draws.npz", allow_pickle=True)
    party_data = np.load(enki_dir / "booth_party_draws.npz", allow_pickle=True)
    
    vol_names = [str(b).strip() for b in vol_data["names"]]
    volume_matrix = vol_data["V"]  # (D, B)
    
    party_booths = [str(b).strip() for b in party_data["booths"]]
    party_names = [str(p).strip() for p in party_data["parties"]]
    
    raw_party_tensor = party_data["booth_party_draws"]  # (D, B, K)
    party_tensor = raw_party_tensor  # (D, B, K)

    assert party_tensor.shape[1] == len(party_booths), \
        f"tensor has {party_tensor.shape[1]} booth columns but {len(party_booths)} labels"
    assert volume_matrix.shape[1] == len(vol_names), \
        f"volume matrix has {volume_matrix.shape[1]} booth columns but {len(vol_names)} labels"
    missing = set(vol_names) - set(party_booths)
    assert not missing, f"booths in volume_draws.npz but not in booth_party_draws.npz: {sorted(missing)}"

    booth_idx_map = {b: i for i, b in enumerate(party_booths)}
    aligned_indices = [booth_idx_map[b] for b in vol_names]
    aligned_party_tensor = party_tensor[:, aligned_indices, :]

    active_mask = volume_matrix.sum(axis=0) > 0
    Vk = volume_matrix[:, active_mask]
    Pk = aligned_party_tensor[:, active_mask, :]
    filtered_booths = [b for b, act in zip(vol_names, active_mask) if act]

    assert np.abs(Pk.sum(axis=2) - 100.0).max() < 1e-6, "party tensor must be in percent (rows sum to 100)"
    seat_all = np.einsum("db,dbk->dk", Vk, Pk) / Vk.sum(axis=1, keepdims=True)
    assert np.abs(seat_all.sum(axis=1) - 100.0).max() < 1e-6, "seat-wide shares must sum to 100 in every draw"
    draws_df = pd.DataFrame(seat_all, columns=party_names)
    
    assert party_booths == vol_names, "booth order differs between ENKI exports"

    return {
        "booth_cols": filtered_booths,
        "volume_matrix": Vk,
        "booth_vote_map": dict(zip(filtered_booths, Vk.mean(axis=0))),
        "party_cols": party_names,
        "party_tensor": Pk,
        "seat_all": seat_all,
        "draws_df": draws_df,
        "n_draws": Pk.shape[0]
    }


def generate_opv_outcome_plots(booth_2cp_df: pd.DataFrame, data_contract: dict, output_dir: Path, finalists=("ALP", "LNP")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    party_colors = {"ALP": "#E13940", "LNP": "#0047AB", "LIB": "#0047AB", "NAT": "#0066CC", "GRN": "#104C34", "ONP": "#FF8C00", "OTHER": "#708090"}

    p_a, p_b = [f.upper() for f in finalists]
    color_a = party_colors.get(p_a, "#E13940")
    color_b = party_colors.get(p_b, "#0047AB")

    # --- PLOT 1: Raw First Preference Vote Share Breakdown ---
    party_tensor = data_contract["party_tensor"]  # (D, B, K)
    booths = data_contract["booth_cols"]
    party_cols = data_contract["party_cols"]
    
    mean_party_shares = party_tensor.mean(axis=0)
    pref_df = pd.DataFrame(mean_party_shares, columns=party_cols, index=booths)
    pref_df['booth'] = booths
    
    sort_col = next((c for c in party_cols if c.upper() == p_a), party_cols[0])
    pref_df = pref_df.sort_values(by=sort_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y = range(len(pref_df))
    left_cum = np.zeros(len(pref_df))
    
    for p_col in party_cols:
        col_vals = pref_df[p_col].values
        color = party_colors.get(p_col.upper(), "#333333")
        ax.barh(y, col_vals, left=left_cum, color=color, label=f"{p_col} Primary (%)", edgecolor='white')
        left_cum += col_vals
        
    ax.set_yticks(y)
    ax.set_yticklabels(pref_df['booth'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Percentage of First Preference Formal Votes (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Raw First Preference Vote Shares by Voting Booth (Sorted by {p_a})', fontsize=13, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / "booth_raw_first_preference_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: Final 2CP & Exhausted Outcomes by Booth ---
    df_2cp = booth_2cp_df.copy()
    
    # Dynamic column name resolution matching simulate_opv_preference_flows output
    col_votes_a = f"{p_a}_2cp_votes_mean"
    col_votes_b = f"{p_b}_2cp_votes_mean"
    col_share_a = f"{p_a}_2cp_share_mean"
    col_p05 = f"{p_a}_2cp_p05"
    col_p95 = f"{p_a}_2cp_p95"

    df_2cp['pct_total_a'] = (df_2cp[col_votes_a] / df_2cp['expected_formal_votes']) * 100.0
    df_2cp['pct_total_b'] = (df_2cp[col_votes_b] / df_2cp['expected_formal_votes']) * 100.0
    df_2cp['exh_pct_total'] = (df_2cp['exhausted_votes_mean'] / df_2cp['expected_formal_votes']) * 100.0
    
    df_2cp = df_2cp.sort_values(by='pct_total_a', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_2cp = range(len(df_2cp))
    
    vals_a = df_2cp['pct_total_a'].values
    vals_b = df_2cp['pct_total_b'].values
    exh_vals = df_2cp['exh_pct_total'].values
    
    ax.barh(y_2cp, vals_a, color=color_a, label=f'{p_a} Final Share (%)', edgecolor='white')
    ax.barh(y_2cp, vals_b, left=vals_a, color=color_b, label=f'{p_b} Final Share (%)', edgecolor='white')
    ax.barh(y_2cp, exh_vals, left=vals_a + vals_b, color='#708090', label='Exhausted Share (%)', edgecolor='white')
    
    ax.set_yticks(y_2cp)
    ax.set_yticklabels(df_2cp['booth'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Percentage of Total Formal Votes (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Final Two-Candidate ({p_a} vs {p_b}) & Exhausted Vote Shares by Booth', fontsize=13, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / "booth_2cp_exhausted_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 3: Booth-Level Forecast Uncertainty Forest Plot ---
    df_forest = df_2cp.sort_values(by=col_share_a, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_f = range(len(df_forest))
    
    err_left = df_forest[col_share_a] - df_forest[col_p05]
    err_right = df_forest[col_p95] - df_forest[col_share_a]
    
    ax.errorbar(df_forest[col_share_a], y_f, xerr=[err_left, err_right], fmt='o',
                color=color_a, ecolor=color_a, elinewidth=2, capsize=4, label=f'{p_a} Mean & 90% CI')
    
    ax.axvline(x=50.0, color='black', linestyle='--', linewidth=1.5, label='50% Majority Threshold')
    ax.set_yticks(y_f)
    ax.set_yticklabels(df_forest['booth'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f'{p_a} Final Matchup Vote Share (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Booth-Level {p_a} vs {p_b} Forecast Uncertainty (90% CI)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.savefig(output_dir / "booth_2cp_uncertainty_forest.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 4: Exhausted Ballot Rate vs. Support Scatter Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = (booth_2cp_df['expected_formal_votes'] / booth_2cp_df['expected_formal_votes'].max()) * 400 + 30
    
    sns.regplot(data=df_2cp, x=col_share_a, y='exhausted_share_of_total_mean', ax=ax,
                color='#708090', scatter_kws={'s': sizes, 'alpha': 0.65, 'color': color_a},
                line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 1.5})
                
    ax.set_xlabel(f'{p_a} Final Vote Share (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Exhausted Share of Formal Votes (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Exhausted Ballot Rate vs. {p_a} Support Across Booths', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / "exhausted_vs_labor_support_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-> Successfully generated all sorted booth breakdowns and diagnostic charts.")

def generate_seat_outcome_plot(seat_2cp_draws: pd.DataFrame, output_dir: Path,
                               filename: str = "seat_outcome_distribution.png", finalists=("ALP", "LNP")) -> Path:
    """Seat-wide outcome: KDE lines for the two finalist parties and exhausted share, the 50% majority line with
    P(win), and percentile markers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_a, p_b = [f.upper() for f in finalists]
    col_a = f"{p_a}_2CP_pct"
    col_b = f"{p_b}_2CP_pct"
    
    need = [col_a, col_b, "Exhausted_pct_formal"]
    missing = [c for c in need if c not in seat_2cp_draws.columns]
    if missing:
        raise KeyError(f"seat_2cp_draws is missing columns {missing}")

    vals_a = seat_2cp_draws[col_a].dropna().values
    vals_b = seat_2cp_draws[col_b].dropna().values
    exh = seat_2cp_draws["Exhausted_pct_formal"].dropna().values

    p_win = float((vals_a > 50.0).mean())
    q25, q50, q75 = np.percentile(vals_a, [25, 50, 75])

    party_colors = {"ALP": "#E13940", "LNP": "#0047AB", "LIB": "#0047AB", "GRN": "#104C34", "ONP": "#FF8C00", "OTHER": "#708090"}
    color_a = party_colors.get(p_a, "#E13940")
    color_b = party_colors.get(p_b, "#4A9FE5")

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 6))
    series = [(f"{p_a} Final Share", vals_a, color_a, 2.8),
            (f"{p_b} Final Share", vals_b, color_b, 2.4),
            ("Exhausted Share", exh, "#708090", 2.2)]
    for label, vals, colour, lw in series:
        sns.kdeplot(x=vals, ax=ax, color=colour, linewidth=lw, label=label, clip=(0, 100))

    top = ax.get_ylim()[1] * 1.30                      
    ax.set_ylim(0, top)

    # 50% majority threshold + win probability
    ax.vlines(50.0, 0, top * 0.88, colors="black", linestyles=":", linewidth=2.4)
    ax.text(50.0, top * 0.90, f"50% Win Margin\n{p_a} WIN CHANCE: {100.0 * p_win:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")

    # Percentile markers
    marks = [(q25, "p(0.25)", "right", -0.8, 0.50),
            (q50, "p(0.50)", "center", 0.0, 0.66),
            (q75, "p(0.75)", "left", 0.8, 0.50)]
    for val, lab, ha, dx, yfrac in marks:
        ax.vlines(val, 0, top * (yfrac - 0.02), colors=color_a, linestyles="--", linewidth=1.6)
        ax.text(val + dx, top * yfrac, f"{p_a} {lab}\n{val:.1f}%", ha=ha, va="bottom",
                fontsize=9, fontweight="bold", color=color_a,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.set_xlim(0, 100)
    ax.set_title(f"Likely Overall Outcome of the Seat ({p_a} vs {p_b})", fontsize=14, fontweight="bold", pad=18)
    ax.set_xlabel("OPV Vote Share (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability", fontsize=11, fontweight="bold", labelpad=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.despine(top=True, right=True)

    plt.tight_layout()
    out_path = output_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Seat outcome plot saved to: {out_path}")
    return out_path


# ----- SANITY CHECKERS ------
def generate_inanna_tensor_sanity_plots(data_contract: dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    booths = data_contract["booth_cols"]
    volume_matrix = data_contract["volume_matrix"]      
    
    # --- PLOT 1: Volume Ridgeline ---
    mean_volumes = volume_matrix.mean(axis=0)
    sorted_indices = np.argsort(mean_volumes)[::-1]
    top_indices = sorted_indices[:15]
    plot_booths = [booths[i] for i in top_indices]
    plot_matrix = volume_matrix[:, top_indices]
    
    fig, axes = plt.subplots(len(plot_booths), 1, figsize=(12, len(plot_booths) * 0.45), sharex=True)
    if len(plot_booths) == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_booths)))
    
    for idx, (i, booth_name) in enumerate(zip(top_indices, plot_booths)):
        ax = axes[idx]
        sim_counts = plot_matrix[:, idx]
        sns.kdeplot(sim_counts, ax=ax, fill=True, color=colors[idx], alpha=0.6, linewidth=1.5)
        
        ax.set_facecolor((0, 0, 0, 0))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False); ax.yaxis.set_visible(False)
        ax.text(-0.03, 0.2, str(booth_name), transform=ax.transAxes, ha='right', va='center', fontsize=9, fontweight='bold', color='black')
        
        if idx < len(plot_booths) - 1:
            ax.spines['bottom'].set_visible(False)
        else:
            ax.spines['bottom'].set_visible(True)
            ax.set_xlabel("Forecasted Number of Voters ($n$)", fontsize=10, fontweight='bold')

    plt.suptitle("INANNA Booth Volume Distribution (Tensor Sanity Check)", y=0.98, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0.22, 0, 1, 0.96])
    plt.savefig(output_dir / "inanna_tensor_volume_ridgeline.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: Multi-Party Seat Distribution ---
    draws_df = data_contract["draws_df"]
    
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(11, 6))
    party_colors = {"ALP": "#E13940", "LNP": "#0047AB", "LIB": "#0047AB", "NAT": "#0066CC", "GRN": "#104C34", "ONP": "#FF8C00", "OTHER": "#708090"}

    for col in draws_df.columns:
        vals = draws_df[col].dropna()
        p_name = col.upper()
        color = party_colors.get(p_name, "#333333")
        is_alp = (p_name == "ALP")
        sns.kdeplot(data=vals, label=p_name, color=color, linewidth=2.8 if is_alp else 1.8, ax=ax, fill=True, alpha=0.18 if is_alp else 0.05)

    alp_draws = draws_df["ALP"].dropna()
    p05_val, avg_val, p95_val = np.percentile(alp_draws, [5, 50, 95])
    max_y = ax.get_ylim()[1]
    
    targets = [
        {"val": p05_val, "label": f"ALP Worst (0.05)\n{p05_val:.1f}%", "y_pos": max_y * 0.70, "ha": "right", "x_offset": -0.4},
        {"val": avg_val, "label": f"ALP Average (0.50)\n{avg_val:.1f}%", "y_pos": max_y * 0.90, "ha": "center", "x_offset": 0.0},
        {"val": p95_val, "label": f"ALP Best (0.95)\n{p95_val:.1f}%", "y_pos": max_y * 0.70, "ha": "left", "x_offset": 0.4}
    ]

    for item in targets:
        ax.vlines(x=item["val"], ymin=0, ymax=item["y_pos"] - (max_y * 0.03), colors="#E13940", linestyles="dotted", linewidth=2)
        ax.text(x=item["val"] + item["x_offset"], y=item["y_pos"], s=item["label"], ha=item["ha"], va="bottom", fontsize=9, fontweight="bold", color="#E13940")

    ax.set_title("INANNA Multi-Party Primary Vote Outcome Distributions (Tensor Sanity Check)", fontsize=15, fontweight='bold', pad=25)
    ax.set_xlabel("Projected Primary Vote Share (%)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Probability Density", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlim(0, min(100.0, max(70.0, float(np.percentile(draws_df.values, 99.5)) + 5.0)))
    ax.legend(title="Party", frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    sns.despine(top=True, right=True)
    
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.95)
    plt.savefig(output_dir / "inanna_tensor_seat_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Tensor sanity plots successfully generated and saved to: {output_dir}")


# =====================================================================================================================
# BACKTESTS: do the preference flows (and ENKI's primaries feeding them) reproduce REAL 2CP results?
# File-driven: everything ENKI-side is READ from <seat_dir>/ENKI/ (see CONTRACT at the top). No ENKI import, no refits.
# =====================================================================================================================
# ALP two-candidate-preferred % in the seat, per election (all vote types). VERIFY against the NSWEC final results.
#   2023: ALP 24,677 v LIB 23,013 -> 51.74%      2019: LIB 50.50% -> ALP 49.50%

# These are SEAT-SPECIFIC. There is deliberately no default: pass the real results for the seat you are running.


def _require_actual_2cp(actual_2cp):
    if not actual_2cp:
        raise ValueError("actual_2cp is required: pass the REAL 2CP % (primary finalist) by year for THIS seat, "
                        "e.g. actual_2cp=ACTUAL_2CP_ALP_BALMAIN.")
    return actual_2cp


def print_enki_run_meta(seat_dir) -> None:
    f = Path(seat_dir) / "ENKI" / "run_meta.json"
    if not f.exists():
        print("NOTE: no run_meta.json in the ENKI folder (older run); cannot show provenance.")
        return
    m = json.loads(f.read_text())
    print(f"✓ Reading ENKI run: seat={m.get('seat_name')} | built {m.get('saved_at')} | booths={m.get('booths_file')} "
        f"| polls={m.get('polls_file')} | ONP layer={m.get('onp_layer')}")


def load_enki_actual_primaries(seat_dir) -> pd.DataFrame:
    f = Path(seat_dir) / "ENKI" / "actual_primaries.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Re-run ENKI (it writes this file).")
    return pd.read_csv(f)


def load_enki_backtest_draws(seat_dir, years=None) -> dict:
    """ENKI's saved backtest predictions: {year: {booths, parties, booth_party_draws (D,B,K %), actual_party_votes, actual_n}}."""
    enki_dir = Path(seat_dir) / "ENKI"
    out = {}
    for f in sorted(enki_dir.glob("backtest_draws_*.npz")):
        y = int(f.stem.split("_")[-1])
        if years is not None and y not in years:
            continue
        z = np.load(f, allow_pickle=False)
        out[y] = {"booths": [str(b) for b in z["booths"]], "parties": [str(p) for p in z["parties"]],
                "booth_party_draws": z["booth_party_draws"], "actual_party_votes": z["actual_party_votes"],
                "actual_n": z["actual_n"]}
    if not out:
        raise FileNotFoundError(f"No backtest_draws_<year>.npz in {enki_dir}. Re-run ENKI (its backtest step writes them).")
    return out


def _make_contract(booths, parties, volumes, party_pct) -> dict:
    """The same dict simulate_opv_preference_flows() reads for the live forecast (volumes (D,B); party_pct (D,B,K) in %)."""
    return {"booth_cols": list(booths), "party_cols": list(parties),
            "volume_matrix": np.asarray(volumes, dtype=float), "party_tensor": np.asarray(party_pct, dtype=float)}


def _flows_for_year(year, flow_matrix_by_year, default_flow_matrix):
    """Year-specific flows if supplied (use flows from BEFORE that election to avoid look-ahead), else the fallback."""
    if flow_matrix_by_year and year in flow_matrix_by_year:
        return flow_matrix_by_year[year], "year-specific"
    if default_flow_matrix:
        return default_flow_matrix, "fallback (may leak: not from before this election)"
    return STATEWIDE_FLOW_MATRIX, "statewide default"


def _two_cp_from_primaries(parties, votes, flow_matrix, finalists=("ALP", "LNP")) -> dict:
    """Seat-level 2CP from ACTUAL primary vote counts, using the same engine as the live forecast (one draw, one 'booth')."""
    votes = np.asarray(votes, dtype=float)
    n = votes.sum()
    contract = _make_contract(["SEAT"], parties, [[n]], (100.0 * votes / n)[None, None, :])
    sd = simulate_opv_preference_flows(contract, flow_matrix, finalists=finalists)["seat_2cp_draws"].iloc[0]
    p_a, p_b = [f.upper() for f in finalists]
    return {
        "alp_2cp": float(sd[f"{p_a}_2CP_pct"]),
        "lnp_2cp": float(sd[f"{p_b}_2CP_pct"]),
        "exhausted": float(sd["Exhausted_pct_formal"])
    }


def _finalists_for(party, parties, votes, finalists):
    """Finalists for one election from its ACTUAL seat-wide primary votes (or the caller's fixed pair). None if `party` is not in the final two."""
    if finalists is None:
        pair, _ = determine_finalists((100.0 * np.asarray(votes, dtype=float) / np.sum(votes))[None, :], parties, prefer=party, verbose=False)
    else:
        pair = tuple(f.upper() for f in finalists)
    p_a = party.upper()
    if p_a not in pair:
        return None
    return (p_a, pair[1] if pair[0] == p_a else pair[0])


def run_flow_only_test(seat_dir, actual_2cp: Optional[dict] = None, flow_matrix_by_year: Optional[dict] = None,
                    default_flow_matrix: Optional[dict] = None, years=None, finalists=None, party="ALP") -> pd.DataFrame:
    """TEST 1 - FLOWS ONLY. Feed the ACTUAL primary votes of each election (ENKI's actual_primaries.csv) through the preference
    flows and compare the resulting 2CP% of `party` with the real one. The opponent is the other party in that election's actual
    final two (determined from the votes) unless `finalists` is fixed. Years where `party` was not in the final two are skipped.
    error_pts = predicted - actual, in percentage points."""
    actual_2cp = _require_actual_2cp(actual_2cp)
    ap = load_enki_actual_primaries(seat_dir)
    parties = [c for c in ap.columns if c not in ("year", "formal_total")]
    p_a = party.upper()
    rows = []
    for y in (years or sorted(actual_2cp)):
        sub = ap[ap["year"] == y]
        if sub.empty or y not in actual_2cp:
            print(f"  [flow-only] skipping {y}: no actual primaries from ENKI or no actual 2CP supplied")
            continue
        votes = sub[parties].iloc[0].values.astype(float)
        fin = _finalists_for(party, parties, votes, finalists)
        if fin is None:
            print(f"  [flow-only] skipping {y}: {p_a} was not in the actual final two, so the supplied 2CP does not apply")
            continue
        p_b = fin[1]
        flows, source = _flows_for_year(y, flow_matrix_by_year, default_flow_matrix)
        r = _two_cp_from_primaries(parties, votes, flows, finalists=fin)
        act = float(actual_2cp[y])
        row = {"test": "flow_only", "target_year": int(y), "flow_source": source, "opponent": p_b,
            f"predicted_{p_a}_2CP_pct": r["alp_2cp"], f"actual_{p_a}_2CP_pct": act, "error_pts": r["alp_2cp"] - act,
            "predicted_opponent_2CP_pct": r["lnp_2cp"], "actual_opponent_2CP_pct": 100.0 - act,
            "predicted_exhausted_pct": r["exhausted"]}
        for p, v in zip(parties, votes):
            row[f"actual_primary_{p}"] = 100.0 * v / votes.sum()
        rows.append(row)
    return pd.DataFrame(rows)


def run_end_to_end_test(seat_dir, actual_2cp: Optional[dict] = None, flow_matrix_by_year: Optional[dict] = None,
                        default_flow_matrix: Optional[dict] = None, years=None, finalists=None, party="ALP") -> pd.DataFrame:
    """TEST 2 - END TO END. ENKI's saved backtest predicted each target election's booth primaries from earlier elections only
    (using the actual statewide shares and actual booth volumes); those predicted primaries go through the preference flows
    and the predicted 2CP of `party` is compared with the real one. The opponent is the actual final-two opponent that year.
    finalist_pair_pred_prob = fraction of ENKI's predicted draws in which that same pairing is the top two.
    predicted = median over draws; error_pts = predicted - actual (percentage points).
    primary_propagated_error = predicted - (flows applied to the ACTUAL primaries of the same booths): the part of the error
    caused by ENKI's primary forecast, as opposed to the flows."""
    actual_2cp = _require_actual_2cp(actual_2cp)
    ap = load_enki_actual_primaries(seat_dir)
    ap_parties = [c for c in ap.columns if c not in ("year", "formal_total")]
    want = [int(y) for y in (years or sorted(actual_2cp)) if int(y) in actual_2cp]
    draws = load_enki_backtest_draws(seat_dir, years=want)
    p_a = party.upper()
    rows = []
    for y in want:
        if y not in draws:
            print(f"  [end-to-end] ENKI saved no backtest draws for {y}; skipping")
            continue
        d = draws[y]
        P = np.asarray(d["booth_party_draws"], dtype=float)                     # (D, B, K) %
        n_t = np.asarray(d["actual_n"], dtype=float)                            # (B,) actual booth volumes
        D = P.shape[0]
        sub = ap[ap["year"] == y]
        votes_all = sub[ap_parties].iloc[0].values.astype(float) if len(sub) else np.asarray(d["actual_party_votes"]).sum(axis=0)
        fin = _finalists_for(party, ap_parties if len(sub) else d["parties"], votes_all, finalists)
        if fin is None:
            print(f"  [end-to-end] skipping {y}: {p_a} was not in the actual final two, so the supplied 2CP does not apply")
            continue
        p_b = fin[1]
        flows, source = _flows_for_year(y, flow_matrix_by_year, default_flow_matrix)

        contract = _make_contract(d["booths"], d["parties"], np.tile(n_t, (D, 1)), P)
        seat = simulate_opv_preference_flows(contract, flows, finalists=fin)["seat_2cp_draws"][f"{p_a}_2CP_pct"].values
        p05, p50, p95 = np.percentile(seat, [5, 50, 95])
        act = float(actual_2cp[y])

        covered_actual = _two_cp_from_primaries(d["parties"], np.asarray(d["actual_party_votes"]).sum(axis=0), flows, finalists=fin)
        tot = ap.loc[ap["year"] == y, "formal_total"]
        total_n = float(tot.iloc[0]) if len(tot) else np.nan
        seat_pred_primary = np.einsum("db,dbk->dk", np.tile(n_t, (D, 1)), P) / n_t.sum()
        act_primary = 100.0 * np.asarray(d["actual_party_votes"]).sum(axis=0) / n_t.sum()
        pair_probs = _top_two_pairs(seat_pred_primary, [str(p).upper() for p in d["parties"]])

        row = {"test": "end_to_end", "target_year": y, "flow_source": source, "opponent": p_b,
            f"predicted_{p_a}_2CP_pct": float(p50), f"actual_{p_a}_2CP_pct": act, "error_pts": float(p50 - act),
            "pred_p05": float(p05), "pred_p95": float(p95), "actual_in_90_interval": bool(p05 <= act <= p95),
            "P_win_predicted": float((seat > 50.0).mean()),
            "finalist_pair_pred_prob": float(pair_probs.get(frozenset(fin), 0.0)),
            "flows_on_actual_primaries_same_booths": covered_actual["alp_2cp"],
            "primary_propagated_error": float(p50 - covered_actual["alp_2cp"]),
            "votes_covered_pct": 100.0 * n_t.sum() / total_n if total_n and total_n > 0 else np.nan}
        for j, p in enumerate(d["parties"]):
            row[f"primary_err_{p}"] = float(np.median(seat_pred_primary[:, j]) - act_primary[j])
        rows.append(row)
    return pd.DataFrame(rows)


def run_inanna_backtests(seat_dir: Path, actual_2cp: Optional[dict] = None,
                        flow_matrix_by_year: Optional[dict] = None, default_flow_matrix: Optional[dict] = None,
                        years=None, finalists=None, party="ALP") -> pd.DataFrame:
    """Runs both tests, prints one summary table (predicted / real / error for each) and saves CSVs to <seat_dir>/INANNA/.
    finalists=None (default): each year's opponent is whoever actually made the final two with `party`."""
    seat_dir = Path(seat_dir)
    out_dir = seat_dir / "INANNA"
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(actual_2cp=actual_2cp, flow_matrix_by_year=flow_matrix_by_year, default_flow_matrix=default_flow_matrix,
            years=years, finalists=finalists, party=party)
    t1 = run_flow_only_test(seat_dir, **kw)
    t2 = run_end_to_end_test(seat_dir, **kw)
    t1.to_csv(out_dir / "backtest_flow_only.csv", index=False)
    t2.to_csv(out_dir / "backtest_end_to_end.csv", index=False)
    if t1.empty or t2.empty:
        print("-> No backtest years were usable (see skip messages above); no summary produced.")
        return pd.DataFrame()

    p_a = party.upper()
    summary = t1[["target_year", "opponent", f"actual_{p_a}_2CP_pct", f"predicted_{p_a}_2CP_pct", "error_pts"]].rename(
        columns={f"predicted_{p_a}_2CP_pct": "flow_only_pred", "error_pts": "flow_only_error"}).merge(
        t2[["target_year", f"predicted_{p_a}_2CP_pct", "error_pts", "pred_p05", "pred_p95", "actual_in_90_interval",
            "finalist_pair_pred_prob", "primary_propagated_error", "votes_covered_pct"]].rename(
            columns={f"predicted_{p_a}_2CP_pct": "end_to_end_pred", "error_pts": "end_to_end_error"}),
        on="target_year", how="outer")
    summary.to_csv(out_dir / "backtest_2cp_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n--- INANNA BACKTEST ({p_a} 2CP %, opponent = actual final-two opponent that year; error = predicted - actual, pts) ---")
    print(summary.round(2).to_string(index=False))
    print(f"-> Saved backtest CSVs to: {out_dir}")
    return summary


def save_inanna_backtest_table_png(summary_df: pd.DataFrame, output_path: Path):
    """Renders INANNA backtest summary metrics into a clean table PNG for a quick glance."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if summary_df is None or summary_df.empty:
        return

    display_cols = [
        "target_year", "opponent", "actual_ALP_2CP_pct", "flow_only_pred", "flow_only_error",
        "end_to_end_pred", "end_to_end_error", "actual_in_90_interval", "finalist_pair_pred_prob", "primary_propagated_error"
    ]
    existing_cols = [c for c in display_cols if c in summary_df.columns]
    if not existing_cols:
        existing_cols = list(summary_df.columns)
        
    df_plot = summary_df[existing_cols].copy()
    
    for col in df_plot.columns:
        if df_plot[col].dtype in [float, np.floating]:
            df_plot[col] = df_plot[col].round(2)

    fig, ax = plt.subplots(figsize=(max(12, len(existing_cols) * 1.2), len(df_plot) * 0.8 + 2.0))
    ax.axis("off")
    
    table = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
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

    plt.suptitle("INANNA Backtest Error & Validation Summary", fontsize=14, fontweight="bold", y=0.92)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> INANNA backtest summary table PNG saved to: {output_path}")


def find_latest_enki_run(base="forecast_output", seat_name="sample") -> Path:
    """Newest <base>/<seat>/<YYYY-MM-DD>/ folder that contains ENKI's exports. Case-insensitive on the seat folder,
    so it matches what ENKI's output_folder() wrote regardless of OS. ISO dates sort correctly as plain strings."""
    base = Path(base)
    slug = seat_name.lower().replace(" ", "_")
    runs = []
    if base.exists():
        for seat_dir in base.iterdir():
            if seat_dir.is_dir() and seat_dir.name.lower() == slug:
                runs += [r for r in seat_dir.iterdir() if r.is_dir() and (r / "ENKI" / "booth_party_draws.npz").exists()]
    if not runs:
        raise FileNotFoundError(f"No ENKI run found under {base}/{seat_name}/<date>/ENKI. Run ENKI first.")
    latest = max(runs, key=lambda r: r.name)
    print(f"-> INANNA using newest ENKI run: {latest}")
    return latest


def main(seat_name="sample", output_root="forecast_output"):
    seat_path = find_latest_enki_run(output_root, seat_name)
    
    # Pair-keyed flow matrix
    flow_matrix = {
        "ALP_vs_LNP": {
            "GRN":   [0.525, 0.078, 0.397],
            "ONP":   [0.109, 0.180, 0.711],
            "OTHER": [0.450, 0.200, 0.350]
        },
        "ALP_vs_GRN": {
            "LNP":   [0.100, 0.100, 0.800],
            "ONP":   [0.100, 0.100, 0.800],
            "OTHER": [0.300, 0.300, 0.400]
        }
    }

    print("==================================================")
    print(" STEP 1: RUNNING INANNA HISTORICAL BACKTESTS")
    print("==================================================")
    
    backtest_summary = run_inanna_backtests(
        seat_dir=seat_path,
        actual_2cp={2015: 45.20, 2019: 39.98, 2023: 48.20},
        flow_matrix_by_year={2019: flow_matrix, 2023: flow_matrix},
        default_flow_matrix=flow_matrix,
        finalists=None # <-- Allows dynamic finalist detection per election/draw
    )

    print("\n==================================================")
    print(" STEP 2: RUNNING LIVE OPV PREDICTION PIPELINE")
    print("==================================================")
    
    run_opv_pipeline(
        seat_dir=seat_path, 
        party="ALP", 
        custom_flow_matrix=flow_matrix,
        finalists=None # <-- Dynamically determines finalists from ENKI's draws
    )
    
if __name__ == "__main__":
    main()