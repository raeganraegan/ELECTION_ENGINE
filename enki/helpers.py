"""helpers.py -- non-math: IO, plots, strategy PDF, terminal report.
Nothing here fits, simulates, or changes a number. enki.py imports load_x from here.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from . import enki, plots
from .enki import ModelConfig, ENKIOutput, Forecast, jsonable
from .report import print_run_report

log = logging.getLogger("enki")

# ----------------------------------------------------------------------------- IO
def output_folder(base: str, seat_name: str, pred_date: Optional[str] = None, sub: Optional[str] = None) -> Path:
    """<base>/<Seat_Name>/<YYYY-MM-DD>/[sub/]/"""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", seat_name).strip("_") or "Seat"
    p = Path(base) / slug / (pred_date or date.today().isoformat())
    if sub:
        p = p / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_history(res: Forecast, seat_folder: Path, pred_date: str) -> Path:
    row = {"date": pred_date, **{k: res.params.get(k, np.nan) for k in ("S_next", "S_next_low", "S_next_high", "beta")}}
    for _, r in res.seat.iterrows():
        row[f"seat_{r['scenario']}"] = r["primary_share_pct"]
    path = seat_folder / "history.csv"
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["date"] != pred_date], new], ignore_index=True).sort_values("date")
    new.to_csv(path, index=False)
    return path


def save_actual_primaries(prepared: pd.DataFrame, parties: Sequence[str], folder) -> None:
    tables_dir = Path(folder) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"year": int(y), "formal_total": int(g["n"].sum()), **{p: int(g[f"y_{p}"].sum()) for p in parties}}
            for y, g in prepared.groupby("year")]
    pd.DataFrame(rows).to_csv(tables_dir / "actual_primaries.csv", index=False)


def save_run_meta(folder, cfg: ModelConfig, booths_path: str, polls_path: str) -> None:
    meta_dir = Path(folder) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {"seat_name": cfg.seat_name, "party": cfg.party, "parties": list(cfg.parties), "prediction_date": cfg.prediction_date,
            "election_date": cfg.election_date, "booths_file": str(booths_path), "polls_file": str(polls_path),
            "onp_mode": cfg.onp_mode, "saved_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    (meta_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def save_backtest_draws(draws_by_year: dict, folder: Union[str, Path]) -> None:
    bt_dir = Path(folder) / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)
    for y, d in draws_by_year.items():
        np.savez_compressed(bt_dir / f"backtest_draws_{int(y)}.npz", booths=np.array(d["booths"]), parties=np.array(d["parties"]),
                            booth_party_draws=d["booth_party_draws"], actual_party_votes=d["actual_party_votes"], actual_n=d["actual_n"])


def save_trace(folder, trace: dict) -> None:
    draws_dir = Path(folder) / "draws"
    tables_dir = Path(folder) / "tables"
    meta_dir = Path(folder) / "metadata"
    for d in (draws_dir, tables_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    scalars: Dict[str, Any] = {}
    tables: List[str] = []
    for k, v in trace.items():
        if k == "meta":
            continue
        if isinstance(v, pd.DataFrame):
            v.to_csv(tables_dir / f"trace_{k}.csv", index=False)
            tables.append(k)
        elif isinstance(v, (str, bool, int, float, np.generic)):
            scalars[k] = v.item() if isinstance(v, np.generic) else v
        else:
            arrays[k] = np.asarray(v)
    np.savez_compressed(draws_dir / "trace.npz", **arrays)
    (meta_dir / "trace_meta.json").write_text(
        json.dumps({"scalars": scalars, "tables": tables, "meta": trace.get("meta", {})}, indent=2, default=jsonable), encoding="utf-8")


def load_trace(folder) -> dict:
    out = Path(folder)
    meta_dir = out / "metadata"
    draws_dir = out / "draws"
    tables_dir = out / "tables"
    meta = json.loads((meta_dir / "trace_meta.json").read_text(encoding="utf-8"))
    trace: Dict[str, Any] = {}
    with np.load(draws_dir / "trace.npz", allow_pickle=False) as z:
        for k in z.files:
            trace[k] = z[k]
    trace.update(meta.get("scalars", {}))
    for k in meta.get("tables", []):
        df = pd.read_csv(tables_dir / f"trace_{k}.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        trace[k] = df
    trace["meta"] = meta.get("meta", {})
    return trace


# ----------------------------------------------------------------------------- persistence orchestration
def write_all(y: ENKIOutput, out_root: str) -> ENKIOutput:
    """Persist a finished ENKIOutput to <out_root>/<Seat>/<date>/ENKI/."""
    cfg = ModelConfig(**y.config)
    run = enki.RunOptions(**y.run)
    sources = y.data_sources
    seat_folder = output_folder(out_root, cfg.seat_name, cfg.prediction_date)
    out = seat_folder / "ENKI"
    out.mkdir(parents=True, exist_ok=True)
    y.output_dir = str(out)

    res = y.forecast
    if run.run_forecast and res is not None:
        _write_forecast(y, res, cfg, run, out, seat_folder, sources)

    (out / "inputs_resolved.json").write_text(
        json.dumps({"config": y.config, "run": y.run, "data_sources": sources}, indent=2, default=jsonable), encoding="utf-8")
    y.files["inputs_resolved"] = str(out / "inputs_resolved.json")

    if run.print_report:
        print_run_report(y, cfg)
        print(f"\nFinished: seat '{cfg.seat_name}'. All outputs written to: {out}/")
    return y


def _guarded(y: ENKIOutput, name: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        y.warnings.append(f"{name} failed: {e!r}")


def _write_forecast(y: ENKIOutput, res: Forecast, cfg: ModelConfig, run: enki.RunOptions,
                    out: Path, seat_folder: Path, sources: Dict[str, str]) -> None:
    if run.make_plots:
        figs = {
            "seat_distribution.png": lambda p: plots.seat_distribution_plot(res, cfg.party, p, run.plot_dpi),
            "booth_swing_scatter.png": lambda p: plots.booth_swing_scatter(res, p, run.plot_label_all_booths, run.plot_dpi),
            "ridgeline.png": lambda p: plots.booth_ridgeline_plot(res, cfg.party, run.plot_top_n_booths, p, run.plot_ridgeline_max_draws,
                                                                   run.plot_dpi, run.plot_ridgeline_keywords),
            "booth_volume_ridgeline.png": lambda p: plots.booth_volume_ridgeline(res, p, run.plot_top_n_booths,
                                                                                  run.plot_volume_log_scale, run.plot_dpi),
        }
        for fname, fn in figs.items():
            _guarded(y, f"plot {fname}", lambda fn=fn, fname=fname: fn(out / fname))
            y.files[fname] = str(out / fname)
    live_out = out / "live_2027"
    live_out.mkdir(parents=True, exist_ok=True)
    pd.Series(res.params, name="value").to_csv(live_out / "params.csv")
    res.booths.to_csv(live_out / "booths.csv", index=False)
    res.tide.to_csv(live_out / "tide.csv", index=False)
    res.booth_forecast.to_csv(live_out / "forecast.csv", index=False)
    res.seat.to_csv(live_out / "seat.csv", index=False)
    if res.diagnostics is not None:
        res.diagnostics.to_csv(live_out / "diagnostics.csv", index=False)
    if res.attribution is not None:
        res.attribution.to_csv(live_out / "uncertainty_attribution.csv", index=False)
    np.savez_compressed(live_out / "volume_draws.npz", names=np.array(res.booth_names), V=res.volume_matrix)
    np.savez_compressed(live_out / "booth_party_draws.npz", booth_party_draws=res.booth_party_draws,
                        booths=np.array(res.booth_names), parties=np.array(res.party_names))
    _guarded(y, "plot seat_distribution.png",
             lambda: plots.seat_distribution_plot(res, cfg.party, live_out / "seat_distribution.png", run.plot_dpi))
    y.files["seat_distribution"] = str(live_out / "seat_distribution.png")
    if res.trace:
        save_trace(out, res.trace)
        y.files["trace"] = str(out / "trace.npz")
    save_actual_primaries(res.prepared_df, cfg.parties, out)
    save_run_meta(out, cfg, sources.get("booth_results", ""), sources.get("polls", ""))
    y.seat_summary.to_csv(out / "seat_primary_summary.csv")
    append_history(res, seat_folder, cfg.prediction_date)
    if res.opv is not None:
        _write_opv(y, res, run, out)
    if run.make_pdf and res.opv is not None:
        _guarded(y, "strategy PDF", lambda: _write_pdf(y, cfg, run, out))
    y.files["forecast_dir"] = str(out)


def _write_opv(y: ENKIOutput, res: Forecast, run: enki.RunOptions, out: Path) -> None:
    o = res.opv
    d = out / "opv"
    d.mkdir(parents=True, exist_ok=True)
    for name, table, idx in (("booth_2cp", o.booth_2cp, False), ("seat_2cp_draws", o.seat_draws, False),
                             ("finalist_pair_probs", o.pair_probs, True), ("win_probs", o.win_probs, True),
                             ("affinities", o.affinities, False)):
        table.to_csv(d / f"{name}.csv", index=idx)
        y.files[f"opv_{name}"] = str(d / f"{name}.csv")
    if run.make_plots:
        figs = {"booth_raw_first_preference_breakdown.png": plots.opv_first_preference_breakdown,
                "booth_2cp_exhausted_breakdown.png": plots.opv_2cp_breakdown,
                "booth_2cp_uncertainty_forest.png": plots.opv_uncertainty_forest,
                "exhausted_vs_support_scatter.png": plots.opv_exhausted_scatter,
                "seat_outcome_distribution.png": plots.opv_seat_outcome}
        for fname, fn in figs.items():
            _guarded(y, f"plot opv/{fname}", lambda fn=fn, fname=fname: fn(o, d / fname, run.plot_dpi))
            y.files[f"opv_{fname}"] = str(d / fname)


def _write_pdf(y: ENKIOutput, cfg: ModelConfig, run: enki.RunOptions, out: Path) -> None:
    slug = "".join(ch if ch.isalnum() else "_" for ch in cfg.seat_name).strip("_") or "Seat"
    y.files["strategy_pdf"] = str(build_strategy_pdf(y, cfg, out / "report" / f"{slug}_strategy.pdf", run.pdf_tier_cuts))


# ----------------------------------------------------------------------------- strategy PDF (absorbed from module9.py)
NAVY, RED, BLUE, PAGE_BG, INK, GREY, LINE = "#161f32", "#dc2626", "#2563eb", "#f8fafc", "#111827", "#6b7280", "#e5e7eb"
TIERS = {"BASE FIREWALL": ("#166534", "#dcfce7"), "SWING PRECINCT": ("#92400e", "#fef3c7"),
         "CONTESTABLE / TOUGH": ("#991b1b", "#fee2e2"), "SPECIAL / EARLY": ("#6b21a8", "#f3e8ff")}
EARLY_WORDS = ("PRE-POLL", "PREPOLL", "PRE POLL", "POSTAL", "DECLARATION", "ABSENT", "IVOTE", "EARLY")


@dataclass
class Strategy:
    seat: str
    party: str
    opponent: str
    year: str
    prediction_date: str
    win_prob: float
    status: str
    pair_prob: float
    median_2cp: float
    p05_2cp: float
    p95_2cp: float
    mean_exhaust: float
    exh_min: float
    exh_max: float
    pair_probs: Dict[str, float]
    win_probs: Dict[str, float]
    enrolled: Optional[int]
    venues: pd.DataFrame
    notes: List[str]
    diag_line: str
    diag_ok: bool


def _status(p: float) -> str:
    if p >= 0.90:
        return "SAFE"
    if p >= 0.70:
        return "LEANS HOLD"
    if p > 0.30:
        return "TOSS-UP"
    if p > 0.10:
        return "LEANS AGAINST"
    return "LONG SHOT"


def _kind(name: str) -> str:
    u = name.upper()
    if "POSTAL" in u:
        return "Postal vote hub"
    if any(k in u for k in ("PRE-POLL", "PREPOLL", "PRE POLL")):
        return "Pre-poll centre"
    return "Early / special votes" if any(k in u for k in EARLY_WORDS) else "Polling place"


def _directive(row: pd.Series) -> str:
    t = row["tier"]
    if t == "SPECIAL / EARLY":
        return ("Secure postal return tracking and direct chase." if "Postal" in row["kind"]
                else "High early-vote volume: deploy early teams from day one.")
    base = {"BASE FIREWALL": "Hold the base: turnout and volunteer presence.",
            "SWING PRECINCT": "Persuasion focus and active presence.",
            "CONTESTABLE / TOUGH": "Targeted persuasion to narrow the margin."}[t]
    if t != "BASE FIREWALL" and row["opp_pref_edge"] > 0:
        base += " Opponent gains more from preferences here: consider Just Vote 1 material."
    return base


def _mandate(row: pd.Series, n: int) -> str:
    if row["rank"] <= max(1, math.ceil(0.2 * n)):
        return ("Highest priority: senior teams, bank early votes." if row["early"]
                else "Highest priority: concentrate experienced volunteers and persuasion resources.")
    if row["rank"] <= math.ceil(0.5 * n):
        return "Targeted door-knocking and persuasion push."
    return "Core baseline canvassing and postal follow-up."


def _fmt(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def module9_data(y: ENKIOutput, cfg: ModelConfig, tier_cuts: Sequence[float] = (0.40, 0.80)) -> Strategy:
    res, o = y.forecast, y.opv
    if res is None or o is None:
        raise ValueError("Module 9 needs a forecast with the OPV step (run_forecast and run_opv).")
    A, B = o.report_pair
    client = cfg.party.upper()
    if client != A:
        raise ValueError(f"{client} is not in the most frequent finalist pairing ({A} v {B}), so there is no {client} 2CP to report. "
                         f"Pairing probabilities: {dict(o.pair_probs.round(3))}.")
    lo_cut, hi_cut = float(tier_cuts[0]), float(tier_cuts[1])

    sd = o.seat_draws
    sub = sd[sd["pair"].isin([f"{A} v {B}", f"{B} v {A}"])]
    q5, q50, q95 = np.nanpercentile(sub[f"{A}_2CP_pct"], [5, 50, 95])
    win = float(o.win_probs.get(A, 0.0))

    bt = o.booth_2cp.copy()
    kinds = res.prepared_df.groupby("booth")["btype"].max() if "btype" in res.prepared_df.columns else pd.Series(dtype=int)
    v = pd.DataFrame({
        "venue": bt["booth"].astype(str), "votes": bt["expected_formal_votes"], "share": bt[f"{A}_2cp_share_mean"],
        "lo": bt[f"{A}_2cp_p05"], "hi": bt[f"{A}_2cp_p95"], "q": bt[f"{A}_booth_win_prob"].fillna(0.5),
        "exh": bt["exhausted_share_of_total_mean"],
        "opp_pref_edge": ((bt[f"{B}_2cp_votes_mean"] - bt[f"{B}_primary_votes_mean"])
                          - (bt[f"{A}_2cp_votes_mean"] - bt[f"{A}_primary_votes_mean"])) if f"{B}_primary_votes_mean" in bt else 0.0})
    v["kind"] = v["venue"].map(_kind)
    v["early"] = v["venue"].map(lambda b: bool(kinds.get(b, 0))) | (v["kind"] != "Polling place")
    v["tier"] = np.where(v["early"], "SPECIAL / EARLY", np.where(v["q"] >= hi_cut, "BASE FIREWALL",
                         np.where(v["q"] < lo_cut, "CONTESTABLE / TOUGH", "SWING PRECINCT")))
    v["ecv"] = v["votes"] * (1.0 - v["exh"] / 100.0)
    v["score"] = v["ecv"] * 4.0 * v["q"] * (1.0 - v["q"])
    v["rank"] = v[["score", "ecv"]].apply(tuple, axis=1).rank(method="first", ascending=False).astype(int)
    n = len(v)
    v["directive"] = v.apply(_directive, axis=1)
    v["mandate"] = v.apply(lambda r: _mandate(r, n), axis=1)
    v = v.sort_values("votes", ascending=False).reset_index(drop=True)

    p = res.params
    bad = (p["rhat_max"] > cfg.rhat_threshold) or (p["ess_bulk_min"] < cfg.ess_threshold) or (p["divergences"] > 0)
    diag = (f"Posterior: {int(p['posterior_draws'])} draws, max R-hat {p['rhat_max']:.3f}, min ESS {p['ess_bulk_min']:.0f}, "
            f"{int(p['divergences'])} divergences" + (". SAMPLER DIAGNOSTICS ARE POOR: DO NOT RELY ON THESE FIGURES." if bad else "."))

    aff = o.affinities
    n_prior = int(aff["basis"].isin(["default (source average)", "proxy"]).sum())
    vol_mode = (y.trace or {}).get("s5_mode", "")
    onp = {"none": "One Nation is not modelled separately (any ONP vote sits inside OTHER).",
           "brian": "One Nation enters as a fixed transfer from the other parties (onp_mode brian).",
           "entry": "One Nation enters as an entrant scaled from the statewide poll, with assumed siphon weights (onp_mode entry)."}.get(cfg.onp_mode, "")
    notes = [
        f"2CP shares and intervals are conditional on the {A} v {B} finalist pairing ({100 * o.report_pair_prob:.0f}% of draws). "
        f"Win probabilities count every draw and every pairing.",
        f"Preference flows are seat-wide, not booth-specific. {n_prior} of {len(aff)} source-to-candidate preference affinities have no flow "
        f"table behind them and use a wide prior (opv_affinity_sd {cfg.opv_affinity_sd:g}); see opv/affinities.csv.",
        onp,
        "Not modelled: incumbency, retirements, strong independents, candidate quality, boundary changes between the elections used.",
        ("Venue volumes come from registered electors x expected turnout, split by the last election's venue shares; early voting growth "
         "is not projected." if vol_mode == "electors" else "Venue volumes come from the volume model." if vol_mode else ""),
        f"Tiers (P(win venue) cut points {lo_cut:g} / {hi_cut:g}), ECV, priority score (ECV x 4q(1-q)), ranks and directives are rule-based "
        "heuristics, not model results. The seat is decided on the sum of all venues, and the model has no information on persuadability "
        "or volunteer productivity.",
    ]
    en = cfg.registered_electors
    return Strategy(seat=cfg.seat_name, party=A, opponent=B, year=str(cfg.election_date)[:4], prediction_date=str(cfg.prediction_date),
                    win_prob=win, status=_status(win), pair_prob=float(o.report_pair_prob), median_2cp=float(q50), p05_2cp=float(q5),
                    p95_2cp=float(q95), mean_exhaust=float(sub["exhausted_pct_formal"].mean()), exh_min=float(v["exh"].min()),
                    exh_max=float(v["exh"].max()), pair_probs={k: float(x) for k, x in o.pair_probs.items()},
                    win_probs={k: float(x) for k, x in o.win_probs.items()}, enrolled=int(en) if en else None, venues=v,
                    notes=[x for x in notes if x], diag_line=diag, diag_ok=not bad)


def build_strategy_pdf(y: ENKIOutput, cfg: ModelConfig, path, tier_cuts: Sequence[float] = (0.40, 0.80)) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, NextPageTemplate, PageBreak, PageTemplate,
                                        Paragraph, Spacer, Table, TableStyle)
    except ImportError as e:
        raise ImportError("Module 9 needs reportlab: pip install reportlab") from e

    S = module9_data(y, cfg, tier_cuts)
    v = S.venues
    n = len(v)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = A4
    M, HDR = 40, 92
    FW = W - 2 * M
    C = colors.HexColor

    def P(text, size=7, bold=False, color=INK, leading=None, align=0):
        return Paragraph(text, ParagraphStyle("p", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                                              leading=leading or size * 1.3, textColor=C(color), alignment=align))

    class Pill(Flowable):
        def __init__(self, text, fg, bg, size=6.2):
            super().__init__()
            self.t, self.fg, self.bg, self.s = text, fg, bg, size

        def wrap(self, aw, ah):
            self.w, self.h = stringWidth(self.t, "Helvetica-Bold", self.s) + 10, self.s + 5.5
            return self.w, self.h

        def draw(self):
            c = self.canv
            c.setFillColor(C(self.bg)); c.roundRect(0, 0, self.w, self.h, 3, stroke=0, fill=1)
            c.setFillColor(C(self.fg)); c.setFont("Helvetica-Bold", self.s); c.drawString(5, 3.0, self.t)

    class Kpis(Flowable):
        def __init__(self, cards):
            super().__init__()
            self.cards = cards

        def wrap(self, aw, ah):
            self.w = aw
            return aw, 52

        def draw(self):
            c, k, gap = self.canv, len(self.cards), 8
            cw = (self.w - gap * (k - 1)) / k
            for i, (val, lab, accent) in enumerate(self.cards):
                x = i * (cw + gap)
                c.setFillColor(colors.white); c.setStrokeColor(C(LINE)); c.roundRect(x, 0, cw, 48, 4, stroke=1, fill=1)
                if accent:
                    c.setFillColor(C(RED)); c.rect(x, 0, 3, 48, stroke=0, fill=1)
                size = 15
                while stringWidth(val, "Helvetica-Bold", size) > cw - 14 and size > 7:
                    size -= 0.5
                c.setFillColor(C(RED if accent else INK)); c.setFont("Helvetica-Bold", size); c.drawCentredString(x + cw / 2, 27, val)
                c.setFillColor(C(GREY)); c.setFont("Helvetica-Bold", 5.6); c.drawCentredString(x + cw / 2, 12, lab)

    class Head(Flowable):
        def __init__(self, text):
            super().__init__()
            self.t = text

        def wrap(self, aw, ah):
            return aw, 20

        def draw(self):
            c = self.canv
            c.setFillColor(C(BLUE)); c.rect(0, 3, 3, 12, stroke=0, fill=1)
            c.setFillColor(C(NAVY)); c.setFont("Helvetica-Bold", 10); c.drawString(10, 5, self.t)

    def box(title, paras, accent=BLUE):
        inner = [P(title, 7.2, True, accent)] + [P(t, 7.2, leading=10) for t in paras]
        t = Table([[inner]], colWidths=[FW])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C("#eff6ff")), ("LINEBEFORE", (0, 0), (0, -1), 3, C(accent)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                               ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        return t

    def table(head, rows, widths, repeat=1):
        t = Table([[P(h, 5.8, True, "#ffffff") for h in head]] + rows, colWidths=widths, repeatRows=repeat)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), C(NAVY)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("LINEBELOW", (0, 1), (-1, -1), 0.4, C(LINE)), ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                               ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                               ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
        return t

    def venue_cell(r, extra=""):
        return [P(f"<b>{escape(r['venue'])}</b>", 7.2), P(escape(r["kind"]) + extra, 5.8, color=GREY)]

    def share_cell(r):
        return [P(f"<b>{r['share']:.1f}%</b> {escape(S.party)}", 7), P(f"90% range {r['lo']:.0f}-{r['hi']:.0f}%", 5.6, color=GREY)]

    seat_u = S.seat.upper()
    subtitle = f"{n} voting venues | {S.party} two-candidate-preferred against {S.opponent} | State Election {S.year}"
    kp = [(f"{100 * S.win_prob:.1f}%", f"{S.party} WIN PROBABILITY", True), (f"{S.median_2cp:.1f}%", f"MEDIAN {S.party} 2CP SHARE", False),
          (f"{S.mean_exhaust:.1f}%", "MEAN OPV EXHAUSTION", False),
          (_fmt(S.enrolled) if S.enrolled else "n/a", "ENROLLED VOTERS", False)]

    s1 = [Kpis(kp), Spacer(1, 10), Head("1. MODEL RESULT & EXHAUSTION DYNAMICS"), Spacer(1, 3)]
    others = ", ".join(f"{k} {100 * x:.0f}%" for k, x in S.pair_probs.items() if k.replace(" v ", "") != (S.party + S.opponent) and
                       k not in (f"{S.party} v {S.opponent}", f"{S.opponent} v {S.party}") and x >= 0.01)
    top_exh = ", ".join(v.sort_values("exh", ascending=False)["venue"].head(3))
    s1.append(box(f"Classification: {S.status} ({100 * S.win_prob:.1f}% {S.party} win probability)", [
        f"The model projects a median {S.party} two-candidate-preferred share of <b>{S.median_2cp:.1f}%</b> against {S.opponent} "
        f"(90% interval {S.p05_2cp:.1f}% to {S.p95_2cp:.1f}%), in {100 * S.pair_prob:.0f}% of draws, where the final pairing is {S.party} v {S.opponent}. "
        f"The win probability counts all draws" + (f", including other pairings ({escape(others)})." if others else "."),
        f"Venue exhaustion runs from {S.exh_min:.1f}% to {S.exh_max:.1f}% of formal votes; highest at {escape(top_exh)}. "
        "Exhaustion comes from the preference-flow assumptions, not from observed ballots."]))
    s1 += [Spacer(1, 10), Head(f"2. COMPLETE {n}-VENUE PRECINCT MATRIX"), Spacer(1, 3)]
    rows = []
    for _, r in v.iterrows():
        fg, bg = TIERS[r["tier"]]
        rows.append([venue_cell(r), Pill(r["tier"], fg, bg), P(f"<b>{_fmt(r['votes'])}</b>", 7), share_cell(r),
                     P(f"<b>{r['exh']:.1f}%</b>", 7), P(escape(r["directive"]), 6.4, leading=8)])
    s1.append(table(["VENUE / LOCATION", "PRECINCT TIER", "PROJ. VOTES", f"FORECAST {S.party} 2CP", "EXHAUST.", "FIELD ACTION DIRECTIVE"],
                    rows, [128, 86, 48, 76, 42, FW - 380]))

    r2 = v.sort_values("rank")
    s2 = [Kpis(kp[:3] + [(f"{n} Venues", "RANKED CATCHMENTS", False)]), Spacer(1, 10), Head("1. WEIGHTING METHODOLOGY"), Spacer(1, 3),
          box("Effective Continuing Votes and marginality", [
              "Venues are ranked by <b>priority score = ECV x 4q(1-q)</b>, where ECV = projected votes x (1 - exhaustion) and q is the model "
              f"probability that {escape(S.party)} wins the venue's two-candidate count. Marginality 4q(1-q) is 1 for a coin-flip venue and falls to 0 for a "
              "decided one. This is a rule-based way to point attention at large, close venues; it is not a model result."]),
          Spacer(1, 10), Head(f"2. COMPLETE {n}-VENUE RANKED CANVASSING PRIORITY TABLE"), Spacer(1, 3)]
    rows = []
    for _, r in r2.iterrows():
        fg, bg = TIERS[r["tier"]]
        rows.append([P(f"<b>#{r['rank']}</b>", 7.2), venue_cell(r, f" | ECV {_fmt(r['ecv'])} | score {_fmt(r['score'])}"),
                     Pill(r["tier"], fg, bg), P(f"<b>{_fmt(r['votes'])}</b>", 7), share_cell(r), P(f"<b>{r['exh']:.1f}%</b>", 7),
                     P(escape(r["mandate"]), 6.4, leading=8)])
    s2.append(table(["RANK", "VENUE / CATCHMENT", "PRECINCT TIER", "PROJ. VOTES", f"FORECAST {S.party} 2CP", "EXHAUST.", "CANVASSING MANDATE"],
                    rows, [26, 138, 84, 44, 70, 38, FW - 400]))

    t1, t2, t3 = r2.iloc[:3], r2.iloc[3:5], r2.iloc[5:]
    tot = float(v["votes"].sum())
    wshare = lambda d: float((d["share"] * d["votes"]).sum() / max(d["votes"].sum(), 1.0))
    top = r2.iloc[0]["venue"]
    s3 = [Kpis([(top, "TOP PRIORITY VENUE", True), (f"{100 * t1['votes'].sum() / tot:.0f}%", "OF PROJECTED VOTES IN TIER 1", False),
                (f"{S.mean_exhaust:.1f}%", "MEAN OPV EXHAUSTION", False), (f"{n} Venues", "IN THIS PLAN", False)]), Spacer(1, 10),
          Head("1. DEPLOYMENT FRAMEWORK"), Spacer(1, 3),
          box("Concentrate on high-score venues", [
              "Strongest teams and resources go to the highest priority scores (Section II); mid-ranked venues get peak-hour presence; the rest "
              "keep baseline presence and postal follow-up. Tiers here are the top 3, the next 2, and the remainder by priority score."]),
          Spacer(1, 10), Head("2. TIERED PRIORITY MATRIX"), Spacer(1, 3)]
    rows = []
    for _, r in t1.iterrows():
        rows.append([[Pill("TIER 1 (PRIMARY FOCUS)", "#991b1b", "#fee2e2")], venue_cell(r),
                     [P(f"<b>{_fmt(r['votes'])} votes</b>", 7), P(f"({r['share']:.1f}% {escape(S.party)})", 6, color=GREY)],
                     P("<b>Maximum deployment.</b> " + escape(r["mandate"]), 6.6, leading=8.6)])
    if len(t2):
        rows.append([[Pill("TIER 2 (SUPPORTED)", "#92400e", "#fef3c7")], [P("<b>" + escape(" & ".join(t2["venue"])) + "</b>", 7.2)],
                     [P(f"<b>{_fmt(t2['votes'].sum())} votes</b>", 7), P(f"({wshare(t2):.1f}% {escape(S.party)})", 6, color=GREY)],
                     P("<b>Targeted shifts.</b> Staffed through peak hours to manage preference flows and persuasion.", 6.6, leading=8.6)])
    if len(t3):
        rows.append([[Pill("TIER 3 (MAINTAINED)", "#1e293b", "#e2e8f0")], [P(f"<b>Remaining {len(t3)} venues</b>", 7.2)],
                     [P(f"<b>{_fmt(t3['votes'].sum())} votes</b>", 7), P(f"({100 * t3['votes'].sum() / tot:.0f}% of total, {wshare(t3):.1f}% {escape(S.party)})", 6, color=GREY)],
                     P("<b>Core presence maintained.</b> Baseline local volunteers, backed by central postal chasing.", 6.6, leading=8.6)])
    s3.append(table(["PRIORITY TIER", "CORE VENUE / LOCATION", "VOTE VOLUME", "STRENGTH & DEPLOYMENT MANDATE"], rows, [112, 138, 84, FW - 334]))
    s3 += [Spacer(1, 12), Head("3. METHOD NOTES & LIMITATIONS"), Spacer(1, 3),
           box("Read before using these figures", [f"{i + 1}. {escape(t)}" for i, t in enumerate(S.notes)] +
               [f"<font color='{RED if not S.diag_ok else GREY}'>{escape(S.diag_line)}</font>"], accent=RED if not S.diag_ok else BLUE)]

    class Numbered(rl_canvas.Canvas):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._saved = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved)
            for st in self._saved:
                self.__dict__.update(st)
                self.setFont("Helvetica-Bold", 6.2); self.setFillColor(C(GREY))
                self.drawRightString(W - M, 24, f"Page {self._pageNumber} of {total}")
                super().showPage()
            super().save()

    def furniture(c, doc):
        c.saveState()
        c.setFillColor(C(PAGE_BG)); c.rect(0, 0, W, H, stroke=0, fill=1)
        c.setStrokeColor(C(LINE)); c.line(M, 36, W - M, 36)
        c.setFillColor(C(GREY)); c.setFont("Helvetica-Bold", 6.2)
        c.drawString(M, 24, f"CONFIDENTIAL - ENKI ELECTORAL ENGINE | {seat_u} | FORECAST {S.prediction_date}")
        c.restoreState()

    def header(tag, title, sub):
        def draw(c, doc):
            furniture(c, doc)
            c.saveState()
            c.setFillColor(C(NAVY)); c.rect(0, H - HDR, W, HDR, stroke=0, fill=1)
            c.setFillColor(C(RED)); c.roundRect(M, H - 34, stringWidth(tag, "Helvetica-Bold", 6.5) + 12, 12, 2, stroke=0, fill=1)
            c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6.5); c.drawString(M + 6, H - 30.5, tag)
            c.setFont("Helvetica-Bold", 15); c.drawString(M, H - 56, title)
            c.setFillColor(C("#9ca3af")); c.setFont("Helvetica", 7); c.drawString(M, H - 70, sub)
            c.setFont("Helvetica", 6.3); c.drawRightString(W - M, H - 30, "Target Electorate")
            c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 13); c.drawRightString(W - M, H - 47, seat_u)
            c.setFillColor(C("#9ca3af")); c.setFont("Helvetica", 6.3); c.drawRightString(W - M, H - 59, f"{n} Total Venues")
            c.restoreState()
        return draw

    first = Frame(M, 46, FW, H - HDR - 14 - 46, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="f")
    cont = Frame(M, 46, FW, H - 36 - 46, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="c")
    heads = [("ENKI MODULE 9 • SECTION I", "COMPLETE PRECINCT & DECLARATION MATRIX", subtitle),
             ("ENKI MODULE 9 • SECTION II", "WEIGHTED CANVASSING PRIORITY MATRIX", f"All {n} venues ranked by effective continuing votes and marginality | State Election {S.year}"),
             ("ENKI MODULE 9 • SECTION III", f"RESOURCE PRIORITISATION STRATEGY: {seat_u}", f"Priority deployment matrix | State Election {S.year}")]
    templates = []
    for i, h in enumerate(heads, 1):
        templates += [PageTemplate(id=f"s{i}f", frames=[first], onPage=header(*h)), PageTemplate(id=f"s{i}c", frames=[cont], onPage=furniture)]
    doc = BaseDocTemplate(str(path), pagesize=A4, pageTemplates=templates, title=f"ENKI Module 9 - {S.seat}", author="ENKI",
                          leftMargin=M, rightMargin=M, topMargin=0, bottomMargin=0)
    story = [NextPageTemplate("s1c")] + s1 + [NextPageTemplate("s2f"), PageBreak(), NextPageTemplate("s2c")] + s2 + \
            [NextPageTemplate("s3f"), PageBreak(), NextPageTemplate("s3c")] + s3
    doc.build(story, canvasmaker=Numbered)
    return path
