"""api.py -- the blackbox: ENKI(x) -> y.

x is ONE flat dict (or a path to a .json) holding the data, every ModelConfig setting and the RunOptions.
Anything left out takes its default; an unknown key is an error.
"""
from __future__ import annotations

import difflib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import plots
from .backtest import BacktestSpec, backtest_shares, backtest_volumes
from .config import RENAMED, ModelConfig, PollInput, RunOptions, validate
from .data import NSW_STATE_RESULTS, prepare_booths, read_frame
from .forecast import Forecast, run_forecast
from .io import (append_history, jsonable, load_x, output_folder, save_actual_primaries, save_backtest_draws,
                 save_run_meta, save_trace)
from .pipeline import check_trace
from .report import print_run_report

log = logging.getLogger("enki")
_DATA_KEYS = ("booth_results", "polls", "state_results", "projected_formal_votes", "target_year", "backtest_polls")
_TUPLE_FIELDS = ("parties", "early_keywords", "exclude_booths", "scenario_pct", "plot_ridgeline_keywords")


@dataclass
class ENKIOutput:
    forecast: Optional[Forecast] = None
    seat_summary: Optional[pd.DataFrame] = None         # seat-wide share per party: mean + scenario percentiles
    backtest: Optional[pd.DataFrame] = None
    backtest_draws: Optional[dict] = None
    volume_backtest: Optional[pd.DataFrame] = None
    config: Dict[str, Any] = field(default_factory=dict)  # the fully resolved settings that were actually run
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


def configure_logging(verbose: bool) -> None:
    """Narration goes through the 'enki' logger. ENKI(x) attaches a console handler only if the caller has not."""
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
    if cfg_kw.get("prediction_date") is None:
        cfg_kw["prediction_date"] = date.today().strftime("%Y-%m-%d")
    cfg, run = ModelConfig(**cfg_kw), RunOptions(**run_kw)
    warnings_ = validate(cfg, run)

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
        data["state_results"], src["state_results"] = NSW_STATE_RESULTS.copy(), "<built-in NSW_STATE_RESULTS>"
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
    if flat.get("backtest_polls") is not None:
        data["backtest_polls"] = read_frame(flat["backtest_polls"], "backtest_polls")
        src["backtest_polls"] = _source(flat["backtest_polls"])
    elif run.run_backtest and run.backtest_state_mode == "polls":
        raise KeyError("ENKI(x): backtest_state_mode='polls' needs 'backtest_polls' (historical polls with election_year, "
                       "election_date and date columns).")
    if run.run_backtest and run.backtest_state_mode == "oracle":
        warnings_.append("Backtest is conditional on the TRUE statewide result and says nothing about poll error. Use "
                         "backtest_state_mode='polls' (with backtest_polls) or 'noised' for an end-to-end test.")
    if run.run_backtest and run.backtest_volume_mode == "actual":
        warnings_.append("Backtest uses the TRUE booth volumes, so it says nothing about turnout error (backtest_volume_mode='modelled').")
    return Resolved(data, cfg, run, warnings_, src)


def _seat_summary(res: Forecast, cfg: ModelConfig) -> pd.DataFrame:
    lo, mid, hi = (float(v) for v in cfg.scenario_pct)
    q = res.draws_df.quantile([lo / 100, mid / 100, hi / 100]).T
    q.columns = [f"p{lo:g}", f"p{mid:g}", f"p{hi:g}"]
    q.insert(0, "mean", res.draws_df.mean())
    return q


def _guarded(y: ENKIOutput, name: str, fn) -> None:
    """Plots are a convenience: a failed figure is a warning, never a lost forecast."""
    try:
        fn()
    except Exception as e:
        y.warnings.append(f"{name} failed: {e!r}")


def _write_backtests(y: ENKIOutput, run: RunOptions, out: Path) -> None:
    for name, table in (("backtest", y.backtest), ("backtest_volume", y.volume_backtest)):
        if table is not None:
            table.to_csv(out / f"{name}.csv", index=False)
            y.files[name] = str(out / f"{name}.csv")
    if y.backtest_draws:
        save_backtest_draws(y.backtest_draws, out)
    if y.backtest is not None and run.make_plots:
        png = out / "backtest_summary_table.png"
        _guarded(y, "backtest table plot", lambda: plots.backtest_table_png(y.backtest, png, run.plot_dpi))
        y.files["backtest_table_png"] = str(png)


def _write_forecast(y: ENKIOutput, res: Forecast, cfg: ModelConfig, run: RunOptions, out: Path, seat_folder: Path, sources: dict) -> None:
    if run.make_plots:
        figs = {
            "seat_distribution.png": lambda p: plots.seat_distribution_plot(res, cfg.party, p, run.plot_dpi),
            "booth_swing_scatter.png": lambda p: plots.booth_swing_scatter(res, p, run.plot_label_all_booths, run.plot_dpi),
            "ridgeline.png": lambda p: plots.booth_ridgeline_plot(res, cfg.party, run.plot_top_n_booths, p, run.plot_ridgeline_max_draws,
                                                                  run.plot_dpi, run.plot_ridgeline_keywords),
            "booth_volume_ridgeline.png": lambda p: plots.booth_volume_ridgeline(res, p, run.plot_top_n_booths, run.plot_volume_log_scale, run.plot_dpi),
        }
        for fname, fn in figs.items():
            _guarded(y, f"plot {fname}", lambda fn=fn, fname=fname: fn(out / fname))
            y.files[fname] = str(out / fname)
    res.save(out)
    if res.trace:
        save_trace(out, res.trace)
        y.files["trace"] = str(out / "trace.npz")
    save_actual_primaries(res.prepared_df, cfg.parties, out)
    save_run_meta(out, cfg, sources.get("booth_results", ""), sources.get("polls", ""))
    y.seat_summary.to_csv(out / "seat_primary_summary.csv")
    append_history(res, seat_folder, cfg.prediction_date)
    y.files["forecast_dir"] = str(out)


def ENKI(x) -> ENKIOutput:
    """The blackbox. ENKI(x) -> y."""
    r = resolve(x)
    cfg, run, data = r.cfg, r.run, r.data
    configure_logging(run.verbose)
    y = ENKIOutput(config=json.loads(json.dumps(asdict(cfg), default=jsonable)), run=json.loads(json.dumps(asdict(run), default=jsonable)),
                   data_sources=r.sources, warnings=r.warnings)

    out: Optional[Path] = None
    seat_folder: Optional[Path] = None
    if run.output_root:
        seat_folder = output_folder(run.output_root, cfg.seat_name, cfg.prediction_date)
        out = seat_folder / "ENKI"
        out.mkdir(parents=True, exist_ok=True)
        y.output_dir = str(out)

    df = prepare_booths(data["booth_results"], cfg)          # cleaned and checked once; every stage below reads this table
    state_df = data["state_results"]

    if run.run_backtest:
        log.info("RUNNING HISTORICAL BACKTEST")
        spec = BacktestSpec(run.backtest_state_mode, run.backtest_volume_mode, data.get("backtest_polls"), run.backtest_poll_horizon_days)
        bt = backtest_shares(df, state_df, cfg, run.backtest_targets, return_draws=True, spec=spec)
        y.backtest, y.backtest_draws = bt.table, bt.draws
    if run.run_volume_backtest:
        y.volume_backtest = backtest_volumes(df, cfg, run.backtest_volume_sims, run.backtest_targets)
    if out is not None:
        _write_backtests(y, run, out)

    if run.run_forecast:
        log.info("RUNNING FORECAST: %s (target %s)", cfg.seat_name, cfg.election_date)
        res = run_forecast(df, state_df, data["polls"], cfg, data.get("projected_formal_votes"), data.get("target_year"),
                           run.run_attribution, run.record_trace)
        y.forecast, y.trace = res, res.trace
        if res.trace is not None:
            y.warnings += check_trace(res.trace)
        y.seat_summary = _seat_summary(res, cfg)
        if out is not None:
            _write_forecast(y, res, cfg, run, out, seat_folder, r.sources)

    if out is not None:
        (out / "inputs_resolved.json").write_text(
            json.dumps({"config": y.config, "run": y.run, "data_sources": y.data_sources}, indent=2, default=jsonable), encoding="utf-8")
        y.files["inputs_resolved"] = str(out / "inputs_resolved.json")
    if run.print_report:
        print_run_report(y, cfg)
        if out is not None:
            print(f"\nFinished: seat '{cfg.seat_name}'. All outputs written to: {out}/")
    return y


def main(seat_name: str = "sample", booth_data: str = "sample_booth_results.csv", polls_data: str = "sample_polls.csv",
         output_root: str = "forecast_output", statewide_historical: Optional[pd.DataFrame] = None,
         registered_voters: Optional[int] = 62169, expected_turnout: float = 0.91, **overrides) -> ENKIOutput:
    x = dict(seat_name=seat_name, booth_results=booth_data, polls=polls_data, output_root=output_root, state_results=statewide_historical,
             registered_electors=registered_voters, expected_turnout_rate=expected_turnout)
    x.update(overrides)
    return ENKI(x)


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
        return main(**over)
    x = load_x(path)
    x.update(over)
    return ENKI(x)
