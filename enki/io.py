"""io.py -- reading the blackbox input x and writing run files. Nothing here touches the model."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import ModelConfig


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    return str(o)


def load_x(x) -> Dict[str, Any]:
    """dict | path to .json -> one flat dict. Groups ('data', 'config', 'run') are merged; a key may not appear twice."""
    if isinstance(x, (str, Path)):
        x = json.loads(Path(x).read_text(encoding="utf-8"))
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


# ----------------------------------------------------------------------------- run folder
def output_folder(base: str, seat_name: str, pred_date: Optional[str] = None) -> Path:
    """<base>/<Seat_Name>/<YYYY-MM-DD>/"""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", seat_name).strip("_") or "Seat"
    return Path(base) / slug / (pred_date or date.today().isoformat())


def append_history(res, seat_folder: Path, pred_date: str) -> Path:
    """One row per prediction date in <seat folder>/history.csv (a re-run on the same date replaces its row)."""
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
    """Seat-wide ACTUAL primary votes per election (all booths): year, formal_total, one column per modelled party."""
    rows = [{"year": int(y), "formal_total": int(g["n"].sum()), **{p: int(g[f"y_{p}"].sum()) for p in parties}}
            for y, g in prepared.groupby("year")]
    Path(folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(folder) / "actual_primaries.csv", index=False)


def save_run_meta(folder, cfg: ModelConfig, booths_path: str, polls_path: str) -> None:
    """Small provenance record so a downstream stage can see which run it is reading."""
    meta = {"seat_name": cfg.seat_name, "party": cfg.party, "parties": list(cfg.parties), "prediction_date": cfg.prediction_date,
            "election_date": cfg.election_date, "booths_file": str(booths_path), "polls_file": str(polls_path),
            "onp_mode": cfg.onp_mode, "saved_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "run_meta.json").write_text(json.dumps(meta, indent=2))


def save_backtest_draws(draws_by_year: dict, folder: Union[str, Path]) -> None:
    """One compressed file per backtested election: predicted booth x party primaries plus the actual booth votes
    (consumed downstream, e.g. by INANNA)."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    for y, d in draws_by_year.items():
        np.savez_compressed(out / f"backtest_draws_{int(y)}.npz", booths=np.array(d["booths"]), parties=np.array(d["parties"]),
                            booth_party_draws=d["booth_party_draws"], actual_party_votes=d["actual_party_votes"], actual_n=d["actual_n"])


# ----------------------------------------------------------------------------- trace
def save_trace(folder, trace: dict) -> None:
    """Arrays -> trace.npz; DataFrames -> trace_<key>.csv; scalars, strings and `meta` -> trace_meta.json."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    scalars: Dict[str, Any] = {}
    tables: List[str] = []
    for k, v in trace.items():
        if k == "meta":
            continue
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"trace_{k}.csv", index=False)
            tables.append(k)
        elif isinstance(v, (str, bool, int, float, np.generic)):
            scalars[k] = v.item() if isinstance(v, np.generic) else v
        else:
            arrays[k] = np.asarray(v)
    np.savez_compressed(out / "trace.npz", **arrays)
    (out / "trace_meta.json").write_text(
        json.dumps({"scalars": scalars, "tables": tables, "meta": trace.get("meta", {})}, indent=2, default=jsonable), encoding="utf-8")


def load_trace(folder) -> dict:
    """Inverse of save_trace."""
    out = Path(folder)
    meta = json.loads((out / "trace_meta.json").read_text(encoding="utf-8"))
    trace: Dict[str, Any] = {}
    with np.load(out / "trace.npz", allow_pickle=False) as z:
        for k in z.files:
            trace[k] = z[k]
    trace.update(meta.get("scalars", {}))
    for k in meta.get("tables", []):
        df = pd.read_csv(out / f"trace_{k}.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        trace[k] = df
    trace["meta"] = meta.get("meta", {})
    return trace
