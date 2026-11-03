"""io.py -- reading the blackbox input x and writing run files with clean subfolder organization."""
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


# ----------------------------------------------------------------------------- folder structuring
def output_folder(base: str, seat_name: str, pred_date: Optional[str] = None, sub: Optional[str] = None) -> Path:
    """<base>/<Seat_Name>/<YYYY-MM-DD>/[sub/]/"""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", seat_name).strip("_") or "Seat"
    p = Path(base) / slug / (pred_date or date.today().isoformat())
    if sub:
        p = p / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    """Seat-wide ACTUAL primary votes per election stored in tables/."""
    tables_dir = Path(folder) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"year": int(y), "formal_total": int(g["n"].sum()), **{p: int(g[f"y_{p}"].sum()) for p in parties}}
            for y, g in prepared.groupby("year")]
    pd.DataFrame(rows).to_csv(tables_dir / "actual_primaries.csv", index=False)


def save_run_meta(folder, cfg: ModelConfig, booths_path: str, polls_path: str) -> None:
    """Small provenance record stored in metadata/."""
    meta_dir = Path(folder) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {"seat_name": cfg.seat_name, "party": cfg.party, "parties": list(cfg.parties), "prediction_date": cfg.prediction_date,
            "election_date": cfg.election_date, "booths_file": str(booths_path), "polls_file": str(polls_path),
            "onp_mode": cfg.onp_mode, "saved_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    (meta_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def save_backtest_draws(draws_by_year: dict, folder: Union[str, Path]) -> None:
    """All backtest relevant items stored neatly inside backtests/."""
    bt_dir = Path(folder) / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)
    for y, d in draws_by_year.items():
        np.savez_compressed(bt_dir / f"backtest_draws_{int(y)}.npz", booths=np.array(d["booths"]), parties=np.array(d["parties"]),
                            booth_party_draws=d["booth_party_draws"], actual_party_votes=d["actual_party_votes"], actual_n=d["actual_n"])


# ----------------------------------------------------------------------------- trace & outputs
def save_trace(folder, trace: dict) -> None:
    """Arrays -> draws/trace.npz; DataFrames -> tables/trace_<key>.csv; scalars & meta -> metadata/trace_meta.json."""
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
    """Inverse of save_trace, reading from organized subfolders."""
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