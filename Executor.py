"""Executor.py -- run ENKI for one seat from config.jsonc.
Reads config.jsonc (with // comments) and passes the clean dict to ENKI(x). Minimal hardcoding."""
import json
from pathlib import Path

from enki import ENKI
from enki.enki import _strip_comments
from enki.helpers import write_all


def main(config_path: str = "config.jsonc") -> object:
    raw_text = Path(config_path).read_text(encoding="utf-8")
    x = json.loads(_strip_comments(raw_text))
    output_root = x.get("output_root")
    y = ENKI(x)
    if output_root:
        write_all(y, output_root)
    return y


if __name__ == "__main__":
    main()
