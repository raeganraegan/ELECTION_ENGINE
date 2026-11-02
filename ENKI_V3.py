"""ENKI_V3.py -- kept so existing `from ENKI_V3 import ENKI` lines (Executor.py, INANNA) keep working.
The model now lives in the `enki/` package; import from there in new code."""
from enki import ENKI, ModelConfig, PollInput, RunOptions, cli, main

if __name__ == "__main__":
    import sys
    cli(sys.argv[1:])
