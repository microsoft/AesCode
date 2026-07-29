"""Trampoline: re-export scripts/_bootstrap.py so `import _bootstrap` works from here."""
import importlib.util, sys
from pathlib import Path

_real = Path(__file__).resolve().parents[2] / "scripts" / "_bootstrap.py"
_spec = importlib.util.spec_from_file_location("_bootstrap", _real)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_bootstrap"] = _mod
_spec.loader.exec_module(_mod)

