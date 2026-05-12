from typing import List, Optional
from langchain_core.tools import tool
import os
from pathlib import Path
import re
from agent.config import get_security
import subprocess
from agent.utils import BASE_DIR, get_root_dir

ROOT = Path(get_root_dir()).resolve()
CURRENT_DIR = ROOT


def _safe_path(p: str) -> Path:
    base = CURRENT_DIR
    cand = Path(p)
    cand = (base / cand).resolve() if not cand.is_absolute() else cand.resolve()
    if not cand.exists() and not Path(p).is_absolute():
        try:
            alt = (Path(BASE_DIR) / Path(p)).resolve()
            if alt.exists():
                cand = alt
        except Exception:
            pass
    sec = get_security()
    allow_outside = bool(sec.get("allow_outside_root", False))
    if not allow_outside:
        ok_root = os.path.commonpath([str(ROOT), str(cand)]) == str(ROOT)
        ok_go = False
        try:
            ok_go = os.path.commonpath([str(BASE_DIR), str(cand)]) == str(BASE_DIR)
        except Exception:
            ok_go = False
        if not (ok_root or ok_go):
            raise ValueError("Path outside project roots")
    return cand


@tool("bash_set_root", return_direct=False)
def bash_set_root(path: str) -> str:
    """Set the base ROOT directory for all operations; resets CURRENT_DIR to ROOT."""
    global ROOT, CURRENT_DIR
    try:
        target = Path(path).resolve()
    except Exception:
        return f"invalid_path: {path}"
    if not target.exists() or not target.is_dir():
        return f"Not a directory: {path}"
    ROOT = target
    CURRENT_DIR = target
    return str(ROOT)


@tool("bash_exec", return_direct=False)
def bash_exec(command: str, cwd: str = ".", timeout: int = 30, env: Optional[List[str]] = None) -> str:
    """Execute a shell command within project root. Only allow one command at a time."""
    global CURRENT_DIR
    base = _safe_path(cwd or ".")
    if not base.exists() or not base.is_dir():
        return f"Not a directory: {cwd}"
    deny = re.compile(r"(?:\brm\s+-rf\b|\bsudo\b|\bshutdown\b|\breboot\b|\bmkfs\b|\bdd\b|\bkill\s+-9\b)")
    if deny.search(command or ""):
        return "command_rejected"
    env_map = dict(os.environ)
    env_map["PAGER"] = "cat"
    if env:
        for kv in env:
            if isinstance(kv, str) and "=" in kv:
                k, v = kv.split("=", 1)
                env_map[k] = v
    try:
        proc = subprocess.run(["/bin/bash", "-lc", command], cwd=str(base), env=env_map, capture_output=True, text=True, timeout=max(1, int(timeout)))
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as e:
        return f"exec_error: {e}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return output or "empty"
