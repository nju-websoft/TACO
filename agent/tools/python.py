from typing import Optional, List
from langchain_core.tools import tool
from pathlib import Path
import subprocess
import os
from .bash import _safe_path

ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT / ".tmp_exec"
DEFAULT_VENV_PATH = ROOT / ".venv"
DEFAULT_CWD = ROOT

def init_python_env():
    """Initialize the default python environment."""
    _ensure_venv(str(DEFAULT_VENV_PATH))

def _venv_python(venv: Path) -> Path:
    p = venv / "bin" / "python3"
    if p.exists():
        return p
    p2 = venv / "bin" / "python"
    return p2

def _ensure_venv(venv_path: str) -> str:
    v = _safe_path(venv_path)
    if not v.exists():
        try:
            subprocess.run(["python3", "-m", "venv", str(v)], check=True)
        except Exception as e:
            return f"venv_error: {e}"
    py = _venv_python(v)
    if not py.exists():
        return "venv_invalid"
    return str(py)

def _pip_install(py_path: str, cwd: Path, requirements: Optional[str], packages: Optional[List[str]], timeout: int) -> str:
    env = dict(os.environ)
    env["PAGER"] = "cat"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    logs = []
    
    # Ensure pip is installed
    try:
        subprocess.run([py_path, "-m", "ensurepip", "--default-pip"], check=True, capture_output=True)
    except Exception:
        pass
        
    try:
        if requirements:
            req_fp = _safe_path(requirements)
            if not req_fp.exists() or not req_fp.is_file():
                logs.append(f"requirements_missing: {requirements}")
            else:
                proc = subprocess.run([py_path, "-m", "pip", "install", "-r", str(req_fp)], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=max(1, int(timeout)))
                logs.append((proc.stdout or "") + (proc.stderr or ""))
        if packages:
            proc = subprocess.run([py_path, "-m", "pip", "install"] + list(packages), cwd=str(cwd), env=env, capture_output=True, text=True, timeout=max(1, int(timeout)))
            logs.append((proc.stdout or "") + (proc.stderr or ""))
    except subprocess.TimeoutExpired:
        logs.append("pip_timeout")
    except Exception as e:
        logs.append(f"pip_error: {e}")
    return "\n".join([s for s in logs if s])

@tool("python_pip_install", return_direct=False)
def python_pip_install(requirements: Optional[str] = None, packages: Optional[List[str]] = None, timeout: int = 120) -> str:
    """Install python dependencies into the default environment.
    
    Args:
        requirements: Path to a requirements.txt file.
        packages: List of package names to install.
        timeout: Timeout in seconds.
    """
    venv_path = str(DEFAULT_VENV_PATH)
    py_path = _ensure_venv(venv_path)
    if py_path.startswith("venv_") or py_path in {"venv_error", "venv_invalid"}:
        return py_path

    return _pip_install(py_path, DEFAULT_CWD, requirements, packages, timeout)

@tool("python_run", return_direct=False)
def python_run(path: str, script_args: List[str] = None, cwd: str = str(DEFAULT_CWD), timeout: int = 60, venv_path: Optional[str] = None, requirements: Optional[str] = None, packages: Optional[List[str]] = None) -> str:
    """Run a python script within project root. Relative to current dir."""
    base = _safe_path(cwd or ".")
    if not base.exists() or not base.is_dir():
        return f"Not a directory: {cwd}"
    fp = _safe_path(path)
    if not fp.exists() or not fp.is_file():
        return f"Not a file: {path}"
    
    target_venv = venv_path or str(DEFAULT_VENV_PATH)
    
    py_path = _ensure_venv(target_venv)
    if py_path.startswith("venv_") or py_path in {"venv_error", "venv_invalid"}:
        return py_path
        
    pip_logs = _pip_install(py_path, base, requirements, packages, timeout)

    cmd_args = []
    if script_args:
        cmd_args = [str(a) for a in script_args]
    
    py_cmd = [py_path, str(fp)] + cmd_args
    
    env = dict(os.environ)
    env["PAGER"] = "cat"
    try:
        proc = subprocess.run(py_cmd, cwd=str(base), env=env, capture_output=True, text=True, timeout=max(1, int(timeout)))
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as e:
        return f"exec_error: {e}"
    output = (proc.stdout or "") + (proc.stderr or "")
    if pip_logs:
        output = (pip_logs + "\n\n" + output).strip()
    return output or "empty"
