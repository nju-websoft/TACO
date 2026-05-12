"""Configuration helpers for limits and security.

- `load_config` reads config.yaml and merges env vars
- `get_limits` returns numeric caps for messages/history/tool retries
- `get_security` returns booleans for path access policy
"""

from typing import Any, Dict, Optional
import os
import yaml


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or os.path.join(os.getcwd(), "config.yaml")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def get_limits() -> Dict[str, int]:
    cfg = load_config()
    limits = (cfg.get("limits") or {}) if isinstance(cfg, dict) else {}
    return limits


def get_security() -> Dict[str, bool]:
    cfg = load_config()
    sec = (cfg.get("security") or {}) if isinstance(cfg, dict) else {}
    allow_outside = str(os.getenv("ALLOW_OUTSIDE_ROOT", sec.get("allow_outside_root", "false"))).lower() in {"1", "true", "yes"}
    return {"allow_outside_root": allow_outside}


limits = get_limits()
