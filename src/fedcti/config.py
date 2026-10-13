from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted_key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def load_config(*files: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    """Merge configs/base.yaml with the given files (later wins), then dotted overrides."""
    paths = [CONFIG_DIR / "base.yaml", *(Path(f) if Path(f).is_absolute() else CONFIG_DIR / f
                                         for f in files)]
    cfg: dict = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    return cfg


def on_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir()


def storage_root(cfg: dict) -> Path:
    """Drive on Colab, a git-ignored local folder elsewhere; FEDCTI_ROOT overrides both."""
    if os.environ.get("FEDCTI_ROOT"):
        root = Path(os.environ["FEDCTI_ROOT"])
    elif on_colab():
        root = Path(cfg["paths"]["drive_root"])
    else:
        root = REPO_ROOT / cfg["paths"]["local_root"]
    root.mkdir(parents=True, exist_ok=True)
    return root
