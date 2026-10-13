from __future__ import annotations

from pathlib import Path

from fedcti.config import storage_root


def data_dir(cfg: dict, exp: str, stage: str) -> Path:
    """stage is one of raw, clean, splits, clients, gt."""
    path = storage_root(cfg) / "data" / exp / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(cfg: dict, exp: str, method: str, model: str, alpha: float, seed: int) -> Path:
    model_tag = model.split("/")[-1]
    path = storage_root(cfg) / "runs" / exp / method / model_tag / f"alpha{alpha}" / f"seed{seed}"
    for sub in ("adapters", "logs", "preds"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def results_dir(cfg: dict) -> Path:
    path = storage_root(cfg) / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path
