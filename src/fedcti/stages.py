from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from fedcti.repro import run_metadata

MARKER = "_DONE.json"


def is_done(stage_dir: Path) -> bool:
    return (stage_dir / MARKER).exists()


def run_stage(name: str, stage_dir: Path, fn: Callable[[Path], dict | None], cfg: dict,
              force: bool = False) -> dict:
    """Run fn(stage_dir) once; a marker file makes Colab restarts skip finished stages.

    The marker is written only after fn returns, so a crash mid-stage is re-run next time.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    marker = stage_dir / MARKER
    if marker.exists() and not force:
        print(f"[skip] {name}: already done ({stage_dir})")
        return json.loads(marker.read_text(encoding="utf-8")).get("summary", {})
    print(f"[run ] {name}")
    summary = fn(stage_dir) or {}
    record = {"stage": name, "summary": summary, **run_metadata(cfg)}
    marker.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"[done] {name}")
    return summary
