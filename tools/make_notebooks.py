"""Generates the Colab notebooks from cell sources, so notebook code stays reviewable in git diffs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


BOOTSTRAP = code('''
GITHUB_USER, REPO_NAME, BRANCH = "khalerishabh", "fedcti-llm-v3", "main"
REPO_DIR = f"/content/{REPO_NAME}"

import os, subprocess, sys
from google.colab import drive, userdata

drive.mount("/content/drive")
token = userdata.get("GITHUB_TOKEN")
auth_url = f"https://{token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"


def git(*args):
    res = subprocess.run(["git", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout).replace(token, "***"))
    return res.stdout


if os.path.isdir(f"{REPO_DIR}/.git"):
    git("-C", REPO_DIR, "pull", "-q", "--ff-only", auth_url, BRANCH)
else:
    git("clone", "-q", "--branch", BRANCH, auth_url, REPO_DIR)
    git("-C", REPO_DIR, "remote", "set-url", "origin",
        f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git")
git("-C", REPO_DIR, "config", "user.name", "Rishabh Khale")
git("-C", REPO_DIR, "config", "user.email", "khalerishabhproff@gmail.com")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                f"{REPO_DIR}/requirements.txt", "-e", REPO_DIR], check=True)
if f"{REPO_DIR}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_DIR}/src")
print(git("-C", REPO_DIR, "log", "--oneline", "-1"))
''')

SETUP = [
    md('''
# 00 - Colab setup check

Every notebook in this repo starts with the same bootstrap cell (Drive mount, clone or pull,
install). This notebook runs only that cell plus environment checks, so use it once to confirm
the GitHub token, Drive and GPU are working.

Prerequisite: Colab secret `GITHUB_TOKEN` (key icon, left sidebar, notebook access on).
See `docs/colab_workflow.md`.
'''),
    BOOTSTRAP,
    code('''
import torch
from fedcti.config import load_config, storage_root
from fedcti.repro import save_metadata

cfg = load_config("exp2/data.yaml")
root = storage_root(cfg)
print("storage root:", root)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("bf16 supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print(save_metadata(root / "logs" / "setup_smoke.json", cfg))
'''),
]

EXP2_DATA = [
    md('''
# Exp 2 / 01 - TRAM data preparation (E2.0, E2.1)

A CPU runtime is enough for this notebook. Outputs go to `fedcti_v3/data/exp2/` on Drive, and
each stage is skipped on re-run once it has finished (delete its `_DONE.json` to redo it).

| Stage | What it does | Output |
| --- | --- | --- |
| E2.0 acquire | Downloads TRAM `multi_label.json` at a pinned commit and the ATT&CK Enterprise v15.1 STIX bundle; records sha256 | `raw/` |
| E2.1 build | Filters junk rows, maps labels to v15.1 parents, merges re-published reports, removes exact and near duplicates (label vote), downsamples NONE, splits 70/10/20 by document group, runs assertions | `clean/`, `splits/` |

Why splitting by document: sentences from one report share wording and context. If one report
had sentences in both train and test, test accuracy would partly measure memorisation.
'''),
    BOOTSTRAP,
    code('''
import json
import pandas as pd

from fedcti.config import load_config
from fedcti.paths import data_dir
from fedcti.stages import run_stage
from fedcti.data.acquire import download, github_raw_url
from fedcti.data.attack import AttackIndex
from fedcti.data import tram as T

cfg = load_config("exp2/data.yaml")
RAW, CLEAN, SPLITS = (data_dir(cfg, "exp2", s) for s in ("raw", "clean", "splits"))
print(RAW, CLEAN, SPLITS, sep="\\n")
'''),
    md('''
## E2.0 Acquire
'''),
    code('''
def acquire(out):
    tc = cfg["tram"]
    files = {
        "tram_multi_label": download(github_raw_url(tc["repo"], tc["commit"], tc["multi_label_file"]),
                                     out / "tram_multi_label.json"),
        "tram_license": download(github_raw_url(tc["repo"], tc["commit"], tc["license_file"]),
                                 out / "TRAM_LICENSE.txt"),
        "attack_bundle": download(cfg["attack"]["url"], out / "enterprise-attack-15.1.json"),
    }
    manifest = {"tram_commit": tc["commit"], "attack_version": cfg["attack"]["version"], "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {k: v["sha256"][:16] for k, v in files.items()}


run_stage("E2.0 acquire", RAW, acquire, cfg)
'''),
    md('''
## E2.1 Build items, split, assert
'''),
    code('''
def build(out):
    attack = AttackIndex.from_file(RAW / "enterprise-attack-15.1.json")
    rows = T.load_tram_multilabel(RAW / "tram_multi_label.json")
    items, log = T.build_exp2_items(rows, attack, cfg)
    check = T.check_exp2_items(items, attack, cfg)
    report = T.exp2_report(items, attack)

    T.write_jsonl(items, CLEAN / "items.jsonl")
    pk = T.to_protocol_pk(items)
    for split in ("train", "val", "test"):
        T.write_jsonl([r for r in items if r["split"] == split], SPLITS / f"pml_{split}.jsonl")
        T.write_jsonl([r for r in pk if r["split"] == split], SPLITS / f"pk_{split}.jsonl")
    for name, obj in (("build_log", log), ("checks", check), ("report", report)):
        (out / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))
    pd.DataFrame(report["techniques"]).to_csv(out / "techniques.csv", index=False)
    return {"items": len(items), **check["split_sizes"]}


run_stage("E2.1 build", SPLITS, build, cfg)
'''),
    md('''
## Data report
'''),
    code('''
log = json.loads((SPLITS / "build_log.json").read_text())
checks = json.loads((SPLITS / "checks.json").read_text())
report = json.loads((SPLITS / "report.json").read_text())

print("rows raw:", log["rows_raw"], "| filtered:", log["rows_filtered"])
print("labels:", log["labels"]["label_occurrences_by_reason"],
      "| rows dropped (unmappable):", log["labels"]["rows_dropped_all_labels_unmappable"])
print("exact dedup:", log["dedup_exact"])
print("near dedup:", log["dedup_near"])
print("NONE:", log["none_downsampling"])
print("split objective:", log["split_objective"])
print("checks:", {k: v for k, v in checks.items() if k != "split_sizes"})
display(pd.DataFrame(log["doc_merges"]))
display(pd.DataFrame(report["summary"]).T)
'''),
    code('''
tech = pd.DataFrame(report["techniques"])
display(tech.groupby("bucket")[["train", "val", "test"]].agg(["count", "sum"]))
display(pd.DataFrame(report["dominant_tactic"]).fillna(0).astype(int))
display(tech)
'''),
    md('''
## Manual spot check and sign-off

Read the 10 random items below against their labels. If they look right, set `REVIEWER` and
`APPROVED = True` in the next cell and run it; the sign-off is stored next to the splits.
'''),
    code('''
import random

attack = AttackIndex.from_file(RAW / "enterprise-attack-15.1.json")
items = T.read_jsonl(CLEAN / "items.jsonl")
sample = random.Random(7).sample(items, 10)
for r in sample:
    names = [f"{l} {attack.get(l).name}" for l in r["labels"]] or ["NONE"]
    print(f"[{r['split']}] {r['uid']} | {'; '.join(names)}\\n    {r['text'][:300]}\\n")
'''),
    code('''
REVIEWER = ""
APPROVED = False

if APPROVED and REVIEWER:
    from datetime import datetime, timezone
    signoff = {"reviewer": REVIEWER, "approved_utc": datetime.now(timezone.utc).isoformat(),
               "sample_uids": [r["uid"] for r in sample]}
    (SPLITS / "review_signoff.json").write_text(json.dumps(signoff, indent=2))
    print("sign-off saved")
else:
    print("not signed off yet")
'''),
]


def write(cells, rel):
    nb = {"cells": cells,
          "metadata": {"colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:02d}"
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="ascii")
    print("wrote", rel)


if __name__ == "__main__":
    write(SETUP, "notebooks/00_colab_setup.ipynb")
    write(EXP2_DATA, "notebooks/exp2/01_data_prep.ipynb")
    sys.exit(0)
