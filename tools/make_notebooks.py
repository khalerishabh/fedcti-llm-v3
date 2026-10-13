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
token = userdata.get("GITHUB_TOKEN_FEDCTI")
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
git("-C", REPO_DIR, "config", "user.email", "khalerishabh998@gmail.com")

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

Prerequisite: Colab secret `GITHUB_TOKEN_FEDCTI` (key icon, left sidebar, notebook access on).
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


EXP2_SHORTLIST = [
    md("""
# Exp 2 / 02 - Candidate shortlist (E2.2)

Runtime: a T4 GPU is plenty (the dense encoder is 110M parameters); CPU also works but the
encoding step takes several minutes. Requires `01_data_prep` to have finished.

**What this stage does.** The classifier will not choose among all 202 ATT&CK techniques; it
chooses among a short list of candidates. This notebook builds that list for every sentence:

| Stage | What it does |
| --- | --- |
| E2.2a technique docs | One text per active parent technique in ATT&CK v15.1: name, tactics, first 150 words of the description, up to two procedure examples. Procedure examples that overlap any TRAM sentence are dropped, because ATT&CK often quotes the same reports TRAM annotated. |
| E2.2b retrieve | Scores every sentence against all 202 docs with BM25 (word overlap) and bge-base (meaning), then fuses the two rankings with reciprocal rank fusion. |
| E2.2c choose K | Smallest K whose validation coverage (share of gold labels inside the top-K) reaches 90%. |

The retriever is never trained, so the federated contribution stays in the classifier.

**Test split.** Shortlists are computed for test too (later stages need them), but test
coverage is not printed here. Test numbers are opened once, after every choice is frozen.
"""),
    BOOTSTRAP,
    code("""
import json
import numpy as np
import pandas as pd

from fedcti.config import load_config
from fedcti.paths import data_dir, results_dir
from fedcti.stages import run_stage
from fedcti.data.attack import AttackIndex
from fedcti.data import tram as T
from fedcti.retrieval import retrieve as R
from fedcti.retrieval.technique_docs import LeakFilter, build_technique_docs

cfg = load_config("exp2/data.yaml", "exp2/retrieval.yaml")
rc = cfg["retrieval"]
RAW, CLEAN, SPLITS = (data_dir(cfg, "exp2", s) for s in ("raw", "clean", "splits"))
RET = data_dir(cfg, "exp2", "retrieval")
FIG = results_dir(cfg) / "figures"
FIG.mkdir(parents=True, exist_ok=True)

assert (SPLITS / "_DONE.json").exists(), "run 01_data_prep first"
if not (SPLITS / "review_signoff.json").exists():
    print("WARNING: E2.1 spot check not signed off")
"""),
    md("""
## E2.2a Technique documents
"""),
    code("""
DOCS_DIR = RET / "docs"


def make_docs(out):
    attack = AttackIndex.from_file(RAW / "enterprise-attack-15.1.json")
    raw_sentences = [r["text"] for r in T.load_tram_multilabel(RAW / "tram_multi_label.json")]
    leak = LeakFilter(raw_sentences, rc["procedure_leak_min_containment"])
    docs, stats = build_technique_docs(attack, rc, leak)
    T.write_jsonl(docs, out / "technique_docs.jsonl")
    return stats


print(run_stage("E2.2a technique docs", DOCS_DIR, make_docs, cfg))
docs = T.read_jsonl(DOCS_DIR / "technique_docs.jsonl")
DOC_IDS = [d["id"] for d in docs]
print(len(docs), "techniques")
print(next(d for d in docs if d["id"] == "T1055")["text"][:900])
"""),
    md("""
## E2.2b Retrieve (BM25 + dense + fusion)
"""),
    code("""
SCORES_DIR = RET / "scores"
items = T.read_jsonl(CLEAN / "items.jsonl")


def run_retrieval(out):
    import torch

    texts = [r["text"] for r in items]
    bm25 = R.BM25Retriever([d["text"] for d in docs], rc["bm25_k1"], rc["bm25_b"]).scores(texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense_model = R.DenseRetriever(rc["dense_model"], rc["dense_revision"], rc["query_instruction"],
                                   rc["batch_size"], device)
    dense = dense_model.scores(texts, dense_model.encode([d["text"] for d in docs]))
    hybrid = R.rrf([bm25, dense], rc["rrf_k"])
    np.savez_compressed(out / "scores.npz", bm25=bm25, dense=dense, hybrid=hybrid,
                        uids=np.array([r["uid"] for r in items]), doc_ids=np.array(DOC_IDS))

    for split in ("train", "val", "test"):
        idx = [i for i, r in enumerate(items) if r["split"] == split]
        lists = {name: R.top_ids(m[idx], DOC_IDS, rc["keep_top"])
                 for name, m in (("hybrid", hybrid), ("bm25", bm25), ("dense", dense))}
        T.write_jsonl([{"uid": items[i]["uid"], **{n: lists[n][j] for n in lists}}
                       for j, i in enumerate(idx)], out / f"shortlist_{split}.jsonl")
    try:
        from huggingface_hub import model_info
        revision = model_info(rc["dense_model"]).sha
    except Exception as exc:
        revision = f"unresolved: {exc}"
    return {"device": device, "dense_revision": revision, "items": len(items)}


print(run_stage("E2.2b retrieve", SCORES_DIR, run_retrieval, cfg))
"""),
    md("""
## E2.2c Coverage@K on validation and choice of K

Two label spaces are reported:

- **all**: all 202 techniques. This is the main setting and matches Krasovec et al.
- **tram**: only the techniques that occur in the training split (TRAM annotators used about 50).
  It shows how much easier the task becomes when the label space is known in advance.
"""),
    code("""
SELECT_DIR = RET / "selection"
KS = list(range(1, rc["k_max"] + 1))


def select_k(out):
    z = np.load(SCORES_DIR / "scores.npz")
    split_of = [r["split"] for r in items]
    gold = [r["labels"] for r in items]
    tram_space = {lab for r in items if r["split"] == "train" for lab in r["labels"]}
    rows = []
    for split in ("train", "val"):
        idx = [i for i, s in enumerate(split_of) if s == split]
        for retriever in ("hybrid", "bm25", "dense"):
            for space, allowed in (("all", None), ("tram", tram_space)):
                ranked = R.top_ids(z[retriever][idx], DOC_IDS, rc["k_max"], allowed)
                cov = R.coverage_at_k(ranked, [gold[i] for i in idx], KS)
                for k in KS:
                    rows.append({"split": split, "retriever": retriever, "space": space, "k": k,
                                 "label_cov": cov["label"][k], "item_all_cov": cov["item_all"][k]})
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "coverage_curves.csv", index=False)

    main = curves.query("split == 'val' and retriever == 'hybrid' and space == 'all'")
    curve = dict(zip(main["k"], main["label_cov"]))
    k_star = R.smallest_k(curve, rc["coverage_target"])
    selected = {
        "k": k_star, "rule": f"smallest K with val label coverage >= {rc['coverage_target']}",
        "val_label_cov_at_k": curve.get(k_star), "val_label_cov_at_6": curve[6],
        "fallback_needed": k_star is None or k_star > 20,
    }
    (out / "selected_k.json").write_text(json.dumps(selected, indent=2))
    return selected


print(run_stage("E2.2c select K", SELECT_DIR, select_k, cfg))
curves = pd.read_csv(SELECT_DIR / "coverage_curves.csv")
selected = json.loads((SELECT_DIR / "selected_k.json").read_text())
K_STAR = selected["k"]
show_k = sorted({1, 3, 5, 6, 10, 15, 20, 30, K_STAR or 50})
table = (curves.query("split == 'val' and k in @show_k")
         .pivot_table(index=["space", "retriever"], columns="k", values="label_cov")
         .round(3))
display(table)
"""),
    code("""
import matplotlib.pyplot as plt

SERIES = {"hybrid": "#2a78d6", "bm25": "#eb6834", "dense": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
val = curves.query("split == 'val' and space == 'all'")
for name, color in SERIES.items():
    c = val.query("retriever == @name")
    ax.plot(c["k"], c["label_cov"] * 100, color=color, lw=2, label=name)
    ax.annotate(name, (c["k"].iloc[-1], c["label_cov"].iloc[-1] * 100), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK)
ax.axhline(rc["coverage_target"] * 100, color=MUTED, lw=1, ls="--")
ax.text(rc["k_max"], rc["coverage_target"] * 100 + 0.8, "90% target", ha="right", fontsize=8,
        color=MUTED)
kref = rc["krasovec_reference"]
ax.plot([kref["k"]], [kref["coverage"] * 100], marker="D", ms=8, color=INK, ls="none",
        label="Krasovec et al. 2025 (cross-paper)")
if K_STAR:
    ax.axvline(K_STAR, color=GRID, lw=1)
    ax.text(K_STAR + 0.5, 12, f"K* = {K_STAR}", fontsize=8, color=MUTED)
ax.set_xlabel("K (candidates in shortlist)", color=INK)
ax.set_ylabel("Gold labels covered (%)", color=INK)
ax.set_title("Shortlist coverage on validation, 202 ATT&CK techniques", fontsize=10, color=INK)
ax.set_ylim(0, 100)
ax.set_xlim(1, rc["k_max"] + 5)
ax.grid(color=GRID, lw=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "exp2_shortlist_coverage_val.png")
plt.show()
"""),
    md("""
## Where the retriever misses (validation)

Validation items whose gold technique is outside the top-K*. Read a few: misses usually come
from vague sentences or from labels that need context from neighbouring sentences.
"""),
    code("""
val_items = [r for r in items if r["split"] == "val" and r["labels"]]
short = {r["uid"]: r for r in T.read_jsonl(SCORES_DIR / "shortlist_val.jsonl")}
names = {d["id"]: d["name"] for d in docs}
k = K_STAR or rc["k_max"]
misses = [r for r in val_items if not set(r["labels"]) <= set(short[r["uid"]]["hybrid"][:k])]
print(f"{len(misses)} of {len(val_items)} labelled val items have a gold label outside top-{k}")
for r in misses[:12]:
    top = short[r["uid"]]["hybrid"]
    rank = {g: (top.index(g) + 1 if g in top else ">50") for g in r["labels"]}
    print(f"gold {[f'{g} {names[g]} (rank {rank[g]})' for g in r['labels']]}")
    print(f"    top5 {[names[t] for t in top[:5]]}")
    print(f"    {r['text'][:220]}\\n")
"""),
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
    write(EXP2_SHORTLIST, "notebooks/exp2/02_shortlist.ipynb")
    sys.exit(0)
