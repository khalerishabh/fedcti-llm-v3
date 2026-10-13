from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from fedcti.data.attack import TACTIC_ORDER, AttackIndex
from fedcti.data.splits import (group_stratified_split, merge_overlapping_docs,
                                 refine_split, split_objective)
from fedcti.data.text import clean_text, merge_rows, near_duplicate_keys, normalize, word_count

NONE = "NONE"


def load_tram_multilabel(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [{"raw_index": i, "text": clean_text(r["sentence"]), "raw_labels": list(r["labels"]),
             "doc": clean_text(r.get("doc_title") or "UNKNOWN")} for i, r in enumerate(raw)]


def is_header_row(text: str) -> bool:
    low = text.lower()
    return low.startswith("title:") and " url: " in low


def uid_of(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:12]


def resolve_labels(rows: list[dict], attack: AttackIndex, collapse: bool) -> tuple[list[dict], dict]:
    reasons: Counter = Counter()
    mapping: dict[str, tuple[str | None, str]] = {}
    kept_rows, dropped_rows = [], 0
    for r in rows:
        labels = []
        for lab in r["raw_labels"]:
            if lab not in mapping:
                mapping[lab] = attack.resolve(lab, collapse=collapse)
            new, reason = mapping[lab]
            reasons[reason] += 1
            if new is not None:
                labels.append(new)
        labels = list(dict.fromkeys(labels))
        if r["raw_labels"] and not labels:
            dropped_rows += 1
            continue
        kept_rows.append(dict(r, labels=labels))
    log = {
        "label_occurrences_by_reason": dict(reasons),
        "rows_dropped_all_labels_unmappable": dropped_rows,
        "label_mapping": {k: {"to": v[0], "reason": v[1]} for k, v in sorted(mapping.items())
                          if v[1] != "ok"},
    }
    return kept_rows, log


def build_exp2_items(rows: list[dict], attack: AttackIndex, cfg: dict) -> tuple[list[dict], dict]:
    pre, spl = cfg["preprocess"], cfg["split"]
    log: dict = {"rows_raw": len(rows),
                 "raw_label_occurrences": sum(len(r["raw_labels"]) for r in rows)}

    keep = []
    dropped = Counter()
    for r in rows:
        if not r["raw_labels"]:
            if pre.get("drop_header_rows", True) and is_header_row(r["text"]):
                dropped["header_unlabelled"] += 1
                continue
            if word_count(r["text"]) < pre["min_words_unlabelled"]:
                dropped["short_unlabelled"] += 1
                continue
        if not r["text"]:
            dropped["empty"] += 1
            continue
        keep.append(r)
    log["rows_filtered"] = dict(dropped)

    keep, log["labels"] = resolve_labels(keep, attack, pre["collapse_subtechniques"])

    doc_group, log["doc_merges"] = merge_overlapping_docs(
        keep, pre["doc_merge_min_containment"], pre["doc_merge_min_words"])

    dd = pre["dedup"]
    items, log["dedup_exact"] = merge_rows(keep, [normalize(r["text"]) for r in keep],
                                           pre["label_vote_min_frac"])
    keys, pairs = near_duplicate_keys([r["text"] for r in items], dd["minhash_jaccard"],
                                      dd["minhash_num_perm"], dd["minhash_min_words"])
    items, log["dedup_near"] = merge_rows(items, keys, pre["label_vote_min_frac"])
    log["dedup_near"]["confirmed_pairs"] = pairs

    labelled = [r for r in items if r["labels"]]
    unlabelled = [r for r in items if not r["labels"]]
    frac = pre["none_max_fraction"]
    max_none = int(frac / (1 - frac) * len(labelled)) if frac < 1 else len(unlabelled)
    rng = random.Random(spl["seed"])
    if len(unlabelled) > max_none:
        unlabelled = rng.sample(sorted(unlabelled, key=lambda r: r["raw_index"]), max_none)
    log["none_downsampling"] = {"labelled": len(labelled), "none_available":
                                len(items) - len(labelled), "none_kept": len(unlabelled)}
    items = sorted(labelled + unlabelled, key=lambda r: r["raw_index"])

    for r in items:
        r["uid"] = uid_of(r["text"])
        r["group"] = doc_group[r["doc"]]
        r["dominant_tactic"] = attack.get(r["labels"][0]).primary_tactic if r["labels"] else NONE
        r.setdefault("docs", [r["doc"]])

    group_labels: dict[str, Counter] = defaultdict(Counter)
    group_sizes: Counter = Counter()
    for r in items:
        group_labels[r["group"]].update(r["labels"] or [NONE])
        group_sizes[r["group"]] += 1
    fractions = {"train": spl["train"], "val": spl["val"], "test": spl["test"]}
    split_of = group_stratified_split(group_labels, group_sizes, fractions, spl["seed"])
    initial = split_objective(split_of, group_labels, group_sizes, fractions)
    split_of = refine_split(split_of, group_labels, group_sizes, fractions, spl["seed"])
    log["split_objective"] = {"stratified": round(initial, 4), "refined": round(
        split_objective(split_of, group_labels, group_sizes, fractions), 4)}
    for r in items:
        r["split"] = split_of[r["group"]]

    fields = ["uid", "text", "labels", "dominant_tactic", "split", "group", "doc", "docs",
              "n_copies", "raw_index", "raw_labels"]
    items = [{k: r[k] for k in fields} for r in items]
    log["items_out"] = len(items)
    return items, log


def to_protocol_pk(items: list[dict]) -> list[dict]:
    """Krasovec-compatible: drop NONE, one row per (sentence, label)."""
    out = []
    for r in items:
        for j, lab in enumerate(r["labels"]):
            out.append({"kid": f"{r['uid']}-{j}", "uid": r["uid"], "text": r["text"],
                        "label": lab, "split": r["split"], "group": r["group"]})
    return out


def check_exp2_items(items: list[dict], attack: AttackIndex, cfg: dict) -> dict:
    """Ground-truth assertions; raises AssertionError on any hard failure."""
    pre, spl = cfg["preprocess"], cfg["split"]
    active = attack.active_ids(parents_only=pre["collapse_subtechniques"])
    bad = sorted({lab for r in items for lab in r["labels"] if lab not in active})
    assert not bad, f"labels not active in ATT&CK {cfg['attack']['version']}: {bad}"
    if pre["collapse_subtechniques"]:
        assert not any("." in lab for r in items for lab in r["labels"]), "sub-technique left"

    uids = [r["uid"] for r in items]
    assert len(uids) == len(set(uids)), "duplicate uid / normalized text"

    split_by_group: dict[str, set] = defaultdict(set)
    split_by_doc: dict[str, set] = defaultdict(set)
    for r in items:
        split_by_group[r["group"]].add(r["split"])
        for d in r["docs"]:
            split_by_doc[d].add(r["split"])
    leaked = [g for g, s in split_by_group.items() if len(s) > 1]
    assert not leaked, f"groups in more than one split: {leaked[:5]}"
    doc_spread = {d: sorted(s) for d, s in split_by_doc.items() if len(s) > 1}

    n = len(items)
    sizes = Counter(r["split"] for r in items)
    for s in ("train", "val", "test"):
        assert sizes[s] > 0, f"empty split {s}"
        assert abs(sizes[s] / n - spl[s]) <= spl.get("tolerance", 0.05), \
            f"split {s} fraction {sizes[s] / n:.3f} far from target {spl[s]}"

    n_none = sum(not r["labels"] for r in items)
    assert n_none / n <= pre["none_max_fraction"] + 1e-9, "too many NONE items"

    per_split = {s: Counter(lab for r in items if r["split"] == s for lab in r["labels"])
                 for s in ("train", "val", "test")}
    all_labels = set().union(*per_split.values())
    missing_train = sorted(all_labels - set(per_split["train"]))
    assert not missing_train, f"techniques absent from train: {missing_train}"

    return {
        "passed": True,
        "split_sizes": dict(sizes),
        "split_fractions": {s: round(sizes[s] / n, 4) for s in sizes},
        "techniques_missing_from_val": sorted(all_labels - set(per_split["val"])),
        "techniques_missing_from_test": sorted(all_labels - set(per_split["test"])),
        "merged_copies_whose_docs_span_splits": len(doc_spread),
    }


def frequency_buckets(train_counts: Counter) -> dict[str, str]:
    """Head / mid / tail by rank thirds of train frequency."""
    ranked = [lab for lab, _ in sorted(train_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    k = len(ranked)
    return {lab: ("head" if i < k / 3 else "mid" if i < 2 * k / 3 else "tail")
            for i, lab in enumerate(ranked)}


def exp2_report(items: list[dict], attack: AttackIndex) -> dict:
    splits = ("train", "val", "test")
    by_split = {s: [r for r in items if r["split"] == s] for s in splits}
    counts = {s: Counter(lab for r in rs for lab in r["labels"]) for s, rs in by_split.items()}
    buckets = frequency_buckets(counts["train"])
    techniques = []
    for lab in sorted(set().union(*counts.values()), key=lambda t: -counts["train"][t]):
        tech = attack.get(lab)
        techniques.append({"id": lab, "name": tech.name, "primary_tactic": tech.primary_tactic,
                           "bucket": buckets.get(lab, "tail"),
                           **{s: counts[s][lab] for s in splits}})
    summary = {}
    for s, rs in by_split.items():
        labelled = [r for r in rs if r["labels"]]
        summary[s] = {
            "items": len(rs), "labelled": len(labelled), "none": len(rs) - len(labelled),
            "multi_label": sum(len(r["labels"]) > 1 for r in rs),
            "label_occurrences": sum(len(r["labels"]) for r in rs),
            "groups": len({r["group"] for r in rs}),
            "distinct_techniques": len(counts[s]),
        }
    tactic_order = TACTIC_ORDER + [NONE]
    tactics = {s: dict(sorted(Counter(r["dominant_tactic"] for r in rs).items(),
                              key=lambda kv: tactic_order.index(kv[0])))
               for s, rs in by_split.items()}
    return {"summary": summary, "techniques": techniques, "dominant_tactic": tactics}


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
