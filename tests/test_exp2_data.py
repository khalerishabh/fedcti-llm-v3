from collections import Counter

import pytest

from fedcti.data.attack import AttackIndex
from fedcti.data.splits import group_stratified_split, merge_overlapping_docs, refine_split
from fedcti.data.text import merge_rows, near_duplicate_keys, normalize, vote_labels


def _tech(stix_id, ext_id, tactics, sub=False, revoked=False, deprecated=False):
    return {
        "type": "attack-pattern", "id": stix_id, "name": ext_id, "description": "",
        "external_references": [{"source_name": "mitre-attack", "external_id": ext_id}],
        "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": t} for t in tactics],
        "x_mitre_is_subtechnique": sub, "revoked": revoked, "x_mitre_deprecated": deprecated,
    }


@pytest.fixture
def attack():
    return AttackIndex({"objects": [
        _tech("ap--1", "T1003", ["credential-access"]),
        _tech("ap--2", "T1003.001", ["credential-access"], sub=True),
        _tech("ap--3", "T9999", ["execution"], revoked=True),
        _tech("ap--4", "T1059", ["execution"]),
        _tech("ap--5", "T1111", ["defense-evasion"], deprecated=True),
        _tech("ap--6", "T1078", ["persistence", "initial-access", "defense-evasion"]),
        {"type": "relationship", "relationship_type": "revoked-by",
         "source_ref": "ap--3", "target_ref": "ap--4"},
    ]})


def test_resolve_paths(attack):
    assert attack.resolve("T1003.001") == ("T1003", "collapsed")
    assert attack.resolve("T1003.001", collapse=False) == ("T1003.001", "ok")
    assert attack.resolve("T9999") == ("T1059", "remapped")
    assert attack.resolve("T1111") == (None, "deprecated")
    assert attack.resolve("T0000") == (None, "unknown")


def test_primary_tactic_uses_kill_chain_order(attack):
    assert attack.get("T1078").primary_tactic == "initial-access"


def test_vote_labels_majority_and_tie():
    assert vote_labels([["A"], [], [], [], [], []], 0.5) == ([], 1)
    assert vote_labels([["A"], ["B", "A"]], 0.5) == (["A", "B"], 0)


def test_merge_rows_collapses_exact_duplicates():
    rows = [{"text": "Run Mimikatz now", "labels": ["T1003"], "doc": "d1", "raw_index": 0},
            {"text": "run  mimikatz now", "labels": ["T1003"], "doc": "d2", "raw_index": 1},
            {"text": "Other", "labels": [], "doc": "d1", "raw_index": 2}]
    merged, stats = merge_rows(rows, [normalize(r["text"]) for r in rows], 0.5)
    assert len(merged) == 2 and stats["merged_groups"] == 1
    assert next(m for m in merged if m["labels"])["docs"] == ["d1", "d2"]


def test_near_duplicates_cluster_only_close_texts():
    base = "the actor used powershell to download a second stage payload from a remote server"
    texts = [base, base + " today", "a completely different sentence about registry run keys"]
    keys, _ = near_duplicate_keys(texts, 0.8, 128, 6)
    assert keys[0] == keys[1] != keys[2]


def test_overlapping_docs_share_a_group():
    shared = [f"shared sentence number {i} with enough words" for i in range(5)]
    rows = ([{"text": s, "doc": "A"} for s in shared]
            + [{"text": s, "doc": "B"} for s in shared]
            + [{"text": "unique sentence in document C only here", "doc": "C"}])
    groups, merges = merge_overlapping_docs(rows, 0.2, 6)
    assert groups["A"] == groups["B"] != groups["C"] and len(merges) == 1


def test_group_split_assigns_every_group_and_refines():
    group_labels = {f"g{i}": Counter({f"L{i % 5}": 3, "NONE": 2}) for i in range(40)}
    sizes = {g: 5 for g in group_labels}
    fractions = {"train": 0.7, "val": 0.1, "test": 0.2}
    split = refine_split(group_stratified_split(group_labels, sizes, fractions, 0),
                         group_labels, sizes, fractions, 0)
    assert set(split) == set(group_labels)
    counts = Counter(split.values())
    assert counts["train"] == 28 and counts["val"] == 4 and counts["test"] == 8
