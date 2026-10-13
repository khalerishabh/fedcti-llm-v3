from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import combinations

from fedcti.data.text import UnionFind, normalize, word_count


def merge_overlapping_docs(rows: list[dict], min_containment: float,
                           min_words: int) -> tuple[dict[str, str], list[dict]]:
    """Put documents that share many sentences (re-published reports) into one split group.

    Containment = shared sentences / sentences of the smaller document, counting only
    sentences with at least min_words words so boilerplate lines do not merge unrelated docs.
    """
    sents: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if word_count(r["text"]) >= min_words:
            sents[r["doc"]].add(normalize(r["text"]))
    owners: dict[str, set[str]] = defaultdict(set)
    for doc, ss in sents.items():
        for s in ss:
            owners[s].add(doc)
    shared: Counter = Counter()
    for docs in owners.values():
        for a, b in combinations(sorted(docs), 2):
            shared[(a, b)] += 1
    uf = UnionFind({r["doc"] for r in rows})
    merges = []
    for (a, b), n in shared.items():
        containment = n / min(len(sents[a]), len(sents[b]))
        if containment >= min_containment:
            uf.union(a, b)
            merges.append({"doc_a": a, "doc_b": b, "shared": n,
                           "containment": round(containment, 3)})
    roots = {doc: uf.find(doc) for doc in uf.parent}
    ids = {root: f"g{i:03d}" for i, root in enumerate(sorted(set(roots.values())))}
    return {doc: ids[root] for doc, root in roots.items()}, merges


def group_stratified_split(group_labels: dict[str, Counter], group_sizes: dict[str, int],
                           fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Iterative stratification (Sechidis et al. 2011) applied to whole groups.

    Rarest label first: each group holding it goes to the split that still wants the most
    of that label, ties broken by remaining size, then randomly.
    """
    rng = random.Random(seed)
    splits = list(fractions)
    total_labels: Counter = Counter()
    for c in group_labels.values():
        total_labels.update(c)
    want = {s: {lab: fractions[s] * n for lab, n in total_labels.items()} for s in splits}
    want_size = {s: fractions[s] * sum(group_sizes.values()) for s in splits}
    remaining = set(group_labels)
    assignment: dict[str, str] = {}

    def assign(group: str, split: str) -> None:
        assignment[group] = split
        remaining.discard(group)
        for lab, n in group_labels[group].items():
            want[split][lab] -= n
        want_size[split] -= group_sizes[group]

    while remaining:
        left: Counter = Counter()
        for g in remaining:
            left.update(group_labels[g])
        left = Counter({k: v for k, v in left.items() if v > 0})
        if not left:
            break
        rarest = min(left.values())
        label = rng.choice(sorted(k for k, v in left.items() if v == rarest))
        holders = sorted(g for g in remaining if group_labels[g].get(label, 0) > 0)
        rng.shuffle(holders)
        for g in holders:
            best = max(splits, key=lambda s: (want[s][label], want_size[s], rng.random()))
            assign(g, best)
    for g in sorted(remaining):
        assign(g, max(splits, key=lambda s: (want_size[s], rng.random())))
    return assignment


def split_objective(assignment: dict[str, str], group_labels: dict[str, Counter],
                    group_sizes: dict[str, int], fractions: dict[str, float]) -> float:
    """Size deviation (weighted x2) + mean per-label proportion deviation + missing-label
    penalties (1.0 per label absent from the first split, 0.02 per label absent elsewhere)."""
    splits = list(fractions)
    size = Counter()
    lab = defaultdict(Counter)
    for g, s in assignment.items():
        size[s] += group_sizes[g]
        for label, n in group_labels[g].items():
            lab[label][s] += n
    total = sum(size.values())
    size_dev = sum(abs(size[s] / total - fractions[s]) for s in splits)
    label_dev, missing = 0.0, 0.0
    for counts in lab.values():
        n = sum(counts.values())
        label_dev += sum(abs(counts[s] / n - fractions[s]) for s in splits)
        missing += sum((1.0 if i == 0 else 0.02) for i, s in enumerate(splits) if counts[s] == 0)
    return 2 * size_dev + label_dev / max(len(lab), 1) + missing


def refine_split(assignment: dict[str, str], group_labels: dict[str, Counter],
                 group_sizes: dict[str, int], fractions: dict[str, float], seed: int,
                 max_passes: int = 30) -> dict[str, str]:
    """Greedy local search: move one group at a time while the objective improves."""
    rng = random.Random(seed)
    best = dict(assignment)
    best_score = split_objective(best, group_labels, group_sizes, fractions)
    for _ in range(max_passes):
        improved = False
        groups = sorted(best)
        rng.shuffle(groups)
        for g in groups:
            current = best[g]
            for s in fractions:
                if s == current:
                    continue
                best[g] = s
                score = split_objective(best, group_labels, group_sizes, fractions)
                if score < best_score - 1e-12:
                    best_score, current, improved = score, s, True
                else:
                    best[g] = current
        if not improved:
            break
    return best
