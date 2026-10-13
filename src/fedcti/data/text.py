from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def normalize(text: str) -> str:
    return clean_text(text).lower()


def word_count(text: str) -> int:
    return len(clean_text(text).split())


class UnionFind:
    def __init__(self, items=()):
        self.parent = {x: x for x in items}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def vote_labels(label_lists: list[list[str]], min_frac: float) -> tuple[list[str], int]:
    """Keep labels present in at least min_frac of the copies; order by first appearance.

    Returns (labels, number of distinct labels dropped by the vote).
    """
    counts = Counter(lab for labels in label_lists for lab in set(labels))
    n = len(label_lists)
    order = list(dict.fromkeys(lab for labels in label_lists for lab in labels))
    kept = [lab for lab in order if counts[lab] / n >= min_frac]
    return kept, len(order) - len(kept)


def merge_rows(rows: list[dict], key_of: list, min_frac: float) -> tuple[list[dict], dict]:
    """Collapse rows sharing a key into one; labels by vote, doc by majority."""
    groups: dict = {}
    for row, key in zip(rows, key_of):
        groups.setdefault(key, []).append(row)
    merged, conflicts, labels_dropped = [], 0, 0
    for members in groups.values():
        if len(members) == 1:
            merged.append(dict(members[0], n_copies=members[0].get("n_copies", 1)))
            continue
        label_lists = [m["labels"] for m in members]
        if len({tuple(sorted(set(ls))) for ls in label_lists}) > 1:
            conflicts += 1
        labels, dropped = vote_labels(label_lists, min_frac)
        labels_dropped += dropped
        docs = Counter(m["doc"] for m in members)
        rep = max(members, key=lambda m: len(m["text"]))
        merged.append(dict(rep, labels=labels, doc=docs.most_common(1)[0][0],
                           docs=sorted(docs), n_copies=sum(m.get("n_copies", 1) for m in members)))
    stats = {"rows_in": len(rows), "rows_out": len(merged),
             "merged_groups": sum(len(g) > 1 for g in groups.values()),
             "label_conflict_groups": conflicts, "labels_dropped_by_vote": labels_dropped}
    return merged, stats


def shingles(text: str, k: int = 3) -> set[str]:
    words = normalize(text).split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def near_duplicate_keys(texts: list[str], threshold: float, num_perm: int, min_words: int,
                        seed: int = 1) -> tuple[list[int], int]:
    """Cluster id per text (MinHash LSH candidates, confirmed by exact shingle Jaccard).

    Texts shorter than min_words only cluster with themselves.
    Returns (keys, number of confirmed near-duplicate pairs).
    """
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sh = [shingles(t) for t in texts]
    uf = UnionFind(range(len(texts)))
    pairs = 0
    for i, s in enumerate(sh):
        if word_count(texts[i]) < min_words:
            continue
        mh = MinHash(num_perm=num_perm, seed=seed)
        for g in s:
            mh.update(g.encode("utf-8"))
        for j in lsh.query(mh):
            if len(s & sh[j]) / len(s | sh[j]) >= threshold:
                uf.union(i, j)
                pairs += 1
        lsh.insert(i, mh)
    return [uf.find(i) for i in range(len(texts))], pairs
