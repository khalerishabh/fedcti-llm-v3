from __future__ import annotations

import re
from collections import Counter, defaultdict

from fedcti.data.attack import AttackIndex
from fedcti.data.text import shingles

_CITATION = re.compile(r"\(Citation:[^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_TAGS = re.compile(r"</?code>")
_WS = re.compile(r"\s+")


def clean_attack_text(text: str) -> str:
    text = _CITATION.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _TAGS.sub("", text)
    return _WS.sub(" ", text).strip()


def first_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def tactic_label(tactic: str) -> str:
    return tactic.replace("-", " ").title().replace("And", "and")


class LeakFilter:
    """Flags procedure examples that overlap a TRAM sentence (ATT&CK procedures are often
    quoted from the same reports TRAM annotated)."""

    def __init__(self, sentences: list[str], min_containment: float):
        self.min_containment = min_containment
        self.index: dict[str, set[int]] = defaultdict(set)
        for i, s in enumerate(sentences):
            for g in shingles(s):
                self.index[g].add(i)

    def leaks(self, text: str) -> bool:
        sh = shingles(text)
        if not sh:
            return False
        hits: Counter = Counter()
        for g in sh:
            hits.update(self.index.get(g, ()))
        return bool(hits) and max(hits.values()) / len(sh) >= self.min_containment


def build_technique_docs(attack: AttackIndex, rc: dict, leak_filter: LeakFilter | None
                         ) -> tuple[list[dict], dict]:
    docs, stats = [], Counter()
    for tech_id in sorted(attack.active_ids(parents_only=True)):
        tech = attack.get(tech_id)
        desc = clean_attack_text(tech.description)
        procs = []
        for _, raw in attack.procedures(tech_id):
            text = clean_attack_text(raw)
            n = len(text.split())
            if not rc["procedure_min_words"] <= n <= rc["procedure_max_words"]:
                continue
            stats["procedures_considered"] += 1
            if leak_filter is not None and leak_filter.leaks(text):
                stats["procedures_dropped_leak"] += 1
                continue
            procs.append(text)
            if len(procs) == rc["procedures_per_technique"]:
                break
        stats["techniques_without_procedures"] += not procs
        tactics = [tactic_label(t) for t in tech.tactics]
        retrieval_text = " ".join(
            [f"{tech.name}.", f"Tactics: {', '.join(tactics)}.", first_words(desc, rc["doc_desc_words"])]
            + ([f"Examples: {' '.join(procs)}"] if procs else []))
        docs.append({"id": tech_id, "name": tech.name, "tactics": tech.tactics,
                     "prompt_desc": first_words(desc, rc["prompt_desc_words"]),
                     "procedures": procs, "text": retrieval_text})
    stats["techniques"] = len(docs)
    return docs, dict(stats)
