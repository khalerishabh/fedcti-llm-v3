from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_\-]*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were which with will can may also into their they these those".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, doc_texts: list[str], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(t) for t in doc_texts], k1=k1, b=b)

    def scores(self, queries: list[str]) -> np.ndarray:
        return np.vstack([self.bm25.get_scores(tokenize(q)) for q in queries]).astype(np.float32)


class DenseRetriever:
    def __init__(self, model_name: str, revision: str | None, query_instruction: str,
                 batch_size: int, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        self.instruction = query_instruction
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)

    def scores(self, queries: list[str], doc_emb: np.ndarray) -> np.ndarray:
        q = self.encode([self.instruction + x for x in queries])
        return (q @ doc_emb.T).astype(np.float32)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """0-based rank of every doc per query (ties broken by doc order)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    return ranks


def rrf(score_mats: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion: sum over retrievers of 1 / (k + rank), rank starting at 1."""
    return sum(1.0 / (k + 1 + ranks_from_scores(s)) for s in score_mats).astype(np.float32)


def top_ids(scores: np.ndarray, doc_ids: list[str], keep: int,
            allowed: set[str] | None = None) -> list[list[str]]:
    s = scores.copy()
    if allowed is not None:
        s[:, [i for i, d in enumerate(doc_ids) if d not in allowed]] = -np.inf
    order = np.argsort(-s, axis=1, kind="stable")[:, :keep]
    return [[doc_ids[j] for j in row if np.isfinite(s[r, j])] for r, row in enumerate(order)]


def coverage_at_k(ranked: list[list[str]], gold: list[list[str]], ks: list[int]) -> dict:
    """label: share of gold labels within top-K (Krasovec's coverage, one row per label);
    item_all: share of items whose gold labels are all within top-K."""
    pairs = [(r, g) for r, g in zip(ranked, gold) if g]
    n_labels = sum(len(g) for _, g in pairs)
    out = {"label": {}, "item_all": {}, "n_items": len(pairs), "n_labels": n_labels}
    for k in ks:
        hit_labels = sum(len(set(g) & set(r[:k])) for r, g in pairs)
        hit_items = sum(set(g) <= set(r[:k]) for r, g in pairs)
        out["label"][k] = hit_labels / n_labels if n_labels else float("nan")
        out["item_all"][k] = hit_items / len(pairs) if pairs else float("nan")
    return out


def smallest_k(curve: dict[int, float], target: float) -> int | None:
    for k in sorted(curve):
        if curve[k] >= target:
            return k
    return None
