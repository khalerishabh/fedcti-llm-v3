import numpy as np

from fedcti.retrieval.retrieve import (BM25Retriever, coverage_at_k, ranks_from_scores, rrf,
                                       smallest_k, top_ids)
from fedcti.retrieval.technique_docs import LeakFilter, clean_attack_text


def test_clean_attack_text_strips_markup():
    raw = ("[Shamoon](https://attack.mitre.org/software/S0140) runs <code>cmd.exe</code>."
           "(Citation: Palo Alto 2016)")
    assert clean_attack_text(raw) == "Shamoon runs cmd.exe."


def test_rrf_prefers_docs_ranked_high_by_both():
    a = np.array([[3.0, 2.0, 1.0]])
    b = np.array([[1.0, 3.0, 2.0]])
    fused = rrf([a, b], k=60)
    assert int(np.argmax(fused)) == 1
    assert ranks_from_scores(a).tolist() == [[0, 1, 2]]


def test_top_ids_respects_allowed_set():
    s = np.array([[0.9, 0.8, 0.1]])
    assert top_ids(s, ["T1", "T2", "T3"], 2) == [["T1", "T2"]]
    assert top_ids(s, ["T1", "T2", "T3"], 2, allowed={"T2", "T3"}) == [["T2", "T3"]]


def test_coverage_label_and_item_level():
    ranked = [["A", "B", "C"], ["C", "A", "B"]]
    gold = [["A", "C"], ["B"], ]
    cov = coverage_at_k(ranked, gold, [1, 3])
    assert cov["label"][1] == 1 / 3 and cov["label"][3] == 1.0
    assert cov["item_all"][1] == 0.0 and cov["item_all"][3] == 1.0
    assert smallest_k(cov["label"], 0.9) == 3


def test_bm25_ranks_matching_doc_first():
    docs = ["credential dumping lsass mimikatz", "scheduled task persistence", "dns tunneling"]
    s = BM25Retriever(docs).scores(["the actor ran mimikatz against lsass"])
    assert int(np.argmax(s[0])) == 0


def test_leak_filter_flags_quoted_procedure():
    lf = LeakFilter(["the malware uses powershell to download a second stage payload"], 0.5)
    assert lf.leaks("the malware uses powershell to download a second stage payload from c2")
    assert not lf.leaks("APT29 used scheduled tasks to maintain persistence on hosts")


def test_technique_docs_use_subtechnique_procedures_and_skip_leaks():
    from fedcti.data.attack import AttackIndex
    from fedcti.retrieval.technique_docs import build_technique_docs

    def tech(sid, eid, sub=False):
        return {"type": "attack-pattern", "id": sid, "name": f"Name {eid}",
                "description": "Adversaries may do things. (Citation: X)",
                "external_references": [{"source_name": "mitre-attack", "external_id": eid}],
                "kill_chain_phases": [{"kill_chain_name": "mitre-attack",
                                       "phase_name": "credential-access"}],
                "x_mitre_is_subtechnique": sub}

    def uses(rid, target, text):
        return {"type": "relationship", "relationship_type": "uses", "id": rid,
                "source_ref": "malware--1", "target_ref": target, "description": text}

    attack = AttackIndex({"objects": [
        tech("attack-pattern--1", "T1003"), tech("attack-pattern--2", "T1003.001", sub=True),
        uses("relationship--a", "attack-pattern--2", "Mimikatz reads credentials from LSASS process memory on hosts"),
        uses("relationship--b", "attack-pattern--1", "the malware dumps the sam database to a temporary file today"),
    ]})
    rc = {"doc_desc_words": 150, "prompt_desc_words": 40, "procedures_per_technique": 2,
          "procedure_min_words": 8, "procedure_max_words": 60}
    leak = LeakFilter(["the malware dumps the sam database to a temporary file"], 0.5)
    docs, stats = build_technique_docs(attack, rc, leak)
    assert [d["id"] for d in docs] == ["T1003"]
    assert docs[0]["procedures"] == ["Mimikatz reads credentials from LSASS process memory on hosts"]
    assert stats["procedures_dropped_leak"] == 1
    assert "Credential Access" in docs[0]["text"] and "Citation" not in docs[0]["text"]
