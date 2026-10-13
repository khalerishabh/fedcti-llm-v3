from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TACTIC_ORDER = [
    "reconnaissance", "resource-development", "initial-access", "execution", "persistence",
    "privilege-escalation", "defense-evasion", "credential-access", "discovery",
    "lateral-movement", "collection", "command-and-control", "exfiltration", "impact",
]


@dataclass
class Technique:
    id: str
    stix_id: str
    name: str
    description: str
    tactics: list[str] = field(default_factory=list)
    is_subtechnique: bool = False
    revoked: bool = False
    deprecated: bool = False

    @property
    def active(self) -> bool:
        return not (self.revoked or self.deprecated)

    @property
    def primary_tactic(self) -> str | None:
        return self.tactics[0] if self.tactics else None


class AttackIndex:
    """ATT&CK techniques from a STIX 2.1 bundle, with revoked-by remapping."""

    def __init__(self, bundle: dict):
        self.by_stix: dict[str, Technique] = {}
        revoked_by: dict[str, str] = {}
        procedures: dict[str, list[tuple[str, str]]] = {}
        for obj in bundle["objects"]:
            if obj.get("type") == "attack-pattern":
                ext_id = next((r["external_id"] for r in obj.get("external_references", [])
                               if r.get("source_name") == "mitre-attack"), None)
                if ext_id is None:
                    continue
                tactics = {p["phase_name"] for p in obj.get("kill_chain_phases", [])
                           if p.get("kill_chain_name") == "mitre-attack"}
                self.by_stix[obj["id"]] = Technique(
                    id=ext_id,
                    stix_id=obj["id"],
                    name=obj.get("name", ""),
                    description=obj.get("description", ""),
                    tactics=sorted(tactics, key=lambda t: TACTIC_ORDER.index(t)
                                   if t in TACTIC_ORDER else len(TACTIC_ORDER)),
                    is_subtechnique=bool(obj.get("x_mitre_is_subtechnique", False)),
                    revoked=bool(obj.get("revoked", False)),
                    deprecated=bool(obj.get("x_mitre_deprecated", False)),
                )
            elif obj.get("type") == "relationship" and obj.get("relationship_type") == "revoked-by":
                revoked_by[obj["source_ref"]] = obj["target_ref"]
            elif (obj.get("type") == "relationship" and obj.get("relationship_type") == "uses"
                  and obj.get("target_ref", "").startswith("attack-pattern--")
                  and not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False)
                  and obj.get("description")):
                procedures.setdefault(obj["target_ref"], []).append((obj["id"], obj["description"]))
        self.revoked_by = revoked_by
        self.procedures_by_stix = procedures
        self.by_id: dict[str, Technique] = {}
        for tech in self.by_stix.values():
            current = self.by_id.get(tech.id)
            # A revoked object can keep its old ID next to the live one; prefer the live one.
            if current is None or (not current.active and tech.active):
                self.by_id[tech.id] = tech

    @classmethod
    def from_file(cls, path: str | Path) -> "AttackIndex":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def active_ids(self, parents_only: bool = False) -> set[str]:
        return {t.id for t in self.by_id.values()
                if t.active and not (parents_only and t.is_subtechnique)}

    def get(self, tech_id: str) -> Technique:
        return self.by_id[tech_id]

    def _follow_revocations(self, tech: Technique) -> Technique | None:
        seen = set()
        while tech.revoked:
            if tech.stix_id in seen or tech.stix_id not in self.revoked_by:
                return None
            seen.add(tech.stix_id)
            tech = self.by_stix.get(self.revoked_by[tech.stix_id])
            if tech is None:
                return None
        return tech

    def resolve(self, label: str, collapse: bool = True) -> tuple[str | None, str]:
        """Map a raw label to an active ID. Reasons: ok, remapped, collapsed, deprecated,
        unknown, revoked_unresolved."""
        label = label.strip().upper()
        tech = self.by_id.get(label)
        if tech is None:
            return None, "unknown"
        reason = "ok"
        if tech.revoked:
            tech = self._follow_revocations(tech)
            if tech is None:
                return None, "revoked_unresolved"
            reason = "remapped"
        if tech.deprecated:
            return None, "deprecated"
        if collapse and tech.is_subtechnique:
            parent_id, parent_reason = self.resolve(tech.id.split(".")[0], collapse=False)
            if parent_id is None:
                return None, parent_reason
            return parent_id, "collapsed" if reason == "ok" else reason
        return tech.id, reason

    def procedures(self, tech_id: str, include_subtechniques: bool = True) -> list[tuple[str, str]]:
        """(relationship id, raw text) procedure examples, sorted by relationship id."""
        out = []
        for tech in self.by_stix.values():
            if not tech.active:
                continue
            if tech.id == tech_id or (include_subtechniques and tech.id.startswith(tech_id + ".")):
                out.extend(self.procedures_by_stix.get(tech.stix_id, []))
        return sorted(out)
