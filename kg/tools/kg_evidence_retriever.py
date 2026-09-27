from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lanternquest.planning import EvidenceHit

MANUAL_EVIDENCE_STATUSES = {"source_checked", "expert_confirmed", "domain_approved"}


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _character_bigrams(text: str) -> set[str]:
    compact = "".join(text.lower().split())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


class KGEvidenceRetriever:
    """Expose only traceable, reviewed KG evidence to the planning engine."""

    def __init__(self, graph: dict[str, Any]) -> None:
        self.graph = graph
        self.nodes = {node["id"]: node for node in graph["nodes"]}
        self.outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in graph["edges"]:
            self.outgoing[edge["from"]].append(edge)
        self.queries: list[str] = []
        self.excluded_neural_candidates = [
            node["id"]
            for node in graph["nodes"]
            if node["type"] == "AlignmentCandidate"
            and node["attributes"].get("gate_decision") != "accepted"
        ]

    def _manual_hits(self) -> list[EvidenceHit]:
        hits: list[EvidenceHit] = []
        for claim_id, claim in self.nodes.items():
            if claim["type"] != "Claim" or claim["status"] not in MANUAL_EVIDENCE_STATUSES:
                continue
            support_edges = [
                edge
                for edge in self.outgoing[claim_id]
                if edge["type"] == "SUPPORTED_BY"
            ]
            scope_edges = [
                edge for edge in self.outgoing[claim_id] if edge["type"] == "HAS_SCOPE"
            ]
            if not support_edges or not scope_edges:
                continue
            for edge in support_edges:
                evidence = self.nodes.get(edge["to"])
                if evidence is None or evidence["type"] != "EvidenceSegment":
                    continue
                source_edges = [
                    item
                    for item in self.outgoing[evidence["id"]]
                    if item["type"] == "EXTRACTED_FROM"
                ]
                if not source_edges:
                    continue
                hits.append(
                    EvidenceHit(
                        evidence_id=evidence["id"],
                        supported_claim_ids=(claim_id,),
                        source_id=source_edges[0]["to"],
                        content=claim["attributes"].get("claim_text", claim["label"]),
                    )
                )
        return hits

    def _accepted_neural_hits(self) -> list[EvidenceHit]:
        hits: list[EvidenceHit] = []
        for candidate_id, candidate in self.nodes.items():
            if candidate["type"] != "AlignmentCandidate":
                continue
            if candidate["attributes"].get("gate_decision") != "accepted":
                continue
            target_edges = [
                edge
                for edge in self.outgoing[candidate_id]
                if edge["type"] == "ALIGNMENT_TARGET"
            ]
            source_edges = [
                edge
                for edge in self.outgoing[candidate_id]
                if edge["type"] == "ALIGNMENT_SOURCE"
            ]
            for target in target_edges:
                hits.append(
                    EvidenceHit(
                        evidence_id=candidate_id,
                        supported_claim_ids=(target["to"],),
                        source_id=source_edges[0]["to"] if source_edges else "",
                        content=candidate["label"],
                    )
                )
        return hits

    def eligible_hits(self) -> list[EvidenceHit]:
        by_id = {
            hit.evidence_id: hit
            for hit in [*self._manual_hits(), *self._accepted_neural_hits()]
        }
        return [by_id[item_id] for item_id in sorted(by_id)]

    def retrieve(self, query: str, limit: int) -> list[EvidenceHit]:
        self.queries.append(query)
        query_bigrams = _character_bigrams(query)
        ranked: list[tuple[float, str, EvidenceHit]] = []
        for hit in self.eligible_hits():
            claim_id = hit.supported_claim_ids[0]
            if claim_id in query:
                score = 2.0
            else:
                content_bigrams = _character_bigrams(hit.content)
                union = query_bigrams | content_bigrams
                score = len(query_bigrams & content_bigrams) / len(union) if union else 0.0
            if score > 0:
                ranked.append((score, hit.evidence_id, hit))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked[:limit]]
