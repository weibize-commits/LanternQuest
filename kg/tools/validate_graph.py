from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ALLOWED_GRAPH_STATUSES = {
    "candidate",
    "source_checked",
    "domain_pending",
    "expert_confirmed",
    "rejected",
}
ALLOWED_ITEM_STATUSES = ALLOWED_GRAPH_STATUSES | {"unreviewed"}
EVIDENCE_SEGMENT_TYPES = {
    "EvidenceSegment",
    "ImageRegion",
    "VideoSegment",
    "AudioSegment",
    "TranscriptSegment",
    "DocumentRegion",
}
MEDIA_ASSET_TYPES = {
    "MediaAsset",
    "ImageAsset",
    "VideoAsset",
    "AudioAsset",
    "TextAsset",
    "DocumentAsset",
    "WebIndexAsset",
}


def validate_graph(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"schema_version", "graph_id", "status", "source_task_id", "nodes", "edges"}
    missing = required - graph.keys()
    if missing:
        errors.append(f"graph missing required fields: {sorted(missing)}")
        return errors
    if graph["schema_version"] != "0.1":
        errors.append("schema_version must be '0.1'")
    if graph["status"] not in ALLOWED_GRAPH_STATUSES:
        errors.append(f"invalid graph status: {graph['status']!r}")
    if not isinstance(graph["nodes"], list) or not isinstance(graph["edges"], list):
        errors.append("nodes and edges must be arrays")
        return errors

    node_ids = [item.get("id") for item in graph["nodes"]]
    edge_ids = [item.get("id") for item in graph["edges"]]
    for item_id, count in Counter(node_ids).items():
        if not item_id:
            errors.append("node id must be a non-empty string")
        elif count > 1:
            errors.append(f"duplicate node id: {item_id}")
    for item_id, count in Counter(edge_ids).items():
        if not item_id:
            errors.append("edge id must be a non-empty string")
        elif count > 1:
            errors.append(f"duplicate edge id: {item_id}")

    nodes = {item.get("id"): item for item in graph["nodes"] if item.get("id")}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for node in graph["nodes"]:
        node_id = node.get("id", "<missing>")
        for field in ("type", "label", "status", "attributes"):
            if field not in node:
                errors.append(f"node {node_id} missing {field}")
        if node.get("status") not in ALLOWED_ITEM_STATUSES:
            errors.append(f"node {node_id} has invalid status {node.get('status')!r}")
        if "attributes" in node and not isinstance(node["attributes"], dict):
            errors.append(f"node {node_id} attributes must be an object")

    for edge in graph["edges"]:
        edge_id = edge.get("id", "<missing>")
        for field in ("type", "from", "to", "status", "attributes"):
            if field not in edge:
                errors.append(f"edge {edge_id} missing {field}")
        source = edge.get("from")
        target = edge.get("to")
        if source not in nodes:
            errors.append(f"edge {edge_id} has missing source node {source!r}")
        if target not in nodes:
            errors.append(f"edge {edge_id} has missing target node {target!r}")
        if edge.get("status") not in ALLOWED_ITEM_STATUSES:
            errors.append(f"edge {edge_id} has invalid status {edge.get('status')!r}")
        if source in nodes:
            outgoing[source].append(edge)
        if target in nodes:
            incoming[target].append(edge)

    def outgoing_of(node_id: str, relation: str) -> list[dict[str, Any]]:
        return [edge for edge in outgoing[node_id] if edge.get("type") == relation]

    for node_id, node in nodes.items():
        node_type = node.get("type")
        attrs = node.get("attributes", {})
        if node_type == "Claim":
            if not outgoing_of(node_id, "SUPPORTED_BY"):
                errors.append(f"claim {node_id} has no SUPPORTED_BY edge")
            if not outgoing_of(node_id, "HAS_SCOPE"):
                errors.append(f"claim {node_id} has no HAS_SCOPE edge")
            if not attrs.get("claim_text"):
                errors.append(f"claim {node_id} has no claim_text")
        elif node_type in EVIDENCE_SEGMENT_TYPES:
            if not attrs.get("locator"):
                errors.append(f"evidence segment {node_id} has no locator")
            if not outgoing_of(node_id, "EXTRACTED_FROM"):
                errors.append(f"evidence segment {node_id} has no EXTRACTED_FROM edge")
        elif node_type == "SourceRecord":
            for field in ("source_id", "relative_path", "sha256", "rights_status", "review_status"):
                if not attrs.get(field):
                    errors.append(f"source {node_id} has no {field}")
            sha256 = attrs.get("sha256", "")
            if len(sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in sha256):
                errors.append(f"source {node_id} has invalid sha256")
        elif node_type == "DecisionOption":
            if not outgoing_of(node_id, "REQUIRES_EVIDENCE"):
                errors.append(f"decision option {node_id} has no evidence obligation")
        elif node_type == "EvidenceObligation":
            if not outgoing_of(node_id, "OBLIGATION_FOR_CLAIM"):
                errors.append(f"evidence obligation {node_id} has no target claim")
        elif node_type == "UnknownCondition":
            if attrs.get("state_value") != "unknown":
                errors.append(f"unknown condition {node_id} must have state_value='unknown'")

    for edge in graph["edges"]:
        source_type = nodes.get(edge.get("from"), {}).get("type")
        target_type = nodes.get(edge.get("to"), {}).get("type")
        relation = edge.get("type")
        if relation in {"EXTRACTED_FROM", "ACCESSIBLE_VIA"}:
            if source_type not in EVIDENCE_SEGMENT_TYPES or target_type != "SourceRecord":
                errors.append(
                    f"edge {edge.get('id')} relation {relation} expects an evidence "
                    f"segment and SourceRecord, got {(source_type, target_type)}"
                )
            continue
        if relation == "HAS_SEGMENT":
            if source_type not in MEDIA_ASSET_TYPES or target_type not in EVIDENCE_SEGMENT_TYPES:
                errors.append(
                    f"edge {edge.get('id')} relation HAS_SEGMENT expects a media asset "
                    f"and evidence segment, got {(source_type, target_type)}"
                )
            continue
        expected: dict[str, tuple[str, str]] = {
            "SUPPORTED_BY": ("Claim", "EvidenceSegment"),
            "HAS_SCOPE": ("Claim", "ApplicabilityScope"),
            "HAS_OPTION": ("Decision", "DecisionOption"),
            "REQUIRES_EVIDENCE": ("DecisionOption", "EvidenceObligation"),
            "OBLIGATION_FOR_CLAIM": ("EvidenceObligation", "Claim"),
        }
        if relation in expected and (source_type, target_type) != expected[relation]:
            errors.append(
                f"edge {edge.get('id')} relation {relation} expects {expected[relation]}, "
                f"got {(source_type, target_type)}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a LanternQuest graph JSON file")
    parser.add_argument("graph", type=Path)
    args = parser.parse_args()
    with args.graph.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    errors = validate_graph(graph)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"valid: {args.graph} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
