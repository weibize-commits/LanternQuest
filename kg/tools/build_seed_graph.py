from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = ROOT / "annotations" / "seed" / "task_needle_piercing_v0.json"
DEFAULT_INVENTORY = ROOT / "artifacts" / "provenance" / "source_inventory.jsonl"
DEFAULT_OUTPUT = ROOT / "kg" / "data" / "dev_seed_graph.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inventory(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records[item["source_id"]] = item
    return records


def build_graph(task: dict[str, Any], inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    def add_node(
        node_id: str,
        node_type: str,
        label: str,
        status: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if node_id in seen_nodes:
            return
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "status": status,
                "attributes": attributes or {},
            }
        )
        seen_nodes.add(node_id)

    def add_edge(
        edge_type: str,
        source: str,
        target: str,
        status: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        edges.append(
            {
                "id": "",
                "type": edge_type,
                "from": source,
                "to": target,
                "status": status,
                "attributes": attributes or {},
            }
        )

    task_id = f"task:{task['task_id']}"
    add_node(
        task_id,
        "Task",
        task["title"],
        task.get("status", "candidate"),
        {
            "included_scope": task.get("included_scope", []),
            "excluded_scope": task.get("excluded_scope", []),
        },
    )
    output_id = f"work_product:{task['task_id']}:target"
    add_node(
        output_id,
        "WorkProduct",
        task["target_output"],
        "candidate",
        {"role": "target_output"},
    )
    add_edge("TARGETS", task_id, output_id, "candidate")

    evidence_by_id = {item["evidence_id"]: item for item in task.get("evidence", [])}
    for evidence_id, evidence in evidence_by_id.items():
        segment_id = f"evidence:{evidence_id}"
        claim_id = f"claim:{evidence_id}"
        scope_id = f"scope:{evidence_id}"
        review_status = evidence.get("review_status", "candidate")
        add_node(
            segment_id,
            "EvidenceSegment",
            evidence["locator"],
            review_status,
            {"locator": evidence["locator"]},
        )
        add_node(
            claim_id,
            "Claim",
            evidence["claim"],
            review_status,
            {"claim_text": evidence["claim"]},
        )
        add_node(
            scope_id,
            "ApplicabilityScope",
            evidence["scope"],
            review_status,
            {"scope_text": evidence["scope"]},
        )
        add_edge("SUPPORTED_BY", claim_id, segment_id, review_status)
        add_edge("HAS_SCOPE", claim_id, scope_id, review_status)

        source_ids = [evidence["primary_source_id"], *evidence.get("access_source_ids", [])]
        for source_id in source_ids:
            if source_id not in inventory:
                raise ValueError(f"Source {source_id!r} is absent from the inventory")
            source = inventory[source_id]
            graph_source_id = f"source:{source_id}"
            add_node(
                graph_source_id,
                "SourceRecord",
                source["filename"],
                source.get("review_status", "unreviewed"),
                {
                    "source_id": source_id,
                    "relative_path": source["relative_path"],
                    "media_type": source["media_type"],
                    "sha256": source["sha256"],
                    "rights_status": source.get("rights_status", "unreviewed"),
                    "review_status": source.get("review_status", "unreviewed"),
                },
            )
            relation = (
                "EXTRACTED_FROM"
                if source_id == evidence["primary_source_id"]
                else "ACCESSIBLE_VIA"
            )
            add_edge(relation, segment_id, graph_source_id, review_status)

    for decision in task.get("decision_nodes", []):
        decision_id = f"decision:{decision['decision_id']}"
        decision_status = decision.get("review_status", "domain_pending")
        add_node(
            decision_id,
            "Decision",
            decision["prompt"],
            decision_status,
            {"timing": decision["timing"]},
        )
        add_edge("HAS_DECISION", task_id, decision_id, decision_status)
        for option in decision.get("options", []):
            option_id = f"option:{option['option_id']}"
            add_node(
                option_id,
                "DecisionOption",
                option["label"],
                decision_status,
                {"consequence": option["consequence"]},
            )
            add_edge("HAS_OPTION", decision_id, option_id, decision_status)
            for evidence_id in option.get("evidence_ids", []):
                if evidence_id not in evidence_by_id:
                    raise ValueError(
                        f"Decision option {option['option_id']!r} refers to missing evidence "
                        f"{evidence_id!r}"
                    )
                obligation_id = (
                    f"obligation:{decision['decision_id']}:{option['option_id']}:{evidence_id}"
                )
                add_node(
                    obligation_id,
                    "EvidenceObligation",
                    f"{option['label']} 的证据义务",
                    decision_status,
                    {
                        "decision_id": decision["decision_id"],
                        "option_id": option["option_id"],
                    },
                )
                add_edge("REQUIRES_EVIDENCE", option_id, obligation_id, decision_status)
                add_edge(
                    "OBLIGATION_FOR_CLAIM",
                    obligation_id,
                    f"claim:{evidence_id}",
                    decision_status,
                )

    for index, text in enumerate(task.get("risk_controls", []), start=1):
        risk_id = f"risk:{task['task_id']}:{index:03d}"
        add_node(risk_id, "RiskControl", text, "domain_pending", {"text": text})
        add_edge("HAS_RISK_CONTROL", task_id, risk_id, "domain_pending")

    for index, text in enumerate(task.get("known_unknowns", []), start=1):
        unknown_id = f"unknown:{task['task_id']}:{index:03d}"
        add_node(
            unknown_id,
            "UnknownCondition",
            text,
            "domain_pending",
            {"text": text, "state_value": "unknown"},
        )
        add_edge("HAS_UNKNOWN_CONDITION", task_id, unknown_id, "domain_pending")

    nodes.sort(key=lambda item: item["id"])
    edges.sort(key=lambda item: (item["from"], item["type"], item["to"]))
    for index, edge in enumerate(edges, start=1):
        edge["id"] = f"edge:{index:05d}"

    return {
        "schema_version": "0.1",
        "graph_id": f"dev:{task['task_id']}:v0.1",
        "status": "candidate",
        "source_task_id": task["task_id"],
        "generated_from": [
            "annotations/seed/task_needle_piercing_v0.json",
            "artifacts/provenance/source_inventory.jsonl",
        ],
        "nodes": nodes,
        "edges": edges,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the LanternQuest development seed graph")
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    graph = build_graph(load_json(args.task), load_inventory(args.inventory))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(graph, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"wrote {args.output} with {len(graph['nodes'])} nodes and "
        f"{len(graph['edges'])} edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
