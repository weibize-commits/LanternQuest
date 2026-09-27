from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "kg" / "data" / "dev_multimodal_graph.json"
DEFAULT_PILOT = ROOT / "kg" / "data" / "chinese_clip_first_task_pilot.json"
DEFAULT_OUTPUT = (
    ROOT / "kg" / "data" / "dev_multimodal_neurosymbolic_graph.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def integrate(graph: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = [
        {key: value for key, value in edge.items() if key != "id"}
        for edge in graph["edges"]
    ]
    model = pilot["model"]
    model_node_id = "neural_model:chinese_clip_vit_base_patch16"
    nodes[model_node_id] = {
        "id": model_node_id,
        "type": "NeuralModel",
        "label": model["model_id"],
        "status": "candidate",
        "attributes": {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "weights_sha256": model["combined_weight_file_sha256"],
            "execution": "local",
            "fine_tuned": False,
        },
    }

    decisions: Counter[str] = Counter()
    for alignment in pilot["alignments"]:
        for match in alignment["top_matches"]:
            candidate = match["candidate"]
            gate = match["gate"]
            candidate_id = candidate["candidate_id"]
            decisions[gate["decision"]] += 1
            nodes[candidate_id] = {
                "id": candidate_id,
                "type": "AlignmentCandidate",
                "label": f"{alignment['source_id']} → {match['claim_id']}",
                "status": "candidate",
                "attributes": {
                    "raw_cosine": match["raw_cosine"],
                    "normalized_similarity": match["normalized_similarity"],
                    "fused_score": gate["fused_score"],
                    "gate_decision": gate["decision"],
                    "gate_reasons": gate["reasons"],
                    "reporting_boundary": pilot["reporting_boundary"],
                },
            }
            edges.extend(
                [
                    {
                        "type": "ALIGNMENT_SOURCE",
                        "from": candidate_id,
                        "to": candidate["source_observation_id"],
                        "status": "candidate",
                        "attributes": {},
                    },
                    {
                        "type": "ALIGNMENT_TARGET",
                        "from": candidate_id,
                        "to": candidate["target_entity_id"],
                        "status": "candidate",
                        "attributes": {},
                    },
                    {
                        "type": "GENERATED_BY_MODEL",
                        "from": candidate_id,
                        "to": model_node_id,
                        "status": "candidate",
                        "attributes": {},
                    },
                ]
            )
            for check in candidate["rule_checks"]:
                rule_id = f"symbolic_rule:{check['rule_id']}"
                if rule_id not in nodes:
                    nodes[rule_id] = {
                        "id": rule_id,
                        "type": "SymbolicRule",
                        "label": check["rule_id"],
                        "status": "candidate",
                        "attributes": {"hard": check["hard"]},
                    }
                edges.append(
                    {
                        "type": "EVALUATED_BY_RULE",
                        "from": candidate_id,
                        "to": rule_id,
                        "status": "candidate",
                        "attributes": {"result": check["result"]},
                    }
                )

    graph["graph_id"] = graph["graph_id"].replace(
        ":multimodal:", ":multimodal_neurosymbolic:"
    )
    graph["generated_from"] = [
        *graph["generated_from"],
        "kg/data/chinese_clip_first_task_pilot.json",
        "kg/rules/neurosymbolic_policy_v0.json",
    ]
    graph["statistics"].update(
        {
            "neural_alignment_candidates": sum(decisions.values()),
            "alignment_gate_decisions": dict(sorted(decisions.items())),
            "accepted_neural_assertions": decisions.get("accepted", 0),
            "neural_reporting_boundary": pilot["reporting_boundary"],
        }
    )
    graph["nodes"] = sorted(nodes.values(), key=lambda item: item["id"])
    edges.sort(key=lambda item: (item["from"], item["type"], item["to"]))
    graph["edges"] = [
        {"id": f"edge:{index:05d}", **edge}
        for index, edge in enumerate(edges, start=1)
    ]
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate the neural pilot into the KG")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    graph = integrate(load_json(args.graph), load_json(args.pilot))
    args.output.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} with {len(graph['nodes'])} nodes and "
        f"{len(graph['edges'])} edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
