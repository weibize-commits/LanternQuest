from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = (
    ROOT / "kg" / "data" / "dev_multimodal_neurosymbolic_graph.json"
)
DEFAULT_PILOT = ROOT / "kg" / "data" / "faster_whisper_field07_pilot.json"
DEFAULT_DIAGNOSTIC = (
    ROOT / "kg" / "data" / "faster_whisper_field07_diagnostic.json"
)
DEFAULT_OUTPUT = (
    ROOT / "kg" / "data" / "dev_multimodal_neurosymbolic_asr_graph.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def integrate(
    graph: dict[str, Any],
    pilot: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(graph)
    nodes = {node["id"]: node for node in result["nodes"]}
    edges = [
        {key: value for key, value in edge.items() if key != "id"}
        for edge in result["edges"]
    ]
    source_id = f"source:{pilot['source']['source_id']}"
    asset_id = f"asset:{pilot['source']['sha256']}"
    reference_id = f"source:{pilot['reference_text_source_id']}"
    for required_id in (source_id, asset_id, reference_id):
        if required_id not in nodes:
            raise RuntimeError(f"pilot references missing graph node: {required_id}")

    model = pilot["model"]
    model_node_id = "neural_model:faster_whisper_small"
    nodes[model_node_id] = {
        "id": model_node_id,
        "type": "NeuralModel",
        "label": model["model_id"],
        "status": "candidate",
        "attributes": {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "weights_sha256": model["weight_file_sha256"],
            "execution": model["execution"],
            "compute_type": model["compute_type"],
            "fine_tuned": model["fine_tuned"],
        },
    }

    for segment in pilot["segments"]:
        segment_id = segment["segment_id"]
        nodes[segment_id] = {
            "id": segment_id,
            "type": "TranscriptSegment",
            "label": segment["text"] or f"ASR segment {segment_id}",
            "status": "candidate",
            "attributes": {
                "locator": (
                    f"{segment['start_seconds']:.3f}-"
                    f"{segment['end_seconds']:.3f}s"
                ),
                "start_seconds": segment["start_seconds"],
                "end_seconds": segment["end_seconds"],
                "text": segment["text"],
                "avg_logprob": segment["avg_logprob"],
                "no_speech_probability": segment["no_speech_probability"],
                "review_status": "human_review",
                "rights_status": pilot["source"]["rights_status"],
                "reporting_boundary": pilot["reporting_boundary"],
            },
        }
        edges.extend(
            [
                {
                    "type": "HAS_SEGMENT",
                    "from": asset_id,
                    "to": segment_id,
                    "status": "candidate",
                    "attributes": {},
                },
                {
                    "type": "EXTRACTED_FROM",
                    "from": segment_id,
                    "to": source_id,
                    "status": "candidate",
                    "attributes": {},
                },
                {
                    "type": "GENERATED_BY_MODEL",
                    "from": segment_id,
                    "to": model_node_id,
                    "status": "candidate",
                    "attributes": {},
                },
                {
                    "type": "DIAGNOSTIC_REFERENCE",
                    "from": segment_id,
                    "to": reference_id,
                    "status": "candidate",
                    "attributes": {
                        "reference_is_gold_standard": False,
                    },
                },
            ]
        )

    result["graph_id"] = result["graph_id"].replace(
        ":multimodal_neurosymbolic:",
        ":multimodal_neurosymbolic_asr:",
    )
    result["generated_from"] = [
        *result["generated_from"],
        "kg/data/faster_whisper_field07_pilot.json",
        "kg/data/faster_whisper_field07_diagnostic.json",
    ]
    result["statistics"].update(
        {
            "asr_candidate_segments": len(pilot["segments"]),
            "asr_human_review_segments": len(pilot["segments"]),
            "accepted_asr_assertions": 0,
            "asr_diagnostic_character_error_rate": diagnostic[
                "diagnostic_character_error_rate"
            ],
            "asr_diagnostic_reporting_boundary": diagnostic[
                "reporting_boundary"
            ],
        }
    )
    result["nodes"] = sorted(nodes.values(), key=lambda item: item["id"])
    edges.sort(key=lambda item: (item["from"], item["type"], item["to"]))
    result["edges"] = [
        {"id": f"edge:{index:05d}", **edge}
        for index, edge in enumerate(edges, start=1)
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate ASR candidates into the KG")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    graph = integrate(
        load_json(args.graph),
        load_json(args.pilot),
        load_json(args.diagnostic),
    )
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
