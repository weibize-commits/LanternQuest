from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kg.tools.build_seed_graph import (
    DEFAULT_INVENTORY,
    DEFAULT_TASK,
    ROOT,
    build_graph,
    load_inventory,
    load_json,
)

DEFAULT_SOURCE_ROOT = ROOT / "汴京灯笼张素材资料"
DEFAULT_OUTPUT = ROOT / "kg" / "data" / "dev_multimodal_graph.json"

ASSET_TYPES = {
    "image": "ImageAsset",
    "video": "VideoAsset",
    "audio": "AudioAsset",
    "text": "TextAsset",
    "document": "DocumentAsset",
    "web_index": "WebIndexAsset",
}


def _group_id(label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
    return f"source_group:{digest}"


def _asset_id(sha256: str) -> str:
    return f"asset:{sha256}"


def _enrich(path: Path, media_type: str) -> dict[str, Any]:
    if not path.is_file():
        return {"metadata_status": "source_file_missing"}
    try:
        if media_type == "image":
            from PIL import Image

            with Image.open(path) as image:
                return {
                    "metadata_status": "extracted",
                    "width_pixels": image.width,
                    "height_pixels": image.height,
                    "color_mode": image.mode,
                    "image_format": image.format,
                }
        if media_type in {"text", "web_index"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            return {
                "metadata_status": "extracted",
                "character_count": len(text),
                "line_count": len(text.splitlines()),
            }
        if media_type == "document" and path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(path)
            return {
                "metadata_status": "extracted",
                "page_count": len(reader.pages),
            }
        if media_type in {"audio", "video"}:
            return {
                "metadata_status": "container_probe_pending",
                "duration_seconds": None,
            }
        return {"metadata_status": "basic_inventory_only"}
    except Exception as error:
        return {
            "metadata_status": "extraction_error",
            "metadata_error_type": type(error).__name__,
        }


def build_multimodal_graph(
    task: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    source_root: Path,
) -> dict[str, Any]:
    graph = build_graph(task, inventory)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = [
        {key: value for key, value in edge.items() if key != "id"}
        for edge in graph["edges"]
    ]
    source_ids_by_hash: dict[str, list[str]] = defaultdict(list)
    for source in inventory.values():
        source_ids_by_hash[source["sha256"]].append(source["source_id"])

    asset_labels: dict[str, str] = {}
    for source_id, source in sorted(inventory.items()):
        graph_source_id = f"source:{source_id}"
        source_node = nodes.get(graph_source_id)
        attributes = {
            "source_id": source_id,
            "relative_path": source["relative_path"],
            "media_type": source["media_type"],
            "sha256": source["sha256"],
            "rights_status": source.get("rights_status", "unreviewed"),
            "review_status": source.get("review_status", "unreviewed"),
        }
        if source_node is None:
            nodes[graph_source_id] = {
                "id": graph_source_id,
                "type": "SourceRecord",
                "label": source["filename"],
                "status": source.get("review_status", "unreviewed"),
                "attributes": attributes,
            }
        else:
            source_node["attributes"].update(attributes)

        asset_id = _asset_id(source["sha256"])
        asset_labels.setdefault(asset_id, source["filename"])
        if asset_id not in nodes:
            absolute_path = source_root / Path(source["relative_path"])
            nodes[asset_id] = {
                "id": asset_id,
                "type": ASSET_TYPES.get(source["media_type"], "MediaAsset"),
                "label": asset_labels[asset_id],
                "status": "unreviewed",
                "attributes": {
                    "sha256": source["sha256"],
                    "media_type": source["media_type"],
                    "extension": source["extension"],
                    "size_bytes": source["size_bytes"],
                    "source_location_count": len(source_ids_by_hash[source["sha256"]]),
                    **_enrich(absolute_path, source["media_type"]),
                },
            }
        edges.append(
            {
                "type": "HAS_CONTENT",
                "from": graph_source_id,
                "to": asset_id,
                "status": "unreviewed",
                "attributes": {"content_addressed_by": "sha256"},
            }
        )

        group_label = source["source_group"]
        group_id = _group_id(group_label)
        if group_id not in nodes:
            nodes[group_id] = {
                "id": group_id,
                "type": "SourceGroup",
                "label": group_label,
                "status": "unreviewed",
                "attributes": {"grouping_basis": "top_level_source_directory"},
            }
        edges.append(
            {
                "type": "IN_SOURCE_GROUP",
                "from": graph_source_id,
                "to": group_id,
                "status": "unreviewed",
                "attributes": {},
            }
        )

    sorted_nodes = sorted(nodes.values(), key=lambda item: item["id"])
    edges.sort(key=lambda item: (item["from"], item["type"], item["to"]))
    sorted_edges = []
    for index, edge in enumerate(edges, start=1):
        sorted_edges.append({"id": f"edge:{index:05d}", **edge})

    media_counts = Counter(source["media_type"] for source in inventory.values())
    graph.update(
        {
            "graph_id": f"dev:{task['task_id']}:multimodal:v0.1",
            "generated_from": [
                "annotations/seed/task_needle_piercing_v0.json",
                "artifacts/provenance/source_inventory.jsonl",
                "artifacts/provenance/corpus_audit.json",
            ],
            "statistics": {
                "source_records": len(inventory),
                "unique_content_assets": len(source_ids_by_hash),
                "duplicate_content_groups": sum(
                    len(source_ids) > 1 for source_ids in source_ids_by_hash.values()
                ),
                "media_source_counts": dict(sorted(media_counts.items())),
                "semantic_claims": sum(
                    node["type"] == "Claim" for node in sorted_nodes
                ),
                "semantic_boundary": (
                    "Only the first task's source-checked claims are semantic assertions; "
                    "the remaining corpus nodes are provenance and media assets."
                ),
            },
            "nodes": sorted_nodes,
            "edges": sorted_edges,
        }
    )
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the full multimodal development graph")
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inventory = load_inventory(args.inventory)
    graph = build_multimodal_graph(load_json(args.task), inventory, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} with {len(graph['nodes'])} nodes, "
        f"{len(graph['edges'])} edges, and "
        f"{graph['statistics']['unique_content_assets']} content assets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
