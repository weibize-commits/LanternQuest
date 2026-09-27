from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kg.tools.build_seed_graph import (
    DEFAULT_INVENTORY,
    DEFAULT_TASK,
    ROOT,
    load_inventory,
    load_json,
)
from kg.tools.neurosymbolic_gate import DEFAULT_POLICY, evaluate_candidate

MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
MODEL_REVISION = "f1e7c92d08b9f674ce4c4cb7a1284a8be2da81bb"
DEFAULT_SOURCE_ROOT = ROOT / "汴京灯笼张素材资料"
DEFAULT_OUTPUT = ROOT / "kg" / "data" / "chinese_clip_first_task_pilot.json"
IMAGE_SOURCE_IDS = (
    "src_51c63e8ce4b1d5af",
    "src_abc7513f16332e0b",
    "src_a8f853229e31ecce",
    "src_d01df7f40b72bec7",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_file_manifest(snapshot: Path) -> list[dict[str, Any]]:
    manifest = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        manifest.append(
            {
                "relative_path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return manifest


def _claim_catalog(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "claim_id": f"claim:{evidence['evidence_id']}",
            "text": evidence["claim"],
            "scope": evidence["scope"],
        }
        for evidence in task["evidence"]
    ]


def run_pilot(
    task_path: Path,
    inventory_path: Path,
    source_root: Path,
    cache_dir: Path,
    output: Path,
) -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    task = load_json(task_path)
    inventory = load_inventory(inventory_path)
    claims = _claim_catalog(task)
    image_records = [inventory[source_id] for source_id in IMAGE_SOURCE_IDS]
    image_paths = [source_root / record["relative_path"] for record in image_records]
    if missing := [str(path) for path in image_paths if not path.is_file()]:
        raise FileNotFoundError(f"Pilot image sources are missing: {missing}")

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "vocab.txt", "pytorch_model.bin"],
        )
    )
    model_files = _model_file_manifest(snapshot)
    weight_files = [
        item
        for item in model_files
        if item["relative_path"].endswith((".bin", ".pt", ".safetensors"))
    ]
    if not weight_files:
        raise RuntimeError("Downloaded model snapshot does not contain a weight file")
    combined_hash = hashlib.sha256(
        "".join(item["sha256"] for item in weight_files).encode("ascii")
    ).hexdigest()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ChineseCLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    model = ChineseCLIPModel.from_pretrained(snapshot, local_files_only=True)
    model.to(device)
    model.eval()

    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    text_inputs = processor(
        text=[claim["text"] for claim in claims],
        padding=True,
        return_tensors="pt",
    ).to(device)
    image_inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        image_features = model.get_image_features(**image_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cosine = image_features @ text_features.T

    policy = load_json(DEFAULT_POLICY)
    alignments = []
    for image_index, record in enumerate(image_records):
        ranked = torch.argsort(cosine[image_index], descending=True).tolist()
        matches = []
        for claim_index in ranked[:3]:
            raw_cosine = float(cosine[image_index, claim_index].item())
            normalized_score = max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0))
            claim = claims[claim_index]
            candidate = {
                "schema_version": "0.1",
                "candidate_id": (
                    f"alignment:{record['source_id']}:{claim['claim_id'].split(':', 1)[1]}"
                ),
                "source_observation_id": f"source:{record['source_id']}",
                "target_entity_id": claim["claim_id"],
                "model": {
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "weights_sha256": combined_hash,
                    "modalities": ["image", "text"],
                },
                "scores": {
                    "cross_modal_similarity": normalized_score,
                    "extractor_confidence": 0.5,
                },
                "rule_checks": [
                    {
                        "rule_id": "source_traceable",
                        "result": "pass",
                        "hard": True,
                        "evidence_ids": [record["source_id"]],
                    },
                    {
                        "rule_id": "scope_compatible",
                        "result": "unknown",
                        "hard": True,
                        "evidence_ids": [],
                    },
                    {
                        "rule_id": "no_known_conflict",
                        "result": "unknown",
                        "hard": True,
                        "evidence_ids": [],
                    },
                    {
                        "rule_id": "safety_boundary_preserved",
                        "result": "pass",
                        "hard": True,
                        "evidence_ids": [],
                    },
                ],
                "source_policy": {
                    "review_status": "source_checked",
                    "rights_status": "unreviewed",
                    "processing_scope": "local_internal",
                },
            }
            gate = evaluate_candidate(candidate, policy)
            matches.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["text"],
                    "raw_cosine": round(raw_cosine, 6),
                    "normalized_similarity": round(normalized_score, 6),
                    "candidate": candidate,
                    "gate": gate,
                }
            )
        alignments.append(
            {
                "source_id": record["source_id"],
                "relative_path": record["relative_path"],
                "top_matches": matches,
            }
        )

    payload = {
        "schema_version": "0.1",
        "reporting_boundary": "development_pilot_not_accuracy_evaluation",
        "purpose": "Verify local frozen image-text encoding and symbolic gating plumbing",
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "combined_weight_file_sha256": combined_hash,
            "snapshot_path": str(snapshot),
            "files": model_files,
        },
        "runtime": {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "inputs": {
            "image_count": len(image_records),
            "claim_count": len(claims),
            "local_only": True,
            "ground_truth_labels_available": False,
        },
        "alignments": alignments,
        "interpretation": (
            "Scores are uncalibrated retrieval candidates. Scope and conflict checks are "
            "unknown, so the symbolic gate must not auto-accept them."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Chinese-CLIP first-task pilot")
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "artifacts" / "models" / "huggingface",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_pilot(
        args.task, args.inventory, args.source_root, args.cache_dir, args.output
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runtime": result["runtime"],
                "inputs": result["inputs"],
                "gate_decisions": [
                    match["gate"]["decision"]
                    for item in result["alignments"]
                    for match in item["top_matches"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
