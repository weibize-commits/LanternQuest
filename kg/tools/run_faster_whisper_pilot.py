from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "artifacts" / "provenance" / "source_inventory.jsonl"
DEFAULT_CACHE = ROOT / "artifacts" / "models" / "huggingface"
DEFAULT_OUTPUT = ROOT / "kg" / "data" / "faster_whisper_field07_pilot.json"

SOURCE_ID = "src_24551a9197af8241"
REFERENCE_TEXT_SOURCE_ID = "src_16b2d629f1facdb9"
MODEL_ID = "Systran/faster-whisper-small"
MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_files_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def inventory_record(path: Path, source_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("source_id") == source_id:
                return record
    raise RuntimeError(f"source not found in inventory: {source_id}")


def locate_source(record: dict[str, Any]) -> Path:
    expected_size = int(record["size_bytes"])
    expected_hash = record["sha256"]
    extension = record["extension"].lower()
    ignored_roots = {
        ".git",
        ".venv",
        ".venv-multimodal",
        "artifacts",
        "kg",
    }
    candidates: list[Path] = []
    for child in ROOT.iterdir():
        if not child.is_dir() or child.name in ignored_roots:
            continue
        for path in child.rglob(f"*{extension}"):
            if path.is_file() and path.stat().st_size == expected_size:
                candidates.append(path)
    matches = [path for path in candidates if file_sha256(path) == expected_hash]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one local file for {record['source_id']}, found {len(matches)}"
        )
    return matches[0]


def download_model(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
    )
    return Path(snapshot)


def run_pilot(
    inventory_path: Path,
    cache_dir: Path,
    cpu_threads: int,
) -> dict[str, Any]:
    source = inventory_record(inventory_path, SOURCE_ID)
    audio_path = locate_source(source)
    model_path = download_model(cache_dir)
    weight_path = model_path / "model.bin"
    if not weight_path.exists():
        raise RuntimeError(f"missing frozen model weight file: {weight_path}")

    manifest_files = [
        path
        for path in model_path.iterdir()
        if path.is_file() and path.name not in {"README.md", ".gitattributes"}
    ]
    model = WhisperModel(
        str(model_path),
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
    )
    started = time.perf_counter()
    segment_iterator, info = model.transcribe(
        str(audio_path),
        language="zh",
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    raw_segments = list(segment_iterator)
    elapsed = time.perf_counter() - started

    segments = []
    for index, segment in enumerate(raw_segments, start=1):
        segments.append(
            {
                "segment_id": f"asr:{SOURCE_ID}:{index:04d}",
                "start_seconds": round(float(segment.start), 3),
                "end_seconds": round(float(segment.end), 3),
                "text": segment.text.strip(),
                "avg_logprob": round(float(segment.avg_logprob), 6),
                "no_speech_probability": round(
                    float(segment.no_speech_prob), 6
                ),
                "status": "human_review",
            }
        )

    return {
        "schema_version": "0.1",
        "pilot_id": "field07_faster_whisper_small_v0",
        "reporting_boundary": "development_pilot_not_accuracy_evaluation",
        "source": {
            "source_id": SOURCE_ID,
            "relative_path": source["relative_path"],
            "sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
            "rights_status": source["rights_status"],
            "processing_scope": "local_internal",
        },
        "reference_text_source_id": REFERENCE_TEXT_SOURCE_ID,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "faster_whisper_version": "1.2.1",
            "ctranslate2_version": "4.4.0",
            "weight_file_sha256": file_sha256(weight_path),
            "combined_model_files_sha256": combined_files_sha256(manifest_files),
            "execution": "local_cpu",
            "compute_type": "int8",
            "fine_tuned": False,
        },
        "inference": {
            "language_requested": "zh",
            "language_detected": info.language,
            "language_probability": round(float(info.language_probability), 6),
            "beam_size": 5,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "cpu_threads": cpu_threads,
            "elapsed_seconds": round(elapsed, 3),
            "audio_duration_seconds": round(float(info.duration), 3),
            "duration_after_vad_seconds": round(
                float(info.duration_after_vad), 3
            ),
        },
        "segments": segments,
        "transcript_text": "".join(segment["text"] for segment in segments),
        "interpretation": (
            "ASR output is a time-coded neural candidate. It does not replace the "
            "human field transcript and cannot support a heritage claim until review."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Field07 ASR pilot")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    args = parser.parse_args()
    pilot = run_pilot(args.inventory, args.cache_dir, args.cpu_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(pilot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {len(pilot['segments'])} segments, "
        f"{pilot['inference']['audio_duration_seconds']} s audio in "
        f"{pilot['inference']['elapsed_seconds']} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
