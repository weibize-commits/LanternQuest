from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lanternquest.dynamic_eval import (
    DynamicCaseScore,
    DynamicMethod,
    execute_frozen_controller,
    join_dynamic_to_base,
    load_jsonl,
    score_response,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DYNAMIC = ROOT / "annotations" / "benchmark" / "lanternquest_dynamic_frozen_v2_1.jsonl"
DEFAULT_BASE = ROOT / "annotations" / "benchmark" / "lanternquest_frozen_v1.jsonl"
METHODS: tuple[DynamicMethod, ...] = ("full_ENSR", "ENSR_base", "B4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def aggregate(scores: list[DynamicCaseScore]) -> dict[str, Any]:
    metrics = (
        "terminal_success",
        "persistent_obligation_resolved_correctly",
        "unsupported_action",
        "repair_locality_success",
        "evidence_trace_complete",
        "avoidable_rework_count",
        "model_calls",
        "retrieval_calls",
    )
    output: dict[str, Any] = {}
    for method in METHODS:
        rows = [score for score in scores if score.method == method]
        output[method] = {"n": len(rows)}
        for metric in metrics:
            values = [float(getattr(score, metric)) for score in rows]
            output[method][metric] = sum(values) / len(values) if values else None
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen deterministic dynamic-controller replay"
    )
    parser.add_argument(
        "--split", choices=("train", "development", "test"), required=True
    )
    parser.add_argument("--dynamic-dataset", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = None
    if args.split == "test":
        if args.protocol is None or not args.protocol.is_file():
            raise ValueError("formal test requires a frozen protocol")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
        if protocol.get("status") != "frozen_not_run":
            raise ValueError("protocol must be frozen_not_run")
        if protocol["dataset_sha256"] != sha256(args.dynamic_dataset):
            raise ValueError("dynamic dataset differs from frozen protocol")
        if protocol["base_dataset_sha256"] != sha256(args.base_dataset):
            raise ValueError("base dataset differs from frozen protocol")
        for relative, expected in protocol["implementation_sha256"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"frozen implementation changed: {relative}")

    dynamic = [
        row for row in load_jsonl(args.dynamic_dataset) if row["split"] == args.split
    ]
    joined = join_dynamic_to_base(dynamic, load_jsonl(args.base_dataset))
    expected = {"train": 29, "development": 9, "test": 10}[args.split]
    if len(joined) != expected:
        raise ValueError(f"expected {expected} cases, found {len(joined)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[DynamicCaseScore] = []
    traces: dict[str, Any] = {}
    for dynamic_case, base_case in joined:
        for method in METHODS:
            started = time.perf_counter()
            response = execute_frozen_controller(method, dynamic_case, base_case)
            score = score_response(
                dynamic_case,
                base_case,
                method,
                response,
                wall_seconds=time.perf_counter() - started,
                input_tokens=0,
                output_tokens=0,
            )
            score.model_calls = 0
            scores.append(score)
            traces[f"{dynamic_case['case_id']}::{method}"] = {
                "pre_event_trace_source": "expert-approved frozen base-case trace",
                "response": response.model_dump(mode="json"),
                "score": score.model_dump(mode="json"),
            }

    scores_path = output_dir / "scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(score.model_dump_json() + "\n")
    write_json(output_dir / "traces.json", traces)
    summary = {
        "schema_version": "1.0",
        "run_id": output_dir.name,
        "split": args.split,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "design": (
            "Deterministic controller replay on identical expert-approved pre-event "
            "action traces; neural proposal held fixed; no model training or inference."
        ),
        "dataset_sha256": sha256(args.dynamic_dataset),
        "base_dataset_sha256": sha256(args.base_dataset),
        "implementation_sha256": {
            "src/lanternquest/dynamic_eval.py": sha256(
                ROOT / "src" / "lanternquest" / "dynamic_eval.py"
            ),
            "scripts/run_lanternquest_dynamic_controller_v2_1.py": sha256(
                ROOT / "scripts" / "run_lanternquest_dynamic_controller_v2_1.py"
            ),
        },
        "case_count": len(joined),
        "run_count": len(scores),
        "aggregate": aggregate(scores),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            **summary,
            "status": "complete",
            "scores_sha256": sha256(scores_path),
            "traces_sha256": sha256(output_dir / "traces.json"),
            "protocol_sha256": sha256(args.protocol) if args.protocol else None,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
