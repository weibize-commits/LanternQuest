from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from lanternquest.benchmark import benchmark_fingerprint, load_benchmark_jsonl
from lanternquest.e21 import (
    E21CaseScore,
    E21Method,
    aggregate_scores,
    score_decision,
    score_failed_run,
)
from lanternquest.e21_methods import prepare_e21_method
from lanternquest.experiment import validate_generated_decision
from lanternquest.llama_cpp_adapter import LlamaCppAdapter
from lanternquest.llm import LLMAdapterError
from lanternquest.planning import SearchBudget

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "annotations" / "benchmark" / "lanternquest_frozen_v1.jsonl"
DEFAULT_PROTOCOL = ROOT / "configs" / "lanternquest_e21_protocol_v1_frozen.json"
DEFAULT_PREFLIGHT = ROOT / "artifacts" / "eval" / "lanternquest_e21_preflight_frozen.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "artifacts" / "eval" / "lanternquest_e21_formal_local_qwen3_8b_v1"
)
METHODS: tuple[E21Method, ...] = (
    "b0_full_context",
    "b1_rag",
    "b2_kg_rag",
    "b4_sequential",
    "iper_rag",
    "ensr",
)
EXECUTION_ORDER: tuple[E21Method, ...] = (
    "b2_kg_rag",
    "b4_sequential",
    "iper_rag",
    "ensr",
    "b0_full_context",
    "b1_rag",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_scores(path: Path, scores: dict[tuple[str, str], E21CaseScore]) -> None:
    method_order = {method: index for index, method in enumerate(METHODS)}
    ordered = sorted(
        scores.values(),
        key=lambda item: (item.case_id, method_order[item.method]),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for score in ordered:
            handle.write(score.model_dump_json() + "\n")
    temporary.replace(path)


def load_scores(path: Path) -> dict[tuple[str, str], E21CaseScore]:
    if not path.is_file():
        return {}
    output = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            score = E21CaseScore.model_validate_json(line)
            output[(score.case_id, score.method)] = score
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen E21 formal test split with a local GGUF model"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B-GGUF:Q4_K_M")
    parser.add_argument("--model-sha256")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=800)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if protocol.get("status") != "frozen":
        raise ValueError("formal E21 requires a frozen protocol")
    if preflight.get("ready_for_formal_run") is not True:
        raise ValueError("formal E21 preflight has not passed")
    protocol_key = str(args.protocol.resolve().relative_to(ROOT)).replace("\\", "/")
    if preflight.get("source_sha256", {}).get(protocol_key) != sha256(args.protocol):
        raise ValueError("formal E21 preflight did not audit this protocol file")
    implementation_hashes = protocol.get("implementation_sha256", {})
    if not implementation_hashes:
        raise ValueError("frozen protocol does not record implementation hashes")
    implementation_mismatches = {}
    for relative, expected in implementation_hashes.items():
        path = ROOT / relative
        observed = sha256(path) if path.is_file() else None
        if observed != expected:
            implementation_mismatches[relative] = (expected, observed)
    if implementation_mismatches:
        raise ValueError(
            "formal implementation differs from the frozen protocol: "
            f"{implementation_mismatches}"
        )
    model_profile = protocol.get("model_profile", {})
    if model_profile.get("status") != "frozen":
        raise ValueError("formal E21 model profile is not frozen")
    if model_profile.get("external_processing_required") is not False:
        raise ValueError("formal E21 model profile is not completely local")
    if model_profile.get("model_id") != args.model_id:
        raise ValueError("requested model identifier differs from the frozen profile")
    frozen_runtime = model_profile.get("runtime", {})
    requested_runtime = {
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "max_output_tokens": args.max_output_tokens,
        "temperature": model_profile.get("temperature"),
        "seed": 21021,
    }
    mismatched_runtime = {
        key: (frozen_runtime.get(key), value)
        for key, value in requested_runtime.items()
        if frozen_runtime.get(key) != value
    }
    if mismatched_runtime:
        raise ValueError(
            f"requested runtime differs from the frozen profile: {mismatched_runtime}"
        )
    cases = [
        case
        for case in load_benchmark_jsonl(args.dataset)
        if case.split == protocol["dataset"]["formal_split"]
    ]
    if len(cases) != 12:
        raise ValueError(f"expected 12 formal test cases, found {len(cases)}")
    dataset_fingerprint = benchmark_fingerprint(
        load_benchmark_jsonl(args.dataset)
    )
    if dataset_fingerprint != protocol["dataset"]["expected_fingerprint_sha256"]:
        raise ValueError("dataset fingerprint differs from the E21 protocol")
    model_sha256 = args.model_sha256 or sha256(args.model)
    if args.model_sha256 and sha256(args.model) != args.model_sha256:
        raise ValueError("model SHA-256 does not match the requested profile")
    if model_sha256 != model_profile.get("model_sha256"):
        raise ValueError("model SHA-256 differs from the frozen profile")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.jsonl"
    traces_path = output_dir / "traces.json"
    manifest_path = output_dir / "run_manifest.json"
    summary_path = output_dir / "formal_summary.json"
    expected_manifest = {
        "schema_version": "1.0",
        "run_id": output_dir.name,
        "split": protocol["dataset"]["formal_split"],
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "model_id": args.model_id,
        "model_sha256": model_sha256,
        "protocol_sha256": sha256(args.protocol),
        "preflight_sha256": sha256(args.preflight),
        "external_processing": False,
        "runtime": {
            "n_ctx": args.n_ctx,
            "n_batch": args.n_batch,
            "n_gpu_layers": args.n_gpu_layers,
            "n_threads": args.n_threads,
            "max_output_tokens": args.max_output_tokens,
            "temperature": protocol["model_profile"]["temperature"],
            "seed": 21021,
        },
        "methods": list(METHODS),
        "expected_runs": len(cases) * len(METHODS),
    }
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected_manifest.items():
            if existing_manifest.get(key) != value:
                raise ValueError(f"existing run manifest differs at {key}")
    scores = load_scores(scores_path)
    traces = (
        json.loads(traces_path.read_text(encoding="utf-8"))
        if traces_path.is_file()
        else {}
    )
    budget = SearchBudget.model_validate(protocol["budget"])
    adapter = None

    for method in EXECUTION_ORDER:
        for case in cases:
            key = (case.case_id, method)
            if key in scores:
                continue
            started = time.perf_counter()
            prepared = prepare_e21_method(case, method, budget)
            response = None
            try:
                if prepared.decision is not None:
                    decision = prepared.decision
                else:
                    if adapter is None:
                        adapter = LlamaCppAdapter(
                            args.model,
                            model_id=args.model_id,
                            model_sha256=model_sha256,
                            n_ctx=args.n_ctx,
                            n_batch=args.n_batch,
                            n_gpu_layers=args.n_gpu_layers,
                            n_threads=args.n_threads,
                            max_output_tokens=args.max_output_tokens,
                            temperature=protocol["model_profile"]["temperature"],
                            seed=21021,
                        )
                    if prepared.llm_request is None:
                        raise ValueError("method requires a missing LLM request")
                    response = adapter.complete(prepared.llm_request)
                    decision = validate_generated_decision(
                        response.content,
                        case.to_fixture().case,
                        set(prepared.retrieved_evidence_ids),
                        expected_method=method,
                    )
                wall_seconds = time.perf_counter() - started
                score = score_decision(
                    case,
                    method,
                    decision,
                    model_calls=response.provider_calls if response else 0,
                    input_tokens=(response.input_tokens or 0) if response else 0,
                    output_tokens=(response.output_tokens or 0) if response else 0,
                    wall_seconds=wall_seconds,
                )
                if response:
                    score = score.model_copy(
                        update={
                            "returned_model": response.model_id,
                            "system_fingerprint": response.system_fingerprint,
                        }
                    )
                trace = {
                    "status": "completed",
                    "decision": decision.model_dump(mode="json"),
                    "retrieved_evidence_ids": prepared.retrieved_evidence_ids,
                    "response_metadata": (
                        response.model_dump(mode="json")
                        | {"content": "stored_in_decision"}
                        if response
                        else None
                    ),
                }
            except Exception as error:  # noqa: BLE001 - checkpoint every dev failure
                wall_seconds = time.perf_counter() - started
                failure_status = (
                    "parser_error"
                    if isinstance(
                        error,
                        (LLMAdapterError, ValidationError, ValueError),
                    )
                    else "internal_error"
                )
                score = score_failed_run(
                    case,
                    method,
                    failure_status,
                    failure_detail=f"{type(error).__name__}: {str(error)[:800]}",
                    model_calls=(
                        response.provider_calls
                        if response
                        else getattr(error, "provider_calls", 0)
                    ),
                    input_tokens=(
                        response.input_tokens or 0
                        if response
                        else getattr(error, "input_tokens", 0)
                    ),
                    output_tokens=(
                        response.output_tokens or 0
                        if response
                        else getattr(error, "output_tokens", 0)
                    ),
                    wall_seconds=wall_seconds,
                )
                trace = {
                    "status": "failed",
                    "failure_type": type(error).__name__,
                    "failure_detail": str(error)[:4000],
                    "retrieved_evidence_ids": prepared.retrieved_evidence_ids,
                    "response_metadata": (
                        response.model_dump(mode="json") if response else None
                    ),
                }
            scores[key] = score
            traces[f"{case.case_id}:{method}"] = trace
            write_scores(scores_path, scores)
            atomic_json(traces_path, traces)
            manifest = expected_manifest | {
                "status": "running",
                "updated_at": datetime.now().astimezone().isoformat(),
                "completed_runs": len(scores),
                "run_status_counts": dict(
                    sorted(Counter(item.run_status for item in scores.values()).items())
                ),
            }
            atomic_json(manifest_path, manifest)
            print(
                f"{len(scores)}/{expected_manifest['expected_runs']} "
                f"{case.case_id} {method}: {score.run_status} "
                f"success={score.grounded_task_success}",
                flush=True,
            )

    score_list = list(scores.values())
    strongest = protocol["comparison_plan"][
        "selected_strongest_baseline_method_id"
    ]
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "reporting_boundary": (
            "Formal test matrix executed once under the frozen E21 protocol; "
            "analysis determines whether performance claims are operationally valid."
        ),
        "split": protocol["dataset"]["formal_split"],
        "case_count": len(cases),
        "score_count": len(score_list),
        "model_id": args.model_id,
        "model_sha256": model_sha256,
        "external_processing": False,
        "frozen_strongest_baseline_method_id": strongest,
        "aggregates": [
            item.model_dump(mode="json") for item in aggregate_scores(score_list)
        ],
    }
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        expected_manifest
        | {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "completed_runs": len(scores),
            "run_status_counts": dict(
                sorted(Counter(item.run_status for item in scores.values()).items())
            ),
            "scores_sha256": sha256(scores_path),
            "traces_sha256": sha256(traces_path),
            "summary_sha256": sha256(summary_path),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
