import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_scienceworld_ensr_results import _load_rows, analyze
except ModuleNotFoundError:  # Direct script execution uses the scripts directory.
    from analyze_scienceworld_ensr_results import _load_rows, analyze

PROFILES = (
    "bit_qwen3_235b",
    "bit_ceep_70b",
    "bit_qwen3_32b",
    "bit_qwen3_8b",
    "bit_qwen3_0_6b",
)


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def analyze_models(
    *,
    protocol: dict[str, Any],
    run_root: Path,
    anchor_root: Path,
    seed: int,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for profile in PROFILES:
        source = (
            anchor_root
            if profile == "bit_qwen3_235b"
            else run_root / profile / "test"
        )
        rows, parse_failures = _load_rows(source)
        selected_rows = [row for row in rows if int(row.get("seed", -1)) == seed]
        result = analyze(
            rows=selected_rows,
            protocol=protocol,
            split="test",
            seeds=[seed],
            parse_failures=parse_failures,
        )
        result["profile_id"] = profile
        result["source_root"] = str(source.resolve())
        result["anchor_reused"] = profile == "bit_qwen3_235b"
        models[profile] = result

    score_positive = [
        profile
        for profile, result in models.items()
        if result["ensr_minus_baseline"]["clipped_score"]["point_estimate"] > 0
    ]
    success_nondecreasing = [
        profile
        for profile, result in models.items()
        if result["ensr_minus_baseline"]["task_success_rate"]["point_estimate"]
        >= 0
    ]
    both = sorted(set(score_positive) & set(success_nondecreasing))
    complete = all(result["complete_matrix"] for result in models.values())
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": protocol["experiment_id"],
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "profile_count": len(PROFILES),
        "expected_episode_count": len(PROFILES) * 90,
        "complete_all_models": complete,
        "models": models,
        "robustness_summary": {
            "score_positive_profiles": score_positive,
            "success_nondecreasing_profiles": success_nondecreasing,
            "both_criteria_profiles": both,
            "both_criteria_count": len(both),
            "required_count": 4,
            "cross_model_robustness_pass": complete and len(both) >= 4,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the five-model E14 ENSR robustness matrix."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/scienceworld_protocol_ensr_cross_model_v1_frozen.json"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_cross_model_v1_runs/"
            "scienceworld_ensr_v3"
        ),
    )
    parser.add_argument(
        "--anchor-root",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_v3_runs_formal/"
            "scienceworld_ensr_v3/bit_qwen3_235b/test"
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_cross_model_v1_audit.json"
        ),
    )
    args = parser.parse_args()
    payload = analyze_models(
        protocol=json.loads(args.protocol.read_text(encoding="utf-8")),
        run_root=args.run_root,
        anchor_root=args.anchor_root,
        seed=args.seed,
    )
    _write_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
