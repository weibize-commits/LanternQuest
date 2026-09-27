import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from scienceworld import ScienceWorldEnv


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trajectory_fingerprint(payload: dict[str, Any]) -> str:
    stable = {
        "protocol_id": payload["protocol_id"],
        "task_id": payload["task_id"],
        "task_name": payload["task_name"],
        "variation": payload["variation"],
        "task_description": payload["task_description"],
        "gold_actions": payload["gold_actions"],
        "transition_hashes": [
            {
                "action": transition["action"],
                "observation_before_sha256": transition[
                    "observation_before_sha256"
                ],
                "observation_after_sha256": transition[
                    "observation_after_sha256"
                ],
                "look_before_sha256": transition["look_before_sha256"],
                "look_after_sha256": transition["look_after_sha256"],
                "inventory_before_sha256": transition[
                    "inventory_before_sha256"
                ],
                "inventory_after_sha256": transition[
                    "inventory_after_sha256"
                ],
                "reward": transition["reward"],
                "score": transition["score"],
                "completed": transition["completed"],
            }
            for transition in payload["transitions"]
        ],
    }
    encoded = json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def export_trajectory(
    env: ScienceWorldEnv,
    *,
    experiment_id: str,
    protocol_id: str,
    task_id: str,
    task_name: str,
    variation: int,
) -> dict[str, Any]:
    started = perf_counter()
    env.load(task_name, variationIdx=variation, generateGoldPath=True)
    observation, initial_info = env.reset()
    look = str(initial_info.get("look", ""))
    inventory = str(initial_info.get("inv", ""))
    gold_actions = list(env.get_gold_action_sequence())
    transitions: list[dict[str, Any]] = []
    completed = False
    final_score = 0

    for step_index, action in enumerate(gold_actions, start=1):
        next_observation, reward, completed, info = env.step(action)
        next_look = str(info.get("look", ""))
        next_inventory = str(info.get("inv", ""))
        final_score = int(info.get("score", final_score + reward))
        transitions.append(
            {
                "step_index": step_index,
                "action": action,
                "observation_before": observation,
                "observation_before_sha256": _sha256_text(observation),
                "observation_after": next_observation,
                "observation_after_sha256": _sha256_text(next_observation),
                "look_before": look,
                "look_before_sha256": _sha256_text(look),
                "look_after": next_look,
                "look_after_sha256": _sha256_text(next_look),
                "inventory_before": inventory,
                "inventory_before_sha256": _sha256_text(inventory),
                "inventory_after": next_inventory,
                "inventory_after_sha256": _sha256_text(next_inventory),
                "observation_changed": next_observation != observation,
                "look_changed": next_look != look,
                "inventory_changed": next_inventory != inventory,
                "reward": int(reward),
                "score": final_score,
                "completed": bool(completed),
            }
        )
        observation = next_observation
        look = next_look
        inventory = next_inventory
        if completed:
            break

    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "experiment_id": experiment_id,
        "protocol_id": protocol_id,
        "reporting_boundary": "train_split_gold_trajectory_not_test_result",
        "source_split": "train",
        "task_id": task_id,
        "task_name": task_name,
        "variation": variation,
        "task_description": initial_info["taskDesc"],
        "gold_actions": gold_actions,
        "transitions": transitions,
        "validation": {
            "completed": bool(completed),
            "final_score": final_score,
            "gold_action_count": len(gold_actions),
            "executed_action_count": len(transitions),
            "elapsed_seconds": round(perf_counter() - started, 3),
        },
    }
    payload["trajectory_fingerprint"] = _trajectory_fingerprint(payload)
    payload["skill_fingerprint"] = payload["trajectory_fingerprint"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the locked E12 train trajectories with auditable state transitions."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/scienceworld_protocol_ensr_v2_draft.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/train_trajectories_ensr_v2"
        ),
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "scienceworld_ensr_v2":
        raise ValueError("This exporter only accepts the E12 ENSR-v2 protocol")

    jobs = [
        (selection, int(variation))
        for selection in protocol["selected_variations"]
        for variation in selection["train_variations"]
    ]
    env = ScienceWorldEnv(
        envStepLimit=int(protocol["benchmark"]["max_environment_steps"])
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for index, (selection, variation) in enumerate(jobs, start=1):
            task_id = str(selection["task_id"])
            output_path = (
                args.output_dir
                / f"{task_id.replace('-', '_')}_v{variation:04d}.json"
            )
            if output_path.is_file():
                record = json.loads(output_path.read_text(encoding="utf-8"))
                if (
                    record.get("protocol_id") != protocol["protocol_id"]
                    or int(record.get("variation", -1)) != variation
                ):
                    raise ValueError(f"Cached record does not match job: {output_path}")
                records.append(record)
                print(f"[{index}/{len(jobs)}] cached {task_id} v{variation}", flush=True)
                continue
            try:
                record = export_trajectory(
                    env,
                    experiment_id=str(protocol["experiment_id"]),
                    protocol_id=str(protocol["protocol_id"]),
                    task_id=task_id,
                    task_name=str(selection["task_name"]),
                    variation=variation,
                )
                _write_atomic(output_path, record)
                records.append(record)
                print(
                    f"[{index}/{len(jobs)}] passed {task_id} v{variation} "
                    f"score={record['validation']['final_score']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - preserve every export failure
                failure = {
                    "task_id": task_id,
                    "task_name": selection["task_name"],
                    "variation": variation,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                _write_atomic(output_path.with_suffix(".failed.json"), failure)
                print(
                    f"[{index}/{len(jobs)}] failed {task_id} v{variation}: {exc}",
                    flush=True,
                )
    finally:
        env.close()

    manifest = {
        "schema_version": "2.0",
        "experiment_id": protocol["experiment_id"],
        "protocol_id": protocol["protocol_id"],
        "reporting_boundary": "train_split_gold_trajectory_corpus",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_split": "train",
        "expected_trajectory_count": len(jobs),
        "exported_trajectory_count": len(records),
        "failure_count": len(failures),
        "all_completed": len(records) == len(jobs)
        and all(record["validation"]["completed"] for record in records),
        "trajectories": [
            {
                "task_id": record["task_id"],
                "task_name": record["task_name"],
                "variation": record["variation"],
                "trajectory_fingerprint": record["trajectory_fingerprint"],
            }
            for record in records
        ],
        "failures": failures,
    }
    _write_atomic(
        args.output_dir.parent / "train_trajectory_manifest_ensr_v2.json",
        manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if failures or not manifest["all_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
