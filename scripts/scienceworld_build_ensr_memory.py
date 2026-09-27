import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.scienceworld_ensr_components import (
        TemporalFact,
        parse_observation_facts,
    )
except ModuleNotFoundError:  # Direct script execution uses the scripts directory.
    from scienceworld_ensr_components import (  # type: ignore[no-redef]
        TemporalFact,
        parse_observation_facts,
    )

_EFFECT_RELATIONS = {
    "located_in",
    "visible_in",
    "has_state",
    "in_inventory_of",
    "contains",
}


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _task_target(task_description: str) -> str | None:
    for pattern in (
        r"your task is to (?:boil|melt|freeze) (.+?)\.",
        r"your task is to change the state of matter of (.+?)\.",
        r"your task is to measure the temperature of (.+?), which",
        r"your task is to measure the melting point of (.+?), which",
        r"your task is to determine if (.+?) is electrically conductive",
        r"your task is to turn on (.+?)(?:\.| by )",
    ):
        match = re.search(pattern, task_description, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _abstractor(task_description: str, actions: list[str]):
    target = _task_target(task_description)
    focus_object = next(
        (
            action[len("focus on ") :].strip()
            for action in actions
            if action.casefold().startswith("focus on ")
        ),
        None,
    )
    replacements: dict[str, str] = {}
    if target:
        replacements[target.casefold()] = "<TASK_TARGET>"
    if focus_object and focus_object.casefold() not in replacements:
        replacements[focus_object.casefold()] = "<FOCUS_OBJECT>"
    if not replacements:
        return lambda text: text
    pattern = re.compile(
        "|".join(re.escape(value) for value in sorted(replacements, key=len, reverse=True)),
        flags=re.IGNORECASE,
    )
    return lambda text: pattern.sub(
        lambda match: replacements[match.group(0).casefold()], text
    )


def _fact_payload(fact: TemporalFact, abstract) -> dict[str, Any]:
    return {
        "subject": abstract(fact.subject),
        "relation": fact.relation,
        "object": abstract(fact.object),
        "polarity": fact.polarity,
        "provenance": fact.provenance,
        "confidence": fact.confidence,
    }


def _negative_fact_payload(fact: TemporalFact, abstract) -> dict[str, Any]:
    payload = _fact_payload(fact, abstract)
    payload["polarity"] = False
    return payload


def _state_text(transition: dict[str, Any], side: str) -> str:
    return "\n".join(
        value
        for value in (
            str(transition.get(f"look_{side}", "")),
            str(transition.get(f"inventory_{side}", "")),
        )
        if value
    )


def _relevant_preconditions(
    facts: list[TemporalFact], action: str
) -> list[TemporalFact]:
    action_tokens = set(re.findall(r"[a-z0-9]+", action.casefold()))
    selected = []
    for fact in facts:
        fact_tokens = set(
            re.findall(r"[a-z0-9]+", f"{fact.subject} {fact.object}".casefold())
        )
        if fact.relation == "located_in" or action_tokens & fact_tokens:
            selected.append(fact)
    return selected[:12]


def build_fragments(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("source_split") != "train":
        raise ValueError("ENSR memory may only be built from train-split trajectories")
    abstract = _abstractor(record["task_description"], record["gold_actions"])
    fragments: list[dict[str, Any]] = []

    for transition in record["transitions"]:
        step = int(transition["step_index"])
        before = parse_observation_facts(
            _state_text(transition, "before"),
            step=max(0, step - 1),
            provenance="train_gold_transition",
            support_action=transition["action"],
        )
        after = parse_observation_facts(
            _state_text(transition, "after"),
            step=step,
            provenance="train_gold_transition",
            support_action=transition["action"],
        )
        before_by_key = {fact.key: fact for fact in before}
        after_by_key = {fact.key: fact for fact in after}
        added = [
            fact
            for key, fact in after_by_key.items()
            if key not in before_by_key and fact.relation in _EFFECT_RELATIONS
        ]
        removed = [
            fact
            for key, fact in before_by_key.items()
            if key not in after_by_key and fact.relation in _EFFECT_RELATIONS
        ]
        action = str(transition["action"])
        stable = "|".join(
            (
                str(record["trajectory_fingerprint"]),
                str(step),
                action,
                str(transition["observation_before_sha256"]),
                str(transition["observation_after_sha256"]),
            )
        )
        fragments.append(
            {
                "fragment_id": hashlib.sha256(stable.encode()).hexdigest(),
                "task_id": record["task_id"],
                "task_name": record["task_name"],
                "variation": record["variation"],
                "step_index": step,
                "action_template": abstract(action),
                "operator": action.casefold().split(maxsplit=1)[0],
                "preconditions": [
                    _fact_payload(fact, abstract)
                    for fact in _relevant_preconditions(before, action)
                ],
                "effects": [
                    *[_fact_payload(fact, abstract) for fact in added],
                    *[_negative_fact_payload(fact, abstract) for fact in removed],
                ],
                "outcome": {
                    "reward": transition["reward"],
                    "score": transition["score"],
                    "completed": transition["completed"],
                },
                "evidence": {
                    "trajectory_fingerprint": record["trajectory_fingerprint"],
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
                },
            }
        )
    return fragments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build provenance-preserving ENSR memory from E12 train trajectories."
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/train_trajectories_ensr_v2"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_memory_v2.json"),
    )
    args = parser.parse_args()
    manifest_path = (
        args.trajectory_dir.parent / "train_trajectory_manifest_ensr_v2.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("all_completed"):
        raise ValueError("Trajectory manifest is incomplete")

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.trajectory_dir.glob("*_v*.json"))
        if not path.name.endswith(".failed.json")
    ]
    if len(records) != int(manifest["expected_trajectory_count"]):
        raise ValueError("Trajectory count does not match the manifest")
    fragments = [fragment for record in records for fragment in build_fragments(record)]
    payload = {
        "schema_version": "2.0",
        "experiment_id": manifest["experiment_id"],
        "protocol_id": manifest["protocol_id"],
        "reporting_boundary": "train_split_derived_memory_not_test_result",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.resolve()),
        "trajectory_count": len(records),
        "fragment_count": len(fragments),
        "fragments": fragments,
    }
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "trajectory_count": len(records),
                "fragment_count": len(fragments),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
