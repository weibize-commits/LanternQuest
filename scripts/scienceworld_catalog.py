import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import scienceworld
from scienceworld import ScienceWorldEnv


def build_catalog() -> dict[str, object]:
    package_root = Path(scienceworld.__file__).resolve().parent
    task_metadata = json.loads(
        (package_root / "tasks.json").read_text(encoding="utf-8")
    )
    metadata_by_name = {item["task_name"]: item for item in task_metadata}

    env = ScienceWorldEnv(envStepLimit=100)
    tasks: list[dict[str, object]] = []
    try:
        for task_name in env.get_task_names():
            env.load(task_name, variationIdx=0, generateGoldPath=False)
            train = env.get_variations_train()
            dev = env.get_variations_dev()
            test = env.get_variations_test()
            max_variations = env.get_max_variations(task_name)
            all_variations = train + dev + test
            metadata = metadata_by_name[task_name]
            tasks.append(
                {
                    "task_id": metadata["task_id"],
                    "task_name": task_name,
                    "topic": metadata["topic"],
                    "description": metadata["task"],
                    "max_variations": max_variations,
                    "splits": {"train": train, "dev": dev, "test": test},
                    "split_counts": {
                        "train": len(train),
                        "dev": len(dev),
                        "test": len(test),
                    },
                    "split_validation": {
                        "disjoint": len(all_variations) == len(set(all_variations)),
                        "complete": set(all_variations) == set(range(max_variations)),
                    },
                }
            )
    finally:
        env.close()

    split_totals = {
        split: sum(task["split_counts"][split] for task in tasks)
        for split in ("train", "dev", "test")
    }
    return {
        "schema_version": "1.0",
        "experiment_id": "E10",
        "reporting_boundary": "setup_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scienceworld_version": version("scienceworld"),
        "task_count": len(tasks),
        "split_totals": split_totals,
        "all_splits_disjoint": all(
            task["split_validation"]["disjoint"] for task in tasks
        ),
        "all_splits_complete": all(
            task["split_validation"]["complete"] for task in tasks
        ),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the official ScienceWorld task and variation splits."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/task_catalog.json"),
    )
    args = parser.parse_args()
    payload = build_catalog()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(args.output)
    print(
        json.dumps(
            {
                "task_count": payload["task_count"],
                "split_totals": payload["split_totals"],
                "all_splits_disjoint": payload["all_splits_disjoint"],
                "all_splits_complete": payload["all_splits_complete"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
