import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

from scienceworld import ScienceWorldEnv

from lanternquest.config import Settings
from lanternquest.openai_adapter import OpenAIResponsesAdapter
from lanternquest.scienceworld_experiment import (
    EpisodeBudget,
    ScienceWorldEpisodeResult,
    ScienceWorldExperimentHarness,
    ScienceWorldMethod,
    load_scienceworld_skills,
    summarize_episode_results,
    write_episode_result,
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    paths = [Path("pyproject.toml")]
    paths.extend(sorted(Path("src/lanternquest").glob("*.py")))
    paths.append(Path("scripts/run_scienceworld_experiment.py"))
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_model_profile(path: Path, profile_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        profile for profile in payload["profiles"] if profile["profile_id"] == profile_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Model profile {profile_id!r} was not found exactly once")
    return matches[0]


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen ScienceWorld sanity checks or model comparisons."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/scienceworld_protocol_v0.json"),
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/train_skills"),
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--methods", default="random_sanity,gold_oracle", help="Comma-separated method IDs."
    )
    parser.add_argument(
        "--seeds", default="", help="Comma-separated seeds; default uses the protocol."
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required acknowledgement before touching frozen test variations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/runs"),
    )
    parser.add_argument(
        "--model-profiles",
        type=Path,
        default=Path("configs/formal_model_profiles_v0.json"),
    )
    parser.add_argument(
        "--model-profile",
        default="",
        help="Frozen model profile ID from --model-profiles.",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("--allow-test is required for the frozen test split")

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    known_methods = set(get_args(ScienceWorldMethod))
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown_methods = set(methods) - known_methods
    if unknown_methods:
        parser.error(f"Unknown methods: {sorted(unknown_methods)}")
    seeds = (
        [int(seed) for seed in args.seeds.split(",") if seed.strip()]
        if args.seeds
        else list(protocol["runs"]["formal_seeds"])
    )
    selections = list(protocol["selected_variations"])
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be positive")
        selections = selections[: args.max_cases]

    llm_methods = set(methods) - {"random_sanity", "gold_oracle"}
    adapter = None
    active_profile: dict[str, object] | None = None
    model_profiles_path: Path | None = None
    settings = Settings()
    if llm_methods:
        if args.model_profile:
            model_profiles_path = args.model_profiles.resolve()
            try:
                active_profile = _load_model_profile(
                    model_profiles_path, args.model_profile
                )
            except (KeyError, ValueError) as exc:
                parser.error(str(exc))
            key_setting = str(active_profile["api_key_setting"])
            allowed_key_settings = {
                "bit_api_key": settings.bit_api_key,
                "deepseek_api_key": settings.deepseek_api_key,
            }
            if key_setting not in allowed_key_settings:
                parser.error(f"Unsupported api_key_setting: {key_setting}")
            secret = allowed_key_settings[key_setting]
            api_key = secret.get_secret_value().strip() if secret is not None else ""
            if not api_key:
                parser.error(f"Missing local secret for {key_setting}")
            adapter = OpenAIResponsesAdapter(
                model_id=str(active_profile["model_id"]),
                api_key=api_key,
                base_url=str(active_profile["base_url"]),
                max_output_tokens=int(active_profile["max_output_tokens"]),
                temperature=float(active_profile["temperature"]),
                reasoning_effort=(
                    str(active_profile["reasoning_effort"])
                    if active_profile.get("reasoning_effort") is not None
                    else None
                ),
                structured_output_mode=str(
                    active_profile["structured_output_mode"]
                ),
                profile_id=str(active_profile["profile_id"]),
            )
        else:
            api_key = (
                settings.openai_api_key.get_secret_value().strip()
                if settings.openai_api_key is not None
                else ""
            )
            if not settings.llm_model or not api_key:
                parser.error(
                    "LLM methods require LANTERNQUEST_LLM_MODEL and "
                    "LANTERNQUEST_OPENAI_API_KEY in the local .env"
                )
            adapter = OpenAIResponsesAdapter(
                model_id=settings.llm_model,
                api_key=api_key,
                base_url=settings.openai_base_url,
                max_output_tokens=settings.llm_max_output_tokens,
                temperature=settings.llm_temperature,
                reasoning_effort=settings.llm_reasoning_effort,
                structured_output_mode=settings.llm_structured_output_mode,
            )

    skills = load_scienceworld_skills(args.skills_dir.resolve())
    harness = ScienceWorldExperimentHarness(
        skills,
        adapter=adapter,
        experiment_id=str(protocol["experiment_id"]),
    )
    budget = EpisodeBudget(
        max_environment_steps=protocol["benchmark"]["max_environment_steps"],
        max_model_calls=protocol["runs"]["max_model_calls_per_episode"],
        max_retrieval_calls=protocol["runs"]["max_retrieval_calls_per_episode"],
        initial_retrieval_k=protocol["runs"]["initial_retrieval_k"],
        obligation_retrieval_k=protocol["runs"]["obligation_retrieval_k"],
        automatic_obligation_policy=protocol["runs"].get(
            "automatic_obligation_policy", "disabled"
        ),
        max_automatic_obligations=protocol["runs"].get(
            "max_automatic_obligations_per_episode", 3
        ),
    )
    run_root = args.output_root / protocol["protocol_id"]
    if active_profile is not None:
        run_root = run_root / str(active_profile["profile_id"])
    run_root = run_root / args.split
    results: list[ScienceWorldEpisodeResult] = []
    env = ScienceWorldEnv(envStepLimit=budget.max_environment_steps)
    try:
        total = sum(1 if method == "gold_oracle" else len(seeds) for method in methods)
        total *= len(selections)
        run_index = 0
        for selection in selections:
            for method in methods:
                method_seeds = [seeds[0]] if method == "gold_oracle" else seeds
                for seed in method_seeds:
                    run_index += 1
                    result_path = (
                        run_root
                        / method
                        / (
                            f"{selection['task_id'].replace('-', '_')}_"
                            f"v{selection[args.split]}_seed{seed}.json"
                        )
                    )
                    result = None
                    if result_path.is_file():
                        try:
                            result = ScienceWorldEpisodeResult.model_validate_json(
                                result_path.read_text(encoding="utf-8")
                            )
                        except ValueError:
                            print(
                                f"[{run_index}/{total}] stale cache {method} "
                                f"{selection['task_id']}; recomputing",
                                flush=True,
                            )
                    if result is not None:
                        print(
                            f"[{run_index}/{total}] cached {method} "
                            f"{selection['task_id']} status={result.status}",
                            flush=True,
                        )
                    else:
                        result = harness.run_episode(
                            env,
                            method=method,
                            task_id=selection["task_id"],
                            task_name=selection["task_name"],
                            split=args.split,
                            variation=selection[args.split],
                            seed=seed,
                            budget=budget,
                            simplifications=protocol["benchmark"]["simplifications"],
                        )
                        write_episode_result(result, result_path)
                        print(
                            f"[{run_index}/{total}] {method} {selection['task_id']} "
                            f"status={result.status} score={result.final_score}",
                            flush=True,
                        )
                    results.append(result)
    finally:
        env.close()

    summary = summarize_episode_results(results)
    summary.update(
        {
            "schema_version": "1.0",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": _file_hash(protocol_path),
            "implementation_sha256": _implementation_hash(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "selected_task_count": len(selections),
            "seeds": seeds,
            "model_id": adapter.model_id if adapter is not None else None,
            "model_profile_id": (
                str(active_profile["profile_id"])
                if active_profile is not None
                else None
            ),
            "model_profiles_sha256": (
                _file_hash(model_profiles_path)
                if model_profiles_path is not None
                else None
            ),
            "reporting_boundary": (
                "development_pilot_only"
                if args.split == "dev"
                else (
                    "sanity_or_oracle_only"
                    if set(methods) <= {"random_sanity", "gold_oracle"}
                    else "formal_machine_result"
                )
            ),
        }
    )
    summary_path = run_root / f"summary_{'_'.join(methods)}.json"
    _write_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
