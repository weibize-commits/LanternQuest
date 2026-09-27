import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scienceworld import ScienceWorldEnv

from lanternquest.config import Settings
from lanternquest.model_trace import ModelTraceAdapter
from lanternquest.openai_adapter import OpenAIResponsesAdapter
from lanternquest.scienceworld_experiment import (
    EpisodeBudget,
    ScienceWorldEpisodeResult,
    ScienceWorldExperimentHarness,
    load_scienceworld_skills,
    write_episode_result,
)
from lanternquest.verified_repair import symbolic_state_sha256

try:
    from scripts.scienceworld_ensr_harness import (
        ENSRBudget,
        ENSREpisodeResult,
        ENSRHarness,
        ENSRMemory,
        write_ensr_episode_result,
    )
except ModuleNotFoundError:  # Direct script execution uses the scripts directory.
    from scienceworld_ensr_harness import (  # type: ignore[no-redef]
        ENSRBudget,
        ENSREpisodeResult,
        ENSRHarness,
        ENSRMemory,
        write_ensr_episode_result,
    )

METHODS = {"b4_sequential_planner", "iper_rag", "ensr_v2"}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    paths = [Path("pyproject.toml")]
    paths.extend(sorted(Path("src/lanternquest").glob("*.py")))
    paths.extend(
        Path(name)
        for name in (
            "src/lanternquest/evidence_scheduler.py",
            "scripts/run_scienceworld_ensr_experiment.py",
            "scripts/scienceworld_ensr_components.py",
            "scripts/scienceworld_ensr_harness.py",
            "scripts/scienceworld_build_ensr_memory.py",
            "scripts/scienceworld_export_ensr_trajectories.py",
        )
    )
    for path in paths:
        digest.update(path.as_posix().encode())
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


def _selected_cases(
    protocol: dict[str, object], split: str
) -> list[tuple[dict[str, object], int]]:
    cases: list[tuple[dict[str, object], int]] = []
    for selection in protocol["selected_variations"]:
        variations = (
            selection["dev_variations"] if split == "dev" else [selection["test"]]
        )
        cases.extend((selection, int(variation)) for variation in variations)
    return cases


def _summarize(results: list[ScienceWorldEpisodeResult | ENSREpisodeResult]) -> dict:
    methods: dict[str, object] = {}
    for method in sorted({result.method for result in results}):
        rows = [result for result in results if result.method == method]
        methods[method] = {
            "episodes": len(rows),
            "status_counts": dict(sorted(Counter(row.status for row in rows).items())),
            "task_successes": sum(row.task_success for row in rows),
            "task_success_rate": sum(row.task_success for row in rows) / len(rows),
            "mean_final_score": sum(row.final_score for row in rows) / len(rows),
            "mean_clipped_score": sum(max(0, row.final_score) for row in rows)
            / len(rows),
            "environment_steps": sum(row.environment_steps for row in rows),
            "model_calls": sum(row.model_calls for row in rows),
            "retrieval_calls": sum(row.retrieval_calls for row in rows),
            "input_tokens": sum(row.input_tokens for row in rows),
            "output_tokens": sum(row.output_tokens for row in rows),
            "wall_seconds": sum(row.wall_seconds for row in rows),
        }
        if method == "ensr_v2":
            ensr_rows = [row for row in rows if isinstance(row, ENSREpisodeResult)]
            methods[method]["planner_calls"] = sum(
                row.planner_calls for row in ensr_rows
            )
            methods[method]["obligation_count"] = sum(
                int(row.mechanism_metrics["obligation_count"]) for row in ensr_rows
            )
            methods[method]["local_replan_recovery_count"] = sum(
                int(row.mechanism_metrics["local_replan_recovery_count"])
                for row in ensr_rows
            )
    return {"episodes": len(results), "methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the E12 ENSR development or locked formal matrix."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/scienceworld_protocol_ensr_v2_draft.json"),
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/train_trajectories_ensr_v2"
        ),
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_memory_v2.json"),
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--methods",
        default="b4_sequential_planner,iper_rag,ensr_v2",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated override; dev defaults to the first formal seed.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--case-keys",
        default="",
        help="Optional comma-separated development cases such as 2-2:226,3-1:12.",
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_v2_runs"),
    )
    parser.add_argument(
        "--model-profiles",
        type=Path,
        default=Path("configs/formal_model_profiles_v0.json"),
    )
    parser.add_argument("--model-profile", default="bit_qwen3_235b")
    parser.add_argument(
        "--ablation-arm",
        default="",
        help="Optional arm_id from an E16 mechanism-ablation protocol.",
    )
    parser.add_argument(
        "--model-trace-mode",
        choices=("off", "record", "replay_prefix", "replay_until_divergence"),
        default="off",
    )
    parser.add_argument(
        "--model-trace-root",
        type=Path,
        default=None,
        help="Output root for per-episode validated model-call traces.",
    )
    parser.add_argument(
        "--model-trace-source-root",
        type=Path,
        default=None,
        help="D0 trace root consumed by replay_prefix mode.",
    )
    parser.add_argument(
        "--action-replay-source-root",
        type=Path,
        default=None,
        help=(
            "D0 result output root whose exact environment-action trajectory is "
            "replayed before a D4 repair."
        ),
    )
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_id = str(protocol.get("protocol_id", ""))
    if protocol_id not in {
        "scienceworld_ensr_v2",
        "scienceworld_ensr_v3",
        "scienceworld_ensr_mechanism_ablation_v1",
        "scienceworld_ensr_evidence_scheduler_dev_v1",
        "scienceworld_ensr_evidence_anchor_dev_v1",
        "scienceworld_ensr_evidence_anchor_cross_model_v1",
        "scienceworld_ensr_evidence_anchor_ablation_v1",
    }:
        parser.error("The ENSR runner requires an ENSR-v2 or ENSR-v3 protocol")
    if args.split == "test":
        if not args.allow_test:
            parser.error("--allow-test is required for the frozen test split")
        if protocol.get("status") != "formal_frozen_test_unlocked":
            parser.error(
                "Formal test is locked until dev gates pass and the protocol status is frozen"
            )

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        parser.error(f"Unknown methods: {sorted(unknown)}")
    if args.model_trace_mode != "off":
        if methods != ["ensr_v2"]:
            parser.error("Model tracing requires --methods ensr_v2")
        if args.model_trace_root is None:
            parser.error("Model tracing requires --model-trace-root")
        if (
            args.model_trace_mode in {"replay_prefix", "replay_until_divergence"}
            and args.model_trace_source_root is None
        ):
            parser.error("replay mode requires --model-trace-source-root")
    if (
        args.action_replay_source_root is not None
        and args.model_trace_mode != "replay_prefix"
    ):
        parser.error("action replay requires --model-trace-mode replay_prefix")
    ablation_settings: dict[str, object] = {
        "enable_hierarchical_subgoals": True,
        "enable_symbolic_transition_verifier": True,
        "enable_obligation_conditioned_retrieval": True,
        "enable_evidence_debt_scheduler": bool(
            protocol.get("evidence_debt_scheduler", {}).get("enabled", False)
        ),
        "evidence_debt_retrieval_threshold": float(
            protocol.get("evidence_debt_scheduler", {}).get(
                "retrieval_threshold", 0.58
            )
        ),
        "evidence_debt_anchor_first_subgoal": bool(
            protocol.get("evidence_debt_scheduler", {}).get(
                "anchor_first_debt_per_subgoal", False
            )
        ),
        "fragment_memory_policy": "replace",
    }
    if args.ablation_arm:
        matches = [
            arm
            for arm in protocol.get("arms", [])
            if arm.get("arm_id") == args.ablation_arm
        ]
        if len(matches) != 1:
            parser.error(f"Unknown or duplicate ablation arm: {args.ablation_arm!r}")
        ablation_settings.update(matches[0].get("settings", {}))
        if methods != ["ensr_v2"]:
            parser.error("Mechanism comparison arms must run only ensr_v2")
    formal_seeds = [int(seed) for seed in protocol["runs"]["formal_seeds"]]
    seeds = (
        [int(seed) for seed in args.seeds.split(",") if seed.strip()]
        if args.seeds
        else formal_seeds
        if args.split == "test"
        else formal_seeds[:1]
    )
    cases = _selected_cases(protocol, args.split)
    if args.case_keys:
        requested: set[tuple[str, int]] = set()
        for raw_key in args.case_keys.split(","):
            try:
                task_id, variation_text = raw_key.strip().split(":", maxsplit=1)
                requested.add((task_id, int(variation_text)))
            except ValueError:
                parser.error(f"Invalid --case-keys item: {raw_key!r}")
        available = {
            (str(selection["task_id"]), variation)
            for selection, variation in cases
        }
        unknown_cases = requested - available
        if unknown_cases:
            parser.error(f"Unknown selected cases: {sorted(unknown_cases)}")
        cases = [
            (selection, variation)
            for selection, variation in cases
            if (str(selection["task_id"]), variation) in requested
        ]
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be positive")
        cases = cases[: args.max_cases]

    profile_path = args.model_profiles.resolve()
    try:
        profile = _load_model_profile(profile_path, args.model_profile)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    settings = Settings()
    secrets = {
        "openai_api_key": settings.openai_api_key,
        "bit_api_key": settings.bit_api_key,
        "deepseek_api_key": settings.deepseek_api_key,
        "kimi_api_key": settings.kimi_api_key,
        "glm_api_key": settings.glm_api_key,
    }
    key_setting = str(profile["api_key_setting"])
    if key_setting not in secrets:
        parser.error(f"Unsupported api_key_setting: {key_setting}")
    secret = secrets[key_setting]
    api_key = secret.get_secret_value().strip() if secret is not None else ""
    if not api_key:
        parser.error(f"Missing local secret for {key_setting}")
    base_adapter = OpenAIResponsesAdapter(
        model_id=str(profile["model_id"]),
        api_key=api_key,
        base_url=str(profile["base_url"]),
        max_output_tokens=int(profile["max_output_tokens"]),
        temperature=(
            float(profile["temperature"])
            if profile.get("temperature") is not None
            else None
        ),
        top_p=(float(profile["top_p"]) if profile.get("top_p") is not None else None),
        reasoning_effort=(
            str(profile["reasoning_effort"])
            if profile.get("reasoning_effort") is not None
            else None
        ),
        structured_output_mode=str(profile["structured_output_mode"]),
        profile_id=str(profile["profile_id"]),
        max_input_characters=(
            int(profile["max_input_characters"])
            if profile.get("max_input_characters") is not None
            else None
        ),
        max_retries=(
            int(profile["provider_retry_count"])
            if profile.get("provider_retry_count") is not None
            else None
        ),
        structured_output_retries=int(profile.get("structured_output_retries", 0)),
        chat_template_kwargs=(
            dict(profile["chat_template_kwargs"])
            if profile.get("chat_template_kwargs") is not None
            else None
        ),
        extra_body=(
            dict(profile["extra_body"])
            if profile.get("extra_body") is not None
            else None
        ),
        minimum_call_interval_seconds=float(
            profile.get("minimum_call_interval_seconds", 0.0)
        ),
        api_style=str(profile.get("api_style", "responses")),
        system_instruction_suffix=str(profile.get("system_instruction_suffix", "")),
    )
    adapter = (
        ModelTraceAdapter(base_adapter, mode=args.model_trace_mode)
        if args.model_trace_mode != "off"
        else base_adapter
    )

    skills_dir = args.skills_dir.resolve()
    memory_path = args.memory.resolve()
    skills = load_scienceworld_skills(skills_dir)
    baseline_harness = ScienceWorldExperimentHarness(
        skills,
        adapter=adapter,
        experiment_id=str(protocol["experiment_id"]),
    )
    ensr_harness = ENSRHarness(
        ENSRMemory.load(memory_path),
        adapter,
        experiment_id=str(protocol["experiment_id"]),
        enable_adaptive_slow_path=bool(
            protocol.get("adaptive_slow_path", {}).get("enabled", False)
        ),
        force_llm_after_local_replan=bool(
            protocol.get("adaptive_slow_path", {}).get(
                "force_llm_after_local_replan", True
            )
        ),
        max_continuation_plans_per_episode=int(
            protocol.get("adaptive_slow_path", {}).get(
                "max_continuation_plans_per_episode", 2
            )
        ),
        max_continuation_subgoals=int(
            protocol.get("adaptive_slow_path", {}).get(
                "max_continuation_subgoals", 4
            )
        ),
        enable_controlled_residual_repair=bool(
            protocol.get("controlled_residual_repair", {}).get("enabled", False)
        ),
        max_controlled_repair_requests=int(
            protocol.get("controlled_residual_repair", {}).get(
                "max_repair_requests_per_episode", 1
            )
        ),
        max_controlled_repair_actions=int(
            protocol.get("controlled_residual_repair", {}).get(
                "max_repair_actions", 3
            )
        ),
        enable_verified_graph_repair=bool(
            protocol.get("verified_transition_graph_repair", {}).get(
                "enabled", False
            )
        ),
        max_verified_graph_rank_requests=int(
            protocol.get("verified_transition_graph_repair", {}).get(
                "max_rank_requests_per_episode", 1
            )
        ),
        max_verified_graph_actions=int(
            protocol.get("verified_transition_graph_repair", {}).get(
                "max_repair_actions", 3
            )
        ),
        max_verified_graph_candidates=int(
            protocol.get("verified_transition_graph_repair", {}).get(
                "max_candidates", 8
            )
        ),
        enable_semantic_kg_control=bool(
            protocol.get("knowledge_graph_semantic_control", {}).get(
                "enabled", False
            )
        ),
        enable_hierarchical_subgoals=bool(
            ablation_settings["enable_hierarchical_subgoals"]
        ),
        enable_symbolic_transition_verifier=bool(
            ablation_settings["enable_symbolic_transition_verifier"]
        ),
        enable_obligation_conditioned_retrieval=bool(
            ablation_settings["enable_obligation_conditioned_retrieval"]
        ),
        enable_evidence_debt_scheduler=bool(
            ablation_settings["enable_evidence_debt_scheduler"]
        ),
        evidence_debt_retrieval_threshold=float(
            ablation_settings["evidence_debt_retrieval_threshold"]
        ),
        evidence_debt_anchor_first_subgoal=bool(
            ablation_settings["evidence_debt_anchor_first_subgoal"]
        ),
        fragment_memory_policy=str(ablation_settings["fragment_memory_policy"]),
    )
    baseline_budget = EpisodeBudget(
        max_environment_steps=int(protocol["benchmark"]["max_environment_steps"]),
        max_model_calls=int(protocol["runs"]["max_model_calls_per_episode"]),
        max_retrieval_calls=int(protocol["runs"]["max_retrieval_calls_per_episode"]),
        initial_retrieval_k=int(protocol["runs"]["initial_retrieval_k"]),
        obligation_retrieval_k=int(protocol["runs"]["obligation_retrieval_k"]),
        automatic_obligation_policy=str(
            protocol["runs"]["automatic_obligation_policy"]
        ),
        max_automatic_obligations=int(
            protocol["runs"]["max_automatic_obligations_per_episode"]
        ),
        candidate_action_limit=(
            int(protocol["runs"]["baseline_candidate_action_limit"])
            if protocol["runs"].get("baseline_candidate_action_limit") is not None
            else None
        ),
    )
    ensr_budget = ENSRBudget(
        max_environment_steps=int(protocol["benchmark"]["max_environment_steps"]),
        max_model_calls=int(protocol["runs"]["max_model_calls_per_episode"]),
        max_retrieval_calls=int(protocol["runs"]["max_retrieval_calls_per_episode"]),
        max_planner_calls=int(protocol["runs"]["max_planner_calls_per_episode"]),
        initial_retrieval_k=int(protocol["runs"]["initial_retrieval_k"]),
        obligation_retrieval_k=int(protocol["runs"]["obligation_retrieval_k"]),
        candidate_action_limit=int(protocol["runs"]["candidate_action_limit"]),
    )

    run_root = (
        args.output_root
        / str(protocol["protocol_id"])
        / str(profile["profile_id"])
        / args.split
    )
    if args.ablation_arm:
        run_root = run_root / args.ablation_arm
    results: list[ScienceWorldEpisodeResult | ENSREpisodeResult] = []
    env = ScienceWorldEnv(envStepLimit=baseline_budget.max_environment_steps)
    total = len(cases) * len(methods) * len(seeds)
    run_index = 0
    try:
        for selection, variation in cases:
            for method in methods:
                for seed in seeds:
                    run_index += 1
                    episode_filename = (
                        f"{str(selection['task_id']).replace('-', '_')}_"
                        f"v{variation}_seed{seed}.json"
                    )
                    result_path = (
                        run_root
                        / method
                        / episode_filename
                    )
                    trace_output_path = (
                        args.model_trace_root.resolve()
                        / str(profile["profile_id"])
                        / args.split
                        / episode_filename
                        if args.model_trace_root is not None
                        else None
                    )
                    trace_source_path = (
                        args.model_trace_source_root.resolve()
                        / str(profile["profile_id"])
                        / args.split
                        / episode_filename
                        if args.model_trace_source_root is not None
                        else None
                    )
                    action_replay_source_path = (
                        args.action_replay_source_root.resolve()
                        / str(protocol["protocol_id"])
                        / str(profile["profile_id"])
                        / args.split
                        / "ensr_v2"
                        / episode_filename
                        if args.action_replay_source_root is not None
                        else None
                    )
                    result = None
                    if result_path.is_file():
                        result_type = (
                            ENSREpisodeResult
                            if method == "ensr_v2"
                            else ScienceWorldEpisodeResult
                        )
                        try:
                            result = result_type.model_validate_json(
                                result_path.read_text(encoding="utf-8")
                            )
                        except ValueError:
                            print(
                                f"[{run_index}/{total}] stale cache {method} "
                                f"{selection['task_id']} v{variation}; recomputing",
                                flush=True,
                            )
                    if (
                        result is not None
                        and args.model_trace_mode != "off"
                        and trace_output_path is not None
                        and not trace_output_path.is_file()
                    ):
                        print(
                            f"[{run_index}/{total}] result has no model trace; "
                            f"recomputing {selection['task_id']} v{variation}",
                            flush=True,
                        )
                        result = None
                    if result is None and method == "ensr_v2":
                        if isinstance(adapter, ModelTraceAdapter):
                            if trace_output_path is None:
                                raise ValueError("Missing trace output path")
                            adapter.begin_episode(
                                trace_output_path,
                                source_trace_path=(
                                    trace_source_path
                                    if args.model_trace_mode
                                    in {"replay_prefix", "replay_until_divergence"}
                                    else None
                                ),
                            )
                        try:
                            action_replay_prefix = None
                            if action_replay_source_path is not None:
                                source_result = ENSREpisodeResult.model_validate_json(
                                    action_replay_source_path.read_text(
                                        encoding="utf-8"
                                    )
                                )
                                facts_by_step: dict[
                                    int, list[tuple[str, str, str, bool]]
                                ] = {}
                                for fact in source_result.temporal_fact_history:
                                    facts_by_step.setdefault(fact.step, []).append(
                                        fact.key
                                    )
                                action_replay_prefix = [
                                    {
                                        **step.model_dump(mode="json"),
                                        "symbolic_state_sha256": (
                                            symbolic_state_sha256(
                                                facts_by_step.get(
                                                    step.step_index, []
                                                )
                                            )
                                        ),
                                    }
                                    for step in source_result.trajectory
                                ]
                            result = ensr_harness.run_episode(
                                env,
                                task_id=str(selection["task_id"]),
                                task_name=str(selection["task_name"]),
                                split=args.split,
                                variation=variation,
                                seed=seed,
                                budget=ensr_budget,
                                simplifications=str(
                                    protocol["benchmark"]["simplifications"]
                                ),
                                action_replay_prefix=action_replay_prefix,
                            )
                        finally:
                            if isinstance(adapter, ModelTraceAdapter):
                                adapter.end_episode()
                        write_ensr_episode_result(result, result_path)
                    elif result is None:
                        result = baseline_harness.run_episode(
                            env,
                            method=method,
                            task_id=str(selection["task_id"]),
                            task_name=str(selection["task_name"]),
                            split=args.split,
                            variation=variation,
                            seed=seed,
                            budget=baseline_budget,
                            simplifications=str(protocol["benchmark"]["simplifications"]),
                        )
                        write_episode_result(result, result_path)
                    print(
                        f"[{run_index}/{total}] {method} {selection['task_id']} "
                        f"v{variation} status={result.status} score={result.final_score}",
                        flush=True,
                    )
                    results.append(result)
    finally:
        env.close()

    summary = _summarize(results)
    summary.update(
        {
            "schema_version": "2.0",
            "experiment_id": protocol["experiment_id"],
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": _file_hash(protocol_path),
            "implementation_sha256": _implementation_hash(),
            "memory_sha256": _file_hash(memory_path),
            "model_profiles_sha256": _file_hash(profile_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "selected_case_count": len(cases),
            "seeds": seeds,
            "model_id": adapter.model_id,
            "model_profile_id": profile["profile_id"],
            "ablation_arm": args.ablation_arm or None,
            "ablation_settings": (
                ablation_settings if args.ablation_arm else None
            ),
            "model_trace_mode": args.model_trace_mode,
            "model_trace_root": (
                str(args.model_trace_root.resolve())
                if args.model_trace_root is not None
                else None
            ),
            "model_trace_source_root": (
                str(args.model_trace_source_root.resolve())
                if args.model_trace_source_root is not None
                else None
            ),
            "action_replay_source_root": (
                str(args.action_replay_source_root.resolve())
                if args.action_replay_source_root is not None
                else None
            ),
            "reporting_boundary": (
                "development_only"
                if args.split == "dev"
                else "formal_machine_result"
            ),
        }
    )
    summary_path = run_root / f"summary_{'_'.join(methods)}.json"
    _write_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
