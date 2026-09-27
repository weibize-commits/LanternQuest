import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import typer

from lanternquest.audit import create_duplicate_review_queue, load_inventory, write_corpus_audit
from lanternquest.config import Settings
from lanternquest.experiment import (
    ExperimentFixture,
    ExperimentHarness,
    write_smoke_report,
)
from lanternquest.experiment_registry import (
    load_experiment_registry,
    render_registry_status,
    write_registry_snapshot,
)
from lanternquest.inventory import write_inventory
from lanternquest.planning import SearchBudget
from lanternquest.retrieval_eval import (
    evaluate_retrieval,
    load_retrieval_dataset,
    write_evaluation,
)
from lanternquest.rights import (
    create_source_rights_review_queue,
    create_task_source_scope_queue,
    migrate_source_rights_review_queue,
)
from lanternquest.schema import TaskCandidate
from lanternquest.system import find_executable
from lanternquest.text_index import build_text_index, search_text_index

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

app = typer.Typer(
    name="lanternquest",
    help="Local provenance and experiment tooling for LanternQuest.",
    no_args_is_help=True,
)


@app.command("experiment-status")
def experiment_status(
    registry: Annotated[
        Path,
        typer.Option(help="Machine-readable experiment registry."),
    ] = Path("configs/experiment_registry_v0.json"),
    include_details: Annotated[
        bool,
        typer.Option(help="Include every experiment specification in the output."),
    ] = False,
) -> None:
    """Validate the experiment graph and show executable, blocked, and paper-eligible work."""
    registry_path = registry.resolve()
    experiment_registry = load_experiment_registry(registry_path)
    snapshot_path = (
        Settings().resolved_artifact_root()
        / "experiments"
        / f"{experiment_registry.registry_id}_snapshot.json"
    )
    write_registry_snapshot(experiment_registry, snapshot_path)
    payload = render_registry_status(
        experiment_registry, include_details=include_details
    )
    payload["registry"] = str(registry_path)
    payload["snapshot"] = str(snapshot_path.resolve())
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def doctor() -> None:
    """Check the local runtime and source corpus without modifying the corpus."""
    settings = Settings()
    source_root = settings.resolved_source_root()
    try:
        openai_sdk_version = version("openai")
    except PackageNotFoundError:
        openai_sdk_version = None
    api_key_configured = bool(
        settings.openai_api_key
        and settings.openai_api_key.get_secret_value().strip()
    )
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "source_root": str(source_root),
        "source_root_exists": source_root.is_dir(),
        "artifact_root": str(settings.resolved_artifact_root()),
        "ffmpeg": find_executable("ffmpeg"),
        "ffprobe": find_executable("ffprobe"),
        "openai_sdk": openai_sdk_version,
        "llm_model_configured": bool(settings.llm_model),
        "llm_base_url_configured": bool(settings.openai_base_url),
        "llm_api_key_configured": api_key_configured,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if not source_root.is_dir():
        raise typer.Exit(code=1)


@app.command()
def inventory(
    output: Annotated[
        Path | None,
        typer.Option(
            help="JSONL destination; defaults to artifacts/provenance/source_inventory.jsonl."
        ),
    ] = None,
) -> None:
    """Hash every source file and write an atomic, deterministic inventory."""
    settings = Settings()
    source_root = settings.resolved_source_root()
    output_path = output or (
        settings.resolved_artifact_root() / "provenance" / "source_inventory.jsonl"
    )
    summary = write_inventory(
        source_root=source_root,
        output_path=output_path.resolve(),
        chunk_size=settings.hash_chunk_mb * 1024 * 1024,
    )
    typer.echo(summary.model_dump_json(indent=2))


@app.command()
def audit(
    manifest: Annotated[
        Path | None,
        typer.Option(help="Source inventory JSONL to audit."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Audit report destination."),
    ] = None,
) -> None:
    """Create a reviewable duplicate-content report from the source inventory."""
    settings = Settings()
    artifact_root = settings.resolved_artifact_root()
    manifest_path = manifest or artifact_root / "provenance" / "source_inventory.jsonl"
    output_path = output or artifact_root / "provenance" / "corpus_audit.json"
    report = write_corpus_audit(manifest_path.resolve(), output_path.resolve())
    review_queue_path = Path("annotations/review/duplicate_review.csv").resolve()
    review_queue_created = create_duplicate_review_queue(report, review_queue_path)
    typer.echo(
        json.dumps(
            {
                "file_count": report.file_count,
                "unique_source_ids": report.unique_source_ids,
                "unique_paths": report.unique_paths,
                "duplicate_content_groups": len(report.duplicate_content_groups),
                "output": str(output_path.resolve()),
                "review_queue": str(review_queue_path),
                "review_queue_created": review_queue_created,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("rights-queue")
def rights_queue(
    manifest: Annotated[
        Path | None,
        typer.Option(help="Source inventory JSONL used to build the review queue."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Human-editable source-package rights review CSV."),
    ] = Path("annotations/review/source_rights_review.csv"),
) -> None:
    """Create a non-destructive rights review queue grouped by source package."""
    settings = Settings()
    manifest_path = manifest or (
        settings.resolved_artifact_root() / "provenance" / "source_inventory.jsonl"
    )
    if not manifest_path.is_file():
        typer.echo("Inventory does not exist. Run `lanternquest inventory` first.", err=True)
        raise typer.Exit(code=1)

    created, package_count = create_source_rights_review_queue(
        load_inventory(manifest_path.resolve()), output.resolve()
    )
    typer.echo(
        json.dumps(
            {
                "source_package_count": package_count,
                "output": str(output.resolve()),
                "created": created,
                "permission_defaults": "unknown",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("task-rights-scope")
def task_rights_scope(
    task: Annotated[
        Path,
        typer.Option(help="Machine-readable task candidate."),
    ] = Path("annotations/seed/task_needle_piercing_v0.json"),
    manifest: Annotated[
        Path | None,
        typer.Option(help="Source inventory JSONL used to resolve evidence IDs."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Focused, human-editable task source review CSV."),
    ] = Path("annotations/review/first_task_source_scope.csv"),
) -> None:
    """Resolve a task's evidence IDs to the source packages needing priority review."""
    settings = Settings()
    manifest_path = manifest or (
        settings.resolved_artifact_root() / "provenance" / "source_inventory.jsonl"
    )
    if not manifest_path.is_file():
        typer.echo("Inventory does not exist. Run `lanternquest inventory` first.", err=True)
        raise typer.Exit(code=1)

    task_path = task.resolve()
    candidate = TaskCandidate.model_validate_json(task_path.read_text(encoding="utf-8"))
    created, scopes = create_task_source_scope_queue(
        load_inventory(manifest_path.resolve()), candidate, output.resolve()
    )
    typer.echo(
        json.dumps(
            {
                "task_id": candidate.task_id,
                "source_group_count": len(scopes),
                "evidence_source_count": len(
                    {source_id for scope in scopes for source_id in scope.source_ids}
                ),
                "source_groups": [scope.source_group_id for scope in scopes],
                "output": str(output.resolve()),
                "created": created,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("migrate-rights-queue")
def migrate_rights_queue(
    review_queue: Annotated[
        Path,
        typer.Option(help="Existing source-package rights review CSV."),
    ] = Path("annotations/review/source_rights_review.csv"),
) -> None:
    """Add required rights columns without replacing human-entered decisions."""
    review_path = review_queue.resolve()
    migrated = migrate_source_rights_review_queue(review_path)
    typer.echo(
        json.dumps(
            {
                "output": str(review_path),
                "migrated": migrated,
                "added_defaults": (
                    {"allowed_external_processing": "unknown"} if migrated else {}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("index-text")
def index_text(
    max_chars: Annotated[
        int,
        typer.Option(min=200, max=4000, help="Maximum characters per source-grounded chunk."),
    ] = 1200,
) -> None:
    """Build a local SQLite FTS index from the corpus text files."""
    settings = Settings()
    artifact_root = settings.resolved_artifact_root()
    manifest_path = artifact_root / "provenance" / "source_inventory.jsonl"
    database_path = artifact_root / "index" / "lanternquest_text.sqlite3"
    summary = build_text_index(
        source_root=settings.resolved_source_root(),
        records=load_inventory(manifest_path),
        database_path=database_path,
        max_chars=max_chars,
    )
    typer.echo(summary.model_dump_json(indent=2))


@app.command("search-text")
def search_text(
    query: Annotated[str, typer.Argument(help="Literal keyword or space-separated AND terms.")],
    limit: Annotated[int, typer.Option(min=1, max=50)] = 5,
) -> None:
    """Search the local text index and return traceable source line ranges."""
    settings = Settings()
    database_path = settings.resolved_artifact_root() / "index" / "lanternquest_text.sqlite3"
    if not database_path.is_file():
        typer.echo("Text index does not exist. Run `lanternquest index-text` first.", err=True)
        raise typer.Exit(code=1)
    results = search_text_index(database_path, query, limit=limit)
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))


@app.command("eval-retrieval")
def eval_retrieval(
    dataset: Annotated[
        Path,
        typer.Option(help="Retrieval evaluation dataset."),
    ] = Path("annotations/eval/retrieval_smoke_v0.json"),
    top_k: Annotated[int, typer.Option(min=1, max=50)] = 5,
) -> None:
    """Evaluate the local lexical index on a traceable development set."""
    settings = Settings()
    database_path = settings.resolved_artifact_root() / "index" / "lanternquest_text.sqlite3"
    if not database_path.is_file():
        typer.echo("Text index does not exist. Run `lanternquest index-text` first.", err=True)
        raise typer.Exit(code=1)

    dataset_path = dataset.resolve()
    retrieval_dataset = load_retrieval_dataset(dataset_path)
    result = evaluate_retrieval(database_path, retrieval_dataset, top_k=top_k)
    output_path = (
        settings.resolved_artifact_root()
        / "eval"
        / f"{retrieval_dataset.dataset_id}_lexical.json"
    )
    write_evaluation(result, output_path)
    typer.echo(result.model_dump_json(indent=2))


@app.command("eval-planning-smoke")
def eval_planning_smoke(
    fixture: Annotated[
        Path,
        typer.Option(help="Synthetic planning fixture used for plumbing checks."),
    ] = Path("annotations/eval/planning_smoke_v0.json"),
    output: Annotated[
        Path | None,
        typer.Option(help="JSON report destination under local artifacts."),
    ] = None,
    max_retrieval_calls: Annotated[
        int,
        typer.Option(min=1, max=50, help="Shared retrieval-call ceiling."),
    ] = 6,
) -> None:
    """Run all four method entrypoints on a non-domain synthetic smoke case."""
    fixture_path = fixture.resolve()
    experiment_fixture = ExperimentFixture.model_validate_json(
        fixture_path.read_text(encoding="utf-8")
    )
    budget = SearchBudget(max_retrieval_calls=max_retrieval_calls)
    harness = ExperimentHarness(experiment_fixture)
    methods = (
        "b0_full_context",
        "b1_rag",
        "b4_sequential",
        "iper_rag",
    )
    runs = [harness.prepare(method, budget) for method in methods]
    output_path = output or (
        Settings().resolved_artifact_root()
        / "eval"
        / f"{experiment_fixture.fixture_id}_methods.json"
    )
    write_smoke_report(experiment_fixture, runs, budget, output_path.resolve())
    typer.echo(
        json.dumps(
            {
                "fixture_id": experiment_fixture.fixture_id,
                "purpose": experiment_fixture.purpose,
                "reporting_boundary": "plumbing_only",
                "input_fingerprint": runs[0].input_fingerprint,
                "corpus_fingerprint": runs[0].corpus_fingerprint,
                "methods": {
                    run.method: {
                        "status": run.status,
                        "decision_type": (
                            run.decision.decision_type if run.decision else "awaiting_llm"
                        ),
                        "retrieved_evidence_ids": run.retrieved_evidence_ids,
                    }
                    for run in runs
                },
                "output": str(output_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
