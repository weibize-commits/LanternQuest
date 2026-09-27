import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExperimentStatus = Literal[
    "completed",
    "in_progress",
    "ready",
    "pending",
    "blocked",
    "deferred",
]
ReportingBoundary = Literal[
    "setup_only",
    "plumbing_only",
    "formal_machine_result",
    "formal_human_result",
]


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=r"^E\d{2}$")
    title: str
    kind: Literal[
        "setup",
        "benchmark",
        "dataset",
        "ablation",
        "efficiency",
        "expert_evaluation",
        "user_study",
    ]
    status: ExperimentStatus
    reporting_boundary: ReportingBoundary
    hypothesis: str | None = None
    dataset: str
    methods: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reporting_boundary(self) -> "ExperimentSpec":
        is_formal = self.reporting_boundary.startswith("formal_")
        if is_formal and not self.hypothesis:
            raise ValueError("Formal experiments require an explicit hypothesis")
        if self.status == "blocked" and not self.blockers:
            raise ValueError("Blocked experiments must record at least one blocker")
        if self.reporting_boundary == "plumbing_only" and self.hypothesis:
            raise ValueError("Plumbing checks must not claim a research hypothesis")
        return self


class ExperimentRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    registry_id: str
    research_question: str
    primary_claim: str
    status_definitions: dict[str, str]
    experiments: list[ExperimentSpec]

    @model_validator(mode="after")
    def validate_graph(self) -> "ExperimentRegistry":
        experiment_ids = [item.experiment_id for item in self.experiments]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("Experiment identifiers must be unique")
        known_ids = set(experiment_ids)
        for experiment in self.experiments:
            unknown = set(experiment.prerequisites) - known_ids
            if unknown:
                raise ValueError(
                    f"{experiment.experiment_id} has unknown prerequisites: {sorted(unknown)}"
                )
            if experiment.experiment_id in experiment.prerequisites:
                raise ValueError(
                    f"{experiment.experiment_id} cannot depend on itself"
                )
        return self

    def summary(self) -> dict[str, object]:
        status_counts = Counter(item.status for item in self.experiments)
        boundary_counts = Counter(item.reporting_boundary for item in self.experiments)
        paper_eligible = [
            item.experiment_id
            for item in self.experiments
            if item.reporting_boundary.startswith("formal_")
            and item.status == "completed"
        ]
        next_ready = [
            item.experiment_id
            for item in self.experiments
            if item.status in {"ready", "in_progress"}
        ]
        blocked = {
            item.experiment_id: item.blockers
            for item in self.experiments
            if item.status == "blocked"
        }
        return {
            "registry_id": self.registry_id,
            "experiment_count": len(self.experiments),
            "status_counts": dict(sorted(status_counts.items())),
            "reporting_boundary_counts": dict(sorted(boundary_counts.items())),
            "paper_eligible_completed": paper_eligible,
            "next_ready": next_ready,
            "blocked": blocked,
        }


def load_experiment_registry(path: Path) -> ExperimentRegistry:
    return ExperimentRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def render_registry_status(
    registry: ExperimentRegistry, *, include_details: bool = False
) -> dict[str, object]:
    payload = registry.summary()
    if include_details:
        payload["experiments"] = [
            item.model_dump(mode="json") for item in registry.experiments
        ]
    return payload


def write_registry_snapshot(registry: ExperimentRegistry, output_path: Path) -> None:
    payload = {
        "schema_version": registry.schema_version,
        "registry_id": registry.registry_id,
        "summary": registry.summary(),
        "experiments": [item.model_dump(mode="json") for item in registry.experiments],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
