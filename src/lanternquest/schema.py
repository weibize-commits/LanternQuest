from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MediaType = Literal[
    "image",
    "video",
    "audio",
    "text",
    "document",
    "web_index",
    "other",
]


class SourceRecord(BaseModel):
    """One immutable source file from the local corpus."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    relative_path: str
    source_group: str
    filename: str
    extension: str
    media_type: MediaType
    size_bytes: int = Field(ge=0)
    modified_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rights_status: Literal["unreviewed", "approved", "restricted", "unknown"] = "unreviewed"
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"


class InventorySummary(BaseModel):
    schema_version: str = "1.0"
    generated_at: datetime
    source_root: str
    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    counts_by_media_type: dict[str, int]
    counts_by_extension: dict[str, int]
    duplicate_content_groups: int = Field(ge=0)


class DuplicateFile(BaseModel):
    source_id: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    media_type: MediaType


class DuplicateGroup(BaseModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[DuplicateFile] = Field(min_length=2)


class CorpusAudit(BaseModel):
    schema_version: str = "1.0"
    generated_at: datetime
    manifest_path: str
    file_count: int = Field(ge=0)
    unique_source_ids: int = Field(ge=0)
    unique_paths: int = Field(ge=0)
    duplicate_content_groups: list[DuplicateGroup]


ReviewStatus = Literal["source_checked", "domain_pending", "domain_approved", "rejected"]


class EvidenceReference(BaseModel):
    evidence_id: str
    primary_source_id: str
    access_source_ids: list[str] = Field(default_factory=list)
    locator: str
    claim: str
    scope: str
    review_status: ReviewStatus


class DecisionOption(BaseModel):
    option_id: str
    label: str
    consequence: str
    evidence_ids: list[str]


class DecisionNode(BaseModel):
    decision_id: str
    prompt: str
    options: list[DecisionOption] = Field(min_length=2)
    timing: str
    review_status: ReviewStatus


class TaskCandidate(BaseModel):
    schema_version: str = "1.0"
    task_id: str
    title: str
    status: Literal["candidate", "domain_approved", "frozen"]
    target_output: str
    included_scope: list[str]
    excluded_scope: list[str]
    decision_nodes: list[DecisionNode]
    evidence: list[EvidenceReference]
    risk_controls: list[str]
    known_unknowns: list[str]

    @model_validator(mode="after")
    def validate_internal_references(self) -> "TaskCandidate":
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence identifiers must be unique")

        unknown_references = {
            evidence_id
            for decision in self.decision_nodes
            for option in decision.options
            for evidence_id in option.evidence_ids
            if evidence_id not in evidence_ids
        }
        if unknown_references:
            raise ValueError(
                "Decision options reference unknown evidence: "
                + ", ".join(sorted(unknown_references))
            )
        return self


class TextIndexSummary(BaseModel):
    schema_version: str = "1.0"
    generated_at: datetime
    database_path: str
    source_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    tokenizer: str
    decode_failures: list[str]


class RetrievalCase(BaseModel):
    case_id: str
    query: str
    acceptable_source_ids: list[str] = Field(min_length=1)
    expected_claim: str
    construction_note: str


class RetrievalDataset(BaseModel):
    schema_version: str = "1.0"
    dataset_id: str
    split: Literal["dev", "test"]
    purpose: Literal["plumbing_smoke_test", "retrieval_development", "frozen_evaluation"]
    status: str
    source_task_id: str
    created_at: datetime
    cases: list[RetrievalCase] = Field(min_length=1)


class RetrievalCaseResult(BaseModel):
    case_id: str
    query: str
    ranked_source_ids: list[str]
    first_relevant_rank: int | None
    hit_at_k: bool
    recall_at_k: float = Field(ge=0, le=1)


class RetrievalEvaluation(BaseModel):
    schema_version: str = "1.0"
    evaluation_id: str
    generated_at: datetime
    dataset_id: str
    split: Literal["dev", "test"]
    database_path: str
    top_k: int = Field(ge=1)
    case_count: int = Field(ge=1)
    hit_rate_at_k: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    mean_recall_at_k: float = Field(ge=0, le=1)
    cases: list[RetrievalCaseResult]
