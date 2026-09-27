import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lanternquest.schema import SourceRecord, TaskCandidate

NUMBERED_SOURCE_PATTERN = re.compile(r"(?<![A-Za-z0-9])(N\d{3})(?!\d)", re.IGNORECASE)
FIELD_SOURCE_PATTERN = re.compile(r"(田野\s*\d{2})(?!\d)")

RIGHTS_QUEUE_FIELDS = [
    "source_group_id",
    "grouping_basis",
    "file_count",
    "media_types",
    "has_primary_media",
    "has_text_access_copy",
    "representative_path",
    "rights_status",
    "allowed_research_use",
    "allowed_external_processing",
    "allowed_public_release",
    "allowed_model_training",
    "creator",
    "reviewer",
    "reviewed_at",
    "notes",
]


@dataclass(frozen=True)
class SourcePackage:
    source_group_id: str
    grouping_basis: str
    records: tuple[SourceRecord, ...]


@dataclass(frozen=True)
class TaskSourceScope:
    source_group_id: str
    grouping_basis: str
    source_ids: tuple[str, ...]
    primary_source_ids: tuple[str, ...]
    access_source_ids: tuple[str, ...]
    media_types: tuple[str, ...]


def classify_source_group(relative_path: str) -> tuple[str, str]:
    """Return an administrative group key without making a rights claim."""
    numbered_match = NUMBERED_SOURCE_PATTERN.search(relative_path)
    if numbered_match:
        return numbered_match.group(1).upper(), "numbered_reference"

    field_match = FIELD_SOURCE_PATTERN.search(relative_path)
    if field_match:
        normalized = re.sub(r"\s+", "", field_match.group(1))
        return normalized, "field_collection"

    parts = PurePosixPath(relative_path).parts
    if len(parts) == 1:
        bucket = "root_files"
    else:
        bucket = "/".join(parts[: min(3, len(parts) - 1)])
    return f"UNNUMBERED::{bucket}", "directory_bucket"


def group_source_packages(records: list[SourceRecord]) -> list[SourcePackage]:
    grouped: dict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    for record in records:
        group_id, basis = classify_source_group(record.relative_path)
        grouped[(group_id, basis)].append(record)

    packages: list[SourcePackage] = []
    for (group_id, basis), package_records in sorted(
        grouped.items(), key=lambda item: item[0][0].casefold()
    ):
        packages.append(
            SourcePackage(
                source_group_id=group_id,
                grouping_basis=basis,
                records=tuple(
                    sorted(package_records, key=lambda item: item.relative_path.casefold())
                ),
            )
        )
    return packages


def create_source_rights_review_queue(
    records: list[SourceRecord], output_path: Path
) -> tuple[bool, int]:
    """Create the source-package rights queue once and preserve human review edits."""
    packages = group_source_packages(records)
    if output_path.exists():
        return False, len(packages)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RIGHTS_QUEUE_FIELDS)
        writer.writeheader()
        for package in packages:
            media_types = sorted({record.media_type for record in package.records})
            has_primary_media = any(
                media_type in {"image", "video", "audio", "document"}
                for media_type in media_types
            )
            writer.writerow(
                {
                    "source_group_id": package.source_group_id,
                    "grouping_basis": package.grouping_basis,
                    "file_count": len(package.records),
                    "media_types": ";".join(media_types),
                    "has_primary_media": str(has_primary_media).lower(),
                    "has_text_access_copy": str("text" in media_types).lower(),
                    "representative_path": package.records[0].relative_path,
                    "rights_status": "unreviewed",
                    "allowed_research_use": "unknown",
                    "allowed_external_processing": "unknown",
                    "allowed_public_release": "unknown",
                    "allowed_model_training": "unknown",
                    "creator": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "notes": "",
                }
            )
    return True, len(packages)


def migrate_source_rights_review_queue(output_path: Path) -> bool:
    """Add new review columns while preserving every existing human-entered value."""
    if not output_path.is_file():
        raise FileNotFoundError(f"Rights review queue does not exist: {output_path}")

    with output_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Rights review queue has no header")
        original_fields = list(reader.fieldnames)
        rows = list(reader)

    new_field = "allowed_external_processing"
    if new_field in original_fields:
        return False
    if "allowed_research_use" not in original_fields:
        raise ValueError("Rights review queue is missing allowed_research_use")

    insertion_index = original_fields.index("allowed_research_use") + 1
    migrated_fields = list(original_fields)
    migrated_fields.insert(insertion_index, new_field)
    for row in rows:
        row[new_field] = "unknown"

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=migrated_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)
    return True


def build_task_source_scopes(
    records: list[SourceRecord], task: TaskCandidate
) -> list[TaskSourceScope]:
    records_by_id = {record.source_id: record for record in records}
    primary_ids = {item.primary_source_id for item in task.evidence}
    access_ids = {
        source_id for item in task.evidence for source_id in item.access_source_ids
    }
    required_ids = primary_ids | access_ids
    missing_ids = sorted(required_ids - records_by_id.keys())
    if missing_ids:
        raise ValueError(
            "Task references source IDs missing from inventory: " + ", ".join(missing_ids)
        )

    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source_id in required_ids:
        record = records_by_id[source_id]
        group_id, basis = classify_source_group(record.relative_path)
        grouped[(group_id, basis)].add(source_id)

    scopes: list[TaskSourceScope] = []
    for (group_id, basis), source_ids in sorted(
        grouped.items(), key=lambda item: item[0][0].casefold()
    ):
        scopes.append(
            TaskSourceScope(
                source_group_id=group_id,
                grouping_basis=basis,
                source_ids=tuple(sorted(source_ids)),
                primary_source_ids=tuple(sorted(source_ids & primary_ids)),
                access_source_ids=tuple(sorted(source_ids & access_ids)),
                media_types=tuple(
                    sorted({records_by_id[source_id].media_type for source_id in source_ids})
                ),
            )
        )
    return scopes


def create_task_source_scope_queue(
    records: list[SourceRecord], task: TaskCandidate, output_path: Path
) -> tuple[bool, list[TaskSourceScope]]:
    """Create a focused task review queue while preserving later human edits."""
    scopes = build_task_source_scopes(records, task)
    if output_path.exists():
        return False, scopes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "source_group_id",
        "grouping_basis",
        "evidence_source_count",
        "primary_source_count",
        "access_source_count",
        "source_ids",
        "media_types",
        "rights_review_priority",
        "scope_review_status",
        "reviewer",
        "reviewed_at",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for scope in scopes:
            writer.writerow(
                {
                    "task_id": task.task_id,
                    "source_group_id": scope.source_group_id,
                    "grouping_basis": scope.grouping_basis,
                    "evidence_source_count": len(scope.source_ids),
                    "primary_source_count": len(scope.primary_source_ids),
                    "access_source_count": len(scope.access_source_ids),
                    "source_ids": ";".join(scope.source_ids),
                    "media_types": ";".join(scope.media_types),
                    "rights_review_priority": "high",
                    "scope_review_status": "pending",
                    "reviewer": "",
                    "reviewed_at": "",
                    "notes": "",
                }
            )
    return True, scopes
