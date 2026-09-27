import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lanternquest.schema import CorpusAudit, DuplicateFile, DuplicateGroup, SourceRecord


def load_inventory(path: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(SourceRecord.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"Invalid inventory record at line {line_number}") from error
    return records


def build_corpus_audit(records: list[SourceRecord], manifest_path: Path) -> CorpusAudit:
    records_by_hash: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        records_by_hash[record.sha256].append(record)

    duplicate_groups = [
        DuplicateGroup(
            sha256=sha256,
            files=[
                DuplicateFile(
                    source_id=record.source_id,
                    relative_path=record.relative_path,
                    size_bytes=record.size_bytes,
                    media_type=record.media_type,
                )
                for record in sorted(group, key=lambda item: item.relative_path.casefold())
            ],
        )
        for sha256, group in sorted(records_by_hash.items())
        if len(group) > 1
    ]
    return CorpusAudit(
        generated_at=datetime.now(timezone.utc),
        manifest_path=str(manifest_path),
        file_count=len(records),
        unique_source_ids=len({record.source_id for record in records}),
        unique_paths=len({record.relative_path for record in records}),
        duplicate_content_groups=duplicate_groups,
    )


def write_corpus_audit(manifest_path: Path, output_path: Path) -> CorpusAudit:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Inventory does not exist: {manifest_path}")

    audit = build_corpus_audit(load_inventory(manifest_path), manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(audit.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return audit


def create_duplicate_review_queue(audit: CorpusAudit, output_path: Path) -> bool:
    """Create a human review queue once, without overwriting later decisions."""
    if output_path.exists():
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "group_id",
                "sha256",
                "source_id",
                "relative_path",
                "size_bytes",
                "media_type",
                "review_decision",
                "canonical_source_id",
                "notes",
            ],
        )
        writer.writeheader()
        for group_number, group in enumerate(audit.duplicate_content_groups, start=1):
            for file in group.files:
                writer.writerow(
                    {
                        "group_id": f"dup_{group_number:03d}",
                        "sha256": group.sha256,
                        "source_id": file.source_id,
                        "relative_path": file.relative_path,
                        "size_bytes": file.size_bytes,
                        "media_type": file.media_type,
                        "review_decision": "pending",
                        "canonical_source_id": "",
                        "notes": "",
                    }
                )
    return True
