import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from lanternquest.schema import InventorySummary, MediaType, SourceRecord

MEDIA_TYPES: dict[str, MediaType] = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".m4a": "audio",
    ".mp3": "audio",
    ".wav": "audio",
    ".txt": "text",
    ".md": "text",
    ".pdf": "document",
    ".docx": "document",
    ".html": "web_index",
    ".htm": "web_index",
}


def sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_source_id(relative_path: str) -> str:
    path_digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"src_{path_digest}"


def iter_source_files(source_root: Path) -> Iterable[Path]:
    return sorted(
        (path for path in source_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix().casefold(),
    )


def build_record(path: Path, source_root: Path, chunk_size: int) -> SourceRecord:
    relative = path.relative_to(source_root).as_posix()
    parts = Path(relative).parts
    extension = path.suffix.lower()
    stat = path.stat()
    return SourceRecord(
        source_id=stable_source_id(relative),
        relative_path=relative,
        source_group=parts[0] if len(parts) > 1 else "root",
        filename=path.name,
        extension=extension,
        media_type=MEDIA_TYPES.get(extension, "other"),
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        sha256=sha256_file(path, chunk_size),
    )


def write_inventory(source_root: Path, output_path: Path, chunk_size: int) -> InventorySummary:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source corpus directory does not exist: {source_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[SourceRecord] = []
    hashes: dict[str, list[str]] = defaultdict(list)

    for path in iter_source_files(source_root):
        record = build_record(path, source_root, chunk_size)
        records.append(record)
        hashes[record.sha256].append(record.relative_path)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as destination:
        for record in records:
            destination.write(record.model_dump_json() + "\n")
    temporary_path.replace(output_path)

    summary = InventorySummary(
        generated_at=datetime.now(timezone.utc),
        source_root=str(source_root),
        file_count=len(records),
        total_bytes=sum(record.size_bytes for record in records),
        counts_by_media_type=dict(Counter(record.media_type for record in records)),
        counts_by_extension=dict(Counter(record.extension or "[none]" for record in records)),
        duplicate_content_groups=sum(1 for paths in hashes.values() if len(paths) > 1),
    )
    summary_path = output_path.with_name(f"{output_path.stem}.summary.json")
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary

