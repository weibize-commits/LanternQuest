import hashlib
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from lanternquest.schema import SourceRecord, TextIndexSummary


def decode_text(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", payload, 0, len(payload), "unsupported text encoding")


def iter_text_chunks(text: str, max_chars: int = 1200) -> Iterable[tuple[int, int, str]]:
    """Yield paragraph-aware chunks with one-based source line ranges."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    block: list[tuple[int, str]] = []

    def split_block(items: list[tuple[int, str]]) -> Iterable[tuple[int, int, str]]:
        current: list[tuple[int, str]] = []
        current_length = 0
        for line_number, line in items:
            if len(line) > max_chars:
                if current:
                    yield current[0][0], current[-1][0], "\n".join(item[1] for item in current)
                    current = []
                    current_length = 0
                for offset in range(0, len(line), max_chars):
                    yield line_number, line_number, line[offset : offset + max_chars]
                continue

            added_length = len(line) + (1 if current else 0)
            if current and current_length + added_length > max_chars:
                yield current[0][0], current[-1][0], "\n".join(item[1] for item in current)
                current = []
                current_length = 0
            current.append((line_number, line))
            current_length += len(line) + (1 if len(current) > 1 else 0)
        if current:
            yield current[0][0], current[-1][0], "\n".join(item[1] for item in current)

    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            block.append((line_number, line))
            continue
        if block:
            yield from split_block(block)
            block = []
    if block:
        yield from split_block(block)


def safe_source_path(source_root: Path, relative_path: str) -> Path:
    source_root = source_root.resolve()
    candidate = (source_root / Path(PurePosixPath(relative_path))).resolve()
    if not candidate.is_relative_to(source_root):
        raise ValueError(f"Source path escapes corpus root: {relative_path}")
    return candidate


def build_text_index(
    source_root: Path,
    records: list[SourceRecord],
    database_path: Path,
    max_chars: int = 1200,
) -> TextIndexSummary:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(database_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary_path)
    tokenizer = "trigram"
    decode_failures: list[str] = []
    source_count = 0
    chunk_count = 0
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE sources (
                source_id TEXT PRIMARY KEY,
                relative_path TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,
                rights_status TEXT NOT NULL,
                encoding TEXT NOT NULL
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            """
        )
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING "
                "fts5(text, content='chunks', content_rowid='id', tokenize=trigram)"
            )
        except sqlite3.OperationalError:
            tokenizer = "unicode61"
            connection.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING "
                "fts5(text, content='chunks', content_rowid='id', tokenize=unicode61)"
            )

        for record in records:
            if record.media_type != "text":
                continue
            path = safe_source_path(source_root, record.relative_path)
            try:
                text, encoding = decode_text(path)
            except UnicodeDecodeError:
                decode_failures.append(record.relative_path)
                continue

            connection.execute(
                "INSERT INTO sources(source_id, relative_path, sha256, rights_status, encoding) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record.source_id,
                    record.relative_path,
                    record.sha256,
                    record.rights_status,
                    encoding,
                ),
            )
            source_count += 1
            for line_start, line_end, chunk_text in iter_text_chunks(text, max_chars=max_chars):
                chunk_seed = (
                    f"{record.source_id}:{line_start}:{line_end}:{chunk_text}".encode()
                )
                chunk_id = f"chk_{hashlib.sha256(chunk_seed).hexdigest()[:20]}"
                connection.execute(
                    "INSERT INTO chunks(chunk_id, source_id, line_start, line_end, text) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, record.source_id, line_start, line_end, chunk_text),
                )
                chunk_count += 1

        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        generated_at = datetime.now(timezone.utc)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", "1.0"),
                ("generated_at", generated_at.isoformat()),
                ("tokenizer", tokenizer),
                ("max_chars", str(max_chars)),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    temporary_path.replace(database_path)
    return TextIndexSummary(
        generated_at=generated_at,
        database_path=str(database_path),
        source_count=source_count,
        chunk_count=chunk_count,
        tokenizer=tokenizer,
        decode_failures=decode_failures,
    )


def search_text_index(database_path: Path, query: str, limit: int = 5) -> list[dict[str, object]]:
    terms = [term for term in query.split() if term]
    if not terms:
        return []

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        if any(len(term) < 3 for term in terms):
            where_clause = " AND ".join("c.text LIKE ? ESCAPE '\\'" for _ in terms)
            escaped_terms = [
                term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                for term in terms
            ]
            parameters = ["%" + term + "%" for term in escaped_terms]
            rows = connection.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.source_id,
                    s.relative_path,
                    c.line_start,
                    c.line_end,
                    c.text
                FROM chunks AS c
                JOIN sources AS s ON s.source_id = c.source_id
                WHERE {where_clause}
                """,
                parameters,
            ).fetchall()
            results: list[dict[str, object]] = []
            for row in rows:
                item = dict(row)
                text = str(item.pop("text"))
                positions = [text.find(term) for term in terms]
                first_match = min(positions)
                match_count = sum(text.count(term) for term in terms)
                match_span = max(positions) - min(positions)
                start = max(0, first_match - 60)
                end = min(len(text), first_match + 180)
                snippet = text[start:end]
                for term in terms:
                    snippet = snippet.replace(term, f"[{term}]")
                item["snippet"] = (
                    ("…" if start else "")
                    + snippet
                    + ("…" if end < len(text) else "")
                )
                item["score"] = -float(match_count) + match_span / 1_000_000
                results.append(item)
            return sorted(
                results,
                key=lambda item: (
                    item["score"],
                    item["relative_path"],
                    item["line_start"],
                ),
            )[:limit]

        match_query = " AND ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
        )
        rows = connection.execute(
            """
            SELECT
                c.chunk_id,
                c.source_id,
                s.relative_path,
                c.line_start,
                c.line_end,
                snippet(chunks_fts, 0, '[', ']', '…', 24) AS snippet,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks AS c ON c.id = chunks_fts.rowid
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE chunks_fts MATCH ?
            ORDER BY score, c.chunk_id
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
