from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = ROOT / "kg" / "data" / "faster_whisper_field07_pilot.json"
DEFAULT_INVENTORY = ROOT / "artifacts" / "provenance" / "source_inventory.jsonl"
DEFAULT_OUTPUT = ROOT / "kg" / "data" / "faster_whisper_field07_diagnostic.json"
REFERENCE_SOURCE_ID = "src_16b2d629f1facdb9"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_record(path: Path, source_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("source_id") == source_id:
                return record
    raise RuntimeError(f"source not found in inventory: {source_id}")


def locate_reference(record: dict[str, Any]) -> Path:
    candidates = []
    for path in ROOT.rglob(f"*{record['extension']}"):
        if ".venv" in path.parts or "artifacts" in path.parts:
            continue
        if path.stat().st_size == record["size_bytes"]:
            candidates.append(path)
    matches = [path for path in candidates if file_sha256(path) == record["sha256"]]
    if len(matches) != 1:
        raise RuntimeError(f"expected one reference text, found {len(matches)}")
    return matches[0]


def extract_reference_transcript(text: str) -> str:
    marker = "完整转录"
    if marker not in text:
        raise RuntimeError("reference transcript marker not found")
    body = text.rsplit(marker, maxsplit=1)[1]
    utterances = []
    for line in body.splitlines():
        match = re.match(r"^\[\d{2}:\d{2}\]\s*[^：]+：\s*(.+)$", line.strip())
        if match:
            utterances.append(match.group(1))
    if not utterances:
        raise RuntimeError("no timestamped utterances found in reference transcript")
    return "".join(utterances)


def normalize_for_diagnostic(text: str) -> str:
    text = re.sub(r"【.*?】", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in text if character.isalnum())


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def evaluate(
    pilot: dict[str, Any],
    reference: dict[str, Any],
    reference_text: str,
) -> dict[str, Any]:
    extracted_reference = extract_reference_transcript(reference_text)
    normalized_reference = normalize_for_diagnostic(extracted_reference)
    normalized_hypothesis = normalize_for_diagnostic(pilot["transcript_text"])
    distance = levenshtein_distance(normalized_reference, normalized_hypothesis)
    return {
        "schema_version": "0.1",
        "evaluation_id": "field07_asr_rough_reference_diagnostic_v0",
        "reporting_boundary": "diagnostic_only_reference_is_not_gold_standard",
        "pilot_id": pilot["pilot_id"],
        "audio_source_id": pilot["source"]["source_id"],
        "reference_source": {
            "source_id": reference["source_id"],
            "sha256": reference["sha256"],
            "reference_quality": "rough_local_asr_checked_against_user_transcript",
        },
        "normalization": "NFKC lowercase alphanumeric only; editorial brackets removed",
        "reference_character_count": len(normalized_reference),
        "hypothesis_character_count": len(normalized_hypothesis),
        "character_edit_distance": distance,
        "diagnostic_character_error_rate": round(
            distance / len(normalized_reference), 6
        ),
        "interpretation": (
            "This number diagnoses the pilot against an existing rough transcript. "
            "It is not a publishable ASR accuracy result and must not be treated as "
            "a gold-standard CER."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Field07 ASR output")
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    reference = inventory_record(args.inventory, REFERENCE_SOURCE_ID)
    reference_path = locate_reference(reference)
    reference_text = reference_path.read_text(encoding="utf-8-sig")
    result = evaluate(pilot, reference, reference_text)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: diagnostic CER="
        f"{result['diagnostic_character_error_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
