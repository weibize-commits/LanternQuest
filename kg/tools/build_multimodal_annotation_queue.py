from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from kg.tools.evaluate_asr_pilot import inventory_record, locate_reference

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP = ROOT / "kg" / "data" / "chinese_clip_first_task_pilot.json"
DEFAULT_ASR = ROOT / "kg" / "data" / "faster_whisper_field07_pilot.json"
DEFAULT_INVENTORY = ROOT / "artifacts" / "provenance" / "source_inventory.jsonl"
DEFAULT_OUTPUT = ROOT / "annotations" / "eval" / "multimodal_gold_pilot_v0.csv"
REFERENCE_SOURCE_ID = "src_16b2d629f1facdb9"

FIELDNAMES = [
    "row_id",
    "modality",
    "source_id",
    "locator",
    "candidate_id",
    "candidate_target_id",
    "candidate_text",
    "reference_text",
    "neural_score",
    "gate_decision",
    "label_set",
    "annotator_a_label",
    "annotator_b_label",
    "adjudicated_label",
    "error_type",
    "notes",
]


def parse_reference_utterances(text: str, duration: float) -> list[dict[str, Any]]:
    body = text.rsplit("完整转录", maxsplit=1)[1]
    utterances = []
    pattern = re.compile(
        r"^\[(\d{2}):(\d{2})\]\s*([^：]+)：\s*(.+)$"
    )
    for line in body.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        minute, second, speaker, content = match.groups()
        utterances.append(
            {
                "start": int(minute) * 60 + int(second),
                "speaker": speaker,
                "text": content,
            }
        )
    for index, utterance in enumerate(utterances):
        utterance["end"] = (
            utterances[index + 1]["start"]
            if index + 1 < len(utterances)
            else duration
        )
    return utterances


def reference_for_segment(
    segment: dict[str, Any],
    utterances: list[dict[str, Any]],
) -> str:
    midpoint = (segment["start_seconds"] + segment["end_seconds"]) / 2
    matches = [
        utterance
        for utterance in utterances
        if utterance["start"] <= midpoint < utterance["end"]
    ]
    if not matches:
        return ""
    utterance = matches[-1]
    return f"{utterance['speaker']}：{utterance['text']}"


def build_rows(
    clip: dict[str, Any],
    asr: dict[str, Any],
    reference_text: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alignment in clip["alignments"]:
        source_id = alignment["source_id"]
        for match in alignment["top_matches"]:
            rows.append(
                {
                    "row_id": f"mmgold:image:{len(rows) + 1:03d}",
                    "modality": "image_text_alignment",
                    "source_id": source_id,
                    "locator": alignment["relative_path"],
                    "candidate_id": match["candidate"]["candidate_id"],
                    "candidate_target_id": match["claim_id"],
                    "candidate_text": match["claim_text"],
                    "reference_text": "",
                    "neural_score": match["normalized_similarity"],
                    "gate_decision": match["gate"]["decision"],
                    "label_set": "supports|partial|unrelated|contradicts|unclear",
                    "annotator_a_label": "",
                    "annotator_b_label": "",
                    "adjudicated_label": "",
                    "error_type": "",
                    "notes": "",
                }
            )

    utterances = parse_reference_utterances(
        reference_text,
        asr["inference"]["audio_duration_seconds"],
    )
    image_count = len(rows)
    for index, segment in enumerate(asr["segments"], start=1):
        rows.append(
            {
                "row_id": f"mmgold:audio:{index:03d}",
                "modality": "audio_transcription",
                "source_id": asr["source"]["source_id"],
                "locator": (
                    f"{segment['start_seconds']:.3f}-"
                    f"{segment['end_seconds']:.3f}s"
                ),
                "candidate_id": segment["segment_id"],
                "candidate_target_id": "",
                "candidate_text": segment["text"],
                "reference_text": reference_for_segment(segment, utterances),
                "neural_score": segment["avg_logprob"],
                "gate_decision": segment["status"],
                "label_set": (
                    "exact_or_minor|semantic_preserved|substantive_error|"
                    "inaudible|unclear"
                ),
                "annotator_a_label": "",
                "annotator_b_label": "",
                "adjudicated_label": "",
                "error_type": "",
                "notes": "",
            }
        )
    if image_count != 12 or len(rows) - image_count != 22:
        raise RuntimeError("unexpected pilot candidate counts")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the double-annotation queue for multimodal pilot candidates"
    )
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--asr", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    clip = json.loads(args.clip.read_text(encoding="utf-8"))
    asr = json.loads(args.asr.read_text(encoding="utf-8"))
    reference = inventory_record(args.inventory, REFERENCE_SOURCE_ID)
    reference_path = locate_reference(reference)
    rows = build_rows(
        clip,
        asr,
        reference_path.read_text(encoding="utf-8-sig"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}: {len(rows)} double-annotation rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
