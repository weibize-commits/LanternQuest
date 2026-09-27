from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_lanternquest_e27_ie import evaluate  # noqa: E402
from finalize_lanternquest_e27_results import paired_bootstrap  # noqa: E402


ENTITY_TYPES = [
    "LANTERN_TYPE",
    "PROCESS",
    "MATERIAL",
    "TOOL",
    "PATTERN",
    "STRUCTURE",
    "EFFECT",
    "CONDITION_PARAMETER",
    "ACTOR",
]
RELATION_TYPES = [
    "USES_MATERIAL",
    "USES_TOOL",
    "APPLIES_TO",
    "HAS_PATTERN",
    "HAS_COMPONENT",
    "PRECEDES",
    "PRODUCES_EFFECT",
    "HAS_CONDITION",
    "PERFORMED_BY",
]
DOMAIN_RANGE = {
    "USES_MATERIAL": ({"PROCESS"}, {"MATERIAL"}),
    "USES_TOOL": ({"PROCESS"}, {"TOOL"}),
    "APPLIES_TO": ({"PROCESS", "PATTERN", "EFFECT"}, {"LANTERN_TYPE", "STRUCTURE"}),
    "HAS_PATTERN": ({"LANTERN_TYPE", "STRUCTURE"}, {"PATTERN"}),
    "HAS_COMPONENT": ({"LANTERN_TYPE"}, {"STRUCTURE"}),
    "PRECEDES": ({"PROCESS"}, {"PROCESS"}),
    "PRODUCES_EFFECT": ({"PROCESS", "MATERIAL", "PATTERN", "STRUCTURE"}, {"EFFECT"}),
    "HAS_CONDITION": ({"PROCESS", "EFFECT"}, {"CONDITION_PARAMETER"}),
    "PERFORMED_BY": ({"PROCESS"}, {"ACTOR"}),
}

ENTITY_LABELS = ["O"] + [f"{prefix}-{kind}" for kind in ENTITY_TYPES for prefix in ("B", "I")]
RELATION_LABELS = ["NONE"] + RELATION_TYPES
ENTITY_TO_ID = {label: index for index, label in enumerate(ENTITY_LABELS)}
RELATION_TO_ID = {label: index for index, label in enumerate(RELATION_LABELS)}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def make_folds(rows: list[dict[str, Any]], n_splits: int, seed: int) -> list[list[int]]:
    """Stable, source-aware round-robin folds without looking at text or label content."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[str(row["source_group"])].append(index)
    folds = [[] for _ in range(n_splits)]
    rng = random.Random(seed)
    for source in sorted(buckets):
        indices = buckets[source][:]
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            folds[offset % n_splits].append(index)
    for fold in folds:
        fold.sort()
    return folds


def make_source_group_folds(rows: list[dict[str, Any]]) -> list[list[int]]:
    """Leave one complete source group out in each deterministic fold."""
    groups = sorted({str(row["source_group"]) for row in rows})
    return [
        [index for index, row in enumerate(rows) if str(row["source_group"]) == group]
        for group in groups
    ]


def char_bio_labels(row: dict[str, Any]) -> list[int]:
    labels = [ENTITY_TO_ID["O"]] * len(row["text"])
    for entity in sorted(row.get("entities", []), key=lambda item: (item["start"], item["end"])):
        start, end, kind = int(entity["start"]), int(entity["end"]), str(entity["type"])
        if not (0 <= start < end <= len(labels)) or kind not in ENTITY_TYPES:
            continue
        labels[start] = ENTITY_TO_ID[f"B-{kind}"]
        for position in range(start + 1, end):
            labels[position] = ENTITY_TO_ID[f"I-{kind}"]
    return labels


class EntityDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.items = []
        for row in rows:
            chars = list(row["text"])
            char_labels = char_bio_labels(row)
            encoded = tokenizer(
                chars,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            word_ids = encoded.word_ids(batch_index=0)
            labels = []
            previous = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id == previous:
                    labels.append(-100)
                else:
                    labels.append(char_labels[word_id])
                previous = word_id
            self.items.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).squeeze(0),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "word_ids": word_ids,
                    "row": row,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {key: value for key, value in item.items() if key not in {"word_ids", "row"}}


def entity_weights(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    weights = torch.ones(len(ENTITY_LABELS), dtype=torch.float32)
    # Keep B and I equally weighted. Up-weighting the rarer B tags caused the
    # diagnostic model to fragment multi-character Chinese entities.
    weights[ENTITY_TO_ID["O"]] = 0.45
    return weights.to(device)


def train_entity_model(
    train_rows: list[dict[str, Any]],
    tokenizer: Any,
    model_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> Any:
    dataset = EntityDataset(train_rows, tokenizer, max_length)
    generator = torch.Generator().manual_seed(91021)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(ENTITY_LABELS),
        id2label={index: label for index, label in enumerate(ENTITY_LABELS)},
        label2id=ENTITY_TO_ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = max(1, epochs * len(loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
    loss_fn = nn.CrossEntropyLoss(weight=entity_weights(train_rows, device), ignore_index=-100)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits.view(-1, len(ENTITY_LABELS)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return model


def decode_entities(row: dict[str, Any], dataset_item: dict[str, Any], logits: torch.Tensor) -> list[dict[str, Any]]:
    predicted = logits.argmax(dim=-1).tolist()
    char_labels = [ENTITY_TO_ID["O"]] * len(row["text"])
    seen_words = set()
    for token_index, word_id in enumerate(dataset_item["word_ids"]):
        if word_id is None or word_id in seen_words or word_id >= len(char_labels):
            continue
        seen_words.add(word_id)
        char_labels[word_id] = predicted[token_index]
    entities: list[dict[str, Any]] = []
    start = 0
    while start < len(char_labels):
        label = ENTITY_LABELS[char_labels[start]]
        if label == "O":
            start += 1
            continue
        prefix, kind = label.split("-", 1)
        if prefix == "I":
            prefix = "B"
        end = start + 1
        while end < len(char_labels) and ENTITY_LABELS[char_labels[end]] == f"I-{kind}":
            end += 1
        entities.append(
            {
                "id": f"e{len(entities) + 1}",
                "start": start,
                "end": end,
                "text": row["text"][start:end],
                "type": kind,
                "canonical_id": None,
            }
        )
        start = end
    return entities


def predict_entities(model: Any, rows: list[dict[str, Any]], tokenizer: Any, device: torch.device, max_length: int) -> list[list[dict[str, Any]]]:
    dataset = EntityDataset(rows, tokenizer, max_length)
    predictions = []
    model.eval()
    with torch.no_grad():
        for index, row in enumerate(rows):
            tensors = dataset[index]
            inputs = {key: value.unsqueeze(0).to(device) for key, value in tensors.items() if key != "labels"}
            logits = model(**inputs).logits.squeeze(0).cpu()
            predictions.append(decode_entities(row, dataset.items[index], logits))
    return predictions


@dataclass(frozen=True)
class RelationExample:
    text: str
    head: dict[str, Any]
    tail: dict[str, Any]
    label: str


def compatible_relations(head_type: str, tail_type: str) -> list[str]:
    return [
        relation
        for relation, (domains, ranges) in DOMAIN_RANGE.items()
        if head_type in domains and tail_type in ranges
    ]


def build_relation_examples(rows: list[dict[str, Any]], negative_ratio: int, seed: int) -> list[RelationExample]:
    rng = random.Random(seed)
    examples: list[RelationExample] = []
    for row in rows:
        entities = row.get("entities", [])
        relation_lookup = {(rel["head"], rel["tail"]): rel["type"] for rel in row.get("relations", [])}
        positives, negatives = [], []
        for head in entities:
            for tail in entities:
                if head["id"] == tail["id"] or not compatible_relations(head["type"], tail["type"]):
                    continue
                label = relation_lookup.get((head["id"], tail["id"]), "NONE")
                example = RelationExample(row["text"], head, tail, label)
                (positives if label != "NONE" else negatives).append(example)
        rng.shuffle(negatives)
        keep = min(len(negatives), max(4, negative_ratio * max(1, len(positives))))
        examples.extend(positives)
        examples.extend(negatives[:keep])
    rng.shuffle(examples)
    return examples


def relation_text(example: RelationExample) -> str:
    text = example.text
    markers = [
        (int(example.head["end"]), "【主体结束】"),
        (int(example.head["start"]), f"【主体{example.head['type']}】"),
        (int(example.tail["end"]), "【客体结束】"),
        (int(example.tail["start"]), f"【客体{example.tail['type']}】"),
    ]
    # Reverse insertion preserves the original character offsets. At equal
    # positions, end markers are inserted before start markers.
    for position, marker in sorted(markers, key=lambda item: item[0], reverse=True):
        text = text[:position] + marker + text[position:]
    return (
        f"文本：{text} [SEP] 主体：{example.head['text']} [SEP] "
        f"客体：{example.tail['text']}"
    )


class RelationDataset(Dataset):
    def __init__(self, examples: list[RelationExample], tokenizer: Any, max_length: int):
        self.examples = examples
        self.encoded = tokenizer(
            [relation_text(example) for example in examples],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor([RELATION_TO_ID[example.label] for example in examples], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        return item


def relation_weights(examples: list[RelationExample], device: torch.device) -> torch.Tensor:
    counts = Counter(RELATION_TO_ID[example.label] for example in examples)
    weights = torch.ones(len(RELATION_LABELS), dtype=torch.float32)
    weights[0] = 0.20
    positive_counts = [counts[index] for index in range(1, len(RELATION_LABELS)) if counts[index]]
    reference = float(np.median(positive_counts)) if positive_counts else 1.0
    for index in range(1, len(RELATION_LABELS)):
        count = counts.get(index, 0)
        weights[index] = min(4.0, max(0.7, math.sqrt(reference / count))) if count else 0.0
    return weights.to(device)


def train_relation_model(
    train_rows: list[dict[str, Any]],
    tokenizer: Any,
    model_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    negative_ratio: int,
    seed: int,
) -> Any:
    examples = build_relation_examples(train_rows, negative_ratio, seed)
    dataset = RelationDataset(examples, tokenizer, max_length)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(RELATION_LABELS),
        id2label={index: label for index, label in enumerate(RELATION_LABELS)},
        label2id=RELATION_TO_ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = max(1, epochs * len(loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
    loss_fn = nn.CrossEntropyLoss(weight=relation_weights(examples, device))
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return model


def predict_relations(
    model: Any,
    row: dict[str, Any],
    entities: list[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    threshold: float,
) -> list[dict[str, Any]]:
    candidates = []
    allowed_labels = []
    for head in entities:
        for tail in entities:
            if head["id"] == tail["id"]:
                continue
            allowed = compatible_relations(head["type"], tail["type"])
            if allowed:
                candidates.append(RelationExample(row["text"], head, tail, "NONE"))
                allowed_labels.append(allowed)
    if not candidates:
        return []
    dataset = RelationDataset(candidates, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    relations = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits.cpu()
            probabilities = torch.softmax(logits, dim=-1)
            for local_index in range(len(logits)):
                candidate = candidates[offset + local_index]
                allowed = allowed_labels[offset + local_index]
                allowed_ids = [0] + [RELATION_TO_ID[label] for label in allowed]
                scores = probabilities[local_index, allowed_ids]
                best_local = int(scores.argmax())
                best_id = allowed_ids[best_local]
                score = float(scores[best_local])
                if best_id != 0 and score >= threshold:
                    relations.append(
                        {
                            "id": f"r{len(relations) + 1}",
                            "head": candidate.head["id"],
                            "tail": candidate.tail["id"],
                            "type": RELATION_LABELS[best_id],
                            "confidence": score,
                        }
                    )
            offset += len(logits)
    return relations


def evaluate_all(gold_rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    gold_copy = [{**row, "evaluation_partition": "test"} for row in gold_rows]
    pred_copy = [{**row, "evaluation_partition": "test"} for row in predictions]
    return evaluate(gold_copy, pred_copy, partition="test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "e27_ie" / "formal_gold_v1" / "E27_expert_adjudicated_gold_frozen.jsonl",
    )
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "e27_ie" / "supervised_cv_v2",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "e27_ie" / "B0_rule_lexicon_predictions_frozen.jsonl",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--source-group-held-out",
        action="store_true",
        help="Use one complete source group as the validation fold.",
    )
    parser.add_argument("--seed", type=int, default=27021)
    parser.add_argument("--entity-epochs", type=int, default=14)
    parser.add_argument("--relation-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--relation-threshold", type=float, default=0.45)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.gold)
    if len(rows) != 160 or any(row.get("annotation_status") != "frozen" for row in rows):
        raise RuntimeError("The supervised evaluation requires all 160 frozen expert records")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, local_files_only=True)
    folds = (
        make_source_group_folds(rows)
        if args.source_group_held_out
        else make_folds(rows, args.folds, args.seed)
    )
    fold_count = len(folds)
    oof: dict[str, dict[str, Any]] = {}
    fold_reports = []
    for fold_index, validation_indices in enumerate(folds):
        fold_seed = args.seed + fold_index
        set_seed(fold_seed)
        validation_set = set(validation_indices)
        train_rows = [row for index, row in enumerate(rows) if index not in validation_set]
        validation_rows = [rows[index] for index in validation_indices]
        atomic_json(
            args.output_dir / "status.json",
            {
                "state": "training_entity_model",
                "fold": fold_index + 1,
                "folds": fold_count,
                "split_strategy": (
                    "source_group_held_out"
                    if args.source_group_held_out
                    else "record_level_source_stratified"
                ),
                "held_out_source_groups": sorted(
                    {str(row["source_group"]) for row in validation_rows}
                ),
                "train_records": len(train_rows),
                "validation_records": len(validation_rows),
                "updated_at": now(),
            },
        )
        entity_model = train_entity_model(
            train_rows,
            tokenizer,
            args.model_name,
            device,
            args.entity_epochs,
            args.batch_size,
            args.learning_rate,
            args.max_length,
        )
        predicted_entities = predict_entities(entity_model, validation_rows, tokenizer, device, args.max_length)
        del entity_model
        torch.cuda.empty_cache()

        atomic_json(
            args.output_dir / "status.json",
            {
                "state": "training_relation_model",
                "fold": fold_index + 1,
                "folds": fold_count,
                "updated_at": now(),
            },
        )
        relation_model = train_relation_model(
            train_rows,
            tokenizer,
            args.model_name,
            device,
            args.relation_epochs,
            args.batch_size,
            args.learning_rate,
            args.max_length,
            args.negative_ratio,
            fold_seed,
        )
        fold_predictions = []
        gold_endpoint_predictions = []
        for row, entities in zip(validation_rows, predicted_entities):
            relations = predict_relations(
                relation_model,
                row,
                entities,
                tokenizer,
                device,
                args.max_length,
                args.relation_threshold,
            )
            prediction = {
                "segment_id": row["segment_id"],
                "source_group": row["source_group"],
                "evaluation_partition": row["evaluation_partition"],
                "text": row["text"],
                "entities": entities,
                "relations": relations,
                "method_id": "E27v2_supervised_evidence_gated_bert_oof",
                "cv_fold": fold_index + 1,
            }
            fold_predictions.append(prediction)
            oof[row["segment_id"]] = prediction
            gold_entities = [dict(entity) for entity in row.get("entities", [])]
            gold_endpoint_predictions.append(
                {
                    "segment_id": row["segment_id"],
                    "source_group": row["source_group"],
                    "evaluation_partition": row["evaluation_partition"],
                    "text": row["text"],
                    "entities": gold_entities,
                    "relations": predict_relations(
                        relation_model,
                        row,
                        gold_entities,
                        tokenizer,
                        device,
                        args.max_length,
                        args.relation_threshold,
                    ),
                    "method_id": "E27v2_relation_gold_endpoint_diagnostic",
                    "cv_fold": fold_index + 1,
                }
            )
        del relation_model
        torch.cuda.empty_cache()
        fold_report = evaluate_all(validation_rows, fold_predictions)
        gold_endpoint_report = evaluate_all(validation_rows, gold_endpoint_predictions)
        fold_report["fold"] = fold_index + 1
        fold_report["validation_segment_ids"] = [row["segment_id"] for row in validation_rows]
        fold_report["gold_endpoint_relation_diagnostic"] = gold_endpoint_report["relation_end_to_end_micro"]
        fold_reports.append(fold_report)
        atomic_json(args.output_dir / f"fold_{fold_index + 1}_metrics.json", fold_report)
        write_jsonl(args.output_dir / f"fold_{fold_index + 1}_predictions.jsonl", fold_predictions)
        write_jsonl(args.output_dir / f"fold_{fold_index + 1}_gold_endpoint_relations.jsonl", gold_endpoint_predictions)
        atomic_json(
            args.output_dir / "status.json",
            {
                "state": "fold_complete",
                "fold": fold_index + 1,
                "folds": fold_count,
                "entity_f1": fold_report["entity_exact_micro"]["f1"],
                "relation_f1": fold_report["relation_end_to_end_micro"]["f1"],
                "updated_at": now(),
            },
        )

    ordered_predictions = [oof[row["segment_id"]] for row in rows]
    prediction_path = args.output_dir / "E27v2_oof_predictions.jsonl"
    write_jsonl(prediction_path, ordered_predictions)
    overall = evaluate_all(rows, ordered_predictions)
    baseline_predictions = read_jsonl(args.baseline)
    baseline_overall = evaluate_all(rows, baseline_predictions)
    gold_all = [{**row, "evaluation_partition": "test"} for row in rows]
    proposed_all = [{**row, "evaluation_partition": "test"} for row in ordered_predictions]
    baseline_all = [{**row, "evaluation_partition": "test"} for row in baseline_predictions]
    bootstrap = {
        "entity_vs_B0": paired_bootstrap(gold_all, proposed_all, baseline_all, "entity", 5000, args.seed + 91),
        "relation_vs_B0": paired_bootstrap(gold_all, proposed_all, baseline_all, "relation", 5000, args.seed + 92),
    }
    gates = {
        "entity_f1_exceeds_strongest_frozen_baseline": overall["entity_exact_micro"]["f1"] > baseline_overall["entity_exact_micro"]["f1"],
        "relation_f1_exceeds_strongest_frozen_baseline": overall["relation_end_to_end_micro"]["f1"] > baseline_overall["relation_end_to_end_micro"]["f1"],
        "entity_bootstrap_probability_gain_le_zero_below_0_025": bootstrap["entity_vs_B0"]["probability_difference_le_zero"] < 0.025,
        "relation_bootstrap_probability_gain_le_zero_below_0_025": bootstrap["relation_vs_B0"]["probability_difference_le_zero"] < 0.025,
        "grounded_entity_rate_at_least_0_98": overall["audit"]["grounded_entity_rate"] >= 0.98,
        "schema_valid_relation_rate_at_least_0_99": overall["audit"]["schema_valid_relation_rate"] >= 0.99,
    }
    result = {
        "status": "supervised_out_of_fold_evaluation_complete",
        "analysis_role": "post_freeze_recovery_experiment",
        "leakage_control": (
            "Each source group was predicted only by models trained on the other source groups."
            if args.source_group_held_out
            else "Each record was predicted only by models trained on the other folds."
        ),
        "split_strategy": (
            "source_group_held_out"
            if args.source_group_held_out
            else "record_level_source_stratified"
        ),
        "model": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "folds": fold_count,
        "hyperparameters": {
            "entity_epochs": args.entity_epochs,
            "relation_epochs": args.relation_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "negative_ratio": args.negative_ratio,
            "relation_threshold": args.relation_threshold,
        },
        "overall": overall,
        "strongest_frozen_baseline": {"method": "B0_rule_lexicon", "overall": baseline_overall},
        "paired_bootstrap": bootstrap,
        "absolute_target_context": {
            "entity_f1_at_least_0_80": overall["entity_exact_micro"]["f1"] >= 0.80,
            "relation_f1_at_least_0_65": overall["relation_end_to_end_micro"]["f1"] >= 0.65,
            "status": "descriptive_post_hoc_targets_not_in_the_frozen_E27_protocol"
        },
        "fold_reports": fold_reports,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
        "artifacts": {
            "gold_sha256": sha256(args.gold),
            "predictions": str(prediction_path.resolve()),
            "predictions_sha256": sha256(prediction_path),
        },
        "completed_at": now(),
    }
    atomic_json(args.output_dir / "E27v2_supervised_cv_results.json", result)
    atomic_json(
        args.output_dir / "status.json",
        {
            "state": "complete",
            "all_acceptance_gates_passed": all(gates.values()),
            "entity_f1": overall["entity_exact_micro"]["f1"],
            "relation_f1": overall["relation_end_to_end_micro"]["f1"],
            "updated_at": now(),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
