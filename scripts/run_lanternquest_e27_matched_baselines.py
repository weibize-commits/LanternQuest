from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_lanternquest_e27_supervised_cv as core  # noqa: E402
from finalize_lanternquest_e27_results import paired_bootstrap  # noqa: E402


def build_relation_examples_all(
    rows: list[dict[str, Any]], negative_ratio: int, seed: int
) -> list[core.RelationExample]:
    rng = random.Random(seed)
    examples: list[core.RelationExample] = []
    for row in rows:
        entities = row.get("entities", [])
        relation_lookup = {(rel["head"], rel["tail"]): rel["type"] for rel in row.get("relations", [])}
        positives: list[core.RelationExample] = []
        negatives: list[core.RelationExample] = []
        for head in entities:
            for tail in entities:
                if head["id"] == tail["id"]:
                    continue
                label = relation_lookup.get((head["id"], tail["id"]), "NONE")
                example = core.RelationExample(row["text"], head, tail, label)
                (positives if label != "NONE" else negatives).append(example)
        rng.shuffle(negatives)
        keep = min(len(negatives), max(4, negative_ratio * max(1, len(positives))))
        examples.extend(positives)
        examples.extend(negatives[:keep])
    rng.shuffle(examples)
    return examples


def relation_weights(examples: list[core.RelationExample], device: torch.device) -> torch.Tensor:
    counts = Counter(core.RELATION_TO_ID[example.label] for example in examples)
    weights = torch.ones(len(core.RELATION_LABELS), dtype=torch.float32)
    weights[0] = 0.20
    positive_counts = [counts[index] for index in range(1, len(core.RELATION_LABELS)) if counts[index]]
    reference = float(np.median(positive_counts)) if positive_counts else 1.0
    for index in range(1, len(core.RELATION_LABELS)):
        count = counts.get(index, 0)
        weights[index] = min(4.0, max(0.7, math.sqrt(reference / count))) if count else 0.0
    return weights.to(device)


def train_relation_model_all(
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
    examples = build_relation_examples_all(train_rows, negative_ratio, seed)
    dataset = core.RelationDataset(examples, tokenizer, max_length)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(core.RELATION_LABELS),
        id2label={index: label for index, label in enumerate(core.RELATION_LABELS)},
        label2id=core.RELATION_TO_ID,
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


def relation_record(
    relation_id: int,
    candidate: core.RelationExample,
    label_id: int,
    score: float,
) -> dict[str, Any]:
    return {
        "id": f"r{relation_id}",
        "head": candidate.head["id"],
        "tail": candidate.tail["id"],
        "type": core.RELATION_LABELS[label_id],
        "confidence": score,
    }


def evidence_entities(row: dict[str, Any], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for entity in entities:
        start, end = int(entity["start"]), int(entity["end"])
        if 0 <= start < end <= len(row["text"]) and row["text"][start:end] == entity["text"]:
            kept.append(dict(entity))
    return kept


def predict_variants(
    model: Any,
    row: dict[str, Any],
    entities: list[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    candidates = [
        core.RelationExample(row["text"], head, tail, "NONE")
        for head in entities
        for tail in entities
        if head["id"] != tail["id"]
    ]
    unrestricted: list[dict[str, Any]] = []
    ontology: list[dict[str, Any]] = []
    if candidates:
        dataset = core.RelationDataset(candidates, tokenizer, max_length)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        offset = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                logits = model(**{key: value.to(device) for key, value in batch.items()}).logits.cpu()
                probabilities = torch.softmax(logits, dim=-1)
                for local_index in range(len(logits)):
                    candidate = candidates[offset + local_index]
                    all_scores = probabilities[local_index]
                    best_all = int(all_scores.argmax())
                    best_all_score = float(all_scores[best_all])
                    if best_all and best_all_score >= threshold:
                        unrestricted.append(
                            relation_record(len(unrestricted) + 1, candidate, best_all, best_all_score)
                        )

                    allowed = core.compatible_relations(candidate.head["type"], candidate.tail["type"])
                    allowed_ids = [0] + [core.RELATION_TO_ID[label] for label in allowed]
                    allowed_scores = all_scores[allowed_ids]
                    best_local = int(allowed_scores.argmax())
                    best_allowed = allowed_ids[best_local]
                    best_allowed_score = float(allowed_scores[best_local])
                    if best_allowed and best_allowed_score >= threshold:
                        ontology.append(
                            relation_record(len(ontology) + 1, candidate, best_allowed, best_allowed_score)
                        )
                offset += len(logits)

    grounded = evidence_entities(row, entities)
    grounded_ids = {entity["id"] for entity in grounded}
    evidence_relations = []
    for relation in ontology:
        if relation["head"] in grounded_ids and relation["tail"] in grounded_ids:
            evidence_relations.append({**relation, "id": f"r{len(evidence_relations) + 1}"})
    base = {
        "segment_id": row["segment_id"],
        "source_group": row["source_group"],
        "evaluation_partition": row["evaluation_partition"],
        "text": row["text"],
    }
    return {
        "neural": {**base, "entities": [dict(x) for x in entities], "relations": unrestricted},
        "ontology": {**base, "entities": [dict(x) for x in entities], "relations": ontology},
        "evidence": {**base, "entities": grounded, "relations": evidence_relations},
    }


def evaluate_all(gold_rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    return core.evaluate_all(gold_rows, predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "artifacts/benchmarks/e27_ie/formal_gold_v1/E27_expert_adjudicated_gold_frozen.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "artifacts/benchmarks/e27_ie/B0_rule_lexicon_predictions_frozen.jsonl",
    )
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--model-id", default="BERT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=27021)
    parser.add_argument("--entity-epochs", type=int, default=14)
    parser.add_argument("--relation-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--negative-ratio", type=int, default=2)
    parser.add_argument("--relation-threshold", type=float, default=0.30)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = core.read_jsonl(args.gold)
    if len(rows) != 160 or any(row.get("annotation_status") != "frozen" for row in rows):
        raise RuntimeError("E27v3 requires all 160 frozen expert records")
    core.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, local_files_only=args.local_files_only
    )
    folds = core.make_folds(rows, args.folds, args.seed)
    oof: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in ("neural", "ontology", "evidence")}
    fold_reports = []

    for fold_index, validation_indices in enumerate(folds):
        fold_seed = args.seed + fold_index
        core.set_seed(fold_seed)
        validation_set = set(validation_indices)
        train_rows = [row for index, row in enumerate(rows) if index not in validation_set]
        validation_rows = [rows[index] for index in validation_indices]
        core.atomic_json(
            args.output_dir / "status.json",
            {
                "state": "training_entity_model",
                "model_id": args.model_id,
                "fold": fold_index + 1,
                "folds": args.folds,
                "updated_at": core.now(),
            },
        )
        entity_model = core.train_entity_model(
            train_rows,
            tokenizer,
            args.model_name,
            device,
            args.entity_epochs,
            args.batch_size,
            args.learning_rate,
            args.max_length,
        )
        predicted_entities = core.predict_entities(
            entity_model, validation_rows, tokenizer, device, args.max_length
        )
        del entity_model
        torch.cuda.empty_cache()

        core.atomic_json(
            args.output_dir / "status.json",
            {
                "state": "training_relation_model",
                "model_id": args.model_id,
                "fold": fold_index + 1,
                "folds": args.folds,
                "updated_at": core.now(),
            },
        )
        relation_model = train_relation_model_all(
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
        fold_predictions = {name: [] for name in oof}
        for row, entities in zip(validation_rows, predicted_entities):
            variants = predict_variants(
                relation_model,
                row,
                entities,
                tokenizer,
                device,
                args.max_length,
                args.relation_threshold,
            )
            for name, prediction in variants.items():
                prediction["method_id"] = f"{args.model_id}_{name}"
                prediction["cv_fold"] = fold_index + 1
                fold_predictions[name].append(prediction)
                oof[name][row["segment_id"]] = prediction
        del relation_model
        torch.cuda.empty_cache()

        report = {name: evaluate_all(validation_rows, predictions) for name, predictions in fold_predictions.items()}
        fold_reports.append({"fold": fold_index + 1, "segment_count": len(validation_rows), "methods": report})
        core.atomic_json(args.output_dir / f"fold_{fold_index + 1}_metrics.json", fold_reports[-1])
        for name, predictions in fold_predictions.items():
            core.write_jsonl(args.output_dir / f"fold_{fold_index + 1}_{name}_predictions.jsonl", predictions)
        core.atomic_json(
            args.output_dir / "status.json",
            {
                "state": "fold_complete",
                "model_id": args.model_id,
                "fold": fold_index + 1,
                "folds": args.folds,
                "entity_f1": report["evidence"]["entity_exact_micro"]["f1"],
                "relation_f1": report["evidence"]["relation_end_to_end_micro"]["f1"],
                "updated_at": core.now(),
            },
        )

    ordered = {name: [method[row["segment_id"]] for row in rows] for name, method in oof.items()}
    for name, predictions in ordered.items():
        core.write_jsonl(args.output_dir / f"{args.model_id}_{name}_oof_predictions.jsonl", predictions)
    overall = {name: evaluate_all(rows, predictions) for name, predictions in ordered.items()}
    baseline_predictions = core.read_jsonl(args.baseline)
    baseline = evaluate_all(rows, baseline_predictions)
    gold_all = [{**row, "evaluation_partition": "test"} for row in rows]
    baseline_all = [{**row, "evaluation_partition": "test"} for row in baseline_predictions]
    bootstrap = {}
    for name, predictions in ordered.items():
        pred_all = [{**row, "evaluation_partition": "test"} for row in predictions]
        bootstrap[name] = {
            "entity_vs_B0": paired_bootstrap(gold_all, pred_all, baseline_all, "entity", 5000, args.seed + 101),
            "relation_vs_B0": paired_bootstrap(gold_all, pred_all, baseline_all, "relation", 5000, args.seed + 102),
        }
    result = {
        "status": "matched_baseline_evaluation_complete",
        "analysis_role": "post_freeze_extended_comparison",
        "model_id": args.model_id,
        "model_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "folds": args.folds,
        "hyperparameters": {
            "entity_epochs": args.entity_epochs,
            "relation_epochs": args.relation_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "negative_ratio": args.negative_ratio,
            "relation_threshold": args.relation_threshold,
        },
        "baseline_B0": baseline,
        "methods": overall,
        "paired_bootstrap": bootstrap,
        "fold_reports": fold_reports,
        "completed_at": core.now(),
    }
    core.atomic_json(args.output_dir / f"{args.model_id}_matched_results.json", result)
    core.atomic_json(
        args.output_dir / "status.json",
        {
            "state": "complete",
            "model_id": args.model_id,
            "methods": {
                name: {
                    "entity_f1": metrics["entity_exact_micro"]["f1"],
                    "relation_f1": metrics["relation_end_to_end_micro"]["f1"],
                }
                for name, metrics in overall.items()
            },
            "updated_at": core.now(),
        },
    )
    print(json.dumps(json.loads((args.output_dir / "status.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
