from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_lanternquest_e27_supervised_cv as core  # noqa: E402
from finalize_lanternquest_e27_results import paired_bootstrap  # noqa: E402
from run_lanternquest_e27_matched_baselines import (  # noqa: E402
    build_relation_examples_all,
    evidence_entities,
    relation_record,
    relation_weights,
)


class JointExtractor(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        dropout = float(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.entity_dropout = nn.Dropout(dropout)
        self.entity_classifier = nn.Linear(hidden, len(core.ENTITY_LABELS))
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(core.RELATION_LABELS)),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        entity_logits = self.entity_classifier(self.entity_dropout(hidden))
        return hidden, entity_logits

    def relation_logits(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        features = torch.cat([head, tail, torch.abs(head - tail), head * tail], dim=-1)
        return self.relation_classifier(features)


def first_token_by_char(word_ids: list[int | None]) -> dict[int, int]:
    result: dict[int, int] = {}
    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in result:
            result[int(word_id)] = token_index
    return result


def span_vector(
    hidden: torch.Tensor,
    char_to_token: dict[int, int],
    start: int,
    end: int,
) -> torch.Tensor | None:
    token_indices = [char_to_token[pos] for pos in range(start, end) if pos in char_to_token]
    if not token_indices:
        return None
    return hidden[token_indices].mean(dim=0)


def train_joint_model(
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
    relation_loss_weight: float,
) -> JointExtractor:
    entity_dataset = core.EntityDataset(train_rows, tokenizer, max_length)
    pair_examples = []
    for row_index, row in enumerate(train_rows):
        examples = build_relation_examples_all([row], negative_ratio, seed + row_index)
        pair_examples.append(examples)
    flat_examples = [example for examples in pair_examples for example in examples]
    model = JointExtractor(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = max(1, (len(train_rows) + batch_size - 1) // batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
    entity_loss_fn = nn.CrossEntropyLoss(
        weight=core.entity_weights(train_rows, device), ignore_index=-100
    )
    relation_loss_fn = nn.CrossEntropyLoss(weight=relation_weights(flat_examples, device))
    rng = random.Random(seed)
    model.train()
    for _ in range(epochs):
        indices = list(range(len(train_rows)))
        rng.shuffle(indices)
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            input_ids = torch.stack([entity_dataset.items[i]["input_ids"] for i in batch_indices]).to(device)
            attention_mask = torch.stack([entity_dataset.items[i]["attention_mask"] for i in batch_indices]).to(device)
            token_type_ids = torch.stack([entity_dataset.items[i]["token_type_ids"] for i in batch_indices]).to(device)
            labels = torch.stack([entity_dataset.items[i]["labels"] for i in batch_indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden, entity_logits = model(input_ids, attention_mask, token_type_ids)
            entity_loss = entity_loss_fn(entity_logits.view(-1, len(core.ENTITY_LABELS)), labels.view(-1))

            head_vectors = []
            tail_vectors = []
            relation_labels = []
            for local_index, row_index in enumerate(batch_indices):
                char_to_token = first_token_by_char(entity_dataset.items[row_index]["word_ids"])
                for example in pair_examples[row_index]:
                    head = span_vector(
                        hidden[local_index],
                        char_to_token,
                        int(example.head["start"]),
                        int(example.head["end"]),
                    )
                    tail = span_vector(
                        hidden[local_index],
                        char_to_token,
                        int(example.tail["start"]),
                        int(example.tail["end"]),
                    )
                    if head is None or tail is None:
                        continue
                    head_vectors.append(head)
                    tail_vectors.append(tail)
                    relation_labels.append(core.RELATION_TO_ID[example.label])
            if head_vectors:
                relation_logits = model.relation_logits(
                    torch.stack(head_vectors), torch.stack(tail_vectors)
                )
                relation_targets = torch.tensor(relation_labels, dtype=torch.long, device=device)
                relation_loss = relation_loss_fn(relation_logits, relation_targets)
            else:
                relation_loss = entity_loss.new_zeros(())
            loss = entity_loss + relation_loss_weight * relation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return model


def predict_row_variants(
    model: JointExtractor,
    row: dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    threshold: float,
    supplied_entities: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    dataset = core.EntityDataset([row], tokenizer, max_length)
    item = dataset.items[0]
    inputs = {
        key: item[key].unsqueeze(0).to(device)
        for key in ("input_ids", "attention_mask", "token_type_ids")
    }
    model.eval()
    with torch.no_grad():
        hidden, entity_logits = model(**inputs)
    entities = (
        supplied_entities
        if supplied_entities is not None
        else core.decode_entities(row, item, entity_logits.squeeze(0).cpu())
    )
    char_to_token = first_token_by_char(item["word_ids"])
    candidates = [
        core.RelationExample(row["text"], head, tail, "NONE")
        for head in entities
        for tail in entities
        if head["id"] != tail["id"]
    ]
    unrestricted = []
    ontology = []
    if candidates:
        heads, tails, usable = [], [], []
        for candidate in candidates:
            head = span_vector(
                hidden[0], char_to_token, int(candidate.head["start"]), int(candidate.head["end"])
            )
            tail = span_vector(
                hidden[0], char_to_token, int(candidate.tail["start"]), int(candidate.tail["end"])
            )
            if head is not None and tail is not None:
                heads.append(head)
                tails.append(tail)
                usable.append(candidate)
        if usable:
            with torch.no_grad():
                probabilities = torch.softmax(
                    model.relation_logits(torch.stack(heads), torch.stack(tails)), dim=-1
                ).cpu()
            for candidate, scores in zip(usable, probabilities):
                best_all = int(scores.argmax())
                best_all_score = float(scores[best_all])
                if best_all and best_all_score >= threshold:
                    unrestricted.append(
                        relation_record(len(unrestricted) + 1, candidate, best_all, best_all_score)
                    )
                allowed = core.compatible_relations(candidate.head["type"], candidate.tail["type"])
                allowed_ids = [0] + [core.RELATION_TO_ID[label] for label in allowed]
                allowed_scores = scores[allowed_ids]
                best_local = int(allowed_scores.argmax())
                best_allowed = allowed_ids[best_local]
                best_allowed_score = float(allowed_scores[best_local])
                if best_allowed and best_allowed_score >= threshold:
                    ontology.append(
                        relation_record(len(ontology) + 1, candidate, best_allowed, best_allowed_score)
                    )

    grounded = evidence_entities(row, entities)
    grounded_ids = {entity["id"] for entity in grounded}
    evidence_relations = [
        {**relation, "id": f"r{index + 1}"}
        for index, relation in enumerate(ontology)
        if relation["head"] in grounded_ids and relation["tail"] in grounded_ids
    ]
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
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--source-group-held-out",
        action="store_true",
        help="Use one complete source group as the validation fold.",
    )
    parser.add_argument("--seed", type=int, default=27021)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--negative-ratio", type=int, default=2)
    parser.add_argument("--relation-threshold", type=float, default=0.30)
    parser.add_argument("--relation-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = core.read_jsonl(args.gold)
    if len(rows) != 160 or any(row.get("annotation_status") != "frozen" for row in rows):
        raise RuntimeError("Joint E27v3 evaluation requires 160 frozen expert records")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, local_files_only=True)
    folds = (
        core.make_source_group_folds(rows)
        if args.source_group_held_out
        else core.make_folds(rows, args.folds, args.seed)
    )
    fold_count = len(folds)
    oof = {name: {} for name in ("neural", "ontology", "evidence")}
    gold_endpoint_oof = {name: {} for name in ("neural", "ontology", "evidence")}
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
                "state": "training_joint_model",
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
                "updated_at": core.now(),
            },
        )
        model = train_joint_model(
            train_rows,
            tokenizer,
            args.model_name,
            device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.max_length,
            args.negative_ratio,
            fold_seed,
            args.relation_loss_weight,
        )
        fold_predictions = {name: [] for name in oof}
        fold_gold_endpoints = {name: [] for name in oof}
        for row in validation_rows:
            variants = predict_row_variants(
                model, row, tokenizer, device, args.max_length, args.relation_threshold
            )
            oracle_variants = predict_row_variants(
                model,
                row,
                tokenizer,
                device,
                args.max_length,
                args.relation_threshold,
                supplied_entities=[dict(entity) for entity in row.get("entities", [])],
            )
            for name in oof:
                variants[name]["method_id"] = f"JOINT_SHARED_BERT_{name}"
                variants[name]["cv_fold"] = fold_index + 1
                oracle_variants[name]["method_id"] = f"JOINT_SHARED_BERT_{name}_gold_endpoints"
                oracle_variants[name]["cv_fold"] = fold_index + 1
                fold_predictions[name].append(variants[name])
                fold_gold_endpoints[name].append(oracle_variants[name])
                oof[name][row["segment_id"]] = variants[name]
                gold_endpoint_oof[name][row["segment_id"]] = oracle_variants[name]
        del model
        torch.cuda.empty_cache()
        report = {
            name: {
                "end_to_end": core.evaluate_all(validation_rows, fold_predictions[name]),
                "gold_endpoints": core.evaluate_all(validation_rows, fold_gold_endpoints[name]),
            }
            for name in oof
        }
        fold_reports.append({"fold": fold_index + 1, "segment_count": len(validation_rows), "methods": report})
        core.atomic_json(args.output_dir / f"fold_{fold_index + 1}_metrics.json", fold_reports[-1])
        for name in oof:
            core.write_jsonl(
                args.output_dir / f"fold_{fold_index + 1}_{name}_predictions.jsonl",
                fold_predictions[name],
            )
            core.write_jsonl(
                args.output_dir / f"fold_{fold_index + 1}_{name}_gold_endpoints.jsonl",
                fold_gold_endpoints[name],
            )

    ordered = {name: [predictions[row["segment_id"]] for row in rows] for name, predictions in oof.items()}
    oracle_ordered = {
        name: [predictions[row["segment_id"]] for row in rows]
        for name, predictions in gold_endpoint_oof.items()
    }
    for name in oof:
        core.write_jsonl(args.output_dir / f"JOINT_{name}_oof_predictions.jsonl", ordered[name])
        core.write_jsonl(args.output_dir / f"JOINT_{name}_gold_endpoints.jsonl", oracle_ordered[name])
    methods = {
        name: {
            "end_to_end": core.evaluate_all(rows, ordered[name]),
            "gold_endpoints": core.evaluate_all(rows, oracle_ordered[name]),
        }
        for name in oof
    }
    baseline_predictions = core.read_jsonl(args.baseline)
    baseline = core.evaluate_all(rows, baseline_predictions)
    gold_all = [{**row, "evaluation_partition": "test"} for row in rows]
    baseline_all = [{**row, "evaluation_partition": "test"} for row in baseline_predictions]
    bootstrap = {}
    for name, predictions in ordered.items():
        pred_all = [{**row, "evaluation_partition": "test"} for row in predictions]
        bootstrap[name] = {
            "entity_vs_B0": paired_bootstrap(gold_all, pred_all, baseline_all, "entity", 5000, args.seed + 201),
            "relation_vs_B0": paired_bootstrap(gold_all, pred_all, baseline_all, "relation", 5000, args.seed + 202),
        }
    result = {
        "status": "joint_shared_encoder_evaluation_complete",
        "analysis_role": "post_freeze_extended_comparison",
        "model": "shared bert encoder with BIO and ordered-pair heads",
        "device": str(device),
        "seed": args.seed,
        "folds": fold_count,
        "split_strategy": (
            "source_group_held_out"
            if args.source_group_held_out
            else "record_level_source_stratified"
        ),
        "hyperparameters": vars(args) | {"gold": str(args.gold), "baseline": str(args.baseline), "output_dir": str(args.output_dir)},
        "baseline_B0": baseline,
        "methods": methods,
        "paired_bootstrap": bootstrap,
        "fold_reports": fold_reports,
        "completed_at": core.now(),
    }
    core.atomic_json(args.output_dir / "JOINT_SHARED_BERT_results.json", result)
    summary = {
        "state": "complete",
        "methods": {
            name: {
                "entity_f1": report["end_to_end"]["entity_exact_micro"]["f1"],
                "relation_f1": report["end_to_end"]["relation_end_to_end_micro"]["f1"],
                "gold_endpoint_relation_f1": report["gold_endpoints"]["relation_end_to_end_micro"]["f1"],
            }
            for name, report in methods.items()
        },
        "updated_at": core.now(),
    }
    core.atomic_json(args.output_dir / "status.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
