from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ENTITY_TYPES = {
    "LANTERN_TYPE", "PROCESS", "MATERIAL", "TOOL", "PATTERN",
    "STRUCTURE", "EFFECT", "CONDITION_PARAMETER", "ACTOR"
}
RELATION_TYPES = {
    "USES_MATERIAL", "USES_TOOL", "APPLIES_TO", "HAS_PATTERN",
    "HAS_COMPONENT", "PRECEDES", "PRODUCES_EFFECT", "HAS_CONDITION", "PERFORMED_BY"
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def entity_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    return int(entity["start"]), int(entity["end"]), str(entity["type"])


def relation_key(record: dict[str, Any], relation: dict[str, Any]) -> tuple[Any, ...]:
    index = {entity["id"]: entity for entity in record.get("entities", [])}
    head = index[relation["head"]]
    tail = index[relation["tail"]]
    return entity_key(head) + (str(relation["type"]),) + entity_key(tail)


def overlap_match(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> int:
    candidates = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g["type"] != p["type"]:
                continue
            overlap = max(0, min(g["end"], p["end"]) - max(g["start"], p["start"]))
            if overlap:
                candidates.append((overlap, gi, pi))
    used_g, used_p, matches = set(), set(), 0
    for _, gi, pi in sorted(candidates, reverse=True):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matches += 1
    return matches


def validate_record(record: dict[str, Any]) -> Counter:
    errors: Counter = Counter()
    text = record.get("text", "")
    entities = record.get("entities", [])
    ids = [entity.get("id") for entity in entities]
    if len(ids) != len(set(ids)):
        errors["duplicate_entity_id"] += 1
    entity_index = {entity.get("id"): entity for entity in entities}
    for entity in entities:
        if entity.get("type") not in ENTITY_TYPES:
            errors["invalid_entity_type"] += 1
        start, end = entity.get("start"), entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
            errors["invalid_span"] += 1
        elif text[start:end] != entity.get("text"):
            errors["ungrounded_entity"] += 1
    for relation in record.get("relations", []):
        if relation.get("type") not in RELATION_TYPES:
            errors["invalid_relation_type"] += 1
        if relation.get("head") not in entity_index or relation.get("tail") not in entity_index:
            errors["dangling_relation"] += 1
    return errors


def multiset_counts(gold: Iterable[tuple], pred: Iterable[tuple]) -> tuple[int, int, int]:
    gold_counter, pred_counter = Counter(gold), Counter(pred)
    tp = sum((gold_counter & pred_counter).values())
    return tp, sum(pred_counter.values()) - tp, sum(gold_counter.values()) - tp


def evaluate(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], partition: str = "test") -> dict[str, Any]:
    gold_rows = [r for r in gold_rows if r.get("evaluation_partition", "test") == partition]
    unfinished = [r["segment_id"] for r in gold_rows if r.get("annotation_status") not in {"adjudicated", "frozen"}]
    if unfinished:
        raise RuntimeError(
            f"Formal scoring refused: {len(unfinished)} gold records are not adjudicated/frozen. "
            f"First pending record: {unfinished[0]}"
        )
    gold_by_id = {r["segment_id"]: r for r in gold_rows}
    pred_by_id = {r["segment_id"]: r for r in pred_rows if r["segment_id"] in gold_by_id}
    if set(gold_by_id) != set(pred_by_id):
        missing = sorted(set(gold_by_id) - set(pred_by_id))
        raise ValueError(f"Prediction coverage mismatch. missing={missing[:5]}")

    entity_totals = [0, 0, 0]
    relation_totals = [0, 0, 0]
    overlap_tp = 0
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_group: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"entity": [0, 0, 0], "relation": [0, 0, 0]})
    validation_errors: Counter = Counter()
    predicted_entity_count = 0
    predicted_relation_count = 0

    for segment_id, gold in gold_by_id.items():
        pred = pred_by_id[segment_id]
        if pred.get("text") != gold.get("text"):
            raise ValueError(f"Text mismatch for {segment_id}")
        validation_errors.update(validate_record(pred))
        predicted_entity_count += len(pred.get("entities", []))
        predicted_relation_count += len(pred.get("relations", []))

        g_entities = gold.get("entities", [])
        p_entities = pred.get("entities", [])
        ec = multiset_counts(map(entity_key, g_entities), map(entity_key, p_entities))
        overlap_tp += overlap_match(g_entities, p_entities)
        for i in range(3):
            entity_totals[i] += ec[i]
            per_group[gold["source_group"]]["entity"][i] += ec[i]
        for entity_type in ENTITY_TYPES:
            tc = multiset_counts(
                (entity_key(e) for e in g_entities if e["type"] == entity_type),
                (entity_key(e) for e in p_entities if e["type"] == entity_type),
            )
            for i in range(3):
                per_type[entity_type][i] += tc[i]

        try:
            g_relations = [relation_key(gold, r) for r in gold.get("relations", [])]
        except KeyError as exc:
            raise ValueError(f"Gold relation has a dangling endpoint in {segment_id}: {exc}") from exc
        p_relations = []
        for relation in pred.get("relations", []):
            try:
                p_relations.append(relation_key(pred, relation))
            except KeyError:
                continue
        rc = multiset_counts(g_relations, p_relations)
        for i in range(3):
            relation_totals[i] += rc[i]
            per_group[gold["source_group"]]["relation"][i] += rc[i]

    overlap_fp = predicted_entity_count - overlap_tp
    overlap_fn = sum(len(r.get("entities", [])) for r in gold_rows) - overlap_tp
    grounded_failures = validation_errors["invalid_span"] + validation_errors["ungrounded_entity"]
    invalid_relations = validation_errors["invalid_relation_type"] + validation_errors["dangling_relation"]
    return {
        "status": "formal_evaluation_complete",
        "evaluation_partition": partition,
        "segment_count": len(gold_rows),
        "entity_exact_micro": prf(*entity_totals),
        "entity_overlap_micro": prf(overlap_tp, overlap_fp, overlap_fn),
        "relation_end_to_end_micro": prf(*relation_totals),
        "triple_end_to_end_micro": prf(*relation_totals),
        "entity_by_type": {k: prf(*v) for k, v in sorted(per_type.items())},
        "by_source_group": {
            group: {metric: prf(*counts) for metric, counts in metrics.items()}
            for group, metrics in sorted(per_group.items())
        },
        "audit": {
            "validation_errors": dict(validation_errors),
            "grounded_entity_rate": 1.0 - grounded_failures / predicted_entity_count if predicted_entity_count else 1.0,
            "schema_valid_relation_rate": 1.0 - invalid_relations / predicted_relation_count if predicted_relation_count else 1.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partition", choices=["development", "test"], default="test")
    args = parser.parse_args()
    result = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions), partition=args.partition)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
