from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LEXICON = {
    "LANTERN_TYPE": ["万眼萝灯", "万眼萝", "万眼箩灯", "万眼箩", "五谷丰登灯", "五谷彩灯", "无骨灯笼", "木版年画灯", "木版灯画灯", "宫灯", "纱灯", "提灯", "滚地灯"],
    "PROCESS": ["裁纸", "裁出", "绘制", "描画", "装订", "穿刺", "针刺", "定位", "勾勒", "上色", "调和", "涂胶", "粘接", "折叠", "正折", "反折", "压边", "合缝", "固定", "组装", "点亮", "数字呈现", "三维采集", "打印"],
    "MATERIAL": ["纸", "卷纸", "竹篾", "绢纱", "玻璃", "铁丝", "丙烯颜料", "颜料", "清水", "胶水", "植物颜料"],
    "TOOL": ["蜡板", "剪刀", "锥针", "追针", "铅笔", "尺子", "订书机", "镊子", "打印机"],
    "PATTERN": ["纹样", "图案", "花纹", "木版水印", "灯画印版", "京剧脸谱", "佛教八宝"],
    "STRUCTURE": ["灯片", "灯面", "骨架", "灯沿", "底托", "凹面", "凸面", "针孔", "灯孔", "四边形", "梯形", "弧线", "灯体"],
    "EFFECT": ["透光", "散射", "柔和", "明亮", "晶莹剔透", "凹凸有致", "色泽鲜艳", "均匀", "精细", "整体效果"],
    "CONDITION_PARAMETER": ["六片", "6片", "四层", "相同", "不同", "一张一张", "半张A4纸", "大小相同", "完全一致", "成千上万个"],
    "ACTOR": ["张俊涛", "张俊丽", "张金汉", "传承人", "灯笼艺人", "汴京灯笼张博物馆"],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_entities(text: str) -> list[dict[str, Any]]:
    candidates = []
    for entity_type, terms in LEXICON.items():
        for term in terms:
            start = 0
            while True:
                index = text.find(term, start)
                if index < 0:
                    break
                candidates.append((index, index + len(term), entity_type, term))
                start = index + 1
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    selected = []
    occupied: list[tuple[int, int]] = []
    for start, end, entity_type, term in candidates:
        if any(start < old_end and old_start < end for old_start, old_end in occupied):
            continue
        occupied.append((start, end))
        selected.append({"id": f"e{len(selected) + 1}", "start": start, "end": end, "text": term, "type": entity_type, "canonical_id": None})
    return sorted(selected, key=lambda entity: (entity["start"], entity["end"]))


def add_relation(relations: list[dict[str, Any]], head: dict[str, Any], tail: dict[str, Any], relation_type: str) -> None:
    key = (head["id"], tail["id"], relation_type)
    if any((r["head"], r["tail"], r["type"]) == key for r in relations):
        return
    relations.append({"id": f"r{len(relations) + 1}", "head": head["id"], "tail": tail["id"], "type": relation_type})


def extract_relations(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        by_type.setdefault(entity["type"], []).append(entity)
    relations: list[dict[str, Any]] = []
    processes = by_type.get("PROCESS", [])
    for process in processes:
        for material in by_type.get("MATERIAL", []):
            if abs(process["start"] - material["start"]) <= 45:
                add_relation(relations, process, material, "USES_MATERIAL")
        for tool in by_type.get("TOOL", []):
            if abs(process["start"] - tool["start"]) <= 45:
                add_relation(relations, process, tool, "USES_TOOL")
        for condition in by_type.get("CONDITION_PARAMETER", []):
            if abs(process["start"] - condition["start"]) <= 45:
                add_relation(relations, process, condition, "HAS_CONDITION")
        for effect in by_type.get("EFFECT", []):
            if abs(process["start"] - effect["start"]) <= 60 and any(cue in text for cue in ("使", "让", "形成", "显得", "效果", "为了")):
                add_relation(relations, process, effect, "PRODUCES_EFFECT")
    for lantern in by_type.get("LANTERN_TYPE", []):
        for structure in by_type.get("STRUCTURE", []):
            if abs(lantern["start"] - structure["start"]) <= 60 and any(cue in text for cue in ("由", "组成", "分成", "没有", "骨架")):
                add_relation(relations, lantern, structure, "HAS_COMPONENT")
        for pattern in by_type.get("PATTERN", []):
            if abs(lantern["start"] - pattern["start"]) <= 60:
                add_relation(relations, lantern, pattern, "HAS_PATTERN")
    if any(cue in text for cue in ("首先", "之后", "然后", "再", "最后")):
        ordered = sorted(processes, key=lambda entity: entity["start"])
        for first, second in zip(ordered, ordered[1:]):
            add_relation(relations, first, second, "PRECEDES")
    return relations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    predictions = []
    for row in rows:
        entities = extract_entities(row["text"])
        predictions.append(
            {
                "segment_id": row["segment_id"],
                "source_group": row["source_group"],
                "evaluation_partition": row.get("evaluation_partition", "test"),
                "text": row["text"],
                "entities": entities,
                "relations": extract_relations(row["text"], entities),
                "method_id": "B0_rule_lexicon_frozen_v1",
            }
        )
    dump_jsonl(args.out, predictions)
    manifest = {
        "experiment_id": "E27",
        "method_id": "B0_rule_lexicon_frozen_v1",
        "status": "predictions_frozen_before_gold",
        "record_count": len(predictions),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "lexicon_sha256": hashlib.sha256(json.dumps(LEXICON, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
