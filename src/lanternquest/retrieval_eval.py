import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from lanternquest.schema import (
    RetrievalCaseResult,
    RetrievalDataset,
    RetrievalEvaluation,
)
from lanternquest.text_index import search_text_index


def load_retrieval_dataset(path: Path) -> RetrievalDataset:
    return RetrievalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def index_source_ids(database_path: Path) -> set[str]:
    connection = sqlite3.connect(database_path)
    try:
        return {row[0] for row in connection.execute("SELECT source_id FROM sources")}
    finally:
        connection.close()


def evaluate_retrieval(
    database_path: Path,
    dataset: RetrievalDataset,
    top_k: int = 5,
) -> RetrievalEvaluation:
    available_sources = index_source_ids(database_path)
    referenced_sources = {
        source_id for case in dataset.cases for source_id in case.acceptable_source_ids
    }
    missing_sources = sorted(referenced_sources - available_sources)
    if missing_sources:
        raise ValueError(
            "Evaluation dataset references sources absent from the text index: "
            + ", ".join(missing_sources)
        )

    case_results: list[RetrievalCaseResult] = []
    reciprocal_ranks: list[float] = []
    for case in dataset.cases:
        raw_results = search_text_index(database_path, case.query, limit=max(top_k * 10, 50))
        ranked_source_ids: list[str] = []
        for result in raw_results:
            source_id = str(result["source_id"])
            if source_id not in ranked_source_ids:
                ranked_source_ids.append(source_id)
            if len(ranked_source_ids) == top_k:
                break

        acceptable = set(case.acceptable_source_ids)
        first_relevant_rank = next(
            (
                rank
                for rank, source_id in enumerate(ranked_source_ids, start=1)
                if source_id in acceptable
            ),
            None,
        )
        reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        reciprocal_ranks.append(reciprocal_rank)
        retrieved_relevant = len(set(ranked_source_ids) & acceptable)
        recall = retrieved_relevant / len(acceptable)
        case_results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                query=case.query,
                ranked_source_ids=ranked_source_ids,
                first_relevant_rank=first_relevant_rank,
                hit_at_k=first_relevant_rank is not None,
                recall_at_k=recall,
            )
        )

    case_count = len(case_results)
    evaluation_seed = (
        f"{dataset.dataset_id}:{database_path}:{top_k}:{datetime.now(timezone.utc).isoformat()}"
    )
    return RetrievalEvaluation(
        evaluation_id=f"eval_{hashlib.sha256(evaluation_seed.encode()).hexdigest()[:16]}",
        generated_at=datetime.now(timezone.utc),
        dataset_id=dataset.dataset_id,
        split=dataset.split,
        database_path=str(database_path),
        top_k=top_k,
        case_count=case_count,
        hit_rate_at_k=sum(result.hit_at_k for result in case_results) / case_count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / case_count,
        mean_recall_at_k=sum(result.recall_at_k for result in case_results) / case_count,
        cases=case_results,
    )


def write_evaluation(result: RetrievalEvaluation, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)

