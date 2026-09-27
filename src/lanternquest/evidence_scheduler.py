"""Budget-aware scheduling for persistent evidence obligations.

The scheduler is deterministic and model independent.  It treats failed or
unverified transitions as evidence debt, keeps a semantic ledger across steps,
and spends retrieval budget only when the available memory is likely to add
new evidence.  The ledger is intentionally separate from the ScienceWorld
harness so that the policy can be unit tested and reused by LanternQuest.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

EvidenceState = Literal["unknown", "contradicted", "resolved"]
ScheduleAction = Literal["retrieve", "reuse_active_evidence", "defer"]


_SEVERITY = {
    "missing_precondition": 1.00,
    "action_error": 0.95,
    "effect_mismatch": 0.90,
    "retry_exhausted": 0.75,
    "no_progress": 0.55,
}
_CONTRADICTED_SOURCES = {"action_error", "effect_mismatch"}
DEFAULT_PRIORITY_WEIGHTS = {
    "severity": 0.42,
    "uncertainty": 0.23,
    "novelty": 0.20,
    "escalation": 0.10,
    "budget_ratio": 0.05,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def semantic_debt_signature(
    *,
    subgoal_id: str,
    source: str,
    action: str,
    fact_keys: Iterable[tuple[str, str, str, bool]],
) -> str:
    """Identify equivalent debt across changing observation hashes and steps."""
    facts = sorted(
        (
            _normalize(subject),
            _normalize(relation),
            _normalize(object_),
            bool(polarity),
        )
        for subject, relation, object_, polarity in fact_keys
    )
    stable = repr((_normalize(subgoal_id), source, _normalize(action), facts))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class EvidenceDebtSignal:
    obligation_id: str
    subgoal_id: str
    source: str
    action: str
    step: int
    fact_keys: tuple[tuple[str, str, str, bool], ...] = ()


@dataclass
class EvidenceDebtEntry:
    signature: str
    first_obligation_id: str
    latest_obligation_id: str
    subgoal_id: str
    source: str
    action: str
    fact_keys: tuple[tuple[str, str, str, bool], ...]
    state: EvidenceState
    first_step: int
    last_step: int
    occurrence_count: int = 1
    retrieval_count: int = 0
    seen_fragment_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class EvidenceScheduleDecision:
    obligation_id: str
    debt_signature: str
    source: str
    evidence_state: EvidenceState
    action: ScheduleAction
    requested_k: int
    priority: float
    severity: float
    uncertainty: float
    candidate_novelty: float
    budget_ratio: float
    occurrence_count: int
    reason: str


class EvidenceDebtLedger:
    """Persistent three-valued ledger with a budgeted retrieval policy."""

    def __init__(
        self,
        *,
        retrieval_threshold: float = 0.58,
        anchor_first_debt_per_subgoal: bool = False,
        priority_weights: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 <= retrieval_threshold <= 1.0:
            raise ValueError("retrieval_threshold must be between zero and one")
        self.retrieval_threshold = retrieval_threshold
        self.anchor_first_debt_per_subgoal = anchor_first_debt_per_subgoal
        supplied = dict(priority_weights or DEFAULT_PRIORITY_WEIGHTS)
        if set(supplied) != set(DEFAULT_PRIORITY_WEIGHTS):
            raise ValueError(
                "priority_weights must define severity, uncertainty, novelty, "
                "escalation, and budget_ratio"
            )
        if any(float(value) < 0.0 for value in supplied.values()):
            raise ValueError("priority_weights must be non-negative")
        total = sum(float(value) for value in supplied.values())
        if total <= 0.0:
            raise ValueError("priority_weights must have a positive sum")
        self.priority_weights = {
            key: float(value) / total for key, value in supplied.items()
        }
        self.entries: dict[str, EvidenceDebtEntry] = {}
        self.subgoal_retrieval_counts: dict[str, int] = {}

    def register(self, signal: EvidenceDebtSignal) -> EvidenceDebtEntry:
        signature = semantic_debt_signature(
            subgoal_id=signal.subgoal_id,
            source=signal.source,
            action=signal.action,
            fact_keys=signal.fact_keys,
        )
        entry = self.entries.get(signature)
        if entry is None:
            entry = EvidenceDebtEntry(
                signature=signature,
                first_obligation_id=signal.obligation_id,
                latest_obligation_id=signal.obligation_id,
                subgoal_id=signal.subgoal_id,
                source=signal.source,
                action=signal.action,
                fact_keys=signal.fact_keys,
                state=(
                    "contradicted"
                    if signal.source in _CONTRADICTED_SOURCES
                    else "unknown"
                ),
                first_step=signal.step,
                last_step=signal.step,
            )
            self.entries[signature] = entry
            return entry
        entry.latest_obligation_id = signal.obligation_id
        entry.last_step = signal.step
        entry.occurrence_count += 1
        if signal.source in _CONTRADICTED_SOURCES:
            entry.state = "contradicted"
        return entry

    def reconcile(
        self, observed_fact_keys: Iterable[tuple[str, str, str, bool]]
    ) -> int:
        """Resolve ledger entries whose full evidence obligation is now observed."""
        observed = set(observed_fact_keys)
        resolved = 0
        for entry in self.entries.values():
            if (
                entry.state != "resolved"
                and entry.fact_keys
                and set(entry.fact_keys).issubset(observed)
            ):
                entry.state = "resolved"
                resolved += 1
        return resolved

    def schedule(
        self,
        signal: EvidenceDebtSignal,
        *,
        candidate_fragment_ids: Iterable[str],
        active_fragment_ids: Iterable[str],
        remaining_retrieval_calls: int,
        max_retrieval_calls: int,
        max_retrieval_k: int,
    ) -> EvidenceScheduleDecision:
        if max_retrieval_calls < 1:
            raise ValueError("max_retrieval_calls must be positive")
        if max_retrieval_k < 1:
            raise ValueError("max_retrieval_k must be positive")

        entry = self.register(signal)
        candidates = list(dict.fromkeys(str(item) for item in candidate_fragment_ids))
        active = {str(item) for item in active_fragment_ids}
        unseen = [
            item
            for item in candidates
            if item not in active and item not in entry.seen_fragment_ids
        ]
        severity = _SEVERITY.get(signal.source, 0.65)
        fact_count = len(signal.fact_keys)
        uncertainty = min(1.0, max(0.35, fact_count / 3.0))
        novelty = len(unseen) / max(1, len(candidates))
        budget_ratio = min(
            1.0, max(0.0, remaining_retrieval_calls / max_retrieval_calls)
        )
        escalation = min(1.0, max(0, entry.occurrence_count - 1) / 2.0)
        priority = (
            self.priority_weights["severity"] * severity
            + self.priority_weights["uncertainty"] * uncertainty
            + self.priority_weights["novelty"] * novelty
            + self.priority_weights["escalation"] * escalation
            + self.priority_weights["budget_ratio"] * budget_ratio
        )
        priority = round(min(1.0, priority), 6)

        requested_k = min(
            max_retrieval_k,
            len(candidates),
            max(1, 1 + math.floor(priority * (max_retrieval_k - 1))),
        )
        critical = signal.source in {
            "missing_precondition",
            "action_error",
            "effect_mismatch",
        }
        anchor_due = (
            self.anchor_first_debt_per_subgoal
            and self.subgoal_retrieval_counts.get(signal.subgoal_id, 0) == 0
        )

        if entry.state == "resolved":
            action: ScheduleAction = "defer"
            requested_k = 0
            reason = "obligation_already_resolved"
        elif remaining_retrieval_calls <= 0:
            action = "defer"
            requested_k = 0
            reason = "retrieval_budget_exhausted"
        elif not candidates:
            action = "reuse_active_evidence" if active else "defer"
            requested_k = 0
            reason = "no_candidate_evidence"
        elif not unseen:
            action = "reuse_active_evidence"
            requested_k = 0
            reason = "candidate_evidence_redundant"
        elif anchor_due:
            action = "retrieve"
            requested_k = min(max_retrieval_k, len(candidates))
            reason = "first_subgoal_evidence_anchor"
        elif critical or priority >= self.retrieval_threshold:
            action = "retrieve"
            reason = "critical_or_high_value_evidence_debt"
        else:
            action = "reuse_active_evidence"
            requested_k = 0
            reason = "expected_information_gain_below_threshold"

        return EvidenceScheduleDecision(
            obligation_id=signal.obligation_id,
            debt_signature=entry.signature,
            source=signal.source,
            evidence_state=entry.state,
            action=action,
            requested_k=requested_k,
            priority=priority,
            severity=severity,
            uncertainty=round(uncertainty, 6),
            candidate_novelty=round(novelty, 6),
            budget_ratio=round(budget_ratio, 6),
            occurrence_count=entry.occurrence_count,
            reason=reason,
        )

    def commit_retrieval(
        self, debt_signature: str, returned_fragment_ids: Iterable[str]
    ) -> None:
        entry = self.entries[debt_signature]
        entry.retrieval_count += 1
        entry.seen_fragment_ids.update(str(item) for item in returned_fragment_ids)
        self.subgoal_retrieval_counts[entry.subgoal_id] = (
            self.subgoal_retrieval_counts.get(entry.subgoal_id, 0) + 1
        )

    def snapshot(self) -> list[dict[str, object]]:
        return [
            {
                "debt_signature": entry.signature,
                "first_obligation_id": entry.first_obligation_id,
                "latest_obligation_id": entry.latest_obligation_id,
                "subgoal_id": entry.subgoal_id,
                "source": entry.source,
                "action": entry.action,
                "evidence_state": entry.state,
                "first_step": entry.first_step,
                "last_step": entry.last_step,
                "occurrence_count": entry.occurrence_count,
                "retrieval_count": entry.retrieval_count,
                "seen_fragment_ids": sorted(entry.seen_fragment_ids),
            }
            for entry in sorted(self.entries.values(), key=lambda item: item.signature)
        ]
