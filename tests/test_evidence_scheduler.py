from lanternquest.evidence_scheduler import EvidenceDebtLedger, EvidenceDebtSignal

FACT = ("door", "has_state", "open", True)


def _signal(
    *, source: str = "missing_precondition", step: int = 1
) -> EvidenceDebtSignal:
    return EvidenceDebtSignal(
        obligation_id=f"o-{step}",
        subgoal_id="sg-open",
        source=source,
        action="open door",
        step=step,
        fact_keys=(FACT,),
    )


def test_critical_novel_debt_triggers_budgeted_retrieval() -> None:
    ledger = EvidenceDebtLedger()

    decision = ledger.schedule(
        _signal(),
        candidate_fragment_ids=["f1", "f2", "f3"],
        active_fragment_ids=["f0"],
        remaining_retrieval_calls=4,
        max_retrieval_calls=8,
        max_retrieval_k=3,
    )

    assert decision.action == "retrieve"
    assert 1 <= decision.requested_k <= 3
    assert decision.evidence_state == "unknown"
    assert decision.candidate_novelty == 1.0


def test_repeated_candidate_evidence_is_reused_without_spending_budget() -> None:
    ledger = EvidenceDebtLedger()
    first = ledger.schedule(
        _signal(),
        candidate_fragment_ids=["f1", "f2"],
        active_fragment_ids=[],
        remaining_retrieval_calls=4,
        max_retrieval_calls=8,
        max_retrieval_k=3,
    )
    ledger.commit_retrieval(first.debt_signature, ["f1", "f2"])

    repeated = ledger.schedule(
        _signal(step=2),
        candidate_fragment_ids=["f1", "f2"],
        active_fragment_ids=["f1", "f2"],
        remaining_retrieval_calls=3,
        max_retrieval_calls=8,
        max_retrieval_k=3,
    )

    assert repeated.debt_signature == first.debt_signature
    assert repeated.occurrence_count == 2
    assert repeated.action == "reuse_active_evidence"
    assert repeated.requested_k == 0
    assert repeated.reason == "candidate_evidence_redundant"


def test_low_value_no_progress_debt_reuses_active_evidence() -> None:
    ledger = EvidenceDebtLedger()

    decision = ledger.schedule(
        _signal(source="no_progress"),
        candidate_fragment_ids=["f1", "f2"],
        active_fragment_ids=["f0"],
        remaining_retrieval_calls=1,
        max_retrieval_calls=8,
        max_retrieval_k=3,
    )

    assert decision.action == "reuse_active_evidence"
    assert decision.reason == "expected_information_gain_below_threshold"


def test_reconcile_closes_observed_evidence_debt() -> None:
    ledger = EvidenceDebtLedger()
    first = ledger.schedule(
        _signal(source="effect_mismatch"),
        candidate_fragment_ids=["f1"],
        active_fragment_ids=[],
        remaining_retrieval_calls=2,
        max_retrieval_calls=8,
        max_retrieval_k=3,
    )
    assert first.evidence_state == "contradicted"

    assert ledger.reconcile([FACT]) == 1
    snapshot = ledger.snapshot()
    assert snapshot[0]["evidence_state"] == "resolved"


def test_exhausted_budget_defers_even_critical_debt() -> None:
    ledger = EvidenceDebtLedger()

    decision = ledger.schedule(
        _signal(),
        candidate_fragment_ids=["f1"],
        active_fragment_ids=[],
        remaining_retrieval_calls=0,
        max_retrieval_calls=8,
        max_retrieval_k=3,
    )

    assert decision.action == "defer"
    assert decision.requested_k == 0
    assert decision.reason == "retrieval_budget_exhausted"


def test_first_subgoal_anchor_retrieves_full_candidate_set_then_suppresses() -> None:
    ledger = EvidenceDebtLedger(anchor_first_debt_per_subgoal=True)
    first = ledger.schedule(
        _signal(source="no_progress"),
        candidate_fragment_ids=["f1", "f2", "f3"],
        active_fragment_ids=["f0"],
        remaining_retrieval_calls=4,
        max_retrieval_calls=8,
        max_retrieval_k=3,
    )

    assert first.action == "retrieve"
    assert first.requested_k == 3
    assert first.reason == "first_subgoal_evidence_anchor"
    ledger.commit_retrieval(first.debt_signature, ["f1", "f2", "f3"])

    second = ledger.schedule(
        EvidenceDebtSignal(
            obligation_id="o-2",
            subgoal_id="sg-open",
            source="no_progress",
            action="look around",
            step=2,
            fact_keys=(FACT,),
        ),
        candidate_fragment_ids=["f4", "f5"],
        active_fragment_ids=["f1", "f2", "f3"],
        remaining_retrieval_calls=3,
        max_retrieval_calls=8,
        max_retrieval_k=3,
    )

    assert second.action == "reuse_active_evidence"
    assert second.reason == "expected_information_gain_below_threshold"
