from scripts.scienceworld_ensr_components import (
    FactPattern,
    Subgoal,
    parse_observation_facts,
    shortlist_actions,
    verify_transition,
)

HALLWAY = """This room is called the hallway. In it, you see:
\tthe agent
\ta substance called air
\ta picture
You also see:
\tA door to the kitchen (that is closed)
"""


def test_parser_keeps_explicit_location_visibility_and_state() -> None:
    facts = parse_observation_facts(HALLWAY, step=0)
    keys = {fact.key for fact in facts}

    assert ("agent", "located_in", "hallway", True) in keys
    assert ("picture", "visible_in", "hallway", True) in keys
    assert ("door to the kitchen", "has_state", "closed", True) in keys
    assert all(fact.confidence == 1.0 for fact in facts)


def test_parser_keeps_room_and_inventory_sections_separate() -> None:
    combined = HALLWAY + "In your inventory, you see:\n\tan orange\n"
    keys = {fact.key for fact in parse_observation_facts(combined, step=0)}

    assert ("picture", "visible_in", "hallway", True) in keys
    assert ("orange", "in_inventory_of", "agent", True) in keys
    assert ("picture", "in_inventory_of", "agent", True) not in keys


def test_parser_normalizes_named_substances_and_infers_carried_contents() -> None:
    observation = (
        "In your inventory, you see:\n"
        "\ta glass cup (containing a drawer, a substance called soap)\n"
    )
    facts = parse_observation_facts(observation, step=2)
    by_key = {fact.key: fact for fact in facts}

    assert ("glass cup", "contains", "soap", True) in by_key
    assert ("soap", "in_inventory_of", "agent", True) in by_key
    assert (
        by_key[("soap", "in_inventory_of", "agent", True)].provenance
        == "symbolic_inference"
    )


def test_parser_handles_scienceworld_sentence_descriptors_and_nested_contents() -> None:
    observation = (
        "This room is called the kitchen. In it, you see:\n"
        "\ta cupboard. The cupboard door is open. In the cupboard is: "
        "a metal pot (containing liquid apple juice), a drawer.\n"
        "\ta sink, which is turned off. In the sink is: nothing.\n"
    )
    keys = {fact.key for fact in parse_observation_facts(observation, step=0)}

    assert ("cupboard", "visible_in", "kitchen", True) in keys
    assert ("cupboard", "has_state", "open", True) in keys
    assert ("cupboard", "contains", "metal pot", True) in keys
    assert ("metal pot", "contains", "apple juice", True) in keys
    assert ("apple juice", "has_state", "liquid", True) in keys
    assert ("sink", "has_state", "off", True) in keys


def test_verifier_creates_auditable_no_progress_obligation() -> None:
    facts = parse_observation_facts(HALLWAY, step=0)
    subgoal = Subgoal(
        subgoal_id="sg-find-target",
        description="Find the target substance.",
        completion_test="target is visible",
    )

    assessment = verify_transition(
        subgoal=subgoal,
        action="look around",
        step=1,
        observation_before=HALLWAY,
        observation_after=HALLWAY,
        reward=0,
        facts_before=facts,
        facts_after=parse_observation_facts(HALLWAY, step=1),
    )

    assert assessment.obligations[0].source == "no_progress"
    assert "subgoal=sg-find-target" in assessment.obligations[0].retrieval_query()


def test_verifier_accepts_an_observed_expected_effect() -> None:
    opened = HALLWAY.replace("that is closed", "that is open")
    subgoal = Subgoal(
        subgoal_id="sg-open-kitchen",
        description="Open the kitchen door.",
        expected_facts=[
            FactPattern(
                subject="door to the kitchen",
                relation="has_state",
                object="open",
            )
        ],
        completion_test="door state is open",
    )

    assessment = verify_transition(
        subgoal=subgoal,
        action="open door to kitchen",
        step=1,
        observation_before=HALLWAY,
        observation_after=opened,
        reward=0,
        facts_before=parse_observation_facts(HALLWAY, step=0),
        facts_after=parse_observation_facts(opened, step=1),
    )

    assert assessment.verified_expected_facts == subgoal.expected_facts
    assert assessment.obligations == []


def test_verified_effect_takes_precedence_over_retry_exhaustion() -> None:
    opened = HALLWAY.replace("that is closed", "that is open")
    subgoal = Subgoal(
        subgoal_id="sg-open-kitchen",
        description="Open the kitchen door.",
        expected_facts=[
            FactPattern(
                subject="door to the kitchen",
                relation="has_state",
                object="open",
            )
        ],
        completion_test="door state is open",
    )

    assessment = verify_transition(
        subgoal=subgoal,
        action="open door to kitchen",
        step=1,
        observation_before=HALLWAY,
        observation_after=opened,
        reward=0,
        facts_before=parse_observation_facts(HALLWAY, step=0),
        facts_after=parse_observation_facts(opened, step=1),
        retry_exhausted=True,
    )

    assert assessment.verified_expected_facts == subgoal.expected_facts
    assert assessment.obligations == []


def test_shortlist_preserves_observation_and_prefers_relevant_actions() -> None:
    subgoal = Subgoal(
        subgoal_id="sg-heat-water",
        description="Heat the water on the stove.",
        completion_test="water is hot",
    )
    actions = [
        "look around",
        "look in inventory",
        "open door to kitchen",
        "activate stove",
        "pick up picture",
        "focus on water",
    ]

    selected = shortlist_actions(
        actions,
        subgoal=subgoal,
        procedure_actions=["activate stove", "focus on water"],
        limit=4,
    )

    assert "activate stove" in selected
    assert "look around" in selected
    assert "look in inventory" in selected
    assert len(selected) == 4
