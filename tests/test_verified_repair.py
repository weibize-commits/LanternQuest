from lanternquest.verified_repair import (
    ActionTransition,
    fact_key,
    infer_current_action_transitions,
    residual_goal_diff,
    search_verified_transition_graph,
    symbolic_state_sha256,
    validate_selected_candidate,
)


def test_residual_goal_diff_respects_any_completion() -> None:
    red = fact_key("red box", "visible_in", "greenhouse")
    green = fact_key("green box", "visible_in", "greenhouse")
    assert residual_goal_diff([red, green], [red], completion_mode="any") == ()
    assert residual_goal_diff([red, green], [], completion_mode="any") == (red, green)


def test_current_action_inference_uses_manipulated_subject() -> None:
    transitions = infer_current_action_transitions(
        [
            "move organism in seed jar to inventory",
            "pick up seed jar",
        ]
    )
    effects = {item.action: set(item.add_effects) for item in transitions}
    assert fact_key("organism in seed jar", "in_inventory_of", "agent") in effects[
        "move organism in seed jar to inventory"
    ]
    assert fact_key("seed jar", "in_inventory_of", "agent") not in effects[
        "move organism in seed jar to inventory"
    ]
    assert fact_key("seed jar", "in_inventory_of", "agent") in effects[
        "pick up seed jar"
    ]


def test_current_action_inference_represents_terminal_focus() -> None:
    transition = infer_current_action_transitions(["focus on grown lemon"])[0]
    assert transition.add_effects == (
        fact_key("grown lemon", "has_state", "focused"),
    )


def test_search_finds_two_step_verified_path() -> None:
    outside = fact_key("agent", "located_in", "outside")
    open_door = fact_key("door to greenhouse", "has_state", "open")
    greenhouse = fact_key("agent", "located_in", "greenhouse")
    candidates = search_verified_transition_graph(
        observed_facts=[outside],
        residual_facts=[greenhouse],
        transitions=[
            ActionTransition(
                action="open door to greenhouse",
                preconditions=(outside,),
                add_effects=(open_door,),
                currently_legal=True,
                provenance="train_verified_transition",
                evidence_count=4,
            ),
            ActionTransition(
                action="go to greenhouse",
                preconditions=(outside, open_door),
                add_effects=(greenhouse,),
                currently_legal=False,
                provenance="train_verified_transition",
                evidence_count=4,
            ),
        ],
        max_depth=3,
    )
    assert len(candidates) == 1
    assert candidates[0].actions == (
        "open door to greenhouse",
        "go to greenhouse",
    )
    assert candidates[0].satisfied_residual_facts == (greenhouse,)


def test_negative_edge_blocks_failed_first_action() -> None:
    state = [fact_key("agent", "located_in", "kitchen")]
    goal = fact_key("agent", "located_in", "hallway")
    action = "go to hallway"
    candidates = search_verified_transition_graph(
        observed_facts=state,
        residual_facts=[goal],
        transitions=infer_current_action_transitions([action]),
        negative_edges={(symbolic_state_sha256(state), action)},
    )
    assert candidates == []


def test_ungrounded_and_risky_transitions_are_rejected() -> None:
    goal = fact_key("red box", "has_state", "focused")
    candidates = search_verified_transition_graph(
        observed_facts=[],
        residual_facts=[goal],
        transitions=[
            ActionTransition(
                action="focus on red box",
                preconditions=(),
                add_effects=(goal,),
                currently_legal=True,
                grounded=False,
            ),
            ActionTransition(
                action="focus on green box",
                preconditions=(),
                add_effects=(goal,),
                currently_legal=True,
                risk_flags=("irreversible_answer",),
            ),
        ],
    )
    assert candidates == []


def test_candidate_selection_accepts_ids_only_and_rechecks_legality() -> None:
    goal = fact_key("seed jar", "in_inventory_of", "agent")
    candidates = search_verified_transition_graph(
        observed_facts=[],
        residual_facts=[goal],
        transitions=infer_current_action_transitions(["pick up seed jar"]),
    )
    selected = candidates[0]
    assert (
        validate_selected_candidate(
            selected.candidate_id,
            candidates,
            legal_actions=["pick up seed jar"],
        )
        == selected
    )
    assert (
        validate_selected_candidate(
            "pick up seed jar",
            candidates,
            legal_actions=["pick up seed jar"],
        )
        is None
    )
    assert (
        validate_selected_candidate(
            selected.candidate_id,
            candidates,
            legal_actions=["look around"],
        )
        is None
    )
