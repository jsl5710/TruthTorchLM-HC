"""Routing action space (protocol Q6).

Pins the taxonomy the routing engine selects over: the seven actions, their cost ordering
(cheapest-adequate objective), and the safety invariants (which actions never ship an
unreviewed answer). Pure enum/dataclass — loads standalone.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    dotted = "hc_benchmark.routing_actions"
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "hc_benchmark" not in sys.modules:
        pkg = types.ModuleType("hc_benchmark")
        pkg.__path__ = [str(REPO_ROOT / "hc_benchmark")]
        sys.modules["hc_benchmark"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / "hc_benchmark/routing_actions.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


r = _load()
RoutingAction = r.RoutingAction
EscalationTarget = r.EscalationTarget


class TestTaxonomy:
    def test_the_seven_actions_exist(self):
        assert {a.value for a in RoutingAction} == {
            "approve", "rewrite", "clarify", "abstain",
            "escalate_rag", "escalate_larger_llm", "escalate_human",
        }

    def test_every_action_has_a_spec(self):
        assert set(r.ACTIONS) == set(RoutingAction)

    def test_escalation_targets_cover_rag_llm_human(self):
        targets = {s.escalates_to for s in r.ACTIONS.values() if s.escalates_to}
        assert targets == {EscalationTarget.RAG, EscalationTarget.LARGER_LLM,
                           EscalationTarget.HUMAN}


class TestCostOrdering:
    def test_approve_is_cheapest_and_human_is_most_expensive(self):
        ordered = r.actions_by_cost()
        assert ordered[0].action is RoutingAction.APPROVE
        assert ordered[-1].action is RoutingAction.ESCALATE_HUMAN

    def test_cost_is_monotonic_non_decreasing(self):
        tiers = [s.cost_tier for s in r.actions_by_cost()]
        assert tiers == sorted(tiers)

    def test_escalations_cost_more_than_in_place_actions(self):
        in_place = max(r.ACTIONS[a].cost_tier for a in
                       [RoutingAction.APPROVE, RoutingAction.REWRITE,
                        RoutingAction.CLARIFY, RoutingAction.ABSTAIN])
        escalations = min(r.ACTIONS[a].cost_tier for a in
                          [RoutingAction.ESCALATE_RAG, RoutingAction.ESCALATE_LARGER_LLM,
                           RoutingAction.ESCALATE_HUMAN])
        assert escalations > in_place


class TestSafetyInvariants:
    def test_only_approve_and_rewrite_and_escalate_answers_ship_a_model_answer(self):
        ships = {a for a, s in r.ACTIONS.items() if s.ships_answer}
        assert ships == {RoutingAction.APPROVE, RoutingAction.REWRITE,
                         RoutingAction.ESCALATE_RAG, RoutingAction.ESCALATE_LARGER_LLM}

    def test_abstain_clarify_and_human_are_safe_stops(self):
        """The actions that guarantee no unreviewed answer reaches the user."""
        safe = {a for a, s in r.ACTIONS.items() if s.is_safe_stop}
        assert safe == {RoutingAction.ABSTAIN, RoutingAction.CLARIFY,
                        RoutingAction.ESCALATE_HUMAN}

    def test_approve_is_the_only_ship_as_is_without_a_safe_stop_and_no_escalation(self):
        spec = r.ACTIONS[RoutingAction.APPROVE]
        assert spec.ships_answer and not spec.is_safe_stop and spec.escalates_to is None


class TestSerialization:
    def test_describe_action_space_is_serializable_and_ordered(self):
        rows = r.describe_action_space()
        assert rows[0]["action"] == "approve"
        assert rows[-1]["action"] == "escalate_human"
        # every row carries the fields a policy config needs
        for row in rows:
            assert {"action", "cost_tier", "ships_answer", "is_safe_stop",
                    "escalates_to", "typical_trigger"} <= set(row)

    def test_action_spec_lookup_accepts_enum_or_value(self):
        assert r.action_spec(RoutingAction.APPROVE).cost_tier == 0
        assert r.action_spec("escalate_human").escalates_to is EscalationTarget.HUMAN
