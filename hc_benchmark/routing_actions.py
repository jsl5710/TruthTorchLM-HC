"""Intervention & routing action space (protocol Q6).

The Intervention & Routing Engine is the consumer of everything the benchmark measures: it
maps the combined signals — UQ score (M), the OOD/competence gate, risk flags, and the
grounding result — to *what to do next* with a candidate response. This module defines the
**action space** it selects over; the policy that does the selecting (a rule-based MVP
first, a learned MoE later) is built on top and chooses among these.

The protocol frames the router as an accuracy–safety–cost trade, not a plain classifier:
escalating to a larger model or a human is expensive, so the actions carry a **cost tier**
and the natural objective is the *cheapest adequate* action — approve when safe, and reach
for an expensive escalation only when the signals demand it.

Actions, cheapest → most expensive:

    APPROVE   deliver the response to the user as-is
    REWRITE   send it through a self-correction / refinement loop (same model)
    CLARIFY   ask the user a clarifying question instead of answering
    ABSTAIN   decline to answer (safe refusal) — no escalation warranted/possible
    ESCALATE  hand off to a more capable / grounded resource, one of:
                → RAG          retrieve grounding evidence and regenerate
                → LARGER_LLM   route to a larger / reasoning model
                → HUMAN        human-in-the-loop review (highest cost)

`ABSTAIN` and `ESCALATE→HUMAN` are the safety-preserving actions (they never ship an
unreviewed answer); `APPROVE` is the only one that ships as-is.
"""

from dataclasses import dataclass
from enum import Enum

__all__ = ["RoutingAction", "EscalationTarget", "ACTIONS", "action_spec",
           "actions_by_cost", "describe_action_space", "print_action_space"]


class RoutingAction(str, Enum):
    APPROVE = "approve"
    REWRITE = "rewrite"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"
    ESCALATE_RAG = "escalate_rag"
    ESCALATE_LARGER_LLM = "escalate_larger_llm"
    ESCALATE_HUMAN = "escalate_human"


class EscalationTarget(str, Enum):
    """Where an ESCALATE action hands off. None for the non-escalation actions."""

    RAG = "rag"
    LARGER_LLM = "larger_llm"
    HUMAN = "human"


@dataclass(frozen=True)
class ActionSpec:
    action: RoutingAction
    cost_tier: int              # 0 = cheapest (approve) … higher = more expensive
    ships_answer: bool          # does the user get a model answer as a direct result?
    is_safe_stop: bool          # does it guarantee no unreviewed answer reaches the user?
    escalates_to: EscalationTarget  # None unless this is an ESCALATE action
    description: str
    typical_trigger: str        # the signal pattern that should select it


# The action space. cost_tier orders them for the "cheapest adequate action" objective:
# a rule policy walks from APPROVE outward and stops at the first action whose safety /
# quality bar is met.
ACTIONS = {
    RoutingAction.APPROVE: ActionSpec(
        RoutingAction.APPROVE, cost_tier=0, ships_answer=True, is_safe_stop=False,
        escalates_to=None,
        description="Deliver the response to the user unchanged.",
        typical_trigger="high UQ confidence + no risk flag + grounded + in-domain (OOD gate pass).",
    ),
    RoutingAction.REWRITE: ActionSpec(
        RoutingAction.REWRITE, cost_tier=1, ships_answer=True, is_safe_stop=False,
        escalates_to=None,
        description="Refine the response via a self-correction loop on the same model, then re-gate.",
        typical_trigger="borderline UQ / minor grounding gap that a rewrite can plausibly fix.",
    ),
    RoutingAction.CLARIFY: ActionSpec(
        RoutingAction.CLARIFY, cost_tier=1, ships_answer=False, is_safe_stop=True,
        escalates_to=None,
        description="Ask the user a clarifying question instead of answering.",
        typical_trigger="ambiguous / underspecified query (uncertainty is about intent, not knowledge).",
    ),
    RoutingAction.ABSTAIN: ActionSpec(
        RoutingAction.ABSTAIN, cost_tier=1, ships_answer=False, is_safe_stop=True,
        escalates_to=None,
        description="Decline to answer (safe refusal).",
        typical_trigger="out-of-domain (OOD gate fail) or a safety boundary, with no useful escalation.",
    ),
    RoutingAction.ESCALATE_RAG: ActionSpec(
        RoutingAction.ESCALATE_RAG, cost_tier=2, ships_answer=True, is_safe_stop=False,
        escalates_to=EscalationTarget.RAG,
        description="Retrieve grounding evidence from the KB and regenerate, then re-gate.",
        typical_trigger="unsupported / possibly-hallucinated factual claim while a relevant KB exists.",
    ),
    RoutingAction.ESCALATE_LARGER_LLM: ActionSpec(
        RoutingAction.ESCALATE_LARGER_LLM, cost_tier=3, ships_answer=True, is_safe_stop=False,
        escalates_to=EscalationTarget.LARGER_LLM,
        description="Route to a larger / reasoning model, then re-gate its answer.",
        typical_trigger="hard-reasoning failure the base model can't self-correct.",
    ),
    RoutingAction.ESCALATE_HUMAN: ActionSpec(
        RoutingAction.ESCALATE_HUMAN, cost_tier=4, ships_answer=False, is_safe_stop=True,
        escalates_to=EscalationTarget.HUMAN,
        description="Route to a human-in-the-loop reviewer.",
        typical_trigger="safety-critical case (crisis, prescriptive medical advice) — the highest-cost, safest stop.",
    ),
}


def action_spec(action: RoutingAction) -> ActionSpec:
    return ACTIONS[RoutingAction(action)]


def actions_by_cost() -> list:
    """Actions ordered cheapest → most expensive (the order a policy should prefer)."""
    return [ACTIONS[a] for a in sorted(ACTIONS, key=lambda a: ACTIONS[a].cost_tier)]


def describe_action_space() -> list:
    """A serializable view of the action space, for docs / policy config."""
    return [
        {
            "action": s.action.value,
            "cost_tier": s.cost_tier,
            "ships_answer": s.ships_answer,
            "is_safe_stop": s.is_safe_stop,
            "escalates_to": s.escalates_to.value if s.escalates_to else None,
            "description": s.description,
            "typical_trigger": s.typical_trigger,
        }
        for s in actions_by_cost()
    ]


def print_action_space() -> None:
    print("\nIntervention & routing actions (protocol Q6) — cheapest first\n" + "=" * 68)
    for s in actions_by_cost():
        tags = []
        if s.is_safe_stop:
            tags.append("safe-stop")
        if not s.ships_answer:
            tags.append("no answer shipped")
        if s.escalates_to:
            tags.append(f"→ {s.escalates_to.value}")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  [cost {s.cost_tier}] {s.action.value:22s}{tag_str}")
        print(f"            {s.description}")
        print(f"            when: {s.typical_trigger}")
    print()


if __name__ == "__main__":
    print_action_space()
