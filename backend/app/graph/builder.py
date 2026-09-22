"""Assemble and compile the SmartLearn multi-agent StateGraph.

                     +--> doubt_agent <--> doubt_tools ---------------------------------------+
                     |                                                                          |
  START -> load_context -> supervisor -+--> viva_ask --------------------------------------+   |
                                       +--> viva_evaluate -> viva_score* / viva_score_auto -+   |
                                       +--> viva_end                                        |   |
                                       +--> test_generate / test_grade                      +-> END
                                       +--> notes_prepare => [summarizer | latex | diagram_mapper] => notes_merge
   any guarded node that failed  ------------> retry_gate --(same / fallback model)--> failed node
                                                     +--(exhausted)--> error_handler -> END
   * = compiled with interrupt_before (human-in-the-loop scoring gate)
"""
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from backend.app.graph.context import load_context
from backend.app.graph.doubt import build_doubt_tools_node, doubt_agent, route_doubt
from backend.app.graph.notes import (
    WORKERS, diagram_mapper, latex_worker, notes_merge, notes_prepare, route_notes_prepare, summarizer,
)
from backend.app.graph.quiz import test_generate, test_grade
from backend.app.graph.resilience import error_handler, retry_gate, route_end_or_retry
from backend.app.graph.state import AgentState
from backend.app.graph.supervisor import SUPERVISOR_TARGETS, route_supervisor, supervisor_node
from backend.app.graph.viva import (
    make_score_node, route_after_score, route_viva_evaluate, viva_ask, viva_end, viva_evaluate,
)

# Nodes the run pauses *before* so a human can weigh in.
INTERRUPT_BEFORE = ["viva_score"]


def build_graph(checkpointer: Optional[Any] = None):
    g = StateGraph(AgentState)

    g.add_node("load_context", load_context)
    g.add_node("supervisor", supervisor_node)

    # specialists
    g.add_node("doubt_agent", doubt_agent)
    tools_node = build_doubt_tools_node()
    if tools_node is not None:
        g.add_node("doubt_tools", tools_node)
    g.add_node("viva_ask", viva_ask)
    g.add_node("viva_evaluate", viva_evaluate)
    g.add_node("viva_score", make_score_node("viva_score"))            # gated: interrupt_before
    g.add_node("viva_score_auto", make_score_node("viva_score_auto"))  # same logic, no interrupt
    g.add_node("viva_end", viva_end)
    g.add_node("test_generate", test_generate)
    g.add_node("test_grade", test_grade)

    # notes fan-out workers + fan-in merger
    g.add_node("notes_prepare", notes_prepare)
    g.add_node("summarizer", summarizer)
    g.add_node("latex", latex_worker)
    g.add_node("diagram_mapper", diagram_mapper)
    g.add_node("notes_merge", notes_merge)

    # resilience
    g.add_node("retry_gate", retry_gate)
    g.add_node("error_handler", error_handler)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "supervisor")

    supervisor_map = {t: t for t in SUPERVISOR_TARGETS} | {"retry_gate": "retry_gate"}
    g.add_conditional_edges("supervisor", route_supervisor, supervisor_map)

    doubt_map = {END: END, "retry_gate": "retry_gate"}
    if tools_node is not None:
        doubt_map["doubt_tools"] = "doubt_tools"
        g.add_edge("doubt_tools", "doubt_agent")
    g.add_conditional_edges("doubt_agent", route_doubt, doubt_map)

    g.add_conditional_edges("viva_ask", route_end_or_retry, {END: END, "retry_gate": "retry_gate"})
    g.add_conditional_edges(
        "viva_evaluate", route_viva_evaluate,
        {"viva_score": "viva_score", "viva_score_auto": "viva_score_auto", "retry_gate": "retry_gate"},
    )
    for scorer in ("viva_score", "viva_score_auto"):
        g.add_conditional_edges(scorer, route_after_score, {"viva_ask": "viva_ask", END: END, "retry_gate": "retry_gate"})
    g.add_edge("viva_end", END)

    g.add_conditional_edges("test_generate", route_end_or_retry, {END: END, "retry_gate": "retry_gate"})
    g.add_edge("test_grade", END)

    # fan-out: notes_prepare returns the list of worker nodes -> they run in parallel in one superstep
    g.add_conditional_edges("notes_prepare", route_notes_prepare, [*WORKERS, END, "retry_gate"])
    # fan-in: a list-edge waits until *all* workers have finished
    g.add_edge(list(WORKERS), "notes_merge")
    g.add_conditional_edges("notes_merge", route_end_or_retry, {END: END, "retry_gate": "retry_gate"})

    # retry_gate routes with Command(goto=...); error_handler is terminal
    g.add_edge("error_handler", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=INTERRUPT_BEFORE)
