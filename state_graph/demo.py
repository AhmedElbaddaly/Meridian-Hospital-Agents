"""
Demo: run each of the three state graphs, exercise HITL, failure tickets,
and crash-and-resume from durable checkpoints.

Usage:
  python -m state_graph.demo
  python -m state_graph.demo --graph post_op_recovery
  python -m state_graph.demo --crash-resume   # kill-and-restart simulation
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from state_graph.checkpoint import CheckpointStore
from state_graph.runtime import GraphRuntime
from state_graph.hitl import resolve_hitl_task, list_pending_hitl
from state_graph.tickets import resolve_failure_ticket, list_open_tickets
from state_graph.graphs.post_op_recovery import build_post_op_recovery_graph
from state_graph.graphs.insurance_auth import build_insurance_auth_graph
from state_graph.graphs.ed_surge_triage import build_ed_surge_triage_graph


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demo_post_op() -> None:
    banner("GRAPH 1: post_op_recovery (Task Decomposition + RAG + HITL)")
    graph = build_post_op_recovery_graph()
    rt = GraphRuntime(graph)
    state = rt.start(
        {
            "patient_id": 101,
            "procedure": "laparoscopic cholecystectomy",
            "simulate_lab_arrival": True,
            "vitals": {"hr": 95, "sbp": 110, "dbp": 70, "temp_c": 37.4, "pain_score": 4},
        }
    )
    state = rt.run_until_pause(state)
    print(f"  status={state.status} node={state.current_node}")
    print(f"  history={state.history}")
    print(f"  recommendation={state.data.get('recommendation')} conf={state.data.get('confidence')}")

    if state.status == "waiting_hitl":
        print(f"  HITL task opened: {state.hitl_task_id}")
        print("  → Admin approves advance_phase via platform...")
        resolve_hitl_task(
            state.hitl_task_id,
            decision={"plan": "advance_phase", "admin_id": "dr_chen"},
            status="approved",
        )
        state = rt.resume(state.run_id)
        state = rt.run_until_pause(state)
        print(f"  after HITL: status={state.status} applied={state.data.get('applied_plan')}")
    print(f"  checkpoints: {len(rt.store.history(state.run_id))} durable writes")


def demo_insurance() -> None:
    banner("GRAPH 2: insurance_auth (Constrained ReAct + Tree of Thoughts + HITL)")
    graph = build_insurance_auth_graph()
    rt = GraphRuntime(graph)
    state = rt.start(
        {
            "patient_id": 202,
            "cpt_code": "27447",
            "payer": "Aetna",
            "estimated_charge": 18500,
            "simulate_insurer_response": {
                "status": "denied",
                "reason": "medical necessity not established",
            },
        }
    )
    state = rt.run_until_pause(state)
    print(f"  status={state.status} node={state.current_node}")
    print(f"  preauth={state.data.get('preauth_status')} strategy={state.data.get('appeal_strategy')}")

    if state.status == "waiting_hitl":
        print(f"  HITL task opened: {state.hitl_task_id}")
        resolve_hitl_task(
            state.hitl_task_id,
            decision={"approve_appeal": True, "admin_id": "billing_lead"},
            status="approved",
        )
        state = rt.resume(state.run_id)
        state = rt.run_until_pause(state)
        print(f"  after HITL: final_status={state.data.get('final_status')}")
    print(f"  checkpoints: {len(rt.store.history(state.run_id))}")


def demo_ed_surge() -> None:
    banner("GRAPH 3: ed_surge_triage (LATS + Constrained ReAct + HITL)")
    graph = build_ed_surge_triage_graph()
    rt = GraphRuntime(graph)
    state = rt.start(
        {
            "patients": [
                {"id": "P1", "hr": 140, "sbp": 85, "spo2": 88, "chief_complaint": "chest pain"},
                {"id": "P2", "hr": 78, "sbp": 122, "spo2": 98, "chief_complaint": "minor laceration"},
                {"id": "P3", "hr": 110, "sbp": 100, "spo2": 94, "chief_complaint": "trauma fall"},
            ]
        }
    )
    state = rt.run_until_pause(state)
    print(f"  status={state.status} node={state.current_node}")
    print(f"  LATS best={state.data.get('lats_best_name')} order={state.data.get('chosen_order')}")
    print(f"  proposals={state.data.get('proposals')}")

    if state.status == "waiting_hitl":
        print(f"  HITL task opened: {state.hitl_task_id}")
        # Admin approves the irreversible actions
        resolve_hitl_task(
            state.hitl_task_id,
            decision={
                "approved_actions": state.data.get("proposals"),
                "admin_id": "on_call_attending",
            },
            status="approved",
        )
        state = rt.resume(state.run_id)
        state = rt.run_until_pause(state)
        print(f"  after HITL: triage_complete={state.data.get('triage_complete')}")
        print(f"  execution_results={state.data.get('execution_results')}")
    print(f"  checkpoints: {len(rt.store.history(state.run_id))}")


def demo_failure_ticket() -> None:
    banner("FAILURE TICKET path (distinct from HITL)")
    graph = build_post_op_recovery_graph()
    rt = GraphRuntime(graph)
    # Missing patient_id → schema validation failure inside intake node
    state = rt.start({"procedure": "appendectomy"})  # no patient_id
    state = rt.run_until_pause(state)
    print(f"  status={state.status} error={state.error}")
    print(f"  ticket_id={state.ticket_id}")
    tickets = list_open_tickets()
    print(f"  open tickets on platform: {len(tickets)}")
    if state.ticket_id:
        resolve_failure_ticket(
            state.ticket_id,
            resolution_note="Added patient_id=999 and retrying from checkpoint",
        )
        # Resume: inject the missing field into checkpointed state then continue
        state = rt.resume(state.run_id)
        state.data["patient_id"] = 999
        state.data["simulate_lab_arrival"] = True
        state.status = "running"
        state = rt.run_until_pause(state)
        print(f"  after ticket resolve: status={state.status} node={state.current_node}")


def demo_crash_resume() -> None:
    banner("CRASH-AND-RESUME (kill process mid-run, restart from checkpoint)")
    graph = build_post_op_recovery_graph()
    rt = GraphRuntime(graph)
    state = rt.start(
        {
            "patient_id": 303,
            "procedure": "hernia repair",
            "simulate_lab_arrival": False,  # will sit in await
        }
    )
    # Advance a few steps then "crash"
    for _ in range(4):
        state = rt.step(state)
        if state.status != "running":
            break
    run_id = state.run_id
    last_node = state.current_node
    n_cp = len(rt.store.history(run_id))
    print(f"  simulated crash at node={last_node} after {n_cp} checkpoints")
    print(f"  run_id={run_id}")

    # New runtime instance = process restart
    rt2 = GraphRuntime(graph)
    restored = rt2.resume(run_id)
    print(f"  restored: node={restored.current_node} status={restored.status}")
    print(f"  history preserved: {restored.history}")
    assert restored.current_node == last_node
    assert len(rt2.store.history(run_id)) == n_cp
    # Now allow labs and finish
    restored.data["simulate_lab_arrival"] = True
    restored.data["labs_still_pending"] = False
    restored.data["labs_status"] = "ready"
    restored = rt2.run_until_pause(restored)
    if restored.status == "waiting_hitl":
        resolve_hitl_task(
            restored.hitl_task_id,
            decision={"plan": "advance_phase", "admin_id": "dr_resume"},
            status="approved",
        )
        restored = rt2.resume(run_id)
        restored = rt2.run_until_pause(restored)
    print(f"  final status={restored.status} applied={restored.data.get('applied_plan')}")
    print("  ✓ no re-execution of completed steps; state preserved across restart")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=["post_op_recovery", "insurance_auth", "ed_surge_triage", "all"], default="all")
    parser.add_argument("--crash-resume", action="store_true")
    parser.add_argument("--failure", action="store_true")
    args = parser.parse_args()

    if args.crash_resume:
        demo_crash_resume()
        return
    if args.failure:
        demo_failure_ticket()
        return

    if args.graph in ("post_op_recovery", "all"):
        demo_post_op()
    if args.graph in ("insurance_auth", "all"):
        demo_insurance()
    if args.graph in ("ed_surge_triage", "all"):
        demo_ed_surge()
    if args.graph == "all":
        demo_failure_ticket()
        demo_crash_resume()

    banner("Pending HITL tasks / open tickets (platform surface)")
    print(f"  pending HITL: {len(list_pending_hitl())}")
    print(f"  open tickets: {len(list_open_tickets())}")


if __name__ == "__main__":
    main()
