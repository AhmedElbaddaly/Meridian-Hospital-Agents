"""
Person 2 deliverable, part 2: prove the ticket path and the HITL path are
each real and distinct, driven by genuine mcp_bridge/db_helpers errors --
not manually inserted rows and not an auto-approved code path.

Run:
    python -m state_graph.demo_person2_ticket_and_hitl
"""

from __future__ import annotations

import uuid

from .checkpoint import CheckpointStore
from .runtime import GraphRuntime
from .graphs.post_op_recovery import build_post_op_recovery_graph
from . import tickets, hitl


def demo_real_ticket() -> None:
    print("=== A) Failure ticket from a GENUINE tool error (unknown patient) ===")
    run_id = str(uuid.uuid4())
    store = CheckpointStore()
    runtime = GraphRuntime(build_post_op_recovery_graph(), store=store)

    # patient_id 999999 does not exist in the seeded hospital DB -> intake's
    # mcp_bridge.get_patient() call raises ToolCallError for real.
    state = runtime.start(initial_data={"patient_id": 999999, "procedure": "hip replacement"}, run_id=run_id)
    state = runtime.step(state)  # executes 'intake' -> real ToolCallError -> ticket

    print(f"status={state.status}, node={state.current_node}, ticket_id={state.ticket_id}")
    assert state.status == "failed"
    assert state.ticket_id is not None

    ticket = tickets.get_ticket_for_run(run_id)
    assert ticket is not None
    print(f"Ticket opened: type={ticket.error_type}, message={ticket.error_message!r}, status={ticket.status}")
    assert ticket.status == "open"
    assert "999999" in ticket.error_message

    print("An admin investigates and resolves the ticket through the platform...")
    tickets.resolve_failure_ticket(
        ticket.ticket_id,
        resolution_note="Typo in patient_id -- corrected to 1 (Mohamed Adel) and retried.",
        status="resolved",
    )
    # The admin's correction is applied via data_patch, which resume() merges
    # into the DURABLE checkpointed state (not just the in-memory object) --
    # see the Person 2 bug fix in runtime.py.
    resumed = runtime.resume(run_id, data_patch={"patient_id": 1, "resume_next": "intake"})
    assert resumed.status == "running"
    assert resumed.current_node == "intake"
    resumed = runtime.step(resumed)
    print(f"After admin fix + resume: status={resumed.status}, node={resumed.current_node}, "
          f"patient_name={resumed.data.get('patient_name')}")
    assert resumed.status == "running"
    assert resumed.data.get("patient_name") == "Mohamed Adel"
    print("=== A) PASSED: ticket was real, distinct from HITL, and resume worked from the checkpoint ===\n")


def demo_real_hitl() -> None:
    print("=== B) HITL pause + real admin decision, distinct code path from tickets ===")
    run_id = str(uuid.uuid4())
    store = CheckpointStore()
    runtime = GraphRuntime(build_post_op_recovery_graph(), store=store)

    # simulate_lab_arrival lets the genuinely-external await_lab_results wait
    # resolve for this demo run (in production it's set by the lab webhook,
    # not by the caller) so we can reach the HITL node within max_steps.
    state = runtime.start(
        initial_data={"patient_id": 2, "procedure": "appendectomy", "simulate_lab_arrival": True},
        run_id=run_id,
    )
    state = runtime.run_until_pause(state, max_steps=10)

    print(f"status={state.status}, node={state.current_node}, hitl_task_id={state.hitl_task_id}")
    assert state.status == "waiting_hitl"
    assert state.hitl_task_id is not None
    assert state.ticket_id is None, "a HITL pause must never also be recorded as a failure ticket"

    task = hitl.get_hitl_for_run(run_id)
    print(f"HITL task opened: reason={task.reason!r}, status={task.status}")
    assert task.status == "pending"

    print("Admin reviews the persisted state on the platform and approves...")
    hitl.resolve_hitl_task(
        task.task_id,
        decision={"plan": "advance_phase", "admin_id": "dr_hassan"},
        status="approved",
    )
    resumed = runtime.resume(run_id)
    resumed = runtime.run_until_pause(resumed, max_steps=10)
    print(f"After admin approval: status={resumed.status}, node={resumed.current_node}, "
          f"applied_plan={resumed.data.get('applied_plan')}, signed_off_by={resumed.data.get('signed_off_by')}")
    assert resumed.status == "completed"
    assert resumed.data.get("applied_plan") == "advance_phase"
    assert resumed.data.get("signed_off_by") == "dr_hassan"
    print("=== B) PASSED: HITL paused, admin decided through the platform's data path, run resumed correctly ===\n")


if __name__ == "__main__":
    demo_real_ticket()
    demo_real_hitl()
    print("=== ALL PERSON 2 PROOFS PASSED ===")
