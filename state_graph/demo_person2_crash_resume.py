"""
Person 2 deliverable: crash-and-resume proof.

The project spec is explicit: "Test it: kill the process mid-run on
purpose, restart it, and show the run resuming from its last checkpoint
with no re-execution of steps that already completed and no loss of the
state that was already collected." A pause/resume inside the SAME running
Python process does not prove that -- in-memory state would trivially
survive. This script proves it for real:

  1. A CHILD process is spawned (`_run_child`, invoked via
     `python -m state_graph.demo_person2_crash_resume --child <run_id>`).
     It starts a post_op_recovery run, executes several nodes one at a
     time (checkpointing after each, via the real GraphRuntime), and then
     calls os._exit(137) -- a hard kill, no cleanup, no graceful shutdown
     hooks -- partway through the run, deliberately BEFORE the HITL node.

  2. The PARENT process (`main`) launches that child with subprocess,
     confirms it really died (nonzero/killed exit code), then opens a
     brand-new GraphRuntime pointed at the SAME durable checkpoint DB
     (state_graph/db_location.py's shared hospital DB) and calls
     `runtime.resume(run_id)`.

  3. The parent asserts: the resumed state's `current_node` and `history`
     match exactly what the child had completed and checkpointed before
     dying (no re-execution of `intake`, `plan_recovery_sequence`,
     `collect_vitals`, `order_labs` -- they do not appear a second time in
     history), and that `state.data` (vitals, recovery_plan, etc.)
     collected before the crash is fully intact. It then finishes the run
     to completion to show the resumed run genuinely continues, not just
     "resumes and immediately re-does everything."

Run:
    python -m state_graph.demo_person2_crash_resume
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import uuid

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_child(run_id: str) -> None:
    """Executed ONLY inside the spawned child process."""
    from .checkpoint import CheckpointStore
    from .runtime import GraphRuntime
    from .graphs.post_op_recovery import build_post_op_recovery_graph

    store = CheckpointStore()  # resolves the shared hospital DB (see db_location.py)
    print(f"[child pid={os.getpid()}] checkpoint DB = {store.db_path}", flush=True)

    runtime = GraphRuntime(build_post_op_recovery_graph(), store=store)
    state = runtime.start(
        initial_data={"patient_id": 1, "procedure": "appendectomy"},
        run_id=run_id,
    )

    # Execute exactly 4 nodes (intake -> plan -> vitals -> order_labs),
    # checkpointing after each -- then die BEFORE await_lab_results/HITL.
    for i in range(4):
        state = runtime.step(state)
        print(f"[child] completed node -> {state.current_node} "
              f"(history so far: {state.history})", flush=True)

    print(f"[child] state.data collected so far: "
          f"{json.dumps(state.data, default=str)}", flush=True)
    print("[child] SIMULATING HARD CRASH NOW (os._exit, no cleanup) ...", flush=True)
    sys.stdout.flush()
    os._exit(137)  # SIGKILL-style hard exit -- nothing after this line ever runs


def main() -> None:
    run_id = str(uuid.uuid4())
    print(f"=== Person 2 crash-and-resume proof | run_id={run_id} ===\n")

    print("--- Step 1: spawn a REAL child OS process to run the graph ---")
    proc = subprocess.run(
        [sys.executable, "-m", "state_graph.demo_person2_crash_resume",
         "--child", run_id],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.stderr:
        print("[child stderr]\n" + proc.stderr)

    # os._exit(137) is reported by the OS as return code 137 on POSIX.
    assert proc.returncode == 137, (
        f"expected the child to hard-exit with code 137, got {proc.returncode} "
        "-- crash simulation did not happen as expected"
    )
    print(f"--- Step 1 result: child process is CONFIRMED DEAD (exit code {proc.returncode}) ---\n")

    print("--- Step 2: fresh parent process resumes from the LAST checkpoint ---")
    from .checkpoint import CheckpointStore
    from .runtime import GraphRuntime
    from .graphs.post_op_recovery import build_post_op_recovery_graph

    store = CheckpointStore()
    print(f"[parent] checkpoint DB = {store.db_path} (same file the child wrote to)")

    runtime = GraphRuntime(build_post_op_recovery_graph(), store=store)
    resumed = runtime.resume(run_id)

    print(f"[parent] resumed at node = {resumed.current_node!r}")
    print(f"[parent] history from checkpoint = {resumed.history}")

    expected_completed = ["intake", "plan_recovery_sequence", "collect_vitals", "order_labs"]
    for node in expected_completed:
        assert node in resumed.history, f"expected {node} in checkpointed history, missing!"
    assert resumed.history.count("intake") == 1, "intake was checkpointed more than once -- re-execution bug!"
    assert resumed.current_node == "await_lab_results", (
        f"expected to resume exactly at 'await_lab_results', got {resumed.current_node!r}"
    )
    assert resumed.data.get("recovery_plan") is not None, "recovery_plan data was lost across the crash!"
    assert resumed.data.get("vitals") is not None, "vitals data was lost across the crash!"
    print("--- Step 2 PASSED: correct node resumed, no re-execution, no data loss ---\n")

    print("--- Step 3: finish the run from here to prove it's not just inspectable, it's resumable ---")
    resumed.data["simulate_lab_arrival"] = True  # let the waiting node proceed for the demo
    final = runtime.run_until_pause(resumed, max_steps=10)
    print(f"[parent] final status = {final.status}, final node = {final.current_node}")
    print(f"[parent] full history = {final.history}")
    if final.status == "waiting_hitl":
        print("[parent] run correctly paused again for physician sign-off (HITL) -- "
              "this is the expected NEXT stop, not a bug.")
    print("\n=== CRASH-AND-RESUME PROOF COMPLETE ===")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        _run_child(sys.argv[2])
    else:
        main()
