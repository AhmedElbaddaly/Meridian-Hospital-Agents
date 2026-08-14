"""
planning/models.py
-------------------
DAG data model (Task / Plan) used by decomposition-first and dynamic
decomposition alike.

Forked from: github.com/AmrSheta22/task_decomposition_and_planning
             (planning_lab/models.py -- Task/Plan/EnvironmentFeedback classes)

We keep the toolkit's validation guarantees (unique ids, known
dependencies, acyclicity enforced AT CONSTRUCTION TIME via a
model_validator, topological ordering, parallel-safe batches) unchanged --
that is the "Cycle detection" and "Topological ordering" requirement from
the lab, and re-deriving it would be exactly the "unnecessary/duplicated
work" the lab penalizes.

Adapted for Meridian Hospital Network:
- Task gained an optional `mcp_tool` field: when a sub-task maps directly
  onto one of our real MCP tools (see mcp_server/MCP.py), decomposition
  records that mapping on the node itself instead of leaving grounding to
  be inferred later.
- Plan.max_length raised from the toolkit's 8 to 10, because a real
  surge-reshuffle plan (assess -> capacity check -> rank -> assign ->
  validate -> apply) plus branch-specific overflow tasks can legitimately
  need more than 8 nodes -- still small enough to stay inspectable.
"""

from __future__ import annotations

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$")
    instruction: str = Field(min_length=5)
    depends_on: list[str] = Field(default_factory=list)

    # --- Meridian addition -------------------------------------------------
    # Which real system this node touches, so the executor and the router
    # (Person 2's planning/router.py) know how to ground it instead of
    # guessing from free text. None means "needs real reasoning" -- those
    # are exactly the nodes that get routed to PS / ToT / LATS.
    mcp_tool: str | None = None
    # "read" tools are safe to run without confirmation; "write" tools
    # (manage_icu_bed, update_operating_room_status, create_admission,
    # update_patient_status) must only ever be reached from the DAG's
    # single terminal/apply node, never from a mid-plan node.
    kind: str = Field(default="reasoning", pattern=r"^(read|write|reasoning)$")


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=5)
    tasks: list[Task] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_dag(self) -> "Plan":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Task ids must be unique")
        known = set(ids)
        for task in self.tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(f"{task.id} has unknown dependencies: {sorted(missing)}")
            if task.id in task.depends_on:
                raise ValueError(f"{task.id} cannot depend on itself")
        # Acyclicity enforced HERE, at construction, not discovered at
        # execution time -- a plan that could deadlock never gets built.
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = nx.find_cycle(self.graph)
            blocked = sorted({node for edge in cycle for node in edge[:2]})
            raise ValueError(f"Cycle detected; blocked tasks: {blocked}")
        # Meridian safety rule: write tools may only sit on a terminal node.
        write_nodes = [t.id for t in self.tasks if t.kind == "write"]
        terminals = {n for n, deg in self.graph.out_degree if deg == 0}
        bad_writes = [t for t in write_nodes if t not in terminals]
        if bad_writes:
            raise ValueError(
                f"Write-kind tasks must be terminal nodes, found mid-plan writes: {bad_writes}"
            )
        return self

    @property
    def graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(task.id for task in self.tasks)
        graph.add_edges_from(
            (dependency, task.id)
            for task in self.tasks
            for dependency in task.depends_on
        )
        return graph

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def execution_batches(self) -> list[list[str]]:
        """Parallel-safe batches; every dependency sits in an earlier batch."""
        return [sorted(generation) for generation in nx.topological_generations(self.graph)]

    def task(self, task_id: str) -> Task:
        return next(task for task in self.tasks if task.id == task_id)

    def terminal_tasks(self) -> list[str]:
        return [node for node, degree in self.graph.out_degree if degree == 0]


class EnvironmentFeedback(BaseModel):
    """A grounded signal produced outside the language model.
    (Person 3 implements the real source in planning/environment.py;
    this shape is kept identical to the toolkit's so LATS/Reflexion can
    consume decomposition-node results without a schema mismatch.)
    """

    model_config = ConfigDict(extra="forbid")

    success: bool
    score: float = Field(ge=0.0, le=1.0)
    details: list[str] = Field(default_factory=list)
