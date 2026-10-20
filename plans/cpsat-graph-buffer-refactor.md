# CP-SAT as graph + nodes + timeline over one buffer type

## Context

The CP-SAT layout solver (`torch_spyre/_inductor/scratchpad/ilp_solver_ortools.py`,
1508 lines) dispatches on five buffer types: the allocator's `LifetimeBoundBuffer`,
`CoreDivisionBuffer` and `RelayoutCopyBuffer` (`plan_solver.py`), wrapped in its own
`_LifetimeBufferWithCpVars` / `_CoreDivisionBufferWithCpVars`. The distinctions are
really "which decisions are already made": a placement-only buffer is one whose
division is fixed, a barred buffer one whose residency is fixed `False`, a dead relayout
copy one pinned out. Relationships are re-derived ad hoc (`_get_children`,
`_relayout_copies`, `merge_vars` on wrappers, op positions from `uses[0]`).

Goal: one node type, `CoreDivisionBoundBuffer`, whose decision fields (`division`,
`resident`, `offset`) are fixed ahead of time or `None` for the solver to find; the
relationships as an explicit `BufferGraph` of edges; op order as a `Timeline`.
Done in phases (user decision): **Phase 1 is CP-SAT only** and keeps the allocator's
three classes as the external contract; Phase 2 pushes the node type out to the
allocator and the other solvers.

Work happens in worktree `.claude/worktrees/cpsat-graph-model` (branch
`cpsat-graph-model`, off `main` 82edd8f6). No commits (user commits).
Already written there: `scratchpad/buffer_graph.py` (draft of step 1 below).

## Phase 1 — CP-SAT internals

### 1. New module `scratchpad/buffer_graph.py` (drafted)

- `spill_cost(buffer, size=None)` — the formula lifted verbatim from
  `_LifetimeBufferWithCpVars.spill_cost`, as a plain function.
- `CoreDivisionBoundBuffer` (dataclass): `name`, `size` (alignment units, total),
  `core_divisions` (never empty; placement-only gets `(CoreDivision(),)`),
  `start_time`/`end_time`, `spill_cost`, `boundary`, `sym_core_divs`,
  `sym_tile_counts`; decisions `division`, `resident`, `offset` (`None` = free).
  A single candidate fixes `division = 0` in `__post_init__`. Derived:
  `sym_is_lx`, `sym_division` (via `division_symbol`), `per_core_sizes`, `cores_used`,
  `shape_costs`, `tileable`. It satisfies the printer contract
  (`core_divisions`, `sym_core_divs`, `sym_division`) unchanged.
- Edges (frozen): `SliceEdge(producer, consumer, pairs, relayouts)` (the residency
  gate; `relayouts` = `RelayoutRead(copy, candidates)` sorted by group),
  `InPlaceEdge(parent, child, pairs | None)` (`None` = child division fixed upstream,
  static size check already done), `CopyEdge(source, copy, priced)`.
- `Timeline(produced_at)` with `runs(names)` — maximal runs on consecutive ticks;
  replaces the position/adjacency logic in `_tiling_group_ids`.
- `BufferGraph(nodes, slices, in_place, copies, timeline)` + `consumers_of`.
- `build_buffer_graph(buffers, alignment, barred)` — the only place that looks at
  allocator classes. Hard-codes what is known: `resident=False` for barred buffers
  (`record_exclusions`) and for copies whose source does not choose a division or
  that no edge reads; slicing/relayout edges only between division-choosing buffers
  (matches today's no-op `constrain_residency` on placement-only wrappers);
  edge order = producer order, so constraints are added in today's order.

### 2. Rewrite `ilp_solver_ortools.py` on the graph

- Delete `_LifetimeBufferWithCpVars`, `_CoreDivisionBufferWithCpVars`, `_wrap`,
  `_get_children`, `_relayout_copies`, `_add_core_division`, every
  `isinstance(..., RelayoutCopyBuffer/CoreDivisionBuffer)` and `getattr(...,
  "core_divisions")` inside the solve.
- Add `_NodeVars` (one per node): a free field gets a CP var, a fixed one a constant
  (`division` → `new_constant(k)` and constant `eff_size/cores/core_cost/splits`, the
  existing single-candidate fast path generalised to any fixed `k`; `resident`/`offset`
  → var pinned with `model.add(var == value)`, since interval presence needs a literal).
  Exposes `buffer`, `division`, `division_is` for `cpsat_expr_printer` (contract unchanged).
- Solver methods take `(model, graph, vars)`:
  `_add_inplace_relaxation` (merge literal per `InPlaceEdge`; division gate only when
  `pairs is not None`), `_add_no_overlap_2d` (unchanged logic), `_constrain_residency`
  (per `SliceEdge`, returns served literals), `_constrain_relayout_copies` (per
  `CopyEdge`), `_cut_literals` / `_tiling_group_ids` (over `tileable` nodes and
  `graph.timeline.runs`), `_tile_count_terms`, `_minimize_cost_expr` (maps every node).
- Objective ladder unchanged; parallelism/shape/tile terms come only from nodes with
  `division is None` (constants cannot move the optimum; this also skips the no-op
  re-solves a placement-only solve runs today).
- `_extract` fills `node.resident/division/offset` (the "solver finds the undefined
  fields" step) and returns chosen relayouts per consumer; bottom-justify skipped when
  any offset was pinned. `_plan_layout_generic` writes nodes back to the caller's
  buffers (`address` in bytes, `spill_reason`, `chosen_division` only where the caller
  enumerated divisions, `chosen_relayouts` scaled).
- Keep unchanged: `_gate_divisions`, `_PlacementUnit`, `_justify`, `_solve_and_record`,
  capabilities, `__init__`, public `plan_layout` / `plan_layout_and_core_divisions`.
- Rewrite the module docstring around nodes/edges/timeline; short docstrings only.

### 3. Dangling references

- `tests/inductor/test_cooptimization_types.py:361-372` → compare against
  `buffer_graph.spill_cost(b)`.
- Name-only updates where the old class is cited: `plan_solver.py:152`,
  `sa_cooptimizer.py:1295`, `allocator.py:2696`, `cpsat_expr_printer.py:809`.

### 4. Tests for the new module

Add `tests/inductor/test_buffer_graph.py` (unit, no device): node fixing rules
(single candidate, barred, dead copy), edge construction (slice/in-place/copy, order),
`Timeline.runs`, and one solve showing a pre-fixed `division`/`resident`/`offset`
is honoured and free fields come back filled. Register it in a `tests/configs/` shard
and run `make check-all-configs`.

### Verification (Phase 1)

- Baseline on frozen `main` copy (`~/gap-probe/cpsat-graph-model/base`), serial
  (conftest opens the card; xdist workers collide): **1121 passed, 5 skipped,
  11 xfailed** over test_scratchpad_solver, test_solver_relayout_decision,
  test_cooptimization_types, test_cost_expr_dump, test_restickify_cost_model,
  test_sa_cooptimizer, test_op_features, test_coarse_tiling, test_scratchpad_patterns,
  test_perm_layout_solver. Same set on a frozen `git archive` of the change must match.
- Then the device suites that compile through CP-SAT: test_scratchpad_use,
  test_work_division, test_solver_relayout_e2e, test_solver_auto_coarse_tiling — base
  vs change, same serial setup.
- Equivalence spot check: for the synthetic graphs in
  `tests/inductor/synthetic_cooptimization_graphs.py`, solve with
  `torch.use_deterministic_algorithms(True)` (1 worker) on base and change and compare
  residency set, chosen divisions and each ladder stage's objective.
- `pre-commit run --files <changed>` (ruff, mypy).

## Phase 2 — outline (separate plan after Phase 1 lands)

Make `CoreDivisionBoundBuffer` the allocator's output type and retire
`LifetimeBoundBuffer`/`CoreDivisionBuffer`/`RelayoutCopyBuffer`. Sized from the inventory:

- **2a. Node absorbs the allocator-only fields.** The node (moved into
  `plan_solver.py`) gains what the allocator commit path reads: `in_place_parents`,
  `residency_reason`, lifetime overrides, `lx_view`, `spill_reason`, `chosen_relayouts`,
  `lx_relayout_plans`, `paired_with` (greedy only), plus `division_space`,
  `residency_edges`, `op_position`, `tile_reads` (SA co-optimizer). Sizes stay in bytes
  on the node; CP-SAT converts to units internally. `address` stays bytes (every solver
  writes bytes today).
- **2b. Allocator constructs nodes + graph** at its four build sites in `allocator.py`:
  `_build_bound_buffers` (:1456/:1497), `_append_lx_relayout_destinations` (:1819),
  `_build_cd_bound_buffers` (:4074/:4275), `_relayout_copy_buffers` (:4418, via
  `build_relayout_copy`). Its 4 isinstance checks (:713, :2769, :2964, :3099) become
  graph lookups (`graph.copies`). `build_buffer_graph` then reads nodes, not classes.
- **2c. Placement-only solvers** (greedy, first-fit/best-fit, SA, cooling_schedules,
  utils.plot_buffers): read-only field use (~90 lines); swap the type, drop greedy's
  isinstance at :183.
- **2d. Exhaustive search and SA co-optimizer**: read `parents`/`cd_parent_matches`
  from graph edges, write `division`/`resident` on nodes; collapse SA's duplicated
  `_spill_cost` onto `buffer_graph.spill_cost`.
- **2e. Tests**: ~180 constructor calls across 13 files, mostly through helpers
  (`make_buffer_registry`, `synthetic_cooptimization_graphs._buf`,
  `cooptimization_capture_loader._buf`, `test_perm_layout_solver._buf`); switch helpers
  first, then the direct calls in `test_scratchpad_solver.py` (131) and
  `test_solver_relayout_decision.py`.

Each sub-step leaves the full suite green; written up as its own plan when Phase 1 is in.
