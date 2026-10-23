# CP-SAT as a graph-based solver: a staged path on `main`

## Context

PR #5676 (`scratchpad-refactor-experimental`, draft, "DO NOT MERGE") built the plan in
`cpsat-graph-buffer-refactor.md` end to end in about 50 commits. It has a graph, encoding
strategies, five backends, and constraint (`Lazy`) encodings as the default. There is one
allocator, no buffer records, and no solver other than CP-SAT. It showed that the
structure works. It cannot land as it is:

- It is too large to review: about 100 files, +13k / −25k lines against `main`. Phase 1
  is a single squashed commit (`9dab5027`).
- It changes behaviour:
  - default compiles plan 21-30% slower;
  - the greedy fallback is gone, and nothing goes in LX without ortools;
  - committed mode is priced like the open mode;
  - completed-reduction relayouts are lost.
- **It never ported #5246.** Merge `2796be28` overwrote the variable lifetimes, and the
  re-solve after tiling came back. `main` is now past #5246 and #5183 (the uniqueness
  check, `58afed9e`), so its tiling and lifetime encodings cannot be lifted.

This plan lands the structure on `main` in small PRs. Every PR keeps the CP model
byte-identical and the committed plans identical. It implements five things:

1. **An injectable backend.** The pywraplp-shaped protocol from the PR, with CP-SAT as
   its only implementation for now.
2. **One allocator class over the table strategy.** The enumerated menus and pair
   tables are the only encoding; there is no `Lazy`.
3. **A graph that generates the `CoreDivisionBuffer`s itself.**
4. **`DependencyEdge`s and `ExecutionEdge`s.**
5. **Placement-only and co-optimizing as strategies.** They replace the two allocator
   classes and their `division_is_fixed` and capability-flag branches.

Decisions (2026-10-23):
- CP-SAT is the only pywraplp backend.
- Greedy, first-fit, best-fit, placement SA, the SA co-optimizer and exhaustive search
  stay outside the graph model. They are fed by the buffers the graph generates. Greedy
  stays the fallback after a failed solve.
- Placement-only reproduces today's plans exactly. `CO_OPTIMIZING_LX_PLANNING=0` stays
  usable for bisecting a miscompile.

How the items read:
- **Item 2:** after Stage 6 there is one `ScratchpadAllocator`. Its co-optimizing
  strategy produces the table data (menus, match tables, capped relayout copies).
  `TableStrategy` encodes that data for CP-SAT.
- **Item 3:** the graph builder, not the allocator, makes every buffer record, and a node
  references its record.

Compatibility with the non-enumerating end state:
- Menus stay inside the records, and only `TableStrategy` reads them.
- Node and edge decisions are `WorkSplit`s and relayout choices, never menu indices.
- A later `Lazy` strategy can skip generating the records.

## Order

| # | Stage | Item | Needs | Proven by |
|---|---|---|---|---|
| 0 | Harness and gates | — | — | — |
| 1 | Backend seam | 1 | 0 | proto + solver parameters |
| 2 | Graph and dependency edges; CP-SAT reads them | 4 | 0 | proto |
| 3 | Execution edges; CP-SAT's tiling walks read them | 4 | 2 | proto (tiling corpus) |
| 4a | `TableStrategy` makes node vars, posting through the protocol | 1, 4 | 1, 3 | proto |
| 4b | The remaining hooks; the wrappers are deleted | | 4a | proto |
| 4c | `graph.solve`; `CpSatLayoutSolver` becomes an adapter | | 4b | proto + plans |
| 5a | Shared rules and cross-method state extracted | 5 | 0 | plans |
| 5b | `PlacementOnlyStrategy`; the allocator delegates its five mode hooks | 5 | 5a | plans |
| 5c | `CoOptimizingStrategy`; the co-opt methods move | 5 | 5b | plans |
| 6 | One allocator; capability rules in the strategies | 2, 5 | 5c | plans |
| 7 | The graph generates the buffers | 3 | 4c, 6 | proto + plans |
| 8 | Cleanup | — | all | — |

There are 13 PRs in two independent tracks, which meet at Stage 7:
- **CP-SAT internals (1-4)**, proven by an identical proto.
- **Allocator (5-6)**, proven by identical plans.

Two ordering choices:
- **Stage 5 comes before 7** so that the co-optimizing build code moves once: into its
  strategy, which the builder then calls.
- **Protocol posting lands with `TableStrategy` (4a/4b)** so that every posting line
  changes once; #5246's hooks have to be re-derived from `main` anyway.

## Stages

Line numbers refer to `main` at `16926127`. #5183 (`58afed9e`) inserts lines into
`ilp_solver_ortools.py` at :395, :601, :773 and :819. Paths are under
`torch_spyre/_inductor/scratchpad/`.

### Stage 0. Harness and gates (tests and probes only)

- **Harness.** Lift `tests/inductor/cpsat_equivalence.py` from `9dab5027`. It patches
  `_solve_and_record` and `_plan_layout_generic` and records, per ladder stage, the
  proto, status, objective and decisions. Add a record of the solver parameters (time
  limit, workers, presolve, seed), taken **before** the harness forces one worker and
  the deterministic limit.
- **Recorded suites.** Add `test_work_division`, which holds #5246's tests, and
  `test_coarse_tiling` to `record_tests.sh`.
- **Captures.** Re-capture smoke, zoo default and tiling, and Granite on current `main`
  (`deenum/run_profile.sh`). Replay the tiling captures, not only `caps_smoke`.
- **Same-plans check.** Promote `plan_record.py`, `same_plans.sh` and `compare_plans.py`
  from `probes/cpsat-graph/feasibility/` and extend them:
  - Record **after** `_push_allocation` as well, so that inserted clones, drain clones,
    relayout ops and `allocation["lx"]` are compared, not only address, division and
    spill reason.
  - Run under `AUTO_COARSE_TILING=1`, and once with a forced `SolveError` to cover the
    fallback.
  - Use one worker and `max_deterministic_time`, not the wall-clock limit.
  - Cover `layout_solver` × `co_optimizing_lx_planning`, plus `allow_exhaustive_search`.
- **Tooling.** Parameterize `freeze.sh` and `gate.sh`, which hard-code the experimental
  worktree.

### Stage 1. Injectable backend seam (item 1)

- **New `backend.py`.** It holds the `Backend` protocol and `CpSat`, lifted from
  `a47ef8f2` `backend.py` (up to :175).
  - **Guard the ortools import** as `ilp_solver_ortools` does, and drop `pywraplp`,
    `cp_model_pb2` and `text_format`.
  - Resolve the default backend lazily, so the no-ortools paths do not change.
  - Not lifted: `_SolvedRecording`, Mip, BruteForce, Annealing.
- **What goes through the backend:**
  - `CpModel()` (:744) and `CpSolver()` (:1312);
  - the solver parameters: time limit :1313, presolve rule :1320-1335, workers :1336,
    seed :1342;
  - `Solve` (:1283), status names and constants, the `model.proto` and parameter reads
    in `_solve_and_record`, and the `Value` / `BooleanValue` reads;
  - #5183's `clone()`, second `CpSolver`, `parameters.copy_from` and `Solve`, through a
    raw `CpSat.solver` / `CpSat.model`.
- **Injection.** `CpSatLayoutSolver(..., backend=None)` takes a factory that defaults to
  a module-level `CpSat`.
- **Unchanged:**
  - `cpsat_expr_printer` keeps cp_model's expression algebra (`new_int_var_from_domain`,
    the multiplication and division equalities, its `isinstance` checks). While CP-SAT
    is the only backend, those types are the protocol's `Var` / `LinearExpr`.
  - The `ilp_solver_ortools.cp_model` attribute stays: 10 test sites set it to `None`.
  - Tests that patch `CpSolver.Solve` or assert the parameters keep working
    (`test_scratchpad_use.py:1423`, `test_solver_relayout_decision.py:187/222`).
  - The harness stays patch-based.
- **Check:** proto diff empty, and the parameter records are equal.

### Stage 2. Graph and dependency edges; CP-SAT reads them (item 4)

**New `buffer_graph.py`.** This is a redesign of the branch's version, not a trim: the
branch has no `node.buffer`, and its copies are not records.
- **`WorkSplit`** carries every `CoreDivision` field. That includes `reduction_syms` and
  `tile_splits`, which the branch's `of` drops. The record still gets the menu index.
- **`ResourceNode`** has:
  - incoming data: name, size in units, start and end with overrides, `read_count`,
    `first_use_is_read`, `boundary`, `produced_at`, `residency_reason`, `untiled_bar`;
  - decisions: `work_split`, `resident`, `offset`, `spill_reason`;
  - links: `prev` / `next` and `inputs` / `outputs`;
  - `buffer`: its record, in units, read only by the encoding.
- **`DependencyEdge`** has `src`, `dst`, `reads` (a count, since a parent can be listed
  twice), `in_place` (`reads == 0` allowed), and the decisions `merged` and `relayout`.
- **`ExecutionEdge`** is used from Stage 3.
- **`SpyreOptimizationGraph`:**
  - `copies` holds the relayout copies in copy order, kept off the chain;
  - `buffers()` returns the records.
- **`build_graph(buffers, alignment, capacity)`.**

**CP-SAT reads the dependency edges.** `_plan_layout_generic` builds the graph from the
wrapped copies.
- These read `node.outputs` instead of `children_of`:
  - `_add_tiling` (:921);
  - `_cut_traffic_terms` (:1232, and `len(kids)` at :1248);
  - `_add_lifetime_ends` (:1142);
  - `_add_core_division` (:1733).
- The `apart_` / `via_full_` literals (:930, :942) become per read edge.
- `_relayout_copies` (:1492) reads `graph.copies`.
- The in-place loop reads `edge.in_place`.
- `_get_children` and `_relayout_copies` are deleted.

**Exactness rules:**
- A read edge exists only when the child has a menu: the base wrapper's `parents` is
  `[]` (:350).
- Every consumer iterates once per listed read.
- `via_full[child]` stays last-write-wins.

**Tests and check:**
- New `test_buffer_graph.py`, with its `tests/configs` YAML. It checks the edges
  against `_get_children` before that function is deleted.
- Proto diff empty.

### Stage 3. Execution edges; the tiling walks read them (item 4)

- **The chain** is the op outputs in tick order. Op-output ticks are unique
  (`utils.py:73-110`), and graph inputs and copies are not on it.
- **`contiguous`** means a tick difference of 1 *and* a tileable predecessor. That
  reproduces `_add_tiling_groups` (:1071-1108), including the gaps left by ops with no
  node.
- **Walk the chain:**
  - the group derivation (:1051);
  - `_add_nest_ends` (:990);
  - `_add_nest_starts` (:1023).

  The `tile_group_` domain stays `len(position)` (:1102).
- **Keep node order**, because unit-test lists are not in tick order:
  - `loop_id` / `tiled` (:898-910);
  - the cut loop (:915);
  - the per-node `_add_partial_reduction_cut_guard` (:961);
  - `_tile_count_terms`.
- **Check:**
  - proto diff empty on the tiling captures and on `test_work_division` /
    `test_coarse_tiling`;
  - then the device suites.

### Stage 4. `TableStrategy` and `graph.solve`

**4a. Node vars.** New `table_strategy.py` holds `TableStrategy` with `prepare` and
`encode_node`.
- It posts through the protocol's PascalCase methods, which pass through to CP-SAT.
- Each node gets a `NodeVars` in the wrappers' creation order. That includes #5246's
  `loop_id`, `tiled`, `cut`, `nest_start` / `nest_end`, `via_full`, `start`, `end` and
  `slot_span`.
- A node without a menu is `Fixed(COMMITTED)`.
- `NodeVars.division` and a lazily cached `division_is` keep the printer's duck typing,
  in the same creation order.
- The wrappers delegate to `NodeVars`.

**4b. The rest of the encoding.**
- **Hook order** follows `_run` (:1301-1310): tiling, lifetime ends, in-place plus the
  2-D no-overlap, core division with read gates and op constraints, relayout copies,
  tile-count terms.
- **Printer:** its `sym_map` / `buffer_map` come from `NodeVars`, and the model stays
  live if lowering fails partway (:810-837).
- **Uniqueness probe:** per-node `decision_variables` feed #5183's probe. Its warning
  stays on the `ilp_solver_ortools` logger (`test_scratchpad_solver.py:2867`).
- **Deleted:** the wrappers, `_wrap` and the `isinstance` dispatch.
- **Tests updated:**
  - `test_cooptimization_types.py:358-367` (`_LifetimeBufferWithCpVars`);
  - `test_scratchpad_solver.py:2009` (`_gate_divisions`).
- **From the branch:** hooks lift from `9dab5027` `encoding_strategy.py` only where #5246
  did not touch the model.

**4c. `graph.solve(backend, cost_expr, strategy)`.**
- It runs `_finalize`, then today's ladder unchanged (:1369-1450, no tiebreak solves),
  then `_extract` onto the nodes and edges.
- `CpSatLayoutSolver` stays the **only** CP-SAT entry point: `plan_layout*` →
  `build_graph` (or a prebuilt graph, from Stage 7) → `graph.solve` → write-back
  (:753-770). The write-back is the only place that touches records.
- Kept as they are:
  - `record_exclusions()` (:742);
  - `last_solve_stats`, including `{}` for an empty solve and `NOT_LINEARIZABLE`;
  - the ImportError check;
  - `SolveError` as the only failure.

**Check:**
- After each sub-stage: proto diff empty and the same plans.
- After 4c also: the 18 `test_inductor_ops_lx_*` shards, and zoo planning time before
  vs after.

### Stage 5. Placement-only and co-optimizing strategies (item 5)

**5a. Extraction only.**
- **Shared rules.** The residency and in-place helpers go to a rules module that both
  modes use: `_get_op_name`, `_op_inputs_good_for_lx_inplace`, `_inplace_edge_ok`,
  `_residency_reasons`, `_untiled_residency_bar`, `_input_residency_reason`.
- **Mode value.** The `division_is_fixed` branches read a mode value there: 1089, 1102,
  1158, 1230 / 1237, 1283, and the drain validator at 439.
- **`_loop_carry_update_edge`** becomes a module function. `_assert_drain_plan_committed`
  calls it (:725), and moving it with the class would create an import cycle.
- **Cross-method state** gets one explicit home that the base class reads:
  - `_validated_drain_plans`: set at 2968, read at 1944, 3349 and 3790; a verbatim move
    would leave `_push_allocation` reading `{}` and drop the drain clones;
  - `_relayout_pair_costs`;
  - `_prescribed_ops` / `_readers_by_name`;
  - `_relayout_view_groups`;
  - `layout_planning`, `size` and `prune`.

**5b. `PlacementOnlyStrategy`.**
- It takes the five mode hooks: `_prepare_buffers` 868, `_solve` 897,
  `_finalize_lx_relayout_allocation` 901, `_post_solve` 933, `_get_spill_reasons` 942.
- `ScratchpadAllocator` delegates them to `self.strategy`.

**5c. `CoOptimizingStrategy`.**
- **What moves into `co_optimizing.py`:**
  - the overrides: 2957, 3047, 3282, 3333, 3353;
  - menus; match tables, carries and clones; relayout candidates and copies; pricing;
  - the build → price → pin → rebuild loop and `_build_cd_bound_buffers`.
- **The cost dump** keeps today's name (`type(self).__name__`, :3219).
- **`CoOptimizingAllocator`** shrinks to `ScratchpadAllocator` plus this strategy.
- **Tests:** about 70 references in 13 files move with the code. Moved code binds the new
  module's names, so test patches move too.

**Check:**
- Same plans, including the post-push graph and the fallback.
- Protos identical.
- Unit and device suites.

### Stage 6. One allocator; capability rules in the strategies (items 2 and 5)

- **`CoOptimizingAllocator` is deleted.** `select_allocator` returns
  `ScratchpadAllocator(solver_cls, strategy, post_optimization_passes)`.
- **Its mode and solver rules become `strategy.accepts(solver_cls)`** (4978-5012). They
  cover:
  - the SA co-optimizer, which gets no post-passes;
  - exhaustive wrapping;
  - the `ValueError` for co-optimization without a division-capable solver, which is also
    what happens without ortools.
- **Checked in `select_allocator` only, not in the constructor.** Tests build
  `CoOptimizingAllocator(GreedyLayoutSolver, …)` (`test_scratchpad_solver.py:241`,
  `test_work_division_hint.py:815`).
- **Each capability flag** is read only inside the strategy that uses it.
- **The greedy fallback** keeps `post_optimization_passes` (5058-5062).
- **Tests:** the constructors in the 13 test files move.
- **Check:** same plans.

### Stage 7. The graph generates the buffers (item 3)

**`GraphBuilder(strategy, solver_caps).build(scheduler graph)`:**
- The generic part comes from `_generate_buffers` (1680).
- The strategy's build hook makes each record:
  - placement: `_build_bound_buffers` (1425), plus `_append_lx_relayout_destinations`
    (1762) for pairing solvers, which re-ticks to 2u+1;
  - co-optimizing: the 5c code, with its loop run to convergence.
- The builder wires nodes, edges, the chain and `node.buffer` **after** both.

**Who reads what:**
- The allocator keeps the graph. `CpSatLayoutSolver` accepts it prebuilt and does not
  rebuild it, so the harness, `last_solve_stats`, `excluded()` and the dump are unchanged.
- The other solvers and `_push_allocation` read `graph.buffers()`.
- The records stay authoritative after the solve:
  - the SA co-optimizer appends to menus;
  - `_clear_lx_relayout_groups` clears addresses;
  - `apply_chosen_tilings` shifts op indices.
- Exhaustive search, the SA co-optimizer (both build private temporaries) and
  `LxContextSwitchingPass` (it reads only the scheduler graph) work unchanged.

**Check:**
- For one PR, a debug flag compares the records with the old path field by field.
- Then protos identical and the same plans.

### Stage 8. Cleanup

- Docstrings, and a grep for every deleted name.
- Point `cpsat-graph-buffer-refactor.md` at this path.

## Out of scope

- `Lazy`, derived and `Pruned` encodings.
- Pins and `set_solver`.
- The Mip, BruteForce and Annealing backends.
- The other solvers on the graph.
- Deleting `CoreDivisionBuffer`.
- A general `Transform` for restickification and transposes.
- The old plan's Stage 7: it changes tied plans.

## Gates for every PR

- **Proto.** `cpsat_equivalence.py diff` over the unit-test solves plus the default and
  tiling captures, comparing frozen trees of the parent and the change. It must show
  0 differences, and the parameter records must be equal.
- **Plans.** The extended same-plans check from Stage 0 must be identical, including the
  post-push graph, the tiling config and the fallback.
- **Unit suites, run serially:** `test_scratchpad_solver`, `test_solver_relayout_decision`,
  `test_cooptimization_types`, `test_cost_expr_dump`, `test_restickify_cost_model`,
  `test_sa_cooptimizer`, `test_coarse_tiling`, `test_work_division*`,
  `test_scratchpad_patterns`, `test_perm_layout_solver`, `test_scratchpad_use`,
  `test_solver_auto_coarse_tiling`, `test_buffer_graph`.
- **Device suites after Stages 3, 4c, 5c, 6 and 7:** `test_scratchpad_use`,
  `test_work_division`, `test_solver_relayout_e2e`, `test_solver_auto_coarse_tiling`,
  `test_lx_planning_suite`.
- **Housekeeping.** `pre-commit run --files`, and re-record the baselines after every
  rebase onto `main`.

Pitfalls from the branch:
- Build the graph from the allocator's full buffer list, never from a solver's filtered
  view. The placement annealer's `self.buffers` rebinding dropped barred buffers this way.
- Never put a CP-SAT expression in a Python `if`.
- Add no new exception types to the solve path.
- Select no backend from an environment variable.

## References

- PR #5676 and its branch `scratchpad-refactor-experimental`. Commits cited:
  - `9dab5027`: Phase 1, harness, graph, strategy;
  - `cea7f906`: one read gate per listed parent;
  - `a47ef8f2`: backend layer;
  - `2796be28`: the merge that dropped #5246.
- `cpsat-graph-buffer-refactor.md`: the full design this path stages toward.
- Two reports that are not yet in this repository: the feasibility report
  (`cpsat-graph-buffer-refactor-feasibility.md`) and the main-vs-branch comparison
  (`scratchpad-architecture-vs-main.md`). They hold the measurements and the review
  findings quoted above.
