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
  - default compiles plan 21–30% slower;
  - the greedy fallback is gone, and nothing goes in LX without ortools;
  - committed mode is priced like the open mode;
  - completed-reduction relayouts are lost.
- **It never ported #5246.** Merge `2796be28` overwrote the variable lifetimes, and the
  re-solve after tiling came back. `main` is now past #5246 and #5183 (the uniqueness
  check, `58afed9e`), so its tiling and lifetime encodings cannot be lifted.

This plan lands the structure on `main` in small PRs. Every PR keeps the CP model
byte-identical and the committed plans identical. It implements five things:

1. **One integrated graph solver over an injectable OR-Tools backend.** The backend
   protocol is pywraplp-shaped, so any OR-Tools-compliant solver can implement it.
   CP-SAT is the first backend.
2. **One allocator class.** The co-optimizing allocator's division code moves into
   `CoreDivisionStrategy`; it is today's table encoding (menus and match tables), and
   there is no `Lazy`.
3. **A graph that generates the `CoreDivisionBuffer`s itself.**
4. **`DependencyEdge`s and `ExecutionEdge`s.**
5. **Placement-only and co-optimizing as strategies.** They replace the two allocator
   classes and their `division_is_fixed` and capability-flag branches.

Decisions (2026-10-23):
- The graph solver is the only integrated solver. CP-SAT is its first backend.
- Greedy, first-fit, best-fit, placement SA, the SA co-optimizer and exhaustive search
  stay outside the graph model. They are fed by the buffer records the strategies
  generate. Greedy stays the fallback after a failed solve.
- Placement-only reproduces today's plans exactly. `CO_OPTIMIZING_LX_PLANNING=0` stays
  usable for bisecting a miscompile.
- Strategies come in increasing order of difficulty, and each one extends the last:
  placement first, then core division, then co-optimization (relayouts and coarse
  tiling).
- The new encodings are written beside today's CP-SAT code, not by refactoring it in
  place. Each one is checked against today's model on the same input, and today's code
  is deleted once both entry points delegate.

Compatibility with the non-enumerating end state:
- Menus stay inside the records, and only `CoreDivisionStrategy` reads them.
- Node and edge decisions are `WorkSplit`s and relayout choices, never menu indices.
- A later `Lazy` strategy can skip generating the records.

## Terms

| Term | Meaning | Replaces |
|---|---|---|
| graph | `SpyreOptimizationGraph`: `ResourceNode`s, `DependencyEdge`s and `ExecutionEdge`s. It holds incoming data and decisions, with no candidates and no solver state. | the allocator's buffer lists, as the solver's input |
| graph solver | The one integrated solver. It runs a strategy over the graph on a backend, then the objective ladder, then extraction onto the nodes. | `CpSatLayoutSolver`'s model code (the class stays as the entry point) |
| backend | An OR-Tools solver behind the pywraplp-shaped protocol. `CpSat` is the first. | direct `cp_model` calls |
| strategy | Which decisions are open, how the others are filled before encoding, where the candidates come from, and how the open decisions are encoded. | the two allocator classes, `division_is_fixed`, and the old plan's `TableStrategy` and `EncodingStrategy` |
| encoding | The part of a strategy that posts to the backend. | the CP-SAT wrappers' constraint code |
| record | A `LifetimeBoundBuffer`, `CoreDivisionBuffer` or `RelayoutCopyBuffer`. A node references its record, and solvers outside the graph model read records. | — |

The three strategies:

| Strategy | Open decisions | Filled before encoding | Used for |
|---|---|---|---|
| `PlacementStrategy` | residency, offset, in-place merges | the work split (the division work division already chose); no tiling; no relayouts | `co_optimizing_lx_planning=0` |
| `CoreDivisionStrategy` | the above, plus the work split (today's table: menu index and match tables) | no tiling; no relayouts | co-optimizing with `SPYRE_LX_PLANNER_RELAYOUT=0` and coarse tiling off |
| `CoOptimizingStrategy` | the above, plus relayouts and coarse tiling | nothing | `co_optimizing_lx_planning=1` (the default) |

## Order

| # | Stage | Item | Needs | Proven by |
|---|---|---|---|---|
| 0 | Harness and gates | — | — | — |
| 1 | The graph: nodes, dependency and execution edges | 4 | 0 | structural tests |
| 2 | Graph solver, OR-Tools backend, `PlacementStrategy` | 1, 5 | 1 | parity with `plan_layout` |
| 3 | Allocator prep: shared rules and state | — | 0 | plans |
| 4 | `CoreDivisionStrategy`, pulled from the allocator | 2, 5 | 2, 3 | parity with the joint solve, relayouts and tiling off |
| 5a | `CoOptimizingStrategy`: relayouts | 5 | 4 | parity, relayouts on |
| 5b | `CoOptimizingStrategy`: coarse tiling | 5 | 5a | parity, `AUTO_COARSE_TILING=1` |
| 5c | Today's CP-SAT model code deleted | — | 5b | recorded protos |
| 6 | One allocator | 2, 5 | 5b | plans |
| 7 | The graph generates the buffers | 3 | 6 | protos + plans |
| 8 | Cleanup | — | all | — |

That is 11 PRs.

- Stage 3 can run any time after Stage 0, in parallel with Stages 1–2.
- **Parity** means that today's model and the graph solver's model, built from the same
  records in one process, have identical protos and identical decisions. The check is a
  device-free test in CI, so the two cannot drift while both exist (Stages 2–5b).

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
- **Corpus.** The unit-test solves, the synthetic co-optimization graphs, the capture
  loader's graphs, and fresh captures from current `main` (`deenum/run_profile.sh`):
  smoke, zoo default and tiling, and Granite. The parity tests in later stages replay
  this corpus. Add `test_work_division` (#5246's tests) and `test_coarse_tiling` to
  `record_tests.sh`.
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

### Stage 1. The graph (item 4)

The whole structure lands at once, so the encoding tests in later stages run on a
complete graph. No solver changes in this stage. It is a redesign of the branch's
`buffer_graph.py`, not a trim: the branch has no `node.buffer`, and its copies are not
records.

**New `buffer_graph.py`:**
- **`WorkSplit`** carries every `CoreDivision` field, including `reduction_syms` and
  `tile_splits`, which the branch's `of` drops. `COMMITTED` marks a split chosen by work
  division.
- **`ResourceNode`** has:
  - incoming data: name, size in units, start and end with overrides, `read_count`,
    `first_use_is_read`, `boundary`, `produced_at`, `residency_reason`, `untiled_bar`;
  - decisions: `work_split`, `resident`, `offset`, `spill_reason`;
  - links: `prev` / `next` and `inputs` / `outputs`;
  - `buffer`: its record, read only by the strategies.
- **`DependencyEdge`** has `src`, `dst`, `reads`, `in_place`, and the decisions `merged`
  and `relayout`.
  - `reads` is a count, because a parent can be listed twice.
  - An in-place edge may have `reads == 0`.
- **`ExecutionEdge`** has `before`, `after` and `contiguous`.
  - The chain is the op outputs in tick order. Op-output ticks are unique
    (`utils.py:73-110`).
  - Graph inputs and relayout copies are not on the chain.
  - `contiguous` means consecutive ticks. Whether two ops can share a nest is decided by
    the coarse-tiling encoding (Stage 5b), not by the graph.
- **`SpyreOptimizationGraph`** has:
  - `nodes`;
  - `copies`: the relayout copies, in copy order;
  - `first` and `execution_chain()`;
  - `buffers()`: the records, in the allocator's order.
- **`build_graph(records, alignment, capacity)`.** The edges come from the records: a
  read from each listed `parents` entry (only `CoreDivisionBuffer`s carry `parents`),
  and an in-place edge from `in_place_parents`.

**Tests:** a new `test_buffer_graph.py`, registered in `tests/configs`. It checks:
- the edges against `_get_children`;
- `in_place` against `in_place_parents`;
- the chain against the op order today's tiling walks use;
- read counts;
- that `add_child` / `add_successor` refuse duplicates.

### Stage 2. Graph solver, OR-Tools backend and `PlacementStrategy` (items 1 and 5)

**New `backend.py`.** It holds the pywraplp-shaped `Backend` protocol and `CpSat`, lifted
from `a47ef8f2` `backend.py` (up to :175).
- Guard the ortools import as `ilp_solver_ortools` does, and drop the MPSolver-only
  imports.
- Resolve the default backend lazily, so the no-ortools paths do not change.
- The protocol is what another OR-Tools solver implements (the `pywraplp` MPSolver
  backends: SCIP, HiGHS, CBC). None is built here; see *Out of scope* for what one needs.

**New `graph_solver.py`.** `GraphSolver(backend_factory).solve(graph, strategy,
cost_expr)` runs these steps:
1. The strategy fills the decisions it does not open.
2. Its encoding posts the open ones, in today's emission order.
3. The objective ladder runs. The stages are today's (:1369-1450), with no tiebreak
   solves, and each strategy contributes the terms of the stages it opens.
4. Extraction writes the decisions onto the nodes and edges and returns the op outputs
   in execution order.

It also covers:
- the solver parameters (time limit, workers, seed, presolve rule), set as `_run` sets
  them today (:1312-1342);
- #5183's uniqueness probe, over the strategy's decision variables;
- `SolveError` as the only failure.

**`PlacementStrategy`:**
- **Fill:** each node's `work_split` is `COMMITTED`, with no tiling and no relayouts.
- **Encoding**, per node:
  - the residency literal, the offset, and a merge literal per in-place input;
  - the merges: offset equality, both ends resident, and at-most-one in and out;
  - capacity;
  - the 2-D no-overlap, with the incoming-merge shortening;
  - the exclusions from `record_exclusions()`, pinned with their reasons;
  - a node with an `untiled_bar` pinned out, since nothing here can tile it
    (:1724-1731); its spill reason is the bar (:760-761);
  - the residency stage (spill cost).

**`CpSatLayoutSolver.plan_layout`** delegates to `GraphSolver` with `PlacementStrategy`
when no record carries a menu. The class stays the only CP-SAT entry point, so the
harness, `last_solve_stats`, `record_exclusions()`, the ImportError check and the dump
are unchanged.

**Check:**
- The parity test (`test_graph_solver_parity.py`, device-free, in CI). On every corpus
  input without menus, `plan_layout`'s model and the graph solver's model must be
  identical and give identical decisions.
- Same plans for `co_optimizing_lx_planning=0` with `layout_solver=cpsat`.

### Stage 3. Allocator prep (no behaviour change)

`CoreDivisionStrategy` pulls code out of `CoOptimizingAllocator`, but the two allocator
classes share state that a plain move would break. This stage gives that state an
explicit home first.
- **Shared rules.** The residency and in-place helpers go to a rules module that both
  modes use: `_get_op_name`, `_op_inputs_good_for_lx_inplace`, `_inplace_edge_ok`,
  `_residency_reasons`, `_untiled_residency_bar`, `_input_residency_reason`.
- **Mode value.** The `division_is_fixed` branches read a mode value there: 1089, 1102,
  1158, 1230 / 1237, 1283, and the drain validator at 439.
- **`_loop_carry_update_edge`** (4135) becomes a module function.
  `_assert_drain_plan_committed` calls it (:725), and moving it with the class would
  create an import cycle.
- **Cross-method state** gets one explicit home that the base class reads:
  - `_validated_drain_plans`: set at 2968, read at 1944, 3349 and 3790. Moved
    verbatim, `_push_allocation` would read `{}` and silently drop the drain clones.
  - `_relayout_pair_costs` (2948);
  - `_prescribed_ops` / `_readers_by_name`;
  - `_relayout_view_groups`;
  - `layout_planning`, `size` and `prune`.
- **Check:** same plans, including the post-push graph.

### Stage 4. `CoreDivisionStrategy`, pulled from the allocator (items 2 and 5)

It extends `PlacementStrategy` and opens the work split. The old plan's `TableStrategy`
is this strategy's encoding, not a strategy of its own: the menus and match tables are
how core divisions are chosen today.

**Candidates, pulled from `CoOptimizingAllocator`:**
- menus: `_division_map` (3376), `_enumerate_core_divisions` (3511),
  `_tiling_candidates` (3588) and `_division_space` (4303);
- match tables: `_cd_parent_matches` (4475) and `_parent_residency_edges` (4445);
- loop carries: 4135 and 4185;
- clones: 4324 and 4344;
- `_residency_by_buf` (3729);
- the build → price → pin → rebuild loop (2981-3012) and `_build_cd_bound_buffers`
  (3758);
- the pricing behind the cost expression: 3015, 3032, 3237, and the cost bundle in
  `_solve` (3047).

The SA co-optimizer and exhaustive search read the same records, so they keep working
unchanged.

**Encoding:**
- the menu index and its element ties: footprint, cores, core cost, and split and
  tile-count vars;
- the cached `division_is` literals that `cpsat_expr_printer` duck-types;
- the read gates (match tables) and the in-place view gates;
- the parallelism and shape ladder stages;
- the cost expression, through the printer's `sym_map` / `buffer_map`. The model stays
  live if lowering fails partway (:810-837).

**Fill:** no relayouts and no tiling.

**Check:** parity with `plan_layout_and_core_divisions` under `SPYRE_LX_PLANNER_RELAYOUT=0`
with coarse tiling off. That configuration then delegates.

### Stage 5. `CoOptimizingStrategy`: relayouts and coarse tiling (item 5)

It extends `CoreDivisionStrategy` and reuses its division vars and view gates.

**5a. Relayouts.**
- Candidates: `_cd_parent_relayouts` (4606), `_cap_relayout_groups` (4497),
  `_relayout_consumer_costs` (4572), `_views_for_divs` (4875) and
  `_relayout_copy_buffers` (4226).
- The copies become `graph.copies`.
- Encoding:
  - the served literals and the "match or served" gates;
  - copy use (today's `_constrain_relayout_copies`);
  - the relayout charges in the cost, and the fallback copy pin.
- After the solve: `_finalize_lx_relayout_allocation` (3282).
- Check: parity with relayouts on and tiling off.

**5b. Coarse tiling.** Re-derive it from `main`'s #5246 code; the branch has none.
- `loop_id` / `tiled`.
- Groups, nest ends and nest starts along the execution edges: `_add_tiling_groups`
  (:1051), `_add_nest_ends` (:990), `_add_nest_starts` (:1023).
  - Grouping uses consecutive ticks with a tileable predecessor, which reproduces
    :1071-1108.
  - The `tile_group_` domain stays `len(position)` (:1102).
- The cut literals, with `apart_` / `via_full_` per dependency edge (:930, :942).
- The per-node partial-reduction guard (:961).
- The `untiled_bar` rule: a node with a bar that can be tiled is resident only when tiled
  (:1728-1731).
- Lifetime ends and slots (`_add_lifetime_ends` :1142, `_Slots`).
- The tile-count and cut-traffic terms, and the cut stage.
- After the solve: `apply_chosen_tilings`.
- Only the group and nest walks follow the chain. `loop_id` / `tiled`, the cut loop and
  the tile-count terms keep node order, because unit-test record lists are not in tick
  order.
- Check: parity with `AUTO_COARSE_TILING=1`, on the tiling captures and on
  `test_work_division` / `test_coarse_tiling`.

**5c. Today's model code is deleted.** `plan_layout_and_core_divisions` delegates fully.
- Deleted: the wrappers, `_wrap`, the model posting in `_run`, `_get_children` and
  `_relayout_copies`.
- Tests updated: `test_cooptimization_types.py:358-367` (`_LifetimeBufferWithCpVars`)
  and `test_scratchpad_solver.py:2009` (`_gate_divisions`).
- The parity test becomes a recorded-baseline test.
- Check: the recorded protos and plans equal, and the 18 `test_inductor_ops_lx_*` shards
  pass.

### Stage 6. One allocator (items 2 and 5)

- **The remaining mode hooks move into the strategies.**
  - `PlacementStrategy` takes `ScratchpadAllocator`'s: `_prepare_buffers` 868, `_solve`
    897, `_finalize_lx_relayout_allocation` 901, `_post_solve` 933 and
    `_get_spill_reasons` 942.
  - `CoOptimizingStrategy` takes whatever is left of `CoOptimizingAllocator`'s overrides
    (2957, 3047, 3282, 3333, 3353).
  - The cost dump keeps today's name (`type(self).__name__`, :3219).
- **`CoOptimizingAllocator` is deleted.** `select_allocator` returns
  `ScratchpadAllocator(solver_cls, strategy, post_optimization_passes)`.
  - The CP-SAT path runs the graph solver.
  - The solvers outside the graph model take the strategy's records.
- **The mode and solver rules become `strategy.accepts(solver_cls)`** (4978-5012). They
  cover:
  - the SA co-optimizer, which gets no post-passes;
  - exhaustive wrapping;
  - the `ValueError` for co-optimizing without a division-capable solver, which is also
    what happens without ortools.

  They are checked in `select_allocator` only, not in the constructor, because tests
  build `CoOptimizingAllocator(GreedyLayoutSolver, …)` (`test_scratchpad_solver.py:241`,
  `test_work_division_hint.py:815`).
- **Each capability flag** is read only inside the strategy that uses it.
- **The greedy fallback** keeps `post_optimization_passes` (5058-5062).
- **Tests:** about 70 references in 13 files move with the code. Moved code binds the new
  module's names, so test patches move too.
- **Check:** same plans.

### Stage 7. The graph generates the buffers (item 3)

**`GraphBuilder(strategy, solver_caps).build(scheduler graph)`:**
- The generic part comes from `_generate_buffers` (1680).
- The strategy makes each record:
  - placement: `_build_bound_buffers` (1425), plus `_append_lx_relayout_destinations`
    (1762) for pairing solvers, which re-ticks to 2u+1;
  - the co-optimizing strategies: their Stage 4–5 candidate code, with the price loop
    run to convergence.
- The builder wires nodes, edges, the chain and `node.buffer` **after** both.

**Who reads what:**
- The allocator keeps the graph. `CpSatLayoutSolver` accepts it prebuilt and does not
  rebuild it.
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

- **A second OR-Tools backend.** The protocol admits one, but building it needs two
  things.
  - **Linearization.** The model uses CP-SAT-only constraints: element / table, optional
    intervals with the 2-D no-overlap, and products. The branch's `Mip` linearized them
    with big-M, one-hot and pairwise disjunctions. It was exact on small models, but
    found no solution on attention graphs in 60 s.
  - **The cost printer.** `cpsat_expr_printer` builds the cost with cp_model's own
    expression types, so it has to post through the protocol first.
- `Lazy`, derived and `Pruned` encodings.
- Pins and `set_solver`.
- Moving the solvers outside the graph model onto it.
- Deleting `CoreDivisionBuffer`.
- A general `Transform` for restickification and transposes.
- The old plan's Stage 7: it changes tied plans.

## Gates for every PR

- **Parity (Stages 2–5b).** The side-by-side test passes: identical protos and decisions
  on the whole corpus.
- **Proto (all stages).** `cpsat_equivalence.py diff` over the unit-test solves plus the
  default and tiling captures, comparing frozen trees of the parent and the change. It
  must show 0 differences, and the parameter records must be equal.
- **Plans.** The extended same-plans check from Stage 0 must be identical, including the
  post-push graph, the tiling config and the fallback.
- **Unit suites, run serially:** `test_scratchpad_solver`, `test_solver_relayout_decision`,
  `test_cooptimization_types`, `test_cost_expr_dump`, `test_restickify_cost_model`,
  `test_sa_cooptimizer`, `test_coarse_tiling`, `test_work_division*`,
  `test_scratchpad_patterns`, `test_perm_layout_solver`, `test_scratchpad_use`,
  `test_solver_auto_coarse_tiling`, `test_buffer_graph`, `test_graph_solver_parity`.
- **Device suites after Stages 2, 4, 5a, 5b, 6 and 7:** `test_scratchpad_use`,
  `test_work_division`, `test_solver_relayout_e2e`, `test_solver_auto_coarse_tiling`,
  `test_lx_planning_suite`.
- **Housekeeping.** `pre-commit run --files`, and re-record the baselines after every
  rebase onto `main`.

Pitfalls from the branch:
- Build the graph from the allocator's full record list, never from a solver's filtered
  view. The placement annealer's `self.buffers` rebinding dropped barred buffers this way.
- Never put a CP-SAT expression in a Python `if`.
- Add no new exception types to the solve path.
- Select no backend from an environment variable.
- While both CP-SAT builders exist, any change to today's model needs the same change in
  the graph solver. The parity test fails until both match.

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
