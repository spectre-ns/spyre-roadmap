# CP-SAT as a graph of buffer nodes, dependency edges and execution edges

## Context

The CP-SAT layout solver (`torch_spyre/_inductor/scratchpad/ilp_solver_ortools.py`,
1508 lines) dispatches on five buffer types: the allocator's `LifetimeBoundBuffer`,
`CoreDivisionBuffer` and `RelayoutCopyBuffer` (`plan_solver.py`), wrapped in its own
`_LifetimeBufferWithCpVars` / `_CoreDivisionBufferWithCpVars`. The distinctions are
really "which decisions are already made": a placement-only buffer is one whose
division is fixed, a barred buffer one whose residency is fixed `False`, a dead relayout
copy one pinned out. Relationships are re-derived ad hoc (`_get_children`,
`_relayout_copies`, `merge_vars` on wrappers), and op order is recovered from
`uses[0]` inside `_tiling_group_ids`.

Goal: model the solve as a graph.

- **Nodes** — one type, `ComputeNode`. Its decision fields (`division`,
  `resident`, `offset`) are fixed ahead of time or `None` for the solver to find.
- **`DependencyEdge`** — every relationship between two nodes (reads, in-place reuse,
  relayout copies). Edges are shared: the same edge object sits in its source's
  `outputs` and its destination's `inputs`, so either end can walk it.
- **`ExecutionEdge`** — links each op output to its chronological neighbours
  (`node.prev` / `node.next`). Coarse-tiling state lives here: whether two adjacent
  ops run the same loop nest, and where a tiling group breaks.
- **`Graph`** — owns the nodes and edges, and holds the global context (alignment,
  LX capacity, exclusions, whether relayouts are priced). `Graph.timeline` returns the
  op-output nodes in execution order.

The work is split into small stages. Each one lands as its own PR, leaves the full
suite green, and gets an empty equivalence diff (see *Verification*), so if behaviour
changes, the stage that caused it is clear. **Phase 1 is CP-SAT only** and keeps the
allocator's three classes as the external contract. **Phase 2** pushes the node type
out to the allocator and the other solvers.

Work happens in worktree `.claude/worktrees/cpsat-graph-model` (branch
`cpsat-graph-model`, off `main` 82edd8f6). No commits (user commits).
`scratchpad/buffer_graph.py` is already drafted there, with
`SliceEdge`/`InPlaceEdge`/`CopyEdge` and `Timeline.runs`. Stages 1–2 reshape it into
the model below.

## The model

```python
class DependencyKind(Enum):
    READ = auto()          # dst reads src; residency gate + slicing match
    IN_PLACE = auto()      # dst may take over src's storage at the shared tick
    RELAYOUT_COPY = auto() # dst is a relayout destination of src

@dataclass(eq=False)
class DependencyEdge:
    src: ComputeNode
    dst: ComputeNode
    kind: DependencyKind
    pairs: Optional[Pairs]              # (src div, dst div) that slice alike; None = no division gate
    relayouts: tuple[RelayoutRead, ...] = ()   # READ only: copies that can serve this read
    priced: tuple[int, ...] = ()               # RELAYOUT_COPY only: src divisions the copy is priced for

@dataclass(eq=False)
class ExecutionEdge:
    before: ComputeNode
    after: ComputeNode
    contiguous: bool   # produced on consecutive ticks (nothing outside the solve runs between)

    @property
    def joinable(self) -> bool:  # can the two ops share a loop nest at all?
        return self.contiguous and self.before.tileable and self.after.tileable

@dataclass(eq=False)
class ComputeNode:                  # the node
    ... # name, size (units), core_divisions, start/end_time, spill_cost, boundary,
        # sym_core_divs, sym_tile_counts, division/resident/offset (None = free)
    produced_at: Optional[int]      # tick of the producing write; None for inputs and copies
    inputs: list[DependencyEdge]
    outputs: list[DependencyEdge]
    prev: Optional[ExecutionEdge]
    next: Optional[ExecutionEdge]

class Graph:
    alignment: int
    capacity: int                    # LX capacity in alignment units
    exclusions: dict[str, str]       # name -> residency_reason (record_exclusions)
    shuffles_priced: bool            # False under the fallback objective: copies pinned out
    nodes: dict[str, ComputeNode]
    edges: list[DependencyEdge]      # in the order the solver adds their constraints

    def connect(self, edge) -> None  # the only way to add an edge; appends to both ends
    def edges_of(self, kind) -> Iterator[DependencyEdge]
    @property
    def timeline(self) -> list[ComputeNode]  # op outputs by produced_at
    def tiling_segments(self) -> list[list[ComputeNode]]  # maximal joinable chains
```

Notes:

- `eq=False` on nodes and edges: they reference each other, so they compare by identity
  and get a short custom `__repr__` (name, or `src->dst:kind`).
- A pair of nodes can share more than one edge (an in-place child usually reads its
  parent too). The solver iterates edges by kind, so there is no `isinstance` dispatch.
- Only `connect` and the builder mutate the wiring. Once the solve starts, only decision
  fields change.
- `ExecutionEdge`s link *all* op outputs in the solve, not just the tileable ones.
  `joinable` reproduces today's rule that an untileable op, or a gap in ticks, splits a
  tiling segment.

## Phase 1 — CP-SAT internals

### Stage 0. Equivalence harness and baseline

- Add a test helper (`tests/inductor/cpsat_equivalence.py`) that solves a buffer set
  with `torch.use_deterministic_algorithms(True)` (1 worker). It dumps residency,
  chosen division, offset, chosen relayouts and each ladder stage's objective to JSON.
- Run it over `synthetic_cooptimization_graphs.py` plus the captured graphs that
  `cooptimization_capture_loader` provides; store the base dumps outside the repo.
- Record the baseline (below) on a frozen `main` copy.
- No production code changes.

### Stage 1. Data model

- In `scratchpad/buffer_graph.py`: `DependencyKind`, `DependencyEdge`, `ExecutionEdge`,
  `Graph` (context, `connect`, `edges_of`, `timeline`, `tiling_segments`), and the node
  gains `produced_at`/`inputs`/`outputs`/`prev`/`next`.
- `spill_cost` and the node's derived properties (`sym_is_lx`, `sym_division`,
  `per_core_sizes`, `cores_used`, `shape_costs`, `tileable`) stay as drafted.
- Drop `SliceEdge`/`InPlaceEdge`/`CopyEdge`/`Timeline` from the draft.
- Tests (`tests/inductor/test_buffer_graph.py`, no device): `connect` shares one edge
  object between both ends; `timeline` order; execution chain links and `contiguous`;
  `tiling_segments` splits at gaps and at untileable ops; node fixing rules (single
  candidate fixes `division=0`, out-of-range division asserts).

### Stage 2. Builder

- `build_graph(buffers, alignment, capacity, exclusions, shuffles_priced)` is the only
  code that looks at the allocator classes. It encodes what is already known:
  - `resident=False` for excluded buffers, and for copies whose source does not choose
    a division or that no READ edge can use.
  - READ edges (with `pairs` and `relayouts`) only between nodes that both choose a
    division (this matches today's no-op `constrain_residency` on placement-only
    wrappers).
  - IN_PLACE edges with `pairs=None` when the child's division was fixed upstream.
  - RELAYOUT_COPY edges for live copies.
  - `produced_at` from `uses[0]` for op outputs; `ExecutionEdge`s along the timeline.
  - Edge order matches today's constraint order (READ by producer order, then
    RELAYOUT_COPY, then IN_PLACE), so model building stays deterministic.
- Tests: for every synthetic graph, the edges must match what `_get_children`,
  `_relayout_copies` and `in_place_parents` give today. This cross-check is deleted in
  Stage 3 along with those helpers.

### Stage 3. Solver reads relationships from the graph

- `_plan_layout_generic` builds the `Graph` next to the existing wrappers.
- `_get_children` → `node.outputs` of kind READ; `_relayout_copies` →
  `graph.edges_of(RELAYOUT_COPY)`; the `in_place_parents` loop in
  `_add_inplace_relaxation` → `graph.edges_of(IN_PLACE)`.
- Delete `_get_children` and `_relayout_copies`. The wrappers still own the CP vars.
- Equivalence diff must be empty.

### Stage 4. Tiling on execution edges

- Per joinable `ExecutionEdge`: a `same_nest` and a `joined` literal (today's
  `samenest_*` / `joined_*`), and the group id steps across the edge
  (`group[after] = group[before] + 1 - joined`). Non-joinable edges start a new segment
  with group 0.
- `_tiling_group_ids` becomes a walk over `graph.tiling_segments()`. Its `uses[0]`
  position logic and the `RelayoutCopyBuffer` check go away.
- `_cut_literals`: for each tiled node, the consumers on its READ edges in the same
  segment get an `apart` literal over group ids, gated on `pairs`; a consumer outside
  the segment, or a graph output, makes the node unshareable. Same rule as today, over
  edges.
- `_tile_count_terms` iterates nodes.
- Equivalence diff must be empty (run the coarse-tiling suites with
  `auto_coarse_tiling` on).

### Stage 5. `_NodeVars`

- One `_NodeVars` per node replaces the variable-making half of both wrappers. A free
  field gets a CP var; a fixed one gets a constant:
  - `division` → `new_constant(k)` with constant `eff_size`/`cores`/`core_cost`/
    `splits`. This generalises today's single-candidate fast path to any fixed `k`.
  - `resident`/`offset` → a var pinned with `model.add(var == value)`, because interval
    presence needs a literal.
- It exposes `buffer`, `division` and `division_is` for `cpsat_expr_printer`, so that
  contract is unchanged.
- For now the wrappers hold a `_NodeVars` and delegate to it, so this stage changes no
  constraint code.
- Equivalence diff must be empty.

### Stage 6. Constraints per edge; delete the wrappers

- `_constrain_residency(model, graph, vars)` runs per READ edge (slicing match, or a
  resident copy from `relayouts`) and returns the served literals.
- `_constrain_relayout_copies` runs per RELAYOUT_COPY edge (`priced`).
- `_add_inplace_relaxation` gives each IN_PLACE edge a merge literal, with a division
  gate only when `pairs is not None`.
- `_add_no_overlap_2d` keeps its logic, now over `_NodeVars`.
- `free_copies` (the presolve threshold) counts RELAYOUT_COPY edges whose source
  chooses a division.
- Delete `_LifetimeBufferWithCpVars`, `_CoreDivisionBufferWithCpVars`, `_wrap`,
  `_add_core_division`, every `isinstance(..., RelayoutCopyBuffer/CoreDivisionBuffer)`
  and every `getattr(..., "core_divisions")` inside the solve.
- Equivalence diff must be empty. Run the device suites.

### Stage 7. Objective from free decisions only

- Parallelism, shape and tile-count terms come only from nodes with `division is None`.
  Constant terms cannot move the optimum, so a stage with no free terms is skipped.
  That removes the no-op re-solves a placement-only solve runs today.
- This is the only Phase 1 stage meant to change what is observable: ladder objectives
  shift by constants and fewer stages run. The equivalence diff for this stage compares
  decisions only; objectives may differ by a constant per stage.

### Stage 8. Extract onto nodes; honour pre-fixed decisions

- `_extract` fills `node.resident/division/offset` (the solver fills in the undefined
  fields) and returns the chosen relayouts per consumer.
- Bottom-justify is skipped when any offset was pinned.
- `_plan_layout_generic` writes nodes back to the caller's buffers: `address` in bytes,
  `spill_reason`, `chosen_division` only where the caller enumerated divisions, and
  `chosen_relayouts` scaled.
- Tests: a solve with a pre-fixed `division`/`resident`/`offset` honours it, and every
  free field comes back filled.
- Run the device suites.

### Stage 9. Cleanup

- Rewrite the module docstring around nodes, dependency edges, execution edges and the
  timeline; keep docstrings short.
- `tests/inductor/test_cooptimization_types.py:361-372` → compare against
  `buffer_graph.spill_cost(b)`.
- Update the comments that cite the old classes by name: `plan_solver.py:152`,
  `sa_cooptimizer.py:1295`, `allocator.py:2696`, `cpsat_expr_printer.py:809`.
- Register `test_buffer_graph.py` in a `tests/configs/` shard; `make check-all-configs`.

Kept unchanged through Phase 1: `_gate_divisions`, `_PlacementUnit`, `_justify`,
`_solve_and_record`, the capability flags, `__init__`, and the public `plan_layout` /
`plan_layout_and_core_divisions`.

### Verification (every stage)

- Unit suites, serial (conftest opens the card; xdist workers collide), on a frozen
  `git archive` of the change vs the frozen base (`~/gap-probe/cpsat-graph-model/base`):
  **1121 passed, 5 skipped, 11 xfailed** over test_scratchpad_solver,
  test_solver_relayout_decision, test_cooptimization_types, test_cost_expr_dump,
  test_restickify_cost_model, test_sa_cooptimizer, test_op_features,
  test_coarse_tiling, test_scratchpad_patterns, test_perm_layout_solver, plus
  test_buffer_graph from Stage 1.
- Equivalence diff from Stage 0: empty, except Stage 7 as noted.
- After Stages 4, 6 and 8: the device suites that compile through CP-SAT —
  test_scratchpad_use, test_work_division, test_solver_relayout_e2e,
  test_solver_auto_coarse_tiling — base vs change, same serial setup.
- `pre-commit run --files <changed>` (ruff, mypy).

## Phase 2 — outline (separate plan after Phase 1 lands)

Make `ComputeNode` and `Graph` the allocator's output and retire
`LifetimeBoundBuffer`/`CoreDivisionBuffer`/`RelayoutCopyBuffer`. Each stage below is
also its own PR, green on the full suite. Sized from the inventory:

- **2a. Node absorbs the allocator-only fields.** The node (moved into
  `plan_solver.py`) gains what the allocator's commit path reads: `in_place_parents`
  (→ IN_PLACE edges), `residency_reason` (→ `Graph.exclusions`), lifetime overrides,
  `lx_view`, `spill_reason`, `chosen_relayouts`, `lx_relayout_plans`, `paired_with`
  (greedy only). From the SA co-optimizer it gains `division_space`, `residency_edges`
  and `tile_reads`; `op_position` is replaced by `produced_at`/execution edges. Sizes
  stay in bytes on the node and CP-SAT converts to units internally. `address` stays
  in bytes (every solver writes bytes today).
- **2b. The allocator builds nodes and the graph** at its four build sites in
  `allocator.py`: `_build_bound_buffers` (:1456/:1497),
  `_append_lx_relayout_destinations` (:1819), `_build_cd_bound_buffers`
  (:4074/:4275), and `_relayout_copy_buffers` (:4418, via `build_relayout_copy`). Its
  4 isinstance checks (:713, :2769, :2964, :3099) become
  `graph.edges_of(RELAYOUT_COPY)` lookups. `build_graph` then reads nodes, not classes.
- **2c. Placement-only solvers** (greedy, first-fit/best-fit, SA, cooling_schedules,
  utils.plot_buffers) only read fields (~90 lines). Swap the type and drop greedy's
  isinstance at :183.
- **2d. Exhaustive search and the SA co-optimizer** read `parents`/`cd_parent_matches`
  from READ edges and write `division`/`resident` on nodes. SA's duplicated
  `_spill_cost` collapses onto `buffer_graph.spill_cost`, and its op adjacency moves
  onto execution edges.
- **2e. Tests**: ~180 constructor calls across 13 files, mostly through helpers
  (`make_buffer_registry`, `synthetic_cooptimization_graphs._buf`,
  `cooptimization_capture_loader._buf`, `test_perm_layout_solver._buf`). Switch the
  helpers first, then the direct calls in `test_scratchpad_solver.py` (131) and
  `test_solver_relayout_decision.py`.
