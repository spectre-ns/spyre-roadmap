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
- **`DependencyEdge`** — every data dependency between two nodes (reads, in-place
  reuse, relayout copies). The edge owns the core-division mapping across it: which
  (src division, dst division) pairs slice the shared buffer alike, and which pairs a
  relayout copy can bridge. Edges are shared: the same edge object sits in its
  source's `outputs` and its destination's `inputs`, so either end can walk it.
- **`ExecutionEdge`** — links each op output to its chronological neighbours
  (`node.prev` / `node.next`). Coarse tiling is handled here: which division pairs run
  the same loop nest, whether the two ops join one tiling group, and where a group
  breaks.
- **`Graph`** — a collection of nodes (which own the edges), plus the global context
  (alignment, LX capacity, exclusions, whether relayouts are priced).
  `graph.solve()` fills in every free decision and returns the op-output nodes in
  execution order, which is the execution plan.

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
    READ = auto()          # dst's op reads src; residency gate + slicing match
    IN_PLACE = auto()      # dst may take over src's storage at the shared tick
    RELAYOUT_COPY = auto() # dst is a relayout destination of src

@dataclass(frozen=True)
class DivisionMap:
    """(src division, dst division) pairs under which both ops own the shared buffer
    with the same per-core slicing (today's cd_parent_matches entry for the edge)."""
    pairs: tuple[tuple[int, int], ...]

    def allows(self, src_div: int, dst_div: int) -> bool
    def dst_options(self, src_div: int) -> frozenset[int]  # consumer divisions that read src_div in place
    def src_options(self, dst_div: int) -> frozenset[int]  # producer divisions dst_div can read in place

class GraphElement(ABC):
    """Base of nodes and both edge types. Built piecewise; finalized and encoded only
    once the whole graph exists."""
    def _finalize(self, model) -> None  # derive what needs the rest of the graph, then
                                        # post this element's vars / constraints into model

@dataclass(eq=False)
class DependencyEdge(GraphElement):
    src: ComputeNode
    dst: ComputeNode
    kind: DependencyKind
    division_map: Optional[DivisionMap]       # None = dst's division fixed upstream, no gate
    relayouts: tuple[RelayoutRead, ...] = ()  # READ only: pairs a resident relayout copy bridges
    priced: tuple[int, ...] = ()              # RELAYOUT_COPY only: src divisions the copy is priced for

    def slices_alike(self, src_div: int, dst_div: int) -> bool  # division_map, or True if ungated
    def bridges(self, src_div: int, dst_div: int) -> tuple[RelayoutRead, ...]  # copies that serve the pair

@dataclass(eq=False)
class ExecutionEdge(GraphElement):
    before: ComputeNode
    after: ComputeNode
    contiguous: bool  # produced on consecutive ticks (nothing outside the solve runs between)
    nest_pairs: tuple[tuple[int, int], ...]  # (before div, after div) that run the same non-empty loop nest

    @property
    def joinable(self) -> bool:  # can the two ops share a loop nest at all?
        return self.contiguous and self.before.tileable and self.after.tileable

    def joins(self, before_div: int, after_div: int) -> bool  # same tiling group under this pair

@dataclass(eq=False)
class ComputeNode(GraphElement):     # the node
    ... # name, size (units), start/end_time, spill_cost, boundary,
        # sym_core_divs, sym_tile_counts
    # Candidates: core_divisions[i] is a split across cores and core_divisions[i].tiling
    # the coarse tiling that goes with it, so one index picks both.
    core_divisions: list[CoreDivision]
    division: Optional[int]         # chosen candidate; None = free
    resident: Optional[bool]        # None = free
    offset: Optional[int]           # None = free

    @property
    def tilings(self) -> list[TileSpec]   # core_divisions[i].tiling for each candidate
    @property
    def tiling(self) -> Optional[TileSpec]  # the chosen candidate's tiling, once division is set
    @property
    def tileable(self) -> bool      # some candidate tiling is not untiled

    produced_at: int                # tick of the producing write
    inputs: list[DependencyEdge]
    outputs: list[DependencyEdge]
    prev: ExecutionEdge             # always linked: the chain runs graph.head -> ... -> graph.tail
    next: ExecutionEdge

    # decision vars: created by _finalize(model), read back by extraction

    def add_child(
        self,
        child: ComputeNode,
        kind: DependencyKind = DependencyKind.READ,
        *,
        geometry: Optional[ResidencyEdge] = None,  # derive the division map from the ops
        pairs: Optional[Pairs] = None,             # or take it as given
    ) -> DependencyEdge  # the only way to add an edge: derives its mapping, wires both ends
    def add_successor(self, after: ComputeNode) -> ExecutionEdge
        # the only way to add an execution edge: sets self.next / after.prev, derives
        # nest_pairs from both nodes' candidate tilings and contiguous from produced_at

class Graph:
    alignment: int
    capacity: int                    # LX capacity in alignment units
    exclusions: dict[str, str]       # name -> residency_reason (record_exclusions)
    shuffles_priced: bool            # False under the fallback objective: copies pinned out
    nodes: dict[str, ComputeNode]    # insertion order = add_node order
    head: ComputeNode                # boundary sentinels closing the execution chain;
    tail: ComputeNode                # edges to them are never joinable

    def add_node(self, buffer) -> ComputeNode  # makes and registers the node; no vars yet
    def edges_of(self, kind) -> Iterator[DependencyEdge]  # walks nodes' outputs; no edge list
    def solve(
        self,
        model: cp_model.CpModel,
        cost_expr: Optional[sympy.Expr] = None,  # None or unlowerable: fallback objective
    ) -> list[ComputeNode]  # the public entry point: finalize into model, run the
                            # ladder, fill every None field; op outputs in order
    def _finalize(self, model) -> None  # nodes, then dependency edges, then execution edges,
                                        # then the graph-wide constraints
    def tiling_segments(self) -> list[list[ComputeNode]]  # maximal joinable chains
```

Notes:

- `eq=False` on nodes and edges: they reference each other, so they compare by identity
  and get a short custom `__repr__` (name, or `src->dst:kind`).
- A pair of nodes can share more than one edge (an in-place child usually reads its
  parent too). The solver iterates edges by kind, so there is no `isinstance` dispatch.
- `parent.add_child(child, kind)` is the only way to add an edge. It ports the
  allocator's per-edge logic: with `geometry` (the `ResidencyEdge` built from the two
  live ops) it computes the `DivisionMap` itself, as `_cd_parent_matches` does today.
  The priced `relayouts` wait for finalization (below). Graphs with no live ops
  (the synthetic and captured test graphs) pass `pairs` instead. Either way it puts the
  one edge object on both ends: the source's `outputs` and the destination's `inputs`.
  The graph keeps no edge list of its own. Once the solve starts, only decision
  fields change.
- Building is piecewise and does structure only: `add_child` and `add_successor`
  never compute anything that depends on another edge, and nothing touches the model.
  Everything else waits until all children are present. There is no separate encode
  step: `solve(model, cost_expr)` passes the caller's model to `Graph._finalize(model)`, and each
  element's `_finalize(model)` both derives what needs the rest of the graph and posts
  its own vars and constraints. Order:
  1. **Nodes**, in insertion order (today's var order). A node's `_finalize`:
     - read counts → `spill_cost` and the "no LX reader" exclusion (`resident=False`);
     - for a relayout copy: pinned out if its source does not choose a division or no
       consumer's capped relayouts use it;
     - its decision vars (a free field gets a CP var, a fixed one a constant);
     - in Phase 2, lifetimes too, once `end_time` comes from the last child.

     Relayout capping (`_cap_relayout_groups`, which keeps any consumer division that
     another parent already matches for free) is a cached computation on the consumer
     over all of its READ `inputs`. Whichever of the consumer or a copy finalizes first
     triggers it, so the result doesn't depend on node order.
  2. **Dependency edges**, by kind in today's order (READ, then RELAYOUT_COPY, then
     IN_PLACE), then nodes in insertion order, then each node's `outputs` in add order.
     That reproduces today's constraint order. Each posts its constraints:
     `_gate_divisions` over `division_map.pairs`, the served literals over
     `relayouts`, and the in-place merge.
  3. **Execution edges**, along the chain: the `same_nest` / `joined` literals and
     the group-id step, which also give the tiling segments.
  4. **Graph-wide**: cuts (each tiled node's READ edges against the groups), the 2-D
     no-overlap, and the objective terms.

  The edges' queries (`slices_alike`, `bridges`, `joins`) need no model.
- `solve(model, cost_expr)` is the only public entry point after building: it runs
  `_finalize(model)`, then the objective ladder over the sympy `cost_expr` (lowered
  as `_minimize_cost_expr` does today, with today's fallback objective when it is
  `None` or fails to lower), then extraction, and fills in every field that was
  `None`.
- This interface stays inside the CP-SAT solve until the problem definition is unified
  (Phase 2a, `graph.set_solver(...)`). Only after that is it pushed out to the
  allocator.
- `DivisionMap` is read in both directions: CP-SAT gates on `pairs`, while a solver
  that generates divisions asks `dst_options`/`src_options` for the other end of an edge.
- Every division index on an edge (`DivisionMap.pairs`, `nest_pairs`, `relayouts`) is
  an index into the end node's `core_divisions`, so a pair fixes both the core split
  and the coarse tiling of each end.
- `Optional` / `None` on a node means exactly one thing: an open decision that
  `solve()` fills (`division`, `resident`, `offset`). Structural fields (`produced_at`,
  `prev`, `next`, the edge lists) are never `Optional`, since execution order is fixed
  in this plan.
- `add_child` records data dependencies; `add_successor` records execution order. The
  order is a single chain from `graph.head` to `graph.tail`: `add_successor` refuses
  to relink a node that already has a successor or predecessor, so an op cannot be
  inserted between two linked ones later, and `_finalize` asserts the chain is
  complete.
- Open: graph inputs and relayout copies have no slot of their own in the op order
  today. An input's clone-in happens at its first use, and a copy's shuffle is
  scheduled directly before its first consumer, but only if the solve keeps the copy.
  How they get `produced_at`/`prev`/`next` without an `Optional` is being decided on
  the PR.
  `contiguous` is derived as today (consecutive `produced_at` ticks), so a gap still
  splits a tiling segment even when the two nodes are linked.
- Out of scope: solver-chosen op order. Execution order is an input to this plan
  (`add_successor`). Leaving it out, and letting the solve place ops, is what op
  reordering would need; that is separate work and changes none of the stages here.
- `ExecutionEdge`s link *all* op outputs in the solve, not just the tileable ones.
  `joinable` reproduces today's rule that an untileable op, or a gap in ticks, splits a
  tiling segment. `nest_pairs` is today's "same `TileSpec.level_counts`, not untiled"
  test, precomputed per edge instead of re-derived through `loop_id`.
- Coarse tiling then splits across the two edge types the way the data does: group
  membership (which adjacent ops share a nest) is decided on `ExecutionEdge`s, and
  whether a tiled value must be cut to HBM is decided by checking its READ edges
  against those groups.

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

- In `scratchpad/buffer_graph.py`: `GraphElement`, `DependencyKind`, `DivisionMap`,
  `DependencyEdge` (`slices_alike`, `bridges`), `ExecutionEdge` (`nest_pairs`,
  `joinable`, `joins`),
  `Graph` (context, `add_node`, `edges_of`, `tiling_segments`), and the node gains
  `tilings`/`tiling`/`produced_at`/`inputs`/`outputs`/`prev`/`next`, `add_child`
  taking `pairs` only (no `geometry` yet; see Stage 2b), and `add_successor`.
- `spill_cost` and the node's derived properties (`sym_is_lx`, `sym_division`,
  `per_core_sizes`, `cores_used`, `shape_costs`, `tileable`) stay as drafted.
- Drop `SliceEdge`/`InPlaceEdge`/`CopyEdge`/`Timeline` from the draft.
- Tests (`tests/inductor/test_buffer_graph.py`, no device): `add_node` keeps insertion
  order; `add_child` puts one edge object on both ends (and nowhere else), and two
  edges of different kinds can join the same pair; `DivisionMap` answers
  `allows`/`dst_options`/`src_options` consistently with `pairs`; `slices_alike` is
  true on an ungated edge; `bridges`
  returns only the copies priced for that pair; `add_successor` links both ends, derives
  `nest_pairs` and `contiguous`, and refuses a second successor or predecessor;
  links and `contiguous`; `joins` follows `nest_pairs`; `tiling_segments` splits at
  gaps and at untileable ops; node fixing rules (single candidate fixes
  `division=0`, out-of-range division asserts); `tiling` is `None` while `division`
  is free and the candidate's spec once it is set.

### Stage 2a. Builder

- `build_graph(buffers, alignment, capacity, exclusions, shuffles_priced)` makes the
  `Graph`, one `add_node` per buffer in the allocator's order, and the edges through
  `add_child`. It is the only code that looks at the allocator classes. It encodes what
  is already known:
  - `resident=False` for excluded buffers. Copy pinning waits for `_finalize`.
  - READ edges only between nodes that both choose a division (this matches today's
    no-op `constrain_residency` on placement-only wrappers). `division_map` comes from
    the consumer's `cd_parent_matches[src]`, `relayouts` from its
    `cd_parent_relayouts[src]` filtered to groups with a copy in this solve.
  - IN_PLACE edges with `division_map=None` when the child's division was fixed
    upstream, otherwise the same `cd_parent_matches` entry.
  - RELAYOUT_COPY edges for live copies.
  - `produced_at` from `uses[0]` for op outputs, then `add_successor` along the op
    outputs in `produced_at` order. Each edge's `nest_pairs` comes from the two nodes'
    candidate `TileSpec.level_counts` (equal and non-empty).
  - `add_child` call order matches today's constraint order (READ by producer order,
    then RELAYOUT_COPY, then IN_PLACE), so `Graph._finalize` stays deterministic.
- Tests: for every synthetic graph, the edges must match what `_get_children`,
  `_relayout_copies` and `in_place_parents` give today, and `nest_pairs` must agree
  with `loop_id` equality from `_cut_literals`. These cross-checks are deleted in
  Stages 3 and 4a along with those helpers.

### Stage 2b. Port the allocator's edge derivation into `add_child`

- `add_child(..., geometry=...)` derives the `DivisionMap` from the `ResidencyEdge`
  over the two nodes' `core_divisions` (the body of `_cd_parent_matches`), and
  the consumer's cached capping derives the priced relayouts (the per-edge half of
  `_cd_parent_relayouts` plus `_cap_relayout_groups`); `ComputeNode._finalize` does
  the read-count and copy steps. The in-place pairing the allocator does at
  `allocator.py:1552` moves the same way.
- `_finalize(model)` grows over the stages: here it only derives; Stage 5 adds the
  node vars, Stage 6 the edge constraints, and Stage 4a the execution-edge literals.
  Until Stage 8 the solver calls `Graph._finalize(model)` itself.
- The builder passes `geometry` wherever the consumer's `residency_edges` has the
  parent, and `pairs` from `cd_parent_matches` otherwise.
- The allocator keeps computing its tables in Phase 1 (they are still the external
  contract), so the test is that both agree: on the device-backed graphs
  (test_work_division, test_solver_relayout_e2e) the derived map and relayouts must
  equal `cd_parent_matches` / `cd_parent_relayouts` edge for edge.
- Equivalence diff must be empty.

### Stage 3. Solver reads relationships from the graph

- `_plan_layout_generic` builds the `Graph` next to the existing wrappers.
- `_get_children` → `node.outputs` of kind READ; `_relayout_copies` →
  `graph.edges_of(RELAYOUT_COPY)`; the `in_place_parents` loop in
  `_add_inplace_relaxation` → `graph.edges_of(IN_PLACE)`.
- Delete `_get_children` and `_relayout_copies`. The wrappers still own the CP vars.
- Equivalence diff must be empty.

### Stage 4a. Tiling groups on execution edges

- One per-edge function owns an execution edge's coarse-tiling encoding (it becomes
  `ExecutionEdge._finalize` in Stage 6, once the wrappers are gone): a
  `same_nest` and a `joined` literal (today's `samenest_*` / `joined_*`), and the group
  id step across it (`group[after] = group[before] + 1 - joined`). A non-joinable edge
  starts a new segment with group 0.
- `same_nest` keeps today's `loop_id` equality encoding in this stage, so the CP model
  is unchanged.
- `_tiling_group_ids` becomes a walk over `graph.tiling_segments()` calling that
  function per edge. Its `uses[0]` position logic and the `RelayoutCopyBuffer` check go
  away.
- `_cut_literals`: for each tiled node, the consumers on its READ edges in the same
  segment get an `apart` literal over group ids, gated on the edge's `division_map`; a
  consumer outside the segment, or a graph output, makes the node unshareable. Same
  rule as today, over edges.
- `_tile_count_terms` iterates nodes.
- Equivalence diff must be empty (run the coarse-tiling suites with
  `auto_coarse_tiling` on).

### Stage 4b. `joined` from `nest_pairs`

- `same_nest` becomes `_gate_divisions` over `edge.nest_pairs`, so the edge is the only
  source of "these two ops can share a nest". `loop_id` and the nest interning in
  `_cut_literals` go away; `tiled` comes from each node's untiled candidates directly.
- Same constraints, different encoding, so CP-SAT may break ties differently. The
  equivalence check for this stage: every ladder stage's objective is unchanged;
  decisions may differ only where the objectives tie. Report solve times for the
  coarse-tiling suites against Stage 4a; if they regress, drop this stage and keep
  `nest_pairs` as a query for Phase 2.

### Stage 5. Nodes own their vars

- `ComputeNode._finalize(model)` creates the node's decision vars, replacing the
  variable-making half of both wrappers. A free field gets a CP var; a fixed one gets
  a constant:
  - `division` → `new_constant(k)` with constant `eff_size`/`cores`/`core_cost`/
    `splits`. This generalises today's single-candidate fast path to any fixed `k`.
  - `resident`/`offset` → a var pinned with `model.add(var == value)`, because interval
    presence needs a literal.
- The node exposes `buffer`, `division` and `division_is` for `cpsat_expr_printer`, so
  that contract is unchanged. Var names stay as today (`div_<name>`, ...).
- For now the wrappers hold their node and delegate to it, so this stage changes no
  constraint code.
- Equivalence diff must be empty.

### Stage 6. Edges post their own constraints; delete the wrappers

- `DependencyEdge._finalize(model)` posts its constraints, by kind:
  - READ: slicing match over `division_map`, or a resident copy from `relayouts`; the
    served literals are kept on the edge.
  - RELAYOUT_COPY: today's `_constrain_relayout_copies` over `priced`.
  - IN_PLACE: a merge literal, with a division gate only when `division_map is not
    None` (today's `_add_inplace_relaxation`).
- `Graph._finalize(model)` calls them in that node-driven order; the three solver
  methods go away.
- `_add_no_overlap_2d` keeps its logic, now over the nodes.
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
- This is the one Phase 1 stage meant to change what is observable (4b may only break
  ties differently): ladder objectives shift by constants and fewer stages run. The
  equivalence diff for this stage compares decisions only; objectives may differ by a
  constant per stage.

### Stage 8. `graph.solve()`; honour pre-fixed decisions

- `graph.solve(model, cost_expr)` becomes the public entry point, taking the same model
  and cost expression `_run` takes today. It runs `_finalize(model)` (moved out of the
  solver), the objective ladder (`_run`) and extraction, and returns
  the op-output nodes in execution order (following the `ExecutionEdge` chain) with
  every field that was `None` filled in: `division` (so the tiling), `resident` and
  `offset`. The chosen
  relayout per READ edge is recorded on the edge.
- The returned list is the execution plan: each node's chosen candidate (core split
  and tiling), where its output lives, how each input is read (in place, through which
  relayout copy, or from HBM), and the tiling groups along the joined execution edges.
  Inputs and relayout copies are reached through the READ edges of the nodes that use
  them.
- Bottom-justify is skipped when any offset was pinned.
- `_plan_layout_generic` makes the model, calls `graph.solve(model, cost_expr)` and
  writes the nodes back to the
  caller's buffers: `address` in bytes,
  `spill_reason`, `chosen_division` only where the caller enumerated divisions, and
  `chosen_relayouts` scaled.
- Tests: a solve with a pre-fixed `division`/`resident`/`offset` honours it; every free
  field comes back filled; the returned order matches `produced_at`; and the tiling
  groups read off the returned nodes equal `derive_tiling_groups` on the same choices.
- Run the device suites.

### Stage 9. Cleanup

- Rewrite the module docstring around nodes, dependency edges, execution edges and
  `graph.solve()`; keep docstrings short.
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
- Equivalence diff from Stage 0: empty, except Stages 4b and 7 as noted.
- After Stages 2b, 4a, 4b, 6 and 8: the device suites that compile through CP-SAT —
  test_scratchpad_use, test_work_division, test_solver_relayout_e2e,
  test_solver_auto_coarse_tiling — base vs change, same serial setup.
- `pre-commit run --files <changed>` (ruff, mypy).

## Phase 2 — outline (separate plan after Phase 1 lands)

Unify the problem definition first, then make `ComputeNode` and `Graph` the
allocator's output and retire `LifetimeBoundBuffer`/`CoreDivisionBuffer`/
`RelayoutCopyBuffer`. Each stage below is also its own PR, green on the full suite.
Sized from the inventory:

- **2a. One problem definition, pluggable solver.** `graph.set_solver(CpSatSolver())` /
  `graph.set_solver(SimulatedAnnealing())`. The `model` passed to `solve()` (and on to
  `_finalize`) becomes the chosen solver's backend; the sympy `cost_expr` stays the
  solver-neutral objective. Nodes and edges state the problem (candidates, division maps,
  relayouts, nest pairs, capacity), and each backend turns that into its own form:
  CP vars and constraints for CP-SAT, a move space and cost for SA. The derivations in
  `_finalize` stay solver-neutral, and what it posts goes through the backend. `solve()`
  returns the same
  execution-ordered nodes whichever solver ran. Nothing outside the solvers changes
  yet.
- **2b. Node absorbs the allocator-only fields.** The node (moved into
  `plan_solver.py`) gains what the allocator's commit path reads: `in_place_parents`
  (→ IN_PLACE edges), `residency_reason` (→ `Graph.exclusions`), lifetime overrides,
  `lx_view`, `spill_reason`, `chosen_relayouts`, `lx_relayout_plans`, `paired_with`
  (greedy only). From the SA co-optimizer it gains `division_space`; `residency_edges`
  and `tile_reads` move onto READ edges (the geometry behind `division_map`, and
  whether the consumer reads the producer tile by tile); `op_position` is replaced by
  `produced_at`/execution edges. Sizes
  stay in bytes on the node and CP-SAT converts to units internally. `address` stays
  in bytes (every solver writes bytes today).
- **2c. The allocator builds nodes and the graph** at its four build sites in
  `allocator.py`: `_build_bound_buffers` (:1456/:1497),
  `_append_lx_relayout_destinations` (:1819), `_build_cd_bound_buffers`
  (:4074/:4275), and `_relayout_copy_buffers` (:4418, via `build_relayout_copy`). Its
  4 isinstance checks (:713, :2769, :2964, :3099) become
  `graph.edges_of(RELAYOUT_COPY)` lookups. Its `_cd_parent_matches` /
  `_cd_parent_relayouts` tables are deleted: it calls `add_child(geometry=...)`, which
  has owned that logic since Stage 2b. `build_graph` then reads nodes, not classes.
- **2d. Placement-only solvers** (greedy, first-fit/best-fit, SA, cooling_schedules,
  utils.plot_buffers) only read fields (~90 lines). Swap the type and drop greedy's
  isinstance at :183.
- **2e. Exhaustive search and the SA co-optimizer** read `parents`/`cd_parent_matches`
  from READ edges and write `division`/`resident` on nodes. SA's duplicated
  `_spill_cost` collapses onto `buffer_graph.spill_cost`, and its op adjacency moves
  onto execution edges.
- **2f. Tests**: ~180 constructor calls across 13 files, mostly through helpers
  (`make_buffer_registry`, `synthetic_cooptimization_graphs._buf`,
  `cooptimization_capture_loader._buf`, `test_perm_layout_solver._buf`). Switch the
  helpers first, then the direct calls in `test_scratchpad_solver.py` (131) and
  `test_solver_relayout_decision.py`.
