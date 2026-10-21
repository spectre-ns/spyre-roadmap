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

The solve is also **enumerating**: each buffer carries a menu (`core_divisions`), one
int var indexes it, and every relationship is a table over menu indices
(`cd_parent_matches`, per-pair `RelayoutCandidate`s, `loop_id` per candidate). The end
state is a **non-enumerating** solve: per-core views, relayouts, in-place reuse and
coarse tiling expressed as constraints over per-axis split and per-level tile-count
variables, with no candidate menus or pair tables. The graph built here has to carry
that end state without another redesign.

Goal: model the solve as a graph.

- **`ComputeNode`** — an op output (inputs and copies: see *Open questions*). It holds
  a `SplitSpace` (what it may choose: per-axis core splits and per-level tile counts)
  and its decisions (`assignment`, `resident`, `offset`), each fixed ahead of time or
  `None` for the solver to find.
- **`DependencyEdge`** — every data dependency between two nodes (reads, in-place
  reuse, relayout copies). It holds *relations* over the two ends' decisions: the
  per-core view match (`ViewRelation`) and, on a read, the relayouts that can bridge a
  mismatch (`RelayoutRelation`). Edges are shared: the same edge object sits in its
  source's `outputs` and its destination's `inputs`, so either end can walk it.
- **`ExecutionEdge`** — links each op output to its chronological neighbours
  (`node.prev` / `node.next`). Coarse tiling is handled here: whether the two ops run
  the same loop nest (`NestRelation`), whether they join one tiling group, and where a
  group breaks.
- **`Graph`** — a collection of nodes (which own the edges), plus the global context
  (alignment, LX capacity, exclusions). `graph.solve(model, cost_expr)` fills in every
  free decision and returns the op-output nodes in execution order, which is the
  execution plan.

Three phases, each stage its own PR that leaves the full suite green:

- **Phase 1 — re-representation (CP-SAT internals).** The graph carries the
  allocator's final data and every relation uses its *table* encoding, which is
  today's model. Each stage is proven by an identical CP model (see *Verification*).
  The allocator's three classes stay the external contract.
- **Phase 2 — constraint encodings (CP-SAT internals).** Each relation gains a
  constraint encoding over per-axis split / per-level tile vars, behind a flag, checked
  against the table encoding. Then menus stop being materialized.
- **Phase 3 — one problem definition.** `graph.set_solver(...)`, then the allocator
  builds the graph and the other solvers move onto it.

Work happens in worktree `.claude/worktrees/cpsat-graph-model` (branch
`cpsat-graph-model`, off `main` 82edd8f6). No commits (user commits).
`scratchpad/buffer_graph.py` is already drafted there (`CoreDivisionBoundBuffer`,
`BufferGraph`, `build_buffer_graph`, `SliceEdge`/`InPlaceEdge`/`CopyEdge`,
`Timeline.runs`). Stages 1–2 rename and reshape it into the model below.

## Design rules

1. **Relations, not tables.** Every relationship is an object over the two ends'
   decisions with two encodings: a table over menu indices (Phase 1, today's model)
   and constraints over split / tile vars (Phase 2). Nothing outside a relation reads
   menu indices; queries take `Assignment`s.
2. **Phase 1 derives nothing the allocator decided.** Match tables (including the
   loop-carry intersection and input-clone overrides, `allocator.py:4160-4240`),
   capped relayouts and the copies built from them, read counts (`len(uses)`, which
   include extern and placement-only readers), lifetimes and their overrides, and
   exclusions all come in as data. Moving any derivation out of the allocator is
   Phase 2/3 work with its own agreement test.
3. **`None` means an open decision.** `assignment`, `resident`, `offset` only.
   Structural fields are never `Optional`.
4. **Build structure, then finalize once.** `add_node` / `add_child` / `add_successor`
   wire the graph and touch no model. `solve(model, cost_expr)` calls
   `Graph._finalize(model)`, where every element posts its vars and constraints in a
   fixed order.

## The model

```python
class GraphElement(ABC):
    """Base of nodes and both edge types: built piecewise, finalized once."""
    def _finalize(self, model) -> None       # post this element's vars / constraints

class DependencyKind(Enum):
    READ = auto()          # dst's op reads src
    IN_PLACE = auto()      # dst may take over src's storage at the shared tick
    RELAYOUT_COPY = auto() # dst is a relayout destination of src

@dataclass(eq=False)
class DependencyEdge(GraphElement):
    src: ComputeNode
    dst: ComputeNode
    kind: DependencyKind
    view: Optional[ViewRelation]             # None only: IN_PLACE whose child was
                                             # fixed upstream (ungated)
    relayout: Optional[RelayoutRelation]     # READ only
    priced: tuple[int, ...] = ()             # RELAYOUT_COPY only
    chosen: Optional[RelayoutRead] = None    # filled by solve()

@dataclass(eq=False)
class ExecutionEdge(GraphElement):
    before: ComputeNode
    after: ComputeNode
    contiguous: bool                         # consecutive ticks
    nest: NestRelation

    @property
    def joinable(self) -> bool:              # can the two ops share a nest at all?
        return self.contiguous and self.before.tileable and self.after.tileable

@dataclass(eq=False)
class ComputeNode(GraphElement):
    ...  # name, size (alignment units), start/end_time and lifetime overrides,
         # read_count, spill_cost, boundary, sym_core_divs, sym_tile_counts
    space: SplitSpace                        # what the node may choose
    assignment: Optional[Assignment]         # None = free
    resident: Optional[bool]                 # None = free
    offset: Optional[int]                    # None = free
    produced_at: int                         # tick of the producing write
    inputs: list[DependencyEdge]
    outputs: list[DependencyEdge]
    prev: ExecutionEdge                      # always linked:
    next: ExecutionEdge                      # graph.head -> ... -> graph.tail

    @property
    def tiling(self) -> Optional[TileSpec]   # assignment.tiling once set
    @property
    def tileable(self) -> bool               # some allowed tiling is not untiled

    def add_child(
        self,
        child: ComputeNode,
        kind: DependencyKind = DependencyKind.READ,
        *,
        view: Optional[ViewRelation],
        relayout: Optional[RelayoutRelation] = None,
    ) -> DependencyEdge                      # the only way to add an edge
    def add_successor(self, after: ComputeNode) -> ExecutionEdge
                                             # the only way to add an execution edge

class Graph:
    alignment: int
    capacity: int                            # LX capacity in alignment units
    exclusions: dict[str, str]               # name -> residency_reason
    nodes: dict[str, ComputeNode]            # add_node order = today's var order
    head: Sentinel                           # chain ends; not in nodes, no vars,
    tail: Sentinel                           # never returned by solve()

    def add_node(self, buffer, space: SplitSpace) -> ComputeNode
    def edges_of(self, kind) -> Iterator[DependencyEdge]  # walks nodes; no edge list
    def solve(
        self,
        model: cp_model.CpModel,
        cost_expr: Optional[sympy.Expr] = None,  # None or unlowerable: fallback
    ) -> list[ComputeNode]                   # op outputs in execution order
    def _finalize(self, model) -> None       # every element, in today's order
    def tiling_segments(self) -> list[list[ComputeNode]]  # maximal joinable chains
```

`SplitSpace`, `Assignment` and the three relations (`ViewRelation`,
`RelayoutRelation`, `NestRelation`) are what the graph holds; each stage below
introduces only the part of their interface it needs (Stage 1: data and queries;
Stage 5: posting node vars; Stage 6: posting edge constraints; Phase 2: constraint
encodings and generated spaces).

Notes:

- `eq=False` on nodes and edges: they reference each other, so they compare by
  identity and get a short custom `__repr__` (name, or `src->dst:kind`).
- A pair of nodes can share more than one edge (an in-place child usually reads its
  parent too). The solver iterates edges by kind, so there is no `isinstance` dispatch.
- `add_child` puts the one edge object on both ends and nowhere else; the graph keeps
  no edge list. `add_successor` links one chain from `graph.head` to `graph.tail` and
  refuses to relink a node that already has a neighbour; `_finalize` asserts the chain
  is complete. `contiguous` is derived as today (consecutive `produced_at` ticks).
- A missing match entry is an **empty** `DivisionMap`, never `None`: today
  `.get(parent, [])` forces the residency literal false, forbids the merge and blocks
  `apart` in cuts (`ilp_solver_ortools.py:392/410/470`), and a missing key is common
  (any producer with a `residency_reason`).
- **`_finalize(model)` order is today's order**, element by element:
  1. **Nodes**, in `add_node` order: `space.post` (division, `eff_size`, `cores`,
     `core_cost`, split and tile-count vars), `in_buffer`, `offset`, and the `merge_*`
     literal for each IN_PLACE input (today these are made per wrapper, `:226`).
  2. **IN_PLACE edges**, by destination (`dst.inputs`): merge constraints, then
     at-most-one over each node's incoming and outgoing merges.
  3. **Capacity** per node (`offset + eff_size <= M`) and the **2-D no-overlap**
     (with the incoming-merge tick shortening).
  4. **Exclusion pins** (`resident=False` from `Graph.exclusions`).
  5. **READ edges**: parent order, then child order, then the child's `parents`
     order; served literals in sorted relayout-group order (`:440`).
  6. **RELAYOUT_COPY edges** in copy-node order, including today's copy pins (source
     not choosing a division, no consumer can use it).
  7. **Tiling**: `loop_id`/`tiled` per divided node, then execution edges along the
     chain (`same_nest`, `joined`, group step), then cuts, then tile counts.

  The fallback copy pin (`:1044-1048`) is not here: it depends on whether `cost_expr`
  lowered, so it stays in the ladder inside `solve`. Stage 0's proto diff is the check
  on every claim in this list.
- `solve(model, cost_expr)` is the only public entry point after building: it runs
  `_finalize(model)`, then the objective ladder over the sympy `cost_expr` (lowered as
  `_minimize_cost_expr` does today, with today's fallback objective when it is `None`
  or fails to lower), then extraction, and fills in every field that was `None`.
- Coarse tiling splits across the two edge types the way the data does: group
  membership is decided on `ExecutionEdge`s, and whether a tiled value must be cut to
  HBM by checking its READ edges against those groups.
- Out of scope: solver-chosen op order. Execution order is an input (`add_successor`).

## Open questions

- **Graph inputs and relayout copies.** Neither has its own slot in the op order: an
  input's clone-in happens at its first use, and a copy's shuffle runs directly before
  its first consumer only if the solve keeps it. Recommendation: a small node type of
  their own (space, decisions, `inputs`/`outputs`, no `produced_at`/`prev`/`next`),
  joined through `add_child`; `ComputeNode` stays op outputs only. Copies do not break
  tiling runs today, and this keeps that exactly.
- **Tiling vs division coupling** is settled by the end state: in Phase 1 a menu index
  picks both; in Phase 2 splits and tile counts are separate vars constrained jointly
  (split domains depend on the tiling through `predict_frame`).

## Phase 1 — re-representation

### Stage 0. Equivalence harness and baseline

- Add `tests/inductor/cpsat_equivalence.py`. It solves a buffer set
  deterministically: 1 worker, `random_seed=0`, and no wall-clock limit
  (`time_limit_seconds=0`, or `max_deterministic_time`), because the default 60 s
  limit (`config.cpsat_time_limit_seconds`) can return a time-dependent incumbent. It
  fails on any ladder stage that does not end OPTIMAL.
- It wraps `_solve_and_record` to capture every stage's objective (`last_solve_stats`
  keeps only the last), and dumps per stage: the **serialized model proto** (vars and
  constraints in order), residency, chosen division, offset and chosen relayouts.
- Corpus: `synthetic_cooptimization_graphs`, the `cooptimization_capture_loader`
  graphs, and, because those hold only untiled `CoreDivisionBuffer`s, cases from
  `test_solver_relayout_decision` (copies, relayouts), the coarse-tiling tests,
  `plan_layout` (placement-only buffers) and solves with a `cost_expr`.
- Record the baseline (below) on a frozen `main` copy. No production code changes.

### Stage 1. Data model

- Rename the draft (`CoreDivisionBoundBuffer` → `ComputeNode`, `BufferGraph` →
  `Graph`, `build_buffer_graph` → `build_graph`) and reshape it into the graph above.
  Drop `SliceEdge`/`InPlaceEdge`/`CopyEdge`/`Timeline`.
- The supporting types, data and queries only (no solver methods yet):

  ```python
  @dataclass(frozen=True)
  class Assignment:                    # a decided split
      splits: Mapping[SplitKey, int]   # sym_core_divs keys
      tiling: TileSpec

  class SplitSpace(ABC):
      def holds(self, a: Assignment) -> bool
  class FixedSpace(SplitSpace): ...    # placement-only: nothing to choose
  class MenuSpace(SplitSpace):         # the allocator's menu
      core_divisions: list[CoreDivision]

  class ViewRelation:                  # both ends own the buffer with the same view
      table: DivisionMap               # allocator's final pairs; empty = never alike
      def holds(self, src: Assignment, dst: Assignment) -> bool
  class RelayoutRelation:              # READ: copies that can serve a mismatch
      reads: tuple[RelayoutRead, ...]  # allocator's capped candidates
      def bridges(self, src: Assignment, dst: Assignment) -> tuple[RelayoutRead, ...]
  class NestRelation:                  # both ops run the same non-empty loop nest
      table: tuple[tuple[int, int], ...]
      def joins(self, before: Assignment, after: Assignment) -> bool
  ```
- `spill_cost` and the node's derived properties (`sym_is_lx`, `sym_division`,
  `per_core_sizes`, `cores_used`, `shape_costs`, `tileable`) stay as drafted.
- Tests (`tests/inductor/test_buffer_graph.py`, no device): `add_node` order;
  `add_child` puts one edge object on both ends only, and two kinds can join one pair;
  relation queries (`holds`, `bridges`, `joins`) agree with their tables, and an empty
  table never holds; `add_successor` links both ends, derives `contiguous`, and refuses
  a second neighbour; `tiling_segments` splits at gaps and at untileable ops; sentinels
  are not in `nodes`; `tiling` is `None` while `assignment` is.

### Stage 2. Builder

- `build_graph(buffers, alignment, capacity, exclusions)` makes the `Graph`, one
  `add_node` per buffer in the allocator's order (`FixedSpace` for placement-only
  buffers, `MenuSpace` otherwise), and the edges through `add_child`. It is the only
  code that reads the allocator classes, and it copies their final data:
  - READ edges only between nodes that both choose a division (today's no-op
    `constrain_residency` on placement-only wrappers). `view.table` is
    `cd_parent_matches.get(src, ())`; `relayout` is `cd_parent_relayouts[src]`
    filtered to groups with a copy in this solve.
  - IN_PLACE edges with `view=None` when the child's division was fixed upstream,
    otherwise the same table.
  - RELAYOUT_COPY edges for the allocator's copies, in its order.
  - `read_count`, lifetimes and overrides, `spill_cost` inputs and `Graph.exclusions`
    as given.
  - `produced_at` from `uses[0]` for op outputs (today's filter: has `uses`, not
    `first_use_is_read`, not a copy), then `add_successor` in `produced_at` order;
    each `NestRelation` table comes from the two nodes' candidate
    `TileSpec.level_counts` (equal and non-empty).
- Tests: on every corpus graph the edges match `_get_children`, `_relayout_copies` and
  `in_place_parents`, and each nest table equals `loop_id` equality ∧ `tiled[after]`.
  The `_get_children`/`_relayout_copies` checks go in Stage 3; the nest check in
  Phase 2b.

### Stage 3. Solver reads relationships from the graph

- `_plan_layout_generic` builds the `Graph` next to the existing wrappers.
- `_get_children` → `node.outputs` of kind READ; `_relayout_copies` →
  `graph.edges_of(RELAYOUT_COPY)`; the `in_place_parents` loop in
  `_add_inplace_relaxation` → each node's IN_PLACE `inputs`.
- Delete `_get_children` and `_relayout_copies`. The wrappers still own the CP vars.
- Proto diff empty.

### Stage 4. Tiling groups on execution edges

- One per-edge function owns an execution edge's coarse-tiling encoding: `same_nest`
  and `joined` (today's `samenest_*` / `joined_*`) and the group step
  (`group[after] = group[before] + 1 - joined`). A non-joinable edge starts a new
  segment with group 0. `same_nest` keeps today's `loop_id` encoding, `tile_group_*`
  keeps its domain, and `loop_id`/`tiled` are still made first, in node order.
- `_tiling_group_ids` becomes a walk over `graph.tiling_segments()`; its `uses[0]`
  logic and `RelayoutCopyBuffer` check go away.
- `_cut_literals`: for each tiled node, consumers on its READ edges in the same segment
  get an `apart` literal gated on `edge.view`; a consumer outside the segment, or a
  graph output, makes the node unshareable. `_tile_count_terms` iterates nodes.
- Proto diff empty (coarse-tiling corpus with `auto_coarse_tiling` on).

### Stage 5. Nodes post their own vars

- Adds `SplitSpace.post(model, node) -> NodeVars`, the node's vars as the relations
  will see them (`division`, `eff_size`, `cores`, `core_cost`, split and tile-count
  vars, `in_buffer`, `offset`, merge literals). `ComputeNode._finalize(model)` calls
  it, replacing the variable-making half of both wrappers: `MenuSpace` posts today's
  index var, element ties and single-candidate constants; `FixedSpace` posts today's
  placement-only vars (`cores`, `core_cost` `None`, no gates). Merge literals are made
  here, in node order.
- The node also exposes `buffer`, `division` and `division_is` for
  `cpsat_expr_printer`, so that contract is unchanged until Phase 2a. Var names stay
  as today.
- The wrappers hold their node and delegate to it, so no constraint code changes.
- Proto diff empty.

### Stage 6. Edges post their own constraints; delete the wrappers

- Adds the relations' table encodings: `ViewRelation.post(model, src, dst, enforce)`
  (today's `_gate_divisions`), `RelayoutRelation.post(...)` returning the served
  literals, and `NestRelation.post(...)` for the Stage 4 `same_nest` literal.
  `DependencyEdge._finalize(model)` posts, by kind, through them:
  - IN_PLACE: merge literal; `eff_size` equality when the child chooses a division;
    offset equality; both-resident implications; view gate unless `view is None`.
    Node-level parts (at-most-one, tick shortening) stay with the node and no-overlap.
  - READ: `view.post` (match), or a resident copy via `relayout.post`; served literals
    kept on the edge.
  - RELAYOUT_COPY: today's `_constrain_relayout_copies` over `priced`.
- `ExecutionEdge._finalize(model)` takes over the Stage 4 function; capacity,
  no-overlap, exclusion pins, cuts and tile counts move into `Graph._finalize`, all in
  the order listed under *The model*.
- `free_copies` (presolve threshold) keeps today's rule: every copy whose parent
  chooses a division.
- Delete `_LifetimeBufferWithCpVars`, `_CoreDivisionBufferWithCpVars`, `_wrap`,
  `_add_core_division`, `_add_inplace_relaxation`, `_constrain_relayout_copies`, every
  `isinstance(..., RelayoutCopyBuffer/CoreDivisionBuffer)` and every
  `getattr(..., "core_divisions")` inside the solve.
- Proto diff empty. Run the device suites.

### Stage 7. Objective from free decisions only

- Parallelism, shape and tile-count terms come only from nodes whose `assignment` is
  `None`. A stage with no free terms is skipped, removing the no-op re-solves a
  placement-only solve runs today.
- Check: every ladder stage's objective equals the baseline up to a constant per
  stage. Decisions may differ only on ties, because `_extract` reads whichever solve
  ran last and skipping stages changes that.

### Stage 8. `graph.solve(model, cost_expr)`; honour pre-fixed decisions

- `graph.solve` becomes the public entry point with the model and cost expression
  `_run` takes today: `_finalize(model)`, the objective ladder (with the fallback copy
  pin when `cost_expr` is `None` or fails to lower), extraction. It returns the
  op-output nodes along the execution chain with every `None` field filled
  (`assignment`, so the tiling; `resident`; `offset`) and `chosen` set on READ edges.
  That list is the execution plan.
- Pre-fixed decisions: a fixed `assignment` on a `MenuSpace` posts a constant index;
  fixed `resident`/`offset` pin their vars. Bottom-justify is skipped when any offset
  was pinned.
- `_plan_layout_generic` makes the model, calls `graph.solve(model, cost_expr)` and
  writes back: `address` in bytes, `spill_reason`, `chosen_division` only where the
  caller enumerated divisions, `chosen_relayouts` scaled.
- Tests: pre-fixed decisions are honoured; every free field comes back filled; the
  returned order matches `produced_at`.
- Device-suite check: CP-SAT's joined runs *refine* `derive_tiling_groups` on the same
  choices, mismatches logged. Equality is not guaranteed today: CP-SAT approximates
  tile-by-tile reads through the view tables and never consults `tile_misread` (a
  pre-existing gap; Phase 2b closes it).

### Stage 9. Cleanup

- Module docstring around nodes, relations, execution edges and `graph.solve`.
- `tests/inductor/test_cooptimization_types.py:361-372` → compare against
  `buffer_graph.spill_cost(b)`.
- Update comments citing the old classes: `plan_solver.py:152`,
  `sa_cooptimizer.py:1295`, `allocator.py:2696`, `cpsat_expr_printer.py:809`.
- Register `test_buffer_graph.py` in a `tests/configs/` shard; `make check-all-configs`.

Kept unchanged through Phase 1: `_gate_divisions`, `_PlacementUnit`, `_justify`,
`_solve_and_record`, the capability flags, `__init__`, and the public `plan_layout` /
`plan_layout_and_core_divisions`.

## Phase 2 — constraint encodings (outline; own plan after Phase 1)

Each stage adds a constraint encoding behind a flag and an **agreement test** against
the table encoding on the full corpus and the device graphs: same feasible
(src, dst) assignments for a relation, or same optimum for a solve. What is not linear
is called out; it needs `AddMultiplicationEquality` / `AddDivisionEquality` or table
constraints over small per-buffer domains, never per-pair tables.

- **2.0. Probe: is lazy faster than eager? (go / no-go for the rest of Phase 2.)**
  Phase 1 does not depend on it, so it can run alongside. Device-free, built on the
  solver-input capture and replay harness in `~/cpsat_bench/pr5132`
  (`zoo_cap.py`, `granite_cap.py`, `replay.py`):
  1. **Where eager time goes today.** On the captured zoo models and Granite blocks,
     time the enumeration end to end: menu generation per op, `_cd_parent_matches`
     (|D_p| × |D_c| view comparisons per edge), `_cd_parent_relayouts` with
     `predict_ops` per consumer candidate, copy construction, then CP-SAT model
     build, sympy lowering, presolve and solve. This bounds what lazy can save.
  2. **Prototype the lazy encoding** for views (2c) and nests (2b) on per-axis split /
     per-level tile vars, outside the solver, over the same captured graphs. The
     capture has to be extended to record each read's index map
     (`dep_device_coordinates`, extents, stick size), since today it pickles only
     what the solver was handed. Relayouts stay eager in the probe (the hardest
     encoding; a separate probe if 2.0 passes).
  3. **Compare** eager (tables + build + solve) against lazy (build + solve) per graph:
     model size, presolve time, time to OPTIMAL at 1 worker
     (`use_deterministic_algorithms`) and at 16 workers, and end-to-end compile time.
     Both arms run in one process, interleaved, with a plan hash recorded (paired A/B:
     cross-process variance is up to ~25%). Both must reach the same objective.
  - **Pass:** lazy is faster end to end on the median and no slower on the slowest
    graphs. The known risk is that CP-SAT propagates multiplication / division
    constraints worse than small tables, so a lazy model can solve slower even when
    it builds faster.
  - **Fail:** keep menus and replace only the parts the probe shows are hot (for
    example per-pair tables → per-buffer view tables), and revise the end state here.

- **2a. Cost expression over per-axis vars.** Today `cpsat_expr_printer` tabulates
  over the menu: `_division_table` (`:309`), `RelayoutCharge` keyed on `sym_division`
  (`:822`), `log2_` by element on the division, `AppliedUndef` menu tables (`:854`),
  and `extract_op_features` reads `candidate_work_slices`. Tabulate over per-axis
  tuples (`AddAllowedAssignments`, small at ≤ 32 cores) instead. Prerequisite for the
  rest; agreement: identical lowered values at every menu point.
- **2b. `NestRelation` on per-level tile vars.** `NodeVars` gains `level(k)`: per
  node `L0`, `L1` (`max_dims=2`) tied to the per-dim counts; `same_nest` is the
  two-way reified `L0 == L0' ∧ L1 == L1' ∧ tiled` (a one-way `_gate_divisions`
  would let `joined` go false on an equal pair and loosen cuts). Tile-by-tile
  alignment (`tile_misread`) becomes a tile `ViewRelation`, closing the Stage 8 gap.
- **2c. `ViewRelation` constraints: views built from the ops.** A per-core view is
  determined by the op's index map plus divisibility of the buffer's shape, so it is
  posted, not enumerated:
  - **Divisibility.** Each split var's domain is the divisors of its stick-adjusted
    extent (`_legal_split_factors`, `work_division.py:178`): the modulo constraint
    `extent % split == 0` as a domain, cheaper than `AddModuloEquality`.
  - **Placement.** Which device dim each split lands on is static per dep (stride
    lookup in `_per_core_view_from_prep`, `pass_utils.py:3506`), so the per-dim factor
    is a linear identity over the split vars; equal factors per device dim on both
    ends is linear.
  - **Core → slot.** `core_to_slice_mapping` (`core_mapping.py:541`) gives each split
    dim the coordinate `(core_id // stride) % split`, with `stride` the product of the
    splits before it in the op's iteration order. Equal strides are a chain of
    `AddMultiplicationEquality`; when both ops order the dims alike this reduces to
    equal factors. `contiguous_dim` moves a dim first only when its split > 1, so that
    order is a reified choice.
  - **Side conditions.** `cores` equal on both ends; "reduction split == 1" on a
    partial-reduction write; multi-stick rescue (factor `split·k`, divisibility over
    sticks); the interleave rule (an inner split > 1 with an outer symbol unsplit is
    unrepresentable) as an implication; the span cap as `AddDivisionEquality` or a
    per-dim table; a conjunction over every read of the buffer.
  - **Reshapes are just another read order.** The consumer's read already maps its
    loop symbols to the buffer's device coordinates as a mixed-radix expression
    (`dep_device_coordinates`, `pass_utils.py:3410-3414`, e.g.
    `4*outer + floor(inner/64)`), so a reshape is handled like a transposed read:
    - several symbols on one device axis, ordered outer → inner by coefficient: a
      split owns a contiguous slice only if every outer symbol on that axis is fully
      split, `split_inner > 1 ⇒ split_outer == extent_outer` (today's interleave
      rule, `:3700-3725`), and the axis factor is the product of their splits;
    - one symbol spanning several device axes (the collapsed-axis case
      `decompose_fused_split_view` handles): the split factorises in mixed radix,
      `f = f_outer · f_inner` with `f_inner > 1 ⇒ f_outer == d_outer`, each factor
      dividing its axis;
    - `floor(inner / elems_per_stick)` terms make divisibility count in sticks, which
      also covers the partial multi-stick case.

    So views need no table fallback: divisibility, implications and products cover
    them. If an agreement test finds a case these rules miss, that case gets a small
    allowed-tuples table per (node, dep), never a pair table.

  `ViewRelation` gains `geometry: tuple[ResidencyEdge, ...]` (each read's index map
  into the buffer) and a constraint `post` built from it. Loop carries are the
  conjunction of the read and update relations; input clones need
  a relation derived from the clone's consumers (design item). Agreement: the feasible
  pairs equal the allocator's final `cd_parent_matches`, carries and clones included.
- **2d. Relayouts as constraints.** One optional copy per (parent, consumer) READ edge
  whose view is the consumer's; sharing between consumers by reified view equality.
  Legality (`movement_supported`, projection and core-domain gates) and price (per
  split-dim tables × 1/cores, governing dim chosen piecewise) are tables over
  (src view, dst view). Group capping (`_cap_relayout_groups`) exists to bound
  enumeration and goes away, or becomes a cap on views per buffer. Agreement where
  capping did not bind; differences reported where it did.
- **2e. In-place as constraints.** Per-core footprint equality
  (`out_partition · out_tiles`, a product equality) plus the `ViewRelation`. The
  pre-solve gates (pointwise input, equal device layout, one-tick handoff) stay
  builder facts.
- **2f. `GeneratedSpace`.** A third `SplitSpace` wrapping `OpSplitSpace`. Per-axis
  domains from `_legal_split_factors`, product ≤ max cores (multiplication equality),
  at most one reduction split, blocked dims, the span cap as a per-dim table,
  tiling-dependent domains through the level vars. Menus
  are no longer materialized; `MenuSpace` remains for tests and for restricting a node
  to listed candidates. Agreement: same optimum wherever the menu was complete.

## Phase 3 — one problem definition (outline; own plan after Phase 2)

- **3a. `graph.set_solver(...)`.** The model passed to `solve` becomes the backend;
  `cost_expr` stays the solver-neutral objective. Lands with CP-SAT only. SA becomes a
  backend once it consumes relations' geometry (it already works lazily over
  `ResidencyEdge` through `_ViewRelation`, `sa_cooptimizer.py:614`, and generates
  divisions, `:378`), which Phase 2c provides.
- **3b. Node absorbs the allocator-only fields** (`in_place_parents` → IN_PLACE edges,
  `residency_reason` → `Graph.exclusions`, lifetime overrides, `lx_view`,
  `spill_reason`, `chosen_relayouts`, `lx_relayout_plans`, `paired_with` (greedy only),
  `division_space` → `GeneratedSpace`; `residency_edges` and `tile_reads` → relations;
  `op_position` → `produced_at`). Sizes stay in alignment units on the node as the draft
  has them; `address` is written back in bytes.
- **3c. The allocator builds the graph** at `_build_bound_buffers` (:1456/:1497),
  `_append_lx_relayout_destinations` (:1819), `_build_cd_bound_buffers`
  (:4074/:4275) and `_relayout_copy_buffers` (:4418). Its three `RelayoutCopyBuffer`
  checks (:2769, :2964, :3099) become RELAYOUT_COPY lookups and the drain-plan
  `CoreDivisionBuffer` assertion (:713) a node check. `cd_parent_matches` /
  `cd_parent_relayouts` are deleted only once 2c/2d replace them.
- **3d. Placement-only solvers** (greedy, first-fit/best-fit, SA, cooling_schedules,
  utils.plot_buffers) swap the type; greedy drops its isinstance at :183.
- **3e. Exhaustive search and the SA co-optimizer** read relations from edges and
  write `assignment`/`resident` on nodes; SA's `_spill_cost` collapses onto
  `buffer_graph.spill_cost` and its op adjacency onto execution edges.
- **3f. Tests**: 194 constructor calls across 14 files, mostly through helpers
  (`make_buffer_registry`, `synthetic_cooptimization_graphs._buf`,
  `cooptimization_capture_loader._buf`, `test_perm_layout_solver._buf`), plus direct
  calls in `test_scratchpad_solver.py` (131), `test_solver_relayout_decision.py`,
  `test_simulated_annealing.py` (17), `test_work_division_hint.py` (6) and
  `test_solver_cost_model.py` (2).

## Verification (every Phase 1 stage)

- Unit suites, serial (conftest opens the card; xdist workers collide), on a frozen
  `git archive` of the change vs the frozen base (`~/gap-probe/cpsat-graph-model/base`):
  **1121 passed, 5 skipped, 11 xfailed** over test_scratchpad_solver,
  test_solver_relayout_decision, test_cooptimization_types, test_cost_expr_dump,
  test_restickify_cost_model, test_sa_cooptimizer, test_op_features,
  test_coarse_tiling, test_scratchpad_patterns, test_perm_layout_solver, plus
  test_buffer_graph from Stage 1.
- Stage 0 harness: model proto identical per ladder stage, and decisions and
  objectives identical, except Stage 7 as noted.
- After Stages 4, 6 and 8: the device suites that compile through CP-SAT —
  test_scratchpad_use, test_work_division, test_solver_relayout_e2e,
  test_solver_auto_coarse_tiling — base vs change, same serial setup.
- `pre-commit run --files <changed>` (ruff, mypy).
