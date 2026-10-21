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

- **`ResourceNode`** — the main object of the graph: everything else (edges, relayout
  layouts, decisions) is reached from a node, and the graph hands out nodes
  (`get_nodes()`). A node is every resource an op needs: op outputs and graph inputs. A
  relayout is not a node: it is a second layout of the producer's output with a longer
  lifetime, and realizing the plan inserts the copy. A node holds what the op is
  (size, lifetime, its candidate menu, …) and its decisions (`assignment`, `resident`,
  `offset`), each fixed ahead of time or `None` for the solver to find. How each
  decision is enumerated (`Fixed`, `Table`, `Lazy` or `Pruned`) belongs to the
  strategy; see *Design rules*.
- **`DependencyEdge`** — every data dependency between two nodes (reads, in-place
  reuse). It holds *relations* over the two ends' decisions: the
  per-core view match (`ViewRelation`) and, on a read, the relayouts that can bridge a
  mismatch (`RelayoutRelation`). Edges are shared: the same edge object sits in its
  source's `outputs` and its destination's `inputs`, so either end can walk it.
- **`ExecutionEdge`** — links each op output (a node with a `Produced` origin) to its
  chronological neighbours (`origin.prev` / `origin.next`). Coarse tiling is handled
  here: whether the two ops run the same loop nest (`NestRelation`), whether they join
  one tiling group, and where a group breaks.
- **`SpyreOptimizationGraph`** — a collection of nodes (which own the edges), plus the
  global context (alignment, LX capacity). `graph.solve(model, cost_expr)` fills in
  every free decision and returns the op-output nodes in execution order, which is
  the execution plan.

Three phases, each stage its own PR that leaves the full suite green:

- **Phase 1 — re-representation (CP-SAT internals).** The graph carries the
  allocator's final data and every relation uses its *table* encoding, which is
  today's model. Each stage is proven by an identical CP model (see *Verification*).
  The allocator's three classes stay the external contract.
- **Phase 2 — constraint encodings (CP-SAT internals).** Each axis gains a `Lazy`
  strategy and each relation a constraint encoding over per-axis split / per-level tile
  vars, selected through the strategy factory and checked against the table encoding.
  Then menus stop being materialized where the config asks for it.
- **Phase 3 — one problem definition.** `graph.set_solver(...)`, then the allocator
  builds the graph, the other solvers move onto it, and the allocator subclasses
  collapse into one allocator whose behaviour comes from the backend and the
  enumeration strategies.

Work happens in worktree `.claude/worktrees/cpsat-graph-model` (branch
`cpsat-graph-model`, off `main` 82edd8f6). No commits (user commits).
`scratchpad/buffer_graph.py` is already drafted there (`CoreDivisionBoundBuffer`,
`BufferGraph`, `build_buffer_graph`, `SliceEdge`/`InPlaceEdge`/`CopyEdge`,
`Timeline.runs`). Stages 1–2 rename and reshape it into the model below.

## Design rules

1. **Relations, not tables.** Every relationship is an object over the two ends'
   decisions. Its encoding follows the strategies at its two ends (see rule 5): a
   table over menu indices when both are `Table` (Phase 1, today's model), constraints
   over per-axis vars otherwise (Phase 2). Nothing outside a relation reads menu
   indices; queries take `Assignment`s.
2. **Phase 1 derives nothing the allocator decided.** Match tables (including the
   loop-carry intersection and input-clone overrides, `allocator.py:4160-4240`),
   capped relayouts and the copies built from them, read counts (`len(uses)`, which
   include extern and placement-only readers), lifetimes and their overrides, and
   exclusions all come in as data. Moving any derivation out of the allocator is
   Phase 2/3 work with its own agreement test.
3. **`None` means an open decision.** A node has three decisions: `assignment` (core
   division and tiling), `resident` and `offset`. Each is either open (`None`; the
   solver decides) or fixed: by its axis's `Fixed` strategy, or pinned in the model by
   the strategy at finalization. Decisions are fixed in exactly three places:
   - **At finalization, by the strategy's op constraints.**
     `strategy.apply_op_constraints(node, model)` applies the restrictions for the
     node's kind of op (today's `residency_reason`s: no LX reader, partial-reduction
     write, unrepresentable view, and so on). The default strategy pins
     `resident=False` and records the reason; another strategy may post a restriction
     as a constraint instead (for example "a split reduction axis ⇒ not resident"),
     which is why this lives in the strategy and not at construction. In Phase 1 the
     reasons are the allocator's, as given (rule 2); in Phase 3 they come from the op.
   - **At construction, by the builder:** a placement-only buffer's division is
     `Fixed(COMMITTED)`.
   - **By the caller** (Stage 8): pinned `assignment`, `resident` or `offset`.

   Structural fields are never `Optional`. `Fixed` is a strategy on a decision axis,
   not a buffer type: it posts constants (or nothing) where an open axis posts vars.
4. **Build structure, then finalize once, through a strategy.** `add_node` /
   `add_child` / `add_successor` wire the graph and touch no model; nodes and edges are
   pure data. `solve(model, cost_expr, strategy)` calls
   `SpyreOptimizationGraph._finalize(model, strategy)`, which runs the `Strategy`'s
   `apply_*` steps over the graph in a fixed order. All encoding lives in the
   strategy.
5. **Enumeration is a polymorphic strategy, per decision axis.** Nodes do not carry
   enumerations; the `Strategy` does. For each node and each decision axis (core
   division, coarse tiling, residency, offset; relayout choice per READ edge) it holds
   an `Enumeration`, data plus `holds(value)` (a query, no model), built from the
   node's fields, and it realizes that enumeration in the model:
   - `Fixed(value)` — known before the solve; posts constants. `Fixed(COMMITTED)` is a
     placement-only buffer: committed upstream by work division, posts nothing.
   - `Table(candidates)` — explicit candidates: an index var plus element ties
     (today's menu). Core division and tiling share one `Table` (today's joint menu)
     until tiling goes lazy, then they separate.
   - `Lazy(space)` — per-axis vars plus legality as constraints, no candidates.
   - `Pruned(space, keep)` — generate, keep a shortlist, realize it as a `Table`.
     Today's relayout group capping is a `Pruned` relayout enumeration.

   The `EnumerationFactory` validates the config and builds a `Strategy` for a graph,
   choosing each node's enumerations from its fields (for example, `Lazy` tiling
   requires tiling to be its own axis). The default reproduces today: `Table` over the
   node's menu, `Fixed(COMMITTED)` for an empty menu (placement-only), `Pruned`
   relayouts.

   Whatever a node's enumeration, the strategy exposes the same per-axis vars for it
   (`split(key)`, `level(k)`; for a `Table` it makes them by element, as today). So a
   relation has one general encoding, per-axis constraints, plus the pair table as an
   optimization when both ends are `Table`:

   | src \ dst | `Fixed` | `Table` / `Pruned` | `Lazy` |
   |---|---|---|---|
   | `Fixed` | checked at build | allowed indices | per-axis equality to constants |
   | `Table` / `Pruned` | allowed indices | pair table (today) | per-axis equality |
   | `Lazy` | per-axis equality to constants | per-axis equality | per-axis equality |

   **Composing strategies across axes.** Every strategy realizes onto the same
   per-axis vars, so composing them is a conjunction of constraints over those vars:
   - `Fixed` with anything: the fixed axis's vars are constants.
   - `Table` and `Lazy` on independent axes: each posts its own legality; nothing joins
     them.
   - `Table` and `Lazy` on dependent axes (a menu item is legal only for some values
     of the other axis, e.g. a division whose span limit depends on the tiling): the
     menu carries, per item, its condition on the other axis. It is posted either as
     one per-node table over (menu index, the lazy vars the condition reads), or as
     `division == i ⇒ condition_i` per item. Both stay per node, never per pair. For
     division × tiling that is |menu| × at most 64 tiling options.
   - If an item's condition can't be stated without enumerating the lazy axis in full
     (a table past a size bound, or no closed form), the factory rejects that
     composition and keeps both axes in one joint `Table` (today's menu) or makes both
     `Lazy`. Each strategy declares the other axes it reads, and that declaration is
     what the factory validates.

## The model

```python
class DependencyKind(Enum):
    READ = auto()          # dst's op reads src
    IN_PLACE = auto()      # dst may take over src's storage at the shared tick

@dataclass(eq=False)
class DependencyEdge:
    src: ResourceNode
    dst: ResourceNode
    kind: DependencyKind
    view: Optional[ViewRelation]             # None only: IN_PLACE whose child was
                                             # fixed upstream (ungated)
    relayout: Optional[RelayoutRelation]     # READ only: src's relayout layouts
                                             # that can serve this read
    chosen: Optional[RelayoutRead] = None    # filled by solve()

@dataclass(eq=False)
class ExecutionEdge:
    before: ResourceNode
    after: ResourceNode
    contiguous: bool                         # consecutive ticks
    nest: NestRelation

    @property
    def joinable(self) -> bool:              # can the two ops share a nest at all?
        return self.contiguous and self.before.tileable and self.after.tileable

@dataclass(eq=False)
class Produced:                              # written by an op in the solve
    produced_at: int                         # tick of the producing write
    prev: ExecutionEdge                      # always linked:
    next: ExecutionEdge                      # graph.head -> ... -> graph.tail

@dataclass(frozen=True)
class GraphInput:                            # no producing op; not on the chain
    first_read: int                          # tick of its first read (clone-in)

@dataclass(eq=False)
class RelayoutLayout:
    """The node's output re-laid-out from its native view: a second copy in LX with
    its own decisions, live from the first read it serves to the last (today's
    copy lifetime, `build_relayout_copy`)."""
    group: int                               # destination view (today's group)
    candidates: tuple[RelayoutCandidate, ...]  # priced source divisions
    start_time: int                          # first consumer it can serve (the
    end_time: int                            # shuffle runs just before); last one
    resident: Optional[bool]                 # None = free
    offset: Optional[int]                    # None = free

@dataclass(eq=False)
class ResourceNode:
    # what the resource is: filled by add_node from an allocator buffer (Phase 1)
    # or the torch IR (Phase 3); fixed once built
    name: str
    size: int                                # alignment units
    start_time: int                          # lifetime, overrides applied
    end_time: int
    read_count: int                          # every reader, extern ones included
    first_use_is_read: bool
    boundary: BufferType
    sym_core_divs: Mapping[SplitKey, sympy.Symbol]
    sym_tile_counts: Mapping[TileKey, sympy.Symbol]
    sym_division: sympy.Symbol
    menu: Sequence[CoreDivision]             # the op's candidates (empty:
                                             # placement-only); the strategy decides
                                             # whether to enumerate them
    residency_reason: Optional[str]          # Phase 1: the allocator's; Phase 3: from
                                             # the op; applied by apply_op_constraints
    # decisions
    assignment: Optional[Assignment]         # None = free
    resident: Optional[bool]                 # None = free
    offset: Optional[int]                    # None = free
    # structure
    origin: Produced | GraphInput            # where the resource comes from
    relayouts: list[RelayoutLayout]          # alternative layouts of this output
    inputs: list[DependencyEdge]
    outputs: list[DependencyEdge]

    @property
    def tiling(self) -> Optional[TileSpec]   # assignment.tiling once set
    @property
    def tileable(self) -> bool               # some menu tiling is not untiled

    def add_child(
        self,
        child: ResourceNode,
        kind: DependencyKind = DependencyKind.READ,
        *,
        view: Optional[ViewRelation],
        relayout: Optional[RelayoutRelation] = None,
    ) -> DependencyEdge                      # the only way to add an edge
    def add_successor(self, after: ResourceNode) -> ExecutionEdge
                                             # the only way to add an execution edge

class SpyreOptimizationGraph:
    alignment: int
    capacity: int                            # LX capacity in alignment units
    nodes: dict[str, ResourceNode]            # add_node order = today's var order
    head: Sentinel                           # chain ends; not in nodes, no vars,
    tail: Sentinel                           # never returned by solve()

    def add_node(
        self, source: LifetimeBoundBuffer | torch._inductor.ir.Operation
    ) -> ResourceNode
        # the graph fills the node's fields from either source: an allocator buffer
        # (or its CoreDivisionBuffer / RelayoutCopyBuffer subclass) in Phase 1, an
        # IR op from Phase 3 (returns the existing node if that op was added)
    def get_nodes(self) -> Iterator[ResourceNode]  # in add_node order; edges are
                                                   # reached through node.inputs /
                                                   # node.outputs, never listed
    def solve(
        self,
        model: cp_model.CpModel,
        cost_expr: Optional[sympy.Expr] = None,  # None or unlowerable: fallback
        strategy: Optional[Strategy] = None,     # None: the factory's default
    ) -> list[ResourceNode]                   # op outputs in execution order
    def _finalize(self, model, strategy) -> None  # the strategy's apply_* steps,
                                                  # in today's order
```

The `Enumeration`s, `Assignment` and the three relations (`ViewRelation`,
`RelayoutRelation`, `NestRelation`) are data the graph holds; the `Strategy` encodes
them. Each stage below introduces only the part it needs (Stage 1: data and queries;
Stage 5: `Strategy.apply_core_division`; Stage 6: the other `apply_*` steps; Phase 2:
`Lazy` / `Pruned` and the mixed-pair encodings).

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
- **`_finalize(model, strategy)` runs the strategy step by step across all nodes**,
  not node by node, because today's model is built step by step:

  ```python
  def _finalize(self, model, strategy):
      for node in self.nodes: strategy.apply_core_division(node, model)
      for node in self.nodes: strategy.apply_inplace(node, model)
      strategy.apply_placement(self, model)
      for node in self.nodes: strategy.apply_op_constraints(node, model)
      for node in self.nodes: strategy.apply_reads(node, model)
      strategy.apply_relayout(self, model)
      strategy.apply_coarse_tiling(self, model)
      strategy.apply_cloning(self, model)
  ```

  1. **`apply_core_division(node)`**, in `add_node` order: the division, `eff_size`,
     `cores`, `core_cost`, split and tile-count vars, `in_buffer`, `offset`, and the
     `merge_*` literal for each IN_PLACE input (today these are made per wrapper,
     `:226`). Then each node's relayout layouts' vars, after all nodes, in today's
     copy order (copies are appended last, `allocator.py:4308`).
  2. **`apply_inplace(node)`**, over the node's IN_PLACE `inputs` (by destination, as
     today): merge constraints, then at-most-one over its incoming and outgoing
     merges.
  3. **`apply_placement(graph)`**: capacity per node (`offset + eff_size <= M`) and the
     2-D no-overlap (with the incoming-merge tick shortening).
  4. **`apply_op_constraints(node)`**, in `add_node` order: the restrictions for the
     node's kind of op. In the default strategy these are today's exclusion pins
     (`resident=False` with the reason), posted exactly where they are today, after
     the no-overlap and before any read gate; the literal was made in step 1.
  5. **`apply_reads(node)`**, over the node's READ `outputs` (parent order, then child
     order, then the child's `parents` order): per edge, the served literals in sorted
     relayout-group order (`:440`) and the view gate, which is "match or served",
     interleaved as today's `constrain_residency` does.
  6. **`apply_relayout(graph)`**: each relayout layout in copy order: priced source
     divisions, "a resident layout serves some read", and today's pins (source not
     choosing a division, no read can use it).
  7. **`apply_coarse_tiling(graph)`**: `loop_id`/`tiled` per divided node, then
     execution edges along the chain (`same_nest`, `joined`, group step), then cuts,
     then tile counts.
  8. **`apply_cloning(graph)`**: input clones. A no-op in Phase 1, where a clone's
     matches are ordinary READ-edge tables; in Phase 2c it posts the clone's relation
     derived from its consumers.

  The fallback copy pin (`:1044-1048`) is not here: it depends on whether `cost_expr`
  lowered, so it stays in the ladder inside `solve`. Stage 0's proto diff is the check
  on every claim in this list.
- `solve(model, cost_expr, strategy)` is the only public entry point after building:
  it runs `_finalize(model, strategy)`, then the objective ladder over the sympy
  `cost_expr` (lowered as `_minimize_cost_expr` does today, with today's fallback
  objective when it is `None` or fails to lower), then extraction, and fills in every
  field that was `None`.
- Coarse tiling splits across the two edge types the way the data does: group
  membership is decided on `ExecutionEdge`s, and whether a tiled value must be cut to
  HBM by checking its READ edges against those groups.
- Out of scope: solver-chosen op order. Execution order is an input (`add_successor`).

- **Graph inputs are nodes** with a `GraphInput` origin: no producing op and not on
  the execution chain (their first use is another op's read tick). They have a
  division enumeration (today's clone menu) and residency like any node.
- **Relayouts are layouts, not nodes.** A `RelayoutLayout` is the producer's output
  in a destination view, with its own residency and offset and a lifetime extended to
  the last read it serves. Realizing the plan inserts the copy op (the shuffle
  directly before its first consumer) only for resident layouts. Copies do not break
  tiling runs today, and with no copy node on the chain that stays exact.

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

- Rename the draft (`CoreDivisionBoundBuffer` → `ResourceNode`, `BufferGraph` →
  `SpyreOptimizationGraph`, `build_buffer_graph` → `build_graph`) and reshape it into
  the graph above. Drop `SliceEdge`/`InPlaceEdge`/`CopyEdge`/`Timeline`.
- The supporting types, data and queries only (no solver methods yet):

  ```python
  @dataclass(frozen=True)
  class Assignment:                    # a decided split
      splits: Mapping[SplitKey, int]   # sym_core_divs keys
      tiling: TileSpec
      COMMITTED: ClassVar[Assignment]  # fixed upstream by work division; its splits
                                       # are not visible to the solve (placement-only)

  class Enumeration(ABC):              # one decision axis of a node
      def holds(self, value) -> bool
  class Fixed(Enumeration):            # value known before the solve
      value: Any
  class Table(Enumeration):            # explicit candidates (the allocator's menu)
      candidates: list[CoreDivision]   # division and tiling share it for now

  @dataclass(frozen=True)
  class Enumerations:                  # one node's enumeration per axis; held by
      division: Enumeration            # the Strategy, not the node. Division carries
      residency: Enumeration           # tiling while it is a Table; residency is
      offset: Enumeration              # open (apply_op_constraints may pin it)

  class Strategy:                      # encodes a graph; methods arrive in Stages 5-6
      enumerations: dict[ResourceNode, Enumerations]

  class EnumerationFactory:            # validates the config and builds strategies
      def strategy(self, graph: SpyreOptimizationGraph) -> Strategy
          # default: today's model, from each node's fields (menu -> Table,
          # empty menu -> Fixed(COMMITTED))

  class ViewRelation:                  # both ends own the buffer with the same view
      table: DivisionMap               # allocator's final pairs; empty = never alike
      def holds(self, src: Assignment, dst: Assignment) -> bool
  class RelayoutRelation:              # READ: src's relayout layouts that can serve
      reads: tuple[RelayoutRead, ...]  # a mismatch; allocator's capped candidates
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
  a second neighbour; `joinable` is false across a tick gap and at untileable ops;
  sentinels are not in `nodes`; `tiling` is `None` while `assignment` is.

### Stage 2. Builder

- `build_graph(buffers, alignment, capacity)` makes the `SpyreOptimizationGraph`,
  calls `add_node(buffer)` for each allocator buffer in the allocator's order, then
  adds the edges through `add_child`. `add_node` does the conversion inside the
  graph: it fills the node's fields, with the buffer's `core_divisions` as the menu (a
  `LifetimeBoundBuffer` has none: placement-only). The strategy built for the graph
  later turns that into `Table` or `Fixed(COMMITTED)`. The allocator classes are read
  only there and in the edge wiring below; past `add_node` the node never refers back
  to its buffer. The final data copied:
  - READ edges only between nodes that both have a menu, i.e. neither is
    `COMMITTED` (today's no-op `constrain_residency` on placement-only wrappers).
    `view.table` is `cd_parent_matches.get(src, ())`; `relayout` is
    `cd_parent_relayouts[src]` filtered to groups with a copy in this solve.
  - IN_PLACE edges with `view=None` when the child's division was fixed upstream,
    otherwise the same table.
  - Each `RelayoutCopyBuffer` becomes a `RelayoutLayout` on its `relayout_parent`'s
    node (not a node), in the allocator's order; graph inputs become nodes with a
    `GraphInput` origin.
  - `read_count`, lifetimes and overrides and `spill_cost` inputs as given; each
    buffer's `residency_reason` is kept on its node, for `apply_op_constraints` to
    apply.
  - Origins: `Produced(uses[0])` for op outputs (today's filter: has `uses`, not
    `first_use_is_read`, not a copy), `GraphInput(uses[0])` for graph inputs; then
    `add_successor` over the `Produced` nodes in `produced_at` order;
    each `NestRelation` table comes from the two nodes' candidate
    `TileSpec.level_counts` (equal and non-empty).
- Tests: on every corpus graph the edges match `_get_children`, `_relayout_copies` and
  `in_place_parents`, and each nest table equals `loop_id` equality ∧ `tiled[after]`.
  The `_get_children`/`_relayout_copies` checks go in Stage 3; the nest check in
  Phase 2b.

- Where each `CoreDivisionBuffer` field goes. In Phase 1 a node is a
  `CoreDivisionBuffer` redistributed: what it is (including its menu) in its fields,
  how it relates to others on edges, what was decided in its decision fields; how its
  choices are enumerated belongs to the strategy.

  | `CoreDivisionBuffer` field | Phase 1 home |
  |---|---|
  | `name`, `size`, `first_use_is_read`, `boundary` | node fields |
  | `uses`, `lifetime_*_override` | `start_time` / `end_time` / `read_count`; `uses[0]` → `produced_at` |
  | `residency_reason` | `residency_reason`, applied by `apply_op_constraints` |
  | `core_divisions` (with each `.tiling`) | `menu`; the default strategy's `Table` (division and tiling shared) |
  | `parents`, `cd_parent_matches`, `cd_parent_relayouts` | READ edges: `ViewRelation.table`, `RelayoutRelation` |
  | `in_place_parents` | IN_PLACE edges |
  | `RelayoutCopyBuffer` (whole buffer) | a `RelayoutLayout` on its parent's node |
  | `chosen_division`, `address`, `spill_reason`, `chosen_relayouts` | decisions and `chosen`, written back by Stage 8 |
  | `residency_edges`, `division_space`, `tile_reads` | not carried; Phase 2c / 2f / 2b |
  | `op_position` | replaced by `produced_at` and execution edges |
  | `paired_with`, `lx_view`, `lx_relayout_plans` | not read by CP-SAT; Phase 3b |

  A `LifetimeBoundBuffer` maps the same way with an empty menu (`Fixed(COMMITTED)`).

### Stage 3. Solver reads relationships from the graph

- `_plan_layout_generic` builds the `SpyreOptimizationGraph` next to the existing
  wrappers.
- `_get_children` → `node.outputs` of kind READ; `_relayout_copies` → the nodes'
  `relayouts`; the `in_place_parents` loop in
  `_add_inplace_relaxation` → each node's IN_PLACE `inputs`.
- Delete `_get_children` and `_relayout_copies`. The wrappers still own the CP vars.
- Proto diff empty.

### Stage 4. Tiling groups on execution edges

- One per-edge function owns an execution edge's coarse-tiling encoding (it moves
  into `Strategy.apply_coarse_tiling` in Stage 6): `same_nest`
  and `joined` (today's `samenest_*` / `joined_*`) and the group step
  (`group[after] = group[before] + 1 - joined`). A non-joinable edge starts a new
  segment with group 0. `same_nest` keeps today's `loop_id` encoding, `tile_group_*`
  keeps its domain, and `loop_id`/`tiled` are still made first, in node order.
- `_tiling_group_ids` becomes a walk along the execution chain, starting a new segment
  wherever an edge is not `joinable`; the segments are local to the coarse-tiling
  encoding (in `Strategy.apply_coarse_tiling` from Stage 6), not a graph method. Its
  `uses[0]` logic and `RelayoutCopyBuffer` check go away.
- `_cut_literals`: for each tiled node, consumers on its READ edges in the same segment
  get an `apart` literal gated on `edge.view`; a consumer outside the segment, or a
  graph output, makes the node unshareable. `_tile_count_terms` iterates nodes.
- Proto diff empty (coarse-tiling corpus with `auto_coarse_tiling` on).

### Stage 5. `Strategy.apply_core_division`: the strategy makes node vars

- Adds `Strategy` and the default strategy's `apply_core_division(node, model)`,
  which records the node's vars as the relations will see them (`NodeVars`:
  `division`, `eff_size`, `cores`, `core_cost`, split and tile-count vars,
  `in_buffer`, `offset`, merge literals). It dispatches on the node's division
  enumeration, replacing the variable-making half of both wrappers: `Table` posts
  today's index var, element ties and single-candidate constants; `Fixed(COMMITTED)`
  posts no split vars, which is today's placement-only model (`cores`, `core_cost`
  `None`, no gates). Merge literals are made here, in node order.
- The node also exposes `buffer`, `division` and `division_is` for
  `cpsat_expr_printer`, so that contract is unchanged until Phase 2a. `buffer` is a
  read-only view of the node, not an allocator buffer: the printer reads exactly
  `core_divisions` (the `Table` candidates), `sym_core_divs` and `sym_division`
  (node fields) through it (`cpsat_expr_printer.py:249`, `:840`, `:881-895`). Var names
  stay as today.
- The wrappers hold their node and delegate to it, so no constraint code changes.
- Proto diff empty.

### Stage 6. The rest of the strategy; delete the wrappers

- The default strategy gains the remaining steps, with the table encodings of the
  three relations (today's `_gate_divisions` for views, the served literals for
  relayouts, the Stage 4 `same_nest` literal for nests):
  - `apply_inplace(node)`: per IN_PLACE input, the merge literal; `eff_size` equality
    when the child chooses a division; offset equality; both-resident implications;
    the view gate unless `view is None`; then at-most-one.
  - `apply_placement(graph)`: capacity and 2-D no-overlap with tick shortening.
  - `apply_op_constraints(node)`: the exclusion pins, in today's position.
  - `apply_reads(node)`: per READ output, served literals and "match or served"; the
    served literals are kept on the edge.
  - `apply_relayout(graph)`: today's `_constrain_relayout_copies`, over the nodes'
    relayout layouts.
  - `apply_coarse_tiling(graph)`: takes over the Stage 4 function, cuts and tile
    counts.
  - `apply_cloning(graph)`: a no-op in Phase 1.
- `SpyreOptimizationGraph._finalize(model, strategy)` runs them in the order under
  *The model*.
- `free_copies` (presolve threshold) keeps today's rule: every relayout layout whose
  node chooses a division.
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
  `_run` takes today, plus the strategy (default: the factory's):
  `_finalize(model, strategy)`, the objective ladder (with the fallback copy
  pin when `cost_expr` is `None` or fails to lower), extraction. It returns the
  op-output nodes along the execution chain with every `None` field filled
  (`assignment`, so the tiling; `resident`; `offset`) and `chosen` set on READ edges.
  That list is the execution plan.
- Pre-fixed decisions are `Fixed` strategies: a concrete `Fixed` division posts a
  constant index (its edges keep their gates, with that side constant);
  `Fixed(COMMITTED)` posts nothing, as above; `Fixed` residency / offset pin their
  vars. Bottom-justify is skipped when any
  offset was pinned.
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

Each stage adds a strategy or encoding, selectable through the `EnumerationFactory`
(default unchanged), and an **agreement test** against the table encoding on the full
corpus and the device graphs: same feasible (src, dst) assignments for a relation, or
same optimum for a solve. Mixed strategies on one graph (for example `Lazy` on tiled
ops, `Table` elsewhere) use the per-axis encodings from rule 5. What is not linear
is called out; it needs `AddMultiplicationEquality` / `AddDivisionEquality` or table
constraints over small per-buffer domains, never per-pair tables.

- **2.0. Probe: is lazy faster than eager? (go / no-go for the rest of Phase 2.)**
  Phase 1 does not depend on it, so it can run alongside. The code is in
  `probes/cpsat-graph/deenum/` (see its README): phase timers and solver-input
  capture during a compile, then a device-free replay of the captures:
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

  **Minimal probe, run 2026-10-21 on `main` 82edd8f6** (`probes/cpsat-graph/deenum/`:
  `timers.py` phase profile + capture, `lazy_probe.py`, `report_probe.py`; 32 zoo
  graphs and Granite 4.0-micro, default config and `AUTO_COARSE_TILING=1`).

  *Where eager planning time goes today* (exclusive seconds; enumeration = menus +
  match tables):

  | config | planning | menus | match tables | relayout pricing | lowering + cost build | CP-SAT solve |
  |---|---|---|---|---|---|---|
  | zoo, default | 152.5 s | 2.3 s (1.5%) | 3.0 s (2%) | 52.2 s (34%) | 44.4 s (29%) | 44.8 s (29%) |
  | zoo, tiling | 292.4 s | 63.1 s (22%) | 138.6 s (47%) | 56.6 s (19%) | 0 | 28.9 s (10%) |
  | Granite, default | 170.6 s | 1.2 s (1%) | 1.5 s (1%) | 11.1 s (7%) | 29.1 s (17%) | 124.9 s (73%) |
  | Granite, tiling (11 of 15 plans) | 82.3 s | 25.4 s (31%) | 36.1 s (44%) | 10.5 s (13%) | 0 | 9.0 s (11%) |

  Under tiling, `match_pairs` compares 63.7M candidate pairs (62M `same_partition`
  calls) to produce 33K pairs; menus are re-enumerated per tiling.

  *Is the per-axis relation exact?* The per-device-dim `(split, stride)` + core-count
  key reproduces every pair table (385/385 zoo, 200/200 Granite, both configs). The
  inferred per-axis rule (symbol → device dim, iteration order, contiguous dim), plus
  the same rule over tile splits for tile keys, reproduces the allocator's **final**
  table on 382/382 zoo edges in both configs (tiled ones included) and on 192/200
  Granite default / 94/100 Granite tiling edges (irregular owner formulas, and 2
  tile rules that could not be inferred). Tile keys follow two conventions: the
  `tile_splits` order, and a side whose tiling slices none of the buffer reads it
  whole.

  *Solver cost.* One reduced model (residency, footprint, capacity, 2-D no-overlap,
  match gates; no cost expression, relayouts, in-place or cuts) built three ways: E
  (menu index + pair tables, today), L1 (per-axis split and tile vars, lazy gates,
  per-node menu legality) and L2 (no menu: per-axis domains, core bound, one
  reduction split). 30 s limit, seed 0. Proven optima agree everywhere: 0 objective
  mismatches on every pair where both arms reached OPTIMAL, in every config. (A first
  tiling run showed lazy arms beating E's proven optimum; the cause was a probe bug:
  the fallback pair table for edges without view keys ignored tile counts. Fixed, and
  all tiling L1/L2 solves re-run.) Time to OPTIMAL vs E, pairs where E took ≥ 0.2 s:

  | config | workers | n | L1 median (range) | L2 median (range) |
  |---|---|---|---|---|
  | zoo, default | 16 | 8 | 1.02× (0.73–1.41) | 0.96× (0.77–1.28) |
  | zoo, default | 1 | 7 | 1.44× (0.30–2.55) | 1.36× (0.21–3.87) |
  | zoo, default | 16, interleaved | 8 | 1.11× (0.68–2.30) | — |
  | zoo, tiling | 16 | 10 | 0.56× (0.38–1.58) | 0.50× (0.10–1.52) |
  | zoo, tiling | 1 | 5 | 0.80× (0.27–2.79) | 0.09× (0.08–0.57) |
  | Granite, default and tiling | 1 and 16 | 0–2 | too few solves ≥ 0.2 s | |

  Lazy model build is slower in Python (0.38 s vs 0.12 s over all 32 zoo solves) but
  negligible next to planning.

  *Eager + hash join* (the cheap Python fix, `DEENUM_HASHJOIN=1`): exact on 385/385
  edges in both configs. Zoo tiling planning 292 → 179 s (match tables 138.6 →
  23.2 s); default 3.0 → 2.5 s of match tables. Granite produced no usable numbers
  (the default run hit the compile cache, and the tiling run hits a pre-existing
  `Cannot tile buf0 … read-distance limit` failure on `main`).

  *Deterministic multi-worker search* (`interleave_search`, 16 workers, two repeats):
  not reproducible with a wall-clock limit. 33/36 proven optima repeated the same
  solution, but only 1/27 time-limited solves did, and deterministic time never
  matched. Next: a `max_deterministic_time` budget instead of seconds.

  *Expected improvement* (lazy removes menu and match-table construction; relayout
  pricing and cost-model prediction stay until 2a/2d; solve scaled by the medians
  above):

  | config | today | eager + hash join | views + nests lazy | lazy vs hash join |
  |---|---|---|---|---|
  | zoo, tiling | 292 s | 179 s | ~78 s | **~2.3× faster** |
  | Granite, tiling | 82 s | not measured | ~21 s | — |
  | zoo, default, 16 workers | 152 s | ~152 s | ~148 s | ~3% faster |
  | zoo, default, 1 worker | 152 s | ~152 s | ~167 s | ~9% slower |
  | Granite, default | 171 s | — | ~168 s | ~2% (solve-bound) |

  **Verdict: pass for compiles with coarse tiling on, not on its own for default
  compiles.** Conditions, as of the follow-up runs:
  1. **Tile relation — done.** Exact on every zoo tiling edge and 94/100 Granite
     tiling edges, and the lazy solve is faster under tiling at both worker counts.
  2. **1-worker regression — open for the default config** (1.44×; 1.11× with
     interleaved 16-worker search, which is not yet reproducible). Under tiling there
     is no regression (0.80×, n=5).
  3. **Hash join — done.** Lazy views and nests are still ~2.3× faster than eager +
     hash join on zoo tiling. Menus (64 s, re-enumerated per tiling) are now the
     largest eager cost; a per-tiling menu cache is the cheap alternative and has not
     been measured.
  4. **Relayout pricing — legality closed form verified** (2d); price and projection
     gates remain.

- **2a. Cost expression over per-axis vars.** Today `cpsat_expr_printer` tabulates
  over the menu: `_division_table` (`:309`), `RelayoutCharge` keyed on `sym_division`
  (`:822`), `log2_` by element on the division, `AppliedUndef` menu tables (`:854`),
  and `extract_op_features` reads `candidate_work_slices`. Tabulate over per-axis
  tuples (`AddAllowedAssignments`, small at ≤ 32 cores) instead. Prerequisite for the
  rest; agreement: identical lowered values at every menu point.
- **2b. `Lazy` tiling; `NestRelation` on per-level tile vars.** Tiling becomes its
  own axis (the factory requires it for `Lazy` tiling). `NodeVars` gains `level(k)`:
  per node `L0`, `L1` (`max_dims=2`) tied to the per-dim counts; `same_nest` is the
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
  into the buffer), and the strategy gains a constraint encoding for views built from
  it. Loop carries are the conjunction of the read and update relations.

  **Input clones** (`apply_cloning`). Today `_clone_divisions_and_matches`
  (`allocator.py`) builds a clone's menu as the distinct non-broadcast read views of
  its consumers, each projected onto the symbols the read indexes, and each consumer's
  table as "my read view equals that clone view". A clone has no write view of its
  own: its view *is* the one its consumers read with. As constraints:
  - The clone's division axis is a view, not a split: per-device-dim (factor, stride)
    vars plus a core count, with no op behind them and no reduction split.
  - Each READ edge from the clone posts, gated by the clone's residency, the
    consumer's read view (from its split vars through its read map, as above) equal
    to the clone's view. Together these say every consumer reads it alike, which is
    today's "must match all consumers to be placed".
  - The edge also posts non-broadcast: the product of the consumer's factors on the
    buffer's dims equals the consumer's core count. A consumer that splits an axis
    the input lacks would read stale scratchpad on the cores without a copy, and today
    such a view is skipped.
  - A consumer with no `MemoryDep` read of the input or no `MemoryDep` write is an op
    constraint (`apply_op_constraints`): it keeps the clone out of LX, as today's empty
    table does.

  Agreement: the feasible pairs equal the allocator's final `cd_parent_matches`,
  carries and clones included.
- **2d. Relayouts as constraints.** One optional relayout layout per (parent, consumer)
  READ edge whose view is the consumer's; sharing between consumers by reified view
  equality.
  - **Legality has a verified closed form** (`probes/cpsat-graph/deenum/ms_check.py`).
    Over each view's digits (device dim, factor, stride) and core counts:
    1. domain: C_src divides C_dst, each view's core count is its domain, and the
       source split product is C_src; identical views on equal domains are not a
       relayout;
    2. source slices distinct: the source view's core → slot map is a bijection;
    3. destination covered: every destination slice is reached (evenly when the
       domains are equal);
    4. per dim one factor divides the other; fan-in = ∏ max(1, f_src/f_dst) ≤ 8;
    5. a larger destination domain is one-to-one: fan-in 1, and the destination
       slices nested in each source slice are replicated C_dst/C_src times in total.

    2 and 3 are per-view properties, and 5 depends on the destination view and the
    source's factors only, so as constraints these are small per-view tables, per-dim
    tables over (f_src, f_dst), a product bound and implications. Nothing enumerates
    view pairs, which is 73% of relayout time today (`movement_supported` →
    `transfer_edges` on 17,261 distinct pairs for `sdpa_gqa_1x32x8x512x128`).
    It agrees with `movement_supported` on all 61,391 decisions recorded over the 32
    zoo graphs and on 306,000 synthetic pairs (two seeds; `ms_synth.py`), with every
    rejection reason exercised.
  - Still to put in closed form: the projection gates (`work_division_from_view` on
    both frames, `partition_footprint` within LX) and the price (per split-dim tables ×
    1/cores, governing dim chosen piecewise).
  - Relayout choice becomes an axis with strategies: today's group capping
    (`_cap_relayout_groups`) is `Pruned`, and `Lazy` drops the cap (or caps views per
    buffer). Agreement where capping did not bind; differences reported where it did.
- **2e. In-place as constraints.** Per-core footprint equality
  (`out_partition · out_tiles`, a product equality) plus the `ViewRelation`. The
  pre-solve gates (pointwise input, equal device layout, one-tick handoff) stay
  builder facts.
- **2f. `Lazy` core division.** `Lazy(OpSplitSpace)`: per-axis domains from
  `_legal_split_factors`, product ≤ max cores (multiplication equality), at most one
  reduction split, blocked dims, the span cap as a per-dim table, tiling-dependent
  domains through the level vars. Menus are no longer materialized for nodes the
  factory makes `Lazy`; `Table` remains for small menus, tests and listed candidates,
  and `Pruned` for SA-seeded or cost-ranked shortlists. Agreement: same optimum
  wherever the menu was complete.

## Phase 3 — one problem definition (outline; own plan after Phase 2)

- **3a. `graph.set_solver(...)` over a backend protocol.** The model passed to `solve`
  becomes a backend; `cost_expr` stays the solver-neutral objective.
  - **The protocol is ours, shaped like `pywraplp`.** `pywraplp.Solver` (`ortools
    .linear_solver`) is the template for the API: `CreateSolver(id)`, `IntVar` /
    `BoolVar`, `Add`, `Minimize`, `Solve`, `solution_value`. On top of it go the
    operations the elements already post: reification (`enforce_if`), element /
    table, multiplication / division equality and 2-D no-overlap. `pywraplp` itself
    cannot be the protocol: in OR-Tools 9.15 it is linear-only from Python (no
    indicator constraints exposed), and `math_opt` adds indicator and quadratic
    constraints but no globals. Linearizing our model (big-M reification, one-hot
    element/table, pairwise no-overlap disjunctions, binary expansion of products)
    would hand CP-SAT a much weaker model, and the solve is 29–73% of planning today
    (Phase 2.0).
  - **Backends.**
    - `CpSat` — direct pass-through. It is the only backend that lands first.
    - `Mip` — over `pywraplp` (SCIP, HiGHS, CBC, or `CP_SAT` as an MPSolver),
      implementing the extra operations by linearization. It gives a second opinion
      on optima and access to MIP solvers.
    - `Annealing` — a wrapper that lets SA act as a generic solver behind the same
      protocol: it records vars, domains and constraints, then searches assignments
      with constraints as hard checks or penalties and the objective from
      `cost_expr`. A flat constraint list makes a weak SA, so the wrapper also takes
      optional neighbourhood providers from the elements: the view relations'
      geometry (Phase 2c; SA already works lazily over `ResidencyEdge` through
      `_ViewRelation`, `sa_cooptimizer.py:614`) and `Lazy` division generation
      (`:378`).
    - `BruteForce` — an exact oracle for small models. Like `Annealing` it records
      what the strategy posts (rows, `OnlyEnforceIf`, element / table, products,
      intervals, `AddNoOverlap2D`) and evaluates the constraints itself, with no
      propagation and no other solver. `Solve` is a depth-first search over integer
      assignments:
      - variables ordered presence and division first, offsets last, since most
        infeasibility shows up early and offsets have the largest domains;
      - each constraint checked as soon as all its variables are assigned, and
        partially where that is sound (a `≤` row with non-negative coefficients
        already over its bound; two placed rectangles that overlap), pruning the
        subtree;
      - optional branch and bound: an optimistic objective bound per subtree,
        pruned against the best feasible value so far.

      It is exact by construction and trusts no solver, so it is the reference the
      other backends are checked against. It is for tests only: a size guard
      refuses models past a bound on the product of variable domains (roughly 20–25
      binaries, offsets on a coarse grid).
    - `Decomposed` — enumerates only the `Table` axes (the division menus) and solves
      the rest of the model at each leaf with an inner backend (`CpSat`, `Annealing`,
      or a placement heuristic). The model is posted once; each leaf fixes its
      division choices through assumptions or domain changes, so nothing is rebuilt
      per leaf. It is exact when the inner backend is `CpSat`, and it minimizes the
      same `cost_expr` as every other backend (today's `ExhaustiveSearchSolver`
      minimizes total HBM bytes). It generalizes `ExhaustiveSearchSolver.wrapping`
      (3g).
  - **Correctness ladder.** `BruteForce` is the ground truth on tiny models: unit tests
    of each strategy's encodings (`Table` vs `Lazy` relations, the relayout closed
    form inside a model, `apply_op_constraints`) and of each backend adapter. `CpSat`
    must match it there, and is then the ground truth on real models. `Annealing`,
    `Mip` and `Decomposed` are measured against `CpSat`.
  - **Gate.** Before `Mip` or `Annealing` is offered, replay the Phase 2.0 captures
    (`probes/cpsat-graph/deenum/`) through each: same optimum as `CpSat` where both
    prove one, plus solve time. The linearization cost is measured, not assumed. The
    SA side is prototyped in `probes/cpsat-graph/sa_ilp/`: `SaSolver` implements this
    interface, including `OnlyEnforceIf`, optional intervals and `AddNoOverlap2D` as a
    scored global constraint, next to a `cp_model` adapter with the same names.
    Measured there (one builder, three backends): SA was feasible in 56/57 generic ILP
    runs and all 108 2-D layout runs (independently verified). It reached the optimum
    on LX residency in ~0.05 s and on 14/36 layout captures in ~0.15 s. It stalled
    5–47% short on assignment-heavy and tightly packed layouts, so `CpSat` stays the
    default and `Annealing` needs the graph's neighbourhood providers to compete.
  - **Brute-force scope (probe first).** Prototype `BruteForce` in
    `probes/cpsat-graph/sa_ilp/` behind the same interface. Cross-check it against
    `CpSat` and SA on small generic ILPs and on tiny LX layouts with 2-D no-overlap
    (a handful of buffers cut from the captures): same optimum as `CpSat` everywhere
    both finish, and record the size where the guard bites. Then use it as the oracle
    for the strategy unit tests above. `Decomposed` follows in 3g, once the allocator
    is merged.
- **3b. Node absorbs the allocator-only fields** (`in_place_parents` → IN_PLACE edges,
  `residency_reason` → `apply_op_constraints` reading the op, lifetime overrides,
  `lx_view`, `spill_reason`, `chosen_relayouts`, `lx_relayout_plans`, `paired_with`
  (greedy only), `division_space` → `Lazy` division; `residency_edges` and `tile_reads`
  → relations; `op_position` → `produced_at`). Sizes stay in alignment units on the
  node as the draft has them; `address` is written back in bytes.
- **3c. The graph builds nodes from the IR.** `graph.add_node(op)` builds a node
  straight from the torch IR (once per op): its fields from the op's layout
  and its place in `graph.operations`, with the op's menu or its `OpSplitSpace` as the
  data the strategy enumerates (`Table`) or generates from (`Lazy`), and the residency
  reasons that `apply_op_constraints` reads from the op.
  It replaces the allocator's buffer construction at `_build_bound_buffers`
  (:1456/:1497), `_append_lx_relayout_destinations` (:1819), `_build_cd_bound_buffers`
  (:4074/:4275) and `_relayout_copy_buffers` (:4418), so no intermediate buffer is
  built at all; the buffer form of `add_node` stays for tests and capture replay. The
  allocator's three `RelayoutCopyBuffer` checks (:2769, :2964, :3099) become
  relayout-layout lookups and the drain-plan `CoreDivisionBuffer` assertion (:713) a
  node check. `cd_parent_matches` / `cd_parent_relayouts` are deleted only once 2c/2d
  replace them.
- **3d. Placement-only solvers** (greedy, first-fit/best-fit, SA, cooling_schedules,
  utils.plot_buffers) swap the type; greedy drops its isinstance at :183.
- **3e. Exhaustive search and the SA co-optimizer** read relations from edges and
  write `assignment`/`resident` on nodes; SA's `_spill_cost` collapses onto
  `buffer_graph.spill_cost` and its op adjacency onto execution edges.
- **3g. One allocator.** Today `select_allocator` (`allocator.py:5117`) picks
  `ScratchpadAllocator` (33 methods, placement only) or `CoOptimizingAllocator(
  ScratchpadAllocator)` (41 more, joint) from `config.co_optimizing_lx_planning`, and
  the allocator branches on seven solver capability flags (`supports_paired_buffers`,
  `decides_lx_relayouts`, `generates_divisions`, `chooses_tilings`,
  `tilings_from_menu`, `replans_after_tiling`, `linear_cost_only`;
  `plan_solver.py:988-1035`). Every difference is which unknowns are open and who
  resolves them, which the graph now states generically, so the subclass goes:
  - **What is unknown** is the strategies. Placement-only is every division
    `Fixed(COMMITTED)`; joint is `Table` / `Lazy` / `Pruned`. The factory picks them
    from config, so `co_optimizing_lx_planning` becomes a factory setting.
  - **Who resolves it** is the backend (3a). The capability flags become what the
    backend reports it can realize (for example `Annealing` takes `Lazy` division and
    no pair tables), checked by the factory; an invalid combination is rejected at
    construction instead of branched on inside the allocator.
    `ExhaustiveSearchSolver.wrapping` becomes the `Decomposed` backend (3a): it
    enumerates the `Table` division axis around an inner backend and minimizes
    `cost_expr`, not HBM bytes.
  - **Where `CoOptimizingAllocator`'s methods go.**
    - Candidates (`_division_map`, `_enumerate_core_divisions`, `_tiling_candidates`,
      `_division_space`) → the strategy producers behind the factory.
    - Pair derivations (`_cd_parent_matches`, `_cd_parent_relayouts`,
      `_cap_relayout_groups`, `_clone_divisions_and_matches`, `_loop_carry_*`,
      `_parent_residency_edges`) → relations (2c / 2d).
    - Pricing (`_pricing_operations`, `_extract_op_features`,
      `_direct_read_candidates_priced`) → building `cost_expr`, the same for every
      backend.
    - Post-solve (`_post_solve`, `_materialize_selection`, `_apply_chosen_tilings`,
      `_commit_copy_out_divisions`, `_finalize_lx_relayout_allocation`) → one
      consumer of `solve()`'s execution plan, run unconditionally; with nothing open
      it changes nothing.
  - **`select_allocator`** returns the one allocator with (backend, factory) from
    config. Needs 3a, 3c and 3e. Verification: every `layout_solver` ×
    `co_optimizing_lx_planning` combination in the suites compiles to the same plans
    as before the merge.
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

## Probes

All measurement code behind this plan is in **`probes/cpsat-graph/`**, with a README
covering setup and every script. Nothing there changes torch-spyre: each hook is a
process-local monkeypatch. Run outputs and captures go to `deenum/runs/`, which is not
committed. Run with the torch-spyre checkout first on `PYTHONPATH` (the shell drivers
take it from `TORCH_SPYRE`); compiling needs the Spyre card, replays do not.

| probe | where | answers | used in |
|---|---|---|---|
| Eager profile and solver-input capture | `deenum/timers.py`, `zoo_t.py`, `granite_t.py`, `run_profile.sh` | where planning time goes per phase; captures each solve's inputs with per-candidate view and tile keys | 2.0 tables |
| Eager + hash join | `deenum/run_profile_hj.sh` (`DEENUM_HASHJOIN=1`) | the bar lazy has to beat: `match_pairs` as a hash join on the exact view key, checked against the original | 2.0 |
| Lazy vs eager replay | `deenum/lazy_probe.py`, `report_probe.py` | per-edge exactness of the per-axis view and tile relations; solve time and objective for E / L1 / L2 at 1 and 16 workers; `INTERLEAVE=1` for deterministic multi-worker search | 2.0, 2b, 2c |
| Relayout legality | `deenum/zoo_ms.py` (record), `ms_check.py` (closed form), `ms_synth.py` (synthetic views) | whether `movement_supported` has a closed form over view digits | 2d |
| Relayout pricing profile | `deenum/zoo_cprof.py` | where `_cd_parent_relayouts` spends its time | 2d |
| SA as a generic backend | `sa_ilp/sa_solver.py`, `cpsat_backend.py`, `problems.py`, `run_probe.py` | whether a `pywraplp`-shaped interface can drive SCIP, CP-SAT and simulated annealing from one model builder; SA's gap on generic ILPs | 3a |
| LX layout with 2-D no-overlap | `sa_ilp/lx_layout.py` | SA with a scored `AddNoOverlap2D` against CP-SAT on the real layout model from captures, with an independent feasibility check | 3a |
| Brute-force oracle (planned) | `sa_ilp/` (to add) | an exact `BruteForce` backend behind the same interface, cross-checked against CP-SAT and SA on small ILPs and tiny LX layouts; the oracle for strategy unit tests | 3a |

Typical runs:

```bash
bash probes/cpsat-graph/deenum/run_profile.sh                 # profile + captures (card)
PYTHONPATH=$TORCH_SPYRE python3 probes/cpsat-graph/deenum/lazy_probe.py \
    probes/cpsat-graph/deenum/runs/caps_default out.jsonl 1,16 30  # replay (no card)
python3 probes/cpsat-graph/deenum/ms_check.py ms.jsonl         # relayout closed form
python3 probes/cpsat-graph/sa_ilp/run_probe.py ilp.jsonl 10 60 # SA vs SCIP / CP-SAT
```
