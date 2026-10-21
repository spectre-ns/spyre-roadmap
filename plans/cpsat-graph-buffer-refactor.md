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

Time is in **ticks**: op indices in execution order (the values in `uses`). A lifetime
is `[start_time, end_time)`.

Goal: model the solve as a graph.

- **`ResourceNode`** — the main object of the graph: everything else (edges, relayout
  layouts, decisions) is reached from a node, and the graph hands out nodes
  (`get_nodes()`). A node is every resource an op needs: op outputs and graph inputs. A
  relayout is not a node: it is a second layout of the producer's output, live from the
  first read it serves to the last, and realizing the plan inserts the copy. A node holds what the op is
  (size, lifetime, its candidate menu, …) and its decisions (`assignment`, `resident`,
  `offset`), each fixed ahead of time or `None` for the solver to find. How each
  decision is enumerated (`Fixed`, `Table`, `Lazy` or `Pruned`) is an *enumeration*
  held by the strategy, which encodes the graph; see *Design rules*.
- **`DependencyEdge`** — every data dependency between two nodes (reads, in-place
  reuse; in Phase 1 only between nodes that both choose a division, as today). It holds *relations* over the two ends' decisions: the
  per-core view match (`ViewRelation`) and, on a read, the relayouts that can bridge a
  mismatch (`RelayoutRelation`). Edges are shared: the same edge object sits in its
  source's `outputs` and its destination's `inputs`, so either end can walk it.
- **`ExecutionEdge`** — links each op output (a node with a `produced_at` tick) to its
  chronological neighbours (`node.prev` / `node.next`). Coarse tiling is handled
  here: whether the two ops run the same loop nest (`NestRelation`), whether they join
  one tiling group, and where a group breaks.
- **`SpyreOptimizationGraph`** — a collection of nodes (which own the edges), plus the
  global context (alignment, LX capacity). `graph.solve(model, cost_expr)` fills in
  every open decision and returns the op-output nodes in execution order, which is
  the execution plan.

Three phases, each stage its own PR that leaves the full suite green:

- **Phase 1 — re-representation (CP-SAT internals).** The graph carries the
  allocator's final data and every relation uses its *table* encoding, which is
  today's model. Each stage except Stage 7 is proven by an identical CP model (see
  *Verification*).
  The allocator's three classes stay the external contract.
- **Phase 2 — constraint encodings (CP-SAT internals).** Each axis gains a `Lazy`
  enumeration and each relation a constraint encoding over per-axis split / per-level tile
  vars, selected through the strategy factory and checked against the table encoding.
  Then menus stop being materialized where the config asks for it.
- **Phase 3 — one problem definition.** `graph.set_solver(...)`, then the allocator
  builds the graph, the other solvers move onto it, and the allocator subclasses
  collapse into one allocator whose behaviour comes from the backend and the
  enumerations the factory picks.

Work happens in the torch-spyre worktree `.claude/worktrees/cpsat-graph-model` (branch
`cpsat-graph-model`, off `main` 82edd8f6). The owner commits each stage.
`scratchpad/buffer_graph.py` is already drafted there (`CoreDivisionBoundBuffer`,
`BufferGraph`, `build_buffer_graph`, `SliceEdge`/`InPlaceEdge`/`CopyEdge`,
`Timeline.runs`). Stages 1–2 rename and reshape it into the model below.

## Design rules

1. **Relations, not tables.** Every relationship is an object over the two ends'
   decisions. Its encoding follows the enumerations at its two ends (see rule 5): for
   views and relayouts a table over menu indices when both are `Table` (Phase 1, today's
   model), constraints over per-axis vars otherwise (Phase 2). Nests keep today's
   per-node `loop_id` until 2b. Outside the strategy's own encodings (`Table` index
   vars and element ties, `loop_id`) and, until 2a, `cpsat_expr_printer`, nothing reads
   menu indices; queries take `Assignment`s.
2. **Phase 1 derives nothing the allocator decided.** Match tables (including the
   loop-carry intersection and input-clone overrides, `allocator.py:4160-4240`),
   capped relayouts and the copies built from them, read counts (`read_count`:
   `len(uses)` less the producing write, `plan_solver.py:155`; extern and
   placement-only readers included), lifetimes and their overrides, and exclusions
   (`MemoryPlanSolver.excluded(buffer)`: the allocator's `residency_reason`, else the
   capacity check) all come in as data. Moving any derivation out of the allocator is
   Phase 2/3 work with its own agreement test.
3. **`None` means an open decision.** A node has three decisions: `assignment` (core
   division and tiling), `resident` and `offset`. Each is open (`None`; the solver
   decides) or set, and the factory realizes a set decision as a `Fixed` enumeration.
   Decisions are set or pinned in three places:
   - **At construction, by `add_node`:** a node with an empty menu gets
     `assignment = Assignment.COMMITTED` (committed upstream by work division), which
     becomes `Fixed(COMMITTED)` and posts nothing.
   - **By the caller** (Stage 8): pinned `assignment`, `resident` or `offset`.
   - **At finalization, by the strategy's op constraints.**
     `strategy.apply_op_constraints(node, model)` applies the restrictions for the
     node's kind of op (its `residency_reason`: an op not allowed in LX, a mutation
     target, an empty tensor, min footprint over LX capacity, and so on). The default
     strategy pins `resident=False` in the model and records the reason; the field
     stays `None` and `solve` fills it. Another strategy may post a restriction as a
     constraint instead (for example "a split reduction axis ⇒ not resident"), which is
     why this lives in the strategy and not at construction. In Phase 1 the reasons
     are the allocator's, as given (rule 2); in Phase 3 they come from the op.

   Decision fields use `None` only for "open". The structural `Optional`s are an edge's
   `view` (`None`: an IN_PLACE child fixed upstream, ungated), its `relayout` (`None`:
   no relayout can serve the read) and `residency_reason` (`None`: no restriction).
   `Fixed` is an enumeration on a decision axis, not a buffer type: it posts constants
   (or nothing) where an open axis posts vars.
4. **Build structure, then finalize once, through a strategy.** `add_node` /
   `add_child` / `add_successor` wire the graph and touch no model; nodes and edges are
   pure data. `solve(model, cost_expr, strategy)` calls
   `SpyreOptimizationGraph._finalize(model, strategy)`, which runs the `Strategy`'s
   `apply_*` steps over the graph in a fixed order. All encoding, and every CP var and
   literal, lives in the strategy.
5. **Enumeration is polymorphic, per decision axis.** Nodes do not carry
   enumerations; the `Strategy` does. For each node and each decision axis (core
   division, coarse tiling, residency, offset; from 2d, relayout choice per READ edge)
   it holds
   an `Enumeration`, data plus `holds(value)` (a query, no model), built from the
   node's fields, and it realizes that enumeration in the model:
   - `Fixed(value)` — known before the solve; posts constants. `Fixed(COMMITTED)` is a
     placement-only buffer: committed upstream by work division, posts nothing.
   - `Table(candidates)` — explicit candidates: an index var plus element ties
     (today's menu). Core division and tiling share one `Table` (today's joint menu)
     until tiling goes lazy, then they separate.
   - `Lazy(space)` — per-axis vars plus legality as constraints, no candidates.
   - `Pruned(space, keep)` — generate, keep a shortlist, realize it as a `Table`.
     Today's relayout group capping (`_cap_relayout_groups`) becomes a `Pruned`
     relayout enumeration in 2d; in Phase 1 its output arrives as data on
     `RelayoutRelation` (rule 2).

   The `EnumerationFactory` validates the config and builds a `Strategy` for a graph,
   choosing each node's enumerations from its fields (for example, `Lazy` tiling
   requires tiling to be its own axis). The Phase 1 default reproduces today: `Table`
   over the node's menu, `Fixed(value)` for a set decision (so `Fixed(COMMITTED)` for a
   placement-only node), relayouts as data.

   Every enumeration except `Fixed(COMMITTED)` exposes the same per-axis vars:
   `split(key)` and a tile count per `TileKey` (a `Table` makes them by element, as
   today; per-level `level(k)` arrive in 2b). `Fixed(COMMITTED)` exposes none, and its
   edges are ungated. So a relation has one general encoding, *derived-view equality*:
   each end's per-device-dim factor and stride and its core count, derived from its
   split vars (products and a reified order, so not linear; see 2c). The pair table is
   an optimization when both ends are `Table`, and required wherever the derived
   relation is inexact:

   | src \ dst | `Fixed` | `Table` / `Pruned` | `Lazy` |
   |---|---|---|---|
   | `Fixed` | checked at build | allowed indices | derived-view equality to constants |
   | `Table` / `Pruned` | allowed indices | pair table (today) | derived-view equality |
   | `Lazy` | derived-view equality to constants | derived-view equality | derived-view equality |

   Every cell is posted under the edge's enforcement literal (`in_buffer` for a view
   gate, the match or served literal when relayouts serve the read, the merge literal
   for IN_PLACE), as `_gate_divisions` does today. "Checked at build": a compatible
   fixed pair posts nothing, an incompatible one forces that literal false, as an empty
   table does today.

   **Composing enumerations across axes.** The strategy realizes every enumeration onto
   the same per-axis vars, so composing them is a conjunction of constraints over those
   vars:
   - `Fixed` with anything: the fixed axis's vars are constants.
   - `Table` and `Lazy` on independent axes: each posts its own legality; nothing joins
     them.
   - `Table` and `Lazy` on dependent axes (a menu item is legal only for some values
     of the other axis, e.g. a division whose span limit depends on the tiling): the
     menu carries, per item, its condition on the other axis. It is posted either as
     one per-node table over (menu index, the lazy vars the condition reads), or as
     `division == i ⇒ condition_i` per item. Both stay per node, never per pair. For
     division × tiling that is |menu| × the tiling options: at most 64 for a `Table`
     list (`_MAX_TILE_OPTIONS`); a `Lazy` tiling space has no cap, so the factory's
     size bound applies to the product.
   - If an item's condition can't be stated without enumerating the lazy axis in full
     (a table past a size bound, or no closed form), the factory rejects that
     composition and keeps both axes in one joint `Table` (today's menu) or makes both
     `Lazy`. Each enumeration declares the other axes it reads, and that declaration
     is what the factory validates.

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
    chosen: Optional[RelayoutRead] = None    # filled by solve(); None = read
                                             # directly

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
class RelayoutLayout:
    """The node's output re-laid-out from its native view: a second copy in LX with
    its own decisions, live from the first read it serves to the last (today's
    copy lifetime, `build_relayout_copy`). Phase 1 copies its fields from the
    allocator's `RelayoutCopyBuffer` (rule 2)."""
    name: str                                # today's copy name (var names, is_lx_*)
    group: int                               # destination view (today's group)
    size: int                                # alignment units
    read_count: int
    division: CoreDivision                   # the destination's one division
    candidates: tuple[RelayoutCandidate, ...]  # priced source divisions
    start_time: int                          # [first consumer tick, last + 1): inside
    end_time: int                            # the producer's lifetime
    resident: Optional[bool]                 # None = open
    offset: Optional[int]                    # None = open; None when not resident

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
    residency_reason: Optional[str]          # Phase 1: excluded(buffer); Phase 3:
                                             # from the op; apply_op_constraints
    # decisions (None = open; solve() fills them)
    assignment: Optional[Assignment]         # COMMITTED when the menu is empty
    resident: Optional[bool]
    offset: Optional[int]                    # stays None when not resident
    # structure
    produced_at: Optional[int]               # tick of the producing write (uses[0]);
                                             # None for a graph input. Not start_time,
                                             # which a start override can move earlier
    prev: Optional[ExecutionEdge]            # set by add_successor; None for a graph
    next: Optional[ExecutionEdge]            # input and at the chain ends
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
    first: Optional[ResourceNode]            # start of the execution chain

    def add_node(
        self, source: LifetimeBoundBuffer | torch._inductor.ir.Operation
    ) -> ResourceNode
        # the graph fills the node's fields from either source: an allocator buffer
        # (or its CoreDivisionBuffer subclass) in Phase 1, an IR op from Phase 3
        # (returns the existing node if that op was added). A RelayoutCopyBuffer
        # becomes a RelayoutLayout on its relayout_parent's node, which is returned.
    def get_nodes(self) -> Iterator[ResourceNode]  # in add_node order; edges are
                                                   # reached through node.inputs /
                                                   # node.outputs, never listed
    def solve(
        self,
        model: cp_model.CpModel,
        cost_expr: Optional[sympy.Expr] = None,  # None or unlowerable: fallback
        strategy: Optional[Strategy] = None,     # None: the factory's default
    ) -> list[ResourceNode]                   # op outputs in execution order;
                                              # raises SolveError as _run does
    def _finalize(self, model, strategy) -> None  # the strategy's apply_* steps,
                                                  # in today's order
```

`Assignment` and the three relations (`ViewRelation`, `RelayoutRelation`,
`NestRelation`) are data the graph holds. The `Enumeration`s are held by the
`Strategy`, which the `EnumerationFactory` builds from node fields and config. The
strategy encodes everything and keeps all CP state: a `NodeVars` per node and per
relayout layout, and the served literals per READ edge. Each stage below introduces
only the part it needs (Stage 1: data and queries; Stage 5:
`Strategy.apply_core_division` and `NodeVars`; Stage 6: the other `apply_*` steps;
Phase 2: `Lazy` / `Pruned` and the mixed-pair encodings).

Notes:

- **Graph inputs are nodes** with `produced_at = None`: no producing op and not on
  the execution chain (their first use is another op's read tick). They have a menu
  (today's clone menu), which the strategy enumerates, and residency like any node.
- **Relayouts are layouts, not nodes.** A `RelayoutLayout` is the producer's output
  in a destination view, with its own residency and offset, live from the first read
  it serves to the last. Realizing the plan inserts the copy op (the shuffle directly
  before its first consumer) only for resident layouts. Copies do not break tiling
  runs today, and with no copy node on the chain that stays exact.
- **Copy order** is every relayout layout sorted by (producer name, `group`), after all
  nodes: today's `sorted(groups.items())` (`allocator.py:4444`), with copies appended
  after every other buffer. Layouts live on their nodes, so the strategy has one helper
  that yields them in copy order; every copy-dependent step uses it.
- `eq=False` on nodes and edges: they reference each other, so they compare by
  identity and get a short custom `__repr__` (name, or `src->dst:kind`).
- A pair of nodes can share more than one edge (an in-place child usually reads its
  parent too). The solver iterates edges by kind, so there is no `isinstance` dispatch.
- `add_child` puts the one edge object on both ends and nowhere else; the graph keeps
  no edge list. `add_successor` links one chain, starting at `graph.first`, and refuses
  to relink a node that already has a neighbour; `_finalize` asserts every node with a
  `produced_at` is on it. The chain ends have no edge (`prev` / `next` is `None`), which the
  coarse-tiling walk treats like a non-joinable edge. `contiguous` is derived as today (consecutive `produced_at` ticks).
- A missing match entry is an **empty** `Pairs`, never `None`: today
  `.get(parent, [])` (`ilp_solver_ortools.py:410`) gives an empty table, which
  `_gate_divisions` turns into its enforcement literal forced false: the residency
  literal (or `match_*` when a relayout serves the edge), the merge, and not-apart in
  cuts, so that pair is forced apart. A missing key is common (any producer with a
  `residency_reason`).
- `solve(model, cost_expr, strategy)` is the only public entry point after building:
  it runs `_finalize(model, strategy)`, then the **objective ladder**, then
  extraction, and fills in every open decision. The ladder is one `cost_expr` solve
  (lowered as `_minimize_cost_expr` does today) or, when `cost_expr` is `None` or fails
  to lower, today's fallback: the copy pin, then the residency, cut, parallelism, shape
  and tile-count stages (`ilp_solver_ortools.py:1040-1130`). Each solve is a *ladder
  stage*.
- Coarse tiling splits across the two edge types the way the data does: group
  membership is decided on `ExecutionEdge`s, and whether a tiled value must be cut to
  HBM by checking its READ edges against those groups.
- Out of scope: solver-chosen op order. Execution order is an input (`add_successor`).

### Finalization order

`_finalize(model, strategy)` runs the strategy step by step across all nodes, not node
by node, because today's model is built step by step (`_run`,
`ilp_solver_ortools.py:945-1000`). Steps that today post across all buffers take the
graph:

```python
def _finalize(self, model, strategy):
    strategy.apply_core_division(self, model)
    strategy.apply_inplace(self, model)
    strategy.apply_placement(self, model)
    for node in self.get_nodes(): strategy.apply_op_constraints(node, model)
    for node in self.get_nodes(): strategy.apply_reads(node, model)
    strategy.apply_relayout(self, model)
    strategy.apply_coarse_tiling(self, model)
    strategy.apply_cloning(self, model)
```

Each step in emission order:

1. **`apply_core_division(graph)`**: each node's `NodeVars` in `add_node` order, as
   today's wrappers make them: `in_buffer`, `off`, a `merge_*` literal per IN_PLACE
   input, then `div`, `eff_size`, `core_cost`, `occ` (cores), split and tile-count vars
   each with its element, then the `eff_size` / `cores` / `core_cost` elements
   (`:216-394`); cached constants for a one-candidate menu, and only `in_buffer`, `off`
   and merges for `COMMITTED`. Then each relayout layout's vars, in copy order.
2. **`apply_inplace(graph)`**: every merge row first, per destination in `add_node`
   order and per IN_PLACE input in `in_place_parents` order: offset equality, merge ⇒
   parent resident, merge ⇒ child resident, then, when the child chooses a division,
   `eff_size` equality and the view gate. Then at-most-one over each node's incoming
   merges, then over each node's outgoing merges (first-seen order)
   (`:1200-1240`).
3. **`apply_placement(graph)`**: capacity (`offset + eff_size <= M` under the
   residency literal) per node, then per layout; then the 2-D no-overlap over both,
   with the incoming-merge tick shortening.
4. **`apply_op_constraints(node)`**, in `add_node` order: the restrictions for the
   node's kind of op. In the default strategy these are today's exclusion pins
   (`resident=False` with the reason), posted where they are today, after the
   no-overlap and before any read gate (`_add_core_division`); a layout the capacity
   check excludes is pinned after the nodes.
5. **`apply_reads(node)`**, over the node's READ `outputs` (parent order, then child
   order, then the child's `parents` order): per edge, the served literals (one per
   layout that can serve the read: "this read goes through that layout") in sorted
   relayout-group order (`:441`) and the view gate, which is "match or served",
   interleaved as today's `constrain_residency` does.
6. **`apply_relayout(graph)`**, per layout in copy order: if the producer chooses no
   division, pin the layout out; otherwise layout resident ⇒ producer resident,
   `bool_or(division_is(i) for i in sorted(priced))` enforced by the layout's literal,
   then `bool_or(serves)` enforced by it, or the pin when no read can use it
   (`_constrain_relayout_copies`).
7. **`apply_coarse_tiling(graph)`**: `loop_id`/`tiled` per node with a menu and per
   layout (today's one-candidate copies), then execution edges along the chain
   (`same_nest`, `joined`, group step), then cuts, then tile counts.
8. **`apply_cloning(graph)`**: input clones. A no-op in Phase 1, where a clone's
   matches are ordinary READ-edge tables; in Phase 2c it posts the clone's relation
   derived from its consumers.

The fallback copy pin (`:1044-1048`) is not here: it depends on whether `cost_expr`
lowered, so it stays in the ladder inside `solve`. Stage 0's proto diff is the check
on every claim in this list.

## Phase 1 — re-representation

**Sequencing.** Two open torch-spyre PRs change the model Phase 1 freezes as
"today's": #5246 (variable buffer lifetimes: lifetime ends, nest starts / ends and
tiling groups become solve decisions) and #5587 (a cost expression under coarse
tiling, draft). Land them before Stage 0 if possible. Otherwise the stage in flight
rebases when one lands, re-records the baseline, and the finalization steps follow the
new `main` (with #5246, lifetime ends become a decision the strategy posts, and
`start_time` / `end_time` stay the fixed bounds).

### Stage 0. Equivalence harness and baseline

- Add `tests/inductor/cpsat_equivalence.py`. It solves a buffer set
  deterministically: 1 worker, `random_seed=0`, and never a wall-clock limit, because
  the default 60 s limit (`config.cpsat_time_limit_seconds`) can return a
  time-dependent incumbent. With no limit (`time_limit_seconds=0`) every ladder stage
  must end OPTIMAL; under a fixed `max_deterministic_time` FEASIBLE is accepted, and
  status, objectives and decisions are compared exactly.
- It wraps `_solve_and_record` to capture every ladder stage's objective
  (`last_solve_stats` keeps only the last), and dumps per ladder stage: the model proto
  (`model.export_to_file(...)` or `str(model.proto)`; OR-Tools 9.15's pybind proto has
  no `SerializeToString`), residency, chosen division, offset and chosen relayouts. A
  **proto diff** compares all of these with the baseline, ladder stage by ladder stage;
  "proto diff empty" means identical.
- Corpus: `synthetic_cooptimization_graphs`, the `cooptimization_capture_loader`
  graphs, cases from `test_solver_relayout_decision` (copies, relayouts), the
  coarse-tiling tests, `plan_layout` (placement-only buffers) and solves with a
  `cost_expr`. Those graphs are small and mostly untiled with almost no execution
  chain, so the corpus also replays the 89 Phase 2.0 captures device-free through
  `CpSatLayoutSolver(...).plan_layout_and_core_divisions(cost_expr)` (not in git;
  regenerated with `run_profile.sh` when the base moves). It must contain a tiled solve
  with copies, copies of several parents (copy order), an execution chain,
  placement-only buffers, a `cost_expr` and a multi-parent in-place child.
- Record the baseline on the stage's parent commit (see *Verification*). No production
  code changes.

### Stage 1. Data model

- Rename the draft (`CoreDivisionBoundBuffer` → `ResourceNode`, `BufferGraph` →
  `SpyreOptimizationGraph`, `build_buffer_graph` → `build_graph`) and reshape it into
  the graph above. Drop `SliceEdge`/`InPlaceEdge`/`CopyEdge`/`Timeline`.
- The supporting types, data and queries only (no solver methods yet):

  ```python
  Pairs = tuple[tuple[int, int], ...]  # (src, dst) menu-index pairs (the draft's)
  SplitKey = sympy.Symbol              # CoreDivision.splits keys
  TileKey = tuple[int, bool]           # as the draft's sym_tile_counts keys

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
      offset: Enumeration              # open (apply_op_constraints may pin it). No
                                       # relayout axis until 2d.

  class Strategy:                      # encodes a graph; methods arrive in Stages 5-6
      enumerations: dict[ResourceNode, Enumerations]

  class EnumerationFactory:            # validates the config and builds strategies
      def strategy(self, graph: SpyreOptimizationGraph) -> Strategy
          # default: today's model, from each node's fields (menu -> Table, a set
          # decision -> Fixed(value), so COMMITTED -> Fixed(COMMITTED))

  @dataclass(frozen=True)
  class RelayoutRead:                  # one src layout this read can go through
      layout: RelayoutLayout
      candidates: tuple[RelayoutCandidate, ...]  # its priced (src, dst) divisions

  # Relations answer through their edge's two menus (an Assignment is unique in
  # its menu), so queries take Assignments while the tables hold indices.
  class ViewRelation:                  # both ends own the buffer with the same view
      table: Pairs                     # allocator's final pairs; empty = never alike
      def holds(self, src: Assignment, dst: Assignment) -> bool
  class RelayoutRelation:              # READ: src's relayout layouts that can serve
      reads: tuple[RelayoutRead, ...]  # a mismatch; allocator's capped candidates
      def bridges(self, src: Assignment, dst: Assignment) -> tuple[RelayoutRead, ...]
  class NestRelation:                  # both ops run the same non-empty loop nest:
      def joins(self, before: Assignment, after: Assignment) -> bool
                                       # equal, non-empty level_counts; no pair
                                       # table (Phase 1 keeps loop_id, Stage 4)
  ```
- The draft's `RelayoutRead.copy` becomes `layout` and its `pairs` goes. `spill_cost(node)`
  (from `read_count`, `first_use_is_read`, `boundary` and `size`) and the derived
  properties (`sym_is_lx`, `per_core_sizes`, `cores_used`, `shape_costs`, `tileable`)
  come from the draft but read `menu`, which is empty for a `COMMITTED` node;
  `sym_division` is a field. Drop the draft's `__post_init__` (the non-empty-menu
  assert, the `(CoreDivision(),)` stand-in and the automatic `division = 0`).
- Tests (`tests/inductor/test_buffer_graph.py`, device-free logic; the inductor
  conftest still opens the card), registered in a `tests/configs/` shard in the same
  PR (`make check-all-configs`): `add_node` order;
  `add_child` puts one edge object on both ends only, and two kinds can join one pair;
  relation queries (`holds`, `bridges`, `joins`) agree with their tables, and an empty
  table never holds; `add_successor` links both ends, derives `contiguous`, and refuses
  a second neighbour; `joinable` is false across a tick gap and at untileable ops;
  the chain ends have `prev` / `next` `None`; `tiling` is `None` while `assignment` is; `joins`
  agrees with `loop_id` equality ∧ `tiled[after]`.

### Stage 2. Builder

- `build_graph(buffers, alignment, capacity, settings)` makes the
  `SpyreOptimizationGraph`, calls `add_node(buffer)` for each allocator buffer in the
  allocator's order, then adds the edges through `add_child`. `add_node` does the
  conversion inside the graph: it fills the node's fields, with the buffer's
  `core_divisions` as the menu; a `LifetimeBoundBuffer` has none and gets
  `assignment = Assignment.COMMITTED`. A `RelayoutCopyBuffer` becomes a
  `RelayoutLayout` on its `relayout_parent`'s node (not a node), with its name, group,
  size, read count, division, candidates and lifetime. The allocator classes are read
  only there and in the edge wiring below; past `add_node` the node never refers back
  to its buffer. `settings` is the interim `SolveSettings` (Stage 8). The final data
  copied:
  - READ edges only between nodes that both have a menu, i.e. neither is
    `COMMITTED` (today's no-op `constrain_residency` on placement-only wrappers).
    `view.table` is `cd_parent_matches.get(src, ())`; `relayout` is
    `cd_parent_relayouts.get(src, ())` filtered to groups with a copy in this solve and
    grouped into one `RelayoutRead` per layout.
  - IN_PLACE edges with `view=None` when the child's division was fixed upstream,
    otherwise the same table.
  - Graph inputs become nodes with `produced_at = None`.
  - `read_count`, lifetimes and overrides as given. A plain `LifetimeBoundBuffer` has no
    `boundary`; it maps to `Input` if `first_use_is_read`, else `Intermediate`, never
    `Output` (today's fallback, `ilp_solver_ortools.py:263-270`). Each node's
    `residency_reason` is `MemoryPlanSolver.excluded(buffer)`: the allocator's reason,
    else the capacity check (`record_exclusions`, `ilp_solver_ortools.py:630`), for
    `apply_op_constraints` to apply and Stage 8 to write back as `spill_reason`.
  - `produced_at = uses[0]` for op outputs (today's filter: has `uses`, not
    `first_use_is_read`, not a copy), `None` for graph inputs; then `add_successor`
    over those nodes in `produced_at` order, with `graph.first` the earliest.
- Tests, while the functions still exist: on every corpus graph IN_PLACE edges equal
  `in_place_parents`, READ edges equal `_get_children` restricted to pairs where both
  ends are in the solve and have a menu, and the layouts equal `_relayout_copies`.
  Stage 3 deletes those functions, and from then the proto diff covers the edges.

- Where each `CoreDivisionBuffer` field goes. In Phase 1 a node is a
  `CoreDivisionBuffer` redistributed: what it is (including its menu) in its fields,
  how it relates to others on edges, what was decided in its decision fields; how its
  choices are enumerated belongs to the strategy.

  | `CoreDivisionBuffer` field | Phase 1 home |
  |---|---|
  | `name`, `size`, `first_use_is_read`, `boundary` | node fields |
  | `uses`, `lifetime_*_override` | `start_time` / `end_time` / `read_count`; `uses[0]` → `produced_at` |
  | `residency_reason` (or the capacity check) | `residency_reason` = `excluded(buffer)`, applied by `apply_op_constraints` |
  | `core_divisions` (with each `.tiling`) | `menu`; the default strategy's `Table` (division and tiling shared) |
  | `parents`, `cd_parent_matches`, `cd_parent_relayouts` | READ edges: `ViewRelation.table`, `RelayoutRelation` |
  | `in_place_parents` | IN_PLACE edges |
  | `RelayoutCopyBuffer` (whole buffer) | a `RelayoutLayout` on its parent's node, with its name, group, size, read count, division, candidates and lifetime |
  | `chosen_division`, `address`, `spill_reason`, `chosen_relayouts` | decisions and `chosen`, written back by Stage 8 |
  | `residency_edges`, `division_space`, `tile_reads` | not carried; Phase 2c / 2f / 2b |
  | `op_position` | replaced by `produced_at` and execution edges |
  | `paired_with`, `lx_view`, `lx_relayout_plans` | not read by CP-SAT; Phase 3b |

  A `LifetimeBoundBuffer` maps the same way with an empty menu, `assignment =
  COMMITTED` and `boundary = Input if first_use_is_read else Intermediate`.

### Stage 3. Solver reads relationships from the graph

- `_plan_layout_generic` builds the `SpyreOptimizationGraph` next to the existing
  wrappers.
- `_get_children` → `node.outputs` of kind READ; `_relayout_copies` → the nodes'
  `relayouts` in copy order (the group-key → wrapper map is built in that order); the
  `in_place_parents` loop in
  `_add_inplace_relaxation` → each node's IN_PLACE `inputs`.
- Delete `_get_children` and `_relayout_copies`. The wrappers still own the CP vars.
- Proto diff empty.

### Stage 4. Tiling groups on execution edges

- One per-edge function owns an execution edge's coarse-tiling encoding (it moves
  into `Strategy.apply_coarse_tiling` in Stage 6): `same_nest`
  and `joined` (today's `samenest_*` / `joined_*`) and the group step
  (`group[after] = group[before] + 1 - joined`). A non-joinable edge starts a new
  segment with group 0. `same_nest` keeps today's `loop_id` encoding, `tile_group_*`
  keeps its domain, and `loop_id`/`tiled` are still made first, for every buffer with
  a menu (copies included, last), in today's order.
- `_tiling_group_ids` becomes a walk along the execution chain from `graph.first`,
  starting a new segment at the first op and wherever an edge is not `joinable`; the segments are local to the coarse-tiling
  encoding (in `Strategy.apply_coarse_tiling` from Stage 6), not a graph method. Its
  `uses[0]` logic and `RelayoutCopyBuffer` check go away.
- `_cut_literals`: for each tiled node, consumers on its READ edges in the same segment
  get an `apart` literal gated on `edge.view`; a consumer outside the segment, or a
  graph output, makes the node unshareable. `_tile_count_terms` iterates nodes.
- Proto diff empty (coarse-tiling corpus with `auto_coarse_tiling` on).

### Stage 5. `Strategy.apply_core_division`: the strategy makes node vars

- Gives the Stage 1 `Strategy` its first method, the default strategy's
  `apply_core_division(graph, model)`. It makes and keeps a `NodeVars` per node, then
  per layout in copy order, in today's emission order (*Finalization order*, step 1):
  `in_buffer`, `offset`, merge literals, `division`, `eff_size`, `core_cost`, `cores`,
  split and tile-count vars. It dispatches on the node's division enumeration,
  replacing the variable-making half of both wrappers: `Table` posts today's index var,
  element ties and single-candidate constants; `Fixed(COMMITTED)` posts no split vars,
  which is today's placement-only model (`cores`, `core_cost` `None`, no gates).
- `NodeVars` carries `cpsat_expr_printer`'s contract until Phase 2a: `buffer`, a
  read-only view exposing exactly `core_divisions` (the `Table` candidates),
  `sym_core_divs` and `sym_division` (`cpsat_expr_printer.py:249`, `:840`,
  `:881-895`); `division`; and `division_is(i)`, cached and created on first call as
  today (`ilp_solver_ortools.py:396-403`), constants for a one-candidate menu. Nodes
  stay pure data. Var names stay as today.
- The wrappers hold their node's `NodeVars` and delegate to it, so no constraint code
  changes.
- Proto diff empty.

### Stage 6. The rest of the strategy; delete the wrappers

- The default strategy gains the remaining steps, with the table encodings of the
  three relations (today's `_gate_divisions` for views, the served literals for
  relayouts, the Stage 4 `same_nest` literal for nests):
  - `apply_inplace(graph)`: every merge row (per IN_PLACE input, using the step-1
    merge literal: offset equality, both-resident implications, then `eff_size`
    equality and the view gate when the child chooses a division and `view` is not
    `None`), then the at-most-ones.
  - `apply_placement(graph)`: capacity for nodes and layouts, then the 2-D no-overlap
    with tick shortening.
  - `apply_op_constraints(node)`: the exclusion pins, in today's position.
  - `apply_reads(node)`: per READ output, served literals and "match or served"; the
    strategy keeps the served literals by edge and tallies them per layout.
  - `apply_relayout(graph)`: today's `_constrain_relayout_copies`, per layout in copy
    order (*Finalization order*, step 6).
  - `apply_coarse_tiling(graph)`: takes over the Stage 4 function, cuts and tile
    counts.
  - `apply_cloning(graph)`: a no-op in Phase 1.
- `SpyreOptimizationGraph._finalize(model, strategy)` runs them in the order under
  *The model*.
- `free_copies` (presolve threshold) keeps today's rule: every relayout layout whose
  node chooses a division.
- The wrappers' other readers move onto `NodeVars` and the layouts' vars:
  `_minimize_cost_expr`'s `sym_map` / `buffer_map` (including `_division_of_*` and
  `is_lx_<copy>`), the ladder's term collection and fallback copy pin (layout terms
  last, in copy order), and `_extract` (justify units include layouts; `chosen` per
  READ edge).
- Delete `_LifetimeBufferWithCpVars`, `_CoreDivisionBufferWithCpVars`, `_wrap`,
  `_add_core_division`, `_add_inplace_relaxation`, `_constrain_relayout_copies`, every
  `isinstance(..., RelayoutCopyBuffer/CoreDivisionBuffer)` and every
  `getattr(..., "core_divisions")` inside the solve. In the same PR, point
  `test_cooptimization_types` (it imports `_LifetimeBufferWithCpVars`) at
  `buffer_graph.spill_cost`, and update every other reference to the deleted names
  (grep each).
- Proto diff empty. Run the device suites.

### Stage 7. Objective from open decisions only

- Parallelism, shape and tile-count terms come only from nodes whose division
  enumeration has more than one value (a `Table` of two or more; later `Lazy`), keyed on
  the strategy, not on `assignment is None`. Layouts contribute none. A ladder stage
  with only constant terms is skipped. A placement-only solve already runs only the
  residency stage (`ilp_solver_ortools.py:1077-1088`); what goes are the constant-term
  re-solves of joint solves whose division-carrying buffers all have one candidate,
  and the constant copy terms.
- Check: every ladder stage's objective equals the baseline up to a constant per
  ladder stage. Decisions may differ only on ties, because `_extract` reads whichever
  solve ran last and skipping ladder stages changes that.

### Stage 8. `graph.solve(model, cost_expr)`; honour pre-fixed decisions

- `graph.solve` becomes the public entry point with the model and cost expression
  `_run` takes today, plus the strategy (default: the factory's):
  `_finalize(model, strategy)`, the objective ladder (with the fallback copy
  pin when `cost_expr` is `None` or fails to lower), extraction. It returns the
  op-output nodes along the execution chain with every open decision filled
  (`assignment`, so the tiling; `resident`; `offset` when resident) and `chosen` set on
  READ edges (`None`: read directly). That list is the execution plan.
- It builds the CP-SAT solver as `_run` does (time limit, workers from deterministic
  mode or the CPU count, `random_seed=0`, presolve off past
  `config.lx_solver_relayout_presolve_max_copies` free copies) and solves every
  ladder stage through `_solve_and_record`, so `last_solve_stats`, the cost dump and
  the Stage 0 hook keep working. Per-instance settings (time limit, `bottom_justify`,
  the recorder) come from the `SolveSettings` given to `build_graph`; 3a moves them
  into the `CpSat` backend.
- Failure keeps `_run`'s behaviour: a ladder stage that ends without OPTIMAL or
  FEASIBLE raises `SolveError` naming the stage; a time-limited FEASIBLE incumbent is
  accepted; a lowering error re-raises when `_cpsat_warn_on_cost_expr` is off. Node
  decisions and `chosen` are written only after extraction succeeds.
- Pre-fixed decisions are `Fixed` enumerations: a concrete `Fixed` division posts a
  constant index (its edges keep their gates, with that side constant);
  `Fixed(COMMITTED)` posts nothing, as above; `Fixed` residency / offset pin their
  vars. Bottom-justify is skipped when any
  offset was pinned.
- `_plan_layout_generic` makes the model, calls `graph.solve(model, cost_expr)` and
  writes back: `address` in bytes, `spill_reason`, `chosen_division` only where the
  caller enumerated divisions, `chosen_relayouts` scaled.
- Tests: pre-fixed decisions are honoured; every open decision comes back filled; the
  returned order matches `produced_at`; an infeasible pin raises `SolveError` and
  leaves every decision `None`.
- Device-suite check: CP-SAT's joined runs *refine* `derive_tiling_groups` on the same
  choices, mismatches logged. Equality is not guaranteed today: CP-SAT approximates
  tile-by-tile reads through the view tables and never consults `tile_misread` (a
  pre-existing gap; Phase 2b closes it).

### Stage 9. Cleanup

- Module docstring around nodes, relations, execution edges and `graph.solve`.
- Comments and docstrings still citing the old classes and methods (for example
  `plan_solver.py:152`, `sa_cooptimizer.py:1295`, `allocator.py:2696`,
  `cpsat_expr_printer.py:809`; grep for each deleted name). Code and test references
  moved in Stage 6.

Kept unchanged through Phase 1: `_gate_divisions`, `_PlacementUnit`, `_justify`,
`_solve_and_record`, the capability flags, `__init__`, and the public `plan_layout` /
`plan_layout_and_core_divisions`.

### Verification (every Phase 1 stage)

Baseline (`main` 82edd8f6): **1121 passed, 5 skipped, 11 xfailed** over the unit suites
below. Every baseline (harness, unit, device) is re-recorded on each stage's parent
commit and again after any rebase, so upstream changes drop out of the diff. Stage 7
has its own check, and later stages compare against the baseline recorded after it.

- Unit suites, serial (conftest opens the card; xdist workers collide), on a frozen
  `git archive` of the change vs the frozen base: test_scratchpad_solver,
  test_solver_relayout_decision, test_cooptimization_types, test_cost_expr_dump,
  test_restickify_cost_model, test_sa_cooptimizer, test_op_features,
  test_coarse_tiling, test_scratchpad_patterns, test_perm_layout_solver, plus
  test_buffer_graph from Stage 1. No new failures or skips vs base; new tests pass.
- Stage 0 harness: proto diff empty per ladder stage, except Stage 7 as noted.
- After Stages 3, 4, 6, 7 and 8: the device suites that compile through CP-SAT —
  test_scratchpad_use, test_work_division, test_solver_relayout_e2e,
  test_solver_auto_coarse_tiling, test_lx_planning_suite — base vs change, same serial
  setup; after Stages 6 and 8 also the 18 `test_inductor_ops_lx_*` shards. A pass/fail
  difference counts only if it reproduces at 1 worker with no time limit.
- After Stages 2, 5 and 8: replay the 2.0 zoo captures with `timers.py` on base and
  change, in one process and interleaved; flag planning-time growth beyond run-to-run
  noise.
- `pre-commit run --files <changed>` (ruff, mypy).

## Phase 2 — constraint encodings (outline; own plan after Phase 1)

Each stage adds an enumeration or encoding, selectable through the
`EnumerationFactory` from settings declared in `torch_spyre/_inductor/config.py` (so
they enter the FX graph cache key; default `Table`; names and any default flip belong
to Phase 2's own plan), and an **agreement test** against the table encoding:
- a relation: the same feasible (src, dst) assignments;
- a node axis: the `Lazy` space equals the menu, per node and tiling. Same optimum
  alone does not certify legality: the probe's L2, a relaxation, matched every proven
  optimum;
- a solve: then the same optimum.

Relation agreement needs each read's geometry, so it runs on device graphs or on
captures extended with index maps; 2a and the table-vs-table checks can use the
existing corpus. Mixed enumerations on one graph (for example `Lazy` on tiled
ops, `Table` elsewhere) use the per-axis encodings from rule 5. What is not linear
is called out; it needs `AddMultiplicationEquality` / `AddDivisionEquality` or table
constraints over small per-buffer domains, never per-pair tables.

- **2.0. Probe: is lazy faster than eager? (go / no-go for the rest of Phase 2.)**
  Phase 1 does not depend on it, so it can run alongside. The code is in
  `probes/cpsat-graph/deenum/` (see `probes/cpsat-graph/README.md`): phase timers and
  solver-input capture during a compile, then a device-free replay of the captures:
  1. **Where eager time goes today.** On the zoo (32 graphs from
     `probes/cpsat-graph/zoo.py`) and Granite 4.0-micro generation,
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
  calls) to produce 33K pairs; menus are re-enumerated per tiling. The profile
  overstates the match tables: `timers.py` wraps `same_partition` in a counting
  wrapper that costs ~0.95 µs per call (measured), ~59 s of the zoo-tiling 138.6 s
  and ~12 s of Granite tiling's 36.1 s. Net of it, zoo tiling plans in ~233 s.

  *Is the per-axis relation exact?* The per-device-dim `(split, stride)` + core-count
  key reproduces every pair table (385/385 zoo in both configs, 200/200 Granite
  default, 100/100 Granite tiling). The per-axis rule (symbol → device dim, iteration
  order, contiguous dim), plus the same rule over tile splits for tile keys,
  reproduces the allocator's **final** table on 382/382 zoo edges in both configs
  (tiled ones included) and on 192/200 Granite default / 94/100 Granite tiling edges
  (irregular owner formulas, and 2 tile rules that could not be inferred). The rule
  was inferred from candidate keys, not built from index maps, and the 37 input-clone
  edges kept pair tables. Tile keys follow two conventions: the `tile_splits` order,
  and a side whose tiling slices none of the buffer reads it whole.

  *Solver cost.* One reduced model (residency, footprint, capacity, 2-D no-overlap,
  match gates, one weighted objective; no cost expression, ladder, relayouts,
  in-place, nests or cuts) built three ways: E (menu index + pair tables, today), L1
  (per-axis split and tile vars, lazy gates, per-node menu legality) and L2 (no menu:
  per-axis domains, core bound, one reduction split; a relaxation). 30 s limit, seed
  0. It is harder than production: E hit the limit on 11 (16 workers) and 14
  (1 worker) of 32 zoo default replays, and on 12 and 17 under tiling. Proven optima
  agree: 0 objective mismatches on every pair where both arms reached OPTIMAL, in
  every config. (A first tiling run showed lazy arms beating E's proven optimum; the
  cause was a probe bug: the fallback pair table for edges without view keys ignored
  tile counts. Fixed, and all tiling L1/L2 solves re-run in a later process, so the
  tiling rows below pair them with E solves from the first run.) Time to OPTIMAL vs E,
  pairs where both proved optimality and E took ≥ 0.2 s; time-limited (the slowest)
  solves are not in the ratios:

  | config | workers | n | L1 median (range) | L2 median (range) |
  |---|---|---|---|---|
  | zoo, default | 16 | 8 | 1.02× (0.73–1.41) | 0.96× (0.77–1.28) |
  | zoo, default | 1 | 7 | 1.44× (0.30–2.55) | 1.36× (0.21–3.87) |
  | zoo, default | 16, interleaved (repeats a / b) | 8 | 1.11× / 1.07× (0.67–2.30) | — |
  | zoo, tiling (unpaired) | 16 | 10 | 0.56× (0.38–1.58) | 0.50× (0.10–1.52) |
  | zoo, tiling (unpaired) | 1 | 5 | 0.80× (0.27–2.79) | 0.09× (0.08–0.57) |
  | Granite, default and tiling | 1 and 16 | 0–2 | too few solves ≥ 0.2 s | |

  Lazy model build is slower in Python: 0.38 s vs 0.12 s over all 32 zoo default
  solves, 4.0 s (L1) and 2.4 s (L2) vs 0.3 s under tiling.

  *Eager + hash join* (the cheap Python fix, `DEENUM_HASHJOIN=1`): exact on 385/385
  edges in both configs. Zoo tiling planning 292 → 179 s (match tables 138.6 →
  23.2 s; ~233 → 179 s net of the counting overhead); default 3.0 → 2.5 s of match
  tables (planning measured 165 s against 152 s, solve variance). Granite produced no
  usable numbers (the default run hit the compile cache, and the tiling run hits a
  pre-existing `Cannot tile buf0 … read-distance limit` failure on `main`).

  *Deterministic multi-worker search* (`interleave_search`, 16 workers, two repeats):
  not reproducible with a wall-clock limit. 33/36 proven optima repeated the same
  solution, but only 1/28 time-limited solves did, and deterministic time never
  matched. Next: a `max_deterministic_time` budget instead of seconds.

  *Expected improvement* (relayout pricing and cost-model prediction stay until
  2a/2d; solve scaled by the L1 medians above, Granite unscaled; L1 build included):

  | config | today | eager + hash join | views lazy, menus kept (L1) | + menus gone (2f; projection) |
  |---|---|---|---|---|
  | zoo, tiling | 292 s (~233 s net) | 179 s | ~145 s | ~78 s |
  | Granite, tiling | 82 s | not measured | ~47 s | ~21 s |
  | zoo, default, 16 workers | 152 s | 165 s | ~151 s | — |
  | zoo, default, 1-worker solve | 152 s | — | ~170 s | — |
  | Granite, default | 171 s | — | ~169 s | — |

  The default rows differ by less than run-to-run variance (the hash-join run's solve
  alone took 56 s against 45 s). The ~78 s column assumes menu construction disappears,
  which needs `Lazy` division (2f) and is not measured.

  **Verdict: provisional pass for compiles with coarse tiling on, not for default
  compiles.** Conditions, as of the follow-up runs:
  1. **Tile relation — exact** on every zoo tiling edge and 94/100 Granite tiling
     edges. The lazy solve looked faster under tiling at both worker counts, but on
     unpaired runs of the reduced model and only where both arms proved optimality:
     re-run all three arms in one process, with time-limited solves reported, before
     Phase 2 relies on it.
  2. **1-worker regression — open for the default config** (1.44×; 1.07–1.11× with
     interleaved 16-worker search, which is not yet reproducible).
  3. **Hash join — done.** Lazy views with menus kept are ~1.2× faster than eager +
     hash join on zoo tiling (~145 vs 179 s); ~2.3× needs the menus gone too. Menus
     (63 s, re-enumerated per tiling) are then the largest eager cost; a per-tiling
     menu cache is the cheap alternative and has not been measured.
  4. **Relayout pricing — `movement_supported` has a verified closed form** (2d); the
     other per-pair gates and the price remain.
  5. **Nests — open.** No `loop_id` / `same_nest` / `joined` / group / cut encoding was
     prototyped or timed.

- **2a. Cost expression over per-axis vars.** Today `cpsat_expr_printer` tabulates
  over the menu: `_division_table` (`:309`), `RelayoutCharge` keyed on `sym_division`
  (`:822`), `log2_` by element on the division, `AppliedUndef` menu tables (`:854`),
  and `extract_op_features` reads `candidate_work_slices`. Tabulate over per-axis
  tuples (`AddAllowedAssignments`, small at ≤ 32 cores) instead. The cost builders
  change too: `RelayoutCharge` is keyed on the division symbol, and a `Lazy` node has
  no division index. Prerequisite for the rest; agreement: identical lowered values
  at every menu point.
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
  - **Placement.** Which device dim a split lands on, and with what factor, is a
    per-(dep, symbol, split value) table (`_per_core_view_from_prep`,
    `pass_utils.py:3506`). A direct stride lookup gives factor = split, linear for a
    single-symbol dim; the multi-stick rescue and `decompose_fused_split_view` give
    other factors.
  - **Core → slot.** `core_to_slice_mapping` (`core_mapping.py:541`) gives each split
    dim the coordinate `(core_id // stride) % split`, with `stride` the product of the
    splits before it in the op's iteration order. Equal strides are a chain of
    `AddMultiplicationEquality`; when both ops order the dims alike this reduces to
    equal factors. `contiguous_dim` moves a dim first only when its split > 1, so that
    order is a reified choice.
  - **Side conditions.** `cores` equal on both ends; "reduction split == 1" on a
    partial-reduction write; multi-stick rescue only when `split·k == num_stick`
    (the view reports factor `split·k`, but its slot stays `(core // stride) %
    split`, so that digit carries its modulus); the interleave rule (an inner split > 1 with an outer symbol unsplit is
    unrepresentable) as an implication; the span cap as `AddDivisionEquality` or a
    per-dim table; a conjunction over every read of the buffer.
  - **Reshapes are just another read order.** The consumer's read already maps its
    loop symbols to the buffer's device coordinates as a mixed-radix expression
    (`dep_device_coordinates`, `pass_utils.py:3410-3414`, e.g.
    `4*outer + floor(inner/64)`), so a reshape is handled like a transposed read:
    - several symbols on one device axis, ordered outer → inner by coefficient:
      `split_inner > 1 ⇒ split_outer > 1` (today's interleave rule, `:3700-3725`).
      Today at most one split symbol owns a device dim; a second goes to
      `decompose_fused_split_view`, so "the axis factor is the product of their
      splits" would be a new, wider rule for the agreement test to flag;
    - one symbol spanning several device axes (the collapsed-axis case
      `decompose_fused_split_view` handles): the split factorises in mixed radix,
      `f = f_outer · f_inner` with `f_inner > 1 ⇒ f_outer == d_outer`, each factor
      dividing its axis;
    - `floor(inner / elems_per_stick)` terms make divisibility count in sticks; partial
      multi-stick coverage stays unrepresentable, as today.

    Views outside this digit form need a fallback: a small allowed-tuples (or view-id)
    table per (node, dep), never a pair table. Known cases: Granite `buf7→buf11` and
    `buf8→buf15` (rescued views). The 2.0 probe inferred these rules from candidate
    keys; 2c records each read's index map and derives them, and the agreement test
    lists what they miss.

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
  - Today's table reads only each consumer's first `MemoryDep` read of the input;
    conjoining every read is a behaviour change the agreement test reports. Not probed:
    the 2.0 lazy arms kept pair tables on the 37 clone edges.

  Agreement: the feasible pairs equal the allocator's final `cd_parent_matches`,
  carries and clones included.
- **2d. Relayouts as constraints.** One optional relayout layout per (parent, consumer)
  READ edge whose view is the consumer's; sharing between consumers by reified view
  equality. A layout either owns a copy or reuses an earlier owner of the same parent
  whose view is equal; the owner's end is a variable ≥ every user's tick + 1, its
  rectangle is the one in the no-overlap, and the price is charged once. So `group`,
  `start_time` and `end_time` become solve outputs for a `Lazy` relayout.
  - **`movement_supported` has a verified closed form**
    (`probes/cpsat-graph/deenum/ms_check.py`; zoo digit views).
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
  - Still to put in closed form, the other per-pair gates of `_cd_parent_relayouts`:
    both ends untiled, the core domain and grouped-gather rejections, the projection
    on both frames (`work_division_from_view`), the transition gate, both spans
    within LX (`partition_footprint`), and the price (per split-dim tables × 1/cores,
    governing dim chosen piecewise).
  - Relayout choice becomes an axis with enumerations: today's group capping
    (`_cap_relayout_groups`) is `Pruned`, and `Lazy` drops the cap (or caps views per
    buffer). `Pruned` reads the division `Table` at both ends and the sibling view
    tables, so the factory rejects it where those are not `Table`s. Agreement where
    capping did not bind; differences reported where it did.
- **2e. In-place as constraints.** Per-core footprint equality
  (`out_partition · out_tiles`, a product equality) plus the `ViewRelation`. The
  pre-solve gates (pointwise input, equal device layout, one-tick handoff) stay
  builder facts.
- **2f. `Lazy` core division.** `Lazy(OpSplitSpace)`: per-axis domains from
  `_legal_split_factors`, product ≤ max cores (multiplication equality), at most one
  reduction split, blocked dims, the span cap as a per-dim table, tiling-dependent
  domains through the level vars. The node gains `space` (today's `division_space`),
  and `add_node` skips the menu for nodes the config can make `Lazy`. Factory rule: a
  menu gives `Table`; no menu but a space gives `Lazy`; `COMMITTED` gives
  `Fixed(COMMITTED)`. `tileable` reads the menu or the space. A `Lazy` node's
  `NodeVars`: cores as a product, `eff_size` by division equality, `core_cost` from
  per-axis tables, tile count `L0·L1`, and no `division_is`. `Table` remains for small
  menus, tests and listed candidates, and `Pruned` for SA-seeded or cost-ranked
  shortlists. Agreement: the space equals the menu per node and tiling, then the same
  optimum.

## Phase 3 — one problem definition (outline; own plan after Phase 2)

- **3a. `graph.set_solver(...)` over a backend protocol.** `set_solver` picks the
  backend class; the `model` passed to `solve` is that backend's model instance (a
  `CpModel` for `CpSat`), so `solve` stays the only entry point. `cost_expr` stays the
  solver-neutral objective, and `cpsat_expr_printer` lowers it onto the protocol.
  - **The protocol is ours, shaped like `pywraplp`.** `pywraplp.Solver` (`ortools
    .linear_solver`) is the template for the API: `CreateSolver(id)`, `IntVar` /
    `BoolVar`, `Add`, `Minimize`, `Solve`, `solution_value`, `SetHint`. On top of it go
    the operations the strategy posts today and in Phase 2 (a census of the Stage 0
    protos): linear rows with `!=` and sparse domains; enforcement by a literal or its
    negation (`OnlyEnforceIf`); `bool_or` / `bool_and` / at-most-one / implication;
    element and enforced table; multiplication / division / min / max equality;
    optional intervals and 2-D no-overlap. `Solve` returns OPTIMAL, FEASIBLE,
    INFEASIBLE or UNKNOWN, and the ladder hints each stage with the previous solution.
    `pywraplp` itself cannot be the protocol: its object API is linear-only, its model
    (`MPModelProto`) has indicator and min / max constraints (SCIP and CP_SAT only,
    through the proto or `model_builder.add_enforced`) but no element / table, interval
    or no-overlap. Linearizing our model (big-M reification, one-hot element/table,
    pairwise no-overlap disjunctions, binary expansion of products) would hand CP-SAT
    a much weaker model, and the solve is 29–73% of planning in the default config
    today (10–11% with coarse tiling; Phase 2.0).
  - **Backends.**
    - `CpSat` — direct pass-through. It is the only backend that lands first.
    - `Mip` — over `pywraplp` (SCIP, HiGHS, CBC, or `CP_SAT` as an MPSolver),
      posting indicators natively where the solver has them and linearizing the rest,
      with `RELATIVE_MIP_GAP = 0` so its OPTIMAL is exact. It gives a second opinion on
      optima and access to MIP solvers.
    - `Annealing` — a wrapper that lets SA act as a generic solver behind the same
      protocol: it records vars, domains and constraints, then searches assignments
      with constraints as hard checks or penalties and the objective from
      `cost_expr`. A flat constraint list makes a weak SA, so the wrapper also takes
      optional neighbourhood providers from the graph and strategy: the view relations'
      geometry (Phase 2c; SA already works lazily over `ResidencyEdge` through
      `_ViewRelation`, `sa_cooptimizer.py:614`) and `Lazy` division generation
      (`:378`).
    - `BruteForce` — an exact oracle for small models. It reads the CP model itself
      (a `CpModelProto`), so it can check the default strategy and Phase 2's encodings
      on tiny graphs before the protocol exists; in 3a it becomes a backend behind it.
      It evaluates every constraint kind the strategy posts itself, with no domain
      propagation and no other solver. `Solve` is a depth-first search:
      - it branches only on variables no constraint defines (presence, division
        index, offset, free literals), presence and division first, offsets last;
        defined variables (element, product, division, min / max, linear equality)
        are computed as their inputs are assigned and checked against their domains;
      - each constraint is checked as soon as its variables are assigned, and
        partially where sound: a row only when all its enforcement literals are true
        and its min / max activity already violates the bound; rectangles intersect
        under CP-SAT's `NoOverlap2D` rule, zero-area boxes included;
      - optional branch and bound: an optimistic objective bound per subtree,
        pruned against the best feasible value so far.

      It is exact by construction and trusts no solver. It is for tests only: a size
      guard refuses models whose branching domains multiply past a bound (roughly
      20–25 binaries). A test that needs it coarsens the model itself (alignment,
      capacity), so `CpSat` and `BruteForce` solve the same model.
    - `Decomposed` — enumerates only the `Table` axes (the division menus, whose index
      vars the strategy marks) and solves the rest of the model at each leaf with an
      inner backend (`CpSat`, `Annealing`, or a placement heuristic). The model is
      posted once; each leaf pins the marked vars by domain changes (not CP-SAT
      assumptions, which take only Boolean literals and force sequential search), so
      nothing is rebuilt per leaf; a placement heuristic rebuilds its per-core inputs
      per leaf, as today. It costs ladder stages × ∏|menu| inner solves and is exact
      only when every leaf ends OPTIMAL or INFEASIBLE. Wrapping today's placement
      solvers it reproduces `ExhaustiveSearchSolver.wrapping` (same inner solver,
      leaves scored by today's HBM bytes as its cost expression), so 3g's same-plans
      check holds; switching it to the real `cost_expr` is a later step of its own
      that lists the tests whose expectations change.
  - **Graph solvers.** A second kind behind `set_solver`: solvers that read nodes,
    enumerations and relation queries directly and write decisions back (greedy,
    first-fit, best-fit, SA placement, `SaCoOptimizingSolver`, `Decomposed`'s heuristic
    inner solver). `graph.solve` runs them in place of `_finalize` and the ladder; the
    greedy fallback after a `SolveError` is the greedy graph solver.
  - **Ground truth.** `BruteForce` on tiny models: the default strategy's encodings,
    then each Phase 2 relation (`Table` vs `Lazy`, the relayout closed form inside a
    model, `apply_op_constraints`), as an oracle independent of CP-SAT next to Phase
    2's agreement tests, and each backend adapter. `CpSat` must match it there, and is
    then the ground truth on real models. `Annealing`, `Mip` and `Decomposed` are
    measured against `CpSat`.
  - **Gate.** Before `Mip` or `Annealing` is offered, replay the Phase 2.0 captures
    (`probes/cpsat-graph/deenum/`) through each, every plan checked and scored by one
    evaluator: `Mip` must reach `CpSat`'s optimum where both prove one (gap 0), with
    solve time; `Annealing` runs only where `CpSat` is OPTIMAL and reports each seed's
    gap at a fixed budget. The linearization cost is measured, not assumed. The SA
    side is prototyped in `probes/cpsat-graph/sa_ilp/`: `SaSolver` implements the
    linear part of this interface (rows, `OnlyEnforceIf` on positive literals,
    optional intervals, `AddNoOverlap2D` as a scored global constraint), next to a
    `cp_model` adapter with the same names; it lacks element / table, products,
    negated literals, `!=` and min / max, and both probes fix the division. Measured
    there: SA was feasible in 56/57 generic ILP runs and all 108 2-D layout runs
    (independently verified). It reached the optimum on 3 of 4 LX-residency instances
    (~0.05 s; on Granite 1 of 3 seeds, after 9 s), and on the 14 of 36 layout captures
    CP-SAT proves within 22 ms (~0.17 s); on the other 22 it fell 1.5–24.5% short
    (median 8.7%), and on generic ILPs up to 47% (GAP). So `CpSat` stays the default,
    and `Annealing` needs the joint operations and the neighbourhood providers to
    compete.
  - **Brute-force scope (probe first).** Prototype `BruteForce` over a `CpModelProto`
    in `probes/cpsat-graph/sa_ilp/`, covering the constraint kinds above. Cross-check
    it against `CpSat` (and SA where the model is linear) on small generic ILPs and on
    tiny LX layouts with 2-D no-overlap (a handful of buffers cut from the captures):
    same optimum as `CpSat` everywhere both finish, and record the size where the
    guard bites. Then use it as the oracle for the strategy unit tests above, from
    Stage 6 on. `Decomposed` follows in 3g.
- **3b. Node absorbs the fields Phase 1 left with the allocator**: `lx_view`,
  `lx_relayout_plans` and `paired_with` (greedy only). Greedy's relayout plans become
  `RelayoutLayout`s on the source (fixed view, all-or-nothing residency), not paired
  nodes. The rest are mapped in Stage 2 and Phase 2. Sizes stay in alignment units on
  the node as the draft has them; `address` is written back in bytes.
- **3c. The graph builds nodes from the IR.** `graph.add_node(op)` builds a node
  straight from the torch IR (once per op): its fields from the op's layout and its
  place in `graph.operations`, with the op's menu or its `OpSplitSpace` as the data the
  strategy enumerates (`Table`) or generates from (`Lazy`), and the residency reasons
  that `apply_op_constraints` reads from the op. It also adds a node with
  `produced_at = None` for each graph input the op reads (an `InputBuffer` or constant is not an `Operation`).
  What it builds depends on the graph's division mode, a build setting read from
  `co_optimizing_lx_planning` when the graph is constructed and kept in its global
  context: sizes, lifetimes (drain plans), residency reasons, in-place candidates,
  clones and the menu all differ between open and committed divisions. The factory
  rejects an enumeration that contradicts the mode a node was built with. In 3c the
  graph from the IR is built beside the allocator's buffers and must agree with the
  graph built from them (same nodes and edges, identical proto) on the corpus and the
  device graphs; the solvers and the commit path stay buffer-in, buffer-out.
- **3d. Placement-only solvers** (greedy, first-fit/best-fit, SA, cooling_schedules,
  utils.plot_buffers) read a placement view of each node (name, size in bytes, uses,
  in-place parents, paired members) and write `address` back in bytes; the SA
  solvers' native packer reads that view. Greedy drops its isinstance at :183.
- **3e. Exhaustive search and the SA co-optimizer** read relations from edges and
  write `assignment`/`resident` on nodes; SA's `_spill_cost` becomes
  `buffer_graph.spill_cost(node, size=<bytes>)` with its `max(0, …)` clamp, and its op
  adjacency moves onto execution edges.
- **3f. Tests**: about 195 constructor calls across 15 files, mostly direct (131 in
  `test_scratchpad_solver.py`, then `test_simulated_annealing.py` 17,
  `test_perm_layout_solver.py` 14, `test_work_division_hint.py` 6), plus helpers
  (`make_buffer_registry`, which constructs through an alias,
  `synthetic_cooptimization_graphs._buf`, `cooptimization_capture_loader._buf`), and
  the tests of the allocator API 3g removes. They move with the stage that changes
  their API.
- **3g. One allocator.** Today `select_allocator` (`allocator.py:5117`) picks
  `ScratchpadAllocator` (33 methods, placement only) or `CoOptimizingAllocator(
  ScratchpadAllocator)` (37 methods: 30 new, 7 overrides; joint) from
  `config.co_optimizing_lx_planning`, `layout_solver` and the solver's
  `chooses_core_divisions()`, and the allocator branches on seven solver capability
  flags (`supports_paired_buffers`, `decides_lx_relayouts`, `generates_divisions`,
  `chooses_tilings`, `tilings_from_menu`, `replans_after_tiling`, `linear_cost_only`;
  `plan_solver.py:988-1035`). Nearly every difference is which unknowns are open and
  who resolves them, which the graph now states generically, so the subclass goes:
  - **What is unknown** is the enumerations. Placement-only is every division
    `Fixed(COMMITTED)`; joint is `Table` / `Lazy` / `Pruned`. The graph's division
    mode (3c) and the factory pick them from config.
  - **Who resolves it** is the backend or graph solver (3a). Five flags
    (`supports_paired_buffers`, `decides_lx_relayouts`, `generates_divisions`,
    `chooses_tilings`, `tilings_from_menu`) become what the backend reports it can
    realize, checked by the factory; an invalid combination is rejected at
    construction instead of branched on inside the allocator. Two stay run-time
    behaviour of the one allocator: `linear_cost_only` (withholds `cost_expr` under
    coarse tiling) and `replans_after_tiling` (apply the tilings, rebuild, re-solve
    placement-only). `ExhaustiveSearchSolver.wrapping` becomes the `Decomposed`
    backend (3a).
  - **Where `CoOptimizingAllocator`'s methods go.**
    - Build (`_prepare_buffers` with `_determine_in_place_division_invariant`,
      `_residency_by_buf`, `_eligible_clone_inputs`, `_check_priced_footprints`) →
      the builder; its build → price → pin → rebuild loop runs `add_node` inside the
      pricing loop.
    - Candidates (`_division_map`, `_enumerate_core_divisions`, `_tiling_candidates`,
      `_division_space`, `_live_splits`) → the strategy producers behind the factory.
    - Pair derivations (`_cd_parent_matches`, `_cd_parent_relayouts`,
      `_cap_relayout_groups`, `_clone_divisions_and_matches`, `_loop_carry_*`,
      `_parent_residency_edges`, `_tile_reads`, `_views_for_divs`,
      `_relayout_consumer_costs`) → relations (2b / 2c / 2d).
    - Pricing (`_pricing_operations`, `_extract_op_features`,
      `_direct_read_candidates_priced`) and `_solve`'s pricing, withholding and cost
      dump → building `cost_expr` around `graph.solve`, the same way for every backend.
    - Post-solve (`_post_solve`, `_materialize_selection`, `_apply_chosen_tilings`,
      `_chosen_tilings`, `_remap_tiled_symbols`, `_commit_copy_out_divisions`,
      `_finalize_lx_relayout_allocation`) → one consumer of `solve()`'s execution plan;
      with nothing open it changes nothing.
    - `_get_spill_reasons` differs from the base class and tests pin both: pick one and
      update the other test. `_solver_generates_divisions` / `_solver_chooses_tilings`
      → factory checks.
  - **`select_allocator`** returns the one allocator with (backend, factory) from
    config. Needs 3a–3f; the allocator's buffer construction (`_build_bound_buffers`
    :1456/:1497, `_append_lx_relayout_destinations` :1819, `_build_cd_bound_buffers`
    :4074/:4275, `_relayout_copy_buffers` :4418) is deleted here, with the buffer form
    of `add_node` kept for tests and capture replay. The allocator's three
    `RelayoutCopyBuffer` checks (:2769, :2964, :3099) become relayout-layout lookups
    and the drain-plan `CoreDivisionBuffer` assertion (:713) a node check;
    `cd_parent_matches` / `cd_parent_relayouts` go once 2c/2d replace them.
    Verification: every `layout_solver` × `co_optimizing_lx_planning` combination in
    the suites compiles to the same plans as before the merge, under the Stage 0
    harness's determinism (1 worker, no wall-clock limit).

## Probes

All measurement code behind this plan is in **`probes/cpsat-graph/`**, with a README
covering setup and every script. Nothing there changes torch-spyre: each hook is a
process-local monkeypatch. Run outputs and captures go to `deenum/runs/`, which is not
committed. Run with the torch-spyre checkout first on `PYTHONPATH` (the shell drivers
take it from `TORCH_SPYRE`); compiling needs the Spyre card, replays do not.

The hooks target torch-spyre `main` 82edd8f6 and patch private solver methods that
Stage 6 deletes, so re-point them (or pin the commit) before re-running on a later
`main`, and record the commit with each run. The phase timers add ~0.95 µs per wrapped
hot-helper call (see 2.0).

| probe | where | answers | used in |
|---|---|---|---|
| Zoo | `zoo.py` | the 32 graphs the zoo runs compile | 2.0 |
| Eager profile and solver-input capture | `deenum/timers.py`, `zoo_t.py`, `granite_t.py`, `run_profile.sh` | where planning time goes per phase; captures each solve's inputs with per-candidate view and tile keys | 2.0 tables |
| Eager + hash join | `deenum/run_profile_hj.sh` (`DEENUM_HASHJOIN=1`) | the bar lazy has to beat: `match_pairs` as a hash join on the exact view key, checked against the original | 2.0 |
| Lazy vs eager replay | `deenum/lazy_probe.py`, `report_probe.py` | per-edge exactness of the per-axis view and tile relations; solve time and objective for E / L1 / L2 at 1 and 16 workers; `INTERLEAVE=1` for deterministic multi-worker search | 2.0, 2b, 2c |
| Relayout legality | `deenum/zoo_ms.py` (record), `ms_check.py` (closed form), `ms_synth.py` (synthetic views) | whether `movement_supported` has a closed form over view digits | 2d |
| Relayout pricing profile | `deenum/zoo_cprof.py` | where `_cd_parent_relayouts` spends its time | 2d |
| SA as a generic backend | `sa_ilp/sa_solver.py`, `cpsat_backend.py`, `problems.py`, `run_probe.py` | whether a `pywraplp`-shaped interface can drive SCIP, CP-SAT and simulated annealing from one model builder; SA's gap on generic ILPs | 3a |
| LX layout with 2-D no-overlap | `sa_ilp/lx_layout.py` | SA with a scored `AddNoOverlap2D` against CP-SAT on the real layout model from captures, with an independent feasibility check | 3a |
| Brute-force oracle (planned) | `sa_ilp/` (to add) | an exact DFS over a `CpModelProto`, cross-checked against CP-SAT and SA on small ILPs and tiny LX layouts; the oracle for strategy unit tests | 3a |

Typical runs:

```bash
TS=${TORCH_SPYRE:-$HOME/torch-spyre}
bash probes/cpsat-graph/deenum/run_profile.sh                 # profile + captures (card)
PYTHONPATH=$TS python3 probes/cpsat-graph/deenum/lazy_probe.py \
    probes/cpsat-graph/deenum/runs/caps_default out.jsonl 1,16 30  # replay (no card)
python3 probes/cpsat-graph/deenum/ms_check.py ms.jsonl         # relayout closed form
PYTHONPATH=$TS python3 probes/cpsat-graph/sa_ilp/run_probe.py ilp.jsonl 10 60
```
