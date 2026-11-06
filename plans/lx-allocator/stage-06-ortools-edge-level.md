# LX allocator, Stage 6: `OrToolsFamily.encode`, edge and op-op level

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0, 1, 2, 3, 4, 5. Needed by: Stages 7, 8, 9, 10, 11, 12, 13.

## Goal

When this PR merges, `OrToolsFamily.encode` builds the whole CP model that `CpSatLayoutSolver` builds
for a joint solve, first objective included, except the 2-D no-overlap section, which Stage 7 adds. That
covers the residency gate, relayout tables and their coupling, the running nest id, copy-outs, the
partial-reduction cut guard, lifetime ends, in-place overwrite, LX prefetch reads, and `encode_objective`, which lowers
Stage 4's terms into `enc.ortools.objective_stages` for Stage 7 to run. The candidate tables move out of
`CoOptimizingAllocator` into one solver-free module, and the allocator delegates to it. Only the test
shadow calls the new encoder. Today's solvers, plans and protos do not change. Completed-reduction rows
exist behind a config flag that is off.

## Today

All of this was read in the snapshot. Citations are under `torch_spyre/_inductor/scratchpad/`
unless they name another path. Section order and variable order were also measured on hand-built
buffers (`exp/stage-06/proto_sections.py`, `exp/stage-06/chain.py`).

**Model order.** `_run` posts in this order (`ilp_solver_ortools.py:1364-1373`): `_add_tiling`,
`_add_lifetime_ends`, `_add_inplace_relaxation` (which ends by calling `_add_no_overlap_2d`, `:1664`),
`_add_core_division`, `_constrain_relayout_copies`. Then `_tile_count_terms` (`:1427`), then the
objective. Measured variable order on a tiled graph with a clone, in-place edges and one copy:

```text
per buffer, clones first, copies last: in_buffer_X off_X merge_* div_ eff_size_ core_cost_ occ_ split_* tile_*
_add_tiling:          loop_id_* tiled_* (samenest_/joined_/tile_group_) (apart_/via_full_) cut_* copy_reads_* nest_end_* nest_start_*
_add_lifetime_ends:   end_* start_*
_add_no_overlap_2d:   length_* top_* start_<in-place child>
_add_core_division:   match_* served_*
_constrain_relayout:  div_<source>_is_<i>
_tile_count_terms:    tile_count_*
_cut_traffic_terms:   resident_cut_* reads_full_*
```

Three facts from that output matter. `merge_*` is created inside the per-buffer block, before the
division variables (`:308-311`). `in_buffer_`, `off_`, `loop_id_`, `tiled_` and `cut_` also exist
for input clones and relayout copies, because `_add_tiling` walks every buffer with a menu
(`:945-949`, `:978`). And removing `_add_no_overlap_2d` removes only `length_`, `top_` and in-place
`start_` variables, plus one contiguous run of constraints. It uses no `new_constant`, so every
other variable keeps its relative order, which is what the gate relies on.

**The gate.** `constrain_residency` (`ilp_solver_ortools.py:534-597`) is the only edge rule on the
joint path. For each reader of a resident buffer it requires one of three ways:

- `match_<p>__<c>`: the division pair is in `cd_parent_matches`;
- `served_<p>__<c>__g<k>`: the reader reads a resident relayout copy, gating the pair to the
  group's candidates;
- `via_full[c]`: the reader reads a copied-out full buffer.

With no alternative it posts the table directly under `in_buffer` (`:585-589`). An empty table
forces `enforce_lit == 0` (`_gate_divisions`, `:248-259`). `_get_children` (`:1764-1777`) builds
readers in buffer order, one entry per listing in `parents`. `parents` comes from
`[dep.name for dep in rw.reads]` (`utils.py:267`, `:276`), one entry per distinct read dep. So
`x * x` is one entry, and `x[..., :8] + x[..., 8:]` is two entries sharing one `ResidencyEdge`.
`_op_readers` (`ilp_solver_ortools.py:211-223`) filters relayout copies out, but a copy never
appears as a reader because `RelayoutCopyBuffer.parents` stays empty (`plan_solver.py:783-786`), so
that filter is a no-op on main and its comment is stale.

**Match tables.** `ResidencyEdge.match_pairs` (`torch_spyre/_inductor/work_division.py:1677-1721`)
compares core views, tile views and `cores_used`. `parent_view` returns `None` for a partial-reduction
write (`:1469-1479`), so a producer division with a reduction split has no match pair at all.
`_parent_residency_edges` (`allocator.py:4830-4870`) drops prefix-contraction edges and barred producers.

**Relayout tables.** `_cd_parent_relayouts` (`allocator.py:5026-5325`) enumerates per consumer and
per parent, gated on `_decides_lx_relayouts`, `config.lx_solver_relayout` and `not
config.ktir_emitter` (`:5056-5061`). `SPYRE_LX_PLANNER_RELAYOUT` does not gate it. The
division-independent gates are `solver_relayout_edge_context` (`lx_relayout.py:873-969`). The per-pair
gates run in a fixed order: partial or unrepresentable source views go first
(`allocator.py:5089-5094`), then tiled candidates (`:5174`, `:5196`), `same_partition` pairs (`:5179`),
the padded-K and local-subrange rules (`:5181-5195`), `core_domain_rejection` and
`grouped_gather_rejection` (`:5203-5206`), projection (`:5207-5227`), the transition check
(`:5232-5238`), spans (`:5239-5245`), and last the price, `solver_relayout_pair_cost`
(`lx_relayout.py:1475-1529`), where `movement_supported` (`:407-421`) applies, times the loop trip count
(`allocator.py:5272`).

Destination views are interned per parent by ownership (`_intern_view_group`,
`allocator.py:2869-2884`), which gives the group id in `served_...__g<k>` and in the copy's name.
Interning runs in consumer order, so group ids depend on it. `_cap_relayout_groups` (`:4893-4989`)
keeps four groups per edge. `_relayout_copy_buffers` (`:4611-4660`) drops groups whose source is
absent or whose span exceeds LX. `build_relayout_copy` (`plan_solver.py:856-900`) makes the copy
live from the first offered consumer to the last offered consumer, whether or not each fires.

**Completed-reduction routes are not on the joint path.** `derive_completed_reduction_routes`
(`lx_relayout.py:579-642`) is called only from `collect_lx_relayout_plans` (`:1731-1736`), which runs
only for a placement allocator whose solver supports paired buffers (`allocator.py:1069-1077`). That
means `GreedyLayoutSolver`, including the `SolveError` fallback (`allocator.py:5495-5515`).
`CpSatLayoutSolver.supports_paired_buffers` is False (`ilp_solver_ortools.py:632`).

**Coupling.** `_constrain_relayout_copies` (`ilp_solver_ortools.py:1578-1606`): a resident copy implies
a resident source, a priced source division and a served reader. A copy with neither is pinned out.

**Nests.** `_add_tiling` (`:902-1027`) interns `TileSpec.level_counts` into `loop_id`.
`_add_tiling_groups` (`:1113-1172`) sorts tileable ops by position, starts a new segment at a gap,
and otherwise makes `samenest_`, `joined_` and `tile_group_<op> == prev + 1 - joined`. Ids compare
only within a segment. A segment's first op holds `new_constant(0)`. `joined` knows nothing about
data edges. Tile alignment comes from the per-reader literal `apart_`: not apart means the pair is
in `cd_parent_matches` (`:993-1001`), and `match_pairs` compares tile views. A reader in another
segment makes the producer `unshareable` and `via_full == tiled`. `cut_<op>` is true when the op
is tiled and either some reader is apart or it is unshareable, which includes a graph output and an
op with no modelled reader (`:1012-1023`).

Note the name clash. Today's `cut_<op>` means "this op's value is copied out". The plan's cut,
`t in graph.cuts`, means "the op at tick `t` and the next op do not share a nest", which is
`not joined_<a>_<b>`.

**Partial-reduction guard.** `_add_partial_reduction_cut_guard` (`:1029-1050`) posts
`¬cut ∨ ¬in_buffer ∨ copy_reads` when any menu entry has a reduction split.

**Nest bounds and lifetimes.** `_add_nest_ends` (`:1052-1083`) walks each run backwards and
`_add_nest_starts` (`:1085-1111`) forwards. `_add_lifetime_ends` (`:1204-1272`) sets `end` on
tileable buffers and on anything a nest reads. It sets `start` only on an input clone read by a
nest, which "never binds" (`:1247-1255`). A tiled op's own box still starts at its write slot
(`:1710`, `:1724`). The only reader of `nest_start` is that clone bound.

**The objective.** With a cost expression, `_minimize_cost_expr` (`:830-900`) binds the symbols, lowers
through `cost_expr_to_cpsat`, which adds helper variables and constraints, and calls `minimize` unless
the result is a constant. A lowering error warns, leaves those helpers in the model and falls to the
ladder (`:886-900`). The ladder path (`:1449-1523`) pins every copy out (`:1454-1455`), creates the
`resident_cut_` and `reads_full_` literals in `_cut_traffic_terms` (`:1274-1323`) and solves up to five
stages. The charges are one `RelayoutCharge` per copy, in name order (`allocator.py:3346-3351`), a table
indexed by source division (`plan_solver.py:487`).

The allocator withholds the cost expression when the solve chooses tilings
(`allocator.py:3275-3299`), so with `AUTO_COARSE_TILING=1` every copy is pinned out
(`ilp_solver_ortools.py:1454`). The solver cannot fire a relayout under `AUTO_COARSE_TILING=1`; the
greedy re-plan after a failed solve still can (`allocator.py:1075-1077`). The ladder also runs with
tiling off when the expression fails to build or lower.

**In place.** `_add_inplace_relaxation` (`ilp_solver_ortools.py:1608-1664`) posts, for each
`merge_<p>_<c>`: shared offset, both resident, `constrain_merge` (equal `eff_size`, match table),
not `via_full`, and `src.end == write_slot(dst.start) + 1`. Then `add_at_most_one` on incoming and
on outgoing, then `offset + eff_size <= M` per buffer, then the no-overlap. Candidates come from
`_determine_in_place_division_invariant` (`allocator.py:4032-4099`), the drain recheck
(`:4279-4286`) and the reverse-clone block (`:4426-4450`), all through `_inplace_edge_ok`
(`:1677-1728`). The list is `op_inputs` order, then reverse clones in last-consumer order.

**Clones.** An LX prefetch is an input clone today. An eligible graph input becomes a `CoreDivisionBuffer` with `first_use_is_read=True`,
`parents=[]` and a menu built from its consumers (`allocator.py:4226-4267`).
`_clone_divisions_and_matches` (`:4730-4828`) collects distinct read views over all consumers, skipping
tiled candidates, unrepresentable views and broadcast reads. Each new view adds a `CoreDivision` keyed
by the consumer's loop symbols. The consumer's `cd_parent_matches[input]` is then `(k, j)` pairs. A
clone has no `ResidencyEdge`, no relayout candidates and no `via_full`, so its gate is the plain table
under `in_buffer`. Inputs not eligible are not in the model at all.

**Loop carries.** Carry updates and reads add match-only entries to `cd_parent_matches` and to
`parents`, for pairs that are not operand reads of the producer's output by that name
(`allocator.py:4326-4375`, `:4408-4410`). They never get relayouts (`:4377-4379`).

**Restickify.** A pre-pass restickify is an ordinary Pointwise op with its own buffer, residency,
tick and readers. Its menu is built like any op's (`allocator.py:3737-3800`), and the only
restickify-specific handling is the deferred direct read (`:3751-3764`). `div_<name>` exists only
when the menu has more than one entry (`ilp_solver_ortools.py:454-479`); `in_buffer_` and `off_`
always do. `_restickify_barrier` (`allocator.py:1434-1483`) runs only on the placement path
(`_BufferResidencyChecks.restickify`, `:847-855`; `_input_residency_reason`, `:1347-1353`). The
joint path enforces the same hazard with the generic gate: a resident buffer read by a restickify
needs a match pair or a served copy, and the solver has no restickify-specific code.

## Design

### Where the tables live

A new solver-free module, `scratchpad/lx_allocator/family/edge_tables.py`, holds everything that
today sits on `CoreDivisionBuffer` for edges. Tables live in the encoder, not on the graph
(`scratchpad-architecture-vs-main.md:155`). Menu indices appear only in `Gate.pairs`, in
`RelayoutCandidate.source_division` and `.consumer_division`, and in `loop_id` values. None of them
reach `DataEdge`, `graph.cuts`, `Pin`, `Step` or `Sym`.

```python
class GateKind(Enum):
    READ = auto()    # producer -> consumer: match or relayout
    PREFETCH = auto()   # LX prefetch -> consumer: match only
    CARRY = auto()   # loop-carry storage -> update or reader: match only

@dataclass(frozen=True)
class Gate:
    """One listing of `source` among `reader`'s parents (today's `parents` entry)."""
    source: OpNode
    reader: OpNode
    kind: GateKind
    pairs: tuple[tuple[int, int], ...]        # source menu index, reader menu index
    relayouts: tuple[RelayoutCandidate, ...]  # empty unless READ
    edges: tuple[DataEdge, ...]               # data edges this listing stands for
    residency: ResidencyEdge | None = field(default=None, compare=False)

@dataclass(frozen=True)
class RelayoutGroup:
    """One destination view of one source: what one shuffle produces."""
    source: OpNode
    index: int
    candidates: tuple[RelayoutCandidate, ...]  # sorted as build_relayout_copy sorts
    readers: tuple[OpNode, ...]
    uses: tuple[int, ...]                      # sorted ticks of the offered readers
    @property
    def name(self) -> str: ...                 # relayout_copy_name(source.name, index)
    @property
    def size_bytes(self) -> int: ...           # span * destination cores, as today
    @property
    def num_cores(self) -> int: ...
    @property
    def cost_by_source_division(self) -> dict[int, float]: ...
    def candidates_for(self, reader: OpNode) -> list[RelayoutCandidate]: ...

@dataclass(frozen=True)
class PrefetchMenu:
    divisions: tuple[CoreDivision, ...]        # keyed by a consumer's symbols, as today
    views: tuple[PerCoreView, ...]             # index-aligned

@dataclass
class EdgeTables:
    gates: dict[str, list[Gate]]               # source name -> readers, in today's order
    groups: dict[tuple[str, int], RelayoutGroup]
    prefetches: dict[str, PrefetchMenu]
    in_place: dict[str, list[str]]             # child name -> candidates, today's order
    def relayout_sources(self) -> set[str]: ...
    def write_view(self, source: OpNode, division: int) -> PerCoreView | None: ...
    def read_view(self, gate: Gate, division: int) -> PerCoreView | None: ...

def build_edge_tables(
    graph: SpyreOptimizationGraph,
    menus: Mapping[OpNode, Sequence[CoreDivision]],
    *,
    capacity: int,
    decides_relayouts: bool,
    prep_cache: dict | None = None,
    pair_costs: dict | None = None,
) -> EdgeTables: ...

def prefetch_menu(input_node: InputNode, tables: EdgeTables) -> PrefetchMenu | None: ...
def in_place_candidates(node: OpNode) -> list[str]: ...
def residency_edge(gate: Gate) -> ResidencyEdge | None: ...
def edge_match_pairs(gate: Gate) -> tuple[tuple[int, int], ...]: ...
def edge_name(edge: DataEdge) -> str: ...      # f"{src}__{dst}__{port}", the Sym.restick key
```

`build_edge_tables` is the body of `_build_cd_bound_buffers` (`allocator.py:4220-4490`) with the
buffer construction removed, in the same order, because order decides results here:

1. carry update edges (`:4196-4218`), from `carry_pairs(graph)`, which applies no residency filter,
   so that filter is here;
2. prefetch menus (`:4226-4247`), with `get_buffer_users(graph.ir)` as today, none without
   `Optimization.PREFETCH`;
3. for each op in `graph.ir.operations` order: residency edges, match tables, carry merges, prefetch
   match merge, relayouts, prefetch re-merge, in-place edges from prefetches (`:4269-4450`);
4. groups (`:4490`).

A node's residency here is `residency_reason(node, graph, decides_relayouts=decides_relayouts,
tileable=...)`, which includes the two rules outside the registry, and `None` for an input with a
`PrefetchMenu` (`:4240`). This is `residency_by_buf`, not `DivisionTable.residency`, which adds the
capacity reason.

Moved or copied:

| old | new home | how |
|---|---|---|
| `_cd_parent_relayouts`, `_cap_relayout_groups`, `_relayout_consumer_costs`, `_intern_view_group`, `_coarse_tile_loop_trip_count`, `_views_for_divs`, `_clone_divisions_and_matches`, `_eligible_clone_inputs`, `_parent_residency_edges`, `_cd_parent_matches`, `_loop_carry_update_edge`, `_loop_carry_read_edges` | `edge_tables.py` | moved; allocator methods become one-line delegates and module functions are re-imported, so main and the new encoder share one body. Stage 13 deletes the delegates. |
| `_restickify_barrier`, `_restickify_read_is_core_local` | `restickify_residency_bar`, `restickify_read_is_core_local` in `edge_tables.py` | moved, allocator delegates |
| `_determine_in_place_division_invariant` | stays; `DataEdge.in_place_candidate` holds the same gates | Stage 13 deletes |
| `_build_cd_bound_buffers`, the `while True` in `_prepare_buffers` | allocator keeps them; the first calls the moved functions | `settle_deferred_reads` is a copy over tables. Stage 13 deletes the old ones |
| `RelayoutCopyBuffer`, `build_relayout_copy`, `_relayout_copy_buffers` | `RelayoutGroup` in `edge_tables.py` | copied, because old solvers still need the buffer; Stage 13 deletes the old ones |
| `constrain_residency`, `_gate_divisions`, `constrain_merge`, `_add_tiling` from `:975`, `_add_tiling_groups`, `_add_nest_*`, `_add_lifetime_ends`, `_add_inplace_relaxation` through `:1662`, `_constrain_relayout_copies`, `_add_partial_reduction_cut_guard`, the variables of `_cut_traffic_terms`, the lowering in `_minimize_cost_expr`, `decision_variables`, `_Slots`, `_TilingGroups` | `family/ortools_encode.py` | copied; `CpSatLayoutSolver` still needs them until Stage 13. `_op_readers` is not copied: it filters nothing, and gates never list a copy as a reader. |

Moving rebinds module names, so a test patching `allocator_module.X` stops applying to a moved body
(feasibility report, deviation 4). `exp/stage-06/moved_patches.py` crosses every patched name in `tests/`
with the names the moved bodies use. To re-point: `tests/inductor/test_solver_relayout_e2e.py:864`
(`_coarse_tile_loop_trip_count`), and the five patches behind `_restickify_barrier` at
`tests/inductor/test_work_division_hint.py:955-970`. `tests/inductor/test_work_division.py:3331` and
`:3369` use `allocator_module.build_residency_edge`, so they switch to `work_division_module`.
`op_read_writes` is patched on the allocator module at six more sites (`test_coarse_tiling.py:9713`,
`test_for_each_tile_lowering.py:2700`, `test_op_features.py:96`, `test_work_division.py:3550`, `:3626`,
`:3686`). I did not trace which of those drive a moved body, so commit 1 runs them and re-points any
that stop applying.

**Deferred direct reads.** The rejection of a staged read depends on relayout sources
(`allocator.py:3203-3216`) and rebuilds everything (`:3145-3183`), so tables and the menus they read
must settle before any variable exists. This stage owns that loop:

```python
def settle_deferred_reads(
    graph: SpyreOptimizationGraph,
    tables: dict[OpNode, DivisionTable],
    *,
    relayout_sources: Callable[[], Collection[str]],
    rebuild: Callable[[], EdgeTables],
) -> EdgeTables: ...
```

`pending` is the nodes with `deferred_direct_read` whose table is not the one fixed entry. Each pass
calls `rebuild()`, returns if nothing is pending, and otherwise prices through
`project_transport_read_copies(graph.ir, ..., relayout_sources=relayout_sources())`, which is called
after `rebuild`, so a closure may read the tables it just built. A node rejected as at
`allocator.py:3159-3165` gets `Pin(node, "split", WorkSplitPin(split, "unproved or unpriced direct read"),
"hazard")`, and its
`DivisionTable` narrows in place to that one entry, through `_legal_fixed_division`, with
`enumerated=False`. Rejection is monotonic, so the loop ends. Interning restarts on each pass
(`allocator.py:4153`) and `pair_costs` persists (`:3119`), as today. Stage 8 passes
`relayout_sources=lambda: ()`.

### Encode order

`OrToolsFamily.encode` is Stage 5's: `begin(graph)`, `build_tables(enc)`, `encode_ops(enc)`. Each
function in `family/ortools_encode.py` takes the `Encoding` first, and the family-private state
sits in `enc.ortools`, Stage 5's `OrToolsState`: its `ops`, `units`, `columns`, `tile_terms`,
`tables` and `edges`, which this stage fills, and this stage's `copies`, `relayout_literals`,
`incoming` and `objective_stages`. The comments give what this stage fills in `encode_ops`:

```python
def encode_ops(enc: Encoding) -> None:   # ortools_encode.py
    settle_tables(enc)            # settle_deferred_reads, then prefetch tables
    encode_op_vars(enc)           # merge_* literals through declare_in_place_vars
    encode_relayout_copies(enc)   # add_unit per group
    encode_loop_columns(enc)
    encode_tiling(enc)            # lines 975-1172, then _encode_lifetimes
    encode_in_place(enc)          # lines 1633-1658
    encode_placement(enc)         # _encode_capacity here; Stage 7 replaces it
    post_residency(enc)
    encode_restickifies(enc)      # no proto in Phase 2
    encode_read_gates(enc)        # gates, then relayout coupling
    encode_tile_counts(enc)
    bind_symbols(enc)
    encode_sticks(enc)
    encode_objective(enc)         # this stage; Stage 7 runs the stages
```

`settle_tables` calls `settle_deferred_reads`, with `rebuild` storing `build_edge_tables(graph,
menus(enc), capacity=graph.target.lx_bytes, decides_relayouts=StepKind.RELAYOUT in
graph.target.steps, ...)` in `enc.ortools.edges` and
`relayout_sources=lambda: enc.ortools.edges.relayout_sources()`. It then builds one
`DivisionTable(node.name, node, list(menu.divisions), enumerated=False)` per `PrefetchMenu`, in
input order, apart from `enc.ortools.tables`. `encode_prefetches` calls `add_unit` on
each, so `encode_op_vars` visits LX prefetches first, as today's buffer order does.

`declare_in_place_vars` runs inside `add_unit` straight after `off_`, because `merge_*` sits between
`off_` and `div_`. It makes one `merge_<p>_<c>` per entry of `EdgeTables.in_place`. A name repeated by a
double read gets a second variable that the dict drops, as `ilp_solver_ortools.py:308-311` does.

`encode_relayout_copies` calls `add_unit` per group with a one-entry `DivisionTable` whose `node` is
`None` and whose entry is `CoreDivision(splits={"relayout_copy": n})`, so `eff_size`, `cores` and
the constants come out as they do for a copy today (`:454-479`). The bundles go in
`enc.ortools.copies: dict[str, OpVars]` by group name, and `_table_owners(enc)` returns `ops` then
`copies`. `PlacementKey` for a copy is that name. `enc.ortools.relayout_literals` maps the same names
to `in_buffer`.

Every constraint goes through `post(enc, subject, rule, lambda cp: ...)` with today's exact call
(`add_implication`, `add_bool_or`, `add_allowed_assignments`, `add_element`, `add_max_equality`,
`add_at_most_one`), one `post` per constraint, so Stage 5's invariant of exactly one `Provenance` per
constraint holds. A rule is a `Rule` flag, or the name of an `ALWAYS` check as a string:

| constraints | rule |
|---|---|
| group ids, `apart_`, `via_full_`, `cut_`, nest ends and starts | `"tiles_line_up"` |
| cut guard | `Rule.RESIDENT_TILE_UNREDUCED` |
| `resident_cut_`, `reads_full_` definitions | `Rule.COPY_OUT_IN_HBM` |
| `end_`, `start_`, `_encode_capacity` | `"boxes_fit_lx"` |
| `merge_*`, `add_at_most_one`, the shared offset | `"in_place_conditions"` |
| gate, reader not a `RestickifyNode` | `"resident_read"` |
| gate, reader a `RestickifyNode` | `"cross_frame"` |
| served literals, coupling, the ladder's pin-out | `"relayout_source"` |
| the printer's helpers, through `tag(enc, "graph", rule, start)` | `TABLE_COLUMN` |

### What the target switches on

The edge-level functions read the card only from `graph.target`. `OrToolsFamily` declares what they
cover: `STEPS` holds `COPY_OUT`, `RESTICKIFY` and `RELAYOUT`, `RULES` holds `RESIDENT_TILE_UNREDUCED`
and `COPY_OUT_IN_HBM` (Stage 5 adds the other two), and `OPTIMIZATIONS` holds `IN_PLACE`, `PREFETCH`
and `COARSE_TILING`. A step kind outside `graph.target.steps` is never offered, and an optimization
outside `graph.target.optimizations` gets no variables. `SPYRE` sets all of them, so none of this
changes the proto.

| off on the target | effect |
|---|---|
| `StepKind.RELAYOUT` | `build_edge_tables` runs with `decides_relayouts=False`: no rows, groups, copies or served literals, and residency reasons as on today's path that does not decide relayouts |
| `StepKind.COPY_OUT` | `cut_<op>` and every `via_full_` literal are posted false, so no tiled entry whose value leaves its nest stays legal |
| `StepKind.RESTICKIFY` | `encode_restickifies` raises on a pinned or forced restickify, because the builder made the graph for this target |
| `Rule.RESIDENT_TILE_UNREDUCED` | the cut guard is not posted |
| `Rule.COPY_OUT_IN_HBM` | nothing: this family has no LX copy-out, so it keeps a copy-out in HBM and is stricter than such a card |
| `Optimization.IN_PLACE` | `EdgeTables.in_place` is empty: no `merge_*`, and `encode_in_place` posts nothing |
| `Optimization.PREFETCH` | no `PrefetchMenu`: no input units, no `PREFETCH` gates, no in-place edge from an input |
| `Optimization.COARSE_TILING` | Stage 5's tables hold no tiled entry, so `encode_tiling` adds nothing, as `_add_tiling` does today |

### The gate

`encode_read_gates` walks units in order and, for each, `enc.ortools.edges.gates[unit.name]`, so one reader
listed twice gets two gates, as today (feasibility report, deviation 1). `gates`, not
`graph.data_edges`, drives the walk, because a listing is not an edge: a `CARRY` gate has no read
edge, and a mutation edge or an edge to a drained collective has no listing (Deviation 7). `Gate.edges`
holds the edges a listing stands for. `x * x` is one dep, so one listing and one edge. A self-matmul
shares one edge across two ports. The body is `ilp_solver_ortools.py:560-597`. A PREFETCH or CARRY gate
has no relayouts. `_gate_pairs` is the one function that reads an `(i, j)` table, so a later encoder can
replace it with view-id equality (feasibility report, 2c) without touching the callers:

```python
def _gate_pairs(enc, subject, rule, pairs, src: OpVars, dst: OpVars, when) -> None:
    """_gate_divisions: `when` implies (src.division, dst.division) is in `pairs`."""
```

Graph mapping of a served literal: true means a `RELAYOUT` step on that gate's edges. Stage 7 decodes
it with `fired_relayouts` below.

### The relayout table

The key is (source layout, producer reduction split, destination layout), as the plan says. In code
the source layout is the source view under source division `i`, so a row is a `RelayoutCandidate`,
extended with two defaulted fields so every existing caller keeps working:

```python
@dataclass(frozen=True)
class RelayoutCandidate:
    ...                                         # existing fields unchanged
    reduction_split: int = 1
    completed_producer_cores: tuple[int, ...] = ()
```

`reduction_split` is 1 when every core holds a finished slice. Views are interned by `same_partition`
per source, because `PerCoreView` equality is structural. The group stays keyed on destination view
alone, as today, so one copy can be fed from any priced source division.

On the graph a relayout appears only as a step of kind `RELAYOUT`. `to_view` is a `PerCoreView` in
the device frame of the layout the step applies to. `to_view`, `reduction_split`, `source_view`,
`source_footprint_bytes` and `destination_footprint_bytes` all come from `FiredRelayout.row`. A price
never reaches the graph by menu index: `cost_domains` keys it by selector value.

"Never across a cut" and "only from a resident source" are table and coupling facts. No row has a
tiled menu entry on either end, so a served literal already implies both ends untiled, hence no
copy-out on that edge. `copy.in_buffer => source.in_buffer` is the coupling.

### Decoding a fired relayout

`scratchpad-architecture-vs-main.md:143` records the bug in PR #5676: `fired()` applied an index into
the node's menu to its point grid, so a relayout was built from the wrong division. Stage 7 owns
`read_back`, but the only route from a literal to a row is here, so the bug cannot return through a
second decoder:

```python
@dataclass(frozen=True)
class FiredRelayout:
    gate: Gate
    group: RelayoutGroup
    row: RelayoutCandidate

def fired_relayouts(enc: Encoding, value: Callable[[Any], int]) -> list[FiredRelayout]:
    """Every served literal that holds, resolved to its one row."""
```

Rules this code follows:

1. **One index space.** `i` and `j` are `value(src.division)` and `value(reader.division)`, the same
   variables `_gate_pairs` constrained, and rows are matched on those fields of the group that minted
   the literal. No index is mapped through any other list.
2. **Exactly one, or a raise.** More than one served literal on a listing, a served literal on a
   spilled copy, or zero or several rows raises `RuleViolation`. Today's `assert`s
   (`ilp_solver_ortools.py:1906-1912`) vanish under `-O`, so they do not carry over.
3. **Verified against recomputed views.** The row's `source_view` must be `same_partition` with
   `EdgeTables.write_view(source, i)` and its `destination_view` with `read_view(gate, j)`. The views
   are computed from the menu entries, not read from the row. This costs one view per fired edge.
4. **An independent check.** `relayout_source` in `check_decisions` gains rule 3's comparison, from
   the decoded `work_split`s and the `RELAYOUT` step's fields, without the tables, so a decoder that
   bypasses this function still fails. With `graph.ir` `None` it counts in `skipped`.

### Nests, cuts and copy-outs

`encode_tiling` starts at line 975, after Stage 5's columns, and copies `_add_tiling`,
`_add_tiling_groups`, `_add_nest_ends` and `_add_nest_starts`, with today's variable names, over
today's unit order for cuts and over sorted positions for groups and nest walks
(`cpsat-graph-staged-conversion.md:325`: unit-test record lists are not in tick order).

How that maps to "a nest is a run of ops, not an object":

| proto variable | meaning | plan name |
|---|---|---|
| `loop_id_<op>` | interned `level_counts` of the chosen entry | private table index |
| `joined_<a>_<b>` | adjacent ops run the same non-empty nest | `a.tick not in graph.cuts` |
| `tile_group_<op>` | running id, steps wherever `joined` is false | the running nest id; not stored on the graph |
| `apart_<a>_<b>` | reader's id differs | a cut lies between the edge's ends |
| `via_full_<a>_<b>` | tiled and apart | this edge carries the `COPY_OUT` step |
| `cut_<op>` | some edge carries a copy-out, or a graph output | `Sym.cut(node)`, not `graph.cuts` |

There is no nest object. `nest_start`, `nest_end` and the group id are variables on the unit's
`OpVars`. Two ops share a nest exactly when their ids agree within a segment, and on the graph exactly
when no tick in `graph.cuts` lies from the first up to the second. Between two adjacent untiled ops
`joined` is false, so the first op's tick decodes into `graph.cuts`: each untiled op is a run of one.
Around an op with no tileable entry the cut is constant: its tick and the tick of the running op
before it are always in `graph.cuts`. Stage 3's nest helper reads the same definition. The helpers
Stage 7 decodes from:

```python
def joined_literal(enc: Encoding, before: OpNode) -> Any: ...  # False across a segment break
def copy_out_literal(enc: Encoding, node: OpNode) -> Any: ...  # cut_<op>, or False with no tiling
def via_full_literal(enc: Encoding, edge: DataEdge) -> Any: ...  # a literal, or False
```

`joined_literal(enc, before)` is the literal for "the next running op shares `before`'s nest", and
`read_back` adds `before.tick` to `graph.cuts` where it is false.

The cut guard is posted at today's position inside the per-op loop. `nest_start` is kept with today's
single reader (Deviation 5). A `COPY_OUT` step is one object per producer, shared across its
`via_full` edges. A graph output or an op with no modelled reader has a copy-out decision and no
such edge (Deviation 6).

### Lifetimes

`_encode_lifetimes`, called at the end of `encode_tiling`, copies `_add_lifetime_ends`. Slots stay
`2t` and `2t + 1`. Each `OpVars` gets `end` and `start`, and the box span stays with Stage 7, which
consumes `end`, `start`, `enc.ortools.incoming: dict[str, list[literal]]` and the slot values.

### In-place overwrite

Eligibility stays today's rule for parity. The read-pattern rule is the parent's Future work and is
not built here. `DataEdge.in_place` is the decoded value of `merge_<p>_<c>`, through `in_place_literal(enc, edge) -> literal
| None`. `DataEdge.in_place_candidate`, from Stage 2, is the division-invariant gate, but a flag has
no list order and Stage 8's parity reads that order, so `in_place_candidates(node)` supplies it: the
sources of flagged edges from computed ops, in `op_read_writes(node.op).reads` order, then flagged
edges from inputs, in `graph_input_names` order. `EdgeTables.in_place[child]` is that list less any
input with no `PrefetchMenu`, which has no unit, and is the ordered truth for the encoder. A caller of
`in_place_candidates` must drop those inputs too. A test holds it equal to today's
`in_place_parents`. `check_in_place_parent_is_read` runs once while building `in_place`, and a
violation raises, like `_check_in_place_relationships` today.

The order inside `encode_in_place` is `ilp_solver_ortools.py:1635-1658` with no change. OR-Tools stays
stricter than Stage 3's physical `in_place_conditions` and `tiles_line_up` by encoding: equal `eff_size`, and a match
pair on every edge inside a nest. The shared offset is one helper, the first post of each edge:

```python
def _post_shared_offset(enc: Encoding, src: OpVars, dst: OpVars, merge) -> None: ...
```

`encode_placement` here only calls `_encode_capacity`, which posts `:1660-1662` for the same parity
reason. Stage 7 replaces the helper with `tie` and `_encode_capacity` with `place` (Deviation 8).

### LX prefetch without a `Buffer`

An `InputNode` holds what an input clone holds today: `resident`, `offset` and
`prefetch_view` in place of a division, because a prefetch is a view. `work_split` stays `None`, since an
input has no axes. A true `in_buffer_<input>` means commit adds the `clone` op; there is no prefetch
variable or class, as in the worked example. Only inputs with a `PrefetchMenu` are units, so an ineligible
input gets no variables, as today. `build_edge_tables` decides eligibility with the moved `_eligible_clone_inputs` logic, and this
stage is the only owner of prefetch menus and `(k, j)` pairs.

LX prefetch reads are `Gate(kind=PREFETCH)`, with `pairs` being `(k, j)`. Row `k` of the menu is a view, not a
division of the input, so `CoreDivision` keys stay consumer symbols, because `split_<input>_<sym>` in
the proto uses them. They never reach the graph. Stage 7 decodes with:

```python
def prefetch_view(enc: Encoding, node: InputNode, value: Callable[[Any], int]) -> PerCoreView:
    """The chosen row of the node's PrefetchMenu."""
```

That is the parent's "LX prefetch reads" bullet, the clone-read constraint under its new name: the prefetch's division selects a view, and each consumer's
division must read that view. Only `_gate_pairs` would change for a non-enumerated form.

### Restickify and the hazard

**The pinned step needs a node.** A pre-pass restickify has a menu, residency, a tick, readers and a
cost term today, and `Step` carries no split. So it is a `RestickifyNode`, encoded as any op, and
the edge into it carries `Step(kind=StepKind.RESTICKIFY, pinned=True, realized_by=node)`. The step
has no footprint, variables or box of its own (Deviation 1).

`encode_restickifies` handles the general case so Stage 11 only lifts pins:

```python
for edge in graph.data_edges:
    if graph.pins_on(edge, "restickify"):  # the step exists: pinned_steps_stay holds
        symbol(enc, Sym.restick(edge_name(edge)).name, True)
        continue
    forced = _neq(enc, stick_of(edge.src, graph), read_stick(edge, graph))  # folds on constants
    symbol(enc, Sym.restick(edge_name(edge)).name, forced)
```

A pinned edge is not compared, because the two sticks are host dims of different buffers: a
stick-swapping transpose can land on the same dim, which would post a false constraint on a legal
graph. With Stage 2's stick pins, `forced` folds to a Python bool, so nothing reaches the proto. In
Phases 1 and 2 the pre-pass inserted every restickify, so a folded True with no restickify pin should not
exist. The encoder raises there, so a Stage 2 gap is found rather than posted as a step commit cannot
insert. Stage 11's 11a builds its beam restickifies as virtual `RestickifyNode`s, so the raise holds
there. Unpinned steps, carrying `target`, first exist at 11b, which changes the raise.

**The hazard.** The rule is that a restickify reads a resident source only through that source's
same-core slice, or after a relayout, or from HBM. On the joint path this is exactly the gate: `match`
is the same-core slice, `served` is the relayout, `via_full` is HBM. So this stage posts no new
constraint. A gate whose reader is a `RestickifyNode` carries the rule `"cross_frame"`. That holds
only while gate tables are built for that restickify's read view, which is true for a node. Stage 11's
solver-placed steps need tables per read stick, and own the hazard constraint then.

For committed divisions, `restickify_residency_bar` is `_restickify_barrier` extracted. It keeps its
stricter behaviour (`SPYRE_LX_PLANNER_RELAYOUT=0` bars every restickify reader, and a double read fails
the proof) and applies when `graph.divisions_committed`, through the residency rule named `restickify`,
which Stage 3 keeps. `restickify_read_is_core_local` takes the same arguments as the old method.

**Order on one edge.** The list order of `DataEdge.steps` is physical and nothing sorts it.
`STEP_KINDS[kind].order` only places a step a decoder appends. The code allows both orders:

- **Relayout before restickify is legal.** `_restickify_read_is_core_local` accepts a source when a
  plan's destination view equals the restickify's read view (`allocator.py:1513-1521`). That view is on
  the source, and materialization inserts the copy before the restickify, which then reads the copy
  (`lx_relayout.py:2110-2114`). `solver_relayout_edge_context` admits a Pointwise consumer
  (`lx_relayout.py:929`), and a restickify is Pointwise (`torch_spyre/_inductor/padding.py:573`). The
  only test is mocked (`tests/inductor/test_work_division_hint.py:981-982`), so whether this fires on a
  real model is unknown.
- **Relayout after restickify occurs.** `_is_activation_source` (`lx_relayout.py:656-696`) allows a
  restickify's output as a relayout source. `test_attention_cache_transpose_handoff`
  (`tests/inductor/test_solver_relayout_e2e.py:1381`) needs this order: the restickify reads a graph
  input, so nothing is resident to relayout beforehand.
- **Both.** The model permits a copy into a restickify and a copy out of it at once (`chain.py` fires
  both on a hand-built P, R, C). A compile reaching it was not shown.

`to_view` indexes the device dims of the layout it applies to, so the order is not cosmetic. With a
node, the two shapes are `P -> R [RELAYOUT, RESTICKIFY]`, where Stage 7 inserts the relayout before
the pinned step with `insert_step`, and a relayout of R's output on the edge out of R. `COPY_OUT`
first agrees with today: a restickify outside the nest reads the full buffer from HBM. `COPY_OUT`
and `RELAYOUT` never share an edge. A restickify *inside* a nest with its value cut sits on two
edges here (`P -> R`, `R -> C[COPY_OUT]`). A solver-placed one could not express that, which Stage
11 has to rule out.

### Completed-reduction rows

These are additive rows behind one flag, off by default, because adding them changes the proto on any
graph with a matmul reader menu (Deviation 4). Stage 12's rollout turns it on:

```python
lx_solver_completed_reduction_routes: bool = (  # torch_spyre/_inductor/config.py
    os.getenv("SPYRE_LX_SOLVER_COMPLETED_ROUTES", "0") == "1"
)
```

With the flag on, for a source view with `partial` set, `relayout_candidates` keeps it when all of the
following hold, which mirrors the collector (`lx_relayout.py:1571-1589`): `completed_reduction_split(op,
dep, name, work_slices)` returns `r`, `cores_used == config.sencores`, and
`config.core_id_k_fast_emission`. For each destination view it replaces `core_domain_rejection` and
`grouped_gather_rejection` with `derive_completed_reduction_routes(pv, cv, r)`, which rejects by raising.
The other gates apply unchanged. The price comes from `relayout_ns(relayout_cost_features(...))` under
the same split range, skipping `movement_supported`, which rejects a partial source. No measured law
exists for this shape, so the price is the reporting path's estimate. The row sets `reduction_split=r`
and `completed_producer_cores=tuple(core for core, _ in routes)`. `FiredRelayoutGroup.plan` passes that
field through and `from_chosen` adds it to its agreement check (`lx_relayout.py:264-276`, `:294-311`). On
the new path Stage 10 derives it from `source_view`, `to_view` and `reduction_split`.
`completed_reduction_split` is `completed_reduction_split_on_buf`
(`torch_spyre/_inductor/pass_utils.py:4244-4281`) taking the split map directly; the old name calls it
with committed ownership.

### The objective

Stage 4 owns the terms, this stage owns `encode_objective` and the lowering, and Stage 7 runs the
stages and adds the locks.

```python
@dataclass(frozen=True)
class SolveStage:                  # family/ortools_encode.py
    name: str                      # "cost", or a ladder name
    expr: "cp_model.LinearExprT | None"   # None: any plan, for a constant cost
    maximize: bool = False

def encode_objective(enc: Encoding) -> None:
    """Fill enc.ortools.objective_stages and set the first stage's objective on enc.model."""
```

In order:

1. **Domains.** `cost_domains(enc)` extends Stage 5's `nodes`: `readers` from `EdgeTables.gates`, repeats
   kept; one `NodeDomain` per group; and one `RelayoutOffer` per group, in name order, as
   `allocator.py:3346-3351` adds charges. `prices_ns` is `{domain.selector_values(entries[i]): price for
   i, price in group.cost_by_source_division.items()}` over the source's menu.
2. **Priced.** `expr = graph.target.cost_model.priced_objective(graph, domains, linear_only=True)`.
   Stage 4's expression already holds the charges, as `SplitKeyedCharge`, so nothing is added here.
   If it is not `None`, `cost_expr_to_cpsat(enc.model, enc.symbols, enc.ortools.columns, expr, strict=True)`
   lowers it. `RuntimeError`, `TypeError` and `ValueError` warn and re-raise unless
   `config._cpsat_warn_on_cost_expr`, as `:886-900` does, and then go to step 3. `UnknownSymbolError`
   is a `KeyError`, so a name no encoder bound is not swallowed. On success `objective_stages` is
   `[SolveStage("cost", cost)]`, with `enc.model.minimize(cost)`, or `[SolveStage("cost", None)]` with
   no objective when `cost` is a constant, and that ends it.
3. **Unpriced.** When `priced_objective` returned `None` or lowering raised, pin every
   `relayout_literals` value to 0 before any term is built (`ilp_solver_ortools.py:1454-1455`).
4. **Ladder symbols.** In unit order, create `resident_cut_<op>` and `reads_full_<op>_<reader>` with
   `_cut_traffic_terms`'s calls, and bind `Sym.cut`, `Sym.resident_cut`, `Sym.reads_full(p, c, listing)`,
   `Sym.cores`, `Sym.shape` and `Sym.tiles`, the last from `tile_terms` in `encode_tile_counts`'s walk.
   `reads_full` is per gate listing, so a double read charges twice, as today. The proto name keeps
   today's spelling on a repeat, and the symbol takes `listing`.
5. **Ladder stages.** For each `ObjectiveStage` in `graph.target.cost_model.ladder(graph, domains)`,
   append `SolveStage(stage.name, linear_terms_to_cpsat(enc.symbols, stage.terms),
   stage.sense is Sense.MAXIMIZE)`, then set the first stage's objective on `enc.model`.

A target whose cost model has `traffic_only=True` reaches steps 3 to 5 and gets `residency` alone,
which is committed mode. A target with `cost_model=None` leaves `objective_stages` empty, for tests
of constraints only.

## Changes

| file | change |
|---|---|
| `scratchpad/lx_allocator/family/edge_tables.py` | new: tables, `Gate`, `RelayoutGroup`, `PrefetchMenu`, moved candidate code, `prefetch_menu`, `in_place_candidates`, `residency_edge`, `edge_match_pairs`, `edge_name`, `restickify_residency_bar`, `restickify_read_is_core_local`, `settle_deferred_reads` |
| `scratchpad/lx_allocator/family/ortools_encode.py` | slots filled; `SolveStage`, `encode_objective`, `cost_domains` extension; `OrToolsState.edges` filled, and `OrToolsState.copies`, `.relayout_literals`, `.incoming`, `.objective_stages` added; `OpVars` fields; `fired_relayouts`, `decision_variables`, decode helpers; the target switches |
| `scratchpad/lx_allocator/family/ortools_family.py` | `OrToolsFamily.STEPS`, `RULES` and `OPTIMIZATIONS` gain this stage's entries |
| `scratchpad/lx_allocator/check.py` | `relayout_source` gains the view comparison |
| `scratchpad/allocator.py` | moved methods become delegates; no behaviour change |
| `scratchpad/lx_relayout.py` | `RelayoutCandidate` fields; `FiredRelayoutGroup` carries `completed_producer_cores` |
| `torch_spyre/_inductor/pass_utils.py` | `completed_reduction_split` |
| `torch_spyre/_inductor/config.py` | `lx_solver_completed_reduction_routes` |
| `tests/inductor/lx_sections.py`, `lx_shadow_plugin.py`, `test_lx_encoder_sections.py`, `test_lx_live_parity.py` | see Parity and gates |
| `tests/inductor/lx_replay_graph.py` | `edge_tables_from_buffers` |
| `tests/inductor/test_lx_gate_tables.py`, `test_lx_ortools_encoder_edges.py`, their `tests/configs/torch_spyre_tests/inductor/` shard configs | new |
| `tests/inductor/test_solver_relayout_e2e.py`, `test_work_division.py`, `test_work_division_hint.py` | re-point the references named above |

Commits, each leaving tests green. Commits 4 to 8 each shrink the `project_today` drop set, so each is
gated on what it adds.

1. **Move candidate code.** `edge_tables.py` with the moved functions, delegates in the allocator,
   re-pointed tests. Stage 0 harness: empty diff on proto and plans.
2. **Add row fields.** `reduction_split`, `completed_producer_cores`, the `plan` passthrough.
3. **Tables as data.** `edge_tables.py`'s types and functions. A test compares them with the allocator's records.
4. **Units.** `settle_tables`, prefetch tables, `merge_*` literals, relayout-group units.
5. **Nests, copy-outs, guard, lifetimes.**
6. **In-place overwrite and capacity.**
7. **Gates and relayout coupling**, with `fired_relayouts`.
8. **Objective.** The drop set is now the no-overlap section alone.
9. **Restickify and the hazard**, with the `relayout_source` view comparison.
10. **Completed rows behind the flag.**
11. **Target switches.** The rows of "What the target switches on", with
    `test_target_switches_remove_variables`. `SPYRE`'s proto does not change.

## Interfaces

Consumes:

| symbol | signature | from Stage | used for |
|---|---|---|---|
| `SpyreOptimizationGraph` | `.ir`, `.nodes` (op order), `.data_edges`, `.target`, `.cuts: set[int]`, `.divisions_committed`, `.pins_on(target, what)`, `.pin(...)`, `.step(node)`, `.lifetime(node)` | 1, filled by 2 | iteration, slots, in-place adjacency |
| `Target` | `.lx_bytes`, `.alignment`, `.rules`, `.steps`, `.optimizations`, `.cost_model`; `select_target(config)` | 1, 2 | capacity, the target switches, the objective |
| `Rule`, `Optimization`, `StepKind` | `Rule.RESIDENT_TILE_UNREDUCED`, `.COPY_OUT_IN_HBM`; `Optimization.IN_PLACE`, `.PREFETCH`, `.COARSE_TILING`; `StepKind.COPY_OUT`, `.RESTICKIFY`, `.RELAYOUT` | 1 | the target switches |
| `DataEdge` | `.src`, `.dst`, `.steps`, `.mutation`, `.in_place_candidate` | 1, filled by 2 | gates, in place |
| `OpNode`, `InputNode` | `.op`, `.graph_output`, `.deferred_direct_read`, `.tick`, `.uses`, `.prefetch_view` | 1, filled by 2 | views; `ResidencyEdge` needs the op |
| `RestickifyNode`, pre-pass step | `Step(kind=StepKind.RESTICKIFY, to_stick=s, realized_by=node, pinned=True)`, `Pin(edge, "restickify", s, "pre-pass")` | 1, 2 | gates, `forced` |
| `Step` of kind `RELAYOUT`, `STEP_KINDS` | `to_view`, `reduction_split`, `source_view`, `source_footprint_bytes`, `destination_footprint_bytes`; list order physical; `STEP_KINDS[kind].order` | 1 | check |
| deferred-read pin | `Pin(node, "split", WorkSplitPin(split, reason), "hazard")`, `work_split_from_division(node, division)` | 1, 2 | rejected read |
| `carry_pairs`, `CarryPair` | `(graph) -> list[CarryPair(storage, reader, role)]` | 2 | CARRY gates |
| `ALWAYS`, `RuleTag`, `TABLE_COLUMN`, `Rule`, `RuleViolation`, `check_decisions` | `ALWAYS` checks `resident_read`, `relayout_source`, `cross_frame`, `in_place_conditions`, `tiles_line_up`, `pinned_steps_stay`, `boxes_fit_lx`, each a tag as its name string; `TABLE_COLUMN`, a tag with no check; `Rule.RESIDENT_TILE_UNREDUCED`, `Rule.COPY_OUT_IN_HBM`; `check_decisions(graph)` | 3 | provenance, the added comparison |
| `residency_reason` | `(node, graph, *, decides_relayouts, tileable) -> str \| None` | 3 | barred producers, prefetch eligibility |
| residency rule | `restickify` still present, calling the delegate | 3 | committed mode |
| `Sym` | `.cut(n)`, `.resident_cut(n)`, `.reads_full(p, c, listing=0)`, `.restick(edge_name, row=None)`, `.cores(n)`, `.shape(n)`, `.tiles(n)` | 4 | symbol binding |
| `SpyreCostModel`, `ObjectiveStage`, `Sense` | `graph.target.cost_model`; `priced_objective(graph, domains, *, linear_only) -> Expr \| None`; `ladder(graph, domains)`; `traffic_only` | 4 | `encode_objective` |
| `CostDomains`, `NodeDomain`, `RelayoutOffer`, `domains_from_buffers` | `(nodes, relayouts=())`; `selector_values(candidate)`; `(buffers) -> CostDomains` | 4 | `cost_domains`, its parity |
| `cost_expr_to_cpsat`, `linear_terms_to_cpsat`, `UnknownSymbolError` | `(model, sym_map, buffer_map, expr, *, strict=False)`; `(sym_map, terms)` | 4 | lowering |
| `OrToolsFamily`, `Encoding`, `OrToolsState`, `Provenance` | `encode(graph) -> Encoding`; `post`, `post_in`, `post_table`, `symbol`, `tag`, each taking `enc` first; `Encoding.model` (`.proto`), `.provenance`, `.symbols`, `.ortools`; `OrToolsState.ops`, `.units`, `.columns`, `.tables`, `.tile_terms`, `.edges` | 5 | everything |
| encode sequence, functions in `family/ortools_encode.py` | `begin(graph) -> Encoding`, `build_tables(enc)`, `encode_ops(enc)`; `settle_tables`, `encode_op_vars`, `add_unit(enc, table, *, size_units, boundary) -> OpVars`, `declare_in_place_vars(enc, v)`, `menus`, `encode_loop_columns`, `post_residency`, `encode_tile_counts`, `bind_symbols`, `encode_sticks`, `cost_domains`, `decision_variables`, and the slots, each `(enc)` | 5 | proto order |
| `OpVars` | `.table`, `.name`, `.units`, `.in_buffer`, `.offset`, `.eff_size`, `.division`, `.division_is(i)`, `.cores`, `.core_cost`, `.loop_id`, `.tiled`; this stage adds `.merge_vars`, `.cut`, `.via_full`, `.nest_end`, `.nest_start`, `.end`, `.start`, `.relayout_reads`, `.serves` | 5 | every constraint |
| `DivisionTable` | `.node`, `.entries`, `.enumerated`, `.tileable`; tables cover non-input nodes | 5 | menus |
| `stick_of`, `read_stick` | `(port, graph)`, `(edge, graph)`; the second walks the edge's steps | 5 | `forced` |
| `build_today`, `project_today`, `section_diff`, `replay_graph`, `lx_shadow_plugin` | Stage 5's helpers; `build_today` gains `cost_expr=None` here | 5 | gate |
| harness | `lx_gate.sh`, tiers `intree`, `fake`, `unit-*`, `failure`, `FIXED_GRAPHS`, `corpus_coverage` | 0 | gate |
| `tests/scripts/lx_check_summary.py` | `<record file>` | 3 | rerun |

Provides:

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `build_edge_tables`, `EdgeTables`, `Gate`, `GateKind`, `PrefetchMenu` | above | 8, 9, 11 | final; Stage 11 adds a stick key to `pairs` |
| `RelayoutGroup` | above; `name` is the box key | 13 | final |
| `settle_deferred_reads` | above | 8 | final |
| `prefetch_menu`, `in_place_candidates`, `residency_edge`, `edge_match_pairs` | `(input_node, tables) -> PrefetchMenu \| None`; `(node) -> list[str]`, today's order; `(gate) -> ResidencyEdge \| None`; `(gate) -> tuple[tuple[int, int], ...]` | 8 | final |
| `RelayoutCandidate.reduction_split`, `.completed_producer_cores` | defaulted fields | 7, 10 | final |
| `SolveStage`, `OrToolsState.objective_stages` | `SolveStage(name, expr, maximize=False)`; `name == "cost"` for the priced stage; `expr=None` for a constant cost; ladder names `residency`, `cut tiebreak`, `parallelism`, `division shape`, `tile count` | 7 | final |
| `OrToolsState.relayout_literals`, `.copies`, `.edges` | `dict[str, literal]`, group name to residency; `dict[str, OpVars]`; `EdgeTables`; reached as `enc.ortools` | 7 | final |
| `fired_relayouts`, `FiredRelayout` | `(enc, value: Callable[[Any], int]) -> list[FiredRelayout]`; `gate`, `group`, `row`; raises unless exactly one row | 7 | final |
| decode helpers | `joined_literal(enc, before: OpNode)`, false where `before.tick` goes in `graph.cuts`; `in_place_literal(enc, edge: DataEdge) -> literal \| None`; `copy_out_literal(enc, node: OpNode)`; `via_full_literal(enc, edge: DataEdge)`; a literal or `False` where not stated; `prefetch_view(enc, node, value) -> PerCoreView` | 7 | final |
| `decision_variables` | `(enc) -> list`: residency, `merge_*`, served and division, as `ilp_solver_ortools.py:805` | 7 | final |
| `encode_in_place`, `OrToolsState.incoming`, `OpVars.end`, `.start`, `_encode_lifetimes` | in-place loop, `add_at_most_one` before `encode_placement`; `dict[str, list[literal]]`; `_Slots` values | 7 | final |
| `_post_shared_offset`, `_encode_capacity` | above | 7 | temporary, removed in Stage 7 |
| `encode_restickifies`, its `Sym.restick` binding | `Sym.restick(edge_name(edge))`, a bool in Phase 2 | 11 | final; 11b changes the raise |
| `restickify_residency_bar` | `(ir, name, uses, *, plans=()) -> str \| None` | none; the `restickify` residency rule calls it | final |
| `restickify_read_is_core_local` | `(ir, name, reader, *, plans=()) -> bool` | 11 | final |
| `_loop_carry_update_edge` | moved to `family/edge_tables.py` | 10, 13 | final |
| `_table_owners` | `(enc) -> Iterable[OpVars]`, `ops` then `copies` | 7 | final |
| `OrToolsFamily.STEPS`, `.RULES`, `.OPTIMIZATIONS` entries | `COPY_OUT`, `RESTICKIFY`, `RELAYOUT`; `RESIDENT_TILE_UNREDUCED`, `COPY_OUT_IN_HBM`; `IN_PLACE`, `PREFETCH`, `COARSE_TILING` | 3 (`coverage`), 7 | final |
| `edge_tables_from_buffers` | `(buffers, graph) -> EdgeTables`, in `tests/inductor/lx_replay_graph.py` | 7, 13 | final; replay arm |
| config `lx_solver_completed_reduction_routes` | bool, default False | 12, 13 | final; Stage 12 flips the default |

## Tests

`tests/inductor/test_lx_gate_tables.py`. Compile-backed cases use `fake` compiles of the fixed set, in
process, with no card.

- `test_tables_equal_the_allocator_records`: per reader, `Gate.pairs` equals `cd_parent_matches`,
  `Gate.relayouts` equals `cd_parent_relayouts` field by field, group ids equal.
- `test_gate_listing_follows_read_deps`: `a + a.permute(...)` gives two listings, `a * a` one, and
  `x[..., :8] + x[..., 8:]` two sharing one `ResidencyEdge`.
- `test_prefetch_menu_matches_allocator`: divisions, views and `(k, j)` pairs match.
  `test_a_broadcast_read_gets_no_prefetch_row`.
- `test_carry_gate_has_no_relayouts`: built from `carry_pairs`; the sibling-pin order stays (carries
  merged before capping).
- `test_in_place_candidates_match_the_allocator`: `EdgeTables.in_place` and `in_place_candidates` against
  `in_place_parents`, in order, with an in-place edge from an LX prefetch and an input with no `PrefetchMenu`.
- `test_residency_edge_and_match_pairs`: they return the gate's edge and pairs.
- `test_tiled_entries_have_no_rows`: menu tilings give no relayout rows, with
  `lx_same_tile_group_relayout` on and off.
- `test_same_tile_group_rows_scale_by_trip_count`: moved from the e2e assertion at
  `test_solver_relayout_e2e.py:864`. `test_group_ids_follow_consumer_order`.
- `test_check_decisions_rejects_a_relayout_from_the_wrong_division`: on a `fake` graph, no tables.
- `test_deferred_read_rejection_rebuilds_tables`: interning restarts, `pair_costs` persists, and the
  rejected node holds a `split` pin and one entry. `test_settle_with_no_relayout_sources`: Stage 8's call.
- `test_completed_rows_are_off_by_default` and `test_completed_rows_when_enabled`: with the flag on, a
  2-way K split on 32 cores gets a row whose `completed_producer_cores` is the last core of each K group,
  equal to `derive_completed_reduction_routes`.

`tests/inductor/test_lx_ortools_encoder_edges.py`, device-free, hand-built graphs through the encoder and a plain
`CpSolver`. Hand-built graphs have no IR, so a derived target with a stub cost model supplies the
expression.

- Ports of `tests/inductor/test_solver_relayout_decision.py`:
  `test_relayout_fires_when_cheaper_than_the_spill`, `test_relayout_declines_when_dearer_than_the_spill`,
  `test_fallback_objective_never_decides_a_relayout`, `test_copy_never_resides_without_its_source`,
  `test_gate_without_a_relayout_table_is_unchanged`, `test_two_consumers_on_one_view_share_one_copy_and_one_charge`,
  `test_a_consumer_reads_the_copy_only_under_a_priced_pair`. The cases needing no-overlap move in Stage 7.
- `test_no_priced_objective_pins_relayouts_out`, `test_lowering_failure_pins_relayouts_out`,
  `test_unknown_symbol_is_not_swallowed`, `test_constant_cost_is_a_stage_with_no_expr`.
- `test_ladder_stage_names_and_senses`: `objective_stages` per path, first objective set on the model.
- `test_offer_prices_key_by_selector_values`: `RelayoutOffer.prices_ns` indexes back to
  `cost_by_source_division` through `index_charge`.
- `test_double_read_binds_two_reads_full_symbols`: two listings, `listing` 0 and 1, today's proto names.
- `test_every_constraint_has_one_provenance`: over tiled and priced models, printer helpers included.
- `test_fired_row_survives_a_shuffled_menu`: reverse each menu, re-encode, solve. The decoded `to_view`,
  source view and `reduction_split` are unchanged. This is the regression test for the `fired()` bug.
- `test_fired_relayout_raises_on_a_row_mismatch`: corrupt a row's `source_view`; `RuleViolation`.
- `test_fired_relayout_raises_on_two_copies`, `test_fired_relayout_raises_on_a_spilled_copy`: under `-O` too.
- `test_cut_is_not_the_copy_out`: chain a, b, c with b untiled. `joined_literal` puts the ticks of a
  and b in `graph.cuts`, and `copy_out_literal(enc, b)` is `False`.
- Ports of the nest cases from `tests/inductor/test_scratchpad_solver.py`:
  `test_cut_stage_splits_a_pair_the_table_does_not_match` (`:1652`),
  `test_cut_stage_joins_a_matched_pair_with_different_specs` (`:1731`),
  `test_a_cut_tile_resides_only_as_its_copy_can_read_it` (`:1828`), which proves the guard,
  `test_a_buffer_read_by_a_nest_lives_until_the_nest_ends` (`:1855`).
- `test_restickify_reader_needs_a_slice_a_relayout_or_hbm`: P resident, R a `RestickifyNode`. With a match
  pair, P resides. With neither, P spills. With a relayout row, P resides and the copy is served. With a
  tiled P cut, P resides through `via_full`. This is the claim "a restickify may not read a core-sliced
  resident source unless a relayout comes first".
- `test_restickify_rule_posts_nothing_new`: the same graph with a `PointwiseNode` in R's place has an
  equal proto, and only the rule on R's gates differs, `"cross_frame"`.
- `test_restickify_residency_bar_matches_the_old_barrier`: runs the cases at
  `tests/inductor/test_work_division_hint.py:978-982`.
- `test_unpinned_forced_restickify_raises`, `test_in_place_excludes_via_full`,
  `test_in_place_at_most_one_each_way`.
- `test_target_switches_remove_variables`: over targets derived from `SPYRE`, each row of "What the
  target switches on" adds or removes exactly the variables it names, and `SPYRE` itself gives the
  same proto as the default build.

Existing suites, serial: all of the parent plan's unit list. `test_solver_relayout_decision`,
`test_solver_relayout_candidates`, `test_work_division*`, `test_coarse_tiling`, `test_scratchpad_solver`
guard the move. Device suites, because the parent plan lists Stage 6: `test_scratchpad_use`,
`test_work_division`, `test_solver_relayout_e2e`, `test_solver_auto_coarse_tiling`,
`test_lx_planning_suite`. They run old solvers on moved code, so they prove commit 1. Nothing in them
reaches the encoder until Stage 9. `make check-all-configs` must pass.

## Parity and gates

Parent plan: "Through Phase 1, and in Phase 2 wherever the rules agree with today's, the CP model stays
byte-identical and the committed plans stay identical. Any other difference is listed by rule in its PR
and either justified or fixed." Stage 6 has no "proven by" line of its own.

**Production unchanged.** The Stage 0 harness between this PR's parent and tip: `intree`, the `fake` zoo
at default and `AUTO_COARSE_TILING=1`, `unit-*` and `failure`, empty on proto, decisions and plans. Each
commit runs `intree` and the fixed set. A model `fake` cannot reach stays card-only and is listed in the
PR. `fake` inputs use default layouts, so this proves parity, not production plans.

**Proto gate.** There is no second comparator. Stage 5's section diff continues with a smaller drop set:

- **A:** `project_today(build_today(solver.buffers, solver.limit, solver.alignment, cost_expr=cost_expr))`.
  `build_today` now stops at the first `_solve_and_record`. The drop set is the `_add_no_overlap_2d`
  range, by its checkpoints, variables and constraints. `project_today` renumbers `objective` as well.
  `vars`, `literals` and `enforcement_literal` stay the only reference fields: `element`, `table` and
  `lin_max` hold `vars` inside their expressions on OR-Tools 9.15 (`exp/stage-06/fields.py`).
- **B:** `OrToolsFamily().encode(build_graph(ir, select_target(config)))`, first-stage objective
  included, `encode_placement` posting capacity only. It runs on live compiles through `lx_shadow_plugin`, at each
  joint solve, on the `fake` zoo at both configs and the unit suites that hold a `GraphLowering`.
- **Pass:** variable strings, constraint strings and the objective are equal, lists in order, for every
  such solve. The corpus has both priced and ladder solves. `cost_domains` also equals
  `domains_from_buffers(solver.buffers)`.

Stage 7 empties the drop set, and the gate becomes the whole proto.

**In CI.** `test_lx_live_parity.py` extends to these sections on `fake` compiles of the fixed set, at
default and tiling. The replay arm in `test_lx_encoder_sections.py` keeps `load_captures`,
`synthetic_graphs()` and the hand-built cases: `edge_tables_from_buffers` feeds `cd_parent_matches` and
`cd_parent_relayouts` to `encode_read_gates` over `replay_graph`, through a test subclass's
`settle_tables`, and stops at `encode_tile_counts`. It tests emission, not the tables, because those
nodes have no IR. Captures are not a route to the encoder.

**Reported in the PR.** Hits per `corpus_coverage` feature, restickify features included. A zero on a
restickify reading a resident source means the hazard was not exercised. `lx_check_summary.py` is rerun
on the `fake` records.

Not byte-gated:

- **Committed mode.** `plan_layout` builds a different model with no gates, so its proto cannot match an
  all-pinned graph. It goes to the Phase 2 exit, on traffic.
- **Plans.** Nothing solves B with placement until Stage 7, so decision parity starts there.
- **`lx_solver_completed_reduction_routes=1`.** Rule-listed difference. Run once on the corpus with
  `check_decisions` clean. Stage 12's rollout owns the comparison.

One difference is expected and listed: where legacy printing fails on a foreign symbol it leaves helper
variables in A, and `priced_objective` returns `None` before printing, so B lacks them. Any other is
listed by rule, or fixed.

## Risks

- **The graph disagrees with the buffers.** A hand port from a graph built without the allocator, in exact
  creation order, has never been compared on a real compile. Stage 5's commit 7 (P4) is the first section
  diff on the `fake` zoo and has not been run. Node, prefetch or menu order differing there changes this
  design before it freezes.
- **Proto order drift.** One helper out of place shifts every later index. Caught by the gate, by name.
- **A ladder stage through sympy changes the proto.** Stage 4 says it does not. Caught by the gate, and
  only then does a CP copy of the terms return.
- **A hash-order dependency.** `ReadWrites.reads` is an `OrderedSet` (`torch/_inductor/dependencies.py:433`),
  but `rw.reads | rw.writes` and the name sets around it are not ordered by contract. Caught by running the
  gate under the hash seeds in `tests/inductor/test_scratchpad_solver.py:1009`.
- **Fake tensors.** `Gate.residency` holds ops. Nothing keeps a table on the graph or in
  `Encoding.provenance`, and the harness must not pickle the encoding's model. Caught by a test that
  pickles `Encoding` metadata and the graph.
- **The rebase race.** Main keeps changing `_cd_parent_relayouts`. The move in commit 1 goes first so later
  fixes land in one body. `test_lx_live_parity.py` fails in CI when the model changes without the encoder.
- **#5695 lands first.** `settle_deferred_reads` then has to call `divide_deferred_op` before
  `_fixed_core_division`, as that PR does for rejected direct reads. Caught by
  `test_deferred_read_rejection_rebuilds_tables`.
- **Completed rows.** Each reduction-split entry can add rows and copies, and copies dominate CP-SAT's model
  size (`sa-vs-cpsat-REPORT.md`). Not measured. The price is an unfitted estimate, so the solver may pick a
  route the hardware does not reward. The flag stays off until Stage 12's comparison and device test.
- **Group sharing across a restickify and a plain reader.** One `RELAYOUT` step sits on both edges, and
  Stage 10 must insert it once. Caught by `test_one_source_two_consumers_share_one_relayout`
  (`tests/inductor/test_solver_relayout_e2e.py:596`) once Stage 10 replaces the commit path.
- **A read that legitimately differs in stick with no restickify.** The raise would be a false alarm, for
  instance on a broadcast dim. Caught by the gate. The fix would belong in `read_stick`, not in the raise.

## Deviations from the parent plan

1. **Pinned restickify as a step.** Plan: "A restickify inserted before the graph is built is a pinned
   step." Code: a restickify op has its own menu, residency, tick and bundle cost (`allocator.py:3737-3800`),
   and byte identity needs its variables. Resolution: a `RestickifyNode` plus
   `Step(kind=StepKind.RESTICKIFY, pinned=True, realized_by=node)` on the edge. Commit inserts nothing
   for the step.

2. **"Today's `_restickify_barrier` enforces this."** Only on the placement path (`allocator.py:847-855`).
   The joint path uses the generic gate (`ilp_solver_ortools.py:534-597`), and the barrier's docstring says
   calling it there would be wrong (#4655, `allocator.py:1454-1461`). Resolution: no new joint constraint;
   the gate carries `"cross_frame"`, and the barrier stays for committed divisions. So
   `SPYRE_LX_PLANNER_RELAYOUT=0` keeps not barring restickify readers on the joint path, even though the
   comment at `torch_spyre/_inductor/config.py:179-180` reads as if it did. It does narrow what is
   representable there, through the ownership proofs at `torch_spyre/_inductor/pass_utils.py:3766`, `:3898`.

3. **"Relayouts are allowed inside a tile when both ends share a nest."** Code: menu tilings never get a row
   (`allocator.py:5130-5132`, `:5174`, `:5196`, and `_views_for_divs`, `:5337-5341`). Only ops already in the
   same `CoarseTileInfo` group do, behind `lx_same_tile_group_relayout`, default off
   (`torch_spyre/_inductor/config.py:488`, `lx_relayout.py:906-919`). And the solver cannot fire a relayout
   under `AUTO_COARSE_TILING=1`; the greedy re-plan after a failed solve still can. Resolution: not built in
   this plan. Stage 6 keeps today's rows. Same-nest relayouts on menu tilings go to the parent's Future work;
   they need the chosen tiling's trip count and a priced tiling objective.

4. **"Including completed-reduction routes."** Code: the joint path has none today (`allocator.py:5089-5094`).
   Adding them changes protos, and they need a price: `movement_supported` rejects a partial source
   (`lx_relayout.py:463`). Resolution: rows behind `lx_solver_completed_reduction_routes`, default off.

5. **"A tiled op's live range starts at its nest's start."** Code: `nest_start` only bounds an LX prefetch's
   `start` and that never binds (`ilp_solver_ortools.py:1247-1255`). A tiled op's box starts at its own write
   slot (`:1710`, `:1724`). Resolution: keep today's, for parity. `check_decisions` and Stage 7's boxes use the
   write slot.

6. **Copy-out with no edge.** Plan: a copy-out is a step on a data edge. Code: a tiled graph output, or an op
   read only by an extern kernel, is cut with no `via_full` edge (`ilp_solver_ortools.py:982`, `:1013-1015`).
   Resolution: the decision stays in the encoder, node-keyed by `copy_out_literal` and `Sym.cut`. `read_back`
   adds no step, its traffic is priced by node, and `verify_copy_outs` skips such producers.

7. **Gates are not data edges.** Plan: "Loops: not modelled." Code: today posts per `parents` listing
   (`utils.py:276`), and carries post match-only gates between storage and each update and read
   (`allocator.py:4326-4375`), which are in the proto. Resolution: `build_edge_tables` makes one gate per
   listing, READ ones linked to data edges, and each `carry_pairs` pair a CARRY gate.

8. **Stage 7 owns placement, but placement sits mid-proto.** Code: `off_`, the shared offset and capacity
   precede `_add_core_division`. Resolution: Stage 6 posts the offset equality and capacity behind helpers
   and gates with the no-overlap range dropped. Stage 7 gates the whole proto.

9. **"The partial-reduction cut guard is an always-on rule."** Its rule, "a copied-out tile that
   stays resident has no reduction split", is the card rule `Rule.RESIDENT_TILE_UNREDUCED` in the
   parent's own `Rule` list. Resolution: the guard is posted only when `graph.target.rules` holds that
   flag, and is tagged with it. `SPYRE` holds it, so the proto does not change.

## Open questions

None.

## Out of scope

- Objective terms, `SplitKeyedCharge` and the printer: Stage 4.
- `no_overlap`, boxes, packing, `Solution`, `read_back`, running the objective stages and their locks: Stage 7.
- SA relations: Stage 8. `optimize`, failure handling, the temporary commit bridge: Stage 9.
- Inserting steps, `_push_allocation`, shared-step insertion, deriving `completed_producer_cores`: Stage 10.
- Free sticks, gate tables per stick pair, `restick_resident`, a solver-placed `RESTICKIFY` step's
  division, box and tick, and the hazard constraint for it: Stage 11. This stage leaves it the literal and the rule tag.
- Turning `lx_solver_completed_reduction_routes` on: Stage 12.
- Deleting the old model code and the delegates: Stage 13.
- Relayouts inside menu tilings, priced tiling objectives: the parent's Future work.
