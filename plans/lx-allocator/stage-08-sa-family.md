# LX allocator, Stage 8: `SAFamily`, today's SA co-optimizer as a family

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 1, 2, 3, 4, 5, 6, 7. Needed by: Stages 9, 10, 11, 12, 13.

## Goal

When this PR merges, the graph can be solved end to end by a second family with no CP-SAT in
it: `SAFamily.encode` builds an `AnnealModel`, inside an `Encoding`, from a
`SpyreOptimizationGraph`, `SAFamily.solve` anneals it with today's moves, and
`SAFamily.read_back` decodes the plan onto the graph. With seed 0, in one process, the plan
and the score equal what `SaCoOptimizingSolver` chooses on the same graph, apart from one
listed tiling refusal.
Nothing in production calls the family yet: `select_allocator` is untouched,
`LAYOUT_SOLVER=simulated_annealing` still runs the old path, and Stage 9 wires the family
in. The one production change is mechanical: the search moves out of `sa_cooptimizer.py`
into a class both paths share, so old and new cannot drift while both exist.

## Today

A citation written `:N` is in the file cited just before it. In this section that is
`sa_cooptimizer.py` unless the sentence names another file.

### The engine

`SaCoOptimizingSolver` (`torch_spyre/_inductor/scratchpad/sa_cooptimizer.py:686`) anneals
the pair `(pi, W)`. `W` is `self.chosen`, one `DivisionConfig` per buffer. `pi` lives in
a composed `NativePermutationLayoutSolver`. Position `i` in `self._bufs` is the index
used by `chosen`, by the packer and by the objective (`sa_cooptimizer.py:701-705`), so
buffer order is part of the search's identity.

| today | where | maps to |
|---|---|---|
| state `(pi, W)` | `packer`, `chosen`, `_n_eligible` (`:1701-1718`) | `AnnealState`, owned by `SAFamily.solve` |
| per-buffer candidates | `_sources[i]` (`:1104-1126`) | `AnnealModel.domains` |
| per-edge verdict and propagation | `_children`, `_relations` (`:904-932`) | `AnnealModel.relations` |
| buffer index order | `self._bufs` | `AnnealModel.order` (fixed index order, not `pi`) |
| residency gate | `_eligible` (`:1167-1192`) | stays in the search |
| moves | `_execute_move` (`:1747-1771`), `_step_reorder` (`:1817-1876`) | `SAFamily.solve`, unchanged |
| score | `_score` (`:1423-1454`), `_build_score_fn` (`:807-857`) | `SAFamily.score` |
| loop | `_anneal` (`:1897-1934`), `_calibrate_temperature` (`:1775-1789`) | `SAFamily.solve`, unchanged |

The loop is one geometric cool over `min(200 * n, 50_000)` steps (`:94-100`), where `n`
is `len(self._bufs)`, pins included, despite the docstring. The moves are reorder 0.5,
flip 0.3, recolor 0.2 (`:106`), with reorder withdrawn once every eligible buffer is
resident (`:1722-1737`). `cooling_schedules.py` is not used by this engine. Only
`SimulatedAnnealingLayoutSolver` uses it (`simulated_annealing.py:53-56,198`), and that
class has a different objective, move set and schedule.

### What the division sources and relations hold

| class | holds | keys on | Graph-side source |
|---|---|---|---|
| `DivisionConfig` (`:171`) | a `CoreDivision`, `menu_index`, a derived `key` | `_canonical_key`: sorted splits, reduction symbols, tiling (`:132-148`). A repeated split map in a clone's menu adds the menu index | Private to the family. Menu positions never reach the graph |
| `_MenuDivisions` (`:334`) | `configs` in menu order, `by_key` | position, then key | the node's division table |
| `_GeneratedDivisions` (`:378`) | `OpSplitSpace` (live op, `WorkDivisionContext`, `TilingSpace`), `seed_config` | `_canonical_key`, memoized | the node's live op |
| `_TableRelation` (`:547`) | `pairs` over `_split_key`s, `down`, `up` | split key, tiling dropped (`:151-168`). The lowest menu position wins (`:596-609`) | an edge's match pairs |
| `_ViewRelation` (`:614`) | `ResidencyEdge`, both sources, `carries`, memos | split key for the verdict, whole key for propagation (`:632-637`) | the live ops at both ends |

`_build_sources` picks `_GeneratedDivisions` wherever `division_space` is set (`:1125`).
The allocator sets it for every enumerated op, whether or not tiling is on, because
`generates_divisions()` is always true here (`allocator.py:4479-4486`,
`sa_cooptimizer.py:719-721`). So at default config every pointwise and reduction op
generates, and `_MenuDivisions` serves only input clones (LX prefetches), pinned ops and ops outside the
enumeration. `_edge_relation` uses `_ViewRelation` only when both ends generate and the
consumer has a `residency_edges` entry, and `_TableRelation` otherwise (`:1071-1102`).
Those fields hold FakeTensor references and get blanked before pickling, so **a replayed
capture runs menus and tables only**. All 25 buffers of the `sdpa` capture have no
generated source and no view relation (measured). The test fixtures exercise a different
trajectory than a live compile does.

How today's edges relate to the graph:

- `CoreDivisionBuffer.parents` is `[dep.name for dep in rw.reads]` (`utils.py:267,276`), and
  `reads` deduplicates equal deps. So `x * x` is one entry and one data edge. Two different
  indexes of one buffer, as in `x[..., :8] + x[..., 8:]`, are two entries, one `ResidencyEdge`
  holding both reads, and two counts in `_companion_charges` (`sa_cooptimizer.py:1387-1394`).
  A Stage 6 `Gate` is one such entry, and a `DataEdge` stands for one distinct read dep.
- A parent outside the solver is skipped (`:911-914`). So an `InputNode` with no clone has
  no relation.
- A loop carry adds parents and pairs that are not reads (`allocator.py:4329-4375`), so no
  data edge stands behind them. A `Gate` of kind `CARRY` holds them.
- A drained collective's edge is removed (`allocator.py:4288-4292`).
- A pre-pass restickify is an op with its own menu and buffer. On the graph it is a
  `RestickifyNode` in `graph.nodes`, with one data edge on each side, so it is an ordinary
  site.

### The seed, and how it fails

The seed is `configs[0]` for every buffer (`:1128-1131`). The docstrings at `:276`, `:798` and
`:1129` call this the committed division, but it is not. `_enumerate_core_divisions` keeps the
order of `enumerate_work_division_candidates` (`allocator.py:3941-3955`), an `itertools.product`
over ascending divisors (`work_division.py:195,1073-1077`). The first legal entry is the lowest
factors: one core unless a span floor or the span cap rejects it (`work_division.py:911`). Only
pinned ops and non-enumerated ops seed at a committed division. The fixture menus are ordered differently: 21 of 25
`sdpa` capture buffers have 32 cores at index 0 (measured).

`pi` comes from `FirstFitLayoutSolver` over per-core sizes, converted by
`SolverToPermutation` (`:1270-1286`, `simulated_annealing.py:69-95`): sorted by address,
unplaced buffers after, ties by index. FirstFit lays out buffers that are transiently
ineligible too, so they keep a slot, and leaves the fixed pins unplaced.

FirstFit does not fail on capacity. A buffer that does not fit keeps `address=None`
(`firstfit_bestfit_solver.py:257-265`). It can only raise:

- `_check_in_place_relationships` raises `ValueError` on a bad handoff or a write-only
  parent (`plan_solver.py:938-967`). The size clause is defused by dropping oversized
  pairs for the seed (`sa_cooptimizer.py:1275-1281`).
- An `assert` fires on a cycle (`firstfit_bestfit_solver.py:76`).
- `IndexError` on empty `uses`.

The native constructor repeats the handoff checks
(`torch_spyre/csrc/perm_layout_native.cpp:250-263`), so
a fallback seed inside the search would hit the same errors unless it also dropped the
offending pairs. `_assert_unsized_buffers_are_pinned` (`:861-883`) is the other hard
stop. `Unsupported` for an op with no legal candidate is raised before any solver runs
(`allocator.py:3801-3802`).

So postmath's statement holds with one precision: the seed is always a valid plan as long
as the first table entries are legal, and at worst that plan spills everything.

### Is every visited state legal?

For divisions, yes. Each value comes from a menu, from `OpSplitSpace.admits`
(`work_division.py:1285-1291`), or from a relation that checks `admits` inside the
inversion (`:1654-1660`). For residency, yes, but not because the domains and relations
make it so. A state is legal because after every move `_atomic_flip` and
`_apply_assignment` recompute `_eligible` for the op and its parents and push the result
into the packer's eligible mask (`sa_cooptimizer.py:1465-1479,1635-1659`). An illegal
pairing is reachable in `W` and is legal because the producer is transparent to the
packer (`perm_layout_native.cpp:648-655`). For placement, yes: the packer only stacks, and
an evicted buffer evicts every later overlapping one (`:529-531`, measured).

With tiling on, the code relies on more than that:

- `apply_tilings` raises `Unsupported` on a region op (`coarse_tiling.py:684-701`).
- `check_priced_footprints` raises (`allocator.py:2947-2996`).
- `commit_divisions` raises on a split the tile squeezed away (`allocator.py:2907-2909`).
- `commit_lx_views` raises if the post-solve judge rejects a resident buffer
  (`allocator.py:3027-3029`), with or without tiling.

Those are solver-free checks after the solve. Nothing is priced as a penalty: companion
traffic is a real cost (`sa_cooptimizer.py:1325-1404`).

### The score

`_build_score_fn` is not integer arithmetic. It lambdifies the whole expression with
`modules="math"`, evaluates it in doubles, and rounds once:
`to_fixed_us(max(0.0, ns) / 1000.0)` (`:849-855`, `utils.py:952-963`). The claim in
`docs/source/compiler/sa_co_optimization.md` that there is "no float accumulation to
reorder" is true of the sum of integers around it, not of the expression inside. `_score`
then adds `_companions`, rounded separately and outside the expression
(`sa_cooptimizer.py:1441-1447`), or uses the memory-only fallback when `_score_fn` is
`None` (`:1449-1454`). The fallback fires when there is no expression, on an
unrecognized symbol, or on a lambdify error (`:846-851`), and
`test_unrecognized_free_symbol_falls_back_to_memory_only` pins it.

Measured on a synthetic 200-buffer expression: a rename that reverses the sort order of
the symbols changes the double on 80% of evaluations and the fixed-point score on 24 of
20,000. A rename that keeps the order changes nothing. Summing the same terms in another
order gives an expression that is not `==`, because `Float` constants accumulate
differently. One unit of score matters: `delta <= 0` short-circuits the RNG draw
(`:1869,1889`), so a difference at a tie shifts every later draw.

### Rule data that depends on the family

- `_prefix_gather` lifts the "partial/offset read" bar only for a solver that decides
  relayouts (`allocator.py:905`).
- `_tiling_candidates` returns untiled for SA (`:4002`), so `tileable` is empty and
  `_untiled_residency_bar` becomes a fixed `residency_reason`
  (`allocator.py:938`, `4187-4194`).
- The pricing view depends on `relayout_sources` (`:3211`), and through it so does the
  deferred-read rebuild in `_prepare_buffers` (`:3152-3179`).
- `select_allocator` gives SA no post passes (`:5432-5435`). So with
  `LAYOUT_SOLVER=simulated_annealing` and every other setting at default,
  `LxContextSwitchingPass` does not run, although `enable_lx_context_switching`
  (`config.py:56`) has turned the extern guard off (`allocator.py:857-862`). A buffer live
  across a single-output, risky `FallbackKernel` is exposed. A default `cpsat` compile is
  not, and a clobber has not been reproduced on a card. The family does not inherit this,
  because `commit(graph)` runs the post passes. This stage's gate compares solves,
  so it does not see the difference.

### What not making relayouts means

`decides_lx_relayouts()` is false (`sa_cooptimizer.py:716`), so the allocator builds no
`RelayoutCopyBuffer` and the objective has no `RelayoutCharge`. `sym_division` binds
`menu_index`, which is `None` for a generated config, and that is sound only because no
such term reaches SA (`:828-836`). The gate compares SA with SA, so it is unaffected. As a
fallback it matters a lot. `sa-vs-cpsat-REPORT.md` has SA at a geomean of 1.117x CP-SAT's
predicted cost on the zoo (max 1.637x) and 1.012x on Granite (max 1.038x). Relayouts are also
most of CP-SAT's extra time, so the graphs that time out are the ones with the most to lose.

Stage 9 owns `config.fallback_solver`, default `""`, and Stage 12 flips it once Gate 4's
forced-failure report exists.

## Design

### Module layout

Everything goes under `torch_spyre/_inductor/scratchpad/lx_allocator/family/sa/`.

| old | new home | move or copy | who deletes the old |
|---|---|---|---|
| `DivisionConfig`, `_canonical_key`, `_split_key`, `_one_axis_apart` | `divisions.py` | move, re-exported from `sa_cooptimizer` | Stage 13 removes the re-export |
| `_DivisionSource`, `_MenuDivisions`, `_GeneratedDivisions` | `divisions.py` as `DivisionSource`, `MenuDivisions`, `GeneratedDivisions` | move, old names aliased | Stage 13 |
| `_EdgeRelation`, `_TableRelation`, `_table_relation`, `_ViewRelation` | `divisions.py` as `EdgeRelation`, `TableRelation`, `table_relation`, `ViewRelation` | move, old names aliased | Stage 13 |
| search methods, `_precompute_topology` down to `_anneal`, except the rows below | `search.py`, class `AnnealSearch` | move | n/a |
| `_lifetime_buffers`, `_build_seed_packer` | `search.py`; the FirstFit call becomes `_seed_permutation(sizes)` | move | n/a |
| body of `_score`; `_build_score_fn` after Stage 4 moved its kernel to `utils.compile_fixed_point` | `score.py`: `fixed_point_score`, `compile_objective` | move; the old methods call them | Stage 13 |
| `_build_sources`, `_edge_relation`, `_spill_cost`, `_precompute_spill_costs`, `_write_back`, `_menu_position` | stay on `SaCoOptimizingSolver`. `SAFamily` has its own graph-side equivalents | copy of the logic | Stage 13 |
| `FirstFitLayoutSolver`, `SolverToPermutation` | stay where they are, called from `_seed_permutation` | none | Stage 13 moves them to `first_fit_order` |

`class SaCoOptimizingSolver(AnnealSearch, CoreDivisionLayoutSolver)` keeps its constructor,
capabilities, `plan_layout_and_core_divisions` and write-back. The 241 lines of private access
in `test_sa_cooptimizer.py` and the `patch.object(sa.SaCoOptimizingSolver, ...)` calls
(`tests/inductor/test_solver_auto_coarse_tiling.py:1362,1365`) keep working because
lookups still go through `self`. `search.py` uses
`get_inductor_logger("scratchpad.sa_cooptimizer")`, because
`test_sa_cooptimizer.py:2271,2285` assert on that logger.

### Model

```python
@dataclass
class AnnealRecord:
    """What the search reads per index. CoreDivisionBuffer satisfies it too."""
    name: str
    size: int
    uses: tuple[int, ...]
    first_use_is_read: bool
    in_place_parents: list[str]
    residency_reason: Optional[str]
    lifetime_start_override: Optional[int]
    lifetime_end_override: Optional[int]
    boundary: BufferType            # Input if not node.runs, Output if graph_output
    op_position: Optional[int]      # node.tick, None for an InputNode
    parents: list[str]              # one per Gate, in today's order
    tile_reads: dict[str, "TileReads"]

@dataclass
class AnnealModel:
    domains: dict[OpNode, DivisionSource]
    relations: dict[Gate, EdgeRelation]
    order: list[PlacementBox]        # order[i] is index i; Encoding.boxes is this list
    records: list[AnnealRecord]      # parallel to order
    menus: list[MenuDivisions]       # parallel to order; table relations and seeds
    prefetch_views: dict[InputNode, tuple[PerCoreView, ...]]  # PrefetchMenu.views
    limit: int                       # bytes, today's self.limit
    alignment: int
    objective: Optional[ObjectiveFn] # None means the memory-only objective
    spill_costs: list[int]           # parallel to order
    hbm_bytes_per_us: float
```

`encode` returns `Encoding(model=anneal_model, ...)` with `boxes=anneal_model.order` and
`capacity=anneal_model.limit // anneal_model.alignment`, in alignment units as Stage 7's
`Encoding.capacity` is. `AnnealModel` is private to the family, so `menu_index` may live in
it. Nothing with a menu index crosses into the graph, which keeps Stage 11 and the
non-enumerating direction open.

### `SAFamily.encode`

```python
class SAFamily:
    ENCODES: ClassVar[Restriction] = (
        Restriction.WHOLE_STICKS | Restriction.PADDED_REDUCTION_UNSPLIT
        | Restriction.MATMUL_STICKS | Restriction.POINTWISE_STICKS
        | Restriction.INDIRECT_ACCESS | Restriction.ENCODED_K_UNSPLIT
        | Restriction.WINDOW_UNSPLIT | Restriction.CONV_SPATIAL_UNSPLIT
        | Restriction.TOPK_SPLITS | Restriction.KEEP_BY_INDEX_SPLITS
        | Restriction.PLAIN_REDUCTION_UNSPLIT)
    RULES: ClassVar[Rule] = ~Rule.NONE
    STEPS: ClassVar[frozenset[StepKind]] = frozenset(
        {StepKind.COPY_OUT, StepKind.RESTICKIFY})   # restickify only as a pinned step
    OPTIMIZATIONS: ClassVar[Optimization] = ~Optimization.NONE
    seed: int

    def __init__(self, seed: int = 0) -> None: ...
    @classmethod
    def from_config(cls, config) -> "SAFamily": ... # cls(seed=0)
    def encode(self, graph: SpyreOptimizationGraph) -> Encoding: ...
    def solve(self, encoding: Encoding, settings: SolveSettings) -> AnnealSolution: ...
    def read_back(self, solution: AnnealSolution,
                  graph: SpyreOptimizationGraph) -> None: ...
    def score(self, encoding: Encoding, state: AnnealState) -> int: ...
```

The cost model is `graph.target.cost_model`; nothing passes one in. `encode` writes no
decision to the graph and nothing to a shared cache, and it draws no randomness. A settled
read adds a hazard pin, which Stage 9 restores on retry. That is what lets Stage 9
re-encode a graph after an OR-Tools timeout. The family holds no state: `solve` returns an
`AnnealSolution` that carries its `encoding`, `read_back` reads `solution.encoding`, and
`score` takes the encoding it scores. The steps below are functions in `family/sa/encode.py`, as
Stage 5's are in `family/ortools_encode.py`. The order of work is:

1. **Refuse what the family cannot encode.** `coverage(SAFamily, graph.target)` runs at
   config time (Stage 9), not here. A pinned `RELAYOUT` step raises `NotImplementedError`
   naming the edge. An unpinned stick raises too, until Stage 11. All eleven flags are in
   `ENCODES` because sticks are pinned and the table obeys the division rules. The four
   `Rule` flags hold through the same table and tiling space, and because a copy-out is
   never placed. `RELAYOUT` is not in `STEPS`: SA never chooses one. A relayout is chosen,
   never forced (`STEP_KINDS`), so leaving it out costs plan quality, not legality:
   `coverage(SAFamily, SPYRE)` lists it in `unused` and has no gaps, and SA serves `SPYRE`
   as the fallback.
2. **Tables.** `build_division_tables(graph, coarse_tiling=False)`, once, with cores from
   `graph.target.cores` and each node's flags from `node.restrictions`. `coarse_tiling` is
   false because SA's menu never carries tilings (`allocator.py:4002`). `table.pruned` and
   `table.decided` are copied to the `Encoding`, and a `VIOLATED` stick table raises
   `RuleViolation`.
3. **Edge tables.** `settle_deferred_reads(graph, tables, relayout_sources=lambda: (),
   rebuild=...)` around `build_edge_tables(graph, menus, capacity=graph.target.lx_bytes,
   decides_relayouts=False)`, with `menus` each table's `entries`. It gives the gates, the
   prefetch menus and the in-place order.
4. **Sites, in today's index order.** LX prefetches first, in `graph.ir.graph_input_names` order
   (`allocator.py:4228,4722-4723`), then running nodes by tick (`:4269`). A
   `RestickifyNode` and an `OpaqueNode` are running nodes. An input with no `PrefetchMenu` is not
   a site (`:4237-4238`).
5. **Domains.** The menu is built exactly as `_build_sources` does, including
   `DivisionConfig(cd, index, canonical in split_maps)`. A site with `table.enumerated` gets
   `GeneratedDivisions(space, configs[0])`, with `space` from `build_op_split_space(node.op,
   graph.target.cores, tiling=...)`. The rest get the menu, and so does any site whose
   `node.restrictions` differ from its type's config in `SPYRE_OPS`, because
   `WorkDivisionContext.is_legal` cannot forget a flag. The division rules need no second
   pass here: `is_legal` and the table share one body in
   `rules.py`, so a generated division obeys what the table obeys.
6. **Tilings.** With `Optimization.COARSE_TILING` in `graph.target.optimizations` (which
   `select_target` keeps only with `config.auto_coarse_tiling` on), `tiling` is
   `build_tiling_space(node.op, readers=...)`, with `readers` the computed ops reading the
   node, in op order (`allocator.py:4177-4181`). A `RestickifyNode`, or a node whose
   `op.get_operation_name()` is in a region of `prescribed_regions(graph.ir.operations)`, gets
   `tiling=None`. This is the same refusal as `_tiling_candidates`
   (`allocator.py:4004-4009`), honours the parent's Loops row, and is a rule difference from
   today's SA (Deviation 7).
7. **Relations.** One per `Gate` in `EdgeTables.gates`, chosen as `_edge_relation` chooses
   (`sa_cooptimizer.py:1079-1102`). `residency_edge(gate)` feeds a `ViewRelation` when both
   ends generate, with `carries = lambda spec: reads.aligned(spec, spec)` when `tile_reads`
   has the parent. Otherwise `table_relation` projects `edge_match_pairs(gate)`. A gate with
   no pairs gets an empty relation, not a cut. Two gates of one producer share one
   `ResidencyEdge`.
8. **Records.** `residency_reason(node, graph, decides_relayouts=False, tileable=False)` gives
   the pin, and `in_place_candidates(node)`, less any input with no `PrefetchMenu`, gives
   `in_place_parents` in today's list order, because that order feeds the `unordered_set` in the native constructor
   (`perm_layout_native.cpp:237-273`). `tile_reads` is empty with tiling off, and otherwise holds
   `TileReads(graph.ir, parent.op, node.op)` per computed parent, as `_tile_reads` does
   (`allocator.py:4494-4506`). `_assert_unsized_buffers_are_pinned` runs in the search as today.
9. **Objective.** See below.
10. **Boxes.** One integer `PlacementBox` per site with `key=node`, `name=node.name`,
    `start, end = graph.lifetime(node)`, `present=False` for a pinned site and `True` otherwise,
    `size` the seed footprint and `max_size` the whole buffer, both in alignment units, and
    `ties` as `(True, parent key)` in `in_place_parents` order. For an annealing box the
    footprint moves with the division, and the search reads it through `_per_core_size`.
11. **Optimizations that are off get no domain.** Step 6 covers `COARSE_TILING`. Without
    `IN_PLACE`, every `in_place_parents` is empty. Without `PREFETCH`, no input is a site.
    Without `LX_RESIDENCY`, every site gets a `residency_reason`. Without `CORE_SPLIT`, a
    site keeps only its one-core menu entries, and a site with none raises `Unsupported`, as
    an op with no legal candidate does today. `SPYRE` has all five, so no plan changes here.

### Score

`compile_objective` wraps Stage 4's kernel with SA's getters, and keeps the four-exception catch
of `_build_score_fn` (`sa_cooptimizer.py:850`), which stays with the caller of that kernel:

```python
ObjectiveFn: TypeAlias = Callable[[list[DivisionConfig], frozenset[str]], int]

def compile_objective(expr: sympy.Expr,
                      getters: Mapping[sympy.Symbol, Callable[..., int]]
                      ) -> Optional[ObjectiveFn]:
    strict = not config._cpsat_warn_on_cost_expr
    try:
        return utils.compile_fixed_point(expr, getters, unit=Unit.NS, strict=strict)
    except (ValueError, TypeError, ZeroDivisionError, RuntimeError):
        if strict:
            raise
        return None

def fixed_point_score(state: AnnealState, objective: Optional[ObjectiveFn],
                      spill_costs: Sequence[int], hbm_bytes_per_us: float) -> int: ...
```

`fixed_point_score` is the body of `_score` unchanged, including the single read of
`packer.addresses`. `SAFamily.score` and `AnnealSearch._score` both call it, so the search
keeps one code path. `AnnealState` is a `Protocol` with `chosen`, `packer` and
`companions(addresses) -> int`.

The expression is `graph.target.cost_model.priced_objective(graph, CostDomains(nodes,
relayouts=()), linear_only=False)`, with one `NodeDomain` per site from its menu, readers
and its space's tile dims. `None` means the memory-only objective, with `spill_cost(node,
clamp_unsized=True)` per site. When `graph.target.cost_model.traffic_only`, it uses
`spill_cost(node, boundary_known=False, clamp_unsized=True,
size=traffic_sizes(graph)[node.name])`, the objective Stage 9 states for committed mode.
`relayouts=()` is what removes charges and `relayout_sources` for this family.

This reproduces today's arithmetic exactly, down to rounding, under one condition: the
expression must be structurally equal to today's `cost_expr`, with the same symbol names.
The getters are:

| symbol | getter | today |
|---|---|---|
| `Sym.resident(n)`, `is_lx_<n>` | `1 if n in resident else 0` | `:825` |
| `Sym.split(n, key)`, `split_<n>_<key>` | `chosen[i].splits.get(key, 1)` | `:837-840` |
| `Sym.tile(n, d, r)`, `tile_<n>_<o\|r><d>` | `chosen[i].tiling.count_for((d, r))` | `:841-844` |

`division_<n>` is not bound, because only a relayout term uses it. An unknown name follows
Deviation 3.

Companion traffic stays in `AnnealSearch._companion_charges`, the loop today's search runs.
Stage 4's test holds it equal to `cut_traffic(count_unmodelled=False)`.

### `SAFamily.solve`

```python
@dataclass
class AnnealSolution(Solution):
    # status is SolveStatus.FEASIBLE: SA never proves anything.
    # objective is the best fixed-point score, today's best_score.
    # offsets are in alignment units, resident sites only, as Solution says.
    # stats stays empty: only CP-SAT has last_solve_stats.
    encoding: Encoding | None = None   # what read_back decodes
    chosen: dict[OpNode, DivisionConfig] = field(default_factory=dict)
    spill_reasons: dict[OpNode, str] = field(default_factory=dict)
    tiled_runs: list[tuple[int, int]] = field(default_factory=list)  # inclusive ticks
    stopped_by: Literal["budget", "no_move", "time_limit"] = "budget"
    steps_run: int = 0
    def score_ns(self) -> Optional[float]: ...
    def off_expression_ns(self) -> dict[str, float]: ...
```

`SAFamily.solve` builds a private `_ModelSearch(AnnealSearch)` whose `_build_sources` and
`_edge_relation` read `encoding.model`. It then runs the same sequence as
`plan_layout_and_core_divisions` (`sa_cooptimizer.py:795-802`): `Random(self.seed)`,
topology, seed configs, seed packer, anneal, with today's moves. An empty model returns an
empty `FEASIBLE` solution. Packer addresses are multiples of the alignment, so `offsets`
divide by `encoding.model.alignment` exactly. `spill_reasons` use the existing
`residency_reason or _SOLVER_CHOSE_SPILL` rule (`:1981`).

`settings.workers` is ignored; the search is single-threaded. `settings.deterministic=True`
gives exactly today's step budget and never reads a clock. Otherwise a positive
`settings.time_limit` sets a deadline from the start of `solve`:

- After calibration, `steps = min(_step_budget(n), remaining / mean_probe_time)`, so the
  cool is shortened rather than cut off while hot.
- Each step checks the deadline and breaks, as a guard, because a reorder sweep costs
  more than a probe.
- `best-seen` is adopted as usual.

If the limit does not bind, `steps` is unchanged and the trajectory is bit-identical,
since no clock value reaches the RNG or the score. `_seed_permutation` is the one seam for
postmath's seeding fallback, and a run with zero steps returns the seed, so a debug greedy
family can reuse this search or stand beside it.

### `read_back`

Nothing is written during `encode` or `solve`. `read_back` runs once, from a plan:

- `node.work_split` is `work_split_from_division(node, config.division)`. No index is written.
- An LX prefetch gets `node.prefetch_view = prefetch_views[node][config.menu_index]`, from
  `solution.encoding.model`, and no `work_split`, because its entries are keyed by a consumer's
  symbols and two can share a split map and differ in slicing (`:183-190`).
- `node.resident` and `node.offset` come from `offsets`, and `node.spill_reason` from
  `spill_reasons`. A `RestickifyNode` decodes like any node. Its pinned step stays and gets
  nothing.
- `edge.in_place` is true when both ends are resident, the child names the parent in
  `in_place_parents`, the addresses are equal, and the pair is in one tiled run or both are
  untiled. The packer reports no in-place flag, but buffers that overlap in time at the handoff
  tick share an address only through an in-place overwrite. Sizes are positive here because `empty()` and
  the unsized pin bar zero-size buffers.
- `graph.cuts` gets the tick of each running op unless it and the next running op are in one
  tiled run, from `solution.tiled_runs`. The last op adds nothing.
- One `Step(kind=StepKind.COPY_OUT)` per escaping tiled producer, shared by the edges to
  readers outside its run, using the same test as `_companion_charges`. It goes through
  `insert_step` at the front, and its `footprint` is the tile's size in alignment units. A
  tiled graph output with no such reader gets no step, and its traffic is priced by node.
- No `RELAYOUT` step is ever written.

### Family registration

`config.layout_solver` gains the value `"sa"`, and `FAMILIES["sa"] = SAFamily` in
`family/registry.py`. `"simulated_annealing"` keeps today's meaning until Stage 13.
`select_allocator` is not touched, so `LAYOUT_SOLVER=sa` raises its existing `ValueError`
until Stage 9 routes it. Stage 9 builds `FAMILIES[key].from_config(config)` and owns
`config.fallback_solver`.

### Keeping the RNG sequence identical

The search code does not change, so what must hold is that the model feeds it the same
inputs. Each item below is compared by the model diff in the gate.

1. `order` equals today's `_bufs` order. Everything is positional:
   `randrange(n)`, `_flippable`, `_anchor_candidates`, `choices(range(n), ...)`.
2. `n` counts every site, so `_step_budget`, the probe count and the burst length match.
3. Menu order and length are identical, because `rng.choice` consumes draws by list length.
4. The same sites generate, since a generated anchor draws differently from a menu anchor
   (`:351-353` against `:463-483`).
5. `space.axes`, ascending `factor_domain` and `tiling_options` come from the same
   `OpSplitSpace`. `build_tiling_space` is called with the same readers, because
   `_unit_tile_breaks_a_reader` changes the count lists. The tiling filter is the one change,
   and the dual run's old side applies it too.
6. Relation type per edge is identical. `TableRelation` and `ViewRelation` pick different
   representatives on purpose (`:623-630`).
7. Sizes, uses, `first_use_is_read` and `in_place_parents` order are identical, because
   `quality()` ranks reorder candidates and float ties break by the sort.
8. The objective is structurally identical, per Score above.
9. `encode` draws no randomness, and `solve` seeds `Random(self.seed)` once at the same point.
10. Python's `Random` and the native build are the same in both arms. `_parents_idx` sets
    are sorted at every use (`:1470,1581,1648`), so hash seed stays irrelevant.

### How long a fallback run takes

If CP-SAT stops with no plan at `cpsat_time_limit_seconds` (default 60,
`config.py:385`, per stage; `SolveError` is raised with no incumbent,
`ilp_solver_ortools.py:871-875`) and `config.fallback_solver` names SA, Stage 9 re-encodes.
SA then returns a plan, unless it hits an error listed under "The seed, and how it fails". The
total is CP-SAT's model build and limit, plus `SAFamily.encode`, plus the anneal. Encoding is
the larger unknown. SA prep on the whole zoo was 28 s, of which 17.5 s was `cost_expr`
(`sa-vs-cpsat-REPORT.md`). Granite's largest graphs took 8.1 to 8.3 s in total against
CP-SAT's 60.6 to 71.0 s.

Today nothing bounds the anneal in wall-clock time (`sa_cooptimizer.py:90-93`). The step
count stops growing at 250 buffers while cost per step does not. Measured with the real
engine and a scratch `-O2` build of the packer (the two golden solves reproduce on it), on
synthetic chains with short lifetimes and three objective terms per buffer, scaled from a
few hundred steps at the hot start of the cool:

| buffers | ms/step | steps | anneal, scaled |
|---:|---:|---:|---:|
| 64 | 0.20 | 12,800 | 3 s |
| 128 | 0.44 | 25,600 | 11 s |
| 250 | 0.8 to 1.1 | 50,000 | 40 to 55 s |
| 500 | 4.2 to 4.8 | 50,000 | 210 to 240 s |
| 1000 | 13 | 50,000 | 650 s |

Real graphs have far larger expressions and denser lifetimes, so treat these as a lower
bound on shape, not a forecast. `native_packer_performance.md` reports 2.95 ms/step at
n=256 for a stress case in the placement-only annealer.

The fallback gets the same `SolveSettings` as the primary, so the worst case is about twice
the time limit plus encoding. A stop is safe at any step because `best-seen` is always legal,
but the seed is mostly one-core divisions, so a run cut short can have poor parallelism.

## Changes

Paths below `.../sa/` are under `torch_spyre/_inductor/scratchpad/lx_allocator/family/`.

| file | change |
|---|---|
| `.../sa/__init__.py` | new, exports |
| `.../sa/divisions.py` | new, moved sources and relations |
| `.../sa/search.py` | new, `AnnealSearch`, `_ModelSearch` |
| `.../sa/score.py` | new, `compile_objective`, `fixed_point_score`, `AnnealState` |
| `.../sa/model.py` | new, `AnnealModel`, `AnnealRecord`, `AnnealSolution` |
| `.../sa/encode.py` | new, the encode steps as functions |
| `.../sa/family.py` | new, `SAFamily` |
| `.../registry.py` | register `"sa"` |
| `torch_spyre/_inductor/scratchpad/sa_cooptimizer.py` | reduced to the solver class, aliases and buffer-side helpers |
| `torch_spyre/_inductor/config.py` | `"sa"` in the `layout_solver` literal, comment |
| `docs/source/compiler/sa_co_optimization.md` | the family, plus corrected seed and float statements |
| `tests/inductor/test_sa_family.py` | new, with the trace probe `sa_trace` |
| `tests/inductor/test_sa_family_live.py` | new, the dual run on the card |
| `tests/configs/torch_spyre_tests/inductor/test_sa_family_config.yaml` | new, mirrors `test_sa_cooptimizer_config.yaml` |
| `tests/configs/torch_spyre_tests/inductor/test_sa_family_live_config.yaml` | new |

Ordered commits, each leaving the suite green:

1. **A trace probe in the tests.** No production change. `sa_trace`, a context manager in
   `test_sa_family.py`, gives `rng_digest`, `score_digest` and `steps_run` for each solve run
   inside it. `sa_cooptimizer.rnd` is the stdlib module, so the probe rebinds the name `rnd` in
   `sa_cooptimizer` with `monkeypatch` and never touches `random.Random` itself. The one
   `rnd.Random` call there (`sa_cooptimizer.py:795`) stays on `SaCoOptimizingSolver`, so
   commit 3 does not move this target. `SAFamily.solve` is the only other place a generator
   is built, so `family/sa/family.py` imports `random as rnd` and the probe rebinds that name
   too once it exists. The probe patches `_score` on `SaCoOptimizingSolver`, which also holds
   after `_score` moves to `AnnealSearch`, and on `_ModelSearch` once it exists. A test fails on
   an SA solve with an empty digest, so a stale target cannot silently cover nothing.
2. **Move sources and relations** to `divisions.py`, with aliases. No logic change.
3. **Move the search** to `AnnealSearch`, with `SaCoOptimizingSolver` inheriting it.
   `test_sa_cooptimizer` passes unchanged, including `GoldenSolveTest`, `DeterminismTest` and
   `CrossProcessDeterminismTest`.
4. **Extract scoring** into `score.py`. The old methods become wrappers. The same check as
   commit 3.
5. **Add `AnnealModel`, `_ModelSearch`, `AnnealSolution` and `SAFamily.solve`,** tested
   with an `Encoding` built from synthetic buffers against `SaCoOptimizingSolver` on the same
   buffers, in one process. Includes the deadline path.
6. **Add `SAFamily.encode`** and the model diff against `SaCoOptimizingSolver`'s
   `_sources`, `_relations` and `_score_fn`.
7. **Add `SAFamily.read_back`** with comparison against `decode_todays_plan`.
8. **Register `SAFamily` in `FAMILIES`,** add the config value, and add
   `test_sa_family_live.py` with its shard config.
9. **Docs.**

## Interfaces

Consumes:

| symbol | signature | from Stage | used for |
|---|---|---|---|
| `SpyreOptimizationGraph`, `DataEdge`, `Pin`, `Step`, `StepKind`, `STEP_KINDS`, `WorkSplit`, `Restriction`, the eleven flags, `Rule`, `Optimization` | parent plan | 1 | everything |
| `Target`, `SPYRE`, `SPYRE_OPS`, `OpFactory.with_type`, `OpNode.restrictions` | parent plan; `restrictions` set by `OpFactory.make` | 1 | class vars, domains, derived-target tests |
| `select_target` | `(config) -> Target` | 2 | dual run |
| `RestickifyNode`, `OpaqueNode` | direct `OpNode` subclasses in `graph.nodes` | 1 | sites |
| `OpNode.op`, `.uses`, `.tick`, `.read_count`, `.first_use_is_read`, `.graph_output`, `.lifetime_start_override`, `.lifetime_end_override` | live op; `tuple[int, ...]`; `int \| None`; `int`; `bool`; `bool`; `int \| None` | 1, 2 | records, tilings |
| `graph.lifetime`; `graph.target.lx_bytes`, `.alignment`, `.cores`, `.optimizations`, `.cost_model` | `lifetime(node) -> tuple[int, int]`, end exclusive; bytes; bytes; `int`; `Optimization`; `SpyreCostModel` | 1, 4 | boxes, tables, objective |
| decision fields | `OpNode.work_split`, `.resident`, `.offset`, `.spill_reason`; `InputNode.prefetch_view`; `DataEdge.in_place`, `.insert_step(index, step)`; `graph.cuts: set[int]`, the tick of the op before each cut; `Step.footprint` | 1, 7 | `read_back` targets |
| `graph.ir` | `GraphLowering`, final | 2 | `TileReads`, `prescribed_regions` |
| `build_graph` | `(ir, target, *, divisions_committed=False, drain_plans=None, lx_relayout_plans=())`, defaults here | 2 | tests, dual run |
| `work_split_from_division` | `(node, division) -> WorkSplit` | 2 | read-back |
| `carry_pairs`, `CarryPair` | `(graph) -> list[CarryPair]` | 2 | `CARRY` gates, through `build_edge_tables` |
| `coverage`, `Coverage` | `(family, target) -> Coverage(gaps, unused)`, run at config time; a chosen step kind is unused, not a gap | 3 | coverage test |
| `check_decisions`, `RuleViolation`, `ALWAYS` | `(graph) -> list[Violation]`; `in_place_conditions` and `tiles_line_up` in physical form | 3 | gate |
| `residency_reason` | `(node, graph, *, decides_relayouts: bool, tileable: bool) -> str \| None` | 3 | pins |
| `decode_todays_plan` | `(graph, allocation, accepted_relayouts) -> list[Mismatch]`, called with `()` | 3 | parity gate |
| `decisions`, `NodeDecision` | `(graph) -> dict[str, NodeDecision]` | 3 | comparison |
| `lx_check_summary.py` | CLI over `SPYRE_LX_CHECK_FILE` records | 3 | level 4 |
| `CostModel.priced_objective` | `(graph, domains, *, linear_only: bool) -> sympy.Expr \| None`, `==` today's `cost_expr` on SA's domains | 4 | score |
| `CostDomains`, `NodeDomain` | `CostDomains(nodes, relayouts=())` | 4 | score |
| `CostModel.spill_cost` | `(node, *, boundary_known=True, clamp_unsized=False, size=None) -> int`, called with `clamp_unsized=True` | 4 | memory-only objective |
| `CostModel.traffic_only`, `traffic_sizes` | `bool`; `(graph) -> dict[str, int]` | 4 | committed-mode objective |
| `Sym.resident`, `.split`, `.tile` | today's exact names and assumptions | 4 | getters |
| `utils.compile_fixed_point`, `Unit`, `UnknownSymbolError` | `(expr, getters, *, unit, strict)`; `class UnknownSymbolError(KeyError)` | 4 | score |
| `Encoding` | as Stage 5: `model`, `provenance`, `pruned`, `symbols`; `ortools` stays `None` | 5 | the encode result |
| `build_division_tables`, `DivisionTable` | `(graph, *, coarse_tiling)`, cores from `graph.target.cores`, flags from `node.restrictions`; `.entries`, `.enumerated`, `.pruned`, `.decided` | 5 | menus |
| `build_edge_tables`, `EdgeTables`, `Gate`, `GateKind` | `(graph, menus, *, capacity, decides_relayouts, ...)` | 6 | gates |
| `settle_deferred_reads` | `(graph, tables, *, relayout_sources: Callable, rebuild: Callable)` | 6 | deferred-read loop |
| `prefetch_menu`, `PrefetchMenu` | `(input_node, tables) -> PrefetchMenu \| None` | 6 | prefetch sites |
| `in_place_candidates` | `(node) -> list[str]`, today's order | 6 | records |
| `residency_edge` | `(gate) -> ResidencyEdge \| None`, in `family/edge_tables.py` | 6 | `ViewRelation` |
| `edge_match_pairs` | `(gate) -> tuple[tuple[int, int], ...]`, in `family/edge_tables.py` | 6 | `table_relation` |
| `PlacementBox`, `Encoding.boxes`, `Encoding.capacity`, `Family`, `Solution`, `SolveStatus`, `SolveSettings`, `FAMILIES` | Stage 7; `Encoding.capacity` in alignment units | 7 | protocol, solution base, registry |

Provides:

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `SAFamily` | `(seed: int = 0)`, `from_config(config)`; `ENCODES` with all eleven flags, `RULES`, `STEPS`, `OPTIMIZATIONS` as above | 9, 11 | final |
| `SAFamily.encode` | `(graph) -> Encoding`, `model` an `AnnealModel` | 9, 11 | final |
| `SAFamily.solve` | `(encoding, settings) -> AnnealSolution` | 9 | final |
| `SAFamily.score` | `(encoding: Encoding, state: AnnealState) -> int` | 9, 11 | final |
| `SAFamily.read_back` | `(solution, graph) -> None`; writes `offset` in alignment units, `spill_reason`, `prefetch_view`, `graph.cuts`, copy-out steps | 9, 10, 11 | final |
| `AnnealModel` | as above | 11 | final, internal to the family |
| `DivisionSource`, `EdgeRelation`, `TableRelation` | base classes and the table relation, in `family/sa/divisions.py` | 11 | final, internal to the family |
| `AnnealSolution` | as above; carries its `encoding`; `status` is `SolveStatus.FEASIBLE`; `score_ns`, `off_expression_ns` feed the cost dump | 9, 10 | final |
| `FAMILIES["sa"]`, `config.layout_solver = "sa"` | `SAFamily`; string | 9, 12 | final |
| `AnnealSearch._seed_permutation` | `(sizes: list[int]) -> list[int]` | 13, greedy, seed fallback | final seam |
| `SaCoOptimizingSolver` on `AnnealSearch` | unchanged API | 9 | temporary, removed in Stage 13 |
| aliases in `sa_cooptimizer.py` | old underscore names | 13 | temporary, removed in Stage 13 |

## Tests

`tests/inductor/test_sa_family.py`, device-free:

- `test_solve_matches_today_on_synthetic_graphs`: for every graph in `synthetic_graphs()`, at
  the capacities `_capacities` gives in `test_sa_cooptimizer.py`, an encoding built from its
  buffers gives the same divisions, addresses, `best_score` and `sa_trace` digests as
  `SaCoOptimizingSolver` on a deep copy of the same buffers, in the same process, with
  `offset * alignment` equal to today's address. `multi_region` and `k_split_consumers` at
  `GoldenSolveTest`'s capacities are among them.
- `test_trace_repeats`, `test_trace_is_not_empty`: two `SaCoOptimizingSolver` runs give equal
  digests, and a run gives a non-empty one.
- `test_compile_objective_matches_build_score_fn`: equal integers on sampled states over the
  op-features corpus.
- `test_unknown_symbol_warns_and_goes_memory_only`, `test_unknown_symbol_raises_when_strict`,
  `test_lambdify_error_goes_memory_only`.
- `test_score_adds_companions_outside_the_objective`.
- `test_encode_writes_nothing_to_the_graph`: decisions and pins compare equal before and
  after, and a second `encode` gives the same sites, menus and relation pairs.
- `test_encoding_boxes_and_capacity`: `Encoding.boxes` is the model's `order`, `capacity` is
  in alignment units, and each box's `start`, `end` and `ties` match its node.
- `test_enumerated_op_generates_at_default_config`: pins the source selection with tiling off.
- `test_region_op_and_restickify_get_no_tiling`.
- `test_prefetch_keeps_its_menu_and_disambiguator`, `test_prefetch_decodes_to_prefetch_view`.
- `test_x_times_x_is_one_gate`, `test_two_indexes_of_one_buffer_count_twice_in_companions`.
- `test_edge_with_no_pairs_spills_the_producer`: no cut, no copy-out, flood stops there.
- `test_seed_is_the_first_table_entry`: not the committed division.
- `test_table_pruned_is_copied`, `test_derived_target_forces_a_menu_source`: a graph built
  with `replace(SPYRE, ops=SPYRE_OPS.with_type(...))` that drops a flag.
- `test_optimization_off_gets_no_domain`, one derived target per `Optimization` member.
- `test_read_back_infers_in_place_from_addresses`, `test_read_back_writes_no_relayout`,
  `test_pinned_relayout_is_refused`.
- `test_cuts_skip_only_tiled_runs`: a tick is missing from `graph.cuts` only inside a tiled
  run. `test_copy_outs_match_companions`.
- `test_unpinned_stick_is_refused`, `test_bad_in_place_pair_raises_at_seed`.
- `test_deterministic_ignores_time_limit`, `test_unbound_limit_keeps_the_trajectory`,
  `test_time_limit_returns_best_seen`: an already expired deadline still returns a legal plan.
- `test_family_is_registered`, `test_from_config_seeds_zero`, `test_status_is_feasible`,
  `test_coverage_has_no_gaps`: `coverage(SAFamily, SPYRE).gaps` is empty and `.unused` names
  the `RELAYOUT` step kind and nothing else.
- `test_family_holds_no_state`: two encodings from one `SAFamily` solve independently, and
  `read_back` decodes the encoding its solution carries.

`tests/inductor/test_sa_family_live.py`, on the card:

- `test_dual_run[case, tiling]`: the dual run under Parity and gates, over `test_softplus`,
  `test__simple_attn`, `test_mlp`, `test_rms_norm` and `test_flash_attention` from
  `tests/inductor/test_building_blocks.py`, with `LAYOUT_SOLVER=simulated_annealing`, at default
  config and at `AUTO_COARSE_TILING=1`. Each case runs the existing test in-process as
  `TestBuildingBlocks(case).debug()`, so the compile and its numerical check are the suite's
  own. It stays in CI after this PR merges, so a change to one SA path without the other fails
  there.

Existing suites run serially: the parent's unit list, plus `test_simulated_annealing` and
`test_sa_cooptimizer` with `SA_COOPT_LARGE_CAPTURES=1`. Device suites on the card, per the
parent, run with `LAYOUT_SOLVER=simulated_annealing` so they cover the moved engine.
`pre-commit run --files <changed>` as usual. Planning time is reported, not gated
(Deviation 12).

What proves the claim: the unchanged `test_sa_cooptimizer` suite proves the move changed
nothing on its graphs, the model diff proves `encode` feeds the same inputs, and the dual run
proves the plan.

## Parity and gates

The parent's gate: "on the Stage 0 corpus with the same seed, the plans equal today's
`SaCoOptimizingSolver` plans wherever the rules agree, and `check_decisions` passes on every
one."

Stage 0 is dropped (Deviation 13). SA proves no minimum, so parity here is not an optimum. It
is today's exact, seeded behaviour: in one process, on the same input, `SAFamily` and
`SaCoOptimizingSolver` give the same trace, the same score and the same decisions. Nothing is
stored between runs.

**Cases.** Five `test_building_blocks` compiles on the card, in `test_sa_family_live.py`, with
`LAYOUT_SOLVER=simulated_annealing`, at default config and at `AUTO_COARSE_TILING=1`. They
have to be live, because a pickled capture runs menus only (see Today). The buffer-level
tests in `test_sa_family.py` still cover the moved engine, on menus. Device tests run on card
runners; there is no device-free compile.

**Dual run.** Each case installs two patches with `monkeypatch`:

- `CoOptimizingAllocator._solve` (`allocator.py:3218`), to keep the `GraphLowering` it is
  called with;
- `SaCoOptimizingSolver.plan_layout_and_core_divisions` (`sa_cooptimizer.py:778`), which the
  allocator calls with today's `cost_expr` (`allocator.py:3352`).

The second wrapper runs `build_graph(ir, select_target(config))`, encodes and solves through
`SAFamily.from_config(config)` with `SolveSettings(deterministic=True)`, and reads back onto
that graph, then calls the original. Order matters, because the old solver mutates buffers and
appends to menus. Both solves run inside `sa_trace`. Under tiling the wrapper also rebuilds the
old `division_space` of every site step 6 of `encode` refuses, with
`build_op_split_space(op, graph.target.cores, tiling=None)`, so the comparison stays exact.
The wrapper lets every exception propagate, so a failure in the family fails the case.

Compared, in order, with the first difference reported:

1. **Model diff.** Same site names and order. Same size, uses, flags, pins, lifetimes,
   `in_place_parents`, `op_position`, per index. Same source class and menu keys. Same
   relation class and pair sets per gate. Objective `==` today's `cost_expr`, or both are
   absent. Equal `SAFamily.score` on 1,000 sampled states.
2. **Trace.** `rng_digest` and `score_digest` equal. A mismatch reports the first step.
3. **Plan.** Per site: splits, reduction symbols, tiling, `offset * alignment` against the
   address, spill reason, and `best_score`. The comparison is by division value, not menu
   index, since `_menu_position` appends and `read_back` has no index. `decode_todays_plan`
   with `accepted_relayouts=()` puts today's plan on a second graph, and `decisions` on the
   two graphs must match.
4. **Check.** `check_decisions` on the read-back graph, then `lx_check_summary.py` over the
   records the cases write with `SPYRE_LX_CHECK_FILE` set.

**Pass:** zero differences at all four levels on every case, at both configs. Stage 9 makes
zero level 4 reports at tiling an entry condition.

Level 4 can be zero for these reasons. At default config I found nothing in `ALWAYS` or the
rule tables that today's SA breaks, given pinned sticks and Stage 3's `cross_frame` as view
equality, which is how the joint path enforces it (`allocator.py:1454-1461`). In-place
footprints are equal there: same device layout and, with no reduction on either side, a
compatible pair has equal `output_partition`. With tiling on, SA differs from CP-SAT's
encoding in three ways, and Stage 3's physical rules admit all three:

- the packer allows a smaller in-place child (`perm_layout_native.cpp:547`, measured), where
  CP-SAT requires equality (`ilp_solver_ortools.py:604`);
- an address reused after a dead tile is not called in place by `read_back`, where CP-SAT
  forbids it (`ilp_solver_ortools.py:1647-1648`);
- a nest needs `TileReads.aligned`, as `derive_tiling_groups` does (`coarse_tiling.py:423-467`),
  where CP-SAT also requires a match pair for a spilled buffer
  (`ilp_solver_ortools.py:998-1000`).

Elsewhere SA is stricter: `_read_across_a_tiling_boundary`, no `via_full` exemption, and
`untiled_bar` as a hard pin. The partial-reduction cut guard (`Rule.RESIDENT_TILE_UNREDUCED`)
holds because a reduction-split producer is never compatible. Not verified: no real tiled
compile ran here, so which of the three occur on the live cases is unknown, and the "at most one"
half of `in_place_conditions` for a parent with two children, which needs lifetime overrides,
was not checked.

**A difference** is a bug, which is fixed. The tiling filter is not one, because the old side
applies it too. The PR lists every graph where the filter fired, and each must pass a quality
check against an old run without the dual-run patches: over five seeds, the median predicted
cost is no worse, since `sa_co_optimization.md` warns that one seed proves nothing.

`SimulatedAnnealingLayoutSolver` is not a baseline here.

## Risks

- **Stage 4 returns an equal-valued but not structurally equal objective.** The trajectory
  diverges at the first rounding tie. Caught by level 1 of the gate, which names the first
  differing subexpression.
- **`srepr` round trip in a test.** `_decode_delivery_price` is an implemented function
  (`cost_model.py:1426-1450`), and its `_imp_` does not survive text. Comparing or replaying
  a parsed expression fails at call time. The dual run compares live objects.
- **A pickled capture stands in for a live case.** Parity would hold on menus and say nothing
  about generated sources. `test_enumerated_op_generates_at_default_config` and an assertion in
  the live cases that view relations exist catch it.
- **Encode cost doubles on fallback.** The second family rebuilds tables and the objective.
  Seen in the planning-time report (Deviation 12), not gated. Stage 9 can share a table cache
  later.
- **A move changes the search where only live compiles reach.** Both sides of the dual run
  share `AnnealSearch` from commit 3 on, so the dual run cannot see a change the move itself
  made. `test_sa_cooptimizer` holds the move on synthetic graphs and captures, which run menus
  only, and the device suites under `LAYOUT_SOLVER=simulated_annealing` check results, not
  plans. Open question 1.
- **`check_decisions` raises on a fallback plan.** The compile dies. Level 4 at both configs
  guards this.
- **A tile's address reuse overlaps a child by one tick.** `read_back` leaves `in_place` false
  there, so `boxes_fit_lx` must use Stage 3's nest-aware lifetime. Caught by level 4.
- **Wall-clock stop reduces reproducibility.** Only when the limit binds and
  `deterministic=False`. `stopped_by` is reported so a user can tell.
- **A generated config reaches the graph with a symbol not in the menu.** `sym_core_divs`
  keys come from the menu (`plan_solver.py:434-445`), and a tiled frame can admit a split map
  the menu lacks. That axis is unpriced today too, so parity holds. `read_back` writes the
  split anyway.
- **FakeTensor captures.** `AnnealModel` holds live ops, and the seed's `copy.deepcopy` touches
  only plain `LifetimeBoundBuffer`s. `AnnealModel.__reduce__` raises, so an accidental pickle or
  deepcopy fails at once.

## Deviations from the parent plan

1. **Which ops generate.** The plan says `_MenuDivisions` by default and `_GeneratedDivisions`
   "when coarse tiling is on". The code generates for every enumerated op at any config
   (`sa_cooptimizer.py:1125`, `allocator.py:4479-4486`). Menus serve LX prefetches, pinned ops and
   non-enumerated ops. Planned against the code, because a menu source consumes different
   draws and the gate would fail.

2. **"First-fit packer."** `NativePermutationLayoutSolver` stacks each buffer on the earlier
   overlapping ones and evicts (`perm_layout_native.cpp:520-564`). It is not first-fit. First-fit
   only seeds `pi`. `order` is the fixed index order, and `pi` is search state.

3. **"An unknown name fails."** Today it falls back to the memory-only objective, a test pins
   that, and the fallback family dying on a dynamic-shape symbol would be worse than a weaker
   objective. Resolution: raise when `config._cpsat_warn_on_cost_expr` is false, otherwise warn
   and use the memory-only objective.

4. **"An edge with no compatible pair becomes a cut."** Today it makes the producer ineligible
   (`sa_cooptimizer.py:1189-1192`) and stops the flood (`:1577-1587`). A cut comes only from
   tiling runs (`:1036-1061`). Planned against the code: spill, not cut.

5. **`relations` key.** The diagram keys on `DataEdge`. A loop-carry entry has no data edge,
   and today's relations are per `parents` entry. Resolution: `relations: dict[Gate,
   EdgeRelation]`, with `Gate.edges` naming the data edges behind each.

6. **`score` is more than the compiled objective.** The plan says it compiles the objective into
   an "integer fixed-point evaluator". It is a double evaluation rounded once, plus a separately
   rounded companion term outside the expression, or a memory-only sum. All three are kept, and
   parity needs Stage 4's expression to be structurally equal, not numerically equal.

7. **Plans differ from today's SA in one listed way.** Under tiling, `SAFamily` refuses tilings
   of `for_each_tile`-region ops and restickifies, honouring the parent's Loops row. Today's SA
   does not: `_prescribed_ops` is filled only for menu tilings (`allocator.py:3701`) and
   `_tileable` catches stamped ops only (`enumerate_tilings.py:409-423`), so it can tile an
   unstamped region op, and `apply_tilings` then raises `Unsupported`
   (`coarse_tiling.py:684-701`). This is confirmed by reading only. Whether an anneal lands
   there is unknown.

8. **"Every state the annealer visits is legal."** True, but through the eligibility mask as well
   as domains and relations, and with tiling on it also leans on post-solve raises. See Today.

9. **Family-dependent rule inputs.** The plan says the rules come from the same data. Residency,
   the tiling menu and pricing all depend on family capabilities. Hence
   `decides_relayouts=False`, `tileable=False`, `coarse_tiling=False` and `relayouts=()`.

10. **`SolveSettings.time_limit` has no meaning today.** Today's SA has no clock. Resolution: the
    deadline in `SAFamily.solve`, off when `deterministic=True`.

11. **SA is not yet the default fallback here.** The plan makes it so at this stage. Stage 9's
    `config.fallback_solver` defaults to `""` and Stage 12 flips it, because the seed is not the
    committed division: best-seen is no worse than a mostly one-core seed, so SA has no floor
    against greedy on committed divisions.

12. **Planning time does not replay.** The plan replays zoo captures in one process after this
    stage. SA cannot replay a capture's trajectory, and two trees cannot share a process.
    Resolution: the PR reports the time of `test_sa_cooptimizer` with
    `SA_COOPT_LARGE_CAPTURES=1` and of `test_sa_family_live.py`, run on its parent and on its
    tip in alternating processes. It is a report, not a gate, and nothing is stored.

13. **Stage 0 is dropped (owner, PR #7, 2026-11-12).** The plan's corpus, recordings, cells and
    observer go. The dual run is now `test_sa_family_live.py`: five `test_building_blocks`
    compiles on the card, with the allocator and `SaCoOptimizingSolver.plan_layout_and_core_divisions`
    patched in the test, comparing model, trace, score and decisions in one process. The trace
    digests come from `sa_trace`, a test helper, not from recordings. Commits 2 to 4 are held by
    the existing `test_sa_cooptimizer` suite. `buffer_decision` is not used.

## Open questions

1. **Do commits 2 to 4 need a check across trees?** Without recordings, nothing compares the
   search before the move with the search after it on a live compile, where generated sources
   and view relations run. `test_sa_cooptimizer` compares it only on menus.
   - Recommended: no gate. The commits are pure moves, reviewed as such, and
     `test_sa_cooptimizer` with `GoldenSolveTest` passes unchanged. In addition, the author
     runs `sa_trace` over the five live cases on the PR's parent, with the test file copied
     into that tree, and on commit 4, and attaches the two digest lists to the PR. Nothing is
     committed.

## Out of scope

- Wiring the family into `plan_allocation`, the failure flow and the bridge to commit: Stage 9.
- `commit(graph)`, which runs the post passes: Stage 10.
- Sticks in SA's domains and layout-dependent facts: Stage 11.
- Relayouts or in-place decisions as SA moves: not in this plan. SA stays without copies.
- Tuning the search, the seed or the budget: nothing here changes a plan.
- A debug greedy family: not built, and nothing here rules it out.
- Deleting `SaCoOptimizingSolver`, `SimulatedAnnealingLayoutSolver` and `cooling_schedules.py`: Stage 13.
- Default flip, the fallback flip and CI comparison: Stage 12.
