# LX allocator, Stage 4: Cost model

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0, 1, 2, with Stage 3's Phase 1 exit closed.
Needed by: Stages 5, 6, 7, 8, 9, 10, 11, 12, 13.

## Goal

When this PR merges, `SpyreCostModel`, `Sym` and the objective types exist,
`select_target` attaches a `SpyreCostModel` to the `SPYRE` target it returns, and the
printer can lower them by name. The cost model
produces, from the graph plus the encoder's domains, the same sympy expression
`CoOptimizingAllocator._solve` builds today, and the same terms as each stage of
`CpSatLayoutSolver._run`'s lexicographic ladder. A shadow check proves that solve for
solve. Nothing in production reads `graph.target.cost_model` yet: the planner, every CP
model, every SA score and every committed plan stay as they were.

## Today

### There are four objectives, not one

| path | runs when | what is minimized | unit | code |
|---|---|---|---|---|
| CP-SAT priced | `cost_expr` built and lowered (default config) | one `model.minimize` over `sum(bundle latency) + sum(RelayoutCharge)` | ns, float | `allocator.py:3320`, `:3346-3352`; `ilp_solver_ortools.py:1442-1447`, `:830-900` |
| CP-SAT ladder | `cost_expr` is `None`: `AUTO_COARSE_TILING=1` with a linear-only solver (`allocator.py:3275-3299`), the build failed (`:3321-3336`), the lowering failed (`ilp_solver_ortools.py:886-900`), or placement-only `plan_layout` (`:687-697`) | up to five solves, each locking the last | bytes, then counts | `ilp_solver_ortools.py:1449-1523` |
| SA priced | `_build_score_fn` returned a callable | `to_fixed_us(max(0, ns) / 1000) + companions` | integer fixed-point us | `sa_cooptimizer.py:853-857`, `:1441-1447` |
| SA memory-only | `cost_expr` is `None` or names an unknown symbol | `to_fixed_us(spilled bytes / rate) + companions` | integer fixed-point us | `sa_cooptimizer.py:1449-1454` |

The priced path has no `spill_cost`, no cut traffic and no tie-break stage. Cores reach
it only as prices, through split symbols and `_product_*` bound to `t.cores`
(`ilp_solver_ortools.py:855-858`). The ladder has no bundle term and no relayout charge,
and its first act is to pin every relayout copy out (`:1454-1455`). Summing the two is
not meaningful either, since they are in ns and bytes. See Deviation 1.
`ExhaustiveSearchSolver` ignores the expression (`exhaustive_search.py:129`, a TODO).

### The priced expression

`_solve` (`allocator.py:3218-3406`):

1. `pricing_ops = project_transport_read_copies(graph, {buf: [splits...]}, relayout_sources=...)`
   (`:3203-3216`, `read_copy_elision.py:548`). It reads every candidate's splits.
2. `is_lx = {buf.name: buf.sym_is_lx}` over `solver.buffers` (`:3224`). A name missing
   from this map falls back to the committed layout, a Python bool
   (`cost_features.py:1376-1379`, `:1604-1607`).
3. One `OpFeatures` per op whose buffer is a `ComputedBuffer` in the solve
   (`:3234-3240`, `_extract_op_features`, `:3408-3451`), with symbolic work slices from
   `sym_core_divs`, `sym_tiling` from `sym_tile_counts` under tiling, and the whole menu
   as `candidate_work_slices`.
4. `predict_bundles` (`cost_model.py:3104-3133`) groups by `estimate_bundles`, applies
   `charge_boundary_reads_once`, and calls `predict_ops` per bundle.
5. `cost_expr = sympify(sum(terms))`, then one `+ copy.cost_term()` per copy
   (`allocator.py:3320`, `:3347-3351`).

So the split symbols that exist, the delivery estimate, the transport projection and
the relayout table all depend on the menu, not on the graph. See Deviation 2.

The parameters are `_COST_PARAMS` (`allocator.py:170-192`). `overlap_gamma=0.46`
replaces the roofline with a partial overlap, and the zero exponents make `eff` and
`spill_derate` exactly 1.

`cost_model_pass.py` is not an objective. It prices committed concrete features after
planning with default `CostParams` (`:396-404`) and no candidate menu, reusing `predict_ops`
only. `cost_model.py` is the pure pricing library both use. Neither changes here.

### The bundle terms

`predict_ops` (`cost_model.py:2673-2920`) returns, for one bundle:

```text
t = compute + mem_t - gamma*Min(compute, mem_t) + rel + clone + transport_dma
    + loop_repeated_read + read_burst + operand_requests
```

### The ladder

`_run` (`ilp_solver_ortools.py:1449-1523`), over every wrapper, relayout copies included:

| stage | sense | terms | lock after it |
|---|---|---|---|
| residency | min | `sb.spill_cost() * (1 - in_buffer)` plus `_cut_traffic_terms` | `sum <= round(obj)`, if a later stage runs (`:1470-1471`) |
| cut tiebreak | min | `sb.cut` where not `None` (`:1419`) | `sum <= cuts`, if `core_terms` (`:1497-1498`) |
| parallelism | max | `sb.cores` (`:1489`) | `sum >= occupancy` (`:1509`) |
| division shape | min | `sb.core_cost`, the summed squared splits (`:446-448`) | `sum <= round(obj)`, if `tile_terms` (`:1520-1521`) |
| tile count | min | `tile_count_<name>`, only where the menu's counts differ (`:1174-1202`) | none |

A lock is added right after a stage's solve exactly when a later stage runs. That one
rule covers all four.

`spill_cost` (`:366-388`): `(reads_served + intermediate) * size`, where
`reads_served = read_count - first_use_is_read`. `intermediate` is
`boundary == Intermediate`, or `not first_use_is_read` for a plain
`LifetimeBoundBuffer`, which has no boundary, so a placement-only graph output is priced
as an intermediate. On the joint path `size` is the total and is unclamped, so an unsized
buffer (`mem_usage` -1, `utils.py:226`, `allocator.py:4287`) gives a negative coefficient
on a pinned-out var. On the placement path `size` is already per core: `max(0,
size_per_core)` (`allocator.py:1582`), or `_input_footprint` for an input (`:1616`,
`:1731-1742`). No relayout reaches that path on this engine: `supports_paired_buffers()`
is False (`ilp_solver_ortools.py:632-633`), so `_prepare_buffers` calls
`_generate_buffers(graph)` with no plans (`allocator.py:1069-1075`).

`_cut_traffic_terms` (`:1275-1323`), per buffer with `cut` and `nest_end`, in bytes:
`2*s*(cut - resident_cut) + (I + U)*s*resident_cut + sum_kid s*(via_full_kid and in_buffer)`,
where `I` is 1 for an intermediate and `U = max(0, read_count - len(kids))` counts
readers the model does not carry. `resident_cut_<n>` and `reads_full_<n>_<kid>` are
created inside this method.

`nest_end` exists only for an op output with a tiled candidate
(`_add_tiling_groups`, `:1134-1140`). `cut` exists for every divided wrapper, copies
included, once any candidate anywhere is tiled (`_add_tiling`, `:951-956`, `:961`,
`:978-1023`).

### The SA evaluator

`_build_score_fn` (`sa_cooptimizer.py:807-857`) binds each free symbol to a getter over
`(chosen, resident)`: `is_lx` from `name in resident`, `division` from
`chosen[idx].menu_index`, splits and tile counts from the config. It calls
`sympy.lambdify(free, expr, modules="math")`, with `free` sorted by name. Unknown symbol:
returns `None`, the memory-only signal. SA does no rewriting, so it evaluates the exact
`log`, `floor` and fractional powers that CP-SAT approximates. `linear_cost_only()` is
False (`:731-733`), so SA keeps its expression under `AUTO_COARSE_TILING=1` and CP-SAT
does not. Hint-tiled and `for_each_tile` graphs with the flag off keep it on both.

`_spill_cost` (`:1299-1316`) duplicates `spill_cost` with `max(0, size)`.

`_companion_charges` (`:1325-1404`) is SA's cut traffic, added outside the expression
and rounded on its own (`:1406-1410`). With `s = size` and `r` readers outside the run:

| state | SA companions | CP `_cut_traffic_terms` |
|---|---|---|
| spilled intermediate | `2s` | `2s` |
| spilled graph output | `2s` | `2s` |
| resident intermediate | `s + r*s` | `s + U*s + r*s` |
| resident graph output | `r*s` | `U*s + r*s` |

The two formulas differ only by `U`. They also bind `cut` differently in one case: a
tiled op read only by an extern kernel is a cut in CP-SAT
(`ilp_solver_ortools.py:1013`), and escapes nothing in SA (`:1396-1399`).

### Symbols and the printer

Names are built on the buffers: `is_lx_<buf>` (`plan_solver.py:181-183`),
`split_<buf>_<key>` (`:429-445`), `tile_<buf>_<o|r><dim>` (`:448-476`),
`division_<buf>` (`:417-426`, `:479-484`). A split key is the op's iteration symbol, a
coefficient in test fixtures, or the string `relayout_copy` for a copy (`:896`).

`_minimize_cost_expr` (`ilp_solver_ortools.py:837-862`) fills two dicts:

- `sym_map`: name to CP var, plus `_division_of_<division sym>` to the wrapper, plus
  `_product_<sorted split names>` to `t.cores`. The printer adds `log2_*`, `inv_*`,
  `_product_<var indices>` and `_imp_<expr>` entries as it goes.
- `buffer_map`: split, tile or product name to `(wrapper, candidate values)`.

The printer reads names by prefix: `split_` and `tile_` mark a factor
(`cpsat_expr_printer.py:405-409`), `log2_` and `inv_` mark derived symbols
(`:242-243`, `:789-791`), and `_product_` marks the core count (`:502-521`). It reaches
through the wrapper to `.buffer.core_divisions`, `.buffer.sym_core_divs` and
`.buffer.sym_division` (`:249`, `:840-841`, `:881-896`).

An unknown name does not fail. `_print_Symbol` raises `NotImplementedError`, a
`RuntimeError`, which `_minimize_cost_expr` catches and turns into the ladder
(`ilp_solver_ortools.py:886-900`). `_print_RelayoutCharge` and `_print_AppliedUndef`
return 0 for a term they cannot place (`cpsat_expr_printer.py:837-839`, `:878-883`,
`:891-892`).

### Scaling today

| constant | value | where | effect |
|---|---|---|---|
| `_CORE_LOG_SCALE` | 32.0 | printer `:53`, `:260`, `:795` | `log2_x = round(32*log2 v)` |
| `_CORE_INV_SCALE` and cap | 1024, widened past 32 | printer `:57-60`, `_inv_scale` `:411-459` | `inv_x = scale // v`; the scale is the LCM of the menu's values when under the cap, so it depends on the domain |
| `_MAX_PRODUCT_BOUND` | `2**30` | printer `:62`, `:330`, `:747` | a wider product raises, a wider table lowers structurally |
| `_truncate_floats_min` | `m = 10000` | printer `:622-647` | `Min`/`Max` operands rounded to 1e-4 |
| quarter-power fit | two segments | printer `:477-496` | `split**0.25`, core splits only |
| relayout price | `Integer(round(ns))` | `plan_solver.py:847-850` | banker's rounding of a float, to integer ns |
| decode delivery price | `int(round(...))` | `cost_model.py:1440-1442` | integer ns |
| objective coefficients | float | `model.minimize` | lands in `floating_point_objective`; CP-SAT scales it |
| SA ns | `to_fixed_us(max(0.0, ns) / 1000.0)` | `sa_cooptimizer.py:855` | one round-half-up, whole expression |
| SA bytes | `to_fixed_us(bytes / hbm_bytes_per_us())` | `:1408-1410`, `:1454` | rounded apart from the ns part |
| scale | `US_FIXED_POINT_SCALE = 1_000_000` | `utils.py:949-976` | picosecond resolution |

### Measurements

Device-free, on the 47 captures with an expression in
`probes/cpsat-graph/deenum/runs/caps_default` and `caps_granite_default`, re-lowered by
this snapshot's printer. The captures date from 2026-10-21 (main `82edd8f6`), so they
predate some terms: none has an `AppliedUndef`, the decode price. Treat the sizes as
lower bounds. `tree` counts nodes with repeats, `dag` distinct nodes.

| capture | buffers (copies) | top-level terms | tree | dag | rewrite s | lowering s | `+=` charges s | bulk `Add` s | lambdify s | eval us |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_swiglu_128 | 60 (54) | 81 | 6,093 | 1,128 | 0.60 | 0.67 | 0.016 | 0.0002 | 0.43 | 30 |
| sdpa_gqa_1x32x8 | 203 (193) | 228 | 21,483 | 2,088 | 0.93 | 1.07 | 0.086 | 0.0005 | 4.35 | 268 |
| attn_block_256 | 196 (178) | 243 | 14,654 | 2,744 | 2.22 | 2.37 | 0.119 | 0.0007 | 3.26 | 83 |
| transformer_block_256 | 303 (266) | 385 | 22,927 | 4,465 | 4.49 | 4.76 | 0.262 | 0.0011 | 7.48 | 125 |
| granite_micro.13 | 79 (28) | 169 | 16,904 | 4,385 | 4.60 | 4.79 | 0.023 | 0.0008 | 0.82 | 26 |
| granite_micro.3 | 312 (261) | 409 | 31,156 | 6,220 | 5.16 | 5.68 | 0.284 | 0.0012 | 10.80 | 134 |

- Across all 47: 47.0 s of rewrite in 51.1 s of lowering, and `+=` 2.23 s against bulk
  0.017 s. The rewritten tree is 0.94x to 1.71x the raw tree.
- Bulk `sympy.Add(rest, *charges)` gave the same expression as the `+=` loop on all 47:
  equal under `==`, `srepr` and against the captured `cost_expr`.
- Synthetic, 5,789 `RelayoutCharge`s over 500 terms (the count the comment at
  `plan_solver.py:493` reports for a 304-op graph): `+=` 39.24 s, bulk 0.02 s, equal.
- On granite_micro.3, lambdify takes 0.31 s for the bundles alone (eval 16 us), 1.98 s
  for the charges alone, and 10.80 s for both together. I did not find out why the sum
  costs more than its parts. SA is never handed a charge today.
- Ladder terms through sympy: 1,000 nodes 0.08 s to build the terms, 0.21 s more to
  `Add` them; 20,000 nodes 1.6 s and 2.1 s.
- The expression's build (features, `predict_bundles`) needs a live graph, so I could
  not measure it. `sa-vs-cpsat-REPORT.md` gives 21.2 s summed over the zoo and 14.3 s
  over Granite on main `82edd8f6`, cost dump included.

Facts about CP-SAT 9.15.6755 that this plan relies on, checked directly: an objective
and a `<=` constraint serialize with vars sorted by index whatever the term order, a
zero coefficient vanishes, and `minimize(7)` gives an offset alone. So a linear stage
built through sympy has the same proto as the Python `sum` it replaces.

Relayout prices as a function of split values: across 3,380 copies and 3,996 divided
buffers in those captures, no copy has two menu positions with the same split and tile
values but different prices, and no menu has two positions with the same identity.
`division_*` is read by `RelayoutCharge` only; `KroneckerDelta` appears in no expression.

## Design

### What moves and what is copied

The orchestration is copied, not moved, so the legacy path stays an independent oracle
for the shadow check until Stage 13. Everything below it stays shared.

| old | new home | moves or copied | who deletes the old |
|---|---|---|---|
| `_solve`'s pricing steps, `_extract_op_features`, `_pricing_operations` | `SpyreCostModel.bundle_terms` | copied | Stage 13, with `CoOptimizingAllocator` |
| `RelayoutCopyBuffer.cost_term` | `SpyreCostModel._charge` | copied, re-keyed | Stage 13 |
| `spill_cost`, `_spill_cost` | `SpyreCostModel.spill_cost` | copied, a test pins all three equal | Stage 13 |
| `_cut_traffic_terms` | `SpyreCostModel.cut_traffic` | copied | Stage 13 |
| ladder term lists in `_run` | `SpyreCostModel.ladder` | copied | Stage 13 |
| `_COST_PARAMS` | `ALLOCATOR_COST_PARAMS` in `lx_allocator/cost.py` | moved; `allocator._COST_PARAMS` stays an alias (`test_cost_model_decode.py:820`) | stays |
| `sa_cooptimizer._work_slices` | `utils.division_work_slices` | moved; alias stays | Stage 13 |
| `sym_is_lx`, `sym_core_divs`, `sym_tile_counts` name strings | `Sym` | moved; the properties delegate | Stage 13 |
| lambdify and rounding in `_build_score_fn` | `utils.compile_fixed_point` | moved; `_build_score_fn` delegates with `strict=False`, Stage 8's `compile_objective` wraps it | Stage 13 |
| `ScratchpadAllocator._input_footprint` | `utils.input_footprint` | moved; the method delegates | Stage 13 |
| `_minimize_cost_expr`'s binding loop | `bind_division_owner` | moved | stays, Stage 5 calls it |
| `predict_bundles`, `extract_op_features`, every `cost_model.py` term | unchanged | reused | never |

### `Sym`

`Sym` keeps today's strings exactly. This is not cosmetic. `Add` and `Mul` sort by
`Basic.compare`, which reads names, so a new spelling reorders printing, which reorders
auxiliary variables and breaks the proto diff. The strings also appear in proto variable
names (`log2_*`, `inv_*`, `relayout_price_*`). Each method returns a `sympy.Symbol`,
because the assumptions are part of the contract: `_tighten_integer_bound` and
`_piecewise_canonical` read them, and a symbol with other assumptions is a different
symbol. A subject is an `OpNode` or a buffer name, because relayout copies and LX
prefetches have no node.

`scratchpad/lx_allocator/graph/symbols.py` imports sympy only, so it sits inside Stage 1's
graph boundary. All methods are static:

```python
SplitKey = sympy.Symbol | int | str   # a key of CoreDivision.splits
Subject = "OpNode | str"              # uses subject.name when given a node

class Sym:
    RESIDENT = "is_lx_"; SPLIT = "split_"; TILE = "tile_"
    CORES = "cores_"; SHAPE = "shape_"; TILES = "tiles_"
    CUT = "cut_"; RESIDENT_CUT = "resident_cut_"; READS_FULL = "reads_full_"
    RESTICK = "restick_"; STICK = "stick_"
    LOG2 = "log2_"; INV = "inv_"; PRODUCT = "_product_"

    def resident(subject: Subject) -> sympy.Symbol: ...
    def split(subject: Subject, key: SplitKey) -> sympy.Symbol: ...
    def tile(subject: Subject, dim: int, is_reduction: bool) -> sympy.Symbol: ...
    def cores(subject: Subject) -> sympy.Symbol: ...          # likewise shape, tiles,
    def cut(subject: Subject) -> sympy.Symbol: ...            # and resident_cut
    def reads_full(producer: Subject, consumer: Subject, listing: int = 0) -> sympy.Symbol: ...
    def restick(edge_name: str, row: str | None = None) -> sympy.Symbol: ...
    def stick(subject: Subject) -> sympy.Symbol: ...
    def split_product(split_names: Iterable[str]) -> sympy.Symbol: ...
    def is_factor(symbol: sympy.Symbol) -> bool: ...   # split_ or tile_, for _is_split_sym
    def is_known(symbol: sympy.Symbol) -> bool: ...    # any prefix above
```

| symbol | name | assumptions | means | in which objective |
|---|---|---|---|---|
| `resident(n)` | `is_lx_<n>` | integer, nonnegative | node, LX prefetch or copy is in LX; for a copy, the shuffle fires | priced, ladder |
| `split(n, k)` | `split_<n>_<k>` | integer, positive | split factor on axis `k` | priced |
| `tile(n, d, r)` | `tile_<n>_<o\|r><d>` | integer, positive | tile count on output or reduction axis `d` | priced (SA only, under tiling) |
| `cores(n)` | `cores_<n>` | integer, positive | core count | ladder |
| `shape(n)` | `shape_<n>` | integer, nonnegative | sum of squared splits | ladder |
| `tiles(n)` | `tiles_<n>` | integer, positive | tile count of the chosen tiling | ladder |
| `cut(n)` | `cut_<n>` | integer, nonnegative | `n` is tiled and is copied out | ladder, companions |
| `resident_cut(n)` | `resident_cut_<n>` | integer, nonnegative | cut and resident | ladder, companions |
| `reads_full(p, c, listing)` | `reads_full_<p>_<c>`, then `_<listing>` past the first | integer, nonnegative | `c` reads `p`'s full buffer and `p`'s tile is resident | ladder, companions |
| `restick(e, row)` | `restick_<e>`, or `restick_<e>_<row>` | integer, nonnegative | a restickify on edge `e`, in the table row keyed `row` | Stage 6 binds it, Stage 11 prices it |
| `stick(n)` | `stick_<n>` | integer, nonnegative | `n`'s solved stick | Stage 11 |
| `split_product(names)` | `_product_<sorted names>` | none | printer only, as today | printer |
| derived | `log2_<factor>`, `inv_<factor>`, `inv__product_...` | integer, nonnegative | made by the rewrite, never by the cost model | printer |

Rules:

- `reads_full` is per listing in `parents`, because `_cut_traffic_terms` makes one
  variable per entry of `kids` (`ilp_solver_ortools.py:1295`, `:1315-1322`).
  `NodeDomain.readers` keeps repeats, so `U` and the terms count what today counts. None
  of the 1,581 listings in 89 captures repeats, so only a test exercises `listing > 0`.
- `tiles_` is new on purpose. Today's CP var is `tile_count_<n>`, which starts with
  `tile_` and would read as a factor to `_is_split_sym`. Ladder symbols are only ever
  bound, never printed by name into the proto, so the spelling is free.
- `cores_` is distinct from `_product_`. The product name is built from split names, so
  every node with no split key shares `_product_` (`ilp_solver_ortools.py:856`). That is
  harmless where nothing reads it and wrong for a per-node ladder term.
- `division_<n>` is not in `Sym`. It is a menu position, so it stays inside the encoder
  (`plan_solver.division_symbol`). The cost model never emits it.
- A new kind of symbol needs a prefix here, a binding in each family and a row in this
  table. The cost model emits neither `restick_` nor `stick_` in this stage.

### A relayout charge with no menu index

`RelayoutCharge(is_lx, division, *prices)` is a table over menu positions, so it cannot
appear in the graph layer's output. The cost model emits this instead:

```python
class SplitKeyedCharge(sympy.Function):
    """``is_lx * price(selectors)``. Args: is_lx, source (sympy.Str), selectors
    (Tuple of split and tile symbols), rows (Tuple of Tuple(v..., price)).
    A selector value with no row prices 0. ``eval`` folds at is_lx == 0, or once
    every selector is numeric. ``_imp_`` evaluates it for lambdify."""
```

Rows hold what the table holds today, `Integer(round(ns))`, with a zero price kept as an
explicit row. Rows are sorted, so equal tables compare equal. `source` is a `Str` so it
adds no free symbol. The measurement above shows the key is sound on every capture.

The OR-Tools family resolves this to today's node before printing, in
`_SympyExprToCpSat._resolve_keyed_charges`, first thing in `convert`:

- Take top-level `Add` args only; `objective` puts a charge nowhere else.
- For each menu position `i` in `range(owner.menu_size)`, read each selector's value
  from `buffer_map[selector.name][1][i]`, which gives one key tuple per position.
- `index_charge(charge, keys)`, a module-level function in the printer, builds
  `table = [row price or 0 for positions 0..last]`, where `last` is the last position
  with a row. That reproduces `range(max(prices) + 1)` from `plan_solver.py:849`,
  trailing explicit zeros included.
- Rebuild `sympy.Add(*args)`.

The result is `RelayoutCharge(is_lx, division_symbol(source), *table)`, argument for
argument. The canonical sort then sees what it sees today, and `_print_RelayoutCharge`
adds the same `relayout_price_*` and `relayout_charge_*` variables in the same order. A
new class name or a new argument shape left in the `Add` would change that sort, which
is why it is resolved rather than printed directly.

### Domains

`objective` needs what the encoder's table offers, in graph vocabulary. Indices stay out:
candidate order is not meaningful here, and it is used as a set.

```python
@dataclass(frozen=True)
class NodeDomain:
    name: str
    candidates: tuple[CoreDivision, ...]     # empty: a placement-only node
    readers: tuple[str, ...] = ()            # modelled op consumers, one per listing
    extra_tile_keys: tuple[tuple[int, bool], ...] = ()   # a generated space's dims

    @property
    def split_keys(self) -> tuple[SplitKey, ...]: ...    # first-seen order
    @property
    def tile_keys(self) -> tuple[tuple[int, bool], ...]: ...
    @property
    def selectors(self) -> tuple[sympy.Symbol, ...]: ... # split then tile, sorted by name
    def selector_values(self, candidate: CoreDivision) -> tuple[int, ...]: ...
    @property
    def tileable(self) -> bool: ...   # an op output with a tiled candidate

@dataclass(frozen=True)
class RelayoutOffer:
    copy_name: str                    # relayout_copy_name(source, group)
    source: str
    prices_ns: Mapping[tuple[int, ...], float]   # NodeDomain.selector_values -> ns
    size: int
    read_count: int
    first_use_is_read: bool = False
    boundary: BufferType = BufferType.Intermediate

@dataclass(frozen=True)
class CostDomains:
    nodes: Mapping[str, NodeDomain]   # every name in the solve, copies included
    relayouts: tuple[RelayoutOffer, ...] = ()

def domains_from_buffers(buffers: Sequence[LifetimeBoundBuffer]) -> CostDomains: ...
```

`domains_from_buffers` is the adapter from today's solver input. It computes `readers`
as `_get_children` and `_op_readers` do, and `tileable` as the `position` filter in
`_add_tiling_groups` does. `nodes` decides which `is_lx_*` symbols exist: an input with
no LX prefetch is not in the solve (`allocator.py:4237-4238`), so it has no symbol and prices
as a constant, as today.

### `SpyreCostModel`

The target holds the cost model, so `encode` takes none. Every caller reaches it as
`graph.target.cost_model`, and the cost model takes no target argument: any card size it
reads comes from `graph.target`, the same card the encoder and `check_decisions` read.
The `CostModel` protocol, with the public methods below, and the per-card table live in
`cost.py`:

```python
COST_MODELS: Mapping[str, CostModel] = {"spyre": SpyreCostModel()}   # keyed by Target.name
```

`graph/` must not import `cost.py`, which reads today's cost features, so `SPYRE.cost_model`
stays `None` in `graph/target.py`, and `target.py` names `CostModel` only in an
`if TYPE_CHECKING:` import, which the boundary test skips. This stage adds one line to Stage
2's `select_target`, which sits outside `graph/`: its `replace` call also passes
`cost_model=COST_MODELS.get(target.name)`. Every compiled graph therefore reaches
`SpyreCostModel()` as `graph.target.cost_model`. A derived target swaps the model with
`replace(target, cost_model=SpyreCostModel(traffic_only=True))`, which is how committed mode
reaches the encoder (Stage 9).

```python
class Sense(Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"

class Unit(Enum):              # in symbols.py, so utils.py can name it
    NS = "ns"
    BYTES = "bytes"
    COUNT = "count"

@dataclass(frozen=True)
class ObjectiveStage:
    name: str     # residency | cut tiebreak | parallelism | division shape | tile count
    sense: Sense
    unit: Unit
    terms: tuple[sympy.Expr, ...]   # integer-linear over Sym names; never summed

class SpyreCostModel:          # implements CostModel
    def __init__(self, params: CostParams = ALLOCATOR_COST_PARAMS, *,
                 traffic_only: bool = False) -> None: ...

    def objective(self, graph: SpyreOptimizationGraph, domains: CostDomains) -> sympy.Expr:
        """Today's ``cost_expr``: bundle terms plus relayout charges, in ns.
        Raises whatever feature extraction raises."""

    def priced_objective(self, graph, domains, *, linear_only: bool) -> sympy.Expr | None:
        """``objective``, or ``None`` when the solve has no priced expression."""

    def bundle_terms(self, graph, domains) -> list[tuple[list[str], sympy.Expr]]: ...
    def ladder(self, graph, domains) -> tuple[ObjectiveStage, ...]: ...
    def spill_cost(self, node, *, boundary_known: bool = True, clamp_unsized: bool = False,
                   size: int | None = None) -> int: ...
    def cut_traffic(self, node, domains, *, count_unmodelled: bool = True) -> list[sympy.Expr]: ...
```

`bundle_terms`, in order. It reads `graph.ir`, and no buffer.

1. `pricing = project_transport_read_copies(ir, {n: [cd.splits for cd in d.candidates]},
   relayout_sources={o.source for o in domains.relayouts})`. Lazy import, as
   `allocator.py:3204` does.
2. `is_lx = {name: Sym.resident(name) for name in domains.nodes}`.
3. For `op` in `ir.operations`, in order: skip unless the buffer is a `ComputedBuffer`,
   and `op.name` is in `domains.nodes` and in `pricing` by `get_name()`. This iterates
   what `mem_usage_by_buf` iterates (`utils.py:242`) without its core-count work, so it
   does not depend on a committed split, which suits #5695.
4. `work_slices = {s: split_of.get(s, 1) for s in iteration_space_from_op(op)}` where
   `split_of = {k: Sym.split(name, k) for k in domain.split_keys}`.
5. Under `config.auto_coarse_tiling`, build `sym_tiling` as `allocator.py:3435-3442`
   does, with `try_resolve_tile_axis_loop_vars`. `Unsupported` is a `RuntimeError`
   (`errors.py`), so it reaches the same `except`.
6. `extract_op_features(op, work_slices, is_lx=is_lx, sym_tiling=sym_tiling or None,
   candidate_work_slices=[division_work_slices(op, cd) for cd in domain.candidates])`.
7. `predict_bundles(pricing, features, params=self.params)`.

`objective` is `sympy.Add(*(t for _, t in bundle_terms), *charges)`, with one
`SplitKeyedCharge(Sym.resident(o.copy_name), Str(o.source), selectors, rows)` per offer.

`priced_objective` returns `None` first when `traffic_only` is set, then keeps today's
branches in today's order:

1. `config.auto_coarse_tiling and linear_only`: return `None`. `chooses_tilings()` is
   true for both families, so it adds nothing here.
2. `objective` raising `ValueError`, `RuntimeError` or `TypeError`: warn with the
   traceback as `allocator.py:3322-3328` does, re-raise if not
   `config._cpsat_warn_on_cost_expr`, else return `None`.
3. A free symbol that is not `Sym.is_known`, such as a dynamic-shape symbol: warn and
   return `None`. This is today's outcome on both engines (`sa_cooptimizer.py:847`,
   `ilp_solver_ortools.py:886`), found earlier.

`spill_cost` and `cut_traffic` are as in Today. `spill_cost` reads `size`, `read_count` and
`first_use_is_read` from an `OpNode` or a `RelayoutOffer`. For a node, `cost.py` computes
the boundary: `Input` when `not node.runs`, `Output` when `node.graph_output`, otherwise
`Intermediate`. `boundary_known=False` applies the plain-buffer rule, `clamp_unsized=True`
gives SA's `max(0, size)`, and `size` replaces `node.size`.

`ladder` builds, from `domains`:

- `residency` (`BYTES`): `spill_cost * (1 - Sym.resident(x))` for every name in
  `domains.nodes`, read from `graph.node(x)` or, for a copy, its `RelayoutOffer`, then
  `cut_traffic` for every `tileable` node.
- `cut tiebreak` (`COUNT`): `Sym.cut(x)` for every divided name, copies included, when
  any candidate anywhere is tiled.
- `parallelism` (`MAXIMIZE`, `COUNT`): for every divided name, `Sym.cores(x)`, or the
  integer when the node has one candidate (`ilp_solver_ortools.py:454-465`).
- `division shape` (`COUNT`): `Sym.shape(x)` or the integer, the same way.
- `tile count` (`COUNT`): `Sym.tiles(x)` where the candidates' `tile_count`s differ.

An empty stage is omitted, `division shape` runs iff `parallelism` does, and `tile count`
runs only after them. A stage is locked right after its solve iff a later stage exists,
at `round(value)`, `<=` for a minimize and `>=` for a maximize.

With `traffic_only=True`, `ladder` returns `residency` alone, priced as `plan_layout`
prices it: no `cut_traffic`, `boundary_known=False`, and `size` from `traffic_sizes`. An
offer keeps its own `size`. This is committed mode's objective.

```python
def traffic_sizes(graph: SpyreOptimizationGraph) -> dict[str, int]:
    """Per-core bytes by name, as ``plan_layout``'s buffers carry them:
    ``max(0, size_per_core)`` from ``mem_usage_by_buf(graph.ir)``, and for an input
    ``input_footprint(graph.ir, name, ncores)``, with ``ncores`` from
    ``get_ncores_for_buffers(graph.ir)``."""
```

### Term table

Every term of today's objective, and what `SpyreCostModel` does with it. "Reused" means
the code in that row is called as it stands.

| term | source | symbols and sympy shape | in `SpyreCostModel` |
|---|---|---|---|
| HBM read, write | `_fused_hbm_bytes`, `cost_model.py:1346` | linear in `1 - is_lx`, `Max` for shared inputs; `Piecewise` from replication | reused |
| peak memory | `_peak_mem_ns`, `:2256` | `(r+w)/bw + c*Min(r, w)`, unevaluated `Min` | reused |
| per-op bandwidth | `_eff_bw`, `:2705-2721` | off when any `is_lx` is symbolic, so the default turnaround applies | reused |
| standalone reduction | `:2742-2764` | `_select` over `tiles_output_dim` | reused |
| store excess | `_store_core_excess_ns`, `:2262` | `1/(cores*rate)` under `Max` | reused |
| replicated reads | `_replicated_operand_reads`, `:1389` | bytes over `cores`, `Piecewise` on replication | reused |
| shared read excess | `_shared_operand_read_excess`, `:1536` | `Max`, multicast penalty | reused |
| partitioned read excess | `_partitioned_operand_read_excess`, `:1582` | `Piecewise`, `Max` | reused |
| decode weight excess | `_decode_weight_delivery_excess`, `:1486` | `AppliedUndef` with `_imp_`, gated by residency | reused |
| underfill, spill derates | `:2828-2842` | exactly 1 under `ALLOCATOR_COST_PARAMS` | reused |
| matmul compute | `_matmul_ns_upstream`, `:2097`, through `_matmul_execution_cost`, `work_division.py:2389` | `log2` of splits under `Max`, powers of `_PT_EFFICIENCY_EXPONENT` (0.25) | reused |
| fused reduction compute | `_fused_reduction_compute_ns`, `:2002` | `elems/cores/rate`, `Max` | reused |
| overlap | `predict_ops:2902` | `-gamma*Min(compute, mem_t)`, unevaluated | reused |
| relayout inside a bundle | `relayout_ns`, `:1313` | concrete, 0 at planning | reused |
| LX prefetch load | `_clone_in_bytes`, `:1373` | linear in `is_lx`, once per input | reused |
| transport DMA | `_transport_dma_excess_ns`, `:2436` | `Piecewise` over splits | reused |
| loop repeated read | `_loop_repeated_read_excess_ns`, `:1737` | as extracted | reused |
| read burst | `_read_burst_excess_ns`, `:2577` | `Piecewise` over `Eq(cores, n)` | reused |
| operand requests | `_loop_operand_request_excess`, `:2510` | `Max` composition | reused |
| relayout charge | `cost_term`, `plan_solver.py:834` | one `RelayoutCharge` per copy | `_charge`: `SplitKeyedCharge` |
| spill | `spill_cost`, `ilp_solver_ortools.py:366` | CP linear | `spill_cost`, stage `residency` |
| cut traffic | `_cut_traffic_terms`, `:1275` | CP linear | `cut_traffic`, stage `residency` |
| cut count | `:1419` | CP linear | stage `cut tiebreak` |
| cores | `:1489` | CP linear or int | stage `parallelism` |
| shape | `:1493` | CP linear or int | stage `division shape` |
| tile count | `:1174` | CP linear | stage `tile count` |
| companions | `sa_cooptimizer.py:1325` | Python loop, bytes | `cut_traffic(count_unmodelled=False)`, SA binds it |

### Printer changes

Each keeps the proto byte-identical on the legacy path.

1. **Prefixes come from `Sym`.** `_is_split_sym`, `_evaluate_candidates`,
   `_inv_log_sym` and `_print_Symbol` use the constants, and `_is_split_sym` calls
   `Sym.is_factor`. No string changes.
2. **`DivisionOwner` replaces reaching into `.buffer`:**

   ```python
   class DivisionOwner(Protocol):
       division: "cp_model.IntVar"
       selector_name: str                    # today's buffer.sym_division.name
       cores: Any; cores_used: Sequence[int]
       cp_core_divs: Mapping; cp_core_divs_raw: Mapping
       cp_tile_counts: Mapping; cp_tile_counts_raw: Mapping
       split_symbols: Mapping[SplitKey, sympy.Symbol]
       tile_symbols: Mapping[tuple[int, bool], sympy.Symbol]
       menu_size: int
       def division_is(self, i: int) -> Any: ...
       def splits_at(self, i: int) -> Mapping[SplitKey, int]: ...
   ```

   `_CoreDivisionBufferWithCpVars` gains the four missing members as properties. Lines
   249, 840-841 and 881-896 use them. `selector_name` keeps the proto variable name
   `<func>_division_<name>_<n>` at line 896.
3. **`bind_division_owner(sym_map, buffer_map, owner)`** is the body of
   `ilp_solver_ortools.py:843-862`, moved. `sym_map[sym_is_lx.name] = in_buffer` stays
   in the caller. `OpVars` implements `DivisionOwner`, `bind_symbols` (Stage 5) calls this
   per node, and `Encoding.symbols` is `sym_map`. `buffer_map` stays private to the OR-Tools
   family's encoding.
4. **`strict`.** `cost_expr_to_cpsat(model, sym_map, buffer_map, cost_expr, *,
   strict=False)`, and the same keyword on `_SympyExprToCpSat.__init__`, whose three
   positional arguments stay because `test_scratchpad_solver.py` calls it directly.
   Under `strict=True`, these raise `UnknownSymbolError` rather than print 0 or raise
   `NotImplementedError`: an unbound symbol in `_print_Symbol`, a missing owner in
   `_print_RelayoutCharge` and `_resolve_keyed_charges`, and the three zero returns in
   `_print_AppliedUndef`. `UnknownSymbolError` subclasses `KeyError`, so the
   `except (RuntimeError, TypeError, ValueError)` at `ilp_solver_ortools.py:886` does
   not swallow it. The legacy caller stays non-strict, and `OrToolsFamily.encode` lowers
   strict.
5. **`_resolve_keyed_charges`**, as above, plus `_print_SplitKeyedCharge`, which raises.
   A keyed charge reaching print means it was nested, which `objective` never does.
6. **`linear_terms_to_cpsat(sym_map, terms)`** lowers a ladder stage: for each term,
   for each `Add` arg, `as_coeff_Mul`, then an integer coefficient times a bound symbol
   or a constant. Anything else raises `ValueError`, and an unbound name raises
   `UnknownSymbolError`. There is no rewrite, no `expand` and no new variable. The
   coefficients stay Python ints, and the proto matches today's `sum` (checked above).

`_rewrite`, every `_print_*` lowering, `_division_table`, `_inv_scale` and all scaling
constants are otherwise untouched.

### Compiling one objective into both families

Both families consume the raw expression. CP-SAT's approximations stay inside
`_rewrite`, and SA evaluates the raw form, so their values keep differing by the
documented error (`cpsat_expr_printer.py:52-57`, `sa-vs-cpsat-REPORT.md` item 4). That
difference is today's and is preserved.

`compile_fixed_point` is the kernel of `_build_score_fn`, moved to `utils.py`:

```python
def compile_fixed_point(
    expr: sympy.Expr,
    getters: Mapping[sympy.Symbol, Callable[..., int]],
    *,
    unit: "Unit",
    strict: bool,
) -> Callable[..., int] | None:
    """``state -> fixed-point int``. ``free`` sorted by str, as today.
    NS:    to_fixed_us(max(0.0, v) / 1000.0)
    BYTES: to_fixed_us(v / hbm_bytes_per_us())
    An unbound symbol: None when not strict, UnknownSymbolError when strict."""
```

`_build_score_fn` becomes its `value_of` loop plus
`compile_fixed_point(cost_expr, value_of, unit=Unit.NS, strict=False)`, and catches the
same four exceptions.

Stage 8's `compile_objective` wraps this with `unit=Unit.NS,
strict=not config._cpsat_warn_on_cost_expr`. SA's score has two parts rounded apart. The
second is the companion loop, which Stage 8 keeps, and `cut_traffic(count_unmodelled=False)`
is that loop as an expression. Summed over nodes and compiled at `Unit.BYTES`, with
`cut_<n>` bound to "tiled and escapes its run", `resident_cut_<n>` to that and resident,
and `reads_full_<p>_<c>` to "`c` outside `p`'s run and `p` resident", it equals
`_companion_bytes` by the state table under Today, and a test here holds the two equal.
`count_unmodelled` stays through Stage 8's parity gate. Charging `U` in both families is a
listed rule change for a later PR.

### Scaling, preserved exactly

- CP-SAT: every constant in the Scaling table stays in the printer, and every value it
  depends on (`_inv_scale`'s menu values) still comes through `buffer_map`. Since the
  resolved expression equals `cost_expr` and the binding code is the same code, the
  lowering is the same computation.
- Relayout prices: `RelayoutOffer.prices_ns` keeps the raw float, and `_charge` applies
  the same `sympy.Integer(round(p))`. Rounding twice, or rounding before merging, would
  change a `.5` case.
- SA: the rounding is the moved code, and rounding stays once per part, not once per
  term. Summing terms in another order is not a risk at this level, because the
  expression is equal and lambdify prints it in `Add.args` order.
- Ladder: coefficients are exact integers in `sympy.Integer`, converted with `int`.
  `size` is not clamped on the joint path.

### Keeping the expression cheap

Guards are sizes, not times, per the issue-5682 guards. granite_micro.3 has about 50 op
buffers. Scaled linearly to 2,000 ops that is about 1.2 million tree nodes and over three
minutes of rewrite. This is an extrapolation, not a measurement, but it is why the
items below matter.

1. **One `sympy.Add` call, never a loop.** `objective` builds its sum in one call. This
   removes the only quadratic step in today's build: 39.24 s to 0.02 s at 5,789 charges.
   The legacy allocator takes the same fix for its charges in its own commit,
   `sympy.Add(cost_expr, *charges)`, since the result was equal on all 47 captures. The
   `sum` over bundle terms stays in the legacy path, so the shadow check also tests
   `Add` against a left fold over floats.
2. **Tables are opaque nodes.** `SplitKeyedCharge`, like `RelayoutCharge` and
   `AppliedUndef`, is one node `expand` cannot open. The comment at `plan_solver.py:493`
   gives the alternative: a 39,948-term objective and 183 to 305 s of rewriting.
3. **Keep the lazy `Min`/`Max`** (`_lazy_min`, `_LazyMin`). Nothing here evaluates one.
4. **Ladder stages are not summed and not rewritten.** They hold term tuples and go
   through `linear_terms_to_cpsat`, which saves the 2.1 s `Add` at 20,000 nodes and all
   of `_rewrite`. Term construction, 80 us each, is the remaining sympy cost on a path
   that has none today. It is under 0.1 s at 1,000 nodes.
5. **Encode once.** A family lowers the expression once per model. Stage 7's packing
   retry changes the roof, not the objective, so it must not re-enter `_rewrite`.
6. **Do not lambdify charges with the bundles.** If SA ever takes relayouts, compile
   charges as a plain list of `(is_lx getter, selector getters, dict)`. The numbers above
   give the reason. Nothing changes here, since SA gets no offers.
7. **A size guard.** `objective_size(expr) -> ObjectiveSize(terms, tree, dag)`, in
   `lx_allocator/cost.py`. This stage extends Stage 0's gate report with the three
   counters, read from the shadow plug-in's output, so the record format is unchanged. A
   test fails if the rewritten tree exceeds 2.5x the raw tree (measured maximum 1.71x),
   which is what the 39,948-term shape would trip.

Not in this stage: a memoizing `_rewrite`. The tree-to-dag ratio is 5x on
granite_micro.3, and rewrite is 91% of its lowering, so it looks like the largest
remaining saving. It risks the proto, needs its own gate, and is unmeasured.

### What a non-enumerating encoder adds

Nothing in `Sym`, `SpyreCostModel` or the expression changes. It binds `is_lx_*`, `split_*`
and `tile_*` to its own variables and supplies domains through `buffer_map` entries of
`(None, values)`, which the printer accepts today for `inv_*` (`:251-255`). The printer
then needs three things, none built here because nothing would call them:

- `log2_x` without a selector: a table over `(x, log2_x)` in place of the `add_element`
  at `:798`, which dereferences `b.division` on `None` today.
- `_print_AppliedUndef` over a split tuple rather than a menu.
- `SplitKeyedCharge` lowered directly as a table over its selectors, under the residency
  literal, instead of resolved.

Feasibility 2a measured the same shape end to end: 8,363 menu tables at most 576 rows as
tuple tables, same objective 64/64.

## Changes

| file | change |
|---|---|
| `torch_spyre/_inductor/scratchpad/lx_allocator/graph/symbols.py` | new: `Sym`, `SplitKeyedCharge`, `Unit`, `UnknownSymbolError` |
| `torch_spyre/_inductor/scratchpad/lx_allocator/cost.py` | new: `CostModel`, `COST_MODELS`, `SpyreCostModel`, `ObjectiveStage`, `Sense`, `NodeDomain`, `RelayoutOffer`, `CostDomains`, `domains_from_buffers`, `traffic_sizes`, `ALLOCATOR_COST_PARAMS`, `objective_size` |
| `torch_spyre/_inductor/scratchpad/lx_allocator/graph/target.py` | `Target.cost_model` annotated `CostModel \| None` through an `if TYPE_CHECKING:` import from `cost.py`; `SPYRE.cost_model` stays `None` |
| `torch_spyre/_inductor/scratchpad/lx_allocator/graph_builder.py` | `select_target` passes `cost_model=COST_MODELS.get(target.name)` to `replace` |
| `torch_spyre/_inductor/scratchpad/cpsat_expr_printer.py` | printer changes 1 to 6 |
| `torch_spyre/_inductor/scratchpad/ilp_solver_ortools.py` | `_minimize_cost_expr` calls `bind_division_owner`; `DivisionOwner` members on the wrapper |
| `torch_spyre/_inductor/scratchpad/plan_solver.py` | symbol properties delegate to `Sym` |
| `torch_spyre/_inductor/scratchpad/sa_cooptimizer.py` | `_build_score_fn` delegates; `_work_slices` becomes an alias |
| `torch_spyre/_inductor/scratchpad/utils.py` | `compile_fixed_point`, `division_work_slices`, `input_footprint` |
| `torch_spyre/_inductor/scratchpad/allocator.py` | `_COST_PARAMS` alias; `_input_footprint` delegates; bulk `Add` for charges |
| `tests/inductor/test_lx_cost_symbols.py` | new |
| `tests/inductor/test_lx_cost_model.py` | new |
| `tests/inductor/test_lx_cost_printer.py` | new |
| `tests/inductor/test_lx_cost_parity.py` | new |
| `tests/inductor/lx_objective_shadow.py` | new: the shadow check as a Stage 0 `Observer` |
| `tests/configs/torch_spyre_tests/inductor/test_lx_cost_*_config.yaml` | new shard configs, one per test file; `make check-all-configs` |
| `docs/source/compiler/cost_model.md` | a short section on `SpyreCostModel`, the target that holds it, and `Sym` |

Commits, each green with `intree` and the fixed set unchanged:

1. `Sym` and its tests; `plan_solver` symbol properties delegate.
2. `SplitKeyedCharge`, `NodeDomain`, `RelayoutOffer`, `CostDomains`,
   `domains_from_buffers`.
3. Printer: `Sym` prefixes, `DivisionOwner`, `bind_division_owner`.
4. Printer: `strict`, `UnknownSymbolError`, `linear_terms_to_cpsat`,
   `_resolve_keyed_charges`.
5. `compile_fixed_point`, `division_work_slices`, `input_footprint`,
   `ALLOCATOR_COST_PARAMS`; legacy callers delegate.
6. `SpyreCostModel.spill_cost`, `cut_traffic`, `ladder`, `traffic_only`, `traffic_sizes`,
   with replay parity on the fixtures.
7. `SpyreCostModel.bundle_terms`, `objective`, `priced_objective`, with stubbed-op tests;
   the `CostModel` protocol, `COST_MODELS`, and `select_target` attaching it.
8. Shadow plug-in, corpus parity test, `objective_size` guard.
9. Bulk `Add` in the allocator, on its own so it can be reverted alone.
10. Docs.

## Interfaces

Consumes:

| symbol | signature | from Stage | used for |
|---|---|---|---|
| Phase 1 exit | closed before this PR merges | 3 | dependency only; nothing else from Stage 3 is used |
| `build_graph` | `build_graph(ir, target, *, divisions_committed=False, drain_plans=None, lx_relayout_plans=()) -> SpyreOptimizationGraph`, called with keyword defaults | 2 | shadow check |
| `select_target` | `(config) -> Target`, built with `replace(TARGETS[config.target], ...)` | 2 | shadow check; this stage adds `cost_model=COST_MODELS.get(target.name)` to its `replace` |
| `Target`, `SPYRE` | frozen dataclass with `cost_model: CostModel \| None = None`; `SPYRE.cost_model` is `None` | 1 | `Target.name` keys `COST_MODELS` |
| `SpyreOptimizationGraph.target` | `Target`, stored by `build_graph` | 1, filled by 2 | `graph.target.cost_model` in the shadow and tests |
| `SpyreOptimizationGraph.nodes`, `.node` | `list[OpNode]`; `node(name) -> OpNode` | 1 | ladder, shadow |
| `SpyreOptimizationGraph.ir` | `GraphLowering`, final | 1, filled by 2 | `bundle_terms`, `traffic_sizes` |
| `OpNode.name` | `str`, the op's buffer name; the input's name on an `InputNode` | 1 | `Sym`, domain lookup |
| `OpNode.size` | `int`, equal to `mem_usage[name]["size"]`, -1 kept | 1, filled by 2 | `spill_cost`, `cut_traffic` |
| `OpNode.read_count`, `.first_use_is_read` | properties over `OpNode.uses` and `runs`, equal to `LifetimeBoundBuffer.read_count` and `.first_use_is_read` | 1, filled by 2 | `spill_cost`, `cut_traffic` |
| `OpNode.runs`, `OpNode.graph_output` | `bool` | 1 | the boundary, computed in `cost.py` |
| `Observer.on_solve_begin` | `(allocator, solver, ir, buffers, cost_expr)`, at every solve of either allocator; `allocator` and `ir` are `None` on a solver-only run | 0 | G1, `traffic_sizes` test |
| `Observer.on_solve_end` | `(solver, buffers, cost_expr, record)` | 0 | G3 |
| `record_compile` | `(name, cell, out, *, device="fake", observers=(), force_failure=None) -> Path` | 0 | the shadow on `fake` compiles |
| `load_solves`, `replay_intree`, `replay_capture` | `(path) -> dict[str, SolveRecord]`; `(case, *, arm)`; `(path, *, case)` | 0 | G2 |
| `SolveRecord` stages | one `StageRecord` per `_solve_and_record`; text at `protos/<sha256>.txt.gz` by `proto_sha` | 0 | G2 |
| corpus tiers | `intree`, `fake`, `unit-*`, `failure`, fixed set | 0 | gates |

Provides:

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `Sym` | as above, in `lx_allocator/graph/symbols.py` | 5, 6, 8, 10, 11 | final |
| `SplitKeyedCharge` | `(is_lx, source: Str, selectors: Tuple, rows: Tuple)` | 6, 10 | final |
| `UnknownSymbolError` | `class UnknownSymbolError(KeyError)` | 6, 8 | final |
| `Unit`, `Sense` | enums, in `graph/symbols.py` and `lx_allocator/cost.py` | 6, 8 | final |
| `ObjectiveStage` | `(name, sense, unit, terms)` | 6 | final |
| `NodeDomain` | as above | 5, 6, 8, 10 | final |
| `RelayoutOffer` | as above | 6 | final |
| `CostDomains` | `(nodes, relayouts=())` | 5, 6, 8, 10 | final |
| `CostModel` | protocol in `lx_allocator/cost.py`: `objective`, `priced_objective`, `bundle_terms`, `ladder`, `spill_cost`, `cut_traffic`; `graph/target.py` imports it only under `TYPE_CHECKING` | 5-13, through `graph.target.cost_model` | final |
| `SpyreCostModel` | `(params=ALLOCATOR_COST_PARAMS, *, traffic_only=False)`, in `lx_allocator/cost.py` | 9 (committed mode), tests | final |
| `COST_MODELS` | `Mapping[str, CostModel]`, `{"spyre": SpyreCostModel()}`, in `lx_allocator/cost.py`, keyed by `Target.name` | 2 (`select_target` attaches `COST_MODELS.get(target.name)`), 5-13 through `graph.target.cost_model` | final |
| `traffic_sizes` | `(graph) -> dict[str, int]` | 8, 9 | final |
| `CostModel.priced_objective` | `(graph, domains, *, linear_only) -> sympy.Expr \| None` | 6, 8, 11 | final |
| `CostModel.objective` | `(graph, domains) -> sympy.Expr`, may raise | 11, row terms | final |
| `CostModel.bundle_terms` | `(graph, domains) -> list[tuple[list[str], sympy.Expr]]` | 10, for the dump | final |
| `CostModel.ladder` | `(graph, domains) -> tuple[ObjectiveStage, ...]` | 6, 9 | final |
| `CostModel.spill_cost` | `(node, *, boundary_known=True, clamp_unsized=False, size=None) -> int` | 8 | final |
| `CostModel.cut_traffic` | `(node, domains, *, count_unmodelled=True) -> list[sympy.Expr]` | 8, as the reference for its companion loop | final |
| `ALLOCATOR_COST_PARAMS` | `CostParams`, in `lx_allocator/cost.py`; `SpyreCostModel`'s default | 9, through that default | final |
| `objective_size` | `(expr) -> ObjectiveSize` | this stage's size guard | final |
| `domains_from_buffers` | `(buffers) -> CostDomains` | 5, 6 for parity | temporary, removed in Stage 13 |
| `cost_expr_to_cpsat` | `(model, sym_map, buffer_map, cost_expr, *, strict=False)` | 5, 6 | final |
| `linear_terms_to_cpsat` | `(sym_map, terms) -> LinearExprT` | 6 | final |
| `bind_division_owner` | `(sym_map, buffer_map, owner) -> None` | 5 | final |
| `DivisionOwner` | protocol above, implemented by `OpVars` | 5 | final |
| `compile_fixed_point` | `(expr, getters, *, unit, strict)`, in `utils.py` | 8 | final |
| `division_work_slices` | `(op, division) -> dict`, in `utils.py` | 13 | final |

What each later stage does with these:

- Stage 5: `cost_domains`, called from `OrToolsFamily.encode`, builds `nodes`.
  `bind_symbols` calls `bind_division_owner`.
- Stage 6: extends `cost_domains` with one `RelayoutOffer` per group, with `prices_ns`
  keyed by `NodeDomain.selector_values`. It registers `cut_*`, `resident_cut_*`,
  `reads_full_*`, `cores_*`, `shape_*` and `tiles_*`. `encode_objective` calls
  `graph.target.cost_model.priced_objective(linear_only=True)` and lowers it with
  `strict=True` into one `SolveStage` named `cost`. When that returns `None` or lowering
  raises, it pins relayouts out and lowers each ladder stage with `linear_terms_to_cpsat`
  into a `SolveStage`.
- Stage 7: `OrToolsFamily.solve` runs `enc.ortools.objective_stages` and adds the locks.
- Stage 8: through `graph.target.cost_model`,
  `priced_objective(graph, CostDomains(nodes, relayouts=()), linear_only=False)`,
  `compile_fixed_point` through `compile_objective`, `spill_cost(node, clamp_unsized=True)`.
- Stage 9: committed mode builds with
  `replace(target, cost_model=SpyreCostModel(traffic_only=True))`.
- Stage 10: `bundle_terms` for the full `cost_expr_record`.

## Tests

`tests/inductor/test_lx_cost_symbols.py`:

| test | asserts |
|---|---|
| `test_sym_equals_buffer_symbols` | `Sym.resident`, `.split`, `.tile` equal `sym_is_lx`, `sym_core_divs[k]`, `sym_tile_counts[k]`, by name and assumptions, on int, `Symbol` and `"relayout_copy"` keys |
| `test_sym_accepts_a_node_or_a_name` | both give the same symbol |
| `test_factor_prefixes_exclude_ladder_symbols` | `is_factor` is false for `tiles_`, `cores_`, `shape_` |
| `test_is_known_rejects_foreign_symbols` | a dynamic-shape symbol is unknown |
| `test_keyed_charge_folds` | `is_lx=0` folds to 0; numeric selectors fold to the row price; a missing row is 0 |
| `test_keyed_charge_rows_are_canonical` | row order does not change equality or hash |
| `test_keyed_charge_lambdifies` | lambdify evaluates it through `_imp_` |

`tests/inductor/test_lx_cost_model.py`:

| test | asserts |
|---|---|
| `test_spill_cost_matches_both_engines` | parametrized over input, output, intermediate, placement-only, unsized: equal to `_LifetimeBufferWithCpVars.spill_cost` and, with `clamp_unsized`, `SaCoOptimizingSolver._spill_cost` |
| `test_cut_traffic_matches_cp_terms` | for `(cut, resident, via_full)` assignments on a tiny CP model, the expression equals the flattened `_cut_traffic_terms`, with `U` of 0 and 1 |
| `test_cut_traffic_matches_companions` | on synthetic SA states, the expression with `count_unmodelled=False` under SA's binding equals `_companion_bytes` |
| `test_extern_only_reader_is_cut_in_cp_only` | the binding difference stated under Today |
| `test_reads_full_keeps_repeated_listings` | a reader listed twice gives two symbols and counts twice in `U`, as the two entries of `kids` do |
| `test_ladder_stages_and_skips` | untiled joint and tiled joint: stage names and order match `_run`'s branches and `_solve_stage` strings |
| `test_traffic_only_has_no_expression` | `priced_objective` returns `None` whatever the config, and `ladder` is `residency` alone |
| `test_traffic_only_matches_plan_layout_objective` | on placement-only fixtures, the stage's coefficients equal `plan_layout`'s objective proto |
| `test_traffic_sizes_match_placement_buffers` | on a live `ScratchpadAllocator` solve, `traffic_sizes` equals `{b.name: b.size}` |
| `test_ladder_folds_single_candidates` | a one-candidate node gives an integer core and shape term |
| `test_ladder_covers_relayout_copies` | each copy has a `residency` term, as `hbm_terms` does |
| `test_priced_withheld_only_for_linear_families` | tiling on: `None` for `linear_only=True`, an expression for False |
| `test_failed_build_falls_back_or_raises` | `_cpsat_warn_on_cost_expr` both ways |
| `test_foreign_symbol_gives_none` | a dynamic-shape symbol gives `None` and a warning |
| `test_bundle_terms_match_the_allocator_on_stub_ops` | with `estimate_bundles` stubbed as `test_cost_expr_dump.py:200` does and `extract_op_features` replaced by a spy on the importing module as `test_cost_model_loop_operands.py:136` does, `is_lx`, work slices and menu arrive as `_extract_op_features` passes them |
| `test_expression_has_no_menu_symbol` | no free symbol starts `division_` |
| `test_bulk_add_equals_left_fold` | float coefficients on shared monomials across bundles: `Add(*terms)` equals `sum(terms)` under `srepr` |
| `test_cost_imports_no_solver` | importing `lx_allocator.cost`, or `graph.target` with it, loads neither `ortools` nor `sa_cooptimizer` |
| `test_select_target_attaches_the_cost_model` | `SPYRE.cost_model` is `None`; `select_target(config).cost_model` is `COST_MODELS["spyre"]`, a `SpyreCostModel` with `ALLOCATOR_COST_PARAMS` and `traffic_only` false, and a graph built for that target reaches the same object as `graph.target.cost_model` |
| `test_derived_target_swaps_the_cost_model` | a graph built for `replace(SPYRE, cost_model=SpyreCostModel(traffic_only=True))` gets `residency` alone from `graph.target.cost_model.ladder` |

`tests/inductor/test_lx_cost_printer.py`:

| test | asserts |
|---|---|
| `test_bind_owner_keeps_the_maps` | on capture fixtures, the two dicts equal what today's loop built, key for key |
| `test_resolved_charge_equals_cost_term` | on capture fixtures and a synthetic trailing-zero price, `_resolve_keyed_charges` gives `copy.cost_term()` exactly |
| `test_keyed_and_indexed_charges_print_the_same_proto` | byte-equal protos |
| `test_strict_unknown_symbol_raises` | `UnknownSymbolError`, and it passes through `_minimize_cost_expr`'s `except` |
| `test_strict_unowned_function_raises` | an `AppliedUndef` over two buffers raises; non-strict still prices 0 |
| `test_non_strict_paths_unchanged` | the three zero returns stay |
| `test_linear_terms_match_python_sum` | objective and lock protos byte-equal to the direct `sum` |
| `test_linear_terms_reject_nonlinear` | a product raises `ValueError` |
| `test_scaling_constants_are_pinned` | every constant in the Scaling table |
| `test_rewrite_expansion_is_bounded` | rewritten tree at most 2.5x raw on the fixtures |

`tests/inductor/test_lx_cost_parity.py`, device-free:

| test | asserts |
|---|---|
| `test_ladder_equals_recorded_stage_objectives` | G2 on `cooptimization_captures*.json` and the synthetic graphs, using fixture nodes built from buffers |
| `test_score_equals_legacy_score_fn` | G3 on the same, in integers |
| `test_shadow_on_a_real_allocator_solve` | G1 on the `staged_transport_graph` fixture (`test_restickify_cost_model.py:385`), which `:556` already drives through the real allocator with no device |

Existing suites, run serially: `test_scratchpad_solver`, `test_solver_relayout_decision`,
`test_cooptimization_types`, `test_cost_expr_dump`, `test_restickify_cost_model`,
`test_sa_cooptimizer`, `test_coarse_tiling`, `test_work_division*`,
`test_scratchpad_patterns`, `test_perm_layout_solver`, `test_scratchpad_use`,
`test_solver_auto_coarse_tiling`, plus `test_solver_cost_model`,
`test_cost_model_decode`, `test_cost_model_loop_operands`. The parent plan requires no
device suite after this stage, and none is planned: no production path changes.

G1 and G2 on the Stage 0 corpus prove the parent plan's claim.

## Parity and gates

The parent plan's gate: "Proven by: on the Stage 0 corpus, the objective equals today's,
solve for solve." Verification adds the Stage 0 diff against the parent commit.

No new encoder exists, so the check runs beside today's solve, not instead of it.
`lx_objective_shadow.py` is a Stage 0 `Observer`. Tiers at the PR tip: `intree`, the `fake`
zoo at both configs, `unit-*` and `failure`. Per commit: `intree` and the fixed set. A
model `fake` cannot reach stays card-only. `fake` uses default layouts on fake inputs, so
it proves parity, not production plans. No gate here needs a card otherwise.

**G1, priced expression.** In `on_solve_begin(allocator, solver, ir, buffers, cost_expr)`,
on the joint allocator:

```python
graph = build_graph(ir, select_target(config))
domains = domains_from_buffers(buffers)
new = graph.target.cost_model.priced_objective(
    graph, domains, linear_only=solver.linear_cost_only()
)
```

`None` must match `None`, or a legacy expression holding a free symbol `Sym.is_known`
rejects, which is branch 3 of `priced_objective` reached earlier. Otherwise it resolves
`new`'s keyed charges with `index_charge`, taking keys from `NodeDomain.selector_values`
over each source's menu, and requires `==` and equal `srepr` against `cost_expr`. This
needs no CP model. It covers every solve that
reaches `_solve`, in every `layout_solver` × `co_optimizing_lx_planning` combination that
builds an expression, including the forced failure.

The check is structural because equal expressions make everything downstream equal by
construction: lowering, proto, lambdify source, and the value Stage 0 records as
`objective_ns`, which G1 therefore does not re-evaluate. It needs a live `GraphLowering`,
since pickled captures hold none, so the `fake` tier carries it. `test_lx_cost_parity.py`
carries the share CI runs.

**G2, ladder.** For each ladder stage in a Stage 0 record, take the stored proto's
objective, translate var names by the table below, and compare the `{name: coefficient}`
map and the offset with `ObjectiveStage.terms`, after substituting known constants. A
placement-only solve compares with `SpyreCostModel(traffic_only=True)`. This is exact linear
equality over all assignments, not agreement at the solved point, and it replays from
recorded protos.

| proto var | `Sym` |
|---|---|
| `in_buffer_<n>` | `is_lx_<n>` |
| `cut_<n>` | `cut_<n>` |
| `resident_cut_<n>` | `resident_cut_<n>` |
| `reads_full_<n>_<c>`, k-th with that name | `reads_full(n, c, k)` |
| `occ_<n>` | `cores_<n>` |
| `core_cost_<n>` | `shape_<n>` |
| `tile_count_<n>` | `tiles_<n>` |

A maximize is compared after the proto's sign and `scaling_factor`.

**G3, SA score.** On every SA solve, `compile_fixed_point(new, getters, unit=Unit.NS,
strict=True)` equals `solver._score_fn` on the seed state and on 32 states drawn from
the seeded RNG, in integers, and the two-part score equals `_score()` under tiling.

**G4, no change.** The Stage 0 diff between the parent commit and this PR is empty on the
tiers above: proto, status, objective, decisions and committed plan, with `strict=False`
on the legacy path.

**Strict sweep.** On every priced CP-SAT solve, lowering `new` with `strict=True` on
`model.clone()` inside `_minimize_cost_expr` raises nothing.

Pass: all of the above, with no exceptions. No rule changes in this stage, so a difference
is a bug in this PR. It is fixed, not listed.

## Risks

- **Names drift between `Sym` and the buffers.** Proto order would move with no plan
  change at first. Caught by `test_sym_equals_buffer_symbols` and G4.
- **`Add` differs from the left fold on float coefficients.** I reasoned they match,
  since `Add.flatten` folds in sequence, and they matched on all 47 captures for charges.
  Bundle sums were not checked, because captures hold only the total. Caught by G1 and
  `test_bulk_add_equals_left_fold`. If they differ, `objective` uses
  `functools.reduce(operator.add, terms)` for the bundles and keeps bulk for charges.
- **`V.graph` is not active.** `_graph_boundary_names` then returns `None` and args stay
  unstamped (`cost_features.py:1139-1144`), which silently changes LX prefetch pricing.
  Caught by G1. Stage 9 must call the family's `encode` inside the same context.
- **`cost_domains` differs from `domains_from_buffers`.** Then a symbol set changes.
  This stage cannot catch it. Stage 6's gate, which includes the first-stage objective,
  does.
- **A ladder stage through sympy changes the proto.** The CP-SAT facts under Today say it
  does not. Stage 6's gate would catch it, and only then would a CP copy return.
- **`traffic_sizes` differs from the placement path.** The `ncores` handling is ported by
  reading only. I have not run it on a placement solve. Caught by
  `test_traffic_sizes_match_placement_buffers` and G2 on `co_optimizing=0`.
- **Strict mode fires on `AppliedUndef` on current graphs.** No capture has one, so the
  three zero returns are unexercised by my measurements. The strict sweep finds this. If
  it fires on a legitimate shape, such as a concrete functor over no owner, that shape
  gets an explicit case, listed in the PR.
- **Foreign symbols.** Legacy CP-SAT prints part of the expression before failing and
  leaves its auxiliary variables in the model. `priced_objective` returns `None` before
  printing, so the new path's proto would lack them. This cannot appear before Stage 6,
  and it should be rare. It belongs in Stage 6's list of justified proto differences.
- **The shadow disturbs the solve.** `project_transport_read_copies` and feature
  extraction are non-mutating by their tests (`test_restickify_cost_model.py:593-595`),
  and sympy's cache affects time only. G4 would show otherwise, since it compares a run
  with the shadow off against one with it on.
- **The captures are stale.** Sizes are from October and lack the decode term. The size
  guard re-reads real sizes on the Stage 0 corpus, and the 2.5x threshold can move with
  evidence.
- **#5695 lands first.** `bundle_terms` avoids `mem_usage_by_buf` and passes explicit
  work slices, and that PR reports identical dumps. `traffic_sizes` does call it, which
  needs a committed split, so it stays on committed mode only. Baselines are re-recorded
  on rebase.
- **`test_cut_traffic_matches_companions` and G3 need `torch_spyre._C`.** It is not in
  this snapshot, so I have not run them. They need a built tree.
- **Copies in the ladder.** Forgetting them would pass a decision-level check, since
  they are pinned out, and fail only on the proto. G2 compares coefficients for that
  reason.

## Deviations from the parent plan

1. **The objective is not one expression.** The plan says
   `SpyreCostModel.objective(graph)`, held by the `SPYRE` target, ports `spill_cost`,
   `_cut_traffic_terms`, the per-bundle terms, relayout charges, and cores "only to break
   ties", as one sympy expression. The code has separate objectives
   that never combine (`ilp_solver_ortools.py:1442-1523`), in different units, and the
   ladder's locks cannot be written as one expression without weights that change plans.
   Cores do reach the priced expression, through split symbols and `_product_*` bound to
   `t.cores` (`:855-858`). What it lacks is a tie-break stage, which exists only on the
   ladder. Resolution: `objective` is today's `cost_expr`, and `ladder` returns the stages
   holding `spill_cost`, `cut_traffic` and cores. The ladder stays through Stage 12: it is
   the only CP-SAT objective with `AUTO_COARSE_TILING=1`, because the cost model cannot
   score tile size or cut count (`allocator.py:3280-3290`). `traffic_only` runs its
   `residency` stage for committed mode.

2. **`objective(graph)` also needs domains, and `cut_traffic(node)` too.** The split
   symbols, the delivery estimate, the transport projection and the relayout table
   depend on the menu (`allocator.py:3210`, `:3448-3450`, `plan_solver.py:440`,
   `cost_features.py:1287`), and which symbols exist depends on which buffers are in the
   solve (`:3224`). The menu belongs to the encoder (parent, Stage 5), and relayout
   candidates do not exist on the graph before a decision. Resolution:
   `objective(graph, domains)` and `cut_traffic(node, domains)`, where `domains` carries
   candidates and offers with no indices.

3. **`spill_cost(node)` reads what the diagram does not show.** `OpNode` lists `size`, but
   `read_count` counts extern readers, so it cannot come from `out.edges`
   (`plan_solver.py:147-160`, `ilp_solver_ortools.py:1311`), and pricing reads IR ops.
   Resolution: `read_count` and `graph.ir` from Stages 1 and 2. The boundary is computed
   in `cost.py` from `runs` and `graph_output`.

4. **`score` is not one compiled expression.** Stage 8 says `score` compiles the sympy
   objective into `_build_score_fn`'s evaluator. Today's score also adds companions,
   rounded apart (`sa_cooptimizer.py:1447`), and companions differ from
   `_cut_traffic_terms` by `U`. Resolution: `SAFamily.score` adds the companions to the
   compiled expression as a second integer, and `cut_traffic(count_unmodelled=False)` is
   held equal to them, so Stage 8 can match today's SA exactly.

5. **"An unknown name fails" holds for OR-Tools, and for SA only when
   `_cpsat_warn_on_cost_expr` is false.** Today it silently degrades
   (`ilp_solver_ortools.py:886-900`, `sa_cooptimizer.py:847-848`), and the legacy path
   stays as it is for G4. A dynamic-shape symbol is not an encoder bug either.
   Resolution: `strict=True` fails with `UnknownSymbolError` for a `Sym` name no encoder
   bound, `compile_objective` passes `strict=not config._cpsat_warn_on_cost_expr`, and
   `priced_objective` keeps today's fallback for a foreign symbol.

6. **The gate is met by a shadow check.** It cannot compare two encoders, because the
   second does not exist. Resolution: G1 to G4.
7. **`SPYRE` does not hold the cost model.** The parent builds `SPYRE` with
   `cost_model=SpyreCostModel()` in `graph/target.py`, but the `graph/` boundary test forbids
   that module importing `cost.py`. Resolution: `SPYRE.cost_model` stays `None`, `cost.py`
   holds the `CostModel` protocol and `COST_MODELS`, and `select_target` attaches
   `COST_MODELS.get(target.name)`, so every compiled graph still reads its cost model from
   `graph.target.cost_model`.

## Open questions

None.

## Out of scope

- `symbol` and `cost_domains`: Stage 5.
- `encode_objective`, lowering, registering cut, ladder and relayout symbols, producing
  `RelayoutOffer`s, and the relayout pin: Stage 6.
- Running the stages and adding the locks: Stage 7.
- `SAFamily.score` and SA's symbol binding: Stage 8.
- Choosing committed mode: Stage 9.
- The cost dump on the new path (`cost_expr_record`, `solved_bindings`): Stage 10.
- A restickify's compute time and any layout-dependent term: Stage 11.
- Deleting the old orchestration, the duplicate spill and cut formulas,
  `RelayoutCharge` emission from `cost_term`, and `domains_from_buffers`: Stage 13.
- The relayout price law (`solver_relayout_pair_cost`, `relayout_ns`), enumeration
  capping and `_relayout_consumer_costs`: Stage 6 keeps them. The cost model takes prices as
  data.
- `cost_model_pass.py`, `dump_cost_model.py` and the `cost_model.py` terms: unchanged.
- A memoizing `_rewrite` and the non-enumerating lowerings: future work.
