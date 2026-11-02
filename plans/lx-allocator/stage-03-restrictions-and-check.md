# LX allocator, Stage 3: Restrictions and the check

Parent plan: `plans/lx-allocator.md` at `5aff4a6`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0, 1, 2. Needed by: Stage 4 (the Phase 1 exit only) and Stages 5 to 13.

## Goal

When this PR merges, every legality rule the allocator obeys is reachable as plain data
through `splits_while_stuck`, `fact_splits`, `always_splits`, `stick_tables` and
`residency_reason`. `check_decisions` judges a decoded plan without a solver, and an
adapter puts every plan today's allocators commit onto the graph so the check runs inside
each compile. It only reports. It cannot change a plan, a model or the outcome of a
compile. The solvers, the enumeration, `_push_allocation` and every committed plan stay
as they are, and no solver reads the new data yet.

## Today

Path aliases used below, to keep tables readable. Each stands for the full path.

| alias | path |
|---|---|
| `WDC` | `torch_spyre/_inductor/work_division_constraints.py` |
| `WD` | `torch_spyre/_inductor/work_division.py` |
| `ALLOC` | `torch_spyre/_inductor/scratchpad/allocator.py` |
| `PS` | `torch_spyre/_inductor/scratchpad/plan_solver.py` |
| `ILP` | `torch_spyre/_inductor/scratchpad/ilp_solver_ortools.py` |
| `SA` | `torch_spyre/_inductor/scratchpad/sa_cooptimizer.py` |
| `RR` | `torch_spyre/_inductor/scratchpad/residency_rules.py` |
| `CT` | `torch_spyre/_inductor/scratchpad/coarse_tiling.py` |
| `PL` | `torch_spyre/_inductor/propagate_layouts.py` |
| `OSV` | `torch_spyre/_inductor/op_spec_validation.py` |
| `PU` | `torch_spyre/_inductor/pass_utils.py` |

`collect_work_division_constraints` (`WDC:125`) runs 16 constraint functions and merges
them: `blocked` by union, `allowed_splits` by intersection. `WorkDivisionContext.is_legal`
(`WD:901`) applies the result plus five other clauses. `enumerate_work_division_candidates`
(`WD:1051`) is the cross product of `factor_domain` filtered by `is_legal`. Residency is
an ordered first-failure registry of 15 rules and 4 exception predicates
(`RR:84`, checked by running it, `exp/stage-03/reg.py`), evaluated by
`_BufferResidencyChecks` (`ALLOC:789`), which holds a back-reference to the allocator.

Things in the code that the rest of this plan depends on, several of them surprising:

1. **Rule data is relative to a tiling.** `work_division_context_for_op(op, max_cores,
   tiling)` (`WD:975`) rebuilds the iteration space per tiling and reruns every
   constraint on that frame. The feasibility report found 50 of 2,674 tiled frames
   block an axis the untiled frame leaves free.
2. **`reduction_vars` is not "the K axes".** It is every axis absent from the output's
   `device_coords[:-1]` (`WD:1018-1024`), which includes an output stick axis that fits
   in one stick. `CoreDivision.reduction_syms` is a different test,
   `write.index.coeff(s) == 0` (`WD:1109-1117`). The two must not be merged into one
   `Axis.reduction`.
3. **A stick axis is judged in sticks, not elements.** `factor_domain` uses the stick
   count as the divisor basis (`WD:889-898`, `WD:491-493`), rounded up. 130 elements at
   64 per stick is 3 sticks, so a split of 3 is legal although 3 does not divide 130.
4. **Mixed dtypes on one stick axis use the largest stick.** `adjust_it_space_for_sticks`
   keeps the maximum `elems_per_stick` over all tensors stuck on the axis (`WD:438-484`).
   With fp16 and fp32 on 96 elements the basis is 2, and split 2 leaves the fp32 tensor
   with 1.5 sticks per core. That plan is legal today.
5. **Domains are not always `{1}`.** `overlapping_window_split_domains` pins an outer axis
   to `{extent}`, one value per core (`WDC:1231-1232`). `topk_split_domains` and
   `keep_by_index_k_split_constraint` pin to the smallest legal factor (`WDC:832`,
   `WDC:889`).
6. **`is_legal` is not the only gate on a division.** Pinned ops go through
   `_legal_fixed_division` (`ALLOC:2376`), which asks only `obeys_op_constraints` and
   `meets_span_floors` (`WD:1097`). The pruned exhaustive menu goes through
   `_legal_split_options` (`ALLOC:2422`). The placement path never asks: it reads committed
   divisions. `SaCoOptimizingSolver._write_back` can append a generated division that was
   never on the menu (`SA:1966-1993`).
7. **Pinning has a wider list of reasons than the parent names.** `_division_map` pins cpu/host
   buffers, offset-mutation closures, windowed pools, keep_by_index and fp8 layout groups,
   indirect access, offset slice reads, deferred direct reads and hints (`ALLOC:3739-3770`).
8. **`is_output_stick_aligned_for_entry` does not exist,** and `indirect_forbidden_split_syms`
   has no padding exception: a partial last index stick stays unsplit, judged from the index's
   logical entry count (`PU:1812-1845`, asserted by
   `tests/inductor/test_indirect_access_gather.py:1205-1215`). `WDC:1038-1039`, `PU:1673` and
   `torch_spyre/_inductor/enforce_indirect_access_layout.py:75-79` are stale. Padding
   (`padded_entry_output_stl`, `PU:1702`) enables no split, but it does change the buffer size.
9. **Sticks are already re-judged on every compile.** `_check_stick_constraints` (`OSV:643`)
   runs when `validate_op_specs` is on, its default
   (`torch_spyre/_inductor/spyre_kernel.py:1484`, `torch_spyre/_inductor/config.py:440`).
   `_check_stick_all_same` covers pointwise, dtype and most reduction ops (`OSV:668-672`).
   `commit_lx_views` re-judges residency views and raises `Unsupported` (`ALLOC:3020-3029`).
10. **QFP8WT reaches only two places.** `qfp8wt_split_domains` looks at `input_tds[1]` and
    the output (`WDC:739-751`), not every operand. The batch-dim form applies only when
    the tensor has more than two host dims (`WDC:719-721`).
11. **`ENCODED_K_UNSPLIT`'s source pins all of `reduction_vars`**, not only K
    (`WDC:781-783`).
12. **`SPAN_CAP` differs between its two sources.** `_spans_within_cap` measures every
    tensor (`WD:954-959`); `raise_if_per_core_overflow` skips indirectly accessed ones
    (`WD:560-562`). Indirect ops are pinned on the joint path, so `is_legal` never asks.
13. **Restickify ops have their own division and buffer.** They appear in `mem_usage` like
    any op, get a menu, and `restickify_padding_blocked_vars` (`WDC:661`) constrains that
    division, so all of that lives on the `RestickifyNode`, not on the pinned step.
14. **Reductions can have several inputs.** `conv2d`, `keepbyindex` and depthwise conv have
    two (`torch_spyre/_inductor/constants.py:261`, `:268`), and a fused prologue can add more
    (`torch_spyre/_inductor/wsr/loop_tiling_common.py:450-451`).
15. **A placement is added after the solve.** `_stage_collective_outputs` (`ALLOC:3525`),
    on by default (`torch_spyre/_inductor/config.py:494`), gives a spilled matmul output an
    LX address inside `_post_solve`, in a gap unused for the entire graph.
16. **The graph is unchanged until `_post_solve`.** The fallback relies on it
    (`ALLOC:5496-5500`), and `apply_chosen_tilings` is the first mutation (`ALLOC:3513`).

What a plan exposes once `self._solve` returns:

| decision | CP-SAT joint | SA joint | exhaustive joint | placement solvers |
|---|---|---|---|---|
| address, bytes | `address` | `address` | `address` | `address` |
| division | `chosen_division` into `core_divisions` (`ILP:783-784`) | same (`SA:1979`) | same | none: read `op.iteration_space_ownership` (`ALLOC:2370`) |
| tiling | `CoreDivision.tiling` | same, may be appended | same | none |
| relayouts | `chosen_relayouts`, then `LXRelayoutPlan` from `_finalize_lx_relayout_allocation` (`ALLOC:3453`) | none (`SA:716`) | none | `lx_relayout_plans`, then `_accepted_plans` (`ALLOC:2008`) |
| residency bars | `residency_reason`, `untiled_bar` | same | same | `residency_reason` |
| spill cause | `spill_reason` | same | same | same |
| per-core view | set later by `commit_lx_views` | same | same | `lx_view` at build |

Not exposed by any solver: which in-place merges fired (`merge_vars`, read only inside
`_extract`, `ILP:1849-1852`), cuts and `via_full`, nest ids, and the solved lifetime ends
(`slot_span`, `ILP:1734`). `in_place_parents` lists candidates, not decisions. Section 6 of
the design recovers each of these from what is exposed, with no change to any solver.

## Design

### 1. Files and layers

The three modules sit in `torch_spyre/_inductor/scratchpad/lx_allocator/`, outside `graph/`,
so they may reach allocator helpers. `rules.py` holds rule data and imports only sympy, the
enum module and `graph/` at module level. IR helpers are imported inside the functions that
need them, as `ALLOC` does throughout. `check.py` holds the check. `plan_adapter.py` holds
everything that touches today's buffers and is removed in Stage 13.

**Moved, not copied.** A rule's computation moves to `rules.py` as a pure function over a
`Frame`, and the old function in `WDC` keeps its name and signature as a thin adapter from
`WorkDivConstraintContext`. `work_distribution` still needs
`collect_work_division_constraints` after Stage 13 (`LX_PLANNING=0`, pinned ops, #5695), so a
copy would leave every rule twice forever. Old names stay patch targets:
`tests/inductor/test_work_division.py` patches `WDC` names directly, and
`torch_spyre/_inductor/scratchpad/utils.py:821` imports `overlapping_coordinate_extents`. A
rule that needs live IR stays where it is and `rules.py` calls it through a local import. The
tables below mark each one.

`_BufferResidencyChecks` and its helpers stay in `ALLOC`, through Stage 13 as well. 29 test
patches target `torch_spyre._inductor.scratchpad.allocator.*`. `rules.py` reaches them through
a small host, a `ScratchpadAllocator` built with `GreedyLayoutSolver` that never solves, with
`_decides_lx_relayouts` set from the call. Stage 13's smaller host replaces it.

### 2. Vocabulary

```python
class Always(str, Enum):
    # The parent plan's list.
    SPLIT_DIVIDES = "split divides the axis"
    CORE_BUDGET = "within the core budget"
    ONE_REDUCTION_SPLIT = "at most one reduction axis split"
    TILES = "tiles on output axes, never the stick dim"
    CUT_TILE = "a copied-out tile that stays resident has no reduction split"
    RESIDENCY = "the residency rules hold"
    RESTICKIFY = "a restickify only where the read stick differs"
    PINNED = "pinned steps stay"
    COPY_OUT = "a copy-out lives in HBM"
    IN_PLACE = "in-place conditions hold"
    PORT_FACTS = "the rules each port fact names hold"
    TILE_ALIGNED = "tiles line up inside a nest"
    NO_OVERLAP = "resident boxes fit LX and never overlap"
    # Added here: existing rules with no entry above (see Deviations).
    SPAN_CAP = "per-core span within the read-distance limit"
    SPAN_FLOOR = "split meets the committed span floor"
    RESIDENT_READ = "a resident source is read under its own view, or through a copy"
    RELAYOUT_SOURCE = "a resident relayout has a resident, priced source and a reader"
    TILE_LOCAL = "a coarse-tile-local dim stays whole"
    ALIGNED_OWNERSHIP = "a split keeps each core's share one contiguous block"
    RESTICKIFY_PADDING = "an unaligned restickify stick dim stays on one core"
    SPLIT_PINS = "a pinned division stays"
    CROSS_FRAME = "a restickify reads a resident source under its own view"
    TABLE_COLUMN = "a table column follows its selector"   # provenance tag, no check

RuleTag = Restriction | PortFact | Always
CONSTRAINT_RULES: dict[str, RuleTag]   # keyed by constraint function name, the 16 of table A

@dataclass(frozen=True)
class SplitRule:
    rule: RuleTag                 # provenance tag, what Encoder.post records
    axis: Axis
    factors: frozenset[int]       # legal factors; empty means no legal split
    port: Port | None = None      # None: holds whatever is stuck
    dim: int | None = None        # the rule holds while `port` is stuck on this dim

@dataclass(frozen=True)
class StickTable:
    rule: Restriction
    ports: tuple[Port, ...]              # columns, in ports and/or the out port
    rows: frozenset[tuple[int, ...]]     # legal stick dims, one value per column

@dataclass(frozen=True)
class Frame:
    extents: Mapping[Symbol, Expr]       # element-valued: today's it_space
    sticks: Mapping[Symbol, Expr]        # stick-adjusted: today's it_space_adjusted
    stick_elems: Mapping[Symbol, int]    # today's stick_vars, max over tensors
    reduction_axes: tuple[Symbol, ...]   # today's reduction_vars, in its order
    inputs: tuple[TensorDep, ...]
    output: TensorDep
    meta: SymbolMeta                     # deprecated symbolic route, usually empty

    def basis(self, axis: Symbol) -> int: ...   # factor_domain's divisor basis

def frame_of(node: OpNode, tiling: TileSpec | None = None) -> Frame | None: ...
def splits_while_stuck(node: OpNode, flag: Restriction,
                       tiling: TileSpec | None = None) -> list[SplitRule]: ...
def fact_splits(node: OpNode, tiling: TileSpec | None = None) -> list[SplitRule]: ...
def always_splits(node: OpNode, tiling: TileSpec | None = None) -> list[SplitRule]: ...
def stick_tables(node: OpNode, flag: Restriction) -> list[StickTable]: ...
def stuck_dims(port: Port) -> frozenset[int]: ...
def is_stuck(rule: SplitRule) -> bool: ...
def residency_reason(node: OpNode, graph: SpyreOptimizationGraph, *,
                     decides_relayouts: bool, tileable: bool) -> str | None: ...
def untiled_bar(node: OpNode, graph: SpyreOptimizationGraph) -> str | None: ...
def residency_bar(node: OpNode, graph: SpyreOptimizationGraph, split: WorkSplit | None,
                  *, decides_relayouts: bool = True) -> str | None: ...
```

`frame_of` returns `None` for a tiling `predict_frame` rejects
(`torch_spyre/_inductor/wsr/tile_prediction.py:291`) and for a node with no iteration space
(an `OpaqueNode`). It rebuilds what `work_division_context_for_op` builds, without running
the constraints, from `iteration_space_from_op`, `adjust_it_space_for_sticks` and the
`reduction_vars` formula. `Frame.from_context(ctx)` is the reverse direction for the old
adapters. A synthetic node with `op=None` uses `Axis` extents, divided by the tile count. A
`WorkSplit` holds `TileLevel`s, so the check calls `frame_of(node, tile_spec_of(node, split))`.

`Always` is a `str` enum so a member works as the `rule` string in `OrToolsEncoder.post` and
in provenance. `Flag` iteration over a composite needs Python 3.11 (`pyproject.toml:42`).

### 3. The two data shapes

Both encoder families and the check read exactly these. A row names a port and an axis,
never a menu position, so a later non-enumerating encoder can post it as a constraint.

**`splits_while_stuck(node, flag, tiling)`** returns `SplitRule`s. A row with `port=None`
holds unconditionally. A row with a port holds only while that port is stuck on `dim`, one
of `port.dims_of(axis)`. While sticks are pinned, `is_stuck(rule)` answers that with a
constant. From Stage 11 an encoder replaces it with a literal.

```python
for rule in splits_while_stuck(node, flag, tiling):
    if rule.port is None or is_stuck(rule):
        prune(entries in this tiling frame whose split on rule.axis not in rule.factors)
```

| flag | rows | tricky case |
|---|---|---|
| `WHOLE_STICKS` | One per port, axis and `dim` in `port.dims_of(axis)`: `(port, dim, axis, divisors(ceil(extent / eps)))` | `eps` is the largest `elems_per_stick` over this port and every port currently stuck on the axis (surprise 4). A port whose `device_coords[-1]` has other than one free symbol is not stuck on anything, as in `WD:464`. QFP8WT sticks on both of `device_coords[-2:]` (`WD:443-461`). |
| `PADDED_REDUCTION_UNSPLIT` | One per reduction axis and port reading it, where `extent % port_eps != 0`: `(port, dim, axis, {1})` | Per-port is equivalent to the old max-over-tensors test because stick sizes are powers of two. Extents come from `frame.extents`, never `frame.sticks`, as the docstring at `WDC:352-358` requires. |
| `ENCODED_K_UNSPLIT` | `(None, None, a, {1})` for each `a` in `frame.reduction_axes`, only when some port has `QFP8WT` or `STAGGERED_EA` | All reduction axes, not only `node.k` (surprise 11). |
| `INDIRECT_ACCESS` | `(None, None, a, {1})` for each shared data axis of port `table`; `(index, dim, entry, {1})` for a partial last index stick | Coordinates of `table` carry `IndirectAccess`; the mixed-coordinate case `d0 + 128*IndirectAccess(idx)` still reports `d0` (`PU:1779-1794`). Partial uses the index's logical range. No padding exception (surprise 8). |
| new flags | `(None, None, a, factors)` | See table A. |

`MATMUL_STICKS` and `POINTWISE_STICKS` return no split rows. Any flag on a node type it
does not fit returns `[]` rather than raising, so an override can add one freely.

`fact_splits` and `always_splits` return the same rows, tagged with the `PortFact` or
`Always` member. Rows are per frame, so an encoder calls these once per tiling in its table.

**`stick_tables(node, flag)`** returns `StickTable`s. A row is one legal assignment of
stick dims to the table's ports. `stuck_dims(port)` wraps Stage 1's `stick_dims`, the only
layout reader: `frozenset(stick_dims(port.current_layout()) or (SPARSE_STICK,))`, two dims
for QFP8WT, and empty for a port with no layout. On a self-matmul `x` and `y` share one edge.

| flag | tables |
|---|---|
| `MATMUL_STICKS` | Three one-column tables: `x` on `k`, `y` on `n`, out on `n`. If `n` is `None` (N = 1): `y` on `SPARSE_STICK`, out on its last host dim (`PL:1539-1555`). If `k` is `None` (K = 1) there is no `x` table, as `OSV:869-872` skips it. A QFP8WT `y` passes because `stuck_dims` holds two dims. |
| `POINTWISE_STICKS` | One table over `(out, in0 .. inN)`. For each output dim `o` with axis `a`, the row set is the product over in ports of `port.dims_of(a)`, plus `broadcast_dims(port)`, plus `SPARSE_STICK`. For `IndirectNode` the index ports are left out. A `RestickifyNode` has no such flag: `_check_stick_restickify` wants the opposite. |

A fused dim, where two axes map to one host dim, makes the table slightly looser than
`_check_stick_all_same`, which compares `min(free_symbols)` of each stick coordinate, so the
check also compares that symbol across the table's ports. The encoder posts only the table.

### 4. Inventory

Fate column: **moves** means the computation moves to `rules.py` and the old function
stays as an adapter over it. **stays** means the old code remains and is called.

**A. `work_division_constraints.py`.** Every function, 29 by `ast` count, plus 2 classes.

| old | new home | node types | data | fate |
|---|---|---|---|---|
| `collect_work_division_constraints` `WDC:125` | none, it is the merge | | | stays; conflict checks against `committed_splits` serve `work_distribution` |
| `WorkDivConstraintContext`, `ConstraintResult` | `Frame` is the context minus `op` | | | stays |
| `coordinate_mask_blocked_vars` `WDC:351` | `PADDED_REDUCTION_UNSPLIT` | Matmul, Reduction | frame | moves |
| `conv_spatial_blocked_vars` `WDC:368` | new flag `CONV_SPATIAL_UNSPLIT` | Reduction | `node.conv_params`, config read by builder, last two write ranges | moves |
| `reduction_window_blocked_vars` `WDC:400` | new flag `WINDOW_UNSPLIT` | Reduction | `node.reduction_type`, `node.conv_params`; depthwise uses write symbols | moves |
| `coarse_tile_local_dim_split_domains` `WDC:496`, `_hinted_work_div_syms` `WDC:468` | `Always.TILE_LOCAL`, every op | all running | `node.tile_local_axes` | stays; reads `loop_info`, hints, raw positions |
| `direct_read_source_stick_split_domains` `WDC:237` | `PortFact.DIRECT_READ_SOURCE` | fact | `InPort.direct_read: DirectReadStick(axis, eps)` | domain arithmetic moves; loop-trip gate and recovery stay in builder |
| `_direct_read_source_dep` `WDC:192` | builder | | | stays |
| `aligned_ownership_split_domains` `WDC:299` | `Always.ALIGNED_OWNERSHIP`, every op | all running | frame | stays; reads `align_tensors_pure`, called per frame |
| `restickify_padding_blocked_vars` `WDC:661` | `Always.RESTICKIFY_PADDING` | `RestickifyNode` | frame | moves |
| `_has_ea_tensor`, `has_qfp8wt_tensor`, `has_staggered_ea_tensor` `WDC:683-705` | `port.facts` | | | stays; `WD:2779` imports `has_qfp8wt_tensor` |
| `_qfp8wt_unsplittable_coords` `WDC:708` | `QFP8WT` rule body | | | moves |
| `qfp8wt_split_domains` `WDC:724` | `PortFact.QFP8WT` | out port, and in port with `operand == 1` | port coords | moves; scope kept (surprise 10) |
| `qfp8wt_matmul_k_split_domains` `WDC:756` | `ENCODED_K_UNSPLIT` | Matmul | `port.facts` | moves |
| `_topk_output_k_var` `WDC:786`, `topk_split_domains` `WDC:801` | new flag `TOPK_SPLITS` | Reduction | `node.reduction_type`, `graph.max_cores` | moves |
| `_keep_by_index_axes` `WDC:836`, `keep_by_index_k_split_constraint` `WDC:871`, `_keep_by_index_search_axis` `WDC:893`, `keep_by_index_pinned_search_space_vars` `WDC:926`, `keep_by_index_search_adjacent_blocked_vars` `WDC:938` | new flag `KEEP_BY_INDEX_SPLITS`, one flag for the three rules | Reduction | `in0`, `node.in_ports.get("in1")`, `graph.max_cores` | moves |
| `plain_reduction_k_split_domains` `WDC:1008` | new flag `PLAIN_REDUCTION_UNSPLIT` | Reduction | `node.reduction_type`, frame | moves |
| `indirect_access_split_domains` `WDC:1035`, `indirect_forbidden_split_syms` `PU:1812` | `INDIRECT_ACCESS` | Indirect | ports `table`, `index` | moves; `PU` keeps its name, `indirect_store_entry_syms` calls it |
| `_ordered_affine_coordinate_terms` `WDC:1050`, `overlapping_coordinate_extents` `WDC:1072`, `_moves_coordinate` `WDC:1105`, `overlapping_window_split_domains` `WDC:1112` | `PortFact.OVERLAPPING_READ` | in ports | `port.device_coords`, `stride_map` | moves; old names re-exported |

**B. `WorkDivisionContext.is_legal`.**

| clause | new home | note |
|---|---|---|
| `_factors_in_domain` `WD:939` | `Always.SPLIT_DIVIDES` | Uses `frame.basis(axis)`: sticks, granularity or elements, as `factor_domain` does. An axis not in the frame fails. |
| `_within_core_budget` `WD:951` | `Always.CORE_BUDGET` | `graph.max_cores`, also `ALLOC:3811`. |
| `_one_reduction_split_at_most` `WD:961` | `Always.ONE_REDUCTION_SPLIT` | Over `frame.reduction_axes`, not `Axis.reduction`. |
| `_spans_within_cap` `WD:954` | `Always.SPAN_CAP`, no parent entry | Calls `get_per_core_span` (stays) per port, skipping indirect tensors as `WD:561`. |
| `_in_split_domains` `WD:964` | the flags, port facts and `always_splits` of table A | |
| `meets_span_floors` `WD:930` | `Always.SPAN_FLOOR`, no parent entry | `node.span_floors`, from `_span_min_splits` (`WD:201`). |

All old methods stay until Stage 13 at least; `WorkDivisionContext` serves
`enumerate_work_division_candidates`, which Stage 5 keeps calling.

**C. `residency_rules.py` and `ResidencyChecks`.** All map to `Always.RESIDENCY`. The
registry stays exactly where it is. It is already solver-free, and its order is a contract
(`RR:83`). `residency_reason` runs today's checks through `node.op`, `graph.ir`,
`graph.divisions_committed`, `graph.drain_plans` and `graph.lx_relayout_plans`, with the
graph-wide facts of `_residency_reasons` computed once per graph, so laziness and order are
preserved. The verdict depends on the family, so it is a call, not a stored field. Stage 5
passes `decides_relayouts=True, tileable=table.tileable`; Stage 8 passes
`decides_relayouts=False, tileable=False`. The check calls `residency_bar`, which is
`decides_relayouts=True, tileable=bool(split and split.tiling)`, the most permissive judge.

| check | kind | depends on the division? | evaluated as |
|---|---|---|---|
| `op_allowed` `RR:85` | rule | no | in `residency_reason` |
| `sized`, `empty` | rule | no | same |
| `mutation`, with exception `loop_carry` | rule | only `division_is_fixed` gates the exception (`ALLOC:832`) | same, from `graph.divisions_committed` |
| `advancing` | rule | no | same |
| `restickify` | rule | placement path only (`ALLOC:848`) | same; on the joint path `CROSS_FRAME` covers it. |
| `extern`, `multi_output_extern`, `indirect` | rule | no | same |
| `reinterpret_output`, `mutated_output` with exception `drain` | rule | no | same, from `graph.drain_plans` |
| `partial_read` with `prefix_gather`, `local_subrange` | rule | `prefix_gather` differs by mode (`ALLOC:903-913`) | same |
| `core_division` | rule | placement path only | same |
| `untiled` | rule | yes, lifted by a tiling | `untiled_bar`; `residency_bar` applies it when `not split.tiling` (`ILP:1797-1804`). A solver with `tilings_from_menu()` false treats it as fixed (`PS:1105`), which only narrows. |
| `back_gap` with `prefix_gather` | rule | no | in `residency_reason` |

Rules outside the registry with the same effect, none named in the parent:
`_input_residency_reason` (`ALLOC:1303`), a second unregistered list for graph inputs;
"loop carry update ownership unavailable" (`ALLOC:4214`). `residency_reason` applies both.
`TRANSFER_CERTIFICATES` (`RR:343`) "never dispatch admission" and need no mapping. The
capacity clause of `excluded()` (`PS:1108`) is `NO_OVERLAP`'s bound.

**D. `check_in_place_parent_is_read` and the in-place conditions.** All map to
`Always.IN_PLACE`. The sources stay. `IN_PLACE` and `TILE_ALIGNED` state the physical rule,
deliberately the union of what both solvers guarantee: child footprint at most the parent's,
no in-place across a dead tile, and tile alignment by `TileReads.aligned`. The SA packer
relies on the first (`torch_spyre/csrc/perm_layout_native.cpp:547`). OR-Tools stays stricter
by encoding: equal footprints (`ILP:604`), and a match pair inside a nest even for a spilled
buffer (`ILP:998-1000`).

| source | what the check evaluates |
|---|---|
| `check_in_place_parent_is_read` `PS:903` | A non-input parent has a use after its first. Input parents are exempt. |
| `_check_in_place_relationships` `PS:938` | Parent's last tick equals child's first tick, widened by overrides. |
| `_inplace_edge_ok` `ALLOC:1678` | `same_index_read(port)`, equal `device_layout`, child footprint at most the parent's. |
| `_op_inputs_good_for_lx_inplace` `ALLOC:1426` | `DataEdge.in_place_candidate`, since it reads `origin_node`. |
| `_add_inplace_relaxation` `ILP:1608` | Both resident, equal offset, at most one in and one out per node, no in-place across a dead tile: no `CopyOut` on the edge (`ILP:1647-1648`). |
| `constrain_merge` `ILP:599` | `RESIDENT_READ` on the edge. |

An input is a legal in-place parent (#3212, `ALLOC:1631`). Eligibility stays today's rule for
parity. The read-pattern rule is the parent's Future work and is not built in these stages.

**E. Sticks.** The sources stay.

| source | new home |
|---|---|
| `_matmul_layouts` `PL:1464`, stick choice | `MATMUL_STICKS` |
| `_matmul_layouts` N = 1 `PL:1539`, M = 1 `PL:1556` | `MATMUL_STICKS`, through `MatmulNode.k`, `.n` |
| `_matmul_layouts` output dim order `PL:1597` | none needed: non-stick order, pinned |
| `_check_supported_input_sticks` `PL:383` | nothing: offset-free stick. Pinned until Stage 11. |
| `_violates_matmul_outer_dim_constraints` `PL:1249` | nothing: outer-dim shape of a matmul operand. Pinned until Stage 11. |
| `_find_quantized_candidate` `PL:1294` | nothing: non-STANDARD layouts are candidates, not a stick rule |
| `_multi_arg_pointwise_layouts` Phase 1, lane-0 index pruning `PL:1909` | nothing: index operands are excluded from `POINTWISE_STICKS` |
| Phase 2, `_resolve_pointwise_ea` `PL:1715` | nothing: EA compatibility. Pinned until Stage 11. |
| Phase 3, candidates and `_is_supported_pointwise_layout` `PL:1848` | `POINTWISE_STICKS`, except its offset-free and index-symbol clauses, which are nothing until Stage 11 |
| Phase 4, in-place promotion `PL:2052` | no rule: a tie preference |
| `_check_stick_all_same` `OSV:675` | `POINTWISE_STICKS` on Pointwise, Reduction and Indirect nodes |

**F. Every `Always` entry and its source.**

| `Always` | existing source | Phase 1 evaluation |
|---|---|---|
| `SPLIT_DIVIDES` | `WD:939` | table B |
| `CORE_BUDGET` | `WD:951`, `ALLOC:3811` | table B |
| `ONE_REDUCTION_SPLIT` | `WD:961` | table B |
| `TILES` | `ALLOC:3960` (`is_clean`), `enumerate_tile_options`, `predict_frame` | Each axis of a solve-chosen tiling is non-reduction, not on the stick dim, and `frame_of` is not `None`. Ops already carrying `loop_info` are left alone. A `RestickifyNode` or a `prescribed_regions` op gets no solve-chosen tiling (`ALLOC:4004-4009`), the parent's Loops row. |
| `CUT_TILE` | `ILP:1030` | Tiled, resident, some reader or `graph_output` outside the nest, and a non-empty `reduction_splits`. |
| `RESIDENCY` | table C | `node.resident and residency_bar(node, graph, node.work_split) is not None` |
| `RESTICKIFY` | `OSV:705`, `is_restickify_coords` `PU:771` | At most one `Restickify` per edge; the stick symbols of `realized_by`'s `in0` and out differ. Compares axes, not dims, because the output buffer can permute. |
| `PINNED` | **none**: today's pre-pass restickify is an op | Every `restickify` pin has a matching step. Vacuous on the adapter, live from Stage 9. |
| `COPY_OUT` | **none executable**: the docstring at `ILP:932-934` and the structure of `apply_tilings` | No `CopyOut` step is resident. Vacuous on the adapter, live from Stage 9. |
| `IN_PLACE` | table D | as table D |
| `PORT_FACTS` | table A | `fact_splits` rows |
| `TILE_ALIGNED` | `tile_misread` `CT:522`, `derive_tiling_groups` `CT:423` | `TileReads(graph.ir, src.op, dst.op).aligned(...)` over `tile_spec_of` on each edge inside a nest; per-axis fallback when `graph.ir` is `None`. |
| `NO_OVERLAP` | `ILP:1666`, `ILP:1662` | section 5 |
| `SPAN_CAP`, `SPAN_FLOOR` | table B | table B, on every running op, pinned or not. The limit is a hardware one. |
| `RESIDENT_READ` | `constrain_residency` `ILP:534`, `ResidencyEdge.match_pairs` `WD:1677` | section 5 |
| `RELAYOUT_SOURCE` | `_constrain_relayout_copies` `ILP:1579` | A resident step has a resident source and at least one reader; no edge carries both `CopyOut` and `Relayout`; a resident source with a reduction split has every reader served. |
| `CROSS_FRAME` | `_restickify_barrier` `ALLOC:1434`, placement path; `ILP:534` on the joint path | section 5 |
| `TILE_LOCAL`, `ALIGNED_OWNERSHIP`, `RESTICKIFY_PADDING` | table A | `always_splits` rows |
| `SPLIT_PINS` | `_division_map` `ALLOC:3739-3770` | Every `split` pin with empty `seeds` has `pin.value.split == node.work_split`. A pin with `seeds` is not compared, because the table may hold the seed-aligned division (`ALLOC:3773`); the flag and `SPLIT_DIVIDES` rows judge it. |
| `TABLE_COLUMN` | **none**: a provenance tag for Stage 5's column ties | No check. |

Other existing gates that need no new entry: `_read_across_a_tiling_boundary` (`SA:1194`)
is stricter than `RESIDENT_READ`, because `match_pairs` admits a tiled consumer whose
read does not advance (`WD:1710`). The check enforces the weaker rule both engines share.

### 5. `check_decisions`

```python
@dataclass(frozen=True)
class Violation:
    rule: RuleTag
    subject: str          # node, edge "src->dst.port", or "graph"
    detail: str

@dataclass
class CheckReport:
    violations: list[Violation]
    errors: list[Violation]            # a rule raised while being evaluated
    evaluated: Counter[str]            # per rule tag name
    skipped: Counter[str]              # per "tag: reason"

class RuleViolation(Exception):
    def __init__(self, violations: Sequence[Violation]) -> None: ...
    violations: tuple[Violation, ...]

def check_report(graph: SpyreOptimizationGraph,
                 overrides: RestrictionOverrides | None = None) -> CheckReport: ...
def check_decisions(graph: SpyreOptimizationGraph,
                    overrides: RestrictionOverrides | None = None) -> list[Violation]: ...
def coverage(graph: SpyreOptimizationGraph, encoder: HasEncodes,
             overrides: RestrictionOverrides | None = None) -> list[str]: ...
def nests(graph: SpyreOptimizationGraph) -> dict[OpNode, int]: ...   # the id steps at every cut
def live_boxes(graph: SpyreOptimizationGraph) -> list[LiveBox]: ...
def no_overlap(graph: SpyreOptimizationGraph) -> list[Violation]: ...

@dataclass(frozen=True)
class LiveBox:
    subject: str          # node, or the relayout's first edge
    offset: int           # alignment units, as size
    size: int
    start: int            # _Slots time, end exclusive
    end: int

@dataclass(frozen=True)
class NodeDecision:
    resident: bool
    offset: int | None               # alignment units
    splits: Mapping[str, int]        # by axis name
    reduction: tuple[str, ...]       # names of split reduction axes
    tile_splits: tuple[tuple[str, int], ...]
    spill_reason: str | None
    view: tuple | None               # prefetch_view's work_slice_dims and core_to_slot

def decisions(graph: SpyreOptimizationGraph) -> dict[str, NodeDecision]: ...
```

`check_decisions` is `check_report(...).violations`. Each rule on each subject runs under
its own `try`: a rule that raises lands in `errors` and the others still run. A caller that
must fail on an unjudged rule reads `errors` from `check_report`, as Stage 9 does.

The order per node, for `node.runs`: frame the node, evaluate `SPLIT_DIVIDES`, `CORE_BUDGET`,
`ONE_REDUCTION_SPLIT`, `SPAN_*`, `TILES`, `CUT_TILE`, each flag from `restrictions_for(node,
overrides)`, then `fact_splits` and `always_splits`. A row is evaluated as `split.get(axis,
1) in factors` when `port is None or is_stuck(row)`. For every node, including inputs:
`RESIDENCY`, then the edge rules on `node.out.edges`:
`RESTICKIFY`, `PINNED`, `COPY_OUT`, `IN_PLACE`, `RELAYOUT_SOURCE`, `RESIDENT_READ`,
`CROSS_FRAME`, `TILE_ALIGNED`. Last, once for the graph, `NO_OVERLAP`. A `RestickifyNode` is
judged as a node, by its frozen flags plus `RESTICKIFY_PADDING`. A node with no frame counts
in `skipped` rather than passing silently.

**`RESIDENT_READ`.** For a data edge from a resident source to a reader, with no
`CopyOut` or `Relayout` step:

- On the joint path, the pair must satisfy `match_pairs([src_division], [dst_division])`,
  with each division from `division_from_work_split`, on an edge from
  `CoOptimizingAllocator._parent_residency_edges`, a static method that also carries the
  prefix-contraction exclusion (`ALLOC:4851-4854`). The view code stays where it is.
- For an `InputNode`, each reader's view under its own division must be representable,
  cover its whole core count, and be `same_partition` with `node.prefetch_view`, the test in
  `_clone_divisions_and_matches` (`ALLOC:4782-4794`).
- Pairs from `carry_pairs(graph)` are counted in `skipped`. `get_ncores_for_buffers` judges
  them at commit (`ALLOC:3020`).
- With `graph.divisions_committed` the rule is skipped, because the `core_division`
  residency rule is the same judge and already ran.
- With `graph.ir` `None`, on a synthetic graph, it compares axis splits through `axis_map`.

**`CROSS_FRAME`.** This is `RESIDENT_READ` on an edge whose reader is a `RestickifyNode`, on
the joint path, reported by name so the hazard is visible. A resident source passes if its
write view is `same_partition` with the restickify's read view, or if a `Relayout` that
precedes the `Restickify` in `transforms` has that `to_view`. List order is physical, and
that route is legal.

**`NO_OVERLAP`.** `live_boxes` gives one box per resident node and resident `Relayout`.
A box's `size` is `node.footprint(split.splits, split.tiling, graph.alignment)`, for a
resident input `ceil_div(node.size, graph.alignment * node.prefetch_view.split_product)`, and
`step.footprint` for a `Relayout`. Boxes are
in `_Slots` time (op `t` writes at `2t`, `ILP:229`). A box's span is the **lower bound the
decisions imply**, the same bounds `_add_lifetime_ends` posts (`ILP:1205`), derived
without the solver:

| subject | span |
|---|---|
| untiled, no tiled reader | `[2*start, 2*end - 1)` from `graph.lifetime(node)` |
| untiled, read by a tiled node | end raised to that reader's nest end; an input's start lowered to its nest start |
| tiled | `[2w, e)`, `e` the maximum of `2w + 1`, `2w + 2` if cut, and `2r + 1` for each reader `r` with no `CopyOut` |
| reads through a `Relayout` | do not extend the source: they count as a read at the transfer |
| `Relayout` step, shared by its edges | `[2*start, 2*end - 1)` from `graph.lifetime(step)` |
| in-place child | start plus one slot (`ILP:1714-1722`); parent's end is `2*child + 1` |

A shorter box can only miss an overlap, never invent one, so this cannot produce a false
report. It also avoids `slot_span`, which would make the check trust the values it judges.
The bound is: every box inside `[0, capacity / alignment]`, and no two boxes intersect in
time and space. Boxes that merely touch are fine, as in the 2-D no-overlap. An in-place pair
needs no special case because the child gives up the shared slot.

**`coverage`** is the union of `restrictions_for(node, overrides)` over nodes, minus
`encoder.ENCODES`, one line per missing flag. `HasEncodes` is a `Protocol` with `ENCODES:
Restriction`, so this stage needs no encoder.

### 6. The adapter

```python
def decode_todays_plan(graph: SpyreOptimizationGraph,
                       allocation: Sequence[LifetimeBoundBuffer],
                       accepted_relayouts: Sequence[LXRelayoutPlan]) -> list[Mismatch]: ...

def check_todays_plan(ir: GraphLowering, allocator: ScratchpadAllocator,
                      allocation: Sequence[LifetimeBoundBuffer],
                      accepted_relayouts: Sequence[LXRelayoutPlan],
                      overrides: RestrictionOverrides | None = None) -> PlanCheck: ...

@dataclass(frozen=True)
class Mismatch:
    kind: str       # "lifetime", "size", "residency_reason", "unknown_buffer", "no_edge"
    subject: str
    detail: str

@dataclass
class PlanCheck:
    graph: SpyreOptimizationGraph
    report: CheckReport
    mismatches: list[Mismatch]
    seconds: float
```

`check_todays_plan` calls `build_graph(ir, capacity=allocator.size, divisions_committed=d,
drain_plans=allocator._validated_drain_plans, lx_relayout_plans=plans)`. `d` is true for a
plain `ScratchpadAllocator`, and `plans` are the ones the allocation started from, recorded
by one new line in `_prepare_buffers` (`ALLOC:1086`). `decode_todays_plan` writes decisions
onto that graph and never touches a buffer or an op. In order:

1. **Nodes.** For each buffer that is not a `__spyre_lx_relayout__:` name, take
   `graph.node(b.name)`; a restickify op is a node like any other, and its step gets
   nothing. Set `resident = b.address is not None`, `offset = b.address // graph.alignment`
   and `spill_reason = b.spill_reason`. Set `work_split` with `work_split_from_division(node,
   core_divisions[chosen_division])` when the buffer has one, else from the op's committed
   ownership as `_fixed_core_division` reads it. An op with no buffer stays non-resident
   and takes its committed split. An input has no `work_split`: a resident input sets
   `prefetch_view` from `b.lx_view`, or on the joint path, where that is unset until
   `_post_solve`, from its first reader's view (`ALLOC:5328`).
2. **Agreement.** Let `t` be true when the buffer's menu offers a tiling (`ALLOC:4187-4191`).
   Compare `graph.lifetime(node)`, the total size, `residency_reason(node, graph,
   decides_relayouts=allocator._decides_lx_relayouts, tileable=t)` and, when `t`,
   `untiled_bar(node, graph)` with `b.start_time`, `b.end_time`, `b.size`,
   `b.residency_reason` and `b.untiled_bar`. A difference is a `Mismatch`, a defect in Stage 2
   or here, not in the plan. The feasibility report found multi-output ops and the `-1` size
   sentinel as the likely sources, so a sentinel buffer is compared as "unsized".
3. **Nests and cuts.** `choices = chosen_tilings(ir, allocation)` and
   `derive_tiling_groups(ir, choices)`, the same pair `apply_chosen_tilings` uses. Set
   `ExecutionEdge.cut` wherever two adjacent ops are not in one group. One shared
   `CopyOut` per tiled producer goes on each edge to a reader outside its group, through
   `insert_step(0, step)` so it sits ahead of a pinned step, matching the parent's "one per
   producer". `CopyOut.resident` is `False`.
4. **Relayouts.** One shared `Relayout` per `LXRelayoutPlan`, on each edge from
   `plan.source_name` to a name in `plan.consumer_names`, with `to_view =
   plan.destination_view`, `source_view`, `source_footprint_bytes`,
   `destination_footprint_bytes`, `resident = True`, and `offset` and `footprint` in
   alignment units. `reduction_split` is `plan.num_cores //
   len(plan.completed_producer_cores)` on a completed route
   (`torch_spyre/_inductor/scratchpad/lx_relayout.py:598`), else 1. On an edge with a pinned
   `Restickify` it goes before the step, with `insert_step`, since it reads the producer;
   otherwise `add_step`. `accepted_relayouts` has the same type on both paths.
5. **In place.** Set `edge.in_place` when `src` is in `b.in_place_parents`, both are
   resident at the same address, and the edge has no `CopyOut`. This is exact: without an
   in-place overwrite the two boxes would share the handoff tick, which the 2-D no-overlap forbids
   (`ILP:1682-1684`), except for a tiled producer that frees before a reader of its
   copied-out buffer, which the `CopyOut` condition excludes. A parent listed with no edge
   is a `Mismatch` of kind `no_edge`.

Rules needing live views, tilings or spans reach them through `node.op`, `port.dep` and
`graph.ir`, so they work with no buffers present. `check.py` never imports the adapter.

**Later reuse.** Stages 7, 8, 9 and 10 decode today's plan onto a graph, take
`decisions(graph)`, and compare with `decisions` of the new family's `read_back` graph. SA
passes `accepted_relayouts=()`.

### 7. Running inside the compile

The hook is in the template method, between the solve and the first IR mutation, so it sees
the IR the solve saw and works for every allocator and solver, including the greedy fallback.

```python
allocation = self._solve(solver, graph)
accepted_lx_relayouts = self._finalize_lx_relayout_allocation(allocation, graph)
plan_check = self._check_plan(graph, allocation, accepted_lx_relayouts)    # new
self._post_solve(graph, allocation, accepted_lx_relayouts)
self._finish_plan_check(plan_check, allocation)                            # new
reasons = self._get_spill_reasons(graph, allocation)
```

```python
def _check_plan(self, graph, allocation, accepted) -> "PlanCheck | None":
    if not config.lx_check_decisions:
        return None
    try:
        from .lx_allocator.plan_adapter import check_todays_plan
        return check_todays_plan(graph, self, allocation, accepted)
    except Exception:  # noqa: BLE001 - report-only, see the docstring
        logger.warning("check_decisions could not run", exc_info=True)
        return None
```

- **Staying off the result.** `check_todays_plan` builds its own graph and returns a
  `PlanCheck` holding it. Neither `_post_solve` nor `_push_allocation` reads it, and it takes
  no solver path, so the CP model cannot change.
- **Staying off the failure path.** A `SolveError` comes from `_solve`, before the hook
  (`ALLOC:5493-5495`), so the failed plan is never checked. The greedy re-plan runs the
  template again and is checked once. The `except` is broad on purpose: an error in a
  report-only check must not reach the `SolveError` handler or fail a compile.
  `BaseException` passes through. An `Unsupported` from `_post_solve` fails as today.
- **Late placements.** `_finish_plan_check` handles surprise 15. It sets `resident` and
  `offset` on any node that gained an address and runs `no_overlap(graph)` alone, which reads
  no IR. It does not run `RESIDENCY`: a staged output is exempt from the bars by design. This
  also runs under `try` and writes the record. Stage 10 calls `no_overlap` for the same reason.
- **Reporting.** A record per plan goes to `config.lx_check_file` through `emit_json_line`
  (`torch_spyre/_inductor/dump_common.py:42`), which never raises:

  ```json
  {"test": "<PYTEST_CURRENT_TEST>", "allocator": "...", "solver": "...",
   "sencores": 32, "nodes": 41, "evaluated": {"WHOLE_STICKS": 83},
   "skipped": {"RESIDENT_READ: loop carry": 2}, "violations": [],
   "errors": [], "mismatches": [], "seconds": 0.0}
  ```

  There is no kernel name at this point (`ALLOC:3363`), so `PYTEST_CURRENT_TEST` and op
  names identify a record. `allocator` and `solver` show a fallback to greedy, which is
  the #5682 solve-status guard for free. A non-empty `violations` also logs one warning
  naming the first few, through the `scratchpad.allocator` logger.
- **Config.** Both are added to `torch_spyre/_inductor/config.py`:

  ```python
  # Run check_decisions on every plan the allocators commit. Report only.
  lx_check_decisions: bool = os.environ.get("SPYRE_LX_CHECK_DECISIONS", "1") == "1"
  # Append one JSON line per checked plan to this file.
  lx_check_file: str = os.environ.get("SPYRE_LX_CHECK_FILE", "")
  ```

  `"lx_check"` joins `_cache_config_ignore_prefix` (`config.py:503`), since neither changes
  generated code. `test_each_cache_key_ignore_prefix_matches_only_known_settings`
  (`tests/inductor/test_solver_relayout_decision.py:320`) gets the new prefix.
  `_counted_compile` (`tests/inductor/test_frontend_work_bounds.py:126`) pins
  `lx_check_decisions` to `False` beside its four pins, so the bounds keep meaning analysis.
- **Default.** On by default only if the measured cost is within noise: `seconds` per record
  against `TORCH_SPYRE_TIMING=1` planning time on a live `zoo` or `fake` run, at default and
  tiling, attached to this PR. Otherwise the default is `"0"` and a `tests/conftest.py`
  fixture turns it on for the suites. The cost is unmeasured until Stage 2 exists. Phase 1
  exit sets a budget from this figure, as a share of planning, for Stage 9's gate.
- **Captures.** `PlanCheck` holds op references. It is dropped when `plan_allocation`
  returns and is never pickled, so the FakeTensor caveat does not arise.

### 8. Overrides

The mapping is Stage 1's `RestrictionOverrides`, resolved by `restrictions_for(node,
overrides)`. `splits_while_stuck` and `stick_tables` take a single flag and never see it, so
the override works by choosing which flags are asked about. The production hook always passes
`None`; only tests build one. An override never touches `Always`, residency or port-fact rules.

### 9. Plans with no IR: `check_buffers`

```python
def check_buffers(allocation: Sequence[LifetimeBoundBuffer], capacity: int,
                  alignment: int = 128) -> list[Violation]: ...
```

Most unit suites call solvers directly on synthetic buffers: `test_scratchpad_solver`,
`test_sa_cooptimizer`, `test_cooptimization_types`, `test_restickify_cost_model`,
`test_perm_layout_solver`, `test_solver_relayout_decision`. They have no `GraphLowering`,
so `build_graph` cannot run. The in-repo captures, Stage 0's `intree` tier, are the same.
`check_buffers` evaluates the part of the check that needs only buffers: `NO_OVERLAP`,
`RESIDENCY` from `b.residency_reason`, `IN_PLACE` structure, `CORE_BUDGET`,
`ONE_REDUCTION_SPLIT` from `reduction_syms`, `RESIDENT_READ` as membership of the chosen
pair in `cd_parent_matches` or a served `chosen_relayouts`, and `RELAYOUT_SOURCE`. It shares
`live_boxes`'s span logic through a small helper.

An autouse fixture in `tests/inductor/conftest.py`, active only when `SPYRE_LX_CHECK_FILE` is
set, wraps `plan_layout` and `plan_layout_and_core_divisions` on the concrete solvers and
appends a record, skipping a solve reached through `plan_allocation`, which the hook checks.

## Changes

| file | change |
|---|---|
| `.../scratchpad/lx_allocator/rules.py` | Added: `Always`, `CONSTRAINT_RULES`, `SplitRule`, `StickTable`, `Frame`, `frame_of`, the data functions, the residency functions, moved rule bodies. |
| `.../scratchpad/lx_allocator/check.py` | Added: `Violation`, `CheckReport`, `RuleViolation`, `NodeDecision`, `LiveBox`, `check_report`, `check_decisions`, `coverage`, `decisions`, `nests`, `live_boxes`, `no_overlap`. |
| `.../scratchpad/lx_allocator/plan_adapter.py` | Added: `decode_todays_plan`, `check_todays_plan`, `check_buffers`, `Mismatch`, `PlanCheck`. |
| `torch_spyre/_inductor/work_division_constraints.py` | Moved rule bodies become adapters over `rules.py`. Names, signatures and behaviour unchanged. |
| `torch_spyre/_inductor/pass_utils.py` | `_non_indirect_coord_syms` body moves to `rules.py`, name stays. |
| `torch_spyre/_inductor/scratchpad/allocator.py` | `plan_allocation` gains two calls; `_check_plan`, `_finish_plan_check` added; `_prepare_buffers` records its plans. |
| `torch_spyre/_inductor/config.py` | `lx_check_decisions`, `lx_check_file`, ignore prefix. |
| `tests/inductor/conftest.py` | Autouse fixture for `check_buffers`. |
| `tests/scripts/lx_check_summary.py` | Added: groups records by rule, op kind and solver; exits non-zero on an unlisted report. |
| `tests/inductor/lx_check_known.json` | Added: `{rule, op kind, issue}`, empty at merge. |
| `tests/inductor/test_lx_rules.py`, `test_lx_check.py`, `test_lx_plan_adapter.py`, `test_lx_check_hook.py` | Added. |
| `tests/configs/torch_spyre_tests/inductor/test_lx_*_config.yaml` | One per new test file, `unlisted_test_mode: mandatory_success`. |
| `tests/inductor/test_solver_relayout_decision.py` | New prefix in the cache-key test. |
| `tests/inductor/test_frontend_work_bounds.py` | `_counted_compile` pins the check off. |
| `docs/source/compiler/co_optimizer.md` | One section: the check, its config and its report. |

Ordered commits, each leaving the suite green:

1. Config values, ignore prefix and the cache-key test.
2. `rules.py` skeleton: `Always`, `SplitRule`, `StickTable`, `Frame`, `Frame.from_context`,
   `frame_of`, with tests that a frame equals the context built from the same op.
3. Move `coordinate_mask`, `conv_spatial`, `reduction_window`, `restickify_padding` and
   the two reduction rules. `tests/inductor/test_work_division.py` is unchanged and is
   the behaviour pin.
4. Move the QFP8WT rules, `topk`, `keep_by_index`.
5. Move `direct_read` arithmetic, indirect and overlap rules, with re-exports.
6. `splits_while_stuck`, `fact_splits`, `always_splits` over synthetic nodes.
7. `stuck_dims`, `is_stuck`, `stick_tables`.
8. `residency_reason`, `untiled_bar`, `residency_bar`, `check.py` without views, one
   violation test per rule.
9. `RESIDENT_READ`, `CROSS_FRAME` and `TILE_ALIGNED` through the IR helpers, with the
   synthetic fallback.
10. `plan_adapter.py`: `decode_todays_plan`, `check_buffers`.
11. The hook, with failure-isolation tests and the `_counted_compile` pin.
12. Fixture, summary script, docs, timing report and the default it decides.

Until commit 11 nothing in production calls the new code, and the tests exercise it
through synthetic graphs and the allocator test doubles.

## Interfaces

Consumes:

| symbol | signature | from Stage | used for |
|---|---|---|---|
| harness tiers | `intree`, `fake`, `zoo`, `unit-*`, `failure`, `FIXED_GRAPHS`, `corpus_coverage`; records proto, decisions, committed plan | 0 | parity gate |
| `graph.flags.Restriction` | `NONE` and eleven members: the parent's six plus `WINDOW_UNSPLIT`, `CONV_SPATIAL_UNSPLIT`, `TOPK_SPLITS`, `KEEP_BY_INDEX_SPLITS`, `PLAIN_REDUCTION_UNSPLIT` | 1 | rule tags |
| `graph.flags.PortFact` | `QFP8WT`, `STAGGERED_EA`, `OVERLAPPING_READ`, `DIRECT_READ_SOURCE` | 1 | fact rules |
| per-type `restrictions` | the five on `ReductionNode`; `POINTWISE_STICKS` on `ReductionNode`, `IndirectNode`; `RestickifyNode.restrictions = OpNode.restrictions` | 1 | flags |
| `restrictions_for`, `RestrictionOverrides` | `(node, overrides=None) -> Restriction`; `Mapping[type[OpNode], Restriction]` | 1 | overrides |
| `OpNode` | `name, axes, size, op, tick, uses, read_count, graph_output, span_floors, tile_local_axes, work_split, resident, offset, spill_reason, in_ports: dict[str, InPort], out, port(name), runs, broadcast_dims(port), same_index_read(port), footprint(splits, tiling, alignment)` | 1, filled by 2; the adapter fills the decisions | rules, adapter, `NodeDecision` |
| `OpNode.padded_reductions()`, `.whole_stick_factors(axis)` | answer for the untiled frame only; not called here. One rule is held by a test, not a call | 1 | `test_node_methods_agree_with_rows` |
| `RestickifyNode`, `OpaqueNode` | direct `OpNode` subclasses | 1 | node kinds |
| `InputNode.prefetch_view` | `PerCoreView \| None`; `work_split` stays `None` on inputs | 1 | `RESIDENT_READ` |
| `Port` | `name, node, axis_map, facts, dep: Dep \| None, coords, device_coords` under the read layout; `current_layout()`, `dims_of(axis)`, `stuck_axes()` | 1, filled by 2 | frames, views |
| `InPort`, `OutPort` | `operand, index, edge, read_layout, direct_read: DirectReadStick \| None`; `layout, index, edges` | 1, filled by 2 | rules |
| `MatmulNode.k`, `.n`, `.reduction_type` | `Axis \| None`, `Axis \| None`, `str` | 1, filled by 2 | `MATMUL_STICKS` |
| `ReductionNode.reduction_type`, `.conv_params` | `str`, `dict[str, Any] \| None` | 1, filled by 2 | new flags |
| `IndirectNode.scatter` | `bool` | 1 | `INDIRECT_ACCESS` |
| `Axis` | `Axis(symbol, extent, reduction=False, granularity=None, off_output_coords=False, symbolic=False)` | 1 | rules |
| `WorkSplit` | `splits: Mapping[Axis, int], tiling: tuple[TileLevel, ...]`, `.reduction_splits` | 1 | decisions |
| `work_split_from_division`, `division_from_work_split`, `tile_spec_of` | `(node, division) -> WorkSplit`, `(node, split) -> CoreDivision`, `(node, split) -> TileSpec`, in `division_convert.py` | 2 | adapter, frames, views |
| `DataEdge` | `src, dst, in_place, transforms, mutation, in_place_candidate`; `read_layout()`, `add_step(step)`, `insert_step(index, step)` | 1, filled by 2 | rules |
| `ExecutionEdge` | `before, after, cut`; true unless both ops share a non-empty nest | 1 | `nests` |
| `Pin`, `WorkSplitPin` | `Pin(target, what, value, source)`, `what="split"`, `WorkSplitPin(split, reason, seeds)` | 1, filled by 2 | pin rules |
| `CopyOut`, `Restickify`, `Relayout` | `Restickify(to_stick, realized_by=None, target=None)`; `Relayout(to_view, reduction_split=1, source_view=None, source_footprint_bytes=0, destination_footprint_bytes=0)`; one instance may sit on several edges | 1 | adapter |
| `stick_dims`, `SPARSE_STICK` | `(layout) -> tuple[int, ...]`; `-1` | 1 | `stuck_dims` |
| `SpyreOptimizationGraph` | `capacity, alignment, max_cores, divisions_committed, ir, drain_plans, lx_relayout_plans, nodes, data_edges, execution_edges, pins; node(name), pins_on, step, read_ticks, lifetime` (`OpNode` and `Relayout`); offsets and footprints in alignment units | 1, filled by 2 | rules |
| `build_graph` | `(ir, *, capacity=None, alignment=128, divisions_committed=False, drain_plans=None, lx_relayout_plans=())`; never writes the IR | 2 | adapter |
| `carry_pairs` | `(graph) -> list[CarryPair(storage, reader, role)]` | 2 | `RESIDENT_READ` skip |
| `DirectReadStick` | `(axis, eps)`, set only past the 3-trip gate `WDC:263` | 1, filled by 2 | `DIRECT_READ_SOURCE` |

Provides, all under `torch_spyre._inductor.scratchpad.lx_allocator`:

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `rules.Always`, `rules.RuleTag` | section 2 | 5, 6, 7, 8, 11 | final |
| `rules.CONSTRAINT_RULES` | `dict[str, RuleTag]` | 5 | final |
| `rules.SplitRule`, `rules.StickTable` | section 2 | 5, 13 | final |
| `rules.Frame`, `rules.frame_of` | section 2 | 5 | final |
| `rules.splits_while_stuck` | `(node, flag, tiling=None) -> list[SplitRule]` | 5, 11 | final |
| `rules.fact_splits`, `rules.always_splits` | `(node, tiling=None) -> list[SplitRule]` | 5 | final |
| `rules.stick_tables` | `(node, flag) -> list[StickTable]` | 5, 11, 13 | final |
| `rules.stuck_dims`, `rules.is_stuck` | `(port) -> frozenset[int]`, `(rule) -> bool` | 5 | final |
| `rules.residency_reason` | `(node, graph, *, decides_relayouts, tileable) -> str \| None` | 5, 6, 8, 13 | final |
| `rules.untiled_bar` | `(node, graph) -> str \| None` | 5 | final |
| `check.check_decisions` | `(graph, overrides=None) -> list[Violation]` | 6, 7, 8, 12, 13 | final |
| `check.check_report` | `(graph, overrides=None) -> CheckReport` | 9, 11 | final |
| `check.coverage` | `(graph, encoder, overrides=None) -> list[str]` | 5, 8, 9, 13 | final |
| `check.RuleViolation`, `.Violation`, `.CheckReport` | section 5 | 6-13 | final |
| `check.nests` | `(graph) -> dict[OpNode, int]` | 7 | final |
| `check.live_boxes`, `.LiveBox` | section 5 | 10, through `no_overlap`; `check_buffers` shares them | final |
| `check.no_overlap` | `(graph) -> list[Violation]` | 9, 10 | final |
| `check.decisions`, `check.NodeDecision` | section 5 | 7, 8, 9 | final |
| `plan_adapter.decode_todays_plan`, `.Mismatch` | `(graph, allocation, accepted_relayouts) -> list[Mismatch]` | 7, 8, 9, 10 | temporary, removed in Stage 13 |
| `plan_adapter.check_todays_plan`, `.check_buffers`, `.PlanCheck` | sections 6, 9 | this stage's hook and fixture | temporary, removed in Stage 13 |
| `tests/scripts/lx_check_summary.py`, `tests/inductor/lx_check_known.json` | `lx_check_summary.py <record file>`, exit status as in Parity and gates | 6, 8 | temporary, removed in Stage 13 |
| `lx_check_decisions`, `lx_check_file`, ignore prefix `lx_check` | section 7 | 9 | temporary, removed in Stage 13 |
| `ScratchpadAllocator._check_plan`, `_finish_plan_check` | section 7 | none | temporary, removed in Stage 13 |

## Tests

New, device-free unless marked.

`tests/inductor/test_lx_rules.py`:

| test | asserts |
|---|---|
| `test_frame_matches_context[...]` | On every op fixture, `frame_of(node)` equals the fields of `work_division_context_for_op`, untiled and tiled. This is the seam the moved rules depend on. |
| `test_domains_match_collect[...]` | The union of `splits_while_stuck`, `fact_splits` and `always_splits` rows, reduced to axis domains, equals `collect_work_division_constraints` on the same op and frame. |
| `test_whole_sticks_uses_largest_stick_on_axis` | fp16 and fp32 stuck on 96 elements gives factors `{1, 2}`. |
| `test_split_divides_counts_sticks` | 130 elements at 64 per stick with split 3 passes. |
| `test_padded_reduction_reads_elements_not_sticks` | A padded K gets `{1}`. |
| `test_overlap_pins_outer_to_extent` | A pinned outer axis has factors `{extent}`, not `{1}`. |
| `test_qfp8wt_reaches_out_and_operand_one_only` | A QFP8WT read on `in0` gives no row. |
| `test_encoded_k_pins_all_reduction_axes` | Includes a single-stick output stick axis. |
| `test_indirect_has_no_padding_exception` | A padded gather's partial entry dim stays `{1}`. |
| `test_matmul_tables_m1_n1_k1` | The three simplified cases. |
| `test_pointwise_table_exempts_broadcast_and_sparse` | |
| `test_constraint_rules_cover_collect` | `CONSTRAINT_RULES` keys equal the 16 functions in `collect_work_division_constraints`. |
| `test_node_methods_agree_with_rows` | `padded_reductions` and `whole_stick_factors` equal the untiled rows. |
| `test_residency_reason_matches_allocator[...]` | Equals `_residency_reasons` on real ops, each rule's first failure, for both modes and both `decides_relayouts`. |
| `test_rows_never_raise_on_a_wrong_node_type[...]` | Each flag on each node type. |
| `test_every_flag_has_data` | Every `Restriction` member but `NONE` is handled by a data function, so a new flag fails until it has one. |

`tests/inductor/test_lx_check.py`:

| test | asserts |
|---|---|
| `test_legal_plan_passes[...]` | A hand-decoded legal plan per node type reports nothing, and `evaluated` is non-empty. |
| `test_each_rule_fires[...]` | One parametrized case per `Always` member, flag and `PortFact`: edit one decision, and exactly that rule is reported on that subject. |
| `test_override_removes_a_flag` | The parent's `ReductionNode` example: the padded split is reported by default and not under the override. |
| `test_override_never_touches_always` | Overriding every type to `NONE` still reports `NO_OVERLAP`, `CORE_BUDGET`. |
| `test_coverage_names_missing_flag` | With a stub `ENCODES`, and with overrides. |
| `test_rule_error_is_contained` | A raising rule lands in `errors`, and other rules still run. |
| `test_shorter_box_never_invents_overlap` | Tiled, in-place and relayout span cases. |
| `test_in_place_is_physical` | A smaller child passes; an in-place overwrite on a `CopyOut` edge is reported. |
| `test_tile_aligned_is_physical` | `TileReads.aligned` decides: an aligned nest passes with no match pair on a spilled buffer, and a misread is reported. |
| `test_decisions_fields` | Alignment units, split names, `reduction`, `tile_splits`, `spill_reason`. |
| `test_relayout_before_restickify_is_legal` | `CROSS_FRAME` passes with the relayout first in `transforms`, fails after. |
| `test_rule_violation_lists_violations` | |

`tests/inductor/test_lx_plan_adapter.py`:

| test | asserts |
|---|---|
| `test_in_place_inferred_from_address` | True on an in-place overwrite, false on a `CopyOut` edge with equal addresses. |
| `test_restickify_op_decodes_on_its_node` | Division, residency and offset land on the node; the step gets none. |
| `test_prefetch_view_from_first_reader` | Joint path, `b.lx_view` unset. |
| `test_relayout_shared_by_group` | One instance on two edges, one box, all byte fields set, `reduction_split` on a completed route. |
| `test_spill_reason_copied` | `b.spill_reason` lands on the node, and `decisions` reports it. |
| `test_copy_out_shared_per_producer` | |
| `test_mismatch_reports_lifetime_and_size` | |
| `test_check_buffers_on_synthetic_graphs[...]` | CP-SAT and SA on `synthetic_cooptimization_graphs.synthetic_graphs()`, no violations. |
| `test_check_buffers_on_captures[...]` | Same on `cooptimization_capture_loader.load_captures()`. |
| `test_check_buffers_catches_edits[...]` | An overlapped address, a resident barred buffer, a pair outside `cd_parent_matches`. |

`tests/inductor/test_lx_check_hook.py`:

| test | asserts |
|---|---|
| `test_plan_identical_with_check_on_and_off` | Addresses, chosen divisions and the proto match, CP-SAT at one worker. |
| `test_check_failure_does_not_reach_the_compile[...]` | `check_todays_plan` raises `Unsupported`, `SolveError`, `AssertionError`: `plan_allocation` finishes unchanged and greedy is not entered. |
| `test_solve_failure_checks_only_the_fallback` | One record, naming `ScratchpadAllocator`. |
| `test_check_sees_the_ir_the_solve_saw` | Hook runs before `apply_chosen_tilings`. |
| `test_late_placement_rechecks_no_overlap` | |
| `test_disabled_runs_nothing` | `lx_check_decisions=False` never builds a graph. |

Existing suites, run serially: the parent's unit list, plus `test_indirect_access_gather`,
`test_frontend_work_bounds`, and `tests/inductor/test_work_division.py` unchanged, which pins
the moved rules. Card suites for the exit gate: `test_scratchpad_use`, `test_work_division`,
`test_solver_relayout_e2e`, `test_solver_auto_coarse_tiling`, `test_lx_planning_suite`, plus
`test_work_division_hint` and `test_lx_context_switching`, which call into the allocator and
are not on the parent's device list.

What proves the parent's claim: `test_each_rule_fires` proves each rule has a check,
`test_domains_match_collect` proves rule data equals today's, `test_every_flag_has_data`
holds "a new flag needs three things", the hook tests prove "reports but never fails a
compile", and the exit gate below proves plans pass.

## Parity and gates

**Every stage (parent):** "the Stage 0 harness, comparing proto, decisions and committed
plans between the stage's parent and the change." Compared at the PR tip: `intree`, the `fake`
zoo at both configs, `unit-*` and `failure`, base against this PR, with the check on. Each
commit runs `intree` and the fixed set. A pass is an empty diff. Any difference is a bug in
this PR, because nothing here may reach a solver. `fake` inputs use default layouts, so this
proves parity, not production plans. The timing report on the zoo at default and tiling is
attached to the PR.

**Phase 1 exit (parent):** "across the unit and device suites, `check_decisions` reports
nothing, or every report traces to a real bug that gets an issue."

How it is met. Run the suites on the card with `SPYRE_LX_CHECK_FILE` set, then run
`tests/scripts/lx_check_summary.py`. It groups by rule, op kind and solver, and fails on a
violation, error or mismatch not listed in `lx_check_known.json`. A port defect is fixed. A
real bug gets an issue and a line in the known file. `evaluated` must be non-zero for every
rule that applies to the corpus, so silence means checked, and `skipped` is reviewed in the
same pass.

The Stage 3 PR does not wait for this gate: it merges on the parity gate and the unit suites.
The exit gate closes Phase 1 and must pass before Stage 4 merges. Stages 6 and 8 rerun the
summary, since they add encodings of these rules and a new family.

**What I expect it to report.** None of this has been run. The snapshot has no built
extension and Stage 2 does not exist yet, so these are reasoned expectations, with no counts.

*Silent, because the same compile already judges them.* `MATMUL_STICKS` and
`POINTWISE_STICKS` (`_check_stick_constraints`), `RESIDENT_READ` (`commit_lx_views` raises),
`CORE_BUDGET` (an assert on the joint path), and on CP-SAT `NO_OVERLAP` and in-place
structure. A report in this group is almost certainly a port defect.

*Port defects to fix in place, most of what the first run will show:*

- `POINTWISE_STICKS` and `WHOLE_STICKS` on fused dims, quantized layouts and sparse sticks,
  where `axis_map` and the stick coordinate disagree.
- Ops that are `Pointwise` by class but not by stick rule: `layernormnorm`, `clone` acting
  as a restickify (`PL:2347`), dtype ops, depending on how Stage 2 classifies them.
- Multi-output ops and unsized buffers, as `Mismatch`. The feasibility report saw 18 read-set
  and 86 size differences of this kind on the zoo.
- `_input_layout_overrides`, where `read_layout()` must give the override.
- `SPLIT_DIVIDES`, `TILES` and span rules on tiled frames, the first time real
  `predict_frame` output meets these rules.

*Could be real.* The paths with no `is_legal` behind them: pinned divisions, the pruned
exhaustive menu, the placement path, the greedy fallback, and SA's generated divisions. Watch
`SPLIT_DIVIDES`, `SPAN_FLOOR` and `ONE_REDUCTION_SPLIT`. On the CP-SAT default path I expect
no real report, because every clause is either a menu filter or a model constraint.

*Will not be exercised much.* `PINNED` and `COPY_OUT` are vacuous here. The tiling rules run
only in `test_solver_auto_coarse_tiling` and with `AUTO_COARSE_TILING=1`. Loop-carry pairs
are skipped and judged at commit. Restickifies are close to unexercised on the zoo; the two
restickify graphs in Stage 0's fixed set give `CROSS_FRAME` and `RESTICKIFY` real input. So
"reports nothing" at this gate is weaker for these rules than for the rest.

From #5682's regression guards, this stage reuses plan equivalence through Stage 0 and
the solve-status guard through the record's allocator and solver fields.

## Risks

| risk | caught by |
|---|---|
| A rule move changes behaviour in a corner. | `test_work_division.py`, `test_indirect_access_gather.py`, the Stage 0 proto diff. |
| `frame_of` diverges from `work_division_context_for_op`, so rule data drifts from the menu. | `test_frame_matches_context`, `test_domains_match_collect`. |
| Tests patching `WDC` names stop intercepting after a move. | Those tests fail in commit 3 to 5; old names stay in the call path, so the wrapper must call through its module name. |
| Import cycle: `WDC` imports `rules.py`, which needs `WD`. | `rules.py` keeps IR imports local; import test runs in every commit. |
| The hook costs real compile time on large graphs. `build_graph` is sympy-heavy, and views are per edge. | `seconds` per record on a live run; over noise, the default is off. |
| The check mutates shared state through a memo or `prep_cache`. | Each check uses its own `prep_cache`. `test_plan_identical_with_check_on_and_off`, Stage 0 diff. |
| `build_graph` needs `V.graph`, set during the pass but not in a test double. | `check_todays_plan` raises, is logged, is contained. Reported as a missing record, which the summary counts. |
| A broad `except` hides that the check never runs. | `evaluated` counters; summary fails if a suite has plans but no records. |
| The solver's lifetime exceeds the bound on purpose, so a real overlap of solver spans goes unseen. | Accepted. The bound is what correctness needs: a longer span "can only cost space" (`ILP:1225`). |
| #5695 lands first. | The adapter reads `chosen_division` for joint buffers and ownership only on the placement path, which #5695 leaves. A rebase re-records baselines. |
| #5682 moves `span_reduction` into the solver. | `SPAN_FLOOR` reads `node.span_floors`; only the builder's source changes. |

## Deviations from the parent plan

1. **Six flags cannot hold "each other op-specific constraint function".** The plan names
   `plain_reduction_k_split_domains` and `reduction_window_blocked_vars`, but its `Restriction`
   has no such members, and the code has five such families (`WDC:368`, `WDC:400`, `WDC:801`,
   `WDC:871-992`, `WDC:1008`). `_check_stick_all_same` also covers dtype, most reductions and
   gathers (`OSV:668-672`), not only `PointwiseNode`. Resolution: Stage 1 adds the five flags
   on `ReductionNode` and `POINTWISE_STICKS` on `ReductionNode` and `IndirectNode`. The stick
   rules of `_keep_by_index_layouts`, `_topk_layouts`, `_conv_layouts`, `_exx2_layout` and
   `_layernormnorm_layout` map to no flag, so those ops stay pinned.
2. **Existing rules that map to nothing in the parent.** Resolution: the added `Always`
   members in section 2, for `WDC:496`, `WDC:299`, `WDC:661`, `WD:954`, `WD:930`,
   `constrain_residency` (`ILP:534`), `_constrain_relayout_copies` (`ILP:1579`), the
   fixed-division pins (`ALLOC:3739-3770`) and the cross-frame hazard. `constrain_residency` is
   the most important wrong-code rule in the allocator and is missing from the parent's
   always-on list. `_input_residency_reason` is folded into `RESIDENCY`.
3. **"The split divides the axis" is false for a stick axis.** `factor_domain` uses only the
   stick count (`WD:892-893`), so 130 elements on 3 sticks splits 3 ways. A literal port
   would reject legal plans. Resolution: `SPLIT_DIVIDES` uses `Frame.basis`.
4. **`splits_while_stuck(node, flag)` needs a tiling.** Rule data is frame-relative
   (`WD:1002-1009`), and 50 of 2,674 frames differ in more than extents. Resolution: an
   optional `tiling` argument.
5. **Port-fact rules and all-op rules need an entry point.** Fact rules have no flag to key
   on. Resolution: `fact_splits` and `always_splits`, returning the same `SplitRule`.
6. **QFP8WT is not "every port that carries it".** The code reaches only the output and
   operand 1 (`WDC:739-751`). A literal port would forbid plans that are legal today.
   Resolution: keep today's scope.
7. **The padding exception does not exist.** The plan's `INDIRECT_ACCESS` comment and Stage 3
   text say a padded gather output lifts the ban, through
   `is_output_stick_aligned_for_entry`. No such name exists and `PU:1812-1845` has no
   exception. Resolution: no exception, no builder fact.
8. **`IN_PLACE` and `TILE_ALIGNED` are physical rules.** The parent's In-place overwrite row, "same
   layout and footprint", is what CP-SAT encodes. SA's packer allows a smaller child
   (`torch_spyre/csrc/perm_layout_native.cpp:547`). Resolution: table D's physical form for
   every family; OR-Tools stays stricter by encoding.
9. **The stick functions stay.** The plan says this stage's flags take them over and Stage 13
   deletes them. `_matmul_layouts` and `_multi_arg_pointwise_layouts` generate candidates and
   set `restick_cost_fn` for the beam (`PL:2314`, `PL:2320`, `PL:3175`, `PL:3212`), and
   `_check_stick_all_same` validates emitted OpSpecs on every compile (`OSV:672`).
   Resolution: all three stay, held to the flags by the check on every compile.
10. **The exit gate needs a card run Verification does not schedule.** It lists device suites
    after Stages 5, 6, 8, 9, 10 and 11, while Phase 1 exit asks for them here. Resolution:
    one card run closes Phase 1.
11. **"Runs on every plan in every suite" cannot hold through `build_graph`.** Direct solver
    suites and in-repo captures have no IR. Resolution: `check_buffers` for those plans, a
    partial check, stated as partial.

## Open questions

None.

## Out of scope

- Encoding any of this: Stages 5, 6 and 8. No solver reads the new data here.
- The builder and its fact recording: Stage 2.
- The `Restriction` and `PortFact` classes and the per-type frozen flags: Stage 1.
- `read_back`: Stages 7 and 8. `RuleViolation` being raised: Stage 9.
- Conditional port facts: Stage 11. Offset-free stick, outer-dim and EA rules, and stick rules
  for conv, topk, keep_by_index, exx2 and layernormnorm: those stay pinned.
- Cross-frame hazard and partial-reduction cut guard encodings: Stage 6.
- Loop-carry views: skipped here, and `get_ncores_for_buffers` judges them at commit.
