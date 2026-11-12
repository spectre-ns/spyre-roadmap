# LX allocator, Stage 3: Restrictions and the check

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 1, 2. Needed by: Stage 4 (the Phase 1 exit only) and Stages 5 to 13.

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
12. **`span_cap` differs between its two sources.** `_spans_within_cap` measures every
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
need them, as `ALLOC` does throughout. `check.py` holds the check: the `ALWAYS` and
`RULE_CHECKS` check functions, `optimizations_used`, the driver and `coverage`.
`plan_adapter.py` holds everything that touches today's buffers and is removed in Stage 13.

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

`Rule` and `Optimization` are Stage 1's flag sets, because `Target` holds them. This stage
gives them their checks.

```python
"""rules.py"""
RuleTag = Restriction | PortFact | Rule | Optimization | str  # str: an ALWAYS check's name
TABLE_COLUMN = "table_column"         # provenance only, for Stage 5's column ties; no check
CONSTRAINT_RULES: dict[str, RuleTag]  # keyed by constraint function name, the 16 of table A

def tag_name(tag: RuleTag) -> str: ...

@dataclass(frozen=True)
class SplitRule:
    rule: RuleTag                 # provenance tag, what ortools_encode.post records
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

```python
"""check.py"""
Subject = OpNode | DataEdge | SpyreOptimizationGraph
Verdict = Sequence[Violation] | Skipped | None   # None: not a subject this check judges
Check = Callable[[SpyreOptimizationGraph, Subject], Verdict]

ALWAYS: list[Check] = [
    # The parent's list.
    split_divides,              # the split divides the axis
    core_budget,                # graph.target.cores holds
    residency,                  # the residency rules hold
    restickify_sticks_differ,   # a restickify only where the read stick differs
    pinned_steps_stay,
    in_place_conditions,
    port_fact_rules,            # the rules each port fact names hold
    tiles_line_up,              # tiles line up inside a nest
    boxes_fit_lx,               # resident boxes fit graph.target.lx_bytes, never overlap
    # Added here: existing rules with no parent entry (see Deviations).
    node_config,                # a node's flags are its type's config in graph.target.ops
    span_cap,                   # per-core span within the read-distance limit
    span_floor,                 # split meets the committed span floor
    resident_read,              # a resident source is read under its own view, or a step's
    relayout_source,            # a resident relayout has a resident, priced source, a reader
    tile_local,                 # a coarse-tile-local dim stays whole
    aligned_ownership,          # a split keeps each core's share one contiguous block
    restickify_padding,         # an unaligned restickify stick dim stays on one core
    split_pins,                 # a pinned division stays
    cross_frame,                # a restickify reads a resident source under its own view
]

RULE_CHECKS: Mapping[Rule, Check] = {   # run only for flags in graph.target.rules
    Rule.ONE_REDUCTION_SPLIT: one_reduction_split,
    Rule.TILES_OFF_STICK: tiles_off_stick,
    Rule.RESIDENT_TILE_UNREDUCED: resident_tile_unreduced,
    Rule.COPY_OUT_IN_HBM: copy_out_in_hbm,
}

def optimizations_used(subject: OpNode | DataEdge) -> Optimization: ...
```

`frame_of` returns `None` for a tiling `predict_frame` rejects
(`torch_spyre/_inductor/wsr/tile_prediction.py:291`) and for a node with no iteration space
(an `OpaqueNode`). It rebuilds what `work_division_context_for_op` builds, without running
the constraints, from `iteration_space_from_op`, `adjust_it_space_for_sticks` and the
`reduction_vars` formula. `Frame.from_context(ctx)` is the reverse direction for the old
adapters. A synthetic node with `op=None` uses `Axis` extents, divided by the tile count. A
`WorkSplit` holds `TileLevel`s, so the check calls `frame_of(node, tile_spec_of(node, split))`.

A tag names the rule a row, a constraint or a violation comes from. An always-on rule's tag
is the `__name__` of its check in `ALWAYS`, so `rules.py` can tag rows without importing
`check.py`. `tag_name(tag)` gives the string Stage 5's `post` records in provenance and
`CheckReport` counts by: a flag member's `name`, or the `str` itself. `Flag` iteration over a
composite needs Python 3.11 (`pyproject.toml:42`).

**Off means rejected.** `optimizations_used` gives the decision axes one subject's decisions
use. The check reports each one missing from `graph.target.optimizations`, tagged with that
`Optimization` member:

| `Optimization` | a subject uses it when |
|---|---|
| `CORE_SPLIT` | a node's `work_split` splits some axis by more than 1 |
| `LX_RESIDENCY` | a running node, or a step on an edge, is resident |
| `IN_PLACE` | an edge has `in_place` set |
| `PREFETCH` | an `InputNode` is resident |
| `COARSE_TILING` | a node's `work_split.tiling` is non-empty and the node has no `tiling` hazard pin, which marks an op tiled before the solve |

**Which rules are `Rule` flags.** The parent leaves the line between `Rule` and `ALWAYS`
provisional and asks this stage to confirm it. The test applied: a rule is a `Rule` flag when
it states how this card and its backend realize a decision, so another card could drop it; it
is in `ALWAYS` when it is arithmetic, or when breaking it gives wrong data on any card.

| rule | source | why | verdict |
|---|---|---|---|
| `Rule.ONE_REDUCTION_SPLIT` | `_one_reduction_split_at_most` `WD:961` | nothing in the arithmetic forbids two split reduction axes; the limit is how the backend combines partial sums | `Rule`, confirmed |
| `Rule.TILES_OFF_STICK` | `_tiling_candidates` `ALLOC:3960` | its docstring: reduction tiling goes through the accumulator combine path, the stick dim is never emitted, and a restickify has no correct tiling after stickification | `Rule`, confirmed |
| `Rule.RESIDENT_TILE_UNREDUCED` | `_add_partial_reduction_cut_guard` `ILP:1030` | a split reduction is combined core to core and the finished slice stays on one core of each group, which the per-tile copy cannot read | `Rule`, confirmed; this is the parent's Stage 6 "partial-reduction cut guard" (Deviations) |
| `Rule.COPY_OUT_IN_HBM` | the docstring at `ILP:932-934` | the copy drains each tile into a full-sized HBM buffer; a card with room could keep it in LX | `Rule`, confirmed |
| `span_cap` | `WD:954` | the read-distance limit is a hardware number, but every card has one | `ALWAYS`; it reads today's `MAX_SPAN_BYTES` until a second card needs the limit on `Target` |
| `residency` | table C | several of the registry's rules describe this backend (extern, indirect), but its order is a contract (`RR:83`) | `ALWAYS`, one check; splitting a rule out waits for a card that differs |
| `resident_read`, `cross_frame`, `tiles_line_up`, `in_place_conditions`, `boxes_fit_lx` | sections 4, 5 | a read or a placement that breaks them gives wrong data on any card | `ALWAYS` |
| `split_divides`, `core_budget`, `span_floor`, `aligned_ownership`, `restickify_padding`, `tile_local`, `split_pins`, `node_config` | tables A, B, F | arithmetic over the frame, the target's sizes, or the pins | `ALWAYS` |

No rule moves between the two lists, so the parent's provisional split stands.

### 3. The two data shapes

Both families and the check read exactly these. A row names a port and an axis, never a
menu position, so a later non-enumerating family can post it as a constraint.

**`splits_while_stuck(node, flag, tiling)`** returns `SplitRule`s. A row with `port=None`
holds unconditionally. A row with a port holds only while that port is stuck on `dim`, one
of `dims_of(port, axis)`. While sticks are pinned, `is_stuck(rule)` answers that with a
constant. From Stage 11 a family's `encode` replaces it with a literal.

```python
for rule in splits_while_stuck(node, flag, tiling):
    if rule.port is None or is_stuck(rule):
        prune(entries in this tiling frame whose split on rule.axis not in rule.factors)
```

| flag | rows | tricky case |
|---|---|---|
| `WHOLE_STICKS` | One per port, axis and `dim` in `dims_of(port, axis)`: `(port, dim, axis, divisors(ceil(extent / eps)))` | `eps` is the largest `elems_per_stick` over this port and every port currently stuck on the axis (surprise 4). A port whose `device_coords[-1]` has other than one free symbol is not stuck on anything, as in `WD:464`. QFP8WT sticks on both of `device_coords[-2:]` (`WD:443-461`). |
| `PADDED_REDUCTION_UNSPLIT` | One per reduction axis and port reading it, where `extent % port_eps != 0`: `(port, dim, axis, {1})` | Per-port is equivalent to the old max-over-tensors test because stick sizes are powers of two. Extents come from `frame.extents`, never `frame.sticks`, as the docstring at `WDC:352-358` requires. |
| `ENCODED_K_UNSPLIT` | `(None, None, a, {1})` for each `a` in `frame.reduction_axes`, only when some port has `QFP8WT` or `STAGGERED_EA` | All reduction axes, not only `node.k` (surprise 11). |
| `INDIRECT_ACCESS` | `(None, None, a, {1})` for each shared data axis of port `table`; `(index, dim, entry, {1})` for a partial last index stick | Coordinates of `table` carry `IndirectAccess`; the mixed-coordinate case `d0 + 128*IndirectAccess(idx)` still reports `d0` (`PU:1779-1794`). Partial uses the index's logical range. No padding exception (surprise 8). |
| new flags | `(None, None, a, factors)` | See table A. |

`MATMUL_STICKS` and `POINTWISE_STICKS` return no split rows. Any flag on a node type it
does not fit returns `[]` rather than raising, so a derived target can add one to any type's
config freely.

`fact_splits` and `always_splits` return the same rows, tagged with the `PortFact` member or
the name of the `ALWAYS` check that judges them. Rows are per frame, so a family's `encode`
calls these once per tiling in its table.

**`stick_tables(node, flag)`** returns `StickTable`s. A row is one legal assignment of
stick dims to the table's ports. `stuck_dims(port)` wraps Stage 1's `stick_dims`, the only
layout reader: `frozenset(stick_dims(current_layout(port)) or (SPARSE_STICK,))`, two dims
for QFP8WT, and empty for a port with no layout. On a self-matmul `x` and `y` share one edge.

| flag | tables |
|---|---|
| `MATMUL_STICKS` | Three one-column tables: `x` on `k`, `y` on `n`, out on `n`. If `n` is `None` (N = 1): `y` on `SPARSE_STICK`, out on its last host dim (`PL:1539-1555`). If `k` is `None` (K = 1) there is no `x` table, as `OSV:869-872` skips it. A QFP8WT `y` passes because `stuck_dims` holds two dims. |
| `POINTWISE_STICKS` | One table over `(out, in0 .. inN)`. For each output dim `o` with axis `a`, the row set is the product over in ports of `dims_of(port, a)`, plus `broadcast_dims(port)`, plus `SPARSE_STICK`. For `IndirectNode` the index ports are left out. A `RestickifyNode` has no such flag: `_check_stick_restickify` wants the opposite. |

A fused dim, where two axes map to one host dim, makes the table slightly looser than
`_check_stick_all_same`, which compares `min(free_symbols)` of each stick coordinate, so the
check also compares that symbol across the table's ports. A family's `encode` posts only the
table.

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
| `coarse_tile_local_dim_split_domains` `WDC:496`, `_hinted_work_div_syms` `WDC:468` | `tile_local`, every op | all running | `node.tile_local_axes` | stays; reads `loop_info`, hints, raw positions |
| `direct_read_source_stick_split_domains` `WDC:237` | `PortFact.DIRECT_READ_SOURCE` | fact | `InPort.direct_read: DirectReadStick(axis, eps)` | domain arithmetic moves; loop-trip gate and recovery stay in builder |
| `_direct_read_source_dep` `WDC:192` | builder | | | stays |
| `aligned_ownership_split_domains` `WDC:299` | `aligned_ownership`, every op | all running | frame | stays; reads `align_tensors_pure`, called per frame |
| `restickify_padding_blocked_vars` `WDC:661` | `restickify_padding` | `RestickifyNode` | frame | moves |
| `_has_ea_tensor`, `has_qfp8wt_tensor`, `has_staggered_ea_tensor` `WDC:683-705` | `port.facts` | | | stays; `WD:2779` imports `has_qfp8wt_tensor` |
| `_qfp8wt_unsplittable_coords` `WDC:708` | `QFP8WT` rule body | | | moves |
| `qfp8wt_split_domains` `WDC:724` | `PortFact.QFP8WT` | out port, and in port with `operand == 1` | port coords | moves; scope kept (surprise 10) |
| `qfp8wt_matmul_k_split_domains` `WDC:756` | `ENCODED_K_UNSPLIT` | Matmul | `port.facts` | moves |
| `_topk_output_k_var` `WDC:786`, `topk_split_domains` `WDC:801` | new flag `TOPK_SPLITS` | Reduction | `node.reduction_type`, `graph.target.cores` | moves |
| `_keep_by_index_axes` `WDC:836`, `keep_by_index_k_split_constraint` `WDC:871`, `_keep_by_index_search_axis` `WDC:893`, `keep_by_index_pinned_search_space_vars` `WDC:926`, `keep_by_index_search_adjacent_blocked_vars` `WDC:938` | new flag `KEEP_BY_INDEX_SPLITS`, one flag for the three rules | Reduction | `in0`, `node.in_ports.get("in1")`, `graph.target.cores` | moves |
| `plain_reduction_k_split_domains` `WDC:1008` | new flag `PLAIN_REDUCTION_UNSPLIT` | Reduction | `node.reduction_type`, frame | moves |
| `indirect_access_split_domains` `WDC:1035`, `indirect_forbidden_split_syms` `PU:1812` | `INDIRECT_ACCESS` | Indirect | ports `table`, `index` | moves; `PU` keeps its name, `indirect_store_entry_syms` calls it |
| `_ordered_affine_coordinate_terms` `WDC:1050`, `overlapping_coordinate_extents` `WDC:1072`, `_moves_coordinate` `WDC:1105`, `overlapping_window_split_domains` `WDC:1112` | `PortFact.OVERLAPPING_READ` | in ports | `port.device_coords`, `stride_map` | moves; old names re-exported |

**B. `WorkDivisionContext.is_legal`.**

| clause | new home | note |
|---|---|---|
| `_factors_in_domain` `WD:939` | `split_divides` | Uses `frame.basis(axis)`: sticks, granularity or elements, as `factor_domain` does. An axis not in the frame fails. |
| `_within_core_budget` `WD:951` | `core_budget` | `graph.target.cores`, also `ALLOC:3811`. |
| `_one_reduction_split_at_most` `WD:961` | `Rule.ONE_REDUCTION_SPLIT` | Over `frame.reduction_axes`, not `Axis.reduction`. Checked when the flag is in `graph.target.rules`. |
| `_spans_within_cap` `WD:954` | `span_cap`, no parent entry | Calls `get_per_core_span` (stays) per port, skipping indirect tensors as `WD:561`. |
| `_in_split_domains` `WD:964` | the node's flags, port facts and `always_splits` of table A | |
| `meets_span_floors` `WD:930` | `span_floor`, no parent entry | `node.span_floors`, from `_span_min_splits` (`WD:201`). |

All old methods stay until Stage 13 at least; `WorkDivisionContext` serves
`enumerate_work_division_candidates`, which Stage 5 keeps calling.

**C. `residency_rules.py` and `ResidencyChecks`.** All map to `residency`. The
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
| `restickify` | rule | placement path only (`ALLOC:848`) | same; on the joint path `cross_frame` covers it. |
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
capacity clause of `excluded()` (`PS:1108`) is `boxes_fit_lx`'s bound.

**D. `check_in_place_parent_is_read` and the in-place conditions.** All map to
`in_place_conditions`. The sources stay. `in_place_conditions` and `tiles_line_up` state the
physical rule, deliberately the union of what both solvers guarantee: child footprint at
most the parent's, no in-place across a dead tile, and tile alignment by
`TileReads.aligned`. The SA packer relies on the first
(`torch_spyre/csrc/perm_layout_native.cpp:547`). OR-Tools stays stricter by encoding: equal
footprints (`ILP:604`), and a match pair inside a nest even for a spilled buffer
(`ILP:998-1000`).

| source | what the check evaluates |
|---|---|
| `check_in_place_parent_is_read` `PS:903` | A non-input parent has a use after its first. Input parents are exempt. |
| `_check_in_place_relationships` `PS:938` | Parent's last tick equals child's first tick, widened by overrides. |
| `_inplace_edge_ok` `ALLOC:1678` | `same_index_read(port)`, equal `device_layout`, child footprint at most the parent's. |
| `_op_inputs_good_for_lx_inplace` `ALLOC:1426` | `DataEdge.in_place_candidate`, since it reads `origin_node`. |
| `_add_inplace_relaxation` `ILP:1608` | Both resident, equal offset, at most one in and one out per node, no in-place across a dead tile: no `COPY_OUT` step on the edge (`ILP:1647-1648`). |
| `constrain_merge` `ILP:599` | `resident_read` on the edge. |

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

**F. Every `ALWAYS` check and `Rule` flag, and its source.** A `Rule` row is evaluated only
when the flag is in `graph.target.rules`; otherwise it counts in `skipped`.

| check or flag | existing source | Phase 1 evaluation |
|---|---|---|
| `split_divides` | `WD:939` | table B |
| `core_budget` | `WD:951`, `ALLOC:3811` | table B |
| `Rule.ONE_REDUCTION_SPLIT` | `WD:961` | table B |
| `Rule.TILES_OFF_STICK` | `ALLOC:3960` (`is_clean`), `enumerate_tile_options`, `predict_frame` | Each axis of a solve-chosen tiling is non-reduction, not on the stick dim, and `frame_of` is not `None`. Ops already carrying `loop_info` are left alone. A `RestickifyNode` or a `prescribed_regions` op gets no solve-chosen tiling (`ALLOC:4004-4009`), the parent's Loops row. |
| `Rule.RESIDENT_TILE_UNREDUCED` | `ILP:1030` | Tiled, resident, some reader or `graph_output` outside the nest, and a non-empty `reduction_splits`. |
| `residency` | table C | `node.resident and residency_bar(node, graph, node.work_split) is not None` |
| `restickify_sticks_differ` | `OSV:705`, `is_restickify_coords` `PU:771` | At most one `RESTICKIFY` step per edge; the stick symbols of `realized_by`'s `in0` and out differ. Compares axes, not dims, because the output buffer can permute. |
| `pinned_steps_stay` | **none**: today's pre-pass restickify is an op | Every `restickify` pin has a matching step. Vacuous on the adapter, live from Stage 9. |
| `Rule.COPY_OUT_IN_HBM` | **none executable**: the docstring at `ILP:932-934` and the structure of `apply_tilings` | No `COPY_OUT` step is resident. Vacuous on the adapter, live from Stage 9. |
| `in_place_conditions` | table D | as table D |
| `port_fact_rules` | table A | `fact_splits` rows |
| `tiles_line_up` | `tile_misread` `CT:522`, `derive_tiling_groups` `CT:423` | `TileReads(graph.ir, src.op, dst.op).aligned(...)` over `tile_spec_of` on each edge inside a nest; per-axis fallback when `graph.ir` is `None`. |
| `boxes_fit_lx` | `ILP:1666`, `ILP:1662` | section 5 |
| `span_cap`, `span_floor` | table B | table B, on every running op, pinned or not. The limit is a hardware one. |
| `resident_read` | `constrain_residency` `ILP:534`, `ResidencyEdge.match_pairs` `WD:1677` | section 5 |
| `relayout_source` | `_constrain_relayout_copies` `ILP:1579` | A resident step has a resident source and at least one reader; no edge carries both a `COPY_OUT` and a `RELAYOUT` step; a resident source with a reduction split has every reader served. |
| `cross_frame` | `_restickify_barrier` `ALLOC:1434`, placement path; `ILP:534` on the joint path | section 5 |
| `tile_local`, `aligned_ownership`, `restickify_padding` | table A | `always_splits` rows |
| `split_pins` | `_division_map` `ALLOC:3739-3770` | Every `split` pin with empty `seeds` has `pin.value.split == node.work_split`. A pin with `seeds` is not compared, because the table may hold the seed-aligned division (`ALLOC:3773`); the flag and `split_divides` rows judge it. |
| `node_config` | **none**: Stage 2's build assertion, repeated | Every node's `restrictions` equals `graph.target.ops.ops[type(node)].restrictions`. |
| `Optimization` members | **none**: today every axis is on, coarse tiling under `auto_coarse_tiling` | `optimizations_used`, section 2. On `SPYRE` only `COARSE_TILING` can be off. |
| `TABLE_COLUMN` | **none**: a provenance tag for Stage 5's column ties | No check. |

Other existing gates that need no new entry: `_read_across_a_tiling_boundary` (`SA:1194`)
is stricter than `resident_read`, because `match_pairs` admits a tiled consumer whose
read does not advance (`WD:1710`). The check enforces the weaker rule both engines share.

### 5. `check_decisions`

```python
@dataclass(frozen=True)
class Violation:
    rule: RuleTag
    subject: str          # node, edge "src->dst.port", or "graph"
    detail: str

@dataclass(frozen=True)
class Skipped:
    reason: str

@dataclass
class CheckReport:
    violations: list[Violation]
    errors: list[Violation]            # a rule raised while being evaluated
    evaluated: Counter[str]            # per rule tag name
    skipped: Counter[str]              # per "tag: reason"

class RuleViolation(Exception):
    def __init__(self, violations: Sequence[Violation]) -> None: ...
    violations: tuple[Violation, ...]

class FamilyCapabilities(Protocol):
    ENCODES: ClassVar[Restriction]
    RULES: ClassVar[Rule]
    STEPS: ClassVar[frozenset[StepKind]]
    OPTIMIZATIONS: ClassVar[Optimization]

def check_report(graph: SpyreOptimizationGraph) -> CheckReport: ...
def check_decisions(graph: SpyreOptimizationGraph) -> list[Violation]: ...
@dataclass(frozen=True)
class Coverage:
    gaps: list[str]       # used by the target, not encodable: the family cannot plan here
    unused: list[str]     # offered by the target, never used by the family: not a gap

def coverage(family: FamilyCapabilities, target: Target) -> Coverage: ...
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

`check_decisions(graph)` is `check_report(graph).violations`. Neither takes a target or an
override: every size, flag set and rule switch comes from `graph.target`, so the check and
the solve judge the same card. Each check on each subject runs under its own `try`: a check
that raises lands in `errors` and the others still run. A caller that must fail on an
unjudged rule reads `errors` from `check_report`, as Stage 9 does.

`check_report` visits each node in `graph.nodes` order, then the edges on `node.out.edges`,
then the graph once. On each subject it runs, in order:

1. `optimizations_used(subject)`, reporting each member outside
   `graph.target.optimizations`;
2. every check in `ALWAYS`, in list order;
3. `RULE_CHECKS[flag]` for each flag in `graph.target.rules`, tagged with the flag. A
   `Rule` member the target leaves out counts in `skipped` as `"<flag>: off on <name>"`;
4. on a running node, each flag in `node.restrictions` through `splits_while_stuck` and
   `stick_tables`. A row is evaluated as `split.get(axis, 1) in factors` when
   `port is None or is_stuck(row)`. `port_fact_rules` and the row-backed checks
   (`tile_local`, `aligned_ownership`, `restickify_padding`) read `fact_splits` and
   `always_splits` the same way.

A check returns `None` for a subject it does not judge, so `evaluated` counts only real
judgements. A check that cannot judge, a node with no frame for example, returns `Skipped`
and counts in `skipped` rather than passing silently. A `RestickifyNode` is judged as a
node, by its config's flags plus `restickify_padding`.

**`resident_read`.** For a data edge from a resident source to a reader, with no
`COPY_OUT` or `RELAYOUT` step:

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

**`cross_frame`.** This is `resident_read` on an edge whose reader is a `RestickifyNode`, on
the joint path, reported by name so the hazard is visible. A resident source passes if its
write view is `same_partition` with the restickify's read view, or if a `RELAYOUT` step
that precedes the `RESTICKIFY` step in `edge.steps` has that `to_view`. List order is
physical, and that route is legal.

**`boxes_fit_lx`.** `live_boxes` gives one box per resident node and resident `RELAYOUT`
step. A box's `size` is `node.footprint(split.splits, split.tiling, graph.target.alignment)`,
in alignment units, for a resident input `ceil_div(node.size, graph.target.alignment *
node.prefetch_view.split_product)`, and `step.footprint` for a `RELAYOUT` step. Boxes are
in `_Slots` time (op `t` writes at `2t`, `ILP:229`). A box's span is the **lower bound the
decisions imply**, the same bounds `_add_lifetime_ends` posts (`ILP:1205`), derived
without the solver:

| subject | span |
|---|---|
| untiled, no tiled reader | `[2*start, 2*end - 1)` from `graph.lifetime(node)` |
| untiled, read by a tiled node | end raised to that reader's nest end; an input's start lowered to its nest start |
| tiled | `[2w, e)`, `e` the maximum of `2w + 1`, `2w + 2` if `w in graph.cuts`, and `2r + 1` for each reader `r` with no `COPY_OUT` step |
| reads through a `RELAYOUT` step | do not extend the source: they count as a read at the transfer |
| `RELAYOUT` step, shared by its edges | `[2*start, 2*end - 1)` from `graph.lifetime(step)` |
| in-place child | start plus one slot (`ILP:1714-1722`); parent's end is `2*child + 1` |

A shorter box can only miss an overlap, never invent one, so this cannot produce a false
report. It also avoids `slot_span`, which would make the check trust the values it judges.
The bound is: every box inside `[0, graph.target.lx_bytes / graph.target.alignment]`, and no
two boxes intersect in time and space. Boxes that merely touch are fine, as in the 2-D
no-overlap. An in-place pair needs no special case because the child gives up the shared
slot.

**`coverage(family, target)`** runs at config time, so it reads the target, not a graph. A
gap is something the target requires that the family cannot honour, one line in `gaps` per
member: each `Restriction` in the OR of `target.ops.ops[t].restrictions` over every node
type in `target.ops`, missing from `ENCODES`; each flag in `target.rules` missing from
`RULES`; each kind in `target.steps` with `STEP_KINDS[kind].forced`, missing from `STEPS`.
A chosen step kind (`RELAYOUT`) or an `Optimization` the family lacks is not a gap: the
family never uses it, and the plan is still legal. Each goes in `unused`, so a report can
say what a family leaves on the table. Stage 7's `Family` protocol satisfies
`FamilyCapabilities`, so this stage needs no family. Stages 7 and 8 run it over every
(family, target) pair, and a non-empty `gaps` is an explicit skip.

### 6. The adapter

```python
def decode_todays_plan(graph: SpyreOptimizationGraph,
                       allocation: Sequence[LifetimeBoundBuffer],
                       accepted_relayouts: Sequence[LXRelayoutPlan]) -> list[Mismatch]: ...

def check_todays_plan(ir: GraphLowering, allocator: ScratchpadAllocator,
                      allocation: Sequence[LifetimeBoundBuffer],
                      accepted_relayouts: Sequence[LXRelayoutPlan]) -> PlanCheck: ...

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

`check_todays_plan` calls `build_graph(ir, target, divisions_committed=d,
drain_plans=allocator._validated_drain_plans, lx_relayout_plans=plans)` with `target =
replace(select_target(config), lx_bytes=allocator.size)`, so the check judges the capacity
the allocator planned against. `d` is true for a plain `ScratchpadAllocator`, and `plans`
are the ones the allocation started from, recorded by one new line in `_prepare_buffers`
(`ALLOC:1086`). `decode_todays_plan` writes decisions onto that graph and never touches a
buffer or an op. In order:

1. **Nodes.** For each buffer that is not a `__spyre_lx_relayout__:` name, take
   `graph.node(b.name)`; a restickify op is a node like any other, and its step gets
   nothing. Set `resident = b.address is not None`, `offset = b.address // graph.target.alignment`
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
   `derive_tiling_groups(ir, choices)`, the same pair `apply_chosen_tilings` uses. Add the
   tick of the earlier op to `graph.cuts` wherever two adjacent running ops are not in one
   group. One shared `Step(kind=StepKind.COPY_OUT, resident=False)` per tiled producer goes
   on each edge to a reader outside its group, through `insert_step(0, step)` so it sits
   ahead of a pinned step, matching the parent's "one per producer".
4. **Relayouts.** One shared `Step(kind=StepKind.RELAYOUT, ...)` per `LXRelayoutPlan`, on
   each edge from
   `plan.source_name` to a name in `plan.consumer_names`, with `to_view =
   plan.destination_view`, `source_view`, `source_footprint_bytes`,
   `destination_footprint_bytes`, `resident = True`, and `offset` and `footprint` in
   alignment units. `reduction_split` is `plan.num_cores //
   len(plan.completed_producer_cores)` on a completed route
   (`torch_spyre/_inductor/scratchpad/lx_relayout.py:598`), else 1. On an edge with a pinned
   `RESTICKIFY` step it goes before that step, with `insert_step`, since it reads the producer;
   otherwise `add_step`. `accepted_relayouts` has the same type on both paths.
5. **In place.** Set `edge.in_place` when `src` is in `b.in_place_parents`, both are
   resident at the same address, and the edge has no `COPY_OUT` step. This is exact: without an
   in-place overwrite the two boxes would share the handoff tick, which the 2-D no-overlap forbids
   (`ILP:1682-1684`), except for a tiled producer that frees before a reader of its
   copied-out buffer, which the `COPY_OUT` condition excludes. A parent listed with no edge
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
  no solver path, so the CP model cannot change. `test_optimum_unchanged_with_check_on_and_off`
  tests this on real compiles.
- **Staying off the failure path.** A `SolveError` comes from `_solve`, before the hook
  (`ALLOC:5493-5495`), so the failed plan is never checked. The greedy re-plan runs the
  template again and is checked once. The `except` is broad on purpose: an error in a
  report-only check must not reach the `SolveError` handler or fail a compile.
  `BaseException` passes through. An `Unsupported` from `_post_solve` fails as today.
- **Late placements.** `_finish_plan_check` handles surprise 15. It sets `resident` and
  `offset` on any node that gained an address and runs `no_overlap(graph)` alone, which reads
  no IR. It does not run `residency`: a staged output is exempt from the bars by design. This
  also runs under `try` and writes the record. Stage 10 calls `no_overlap` for the same reason.
- **Reporting.** A record per plan goes to `config.lx_check_file` through `emit_json_line`
  (`torch_spyre/_inductor/dump_common.py:42`), which never raises:

  ```json
  {"test": "<PYTEST_CURRENT_TEST>", "allocator": "...", "solver": "...",
   "target": "spyre", "sencores": 32, "nodes": 41, "evaluated": {"WHOLE_STICKS": 83},
   "skipped": {"resident_read: loop carry": 2}, "violations": [],
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
  against `TORCH_SPYRE_TIMING=1` planning time on a card run of the existing compile suites,
  `test_building_blocks` among them, at default and with `auto_coarse_tiling=True`, attached
  to this PR. Otherwise the default is `"0"` and a `tests/conftest.py`
  fixture turns it on for the suites. The cost is unmeasured until Stage 2 exists. Phase 1
  exit sets a budget from this figure, as a share of planning, for Stage 9's gate.
- **Captures.** `PlanCheck` holds op references. It is dropped when
  `ScratchpadAllocator.plan_allocation` returns and is never pickled, so the FakeTensor
  caveat does not arise.

### 8. Changing flags in tests

The check reads every flag from `graph.target`, so a test changes flags by building with a
derived target, never by passing anything to `check_decisions` or `coverage`:

```python
no_padded = replace(SPYRE, ops=SPYRE_OPS.with_type(
    ReductionNode,
    restrictions=SPYRE_OPS.ops[ReductionNode].restrictions
    & ~Restriction.PADDED_REDUCTION_UNSPLIT))
two_reductions = replace(SPYRE, rules=SPYRE.rules & ~Rule.ONE_REDUCTION_SPLIT)
no_tiling = replace(SPYRE, optimizations=SPYRE.optimizations & ~Optimization.COARSE_TILING)
```

A node's flags live in its type's `OpConfig` in `SPYRE_OPS`, not on the node type, so
`splits_while_stuck` and `stick_tables`, which take one flag, see a derived target only
through which flags `node.restrictions` holds. The production hook always uses
`select_target(config)`. A target can drop a `Rule` flag or an `Optimization`, but never an
`ALWAYS` check, a residency rule or a port-fact rule.

### 9. Plans with no IR: `check_buffers`

```python
def check_buffers(allocation: Sequence[LifetimeBoundBuffer],
                  target: Target) -> list[Violation]: ...
```

Most unit suites call solvers directly on synthetic buffers: `test_scratchpad_solver`,
`test_sa_cooptimizer`, `test_cooptimization_types`, `test_restickify_cost_model`,
`test_perm_layout_solver`, `test_solver_relayout_decision`. They have no `GraphLowering`,
so `build_graph` cannot run. The in-repo captures (`cooptimization_capture_loader.py`) are
the same.
`target` gives the capacity, alignment, core budget and `Rule` flags; the fixture passes
`replace(select_target(config), lx_bytes=solver_capacity)`. `check_buffers` evaluates the
part of the check that needs only buffers: `boxes_fit_lx`, `residency` from
`b.residency_reason`, `in_place_conditions` structure, `core_budget`,
`Rule.ONE_REDUCTION_SPLIT` from `reduction_syms` when the flag is in `target.rules`,
`resident_read` as membership of the chosen
pair in `cd_parent_matches` or a served `chosen_relayouts`, and `relayout_source`. It shares
`live_boxes`'s span logic through a small helper.

An autouse fixture in `tests/inductor/conftest.py`, active only when `SPYRE_LX_CHECK_FILE` is
set, wraps `plan_layout` and `plan_layout_and_core_divisions` on the concrete solvers and
appends a record, skipping a solve reached through `ScratchpadAllocator.plan_allocation`,
which the hook checks.

## Changes

| file | change |
|---|---|
| `.../scratchpad/lx_allocator/rules.py` | Added: `RuleTag`, `TABLE_COLUMN`, `tag_name`, `CONSTRAINT_RULES`, `SplitRule`, `StickTable`, `Frame`, `frame_of`, the data functions, the residency functions, moved rule bodies. |
| `.../scratchpad/lx_allocator/check.py` | Added: `Violation`, `Skipped`, `Check`, `ALWAYS` and its checks, `RULE_CHECKS` and its checks, `optimizations_used`, `FamilyCapabilities`, `CheckReport`, `RuleViolation`, `NodeDecision`, `LiveBox`, `check_report`, `check_decisions`, `coverage`, `Coverage`, `decisions`, `nests`, `live_boxes`, `no_overlap`. |
| `.../scratchpad/lx_allocator/plan_adapter.py` | Added: `decode_todays_plan`, `check_todays_plan`, `check_buffers`, `Mismatch`, `PlanCheck`. |
| `torch_spyre/_inductor/work_division_constraints.py` | Moved rule bodies become adapters over `rules.py`. Names, signatures and behaviour unchanged. |
| `torch_spyre/_inductor/pass_utils.py` | `_non_indirect_coord_syms` body moves to `rules.py`, name stays. |
| `torch_spyre/_inductor/scratchpad/allocator.py` | `ScratchpadAllocator.plan_allocation` gains two calls; `_check_plan`, `_finish_plan_check` added; `_prepare_buffers` records its plans. |
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
2. `rules.py` skeleton: `RuleTag`, `tag_name`, `SplitRule`, `StickTable`, `Frame`,
   `Frame.from_context`, `frame_of`, with tests that a frame equals the context built from
   the same op.
3. Move `coordinate_mask`, `conv_spatial`, `reduction_window`, `restickify_padding` and
   the two reduction rules. `tests/inductor/test_work_division.py` is unchanged and is
   the behaviour pin.
4. Move the QFP8WT rules, `topk`, `keep_by_index`.
5. Move `direct_read` arithmetic, indirect and overlap rules, with re-exports.
6. `splits_while_stuck`, `fact_splits`, `always_splits` over synthetic nodes.
7. `stuck_dims`, `is_stuck`, `stick_tables`.
8. `residency_reason`, `untiled_bar`, `residency_bar`, `check.py` without views: `ALWAYS`,
   `RULE_CHECKS`, `optimizations_used`, `coverage`, one violation test per check, `Rule`
   flag and `Optimization`.
9. `resident_read`, `cross_frame` and `tiles_line_up` through the IR helpers, with the
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
| `graph.flags.Restriction` | `NONE` and eleven members: the parent's six plus `WINDOW_UNSPLIT`, `CONV_SPATIAL_UNSPLIT`, `TOPK_SPLITS`, `KEEP_BY_INDEX_SPLITS`, `PLAIN_REDUCTION_UNSPLIT` | 1 | rule tags |
| `graph.flags.PortFact` | `QFP8WT`, `STAGGERED_EA`, `OVERLAPPING_READ`, `DIRECT_READ_SOURCE` | 1 | fact rules |
| `graph.flags.Rule` | `NONE`, `ONE_REDUCTION_SPLIT`, `TILES_OFF_STICK`, `RESIDENT_TILE_UNREDUCED`, `COPY_OUT_IN_HBM` | 1 | `RULE_CHECKS`, `coverage` |
| `graph.flags.Optimization` | `NONE`, `CORE_SPLIT`, `LX_RESIDENCY`, `IN_PLACE`, `PREFETCH`, `COARSE_TILING` | 1 | `optimizations_used`, `coverage` |
| `Target`, `SPYRE`, `TARGETS` | `name, ops, cores, lx_bytes, alignment, stick_bytes, rules, steps, optimizations, cost_model`; frozen, derived with `dataclasses.replace` | 1 | sizes, rule switches, `coverage`, test targets |
| `OpFactory`, `OpConfig`, `SPYRE_OPS` | `OpFactory.ops: Mapping[type[OpNode], OpConfig]`, `with_type(node_type, **changes)`; `OpConfig(restrictions, facts)`. `SPYRE_OPS` sets the five new flags in the `ReductionNode` config, `POINTWISE_STICKS` in the `ReductionNode` and `IndirectNode` configs, and `WHOLE_STICKS` alone in the `RestickifyNode` config | 1 | `node_config`, `coverage`, test targets |
| `OpNode` | `restrictions: Restriction` (an instance field `make` sets); `name, axes, size, op, tick, uses, read_count, graph_output, span_floors, tile_local_axes, work_split, resident, offset, spill_reason, in_ports: dict[str, InPort], out, port(name), runs, broadcast_dims(port), same_index_read(port), footprint(splits, tiling, alignment)` (alignment units) | 1, filled by 2; the adapter fills the decisions | rules, adapter, `NodeDecision` |
| `OpNode.padded_reductions()`, `.whole_stick_factors(axis)` | answer for the untiled frame only; not called here. One rule is held by a test, not a call | 1 | `test_node_methods_agree_with_rows` |
| `RestickifyNode`, `OpaqueNode` | direct `OpNode` subclasses | 1 | node kinds |
| `InputNode.prefetch_view` | `PerCoreView \| None`; `work_split` stays `None` on inputs | 1 | `resident_read` |
| `InPort`, `OutPort`, `Port` | both: `name, node, axis_map, facts, dep: Dep \| None, coords, device_coords` under the read layout; `InPort`: `operand, index, edge, read_layout, direct_read: DirectReadStick \| None`; `OutPort`: `layout, index, edges`; `Port = InPort \| OutPort` | 1, filled by 2 | rules, frames, views |
| `current_layout`, `dims_of`, `stuck_axes` | `(port)`, `(port, axis)`, `(port)`, module functions in `nodes.py` | 1 | frames, views, `stuck_dims` |
| `MatmulNode.k`, `.n`, `.reduction_type` | `Axis \| None`, `Axis \| None`, `str` | 1, filled by 2 | `MATMUL_STICKS` |
| `ReductionNode.reduction_type`, `.conv_params` | `str`, `dict[str, Any] \| None` | 1, filled by 2 | new flags |
| `IndirectNode.scatter` | `bool` | 1 | `INDIRECT_ACCESS` |
| `Axis` | `Axis(symbol, extent, reduction=False, granularity=None, off_output_coords=False, symbolic=False)` | 1 | rules |
| `WorkSplit` | `splits: Mapping[Axis, int], tiling: tuple[TileLevel, ...]`, `.reduction_splits` | 1 | decisions |
| `work_split_from_division`, `division_from_work_split`, `tile_spec_of` | `(node, division) -> WorkSplit`, `(node, split) -> CoreDivision`, `(node, split) -> TileSpec`, in `division_convert.py` | 2 | adapter, frames, views |
| `DataEdge` | `src, dst, in_place, steps, mutation, in_place_candidate`; `read_layout()`, `add_step(step)`, `insert_step(index, step)` | 1, filled by 2 | rules |
| `Pin`, `WorkSplitPin` | `Pin(target, what, value, source)`, `what="split"`, `WorkSplitPin(split, reason, seeds)` | 1, filled by 2 | pin rules |
| `Step`, `StepKind` | `Step(kind, resident, offset, footprint, pinned, edges, to_stick, realized_by, target, to_view, reduction_split, source_view, source_footprint_bytes, destination_footprint_bytes)`, `kw_only`, identity comparison; `StepKind.COPY_OUT`, `RESTICKIFY`, `RELAYOUT`; one instance may sit on several edges; `STEP_KINDS[kind].forced` | 1 | adapter, edge checks, `coverage` |
| `stick_dims`, `SPARSE_STICK` | `(layout) -> tuple[int, ...]`; `-1`, in `steps.py` | 1 | `stuck_dims` |
| `SpyreOptimizationGraph` | `target, divisions_committed, ir, drain_plans, lx_relayout_plans, nodes` (op order), `data_edges, pins, cuts: set[int]` (the tick before each cut); `node(name), pins_on, step, read_ticks, lifetime` (`OpNode` and a `RELAYOUT` step); offsets and footprints in units of `target.alignment` | 1, filled by 2 | rules |
| `build_graph`, `select_target` | `(ir, target, *, divisions_committed=False, drain_plans=None, lx_relayout_plans=())`, never writes the IR, asserts one config per type; `(config) -> Target` | 2 | adapter, `check_buffers` fixture |
| `carry_pairs` | `(graph) -> list[CarryPair(storage, reader, role)]` | 2 | `resident_read` skip |
| `DirectReadStick` | `(axis, eps)`, set only past the 3-trip gate `WDC:263` | 1, filled by 2 | `DIRECT_READ_SOURCE` |
| `CpSatLayoutSolver.plan_layout_and_core_divisions`, `.last_solve_stats` | `(self, cost_expr=None)` (`ILP:699`); `last_solve_stats["status"]`, `["objective_used"]`, set by the solve that produced the plan (`ILP:1347`) | existing | the on/off test |
| `solved_bindings`, `_evaluate` | `(buffers) -> dict` (`PS:542`), `(expr, bindings) -> float \| None` (`PS:567`) | existing | `objective_ns` in the on/off test |
| `TestBuildingBlocks` | `tests/inductor/test_building_blocks.py:40` | existing | the on/off test's compiles |

Provides, all under `torch_spyre._inductor.scratchpad.lx_allocator`:

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `rules.RuleTag`, `rules.tag_name`, `rules.TABLE_COLUMN` | section 2; a `str` tag is the `__name__` of a check in `ALWAYS` | 5, 6, 7, 8, 11 | final |
| `rules.CONSTRAINT_RULES` | `dict[str, RuleTag]` | 5 | final |
| `rules.SplitRule`, `rules.StickTable` | section 2 | 5, 13 | final |
| `rules.Frame`, `rules.frame_of` | section 2 | 5 | final |
| `rules.splits_while_stuck` | `(node, flag, tiling=None) -> list[SplitRule]` | 5, 11 | final |
| `rules.fact_splits`, `rules.always_splits` | `(node, tiling=None) -> list[SplitRule]` | 5 | final |
| `rules.stick_tables` | `(node, flag) -> list[StickTable]` | 5, 11, 13 | final |
| `rules.stuck_dims`, `rules.is_stuck` | `(port) -> frozenset[int]`, `(rule) -> bool` | 5 | final |
| `rules.residency_reason` | `(node, graph, *, decides_relayouts, tileable) -> str \| None` | 5, 6, 8, 13 | final |
| `rules.untiled_bar` | `(node, graph) -> str \| None` | 5 | final |
| `check.ALWAYS` | `list[Check]`, `Check = (graph, subject) -> Sequence[Violation] \| Skipped \| None`; names in section 2 | 5, 6, 8, 9 | final |
| `check.RULE_CHECKS` | `Mapping[Rule, Check]`, run only for flags in `graph.target.rules` | 5, 6, 8, 9 | final |
| `check.optimizations_used` | `(subject: OpNode \| DataEdge) -> Optimization` | 5, 6, 8 | final |
| `check.check_decisions` | `(graph) -> list[Violation]`; reads every flag and size from `graph.target` | 6, 7, 8, 9, 12, 13 | final |
| `check.check_report` | `(graph) -> CheckReport` | 9, 11 | final |
| `check.coverage`, `check.Coverage`, `check.FamilyCapabilities` | `(family, target) -> Coverage(gaps, unused)`: `gaps` lists restrictions, rules and forced step kinds the family cannot encode, `unused` lists chosen step kinds and optimizations it never uses; a `Protocol` with `ENCODES`, `RULES`, `STEPS`, `OPTIMIZATIONS` | 5, 7, 8, 9, 13 | final |
| `check.RuleViolation`, `.Violation`, `.Skipped`, `.CheckReport` | section 5 | 6-13 | final |
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
| `test_every_str_tag_names_a_check` | Every `str` tag in `CONSTRAINT_RULES` and in rows from `always_splits` is the `__name__` of a check in `ALWAYS`, or `TABLE_COLUMN`. |

`tests/inductor/test_lx_check.py`:

| test | asserts |
|---|---|
| `test_legal_plan_passes[...]` | A hand-decoded legal plan per node type reports nothing, and `evaluated` is non-empty. |
| `test_each_rule_fires[...]` | One parametrized case per `ALWAYS` check, `Rule` flag, `Restriction` flag and `PortFact`: edit one decision, and exactly that rule is reported on that subject. |
| `test_every_rule_flag_has_a_check`, `test_every_optimization_has_a_use` | Every `Rule` member but `NONE` is a key of `RULE_CHECKS`; `optimizations_used` can return every `Optimization` member but `NONE`. |
| `test_derived_target_removes_a_flag` | The parent's `ReductionNode` example: the padded split is reported on a graph built with `SPYRE` and not on one built with `replace(SPYRE, ops=SPYRE_OPS.with_type(ReductionNode, ...))`. |
| `test_rule_flag_off_is_not_checked` | Two split reduction axes are reported under `SPYRE` and not under `replace(SPYRE, rules=SPYRE.rules & ~Rule.ONE_REDUCTION_SPLIT)`, where `skipped` counts the flag. |
| `test_optimization_off_rejects_use[...]` | One case per `Optimization` member: a plan using it passes under `SPYRE` and is reported, tagged with the member, under a target without it. |
| `test_target_never_drops_always` | A target with every config at `Restriction.NONE`, `rules=Rule.NONE` and every optimization on still reports `boxes_fit_lx` and `core_budget`. |
| `test_node_config_reported` | A node whose `restrictions` differ from its type's config in `graph.target.ops` is reported by `node_config`. |
| `test_check_reads_sizes_from_target` | The same plan passes under `SPYRE` and reports `boxes_fit_lx` and `core_budget` under a target with smaller `lx_bytes` and `cores`. |
| `test_coverage_names_missing_members` | A stub family lacking one `Restriction` that only one node type's config sets, one `Rule` and the forced `StepKind.RESTICKIFY` gets one line each in `gaps`; lacking `StepKind.RELAYOUT` and one `Optimization` puts one line each in `unused` and none in `gaps`; a stub with everything gets neither. |
| `test_rule_error_is_contained` | A raising rule lands in `errors`, and other rules still run. |
| `test_shorter_box_never_invents_overlap` | Tiled, in-place and relayout span cases. |
| `test_in_place_is_physical` | A smaller child passes; an in-place overwrite on an edge with a `COPY_OUT` step is reported. |
| `test_tile_aligned_is_physical` | `TileReads.aligned` decides: an aligned nest passes with no match pair on a spilled buffer, and a misread is reported. |
| `test_decisions_fields` | Alignment units, split names, `reduction`, `tile_splits`, `spill_reason`. |
| `test_relayout_before_restickify_is_legal` | `cross_frame` passes with the `RELAYOUT` step first in `edge.steps`, fails after. |
| `test_rule_violation_lists_violations` | |

`tests/inductor/test_lx_plan_adapter.py`:

| test | asserts |
|---|---|
| `test_in_place_inferred_from_address` | True on an in-place overwrite, false on an edge with a `COPY_OUT` step and equal addresses. |
| `test_restickify_op_decodes_on_its_node` | Division, residency and offset land on the node; the step gets none. |
| `test_prefetch_view_from_first_reader` | Joint path, `b.lx_view` unset. |
| `test_relayout_shared_by_group` | One instance on two edges, one box, all byte fields set, `reduction_split` on a completed route. |
| `test_spill_reason_copied` | `b.spill_reason` lands on the node, and `decisions` reports it. |
| `test_copy_out_shared_per_producer` | |
| `test_cuts_from_tiling_groups` | `graph.cuts` holds the tick of the last op before each tiling-group boundary and nothing else. |
| `test_adapter_target_uses_allocator_capacity` | The graph's `target.lx_bytes` is `allocator.size`; the rest is `select_target(config)`. |
| `test_mismatch_reports_lifetime_and_size` | |
| `test_check_buffers_on_synthetic_graphs[...]` | CP-SAT and SA on `synthetic_cooptimization_graphs.synthetic_graphs()`, no violations. |
| `test_check_buffers_on_captures[...]` | Same on `cooptimization_capture_loader.load_captures()`. These buffer-level graphs and the synthetic ones take the ladder, so both tests judge legality only, never an optimum. |
| `test_check_buffers_catches_edits[...]` | An overlapped address, a resident barred buffer, a pair outside `cd_parent_matches`. |

`tests/inductor/test_lx_check_hook.py`:

| test | asserts |
|---|---|
| `test_optimum_unchanged_with_check_on_and_off[...]` | Card. The five `test_building_blocks` compiles Stage 2's e2e test runs (`test_mlp`, `test__simple_attn`, `test_rms_norm`, `test_flash_attention`, `test_sdpa_lk_uses_for_each_tile`), each at the default config, once with `lx_check_decisions` on and then once off, in one process with the FX graph cache off and dynamo reset between runs. A wrapper on `CpSatLayoutSolver.plan_layout_and_core_divisions` records `last_solve_stats["status"]` and `objective_ns = _evaluate(cost_expr, solved_bindings(result))`. Both runs are `OPTIMAL` with `objective_used`, at equal `objective_ns` (1e-9 relative), with one solve per plan and no `SolveError` fallback. A different plan at the same cost passes. |
| `test_check_failure_does_not_reach_the_compile[...]` | `check_todays_plan` raises `Unsupported`, `SolveError`, `AssertionError`: `ScratchpadAllocator.plan_allocation` finishes unchanged and greedy is not entered. |
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
`test_domains_match_collect` proves rule data equals today's, `test_every_flag_has_data`,
`test_every_rule_flag_has_a_check` and `test_every_optimization_has_a_use` hold "a new flag
needs three things", `test_rule_flag_off_is_not_checked` and
`test_optimization_off_rejects_use` prove the card switches, `test_coverage_names_missing_members`
proves coverage, the hook tests prove "reports but never fails a compile", and the exit gate
below proves plans pass.

## Parity and gates

**No plan change.** The parent's per-stage harness gate is dropped with Stage 0
(Deviation 13). The claim it guarded here is the Goal's: the check cannot change a plan, a
model or the outcome of a compile. After the owner's review of PR #7, a plan claim means every
solve reaches `OPTIMAL` at the same `objective_ns` (`_evaluate(expr, solved_bindings(buffers))`
on the returned plan), and a different plan at an equal cost is a tie. It is met three ways:

- **On the production path,** by `test_optimum_unchanged_with_check_on_and_off` on the card:
  the default config (`layout_solver="cpsat"`, co-optimizing on, `auto_coarse_tiling` off)
  solves one cost-expression model, and `lx_check_decisions` is the switch, so one test runs
  both sides on the same input. The check-on compile runs first, so state the check leaves
  behind reaches the check-off compile's solve.
- **On the ladder paths** (placement-only, auto tiling, the greedy fallback), by position:
  the hook runs after `_solve` returns, so the model is built and solved before the check
  starts. `test_check_failure_does_not_reach_the_compile`,
  `test_solve_failure_checks_only_the_fallback` and `test_disabled_runs_nothing` hold the
  rest. No ladder objective is compared.
- **For the rule moves** in commits 3 to 5, which have no config switch and leave no old path
  to run beside the new one, by `tests/inductor/test_work_division.py` unchanged as the
  behaviour pin, `test_domains_match_collect`, `test_indirect_access_gather`, and the card
  suites under Tests, run with the check on at the PR tip.

Each commit runs the parent's unit suites. Nothing is stored between runs: no baseline, plan
or objective. The timing figure from section 7 is attached to the PR.

**Phase 1 exit (parent):** "across the unit and device suites, `check_decisions` reports
nothing, or every report traces to a real bug that gets an issue."

How it is met. Run the suites on the card with `SPYRE_LX_CHECK_FILE` set, then run
`tests/scripts/lx_check_summary.py`. It groups by rule, op kind and solver, and fails on a
violation, error or mismatch not listed in `lx_check_known.json`. A port defect is fixed. A
real bug gets an issue and a line in the known file. `evaluated` must be non-zero for every
rule that applies to the suites' plans, so silence means checked, and `skipped` is reviewed
in the same pass.

The Stage 3 PR does not wait for this gate: it merges on the no-plan-change tests and the
unit suites. The exit gate closes Phase 1 and must pass before Stage 4 merges. Stages 6 and 8 rerun the
summary, since they add encodings of these rules and a new family.

**What I expect it to report.** None of this has been run. The snapshot has no built
extension and Stage 2 does not exist yet, so these are reasoned expectations, with no counts.

*Silent, because the same compile already judges them.* `MATMUL_STICKS` and
`POINTWISE_STICKS` (`_check_stick_constraints`), `resident_read` (`commit_lx_views` raises),
`core_budget` (an assert on the joint path), and on CP-SAT `boxes_fit_lx` and in-place
structure. A report in this group is almost certainly a port defect.

*Port defects to fix in place, most of what the first run will show:*

- `POINTWISE_STICKS` and `WHOLE_STICKS` on fused dims, quantized layouts and sparse sticks,
  where `axis_map` and the stick coordinate disagree.
- Ops that are `Pointwise` by class but not by stick rule: `layernormnorm`, `clone` acting
  as a restickify (`PL:2347`), dtype ops, depending on how Stage 2 classifies them.
- Multi-output ops and unsized buffers, as `Mismatch`. The feasibility report saw 18 read-set
  and 86 size differences of this kind on the zoo.
- `_input_layout_overrides`, where `read_layout()` must give the override.
- `split_divides`, `Rule.TILES_OFF_STICK` and span rules on tiled frames, the first time real
  `predict_frame` output meets these rules.

*Could be real.* The paths with no `is_legal` behind them: pinned divisions, the pruned
exhaustive menu, the placement path, the greedy fallback, and SA's generated divisions. Watch
`split_divides`, `span_floor` and `Rule.ONE_REDUCTION_SPLIT`. On the CP-SAT default path I expect
no real report, because every clause is either a menu filter or a model constraint.

*Will not be exercised much.* `pinned_steps_stay` and `Rule.COPY_OUT_IN_HBM` are vacuous
here. The tiling rules run only in `test_solver_auto_coarse_tiling` and with
`AUTO_COARSE_TILING=1`. Loop-carry pairs are skipped and judged at commit. Restickifies get
real input only where a suite compile makes one, so `cross_frame` and
`restickify_sticks_differ` are judged only there, and the summary's `evaluated` count for each
shows whether any compile did. So "reports nothing" at this gate is weaker for these rules
than for the rest.

From #5682's regression guards, this stage covers plan equivalence with
`test_optimum_unchanged_with_check_on_and_off`, and the solve-status guard through the
record's allocator and solver fields.

## Risks

| risk | caught by |
|---|---|
| A rule move changes behaviour in a corner. | `test_work_division.py`, `test_indirect_access_gather.py`, `test_domains_match_collect`, and the card suites. No old path is left to compare against in process. |
| `frame_of` diverges from `work_division_context_for_op`, so rule data drifts from the menu. | `test_frame_matches_context`, `test_domains_match_collect`. |
| Tests patching `WDC` names stop intercepting after a move. | Those tests fail in commit 3 to 5; old names stay in the call path, so the wrapper must call through its module name. |
| Import cycle: `WDC` imports `rules.py`, which needs `WD`. | `rules.py` keeps IR imports local; import test runs in every commit. |
| The hook costs real compile time on large graphs. `build_graph` is sympy-heavy, and views are per edge. | `seconds` per record on a live run; over noise, the default is off. |
| The check mutates shared state through a memo or `prep_cache`. | Each check uses its own `prep_cache`. `test_optimum_unchanged_with_check_on_and_off` runs check-on first, so a leak that moves a later solve shows as a changed `objective_ns`. A leak that changes only a commit does not show there. |
| A building-block compile does not reach `OPTIMAL` within the solver's time limit, so the on/off test has no optimum to compare. | The test fails rather than comparing a `FEASIBLE` objective. The case is replaced by another existing compile that proves its optimum. |
| `build_graph` needs `V.graph`, set during the pass but not in a test double. | `check_todays_plan` raises, is logged, is contained. Reported as a missing record, which the summary counts. |
| A broad `except` hides that the check never runs. | `evaluated` counters; summary fails if a suite has plans but no records. |
| The solver's lifetime exceeds the bound on purpose, so a real overlap of solver spans goes unseen. | Accepted. The bound is what correctness needs: a longer span "can only cost space" (`ILP:1225`). |
| #5695 lands first. | The adapter reads `chosen_division` for joint buffers and ownership only on the placement path, which #5695 leaves. A rebase reruns the suites; there are no baselines. |
| #5682 moves `span_reduction` into the solver. | `span_floor` reads `node.span_floors`; only the builder's source changes. |
| An op tiled before the solve (a `spyre_hint` or `for_each_tile` op) decodes with a non-empty `tiling`, so `COARSE_TILING` reads as used while `auto_coarse_tiling` is off. | `optimizations_used` skips a node with a `tiling` hazard pin, which Stage 2 sets on every op with `loop_info`; the exit-gate summary groups reports by tag, so a miss shows as `COARSE_TILING` on one op kind. |
| The adapter judges a different card than the allocator planned for. | The adapter's target is `select_target(config)` with `lx_bytes=allocator.size`; `test_adapter_target_uses_allocator_capacity`. |

## Deviations from the parent plan

1. **Six flags cannot hold "each other op-specific constraint function".** The plan names
   `plain_reduction_k_split_domains` and `reduction_window_blocked_vars`, but its `Restriction`
   has no such members, and the code has five such families (`WDC:368`, `WDC:400`, `WDC:801`,
   `WDC:871-992`, `WDC:1008`). `_check_stick_all_same` also covers dtype, most reductions and
   gathers (`OSV:668-672`), not only `PointwiseNode`. Resolution: Stage 1 adds the five
   members to `Restriction` and sets them in `SPYRE_OPS`'s `ReductionNode` config, and sets
   `POINTWISE_STICKS` in its `ReductionNode` and `IndirectNode` configs. The flags live in
   the per-card configs, not on the node types, so another card's `OpFactory` can leave any
   of them out. The stick
   rules of `_keep_by_index_layouts`, `_topk_layouts`, `_conv_layouts`, `_exx2_layout` and
   `_layernormnorm_layout` map to no flag, so those ops stay pinned.
2. **Existing rules that map to nothing in the parent.** Resolution: the added `ALWAYS`
   checks in section 2, for `WDC:496`, `WDC:299`, `WDC:661`, `WD:954`, `WD:930`,
   `constrain_residency` (`ILP:534`), `_constrain_relayout_copies` (`ILP:1579`), the
   fixed-division pins (`ALLOC:3739-3770`) and the cross-frame hazard, plus `node_config`,
   which repeats Stage 2's one-config-per-type assertion. `constrain_residency` is the most
   important wrong-code rule in the allocator and is missing from the parent's always-on
   list. `_input_residency_reason` is folded into `residency`.
3. **"The split divides the axis" is false for a stick axis.** `factor_domain` uses only the
   stick count (`WD:892-893`), so 130 elements on 3 sticks splits 3 ways. A literal port
   would reject legal plans. Resolution: `split_divides` uses `Frame.basis`.
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
8. **`in_place_conditions` and `tiles_line_up` are physical rules.** The parent's In-place
   overwrite row, "same layout and footprint", is what CP-SAT encodes. SA's packer allows a
   smaller child (`torch_spyre/csrc/perm_layout_native.cpp:547`). Resolution: table D's
   physical form for every family; OR-Tools stays stricter by encoding.
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
12. **The partial-reduction cut guard is a `Rule` flag, not an always-on rule.** The parent's
    Stage 6 lists "the partial-reduction cut guard" as always-on, and its `Rule` list has
    `RESIDENT_TILE_UNREDUCED`. Both are `_add_partial_reduction_cut_guard` (`ILP:1030`), and
    its reason, a split reduction combined core to core, describes this card (section 2).
    Resolution: one rule, `Rule.RESIDENT_TILE_UNREDUCED`, checked when the target sets it;
    Stage 6 encodes it under the same condition.
13. **Stage 0 is dropped (owner, PR #7, 2026-11-12).** This stage's parity gate ran Stage 0's
    tiers, base against this PR, on proto, decisions and committed plan. None is built. The
    claim is now the proven optimum, tested in process by
    `test_optimum_unchanged_with_check_on_and_off` on existing `test_building_blocks` compiles,
    with `lx_check_decisions` as the switch. The ladder paths and the rule moves rest on the
    hook's position, the hook tests and the existing suites. The timing figure for the default
    comes from a card run of the suites, not the zoo.

## Open questions

None.

## Out of scope

- Encoding any of this: Stages 5, 6 and 8. No solver reads the new data here.
- The builder, its fact recording and its config mask, `select_target` and the build-time
  one-config-per-type assertion: Stage 2.
- The flag classes (`Restriction`, `PortFact`, `Rule`, `Optimization`), `Target`, `OpFactory`,
  `OpConfig` and the `SPYRE_OPS` configs: Stage 1.
- Running `coverage` over every (family, target) pair: Stages 7 and 8, once a family exists.
- `read_back`: Stages 7 and 8. `RuleViolation` being raised: Stage 9.
- Conditional port facts: Stage 11. Offset-free stick, outer-dim and EA rules, and stick rules
  for conv, topk, keep_by_index, exx2 and layernormnorm: those stay pinned.
- Encodings of the cross-frame hazard and of `Rule.RESIDENT_TILE_UNREDUCED` (the
  partial-reduction cut guard): Stage 6.
- Loop-carry views: skipped here, and `get_ncores_for_buffers` judges them at commit.
