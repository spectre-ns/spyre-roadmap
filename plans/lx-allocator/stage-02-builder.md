# LX allocator, Stage 2: Builder

Parent plan: `plans/lx-allocator.md` at `5aff4a6`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0, 1. Needed by: Stages 3, 4, 5, 6, 7, 8, 9, 10, 11, 13.

## Goal

When this PR merges, `build_graph(ir)` turns a lowered graph into a complete
`SpyreOptimizationGraph` with no solver and no allocator instance: one node per leaf
op, named ports with axis maps, data and execution edges, port facts and pins. Nothing
in production calls it yet. The one production edit is a set of extractions inside
`allocator.py` with no behaviour change, so the CP model, decisions and committed plans
stay byte-identical. Faithfulness is proven by comparing the graph with the allocator's
records, which become the test oracle rather than the builder's input.

## Today

**The record list.** The allocator's records are `LifetimeBoundBuffer`, `CoreDivisionBuffer`
and `RelayoutCopyBuffer` (`torch_spyre/_inductor/scratchpad/plan_solver.py:60,334,751`). The
joint path builds them in `_prepare_buffers`
(`torch_spyre/_inductor/scratchpad/allocator.py:3128`), through `_division_map` and
`_build_cd_bound_buffers` (`torch_spyre/_inductor/scratchpad/allocator.py:4130`). The
placement path uses `_generate_buffers` and `_build_bound_buffers`
(`torch_spyre/_inductor/scratchpad/allocator.py:1797,1542`).

The "solver's filtered view" in the plan is `SimulatedAnnealingLayoutSolver.__init__`
rebinding `self.buffers` to its placeable subset
(`torch_spyre/_inductor/scratchpad/simulated_annealing.py:155-157`). The rule is really
"never lose a barred op".

The records cannot be the builder's input, for four reasons:

1. **They are not a superset of the graph.** Joint records cover every op
   (`torch_spyre/_inductor/scratchpad/utils.py:242`) but only the graph inputs that pass
   `_eligible_clone_inputs` (`torch_spyre/_inductor/scratchpad/allocator.py:4228-4238`).
   `parents` still names the others
   (`torch_spyre/_inductor/scratchpad/allocator.py:4070-4073`). Placement records have no
   `parents` at all.
2. **They hold names, not reads.** No index expression, no layout, no dep.
3. **Building them runs the enumeration.** `_division_map` calls
   `enumerate_work_division_candidates`, which Stage 5 owns inside the encoder.
4. **They carry path-specific ticks.** `_append_lx_relayout_destinations` rewrites `uses`
   to `2*use+1` (`torch_spyre/_inductor/scratchpad/allocator.py:1905-1910`), and
   `RelayoutCopyBuffer`s are candidates, not ops.

The graph can be built without the allocator. PR #5676's `build_graph_from_ir` did this
over the zoo (feasibility report, "3c graph from the IR"): 64 solves, 762/762
`produced_at`, 744/762 read sets, the 18 misses being `MultiOutput` plumbing. What the
builder needs is module-level: `calculate_liveness`, `counted_loop_lifetime_overrides`
(`torch_spyre/_inductor/scratchpad/utils.py:77,143`), `validated_drain_plans`
(`torch_spyre/_inductor/scratchpad/allocator.py:420`).

**Loops.** `splice_while_loops` ends by calling `collapse`
(`torch_spyre/_inductor/wsr/for_each_tile_lowering.py:3525`), so `graph.operations` holds
`LoopNode`s from then on. Each pre-scheduling pass runs against `OpsView`, a flat view of
the leaf ops (`torch_spyre/_inductor/passes.py:447-476`,
`torch_spyre/_inductor/loop_node.py:193`), and `expand` flattens them at the end
(`torch_spyre/_inductor/passes.py:622`). So the allocator never sees a `LoopNode`, and
every tick is a leaf position. Four things follow.

- `LoopNode` is not "all loops". Only `for_each_tile` loops are collapsed. Span-overflow
  groups stamped later (`torch_spyre/_inductor/passes.py:374`) and solver tilings have
  `loop_info` but no node.
- A `LoopNode` body is a span, not membership: it can hold ops that run once, outside
  the loop (`torch_spyre/_inductor/loop_node.py:382-387`,
  `torch_spyre/_inductor/scratchpad/coarse_tiling.py:355-359`).
- `loop_facts.py` is the stable, write-free read API over `loop_info`, and
  `counted_loop_path` (`torch_spyre/_inductor/loop_facts.py:101`) is its one definition.
- "Not modelled" cannot mean "ignored". Today's model consumes lifetime overrides
  (`torch_spyre/_inductor/scratchpad/utils.py:143`), carry edges with match pairs only
  (`torch_spyre/_inductor/scratchpad/allocator.py:4300-4375`), the no-re-tile rule
  (`torch_spyre/_inductor/scratchpad/allocator.py:4004-4007`, which applies to any op with
  `loop_info`, not only user loops) and advancing reads
  (`torch_spyre/_inductor/loop_facts.py:269,297`).

**Op kinds at planning time.** Clones (an input clone is an LX prefetch), drains and LX relayout copies do not exist yet
when `_prepare_buffers` runs: `_push_allocation` inserts them
(`torch_spyre/_inductor/scratchpad/allocator.py:2036`, `materialize_lx_relayouts` at
`torch_spyre/_inductor/scratchpad/allocator.py:2158`). What the allocator sees is every
`ComputedBuffer`, every extern kernel (`torch_spyre/_inductor/propagate_layouts.py:3485-3536`
lists the branches) and any `NopKernel`.

**Surprises, each checked against the snapshot or by experiment.**

- `Scatter` subclasses `Pointwise`, so `isinstance(op.data, Pointwise)` is true for a
  scatter.
- `ReadWrites.reads` deduplicates. `x * x` gives one `MemoryDep`
  (`torch_spyre/_inductor/propagate_layouts.py:1521-1524` restores the pair for matmul);
  `x[..., :8] + x[..., 8:]` gives two with different indices.
- A gather's read order carries no meaning. `MemoryDep.is_indirect()` is true for the
  table, not the index, despite the comment at `torch_spyre/_inductor/pass_utils.py:568`.
- `host_coordinates(..., None)` silently drops the indirect symbol (`[0, d1]` for a
  table read); with `indirect_sizes_from_op(op)` it keeps it (`[tmp0, d1]`).
- A mutation target is not in `reads` unless the body also reads it
  (`torch_spyre/_inductor/propagate_layouts.py:3130-3132`). Its write dep is named after
  the mutating op (`torch_spyre/_inductor/scratchpad/allocator.py:4518`).
- A restickify is an ordinary op with its own buffer, menu, tick, residency variable,
  boxes and cost features, and a `div_<name>` variable when its menu has more than one entry
  (`torch_spyre/_inductor/scratchpad/ilp_solver_ortools.py:454-479`). The solvers never test
  for it. Codegen validates it as a separate op
  (`torch_spyre/_inductor/op_spec_validation.py:660`).
- `is_output_stick_aligned_for_entry` does not exist. `indirect_forbidden_split_syms`
  (`torch_spyre/_inductor/pass_utils.py:1812-1845`) keys on the index's logical entry count,
  and `tests/inductor/test_indirect_access_gather.py:1184-1240` asserts a partial entry stays
  unsplit. `_pad_output_for_stick_aligned_split`
  (`torch_spyre/_inductor/enforce_indirect_access_layout.py:71-98`) does run: it enlarges
  the output buffer and enables no split. The comments at
  `torch_spyre/_inductor/work_division_constraints.py:1038-1039`,
  `torch_spyre/_inductor/pass_utils.py:1673` and
  `torch_spyre/_inductor/enforce_indirect_access_layout.py:75-79` are stale.
- `WorkDivisionContext.reduction_vars` is not "absent from the write". It is "absent from
  the output's non-stick device coordinates"
  (`torch_spyre/_inductor/work_division.py:1018-1024`). Experiment: `sum` from `[4,32,16]`
  to `[4,32]` gives `reduction_vars=[d1, d2]`, so output stick axis `d1` is pinned to 1 by
  `plain_reduction_k_split_domains`, while `_reduction_syms`
  (`torch_spyre/_inductor/work_division.py:1109`) gives `{d2}`.
- `finalize_layouts` deletes `op.layouts` (`torch_spyre/_inductor/insert_restickify.py:625`),
  so stick candidates are gone by build time.

## Design

### Entry point

```python
"""torch_spyre/_inductor/scratchpad/lx_allocator/graph_builder.py"""
def build_graph(
    ir: GraphLowering,
    *,
    capacity: int | None = None,        # default _lx_planning_size()
    alignment: int = 128,               # MemoryPlanSolver's default, one stick
    divisions_committed: bool = False,  # placement path: pin every running node
    drain_plans: Mapping[str, DrainPlan] | None = None,  # None computes them
    lx_relayout_plans: Sequence[LXRelayoutPlan] = (),
) -> SpyreOptimizationGraph: ...

def classify(op: Operation, ir: GraphLowering) -> type[OpNode]: ...
def matmul_axes(op: ComputedBuffer) -> MatmulAxes: ...
def derive_port_facts(port: Port, layout: FixedTiledLayout | None) -> PortFact: ...
def carry_pairs(graph: SpyreOptimizationGraph) -> list[CarryPair]: ...
```

Preconditions, asserted on entry: `ir` is the active `V.graph`; `not
has_loop_nodes(ir.operations)`, which holds under `OpsView` and fails loudly on a real
nested list, where ticks would be top-level positions; and
`materialized_lx_relayouts(ir)` is empty. Call it after `_distribute_work` and before
`_post_solve`, which can add ops through `apply_chosen_tilings`. The graph is stale after
any IR edit. A solve failure leaves the IR unchanged
(`torch_spyre/_inductor/scratchpad/allocator.py:5495-5500`), so a fallback reuses the same
graph.

`build_graph` raises only on a broken invariant. An `Unsupported` from a coordinate call
is caught per port: the port keeps `coords=None`, `device_coords=None`, `axis_map={}`. A
legal compile must never fail in the builder.

The order of work:

1. `ops = list(all_ops(ir.operations))`, `uses = calculate_liveness(ir)`, the lifetime
   overrides. `drain_plans`, when `None`, is `validated_drain_plans(ir,
   division_is_fixed=divisions_committed)`. Then `_drain_lifetime_end_overrides` as at
   `torch_spyre/_inductor/scratchpad/allocator.py:4162-4166`.
2. Construct the graph with `capacity`, `alignment`, `max_cores=config.sencores`,
   `divisions_committed`, `ir`, `drain_plans` and `tuple(lx_relayout_plans)`. Stage 3's
   `residency_reason` reads these.
3. Leaf nodes: one `InputNode` per `ir.graph_input_names`, including unused and scalar
   inputs, then one per name read by an op that no op writes (constants through
   `try_get_buffer`), in first-read order. A name that resolves to nothing also becomes a
   leaf with `out.layout=None`, and logs once. An `InputNode` has `op=None` and no tick.
4. Op nodes in op order, with `tick` the leaf index. `add_node` links consecutive running
   nodes by execution edges. `writer[name]` comes from each op's `rw.writes`.
5. Ports, axis maps and edges.
6. Port facts, in-place candidates, pins.

### Op kind to node type

`classify` runs these predicates in this order. Order matters: `Scatter` is a
`Pointwise`, and a matmul is a `Reduction`.

| # | predicate | node | codegen counterpart |
|---|---|---|---|
| 1 | `not isinstance(op, ComputedBuffer)` | `OpaqueNode` | no `SpyreKernel` |
| 2 | `_is_cpu_host_buffer(op)` (`torch_spyre/_inductor/scratchpad/allocator.py:2195`) | `OpaqueNode` | host code |
| 3 | `op.data` not `Pointwise`/`Reduction` (`Scan`, `Sort`) | `OpaqueNode` | `iteration_space_from_op` raises (`torch_spyre/_inductor/pass_utils.py:1946`) |
| 4 | `indirect_access_subs_from_op(op)` non-empty, as `_is_indirect_access_op` (`torch_spyre/_inductor/scratchpad/allocator.py:2221`) | `IndirectNode` | `_indirect_syms_used` (`torch_spyre/_inductor/spyre_kernel.py:1308,1339`) |
| 5 | `_is_matmul_op(op)` (`torch_spyre/_inductor/pass_utils.py:3462`) | `MatmulNode` | `MATMUL_REDUCTION_OPS` (`torch_spyre/_inductor/spyre_kernel.py:1999`) |
| 6 | `isinstance(op.data, Reduction)` | `ReductionNode` | `store_reduction` (`torch_spyre/_inductor/spyre_kernel.py:1392`) |
| 7 | `is_restickify_op(op, ir)` (`torch_spyre/_inductor/padding.py:563`), or, when the written buffer has `_input_layout_overrides`, `is_restickify_coords` over `device_coordinates(override.device_layout, in_dep)` and the write's coordinates, because `is_restickify_op` reads `in_buf.get_layout()` (`torch_spyre/_inductor/padding.py:492`) and misses the override on `insert_post_mutation_restickify` step 1 | `RestickifyNode` | `is_restickify_coords` (`torch_spyre/_inductor/spyre_kernel.py:1381`) |
| 8 | otherwise | `PointwiseNode` | `store` (`torch_spyre/_inductor/spyre_kernel.py:1305,1332`) |

How each op that exists today lands:

| IR op | node | notes |
|---|---|---|
| `Pointwise` compute, identity copy, `clone`, reshape, transpose with no stick change | `PointwiseNode` | |
| single-input copy that swaps the stick: `insert_restickify`, `insert_post_mutation_restickify` step 1, the copies of `torch_spyre/_inductor/enforce_indirect_access_layout.py:165` and `torch_spyre/_inductor/nonstick_dim_order.py:1147`, a stick-changing `clone` | `RestickifyNode` | the copy-back has a mutation layout, so `is_restickify_op` is false and it is a `PointwiseNode` |
| gather | `IndirectNode(scatter=False)` | |
| `Scatter` with an indirect write | `IndirectNode(scatter=True)` | |
| `Scatter` with no indirect symbol | `PointwiseNode` or `RestickifyNode` | axes still come from `_scatter_iteration_space` |
| `batchmatmul`, `batchmatmulfp8` | `MatmulNode` | |
| `conv2d`, `depthwiseconv2dnative`, `keepbyindex`, topk, pool, `exx2`, `absmax`, `quantscalepertokenfp8`, `sum`/`mean`/`max`/`min`/`prod` | `ReductionNode` | `reduction_type` and `conv_params` stored |
| any of the above with `MutationLayoutSHOULDREMOVE` | by `op.data` | `out.layout` is the target's via `_resolve_layout`; `mutation_target` is set |
| loop-carry storage or update | by `op.data` | plus a `loop_carry` pin |
| `FallbackKernel`, `MultiOutput`, `DeviceCopy`, `SpyreConstantFallback`, `SpyreEmptyFallback`, the four collectives (`torch_spyre/_inductor/ir.py:565-796`), `DynamicScalar`, `AssertScalar`, unspliced `WhileLoop`, `Conditional`, `InvokeSubgraph`, `NopKernel`, CPU `ComputedBuffer` | `OpaqueNode` | a tick and a record today, so a node here; `opaque` pin |
| graph input, constant | `InputNode` | LX prefetch of a graph input is a decision on the node, not an op |
| `LoopNode` | never | the view hides it; the precondition refuses a nested list |
| LX relayout copy, drain, LX prefetch, output clone | never at build time | decisions; `RelayoutCopyBuffer` records have no node |

`OpaqueNode` runs, has `restrictions = NONE`, takes its committed division and never
resides. `RestickifyNode` has `WHOLE_STICKS` only: it reads one stick and writes another.

### Pre-pass restickifies

The op stays a `RestickifyNode`, which holds the division, residency, box and tick. The
edge into it carries the step the node realizes:

```python
edge = R.port("in0").edge                     # P.out -> R.in0
edge.add_step(Restickify(to_stick=s, pinned=True, realized_by=R))
graph.pin(edge, "restickify", s, "pre-pass")
```

The step has no footprint, variables or box of its own, and `apply` returns
`realized_by.out.layout`. `read_back` keeps pinned steps and adds no second one.

`to_stick` is a host dim of the source: `_host_dim_carrying_sym(in_host_coords,
out_stick_sym)`, the derivation at `torch_spyre/_inductor/padding.py:995`, where
`out_stick_sym` is `_stick_symbol(R.layout.device_layout, write)`
(`torch_spyre/_inductor/padding.py:543`). `SPARSE_STICK` if there is no such dim.

An edge has a `Restickify` step exactly when it has the pin. Before a solve,
`DataEdge.transforms` holds only pinned steps.

### Axes and ports

**Axes.** `iteration_space_from_op(op)` (`torch_spyre/_inductor/pass_utils.py:1923`) gives
the node's axes in dict order, output axes then reduction axes. `OpaqueNode` and
`InputNode` have none.

```python
Axis(symbol=sym, extent=concretize_expr(e), reduction=write.index.coeff(sym) == 0,
     granularity=granularity_of(sym), off_output_coords=sym not in coord_vars,
     symbolic=bool(getattr(e, "free_symbols", None)))
```

- `reduction` is `_reduction_syms`' test (`torch_spyre/_inductor/work_division.py:1117`).
- `off_output_coords` is `WorkDivisionContext.reduction_vars` membership. `coord_vars` is
  computed as at `torch_spyre/_inductor/work_division.py:1018-1023`. It is what
  `_one_reduction_split_at_most`, `coordinate_mask_blocked_vars`,
  `plain_reduction_k_split_domains` and `qfp8wt_matmul_k_split_domains` read.
- `granularity_of` reads `_collect_symbol_metadata(it_space)`
  (`torch_spyre/_inductor/work_division.py:118`). `concretize_expr` gives the shape hint
  under dynamic shapes; `symbolic` says so.

**Binding.** A variadic node is built with `operands=n`, a fixed type with none, and
further ports come from `add_in_port(name, operand=...)`. `operand` is a read's position
among the op's `MemoryDep` reads, in `op_read_writes(op).reads` order, the convention
`tiled_dims_per_read` uses
(`torch_spyre/_inductor/loop_info.py:252`, `torch_spyre/_inductor/loop_facts.py:150`). A
non-`MemoryDep` read has `operand=-1`, `index=None`, `axis_map={}`.

| node | binding |
|---|---|
| Pointwise, Reduction, Restickify, Opaque | `in0..inN`, one per dep in `reads` order, `MemoryDep`s first. `x * x` has one port. Two offsets into one buffer have two ports, two edges from one out port |
| Matmul | filter out a read of the mutation target, then `identify_matmul_inputs` (`torch_spyre/_inductor/pass_utils.py:1268`). `x` is the first, `y` the second. With one dep, the builder connects `x` and sets `y.edge` to the same `DataEdge`, so the edge count matches `parents`. A read of the target binds to a port named `target` |
| Indirect, gather | `table`: `dep.is_indirect()`. `index`: `dep.name` in `indirect_info_from_op(op)[0]`. `values`: anything else, for example a plain operand of an indirect `PointwiseOp` (`torch_spyre/_inductor/spyre_kernel.py:1308`). Decide by predicate, never by position |
| Indirect, scatter | `index` as above. `table`: a read of the mutation target if there is one, else a synthetic port with `dep` set to the write and an edge from `target.out` with `mutation=True`. `values`: the rest |

A further read that fits no named port goes to `table1`, `index1`, `values1` and so on,
and logs once. That covers multi-index gathers, whose existence is unverified.

Every port stores `dep`, `coords` (host) and `device_coords`, and an in port also
`index=dep.index`:

```python
sizes = indirect_sizes_from_op(op)
coords = host_coordinates(read_layout, dep, sizes, op=op)
device = device_coordinates(read_layout.device_layout, dep, None)            # as TensorDep
device = [c.xreplace(subs) for c in device_coordinates(stl, dep, sizes)]     # table ports
```

Table ports follow `_shared_indirect_coords`
(`torch_spyre/_inductor/pass_utils.py:1754-1758`). An `OpaqueNode` port gets none of this:
a CPU op's reads have no `FixedTiledLayout`, and `_fixed_read_layout` would raise. `op=op`
applies `per_trip_index`, so a splice trip counter such as `u0` is not an axis.
`read_layout` here is `input_layout_for_operation(op, dep.name,
_fixed_read_layout(ir.get_buffer(dep.name)))`, the pair work division and codegen use
(`torch_spyre/_inductor/work_division.py:2863`). `InPort.read_layout` is set only when an
`_input_layout_overrides` entry applies. The out port's host coordinates use
`_effective_output_layout(op)` (`torch_spyre/_inductor/pass_utils.py:652`), because a sliced
mutation target has mismatched `size` and `stride` on the mutation layout.

**Axis map.** `{axis: _host_dim_carrying_sym(coords, axis.symbol) for axis in axes if
axis.symbol in coord symbols}` (`torch_spyre/_inductor/padding.py:498`). Dims are host dims
of the read layout. That helper's convention is the outermost dim, the "governing" one.
One axis can reach several dims, so `Port.dims_of(axis)` returns all of them from `coords`.
Checked by experiment on real IR:

| read | coords | `axis_map` |
|---|---|---|
| same index | `[d0, d1]` | `{d0: 0, d1: 1}` |
| transposed | `[d1, d0]` | `{d1: 0, d0: 1}` |
| broadcast from `[1, 8]` | `[0, d1]` | `{d1: 1}`; dim 0 is a broadcast dim |
| flatten, `[4, 8]` read as `[32]` | `[floor(d0/8), Mod(d0, 8)]` | `{d0: 0}`; `dims_of` is `(0, 1)` |
| `unfold` | `[2*d0 + d1]` | `{d0: 0, d1: 0}` |
| gather table | `[tmp0, d1]` | `{d1: 1}`; `tmp0` is not an axis |

**Matmul axes.** `matmul_axes` is a pure port of `_matmul_layouts`
(`torch_spyre/_inductor/propagate_layouts.py:1535-1581`), with no layout work. It takes the
op and returns symbols, and the builder maps them to axes:

```python
@dataclass(frozen=True)
class MatmulAxes:
    k: sympy.Symbol | None
    n: sympy.Symbol | None    # None when N is 1: the symbol has simplified away

k = find_reduction_var((x_dep,), write)
if get_matmul_n_size(op) == 1:   n = None
elif get_matmul_m_size(op) == 1: n = write.var_names[-1]
else:                            n = find_matmul_generated_var(y_dep, x_dep, write, op)
```

`Unsupported` leaves `k` or `n` as `None`, and the builder logs. `_matmul_layouts` has
already passed for this op, so this needs an IR edit since then to fire.

**Out port.** `out.layout` is `_resolve_layout(op)` for a device op, the input's
`FixedTiledLayout` for an `InputNode`, and `None` otherwise (`MultiOutputLayout`,
`NoneLayout`, a plain `FixedLayout`). `out.index` is the write's index. It stays `None` with
no single `MemoryDep` write (`torch_spyre/_inductor/scratchpad/utils.py:737-739`) or no
`origin_node.target` (`torch_spyre/_inductor/scratchpad/allocator.py:1427-1429`), so
`same_index_read` refuses what `op_inputs_good_for_lx_inplace` refuses. `OpNode.size` is
`get_device_size_in_bytes(layout.device_layout)` under the conditions at
`torch_spyre/_inductor/scratchpad/utils.py:256-260`, else the `-1` sentinel. It does not
call `mem_usage_by_buf`, which runs `get_ncores_for_buffers`.

**Rule inputs on the node.** Each is read once:

| field | derivation |
|---|---|
| `reduction_type` | `op.data.reduction_type`, on `ReductionNode` and `MatmulNode` |
| `conv_params` | `op.data.op_info.get("conv_params")`, as `torch_spyre/_inductor/work_division_constraints.py:377-382` |
| `span_floors` | `_span_min_splits(op)` (`torch_spyre/_inductor/work_division.py:201`), keyed by `Axis` |
| `tile_local_axes` | the `allowed_splits` keys of `coarse_tile_local_dim_split_domains` (`torch_spyre/_inductor/work_division_constraints.py:496`), called with a context holding only `op` and `it_space`, the two fields it reads |
| `graph_output` | name in `ir.get_output_names()` |
| `mutation_target` | the node of `op.layout.get_buffer()`, for a `MutationLayoutSHOULDREMOVE` op |

### Port facts

`derive_port_facts(port, layout)` is pure: it reads the port, its node's `op` and the
given layout, so Stage 11 can ask it about a candidate layout. `build_graph` calls it once
per port with the port's own layout.

| fact | port | derivation | old reader |
|---|---|---|---|
| `QFP8WT` | out; in unless the dep `is_indirect()` | `layout.device_layout.element_arrangement == ElementArrangement.QFP8WT` | `has_qfp8wt_tensor` (`torch_spyre/_inductor/work_division_constraints.py:698`) |
| `STAGGERED_EA` | same | `element_arrangement in STAGGERED_EAS` (`torch_spyre/_inductor/constants.py:66`) | `has_staggered_ea_tensor` (`torch_spyre/_inductor/work_division_constraints.py:703`) |
| `OVERLAPPING_READ` | in, non-indirect only, not for `conv2d`, `depthwiseconv2dnative`, `POOL_OPS` | `overlapping_coordinate_extents(c, coord_ranges)` is non-empty for any `c` in `device_coords[:-1]` | `overlapping_window_split_domains` (`torch_spyre/_inductor/work_division_constraints.py:1112`) |
| `DIRECT_READ_SOURCE` | the in port reading `record.copy_name`, else `in0` | every gate of `direct_read_source_stick_split_domains` passes | `torch_spyre/_inductor/work_division_constraints.py:237` |

- The in-port layout is the real read, so a `RestickifyNode`'s facts come from its source,
  not the step's result. Indirect table ports get no layout facts because
  `get_mem_deps_from_rw` (`torch_spyre/_inductor/pass_utils.py:565`) keeps them out of
  `input_tds`, and the rules never see them.
- `coord_ranges` follows `torch_spyre/_inductor/work_division_constraints.py:1143-1185`,
  without the raises. A coordinate the rule would refuse sets the fact anyway, so the rule
  still runs and raises in the same place.
- `DIRECT_READ_SOURCE` requires: `op._read_copy_elision_record` set, `op.data` Pointwise,
  loop trips at least `_MIN_SOURCE_AWARE_DIRECT_READ_TRIPS`, `_direct_read_source_dep(op)`
  not `None`, and an axis whose stick coordinate is `Mod(symbol, eps)`. The flag alone
  cannot serve the rule, so the port also stores `InPort.direct_read:
  DirectReadStick(axis, eps)`.

### In-place candidates

`DataEdge.in_place_candidate` holds the division-invariant gates, computed once. It is
`inplace_edge_ok(..., division_invariant=True)`
(`torch_spyre/_inductor/scratchpad/allocator.py:1678`) over the same inputs as
`_determine_in_place_division_invariant`
(`torch_spyre/_inductor/scratchpad/allocator.py:4032`), with final drain-extended bounds as
at `torch_spyre/_inductor/scratchpad/allocator.py:4279-4286`.

An edge from an `InputNode` is the reverse-parent case at
`torch_spyre/_inductor/scratchpad/allocator.py:4427-4450`: the same gate, for the reader at
the input's last use, with `clone_at_graph_boundaries()` on. Whether that LX prefetch exists
at all depends on residency and on a menu
(`torch_spyre/_inductor/scratchpad/allocator.py:4228-4238`), so on these edges the flag is
necessary, not sufficient. Stage 6's `EdgeTables.in_place` is the ordered truth: it drops
inputs with no `PrefetchMenu` and adds today's list order.

### Lifetime and carries

The builder fills `uses`, `lifetime_start_override` and `lifetime_end_override`, so
`graph.lifetime` equals `LifetimeBoundBuffer.start_time/end_time`
(`torch_spyre/_inductor/scratchpad/plan_solver.py:161-170`). An unused input has `uses=()`.
`uses` are always leaf-op ticks, never `2u+1`.

```python
@dataclass(frozen=True)
class CarryPair:
    storage: OpNode
    reader: OpNode
    role: Literal["update", "read"]
```

`carry_pairs` lists one `update` pair per op whose `_loop_carry_record.update_name` is its
own name, and one `read` pair per op reading such an update, as
`_loop_carry_read_edges` finds them (`torch_spyre/_inductor/scratchpad/allocator.py:4564`).
It applies no residency filter; Stage 6 and Stage 8 do.

### Pins

The value of a `split` pin holds factors, never a menu index. `WorkSplitPin`,
`DimOrderPin` and `DirectReadStick` are Stage 1 types. A stick value is
`stick_dims(layout) or (SPARSE_STICK,)`, which keeps QFP8WT's two dims. It needs no dep, so
it works on an input.

| plan's source | `what`, target | `value` | `source` | traced to |
|---|---|---|---|---|
| graph-input arrival | `stick`, `node.out` | stick dims | `graph input` | `real_input.device_tensor_layout()` (`torch_spyre/_inductor/propagate_layouts.py:3310-3334`), committed at `torch_spyre/_inductor/insert_restickify.py:616`, read from `ir.graph_inputs[name].data.data.layout` |
| pre-pass restickify | `restickify`, the edge into `R` | `to_stick` | `pre-pass` | `is_restickify_op`; inserted by `torch_spyre/_inductor/insert_restickify.py:350,866`, `torch_spyre/_inductor/enforce_indirect_access_layout.py:165`, `torch_spyre/_inductor/nonstick_dim_order.py:1147` |
| same | `stick`, `R.out` | stick dims | `pre-pass` | so Stage 11 keeps it |
| work-division hint | `split`, node | `WorkSplitPin` | `hint` | `has_resolved_work_div_hint` (`torch_spyre/_inductor/work_division.py:1780`), gated by `config.ignore_work_division_hints`; committed by `work_distribution_pass` |
| hazard: loop | `tiling`, node | `()` | `hazard` | `loop_info is not None`, or in `prescribed_regions` (`torch_spyre/_inductor/scratchpad/coarse_tiling.py:383`), as `torch_spyre/_inductor/scratchpad/allocator.py:4004-4007` |
| hazard: loop carry | `loop_carry`, node | `LoopCarryRecord` | `hazard` | `op._loop_carry_record` (`torch_spyre/_inductor/loop_info.py:37`) |
| hazard: drain | `drain`, storage node | `DrainPlan` | `hazard` | `drain_plans` |
| hazard: opaque op | `opaque`, node | type name | `hazard` | predicates 1 to 3; covers an unspliced `WhileLoop`. Stage 5 gives it the committed division |
| hazard: division | `split`, node | `WorkSplitPin` | `hazard` | `division_pin_reasons`, below |
| every op's stick | `stick`, `node.out` | stick dims | `layout` | an op with `out.layout=None` gets `(SPARSE_STICK,)`; otherwise the committed `op.layout.device_layout`: candidates from `propagate_spyre_tensor_layouts`, chosen at `torch_spyre/_inductor/optimize_restickify.py:856`, committed at `torch_spyre/_inductor/insert_restickify.py:634`, possibly rewritten by later passes |
| dim order | `dim_order`, the `table` port | `DimOrderPin` | `layout` | `_dim_order_is_compliant` (`torch_spyre/_inductor/nonstick_dim_order.py:504`) over `_compute_gather_ia_constraint`'s coordinates (`torch_spyre/_inductor/nonstick_dim_order.py:817`) for gathers, `_try_scatter_ia_constraint` (`torch_spyre/_inductor/nonstick_dim_order.py:679`) for scatters |
| placement path | `split`, every running node with no `split` pin | `WorkSplitPin` | `committed` | only with `divisions_committed=True` |

The builder is the only writer of pins from these sources.

**The committed split.** `_committed_work_split(node)` returns
`work_split_from_division(node, _fixed_core_division(node.op))`
(`torch_spyre/_inductor/scratchpad/allocator.py:2370`), and never writes the IR. An
`OpaqueNode` has no axes and no ownership, because `work_distribution` skips it
(`torch_spyre/_inductor/work_division.py:2908-2916`), so its split is empty. If #5695 has
landed, a deferred op has no ownership yet either, and `_fixed_core_division` would read one
core. The function then takes the split from the computing half of `divide_deferred_op` and
does not commit it. In the snapshot, `work_distribution_pass` computes and commits in one
body (`torch_spyre/_inductor/work_division.py:2166`). Whether #5695's
`_work_distribution_op` exposes that split separately is unverified, since the PR is not in
the snapshot; if it does not, the rebase separates the two.

**The division chain.** `_division_map` pins nine reason strings over eight branches
(`torch_spyre/_inductor/scratchpad/allocator.py:3737-3770`). Stage 5's parity needs all of
them, and the chain needs one definition:

```python
"""allocator.py, module level; _division_map calls it."""
def division_pin_reasons(
    graph: GraphLowering, *, allow_deferred_read_candidates: bool
) -> dict[str, str]: ...
```

It holds the chain unchanged, in the same order, building its group sets once. It stays
in `allocator.py`, so tests that patch `allocator._fused_layout_group_ops` and friends
(`tests/inductor/test_work_division.py:3462,3995,4048`) keep working. The builder passes
`allow_deferred_read_candidates=True`, as `_prepare_buffers` does at
`torch_spyre/_inductor/scratchpad/allocator.py:3143`. A restickify left unpinned that way
gets `OpNode.deferred_direct_read=True`, which stays false in committed mode, where every
running node is pinned. Stage 6's `settle_deferred_reads` owns the pricing loop at
`torch_spyre/_inductor/scratchpad/allocator.py:3152-3183`. Fp8 consumers
(`_fp8_matmul_consumer_seeds`) carry `seeds`, from which Stage 5 runs
`_seed_aligned_division`.

### Converting divisions

`WorkSplit` holds no `TileSpec`, and `graph/` cannot import `plan_solver`, so conversion
lives here, for every stage that meets a `CoreDivision`:

```python
"""torch_spyre/_inductor/scratchpad/lx_allocator/division_convert.py"""
def work_split_from_division(node: OpNode, division: CoreDivision) -> WorkSplit: ...
def division_from_work_split(node: OpNode, split: WorkSplit) -> CoreDivision: ...
def tile_spec_of(node: OpNode, split: WorkSplit) -> TileSpec: ...
```

- Forward: `splits` map symbol to axis through `node.axes`, and `tiling` is
  `division.tile_splits`, outermost first, with `Axis` for the symbol.
- Back: `reduction_syms` is the split's axes with `reduction` set, `tiling` is
  `tile_spec_of(node, split)`, and `tile_splits` is `tiling` with symbols.
- `tile_spec_of` gives one `TileAxis` per level, in order. `count` is the level's count and
  `is_reduction` is `axis.reduction`. `host_dim` is `node.out.axis_map[axis]` for an output
  axis, or the position in `reduction_loop_vars(op)`
  (`torch_spyre/_inductor/wsr/loop_tiling_common.py:436`) for a reduction axis, the two
  frames of `TileAxis.host_dim` (`torch_spyre/_inductor/scratchpad/plan_solver.py:187`). An
  output axis that reaches two dims takes the outermost, which names the same loop
  (`torch_spyre/_inductor/scratchpad/coarse_tiling.py:70`).

### Layering

`graph_builder.py` imports twelve `allocator.py` helpers inside its functions:
`validated_drain_plans`, `_drain_lifetime_end_overrides`, `division_pin_reasons`,
`_fixed_core_division`,
`_is_cpu_host_buffer`, `_is_indirect_access_op`, `_fp8_matmul_consumer_seeds`,
`_lx_planning_size`, `_handoff_child_start`, `_handoff_parent_end`, `inplace_edge_ok` and
`op_inputs_good_for_lx_inplace`. The last two become module functions in commit 1. The
class keeps `_inplace_edge_ok` and `_op_inputs_good_for_lx_inplace` as one-line delegates,
because `tests/inductor/test_scratchpad_use.py:2033` calls one and
`tests/inductor/test_coarse_tiling.py:9727` patches the other. All twelve stay in
`allocator.py`.

Nothing here assumes a solver, so a debug greedy solve is unaffected.

## Changes

| file | change |
|---|---|
| `torch_spyre/_inductor/scratchpad/lx_allocator/graph_builder.py` | new: `build_graph`, `classify`, `matmul_axes`, `derive_port_facts`, `carry_pairs`, `MatmulAxes`, `CarryPair`, private helpers |
| `torch_spyre/_inductor/scratchpad/lx_allocator/division_convert.py` | new: the three conversions |
| `torch_spyre/_inductor/scratchpad/allocator.py` | extract `division_pin_reasons`, `inplace_edge_ok`, `op_inputs_good_for_lx_inplace` |
| `tests/inductor/test_graph_builder.py` | new, device-free fixtures |
| `tests/inductor/test_graph_builder_e2e.py` | new, `fake` compiles |
| `tests/inductor/graph_builder_oracle.py` | new: `assert_graph_matches_records`, `make_observer` |
| `tests/configs/torch_spyre_tests/inductor/test_graph_builder_config.yaml`, `test_graph_builder_e2e_config.yaml` | new shard configs |

Commits, each with tests green:

1. The three `allocator.py` extractions. Harness diff empty.
2. `division_convert.py` with its round-trip test.
3. `classify`, nodes, axes, ports, axis maps, `matmul_axes`, node rule inputs.
4. Edges, lifetimes, in-place candidates, `carry_pairs`.
5. `derive_port_facts`.
6. Pins, including pinned restickify steps and committed mode.
7. The oracle and the e2e test.

## Interfaces

**Consumes**

| symbol | signature | from Stage | used for |
|---|---|---|---|
| `SpyreOptimizationGraph` | `capacity, alignment, max_cores, divisions_committed, ir, drain_plans, lx_relayout_plans`; `add_node`, `node`, `connect(out, in_port)`, `pin(target, what, value, source)`, `pins_on`, `read_ticks`, `lifetime` | 1 | the build |
| `OpNode` | `op, tick, uses, size, graph_output, lifetime_start_override, lifetime_end_override, mutation_target, deferred_direct_read, span_floors, tile_local_axes`, `first_use_is_read`; `operands=`, `layout=`; `port(name)`, `add_in_port(name, *, operand=None) -> InPort` | 1 | nodes |
| node subtypes | `MatmulNode.k`, `.n: Axis \| None`, `.reduction_type`; `ReductionNode.reduction_type`, `.conv_params`; `IndirectNode.scatter`; `PointwiseNode`, `InputNode`, `OpaqueNode`, `RestickifyNode` | 1 | nodes |
| `Port`, `InPort`, `OutPort` | `axis_map, facts, dep, coords, device_coords`, `dims_of(axis)`; `operand, index, edge, read_layout, direct_read`; `layout: FixedTiledLayout \| None, index, edges` | 1 | ports |
| `Axis`, `WorkSplit`, `TileLevel` | `Axis(symbol, extent, reduction=False, granularity=None, off_output_coords=False, symbolic=False)`, `WorkSplit(splits={}, tiling=())`, `TileLevel = tuple[Axis, int]` | 1 | axes, pins, conversion |
| `PortFact` | Stage 1's `Flag` members | 1 | facts |
| `DataEdge`, `ExecutionEdge` | `src, dst, transforms, mutation, in_place_candidate`, `add_step(step)`, `read_layout()`; `before, after` | 1 | edges |
| `Restickify` | `Restickify(to_stick, realized_by=None, target=None, **transform_fields)`, built with `pinned=True`, `realized_by=node` | 1 | pinned steps |
| `Pin`, `WorkSplitPin`, `DimOrderPin`, `DirectReadStick` | `WorkSplitPin(split, reason, seeds=())`, `DimOrderPin(indirect_device_pos, stride_map)`, `DirectReadStick(axis, eps)`; sources `graph input`, `pre-pass`, `layout`, `hazard`, `committed`, `greedy`, `hint` | 1 | pins |
| `stick_dims`, `SPARSE_STICK` | `stick_dims(layout) -> tuple[int, ...]`, `SPARSE_STICK = -1` | 1 | stick pins |
| `lx_graph_fixtures.py` | layout builders | 1 | unit fixtures |
| harness | `record_compile(name, cell, out, *, device="fake", observers=...)`, `Observer.on_solve_begin(allocator, solver, ir, buffers, cost_expr)`, `check_guards`, `lx_gate.sh`, `Cell`, `DEFAULT_CELL`, `TILING_CELL`, `FIXED_GRAPHS`, `corpus_coverage`, `CARD_ONLY` | 0 | oracle, gates |

**Provides**

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `build_graph` | above | 3-8, 9 (wrapped as `SpyreOptimizer.build_graph`), 10, 11, 13 | final |
| `classify`, `_committed_work_split` | `(op, ir) -> type[OpNode]`; `(node) -> WorkSplit`, never writes the IR | 11 | final |
| `matmul_axes`, `MatmulAxes` | `(op) -> MatmulAxes` | 13 | final |
| `derive_port_facts` | `(port, layout) -> PortFact` | 11 | final |
| `carry_pairs`, `CarryPair` | `(graph) -> list[CarryPair]`, `CarryPair(storage, reader, role)` | 3, 6, 8 | final |
| `work_split_from_division` | `(node, division) -> WorkSplit` | 3, 5, 6, 7, 8 | final |
| `division_from_work_split` | `(node, split) -> CoreDivision` | 3, 5, 9, 10 | final |
| `tile_spec_of` | `(node, split) -> TileSpec` | 3, 10 | final |
| `division_pin_reasons` | above | 5 | final |
| `inplace_edge_ok`, `op_inputs_good_for_lx_inplace` | module functions, today's semantics | 6 | final |
| graph fields, declared by Stage 1, filled here | `graph.ir`, `.divisions_committed`, `.max_cores`, `.drain_plans`, `.lx_relayout_plans` | 3, 4, 5, 6, 8, 9, 10 | final |
| node fields, same | `OpNode.op`, `.tick`, `.uses`, `.size`, `.graph_output`, lifetime overrides, `.mutation_target`, `.deferred_direct_read`, `.span_floors`, `.tile_local_axes`; `.reduction_type`; `ReductionNode.conv_params`; `MatmulNode.k`, `.n`; `IndirectNode.scatter` | 3-10 | final |
| port and edge fields, same | `Port.axis_map`, `.facts`, `.dep`, `.coords`, `.device_coords`; `InPort.operand`, `.index`, `.read_layout`, `.direct_read`; `OutPort.layout`, `.index`; `DataEdge.in_place_candidate`, `.mutation` | 3, 5, 6, 8, 11 | final |
| pins | `Pin` with `what` one of `stick`, `restickify`, `split`, `tiling`, `loop_carry`, `drain`, `opaque`, `dim_order` | 3, 5, 6, 8, 10, 11 | final |

## Tests

**`tests/inductor/test_graph_builder.py`, device-free.** Fixtures build real
`ComputedBuffer`s, as `_real_pointwise_core` does
(`tests/inductor/test_coarse_tiling.py:272`), under
`V.set_graph_handler(GraphLowering(fx.symbolic_trace(lambda: None)))`, with layouts from
Stage 1's `lx_graph_fixtures.py`. This pattern was run against the snapshot for pointwise,
reduction and gather ops. Extern kinds use
`MagicMock(spec=...)`. No test imports a solver, which is this stage's "no solver involved".

| test | asserts |
|---|---|
| `test_classify_*`, one per row of the op-kind tables | the node type, and that a `Scatter` is never a `PointwiseNode` when its write is indirect |
| `test_one_node_per_leaf_op` | names equal `[op.get_name()]` plus inputs; a barred op still has a node |
| `test_every_read_has_a_source` | each dep is bound to exactly one port, each port with a resolvable name has an edge |
| `test_self_alias_pointwise_has_one_port`, `test_self_alias_matmul_shares_an_edge` | `x * x` gives `in0`; `x.edge is y.edge`, one graph edge |
| `test_two_offsets_two_ports` | two ports, two edges, one source |
| `test_axis_map_*` | every row of the axis-map table, from real deps; plus a size-1 output dim |
| `test_trip_counter_is_not_an_axis`, `test_indirect_symbol_is_not_an_axis` | `u0`, `tmp0` absent |
| `test_matmul_axes[general, m1, n1]` | `k`, `n`; `n is None` when N is 1 |
| `test_gather_ports_by_predicate`, `test_scatter_table_port`, `test_mutation_scatter_edge_is_flagged` | binding regardless of read order; `mutation=True` |
| `test_two_input_reductions_keep_both_ports` | `conv2d`, `keepbyindex` have `in0`, `in1` |
| `test_node_rule_inputs` | `conv_params`, `span_floors`, `tile_local_axes`, `reduction_type`, `graph.max_cores` against their sources |
| `test_out_index` | the write's index; `None` with no single `MemoryDep` write or no `origin_node.target` |
| `test_in_place_input_edge` | set only with `clone_at_graph_boundaries()` on, at the input's last reader; an input with no LX prefetch keeps the flag and has no record |
| `test_port_fact_*` | each fact on and off, on out and in ports, against the old predicate on the same op |
| `test_derive_port_facts_is_pure` | a candidate layout changes the answer and leaves the port's `facts` alone |
| `test_read_layout_override`, `test_edge_read_layout` | an `_input_layout_overrides` op reads the override; on every port except a `RestickifyNode`'s, `edge.read_layout()` equals the real read layout |
| `test_reduction_flags_differ_on_a_sub_stick_output` | `[4,32,16]` sum: `d1.reduction` is false, `d1.off_output_coords` is true, matching `work_division_context_for_op` |
| `test_lifetime_*` | equals a `LifetimeBoundBuffer` built from the same overrides, including a drain |
| `test_carry_pairs` | update and read pairs, against `_loop_carry_update_edge` and `_loop_carry_read_edges` |
| `test_pin_*` | one per row of the pin table; `test_restickify_is_node_and_step` checks node, step, pin, `realized_by` and `step.edges` together; `test_committed_pins_every_running_node` checks inputs stay unpinned and an `OpaqueNode`'s split is empty |
| `test_stick_pin_matches_stick_symbol` | the pin's dims agree with `_stick_symbol`, including sparse |
| `test_drain_plans_passed_in` | a given mapping is used, not recomputed, and stored on the graph |
| `test_division_round_trip` | `division_from_work_split(work_split_from_division(d)) == d`, tiling level order and a reduction level included; `tile_spec_of` resolves back through `try_resolve_tile_axis_loop_vars` (`torch_spyre/_inductor/scratchpad/coarse_tiling.py:268`) |
| `test_division_pin_reasons_matches_division_map` | same reason per op as the chain, for each of the nine strings |
| `test_refuses_nested_list`, `test_ops_view_sees_leaves` | `LoopNode` precondition; ticks through a real `OpsView` are leaf positions |
| `test_build_is_deterministic`, `test_build_does_not_mutate_ir` | same structure over two builds and hash seeds; no op attribute changes except the `op_read_writes` memo. On the rebase over #5695, a deferred pinned op is added, and its ownership stays unset |

**`tests/inductor/test_graph_builder_e2e.py`, on `fake` compiles.** A fake compile needs a
process of its own, so each model runs through `record_compile(model, cell, out,
device="fake", observers=("graph_builder_oracle:make_observer",))`. The observer's
`on_solve_begin` calls `build_graph(ir, divisions_committed=not joint)` on the live graph,
where `joint` is `isinstance(allocator, CoOptimizingAllocator)`, and
`assert_graph_matches_records(graph, buffers)` on the result, and the compile continues. A
mismatch raises inside the observer, `check_guards` reports it, and the case fails. Models
are `FIXED_GRAPHS`, restickify graphs included, at `DEFAULT_CELL` and
`Cell(co_optimizing=False)`. Every comparison below happens before a solver runs.

| record field | graph side |
|---|---|
| every `name`, except `RELAYOUT_COPY_PREFIX` | `graph.node(name)` exists |
| `uses` | `node.uses`, undoing `2u+1` on the placement path |
| `uses`, less the producing write | `graph.read_ticks(node)` |
| `start_time`, `end_time`, undone the same way | `graph.lifetime(node)` |
| `first_use_is_read` | `node.first_use_is_read` |
| `boundary` (joint path) | `node.graph_output` |
| `size`, `op_position` (joint path) | `node.size`, `node.tick` |
| `parents` (joint path) | multiset of edge sources, excluding `mutation` edges and edges to a drained collective, adding `carry_pairs` storages |
| `in_place_parents` (joint path) | sources of `in_place_candidate` in-edges that have a record |
| `core_divisions`, when length 1 and pinned | `division_from_work_split` of the `split` pin |

**Existing suites.** The parent plan's unit suites, serially, and Stage 1's graph tests.
Device, on the card: `test_work_division`, `test_scratchpad_use`, because commit 1 touches
`_division_map`. `make check-all-configs` after adding the configs, and `pre-commit run
--files <changed>`.

**What proves the claim.** `test_classify_*`, `test_axis_map_*`, `test_port_fact_*` and
`test_pin_*` prove the plan's Stage 2 bullets on fixtures. The e2e oracle proves nothing was
lost on real graphs.

## Parity and gates

Parent plan: "Every stage: the Stage 0 harness, comparing proto, decisions and committed
plans between the stage's parent and the change. Baselines are re-recorded after every
rebase." And: "Through Phase 1 ... the CP model stays byte-identical and the committed
plans stay identical."

| gate | compares | corpus | pass | on a difference |
|---|---|---|---|---|
| harness | parent commit vs this PR: proto, status, objective, decisions, committed plan | `intree`, fixed set | zero differences | a bug in commit 1; fix it |
| builder oracle | `build_graph` vs records, in one process | `FIXED_GRAPHS` per commit, in CI; the `fake` zoo at `DEFAULT_CELL` and `TILING_CELL` at the PR tip | zero mismatches | fix the builder, or list the mismatch by cause in the PR. The 18 `MultiOutput` and 62 `-1` cases from #5676 must both match, by construction |
| suites | above | | pass | fix |

`lx_gate.sh` records both sides, and after a rebase only the last rebase before merge is
recorded again. The PR reports oracle hits per `corpus_coverage` feature, `restickify_op`,
`beam_restickify` and `fixup_copy` among them. A model the `fake` tier cannot reach stays
card-only. `fake` inputs use default layouts, so this proves the builder faithful to the
allocator and says nothing about production plans. Pickled captures blank the live ops, so
the oracle cannot replay them. This builder has no zoo measurement yet; the numbers above
are #5676's.

The stage has no planning-time gate because nothing calls the builder. Its cost is one
`indirect_access_subs_from_op` and two coordinate calls per dep, comparable to
`_division_map`'s chain without the enumeration. It has not been measured.

## Risks

| risk | caught by |
|---|---|
| `read_back` or Stage 5 double-counts a restickify, as node and step | `test_restickify_is_node_and_step`; Stage 5's proto diff, since an extra box or variable shows at once |
| A rule uses `Axis.reduction` where today's code uses `reduction_vars` | the table comes from the old function, so the proto diff cannot see it; only `check_decisions` can. `off_output_coords` and its test exist for this |
| The `coord_ranges` copy drifts from `overlapping_window_split_domains` | `test_port_fact_overlapping*` against the rule's own output |
| `stick_dims` returns `()` on a layout with no unique match, so a real stick reads as sparse | `test_stick_pin_matches_stick_symbol`, on every op in the e2e run |
| A coordinate call raises something other than `Unsupported` on a real op | the e2e run |
| #5695 lands first and pinned ops have no committed split | `_committed_work_split` is the one seam, and it must stay read-only, as must `division_pin_reasons`: the rebase keeps `divide_deferred_op` in `_division_map`, outside the extracted chain. The oracle's `core_divisions` row and the #5695 case of `test_build_does_not_mutate_ir` fail until both hold |
| A #5682 Step 2 item moves a stick pass after the solve, so layouts are not final at build time | `stick` pins would pin stale layouts, and the oracle's `size` row fails first. Rebase work for whichever lands second |
| `build_graph` is called after `_post_solve` | stale graph. The precondition catches materialized relayouts, not chosen tilings, so this rests on the contract |
| `node.op` makes the graph unpicklable, because IR holds FakeTensor refs | graphs are not pickled; live compiles replace replay |

## Deviations from the parent plan

1. **Input.** The plan: "Build from the allocator's full record list". The code: records
   are unusable as input, for the four reasons under Today. Resolution: `build_graph` takes
   `GraphLowering`, as the class diagram does, and records are the test oracle.
2. **Five subtypes, and "one node per op" vs "a pinned step".** Extern kernels and CPU
   buffers run, hold a tick and a record, and fit none of the five. Byte-identical protos
   need a node for a restickify, and a `PointwiseNode` would break `POINTWISE_STICKS`.
   Resolution: `OpaqueNode`, and `RestickifyNode` plus a pinned step with `realized_by`.
3. **Fixed `PORTS`.** `conv2d`, `depthwiseconv2dnative` and `keepbyindex` have two reads
   (`torch_spyre/_inductor/constants.py:261,268`), and a fused pointwise prologue can add
   more (`torch_spyre/_inductor/wsr/loop_tiling_common.py:450-451`). A scatter destination
   is usually not a read (`torch_spyre/_inductor/propagate_layouts.py:3130-3132`), and an
   indirect `PointwiseOp` can have plain operands. Resolution: `in0..inN` on reductions; a
   synthetic `table` port with a `mutation=True` edge, `values` for plain operands and
   numbered overflow ports on `IndirectNode`; a `target` port on `MatmulNode`.
4. **`axis_map: dict[Axis, int]`.** One axis can reach several dims (the flatten row in the
   table above). Resolution: the dict keeps the outermost dim, and `Port.dims_of` gives all
   of them.
5. **"An in port holds no layout of its own".** `_input_layout_overrides`
   (`torch_spyre/_inductor/insert_restickify.py:940`,
   `torch_spyre/_inductor/pass_utils.py:161`) makes one op read a graph input in a layout
   different from its committed one. Resolution: `InPort.read_layout`, set only there.
6. **`is_output_stick_aligned_for_entry`.** It does not exist, and no padding exception
   exists, so nothing records it. Padding still changes the buffer size, which Stage 11's
   preview needs.
7. **"Every op's stick as `propagate_layouts` fixed it".** The beam and later passes change
   it. Resolution: the pin reads the committed `device_layout`, under source `layout`, so
   Stage 11 can lift it apart from `graph input` and `pre-pass`. The target is the out port.
8. **Hazards are wider than the plan's list.** All of `_division_map`'s reasons are pins,
   or Stage 5 cannot reach parity, and "user-written loops" means every op with
   `loop_info`, as `torch_spyre/_inductor/scratchpad/allocator.py:4004` does. Resolution:
   `division_pin_reasons`, and `tiling`, `loop_carry`, `drain`, `opaque` pins.
9. **"Loops: not modelled".** Kept in that no loop object joins the graph, and `LoopNode`
   is never read. Loop facts stay on the graph for parity: lifetime overrides on the node,
   `tiling` and `loop_carry` pins, advancing via `loop_facts` from `node.op`.
10. **"Plain data".** Later stages reach parity by calling today's functions, so `OpNode.op`
    and `graph.ir` hold live IR, and `OpNode` carries the fields under Provides beyond the
    diagram.
11. **`OutPort.layout: FixedTiledLayout`** and "One `FixedTiledLayout` per node". Extern and
    CPU ops have none. Resolution: `None`.
12. **`Axis.reduction`.** `WorkDivisionContext.reduction_vars` differs from it. Resolution:
    `off_output_coords` beside it.
13. **Rules keyed on a fact "over every port that carries it".** Today's QFP8WT rule reads
    only the second input and the output
    (`torch_spyre/_inductor/work_division_constraints.py:739-751`), and table reads are
    outside all of these rules. Resolution: table ports carry no layout facts, and
    `InPort.operand` gives the input position.

## Open questions

None.

## Out of scope

- `Restriction` semantics, `splits_while_stuck`, `stick_tables`, the plan-to-graph adapter,
  and `residency_reason`, which depends on the menu and family capability: Stage 3.
- The objective: Stage 4.
- Menus, `_seed_aligned_division`, `_legal_fixed_division`: Stage 5.
- `settle_deferred_reads`, carry match edges, relayout candidates, the cross-frame hazard,
  in-place encoding and its list order: Stage 6.
- `SpyreOptimizer.build_graph` and any production call: Stages 3 and 9.
- Lifting stick pins, and recovering stick candidates that `finalize_layouts` deletes:
  Stage 11.
