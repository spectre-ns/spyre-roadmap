# LX allocator, Stage 10: Commit

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. Needed by: Stages 11, 12, 13.

## Goal

When this PR merges, `commit(graph)` lowers a solved graph onto the `GraphLowering` directly and
runs the post passes, `optimize` writes the full cost dump, and the temporary Stage 9 bridge is
gone. No code in the optimizer reads `LifetimeBoundBuffer` or `CoreDivisionBuffer`, and
none calls `_post_solve` or `_push_allocation`. On every corpus entry the new commit produces the
same IR as those two methods: the same ops in the same order, the same clones, addresses, views,
divisions, loop metadata and relayout registry, and the same `LxContextSwitchingPass` result. The
default configuration is unchanged. The old allocators and `_push_allocation` stay in service for
it until Stage 13, and the extraction refactors below leave them byte-for-byte equivalent in
behaviour. No layout changes, because the Stage 2 stick pins are still in place.

## Today

### The commit is not one method

`_push_allocation` is the last third of the commit. The IR edits that the plan calls
"commit" run across these steps of `ScratchpadAllocator.plan_allocation`
(`torch_spyre/_inductor/scratchpad/allocator.py:1044-1052`), all in this order:

| # | step | citation | effect |
|---|---|---|---|
| 1 | `_finalize_lx_relayout_allocation` | `allocator.py:3453` (co-opt), `:1094` (base) | Folds fired relayouts into `LXRelayoutPlan`s and asserts them against the plan. No IR edit. |
| 2 | `apply_chosen_tilings` | `allocator.py:3513`, `torch_spyre/_inductor/scratchpad/coarse_tiling.py:772` | Builds every coarse-tile loop and every copy-out. Co-opt only. |
| 3 | `check_priced_footprints` | `allocator.py:3514`, `:2947` | Raises `Unsupported` if an applied tile footprint exceeds what the solve placed. Co-opt only. |
| 4 | `commit_divisions` | `allocator.py:3515`, `:2886` | Writes `iteration_space_ownership` on every op the solve divided, on the tiled graph, including copy ops via `planned_splits` (`coarse_tiling.py:819`). Co-opt only. |
| 5 | `commit_lx_views` | `allocator.py:3516`, `:2999` | Sets `buffer.lx_view` from `get_ncores_for_buffers`, with `plan.source_view` taking precedence for relayout sources. Co-opt only. |
| 6 | `_stage_collective_outputs` | `allocator.py:3522`, `:3525` | Gives a spilled matmul output an address in a free LX gap after the solve. Co-opt only, on by default (`torch_spyre/_inductor/config.py:494`). |
| 7 | `log_solver_decisions` | `allocator.py:3523`, `:3033` | DEBUG log. |
| 8 | `_get_spill_reasons` | `allocator.py:3648`, `:1135` | Builds the reason map. It runs before any clone exists, so only tiling-added ops read as "added by coarse tiling". |
| 9 | `_push_allocation` | `allocator.py:2036` | Clones, addresses, drains, relayouts. |
| 10 | `_log_lx_pinning` | `allocator.py:2022` | DEBUG log, the only consumer of the reason map. |
| 11 | `post_optimization_passes` | `allocator.py:1052` | `LxContextSwitchingPass`. |

The placement path (`ScratchpadAllocator`, `co_optimizing_lx_planning=False`) has a no-op
`_post_solve` (`allocator.py:1126`), so it runs only 1, 8, 9, 10 and 11. Its views were set
before the solve (`allocator.py:1817`, `:1826`, `:1593`).

### `_push_allocation`, line by line

All lines are in `torch_spyre/_inductor/scratchpad/allocator.py`.

| lines | what happens | easy to miss |
|---|---|---|
| 2056-2057 | Reads `outputs` and `inputs` from the graph. | This is after tiling, and `_patch_graph_outputs` (`torch_spyre/_inductor/wsr/coarse_tile.py:3970`) may already have repointed the output to `full_buf`. A tiled graph output therefore takes the plain branch. |
| 2059 | `get_buffer_users(graph)`, a snapshot. | Includes writers as well as readers (`torch_spyre/_inductor/scratchpad/utils.py:545`), once per dep, so an op reading through two indexes appears twice. Clones added during the loop are not in it. |
| 2061-2063 | Drain plan maps. | `_validated_drain_plans` was computed once before the solve. It is `{}` on the placement path. |
| 2065-2069 | Skips spilled buffers and the `__spyre_lx_relayout__:` names, then `graph.get_buffer(b.name)`. | By name: `coarse_tile` may have replaced the op object under it. Copies go at line 2158. |
| 2070-2085 | **Graph input.** `_hoisted_input_clone_entry` (`:630`) picks a loop entry or `None`. Then `push_allocation_with_clone(buf, users, input=True, lx_view=..., lower_before=...)`. If hoisted, `_clear_loop_membership_metadata` (`:611`) drops `loop_info` and `_loop_carry_record`. The address and view go on the **clone**. | The input itself gets no allocation. This branch wins when a name is both input and output. |
| 2087-2118 | **Graph output.** With a drain plan: `_assert_drain_plan_committed`, then a clone with `users=[]`, `private=True`, `after_fx=loop_origin`, `lower_anchor=anchor_op`, metadata cleared. Without: a clone with all users. Either way the address and view go on the **original**, and `change_graph_output(buf, clone)` makes the clone the output. | The clone stays HBM. See "Surprises" for where the ordinary clone lands. |
| 2120-2133 | **Intermediate, staged collective.** `push_allocation_with_clone(buf, [collective], input=False, private=True, lower_anchor=buf)`, then `replace_collective_operand`. | Not exclusive with the drain branch below. |
| 2134-2153 | **Intermediate, collective drain.** Same as the output drain, then `replace_collective_operand(plan.collective, ...)`. | |
| 2154 | `_set_one_allocation(buf, ...)` (`:2160`) writes `layout.allocation["lx"]` and `layout.lx_view`. | Raises `RuntimeError` if the view is `None`. |
| 2158 | `materialize_lx_relayouts(graph, accepted)` | Last on purpose. With no plans it still raises if the registry is non-empty, which catches a double commit (`torch_spyre/_inductor/scratchpad/lx_relayout.py:2033`). |

`push_allocation_with_clone` (`torch_spyre/_inductor/scratchpad/graph_editor.py:141`) adds the FX
clone and rewires FX users (`:194-224`), then builds the `ComputedBuffer` from
`layout.unallocated_copy()`, so the clone has no allocation and no view (`:228`,
`torch_spyre/_inductor/ir.py:110`). What the order of work depends on:

- `copy_op_metadata` copies, by reference, `iteration_space_ownership`, `loop_info`, `dim_hints`,
  `_loop_carry_record` and the rest of `_SPYRE_METADATA_ATTRS` (`:238`, `:249`,
  `torch_spyre/_inductor/loop_info.py:339`).
- An input clone, the LX prefetch, derives ownership from `lx_view` (`:258-284`). An output clone copies the
  producer's `work_slices` by symbol (`:285-290`), which only works if divisions were committed.
- Only an input clone rewrites its consumers (`:292-318`). An output clone leaves them reading the
  original.
- The op lands after `lower_anchor`, or before `lower_before`, or before the first user (`:320-336`).

`materialize_lx_relayouts` (`lx_relayout.py:2032`) validates every span before any edit, checks
that each source's committed `allocation["lx"]` and `lx_view` match the plan (`:2076-2081`), inserts
a private clone through `insert_clone_before_consumers` (`graph_editor.py:340`), records it in
`graph._spyre_lx_relayout_copies`, and sets the copy's address and view.

### How each step kind becomes IR today

**Copy-out.** There is no per-edge insertion. `apply_tilings` (`coarse_tiling.py:651`) takes
`{operation name: TileSpec}`, groups consecutive ops by nest and `tile_misread`
(`derive_tiling_groups`, `:423`), and calls `coarse_tile_post_stickify` (`wsr/coarse_tile.py:2061`).
`_plan_tiling_propagation` (`:590`) then decides per **producer**, from the reads it finds, whether
the op is `loop_internal`, `copy_out`, `reduction` or `mutation_write_back`. For `copy_out`,
`_propagate_tiled_op` (`:2616`) makes one `SpyreEmptyFallback` full buffer (`:2887`) and one
`coarse_tile_copy_<name>` op (`:3002`, `:3218`), and patches **all** outside consumers at once
(`:2788-2792`). That is "one per producer, shared by every reader outside the nest": sharing is
structural. A tiled reduction instead gets an accumulator, a fill and a combine op
(`wsr/coarse_tile.py:3523`). A cut's copy takes the drained op's output split
(`coarse_tiling.py:842-850`). The ops `apply_tilings` adds are marked
`JOINT_TILING_AND_DIVISION_ATTR`, and tiled ops get `symbols_on_tile` (`coarse_tiling.py:732-751`).

**Restickify.** Inserted before work division by `insert_restickify`
(`torch_spyre/_inductor/insert_restickify.py:570`), so at LX planning it is an ordinary op with
a buffer, a division menu and a residency decision. Commit treats it like any other op.

**Relayout.** `materialize_lx_relayouts`, above. The solve fires a group per
(source, destination view) (`lx_relayout.py:279`). A relayout of a restickify's output has a real
end-to-end test (`tests/inductor/test_solver_relayout_e2e.py:1380`). A relayout that feeds a
restickify is legal: `_restickify_read_is_core_local` accepts its destination view
(`allocator.py:1513-1521`). Its only test is mocked
(`tests/inductor/test_work_division_hint.py:982`), so whether it fires on a real model is unknown.

**Boundary clones and drains** are not steps. They are per-buffer decisions in `_push_allocation`.

### How an in-place pair lowers

It does not lower to anything. `_add_inplace_relaxation` pins parent and child to one offset
(`torch_spyre/_inductor/scratchpad/ilp_solver_ortools.py:1608`), and `_set_one_allocation` writes
that address to both layouts. No op is inserted, and no edge is recorded on the IR. The reverse-parent
form (#3212) is the same with an input clone as parent. The guarantee is only in the plan, so a
commit that wrote one address differently would corrupt data with no error.

### Spill reasons

CP-SAT writes `forced_reasons.get(name)`, else `untiled_bar` if the op stayed untiled, else
`_SOLVER_CHOSE_SPILL` (`ilp_solver_ortools.py:775-782`). SA writes
`residency_reason or _SOLVER_CHOSE_SPILL` (`torch_spyre/_inductor/scratchpad/sa_cooptimizer.py:1981`),
without the untiled bar. Staging clears the reason (`allocator.py:3638`). The base allocator
substitutes a "no room" string (`allocator.py:1150`). The reasons reach only `_log_lx_pinning`.
Nothing in codegen reads them.

### Cost dump

With `config.dump_cost_expr_file` set, `CoOptimizingAllocator._solve` appends one
`cost_expr_record` line (`allocator.py:3355-3406`, `plan_solver.py:574`) after the solve and before
any commit edit, and only when the solve had a cost expression. It holds what the solve decided
(`bindings`, read from buffers by `solved_bindings`) and what it chose among (`divisions`,
`priced_relayouts`).

### What depends on the result

| reader | what it reads | citation |
|---|---|---|
| `LxContextSwitchingPass` | `calculate_liveness`, `"lx" in layout.allocation`, `origins`, `op_it_space_splits`; adds dump and restore clones and `additional_buffer_deps`. | `torch_spyre/_inductor/scratchpad/lx_context_switching.py:85-112`, `:188-357` |
| `elide_proven_read_copies` | Runs right after planning. Reads `_read_copy_elision_record`, the op list, graph output names, and proves direct reads. It can remove a pinned restickify. | `torch_spyre/_inductor/passes.py:580`, `torch_spyre/_inductor/read_copy_elision.py:607` |
| `cost_model_pass`, `dump_cost_model` | Read-only. Residency from `_mem_of_layout`; a copy is a relayout only if it is in the registry; boundaries from current input and output names, so the output clone is what carries the boundary write; one load per input clone. | `torch_spyre/_inductor/passes.py:642-644`, `torch_spyre/_inductor/cost_features.py:147`, `:1079`, `:1129`, `torch_spyre/_inductor/cost_model.py:246`, `:3056` |
| `finalize_work_division_for_scheduler` | `iteration_space_ownership` on every op including clones; turns it into `op_it_space_splits` and deletes it. | `torch_spyre/_inductor/passes.py:648`, `torch_spyre/_inductor/pass_utils.py:2157`, `:2209` |
| Scheduler, `prepare_spyre_kernels` | Regroups by `loop_info.loop_group_id` and sorts topologically, so loop metadata on clones decides their loop. Demotes LX buffers whose finalization fails, by whole relayout group through the registry. | `torch_spyre/_inductor/scheduler.py:124`, `:313`, `:352`, `:375`, `lx_relayout.py:375` |
| `SpyreKernel` | `lx_view` is mandatory for an LX buffer; ownership comes from it; the registry supplies the certified relayout marker and `completed_producer_cores`. OpSpec validation wants LX on both sides of a relayout, differing ownership, and exactly one of hbm, lx, hbm_pool. | `torch_spyre/_inductor/spyre_kernel.py:879`, `:901`, `:1166-1190`, `torch_spyre/_inductor/op_spec.py:300`, `torch_spyre/_inductor/op_spec_validation.py:493` |
| `hbm_pool_planning` | `"lx" not in layout.allocation`, and aliases by `id(layout.allocation)`. | `torch_spyre/_inductor/hbm_pool_planning.py:257`, `:312` |
| for-each-tile lowering | Runs first (`passes.py:512`) and is never re-entered. It leaves behind what commit consumes: `loop_info`, `_loop_carry_record`, the retained `while_loop` FX node in `DrainPlan.loop_origin`, and `LoopNode`s. | `torch_spyre/_inductor/loop_node.py:193` |

### Surprises

- **The ordinary output clone lands before its producer.** `_push_allocation` passes
  `buffer_users[b.name]`, which includes the writer, so `first_user` is the producer and the
  clone is inserted in front of it (`graph_editor.py:235`, `:332-336`). A stub run mirroring
  `tests/inductor/test_scratchpad_use.py:106` gave `[before, CLONE, source, use]`, so the
  post-commit list is not topological, and `calculate_liveness` in the context-switching pass
  sees it that way. Every FX user moved to the clone while the IR consumers kept the original.
  Both must be preserved for parity, and commit gets them by calling the same editor method.
- **The SA co-optimizer gets no post passes.** `select_allocator` builds
  `CoOptimizingAllocator(layout_planning=SaCoOptimizingSolver, size=size)` with no
  `post_optimization_passes` (`allocator.py:5433-5435`), while `_BufferResidencyChecks.extern`
  skips the extern guard whenever `enable_lx_context_switching` is on (`allocator.py:857-862`),
  which is the default (`config.py:56`). So with `LAYOUT_SOLVER=simulated_annealing` a buffer can
  stay in LX across a single-output, risky `FallbackKernel` with no bracket. A default compile does
  not reach this. `scratchpad_planning`'s comment says it is unsafe (`allocator.py:5501-5504`).
- **Allocation order differs by path.** `_build_cd_bound_buffers` appends input clones first
  (`allocator.py:4228-4266`). `_build_bound_buffers` appends them last (`:1604-1630`). Order sets
  clone names and the order of clones in front of one op.
- **Residency changes after the solve.** Staging (step 6) gives a spilled buffer an address and
  moves no chosen one (`allocator.py:3534-3537`), a documented exception
  (`docs/source/compiler/scratchpad_planning.md`, "Matmul output staging before a collective").
  It runs after `check_decisions` on the new path, so commit judges it again.
- **Committed mode must not re-commit divisions.** `anchor_lx_relayout_ownership` writes a
  chosen owner order with `commit_tensor_work_division` before planning
  (`lx_relayout.py:1890`). `commit_iteration_space_ownership` would rebuild the canonical order
  from split counts.
- **A strict comparator already exists.** `tools/lx_planning/lx_semantic_plan.py:181`
  (`graph_snapshot`) records operations, layouts with allocation and view, dependencies,
  ownership, `loop_info`, outputs, removed buffers and the relayout registry, and
  `capture_off_plan` (`:249`) adds final OpSpecs. Buffer names are significant there. Its
  contract insists on `lx_extended_ownership` being off (`:244`), which is not the default
  (`config.py:170`).

## Design

### Where the code lives

A new module `torch_spyre/_inductor/scratchpad/lx_allocator/commit.py` holds the lowering.
`commit(graph)`, in `lx_allocator/optimizer.py`, is a thin call into it. The dependency runs
one way: `commit.py` imports from `allocator.py`, and `allocator.py` imports nothing from
`commit.py`, so `_push_allocation` and every existing test patch on `allocator_module` keep working.

### Two phases: read, then lower

```python
@dataclass(frozen=True)
class CommitPlan:
    """What the solved graph asks for, in names, read before any IR edit."""
    order: tuple[str, ...]                 # record order, see below
    tilings: Mapping[str, TileSpec]        # operation name -> tile_spec_of(node, split)
    divisions: Mapping[str, CoreDivision]  # buffer name -> division_from_work_split(node, split)
    planned: frozenset[str]                # every name the solve saw
    placements: Mapping[str, int]          # resident name -> byte offset
    spans: Mapping[str, int]               # resident name -> placed per-core bytes
    views: Mapping[str, PerCoreView]       # names whose view is decided, see below
    relayouts: tuple[LXRelayoutPlan, ...]
    copy_outs: Mapping[str, frozenset[str]]  # producer -> consumers on COPY_OUT edges
    drains: Mapping[str, DrainPlan]        # graph.drain_plans

@dataclass(frozen=True)
class CollectiveStage:
    address: int
    lx_view: PerCoreView
    collective: Operation

@dataclass(frozen=True)
class CommitResult:
    spill_reasons: Mapping[str, str]
    stages: Mapping[str, CollectiveStage]

def read_commit_plan(graph: SpyreOptimizationGraph) -> CommitPlan: ...
def lower_commit(graph: SpyreOptimizationGraph, plan: CommitPlan) -> CommitResult: ...

def commit(graph: SpyreOptimizationGraph) -> None:
    if graph.ir is None:                # a hand-built graph in a unit test
        return
    lower_commit(graph, read_commit_plan(graph))
    for post_pass in _context_switching_passes():
        post_pass.apply_pass(graph.ir)
```

`commit` runs the post passes itself, on `graph.ir`, after `lower_commit`. They belong here and not
in `plan_allocation`, because Stage 11's held-plan path calls `commit` directly. They run last, as
at `allocator.py:1052`, and `LxContextSwitchingPass` stays a separate, unchanged pass. The list is
Stage 9's `_context_switching_passes()`, the same for every family. `plan_allocation` calls
`commit` after `optimize` returns with a plan read back, never after a proven-infeasible solve.

`read_commit_plan` makes no IR edit. It is the last point a bug can be reported with the graph
intact. It raises `RuleViolation` for what would otherwise miscompile silently:

- a `RESTICKIFY` step with `pinned=False`, or a step whose kind is not in `graph.target.steps`,
  see "Step kinds";
- a pinned `RESTICKIFY` step whose `realized_by` is `None` or names an op missing from
  `ir.operations`;
- a pinned step of any kind other than `RESTICKIFY` or `COPY_OUT`;
- a `RELAYOUT` step after a pinned `RESTICKIFY` step on one edge, see "Folding relayouts";
- an `in_place` edge whose two ends are not both resident at one offset;
- a `COPY_OUT` step that is resident or has an offset;
- a `RELAYOUT` step that is not resident, or whose source is an `InputNode`, or two `RELAYOUT`
  steps of one source and one `to_view` partition;
- a resident `InputNode` with no `prefetch_view`.

It does not repeat `check_decisions`. Everything after this point names buffers, never holds an
op across `apply_tilings`, because `coarse_tile` replaces op objects under the same name
(`torch_spyre/_inductor/wsr/coarse_tile.py:2153-2163`).

The graph holds alignment units, and this is the edge that scales them:
`placements[name] = node.offset * graph.target.alignment`, and likewise for a `RELAYOUT` step's
`offset`. `spans[name]`
is `ceil_div(node.size, split.output_partition * split.output_tile_count)`, today's unrounded
`per_core_size` (`plan_solver.py:395`), which `check_priced_footprint` compares against. A resident
input has no `WorkSplit`, so its span is `ceil_div(node.size, node.prefetch_view.split_product)`.
`plan.views` holds each relayout source's `source_view` and each resident input's `prefetch_view`,
which take precedence over the judge in `accepted_lx_views`.

`graph.divisions_committed` decides committed mode, not "every division is pinned", because co-opt
with every op pinned still re-commits ownership today.

Lowering knows no solver family. It reads decisions from the graph and never a menu index, so a
debug greedy family that fills in the same fields commits without change. Such a family's
`dump_tables` returns `cost_expr=None`, so it writes no dump line.

### The order of work

`lower_commit` runs these in exactly this order. Each row says why it cannot move.

| # | step | old code | must follow | why |
|---|---|---|---|---|
| 1 | `apply_tiling_choices(ir, plan.tilings)` | `apply_chosen_tilings` | | Everything below reads the tiled graph. |
| 2 | `verify_copy_outs(ir, plan)` | new | 1 | |
| 3 | `check_priced_footprint` per tiled buffer | `check_priced_footprints` | 1 | Reads applied layouts. |
| 4 | `commit_chosen_divisions(ir, plan.divisions, plan.planned)`, skipped if `divisions_committed` | `commit_divisions` | 1 | `planned_splits` restates splits on tile symbols and needs the copy ops. |
| 5 | `accepted_lx_views(...)` | `commit_lx_views` | 4 | The judge reads committed ownership. |
| 6 | `collective_output_stages(...)` then `no_overlap(graph)`, skipped if `divisions_committed` | `_stage_collective_outputs` | 4, 5 | Uses current liveness, spans and committed ownership. See below. |
| 7 | `spill_reasons(...)` | `_get_spill_reasons` | 6, before 8 | A clone must not read as "added by coarse tiling". |
| 8 | `push_one_placement` per name in `plan.order` | `_push_allocation:2065-2154` | 4, 5, 6 | Clones snapshot metadata and copy `work_slices`. |
| 9 | `materialize_lx_relayouts(ir, list(plan.relayouts))` | `:2158` | 8 | It raises unless the source's address and view are already on its layout (`lx_relayout.py:2076-2081`). |
| 10 | Debug logs, `solver_out:` and `lx_pinning:` | `log_solver_decisions`, `_log_lx_pinning` | 9 | `_log_lx_pinning` walks the final op list. |

This is the parent's "steps, then divisions and offsets" only in part. See Deviation 1.

`plan.order` is the record order of the path being replaced:

```python
ops = [op.name for op in ir.operations]          # taken before any tiling
inputs = list(ir.graph_input_names)
order = ops + inputs if graph.divisions_committed else inputs + ops
```

**Judging staged outputs.** Staging adds addresses after `check_decisions`, so commit judges them
again, as `_finish_plan_check` does on the old path. Stages are computed at step 6 and not in
`read_commit_plan`, because their inputs exist only there: `calculate_liveness` on the tiled graph
(`allocator.py:3551`) and the judge on committed ownership (`allocator.py:3613`). Step 6 still edits
no IR, and no address has landed. For each stage it sets `node.resident`, `node.offset` to
`address // graph.target.alignment` and `node.spill_reason` to `None`, the graph form of
`allocator.py:3638`. `:3635` aligns to 128 (`plan_solver.py:1073`), so the division is exact. Stage
3's `no_overlap(graph)`, which reads no IR, then judges every box, and a violation raises
`RuleViolation`.

### Reuse by extraction, not copying

Parity is safest if both paths run one implementation. Each item below is extracted in its own
module with today's public function left as a thin wrapper. Stage 13 deletes the wrappers.

| new function | home | extracted from | old callers |
|---|---|---|---|
| `apply_tiling_choices(graph, choices) -> bool` | `coarse_tiling.py` | `apply_chosen_tilings:781-793` | wrapper passes `chosen_tilings(...)` |
| `splits_from_divisions(graph, chosen, planned_names)` | `coarse_tiling.py` | `planned_splits:830-860` | wrapper |
| `commit_chosen_divisions(graph, chosen, planned_names)` | `allocator.py` | `commit_divisions:2907-2944` | wrapper |
| `check_priced_footprint(graph, name, division, priced, size)` | `allocator.py` | body of the `check_priced_footprints` loop | wrapper loop |
| `accepted_lx_views(graph, names, source_views, drained_readers)` | `allocator.py` | `commit_lx_views:3015-3029` | wrapper sets `buffer.lx_view` |
| `collective_output_stages(graph, placed, spilled, residents, relayouts, capacity)` | `allocator.py` | `_stage_collective_outputs:3549-3646` | method wraps it and writes `b.address`, `b.lx_view`, `b.spill_reason`; the new path passes `graph.target.lx_bytes` |
| `push_one_placement(...)` | `allocator.py` | `_push_allocation:2069-2154` | `_push_allocation` becomes its loop plus line 2158 |
| `cost_record(...)` | `plan_solver.py` | `cost_expr_record`'s body, see "Cost dump" | `cost_expr_record` derives plain values from buffers and calls it |

```python
def push_one_placement(
    graph: GraphLowering, editor: GraphEditor, name: str, address: int,
    lx_view: PerCoreView | None, *,
    users: Mapping[str, list[Operation]], inputs: set[str], outputs: set[str],
    drain_plan: DrainPlan | None, collective: Operation | None,
    check_drain: Callable[[DrainPlan], None],
    set_allocation: Callable[[Any, int, PerCoreView | None], None],
) -> None: ...
```

`check_drain` is injected because `_assert_drain_plan_committed` (`allocator.py:691`) reads
records. The old path passes a closure over it. The new path passes `check_drain_committed`, the
same four structural checks plus the admitted-pair check on the chosen splits alone:
`edge.match_pairs([storage_split], [update_split]) == [(0, 0)]`, with both splits from
`plan.divisions`, where `edge` comes from `_loop_carry_update_edge`, which Stage 6 moved to
`lx_allocator/family/edge_tables.py`. `_hoisted_input_clone_entry`,
`_clear_loop_membership_metadata`, `_drained_collectives` and `DrainPlan` are reused in place.

### Step kinds

Commit does not walk edges inserting steps one at a time. It reads each edge's `DataEdge.steps`
and lowers each `StepKind` in the phase where the IR allows it.

| `step.kind` | how it becomes IR | through |
|---|---|---|
| `COPY_OUT` | Nothing per edge. `tilings` carries each op's `TileSpec`, and `apply_tilings` builds one copy per producer. `plan.copy_outs` is only verified. A pinned `COPY_OUT` step is skipped. | `coarse_tile`, not `GraphEditor` |
| `RESTICKIFY`, pinned | Nothing inserted. Its `realized_by` node gets a division and an address in the ordinary walk. | see "Pinned steps" |
| `RESTICKIFY`, unpinned | `read_commit_plan` raises. Unpinned steps exist only from Stage 11b, carrying `target`, and Stage 11b changes this raise: `insert_restickify` lowers them, so commit inserts nothing. | Stage 11 |
| `RELAYOUT` | Folded into one `LXRelayoutPlan` per group, materialized last. | `GraphEditor.insert_clone_before_consumers` |
| a kind not in `graph.target.steps` | `read_commit_plan` raises. | |

Per edge, the steps are read in list order, which is physical, and nothing sorts them.
`STEP_KINDS[kind].order` only placed a step when a decoder appended it, so commit never reads it
(Deviation 3).

**Verifying copy-outs.**

```python
def verify_copy_outs(ir: GraphLowering, plan: CommitPlan) -> None:
    ops = {op.get_name(): op for op in ir.operations if isinstance(op, ComputedBuffer)}
    for producer, consumers in plan.copy_outs.items():
        propagation = ops[producer].loop_info.propagation
        if propagation.kind not in ("copy_out", "reduction"):
            raise RuleViolation(...)
        if propagation.consumer_lookup_name is None:
            for name in consumers:
                if producer in {d.name for d in op_read_writes(ops[name]).reads}:
                    raise RuleViolation(...)
```

Only that direction is checked. `coarse_tile` also copies out consumers deferred past a reduction
(`wsr/coarse_tile.py:767`), which the solve has no edge step for. A tiled graph output, or a tiled
buffer read only by an extern kernel, is cut (its tick is in `graph.cuts`) with no data edge to
carry a step: `read_back` adds no `COPY_OUT` step, its traffic is priced by node, and
`verify_copy_outs` skips such a producer. Commit passes no cuts to `apply_tilings`, which derives
its own nests; `graph.cuts` serves only that DEBUG log of reverse differences. `COPY_OUT` steps are
deduplicated by producer in `plan.copy_outs`, so three readers mean one set entry and one IR copy.

**Folding relayouts.** `relayout_plans(graph)` walks `graph.all_steps()`, which lists each shared
`RELAYOUT` step once, and folds one `LXRelayoutPlan` per step from `step.edges`:

- The **source** is `edge.src.node.name`, the same on every edge of the step.
- The **consumers** are the `edge.dst.node.name`s. A relayout that feeds a restickify sits before
  the pinned step on the edge into the `RestickifyNode`, so that node is its consumer. A relayout
  of a restickify's output sits on an edge out of that node, so that node is its source. A
  `RELAYOUT` step after a pinned `RESTICKIFY` step on one edge has no meaning, so
  `read_commit_plan` raises.
- Consumers sort by name and plans by `(source, consumer_names)`. `FiredRelayoutGroup.from_chosen`
  sorts by an interned group id, which the graph does not hold and the parent's direction keeps out
  of the graph layer. The cost is the narrow normal form in "Parity and gates".

| `LXRelayoutPlan` field | source |
|---|---|
| `source_name`, `consumer_names` | as above |
| `source_view`, `source_footprint_bytes`, `destination_footprint_bytes` | the `RELAYOUT` step's fields of those names |
| `destination_view` | `step.to_view` |
| `num_cores` | `source_view.num_cores` |
| `completed_producer_cores` | `()` if `reduction_split <= 1`, else the cores of `derive_completed_reduction_routes(source_view, to_view, reduction_split)` |
| `source_address`, `destination_address` | the source's and the step's `offset * graph.target.alignment` |

`LXRelayoutPlan` and `materialize_lx_relayouts` stay as they are, so the registry that
`cost_features`, `scheduler` and `spyre_kernel` read does not change.

### Pinned steps

A step is recognised as existing by `step.pinned`, which Stage 2 sets. Commit never infers it from
layouts. The step's `realized_by` names the `RestickifyNode`, which holds the division, residency,
box and tick; the step holds none of them. One restickify read by two consumers is one node, so
commit handles it once: its op is already in `ir.operations`, so its name falls in `plan.order` at
its normal place and takes the ordinary branch of `push_one_placement`, including the output branch
if it is a graph output. A relayout naming it finds it by name. `elide_proven_read_copies` may
remove it later, as today.

### LX prefetches, output clones and drains

These stay node-level. An `InputNode` with `resident=True` is an LX prefetch, and its view is
`prefetch_view`. A resident node whose name is in `ir.get_output_names()` **after tiling** is an output
clone. An entry in `plan.drains` selects the drain branch. Consumer lists come from
`get_buffer_users(ir)`, not from edges, because edges do not list ops that a step folded away or
that tiling added.

### In-place pairs

`read_commit_plan` checks the pair and then has nothing to lower: both names are in
`plan.placements` with one offset, and `push_one_placement` writes it to each. Under an LX
prefetch parent, the address lands on its `clone` op and on the consumer. The offset is in alignment units
on both ends and scaled once, as `ilp_solver_ortools.py:773` does.

### Spill reasons

`spill_reasons(ir, graph, plan)` starts from `node.spill_reason` on every spilled node, because
only `read_back` knows a solver chose to spill, and staging has cleared a staged node's reason.
Then, if any op carries `JOINT_TILING_AND_DIVISION_ATTR`, every op outside `plan.planned` gets
`setdefault(name, "added by coarse tiling")` (`allocator.py:3659-3668`). In committed mode a node
with no reason takes the base allocator's "no room" string (`allocator.py:1150`). The map goes to
the debug log and to `CommitResult.spill_reasons`, where the harness reads it.

### Failure

- `RuleViolation` from `read_commit_plan`: nothing edited.
- `RuleViolation` from `verify_copy_outs` or step 6, or `Unsupported` from steps 3 to 5: tilings are
  already applied, so a fallback would replan a graph it did not see (`allocator.py:2902-2905`).
  These propagate, because `plan_allocation` calls `commit` after `optimize` returns, outside the
  fallback flow.
- `SolveError` stays before commit, so `scratchpad_planning`'s guarantee (`allocator.py:5496-5500`) holds.

### Removing the Stage 9 bridge

The bridge goes in the same commit that switches `plan_allocation` to `commit(graph)`, so no tree
has neither or both. That commit deletes `lx_allocator/optimizer_bridge.py` (`records_from_graph`,
`_CommitBridge`), `_commit_via_bridge`, Stage 9's reduced cost line and
`tests/inductor/test_optimizer_bridge.py` with its config shard, and adds
`test_optimizer_reads_no_solver_records`. The Stage 3 adapter stays: it runs the other way, records
onto the graph, and the shadow gate needs it. The gate's base still has the bridge, so its removal
costs no comparison.

### Cost dump

`optimize` writes the full `cost_expr_record` schema from decoded values, in place of Stage 9's
reduced line. It writes once `check_decisions` passes and before it returns, because it is the last
holder of the accepted attempt's family, `Encoding` and `Solution`, and because `bundle_terms` reads
the untouched IR, as `_solve` does today: `plan_allocation` calls `commit` only after `optimize`
returns. It writes nothing when the encoding had no priced expression, also as today. The cost
model is `graph.target.cost_model`.

`cost_expr_record`'s body moves to `cost_record`, which takes plain values, so one body builds the
record. Menus stay inside the family's `Encoding`, so this stage adds `dump_tables` to the `Family`
protocol, and adds `score_ns()` and `off_expression_ns()` to `Solution` with the defaults of
`plan_solver.py:1173-1181`. `AnnealSolution` already overrides both.

```python
@dataclass(frozen=True)
class DumpTables:
    cost_expr: sympy.Expr | None                # as `encode` priced it; None writes no line
    domains: CostDomains
    divisions: Mapping[str, Mapping[str, Any]]  # the record's `divisions` value
    priced_relayouts: Mapping[str, Any]         # the record's `priced_relayouts` value

class Family(Protocol):
    ...                                         # Stage 7's members
    def dump_tables(self, encoding: Encoding, solution: Solution) -> DumpTables: ...

def graph_bindings(graph, domains: CostDomains, solution: Solution) -> dict[sympy.Symbol, int]: ...
def write_cost_dump(graph, family: Family, encoding: Encoding, solution: Solution) -> None: ...
```

`graph_bindings` is `solved_bindings` over decoded values. For each name in `domains.nodes` it binds
`Sym.resident` from `node.resident`, then `Sym.split` and `Sym.tile` from
`division_from_work_split(node, node.work_split)`, as `:542-566` does. A relayout copy's name has no
node, and the graph keeps no group name, so its `Sym.resident` is `name in solution.offsets`.
Charges are keyed by split, so no term evaluates on a menu index, which is what made a bridge
record's index 0 wrong.

| record key | source on the new path |
|---|---|
| `bindings` | `graph_bindings`, plus each `division_<name>` from `tables.divisions` |
| `bundles`, `objective_ns` | `graph.target.cost_model.bundle_terms(graph, tables.domains)`, `tables.cost_expr` |
| `relayout_terms` | each `SplitKeyedCharge` in `tables.cost_expr.args` |
| `divisions`, `priced_relayouts` | `tables`. `OrToolsFamily` fills them from `DivisionTable`, `Gate.pairs` and `Gate.relayouts`; `SAFamily` from `AnnealModel.menus` and `relations`, with no relayouts |
| `off_expression_ns`, `score_ns`, `solve` | `solution`, with `solve` from `solution.stats` |
| `buffers`, `buffer_sizes`, `params`, `op_names`, `op_ids`, `env` | as today (`allocator.py:3369-3391`), with `env.solver` the family's `FAMILIES` key and `env.allocator` `"plan_allocation"` |

Not verified: SA's `ViewRelation` holds no pair list, and a generated tiling can lie outside the
menu, so whether `matches` and `chosen` reproduce today's under `AUTO_COARSE_TILING=1` on SA is
unknown. G5 settles it.

### Cache key

At `5268ba95` planner source is not in the FX key. `_uuid` hashes what a pass declares
(`passes.py:164-171`), but nothing calls `CustomPreSchedulingPasses.uuid()`, so `@_runs` alone
reaches no key. Stage 9 fixes that through `optimizer_pass_sources()`, at the cost of one recompile
for every user when Stage 9 lands. This stage adds `lower_commit` to that list, so `commit.py` is
hashed, and every user recompiles once more when this PR lands.

## Changes

| file | change |
|---|---|
| `torch_spyre/_inductor/scratchpad/lx_allocator/commit.py` | New. `CommitPlan`, `CollectiveStage`, `CommitResult`, `read_commit_plan`, `lower_commit`, `relayout_plans`, `verify_copy_outs`, `spill_reasons`, `check_drain_committed`, `graph_bindings`, `write_cost_dump`. |
| `torch_spyre/_inductor/scratchpad/allocator.py` | Extract `push_one_placement`, `commit_chosen_divisions`, `check_priced_footprint`, `accepted_lx_views`, `collective_output_stages`. No behaviour change. |
| `torch_spyre/_inductor/scratchpad/coarse_tiling.py` | Extract `apply_tiling_choices`, `splits_from_divisions`. No behaviour change. |
| `torch_spyre/_inductor/scratchpad/plan_solver.py` | Extract `cost_record`. No behaviour change. |
| `.../lx_allocator/optimizer.py` | Real `commit(graph)`, called by `plan_allocation` after `optimize`; `optimize` calls `write_cost_dump`; bridge call and reduced cost line deleted. |
| `.../lx_allocator/optimizer_bridge.py`, `tests/inductor/test_optimizer_bridge.py`, `tests/configs/torch_spyre_tests/inductor/test_optimizer_bridge_config.yaml` | Deleted. |
| `.../lx_allocator/family/base.py`, `family/ortools_encode.py`, `family/sa/`, `family/placement.py` | `DumpTables`, `Family.dump_tables` with `OrToolsFamily.dump_tables` and `SAFamily.dump_tables`, `Solution.score_ns`, `.off_expression_ns`. |
| `optimizer_pass_sources()` (Stage 9) | Adds `lower_commit`. |
| `tests/inductor/lx_equivalence.py`, `lx_corpus.py` | New record fields, `SCHEMA_VERSION`, `compile_model(stop_at="codegen")`, the commit hook, `commit_shadow`, all under "Parity and gates". |
| `tests/inductor/test_lx_commit.py`, `test_lx_commit_e2e.py` | New host tests, new device tests. |
| `docs/source/compiler/scratchpad_planning.md` | Name `commit(graph)` beside `_push_allocation` at `:308`. |

Commits, each leaving the suites green:

1. **Extract the shared helpers.** Refactor only. The existing tests at
   `tests/inductor/test_coarse_tiling.py:9750-10780`, `tests/inductor/test_work_division.py:3280`
   and `tests/inductor/test_cost_expr_dump.py` prove it unchanged, and G0 confirms it.
2. **Extend the harness record.** Add the fields and `compile_model(stop_at="codegen")`, and bump
   `SCHEMA_VERSION`. Nothing is re-recorded by hand: `lx_gate.sh` records the base tree with HEAD's
   harness, so both sides carry the fields.
3. **Add `read_commit_plan` and `relayout_plans`** with host tests. Nothing calls them yet.
4. **Add `lower_commit` and `verify_copy_outs`** with host tests, plus `commit_shadow`, which
   drives them from the old allocators. This is how the new code runs before production calls it.
5. **Add `dump_tables`, `graph_bindings` and `write_cost_dump`** with host tests, still uncalled.
6. **Switch `plan_allocation` to `commit(graph)`, delete the bridge,** and add the post passes, the
   dump, the `optimizer_pass_sources()` entry, the commit hook, the guard, device tests and the doc
   note.

## Interfaces

### Consumes

| symbol | signature | from Stage | used for |
|---|---|---|---|
| `SpyreOptimizationGraph` | `.ir: GraphLowering \| None`, `.target`, `.divisions_committed`, `.drain_plans`, `.nodes` (op order), `.data_edges`, `.cuts: set[int]`, `.node(name)`, `.all_steps()` | 1, filled by 2, 7, 8 | walking the plan; `commit(graph)` reaches the IR only through `.ir` |
| `Target` | `graph.target.alignment`, `.lx_bytes`, `.steps: frozenset[StepKind]`, `.cost_model` | 1, cost model from 4 | unit scaling, staging capacity, step kinds offered, cost dump |
| `OpNode` | `.name`, `.op`, `.size` (total bytes), `.graph_output`, `.work_split`, `.resident`, `.offset` (alignment units), `.spill_reason` | 1, filled by 2, 7, 8 | placements, spans, reasons |
| `InputNode.prefetch_view` | `PerCoreView \| None`; `work_split` stays `None` on inputs | 1, filled by 7, 8 | LX prefetches |
| `WorkSplit` | `.splits`, `.tiling: tuple[TileLevel, ...]`, `.output_partition`, `.output_tile_count` | 1 | spans |
| `division_from_work_split`, `tile_spec_of` | `(node, split) -> CoreDivision`, `(node, split) -> TileSpec` | 2 | `.reduction_syms`, `.label`, `CommitPlan.tilings` |
| `DataEdge` | `.src`, `.dst`, `.in_place`, `.steps: list[Step]` in physical list order | 1, filled by 2, 7, 8 | steps, in-place |
| `Step`, `StepKind`, `STEP_KINDS` | `Step(kind=..., ...)`: `.kind`, `.resident`, `.offset`, `.footprint` (alignment units), `.pinned`, `.edges`; `StepKind.COPY_OUT`, `.RESTICKIFY`, `.RELAYOUT` | 1, filled by 7, 8 | steps |
| `RESTICKIFY` step, `RestickifyNode` | `Step(kind=StepKind.RESTICKIFY, to_stick=..., pinned=True, realized_by=node)`, with no footprint or box; a direct `OpNode` subclass, decoded as any node | 1, filled by 2 | pinned restickifies |
| `RELAYOUT` step | `.to_view`, `.reduction_split`, `.source_view`, `.source_footprint_bytes`, `.destination_footprint_bytes` | 1, filled by 3, 7, 8 | the plan |
| `no_overlap`, `RuleViolation` | `(graph) -> list[Violation]`, `RuleViolation(violations)` | 3 | step 6, failures |
| `build_graph`, `select_target`, `decode_todays_plan` | `(ir, target, *, divisions_committed, drain_plans, lx_relayout_plans)`; `(config) -> Target`; `(graph, allocation, accepted_relayouts) -> list[Mismatch]`, temporary, removed in Stage 13 | 2, 3 | `commit_shadow` |
| `CostModel.bundle_terms`, `.params`; `CostDomains`, `NodeDomain`, `Sym`, `SplitKeyedCharge` | `(graph, domains) -> list[tuple[list[str], sympy.Expr]]`; as Stage 4 | 4 | cost dump |
| `_loop_carry_update_edge` | moved to `family/edge_tables.py` | 6 | `check_drain_committed` |
| `Family.read_back` | `read_back(solution, graph) -> None`: offsets in alignment units; both ends of an in-place pair; one shared `RELAYOUT` step per group; `graph.cuts`; `OpNode.spill_reason` | 7, 8 | the plan |
| `Family`, `Encoding`, `FAMILIES`; `OrToolsFamily`, `SAFamily` | Stage 7's protocol, which this stage extends with `dump_tables`: `from_config(config)`, `encode(graph) -> Encoding`, `solve(encoding, settings) -> Solution`; `Encoding.model`, `.ortools` (`OrToolsState.tables`, `.edges`); `AnnealModel.menus`, `.relations`; `FAMILIES: Mapping[str, type[Family]]` | 5, 6, 7, 8 | `dump_tables`, cost dump |
| `Solution`, `AnnealSolution` | `.offsets` by `PlacementKey`, resident sites only; `.stats`; `score_ns()`, `off_expression_ns()` on `AnnealSolution` | 7, 8 | cost dump |
| `plan_allocation`, `optimize` | `plan_allocation(ir, config) -> SpyreOptimizationGraph`; `optimize(graph, family, fallback, settings)`: `check_decisions` raises before return, and nothing is read back after a proven-infeasible solve | 9 | `plan_allocation` calls `commit` after `optimize`, outside the fallback flow; `optimize` writes the cost dump |
| `_context_switching_passes` | `() -> list[ScratchpadOptimizationPass]`, the same for every family | 9 | `commit` runs them |
| `optimizer_pass_sources()` | planner sources in the FX key | 9 | adds `lower_commit` |
| `records_from_graph`, `_CommitBridge`, `_commit_via_bridge` | temporary; records in `_build_cd_bound_buffers` order, inputs first | 9 | G2's base, then deleted here |
| harness | `lx_gate.sh`, `Recorder.record_commit`, `snapshot_commit`, `OPTIONAL_TARGETS`, `compile_model`, `corpus_coverage`, `SCHEMA_VERSION` | 0 | gates, extended below |

### Provides

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `commit` | `commit(graph) -> None`, runs `_context_switching_passes()` after `lower_commit` | 11, 12, 13 | final |
| `read_commit_plan`, `lower_commit`, `CommitPlan`, `CommitResult`, `CollectiveStage`; `relayout_plans` | as in Design; `lower_commit` returns the `CommitResult` the harness hook reads; `(graph) -> tuple[LXRelayoutPlan, ...]` | 11 | final |
| `write_cost_dump`, the `cost_expr_record` writer | `(graph, family, encoding, solution) -> None`; today's schema | 12, 13 | final |
| `cost_record`, `DumpTables`, `Family.dump_tables` | as in Design; `dump_tables(encoding, solution) -> DumpTables`, which extends Stage 7's `Family` protocol; `OrToolsFamily` and `SAFamily` implement it | 12, 13 (every family `FAMILIES` holds) | final |
| `push_one_placement`, `commit_chosen_divisions`, `check_priced_footprint`, `accepted_lx_views`, `collective_output_stages`; `apply_tiling_choices`, `splits_from_divisions` | as in Design | 13 | final; they stay in `allocator.py` and `coarse_tiling.py` |

For Stage 13: `commit.py` imports these from `allocator.py`, so they stay there: the five functions
above, `_hoisted_input_clone_entry`, `_clear_loop_membership_metadata`,
`_extern_kernel_in_live_range`, `_multi_output_extern_kernel_in_live_range`, `_drained_collectives`,
`DrainPlan`, `_split_option_is_legal` and `_hard_domain_violation`. `lx_context_switching.py:51`
also imports `ScratchpadOptimizationPass` from there.

## Tests

`tests/inductor/test_lx_commit.py`, host. Solved graphs are hand-built from Stage 1 types. IR is
either a mock `GraphEditor`, as `tests/inductor/test_coarse_tiling.py:9790-9822` does, or the tiny
real IR of `tests/inductor/test_scratchpad_use.py:106`.

| test | asserts |
|---|---|
| `test_read_plan_makes_no_ir_edit` | Operations, FX and layouts are unchanged after `read_commit_plan`. |
| `test_read_plan_rejects_unpinned_restickify`, `..._step_kind_off_target`, `..._pinned_relayout`, `..._relayout_after_restickify`, `..._in_place_offset_mismatch`, `..._spilled_end`, `..._resident_copy_out`, `..._missing_pinned_op`, `..._input_without_prefetch_view` | `RuleViolation` naming the subject. |
| `test_read_plan_scales_units` | `placements` and plan addresses are `offset * graph.target.alignment`, also on a derived target with another alignment; plan footprints are the unrounded byte fields. |
| `test_order_is_inputs_first_in_co_opt`, `test_order_is_inputs_last_when_committed` | `plan.order` matches each allocator's record order. |
| `test_shared_relayout_is_one_plan`, `test_two_views_are_two_plans` | Two edges, one `RELAYOUT` step: one plan, consumers sorted by name. One source, two views: two plans. |
| `test_relayout_group_disagreement_raises` | Two `RELAYOUT` steps of one source and view raise. |
| `test_relayout_before_pinned_restickify_feeds_the_restickify`, `test_relayout_of_restickify_output_reads_the_restickify` | Plan consumer is the `RestickifyNode`; on an edge out of it, plan source is. |
| `test_completed_route_sets_producer_cores` | `reduction_split > 1` yields the same cores as `derive_completed_reduction_routes`. |
| `test_pinned_restickify_is_not_reinserted`, `..._shared_by_two_edges_commits_once`, `..._takes_division_and_address` | No op is added. One node, one division, one address, landing on the existing op. |
| `test_in_place_pair_lowers_to_one_address`, `test_in_place_with_prefetch_parent` | Both layouts hold the offset, no op added; the `clone` op and the consumer hold one address. |
| `test_lower_order` | A recording mock sees tilings, divisions, views, stages, `no_overlap`, placements, relayouts in that order. |
| `test_committed_mode_skips_division_commit` | A preset `core_id_to_work_slice` survives. |
| `test_prefetch_goes_on_clone`, `test_prefetch_hoisted_clears_membership`, `test_output_clone_goes_on_original_and_swaps_output` | Ported from `test_coarse_tiling.py:10665`, `:10683`, `:9844`. |
| `test_output_clone_lands_before_producer`, `test_tiled_output_takes_the_plain_branch` | Records today's order so a change is deliberate. No clone when tiling moved the output. |
| `test_drain_clone_after_anchor`, `test_collective_drain_rewrites_operand` | Ported from `:9824`, `tests/inductor/test_work_division.py:3280`. |
| `test_staged_collective_clones_and_rewrites`, `test_spilled_node_gets_no_clone` | Ported from `test_coarse_tiling.py:10322`, `:9855`. |
| `test_staged_overlap_raises_before_placement` | An overlapping stage raises `RuleViolation` at step 6, and no placement runs. |
| `test_spill_reason_reads_the_node`, `test_spill_reason_added_by_tiling` | The rule in Design, a staged node has none, and clones do not appear. |
| `test_relayouts_materialize_after_placements` | `materialize_lx_relayouts` receives a source that already has its address and view. |
| `test_verify_copy_outs_rejects_missing_copy`, `..._ignores_graph_output_copy` | `RuleViolation`; a producer with no `COPY_OUT` step is skipped. |
| `test_commit_runs_post_passes_after_lower` | A recording pass sees the lowered IR, once, on every family. A graph with `ir=None` returns early. |
| `test_optimizer_reads_no_solver_records` | `lx_allocator/optimizer.py` and `commit.py` do not mention `LifetimeBoundBuffer`, `CoreDivisionBuffer`, `_push_allocation` or `_post_solve`. |
| `test_commit_module_is_in_pass_sources` | `lower_commit` is in `optimizer_pass_sources()`, so `commit.py` is hashed. |
| `test_cost_dump_terms_evaluate`, `test_cost_dump_is_written_before_lowering`, `test_no_cost_dump_without_priced_expression` | No `value_ns` is `None` under `graph_bindings`, relayout charges included. `optimize` writes the line before `plan_allocation` calls `commit`, so it precedes every IR edit. No expression, no line. |

`tests/inductor/test_lx_commit_e2e.py`, device, each with the new family selected and values
checked against CPU: `test_copy_out_one_per_producer` (one `coarse_tile_copy_*` for three outside
readers), `test_grouped_relayout` (with `tests/inductor/relayout_fixtures.py:grouped_relayout_graph`),
`test_relayout_feeds_restickify`, `test_prefetch_in_place`, `test_for_each_tile_drain`,
`test_staged_collective`, `test_context_switching_after_commit` for each family.

Existing suites: the serial list in the parent's Verification, plus `test_cost_expr_dump`,
`test_lx_context_switching`, `test_lx_extern_kernel_clobber`, `test_lx_inplace_layout`,
`test_lx_read_contracts`, `test_kernel_preparation` and `test_solver_relayout_candidates`. Device
suites, per the parent: `test_scratchpad_use`, `test_work_division`, `test_solver_relayout_e2e`,
`test_solver_auto_coarse_tiling`, `test_lx_planning_suite`.

The parent's claim for this stage is proven by G1 and G2 below, not by the unit tests.

## Parity and gates

The parent says: "`commit(graph)` inserts each edge's steps in canonical order (pinned steps
already exist), then commits divisions and offsets through `GraphEditor`. This replaces
`_push_allocation`," "`LxContextSwitchingPass` runs after commit, unchanged," and under
Verification, "comparing proto, decisions and committed plans between the stage's parent and the
change." Stage 0 records "inserted clones, relayout ops, addresses, divisions and spill reasons."

**What that record is missing.** Those five fields do not prove equality of the effects listed
under "What depends on the result". Commit 2 extends the format and bumps `SCHEMA_VERSION`. The
record is taken at two points:

- **P1**, when `_maybe_scratchpad_planning` returns: `graph_snapshot(graph)` from
  `tools/lx_planning/lx_semantic_plan.py:181`, which already covers the op list with names and
  kinds, layouts with allocation and `lx_view`, dependencies, `iteration_space_ownership`,
  `loop_info`, outputs and the registry. Added beside it: `dim_hints`, `symbols_on_tile`,
  `_work_division_chosen_with_tiling`, presence of `_loop_carry_record`, `graph_outputs` wrapper
  types, `name_to_users`, `additional_buffer_deps`, the FX node list with arguments, and spill
  reasons. P1 is after post passes, so it covers `LxContextSwitchingPass`.
- **P2**, at codegen: the `graph`, `op_specs`, `pool_size` and `arguments` that `capture_off_plan`
  records, plus `last_cost_report` with the cost model on. P2 covers elision, demotion, HBM
  pooling and OpSpec emission. `capture_off_plan` itself cannot be used under the default config
  because of its `lx_extended_ownership` check, so the harness calls `graph_snapshot` and `freeze`
  from its own wrapper of `SpyreKernel.codegen_kernel`. Stage 0's driver stops after
  `CustomPreSchedulingPasses.__call__`, so commit 2 adds `compile_model(stop_at="codegen")`. No run
  has shown a `fake` compile reaching codegen. A model that does not gets P2 on the card only.

The commit hook moves from `_push_allocation` to `commit(graph)` by wrapping the `lower_commit` it
calls, so the `CommitRecord` snapshot stays before the post passes, where the old hook takes it. It
calls `snapshot_commit` with `allocator="plan_allocation"` and
`spill_reasons=result.spill_reasons`, and its target joins `OPTIONAL_TARGETS`, because the base has
no `commit`. The `_push_allocation` hook stays for the old path.

**Which corpus entries can run commit.** Unit-test solves, the synthetic graphs and capture pickles
cannot: they hold no IR (`tests/inductor/synthetic_cooptimization_graphs.py`,
`probes-cpsat-graph/README.md`). The `fake` zoo at `ZOO_CELLS`, `matrix` and `failure` can, by
compiling again, with no card. `unit-device` suites can, on a card. `zoo` and `CARD_ONLY` models are
compared at card calibration, Stages 9 and 12.

So "the whole corpus" means every entry that reaches a live `GraphLowering`. The PR-tip gate is
Stage 0's row for this stage: all tiers including `matrix`, through `lx_gate.sh` with HEAD's harness
over the base tree. A commit-level gate is `intree` and the fixed set. `fake` proves parity, not
production plans, because fake inputs use default layouts. Determinism is one worker, seed 0 and a
deterministic budget, through Stage 0's patch of `SolveSettings.from_config`. The PR reports hits per
coverage feature, including the five restickify features.

**The gates.**

- **G0, default path untouched.** The Stage 0 comparison between base and change, default config:
  protos, decisions and P1 and P2 records all equal. This proves the extraction inert.
- **G1, shadow: commit against `_push_allocation` on the same decisions, one tree.** For each
  usable entry, compile twice. Run A is unmodified. Run B patches `_post_solve` to a no-op and
  `_push_allocation` to `build_graph`, then `decode_todays_plan`, then `read_commit_plan` and
  `lower_commit`. `commit_shadow` is a harness monkeypatch, never production code. The adapter and
  `build_graph` run before any edit, because `_post_solve` has been disarmed. Both runs keep the
  allocator's own post passes. This compares against today's real solvers, including the greedy
  replan, and answers the stage's question literally. It stays runnable until Stage 13. B's spill
  reasons come from its `CommitResult`.
- **G2, tree to tree.** On the `ortools` and `sa` cells: base (bridge) against change (commit), the
  same comparison, at both configs and in committed mode.
- **G3, device suites** above, on a card with each family selected, by dispatch of Stage 9's
  `lx_planner_compare.yaml` with `SPYRE_LX_ROUTE_TO_FAMILIES=1`.
- **G4,** `test_optimizer_reads_no_solver_records`.
- **G5, cost dump.** On the fixed set, `ortools` and `sa` cells against the old allocators' records
  on the same decisions: the same keys, and equal `bindings`, `bundles`, `divisions`,
  `objective_ns` and each relayout term's `value_ns`.

**Pass and normal form.** P1 and P2 must be equal, names included, as in `lx_semantic_plan`
("an unstable name stops equality"). The harness applies one normal form, and only this one: when
one source fires two or more relayout groups, a run of adjacent relayout clones of that source
compares as a set, with names mapped by `(source, consumers, destination view)`. The PR reports how
many solves needed it. If none did, the normal form is removed before merge.

**A difference** is fixed unless it is one of these, each listed by rule in the PR:

1. An `sa` cell against an old `simulated_annealing` arm: the family brackets an extern kernel and
   today's allocator does not. This is Deviation 7. Old SA arms carry Stage 9's baseline patch, so it
   shows only on an unpatched arm. G1 and G2 do not meet it: both sides of each run the same passes.
2. Spill reason text: tick numbers in the committed-mode "no room" string. It is debug text only.
3. Relayout clone order, as above.
4. Cost dump text: `env.solver`, `env.allocator`, and a relayout term's `expr`, which prints
   `SplitKeyedCharge` where today's prints `RelayoutCharge`.

Anything else blocks the merge.

## Risks

| risk | caught by |
|---|---|
| A clone's name or position shifts because record order differs. | G1 and G2 compare names and op order. `test_order_*`. |
| An op object is held across `apply_tilings` and goes stale. | Names only in `CommitPlan`. G1 under `AUTO_COARSE_TILING=1`. |
| A relayout source's or input's view differs in slot spelling from today's. | `source_view` and `prefetch_view` carried from the solve; `lx_view` in P1, OpSpecs in P2. |
| An offset or footprint is used in the wrong unit. | `test_read_plan_scales_units`; `allocation` in P1. |
| Metadata is snapshotted before divisions or tilings land. | `test_lower_order`; `iteration_space_ownership` and `loop_info` in P1. |
| A step is lowered twice, or a pinned op re-inserted. | `test_pinned_restickify_*`; op list in P1. `materialize_lx_relayouts` raises on a second run. |
| Committed mode overwrites anchored owners. | `test_committed_mode_skips_division_commit`; G1 on `co_optimizing_lx_planning=False`. |
| A staged output overlaps a live box. | Step 6 raises before any address lands. `test_staged_overlap_raises_before_placement`. |
| Not verified: `live_boxes` may measure a staged node larger than the `partition_footprint` its gap was found with, so step 6 raises on a good plan; and a resident input's span from `prefetch_view` may differ from its prefetch division's `per_core_size`, which feeds only staging's gap search. | G1 and G2 on models with staging. |
| The post passes are lost with the bridge. | `test_commit_runs_post_passes_after_lower`; `test_context_switching_after_commit`; P1. |
| Failure after tilings is treated as recoverable. | `plan_allocation` contract in Consumes; a test that `Unsupported` from step 4 propagates. |
| #5695 lands first and adds deferred divisions. | It changes `commit_divisions`, which this stage extracts, so both paths inherit it. Rebase, re-record. It also adds "after the solve, for ops the solve committed no split to", which belongs in `commit_chosen_divisions`. |
| `scratchpad_planning` runs under `_flat_operations` (`passes.py:447`), so `graph.operations` may be an `OpsView`, where a rebind raises (`passes.py:472`). | Only the editor's and `coarse_tile`'s existing index, insert and remove calls are used. `tests/inductor` loop-node tests in the suites. |
| The FX cache reuses stale compiles after a commit change. | `test_commit_module_is_in_pass_sources`. |
| G1 proves nothing on a path the corpus never takes, such as staging or same-tile relayouts (`lx_same_tile_group_relayout` is off by default). | The PR reports, per branch of `push_one_placement`, how many corpus hits it had. A zero-hit branch needs a device test. Same-tile relayout gets one G1 run with the flag on. |

## Deviations from the parent plan

1. **The commit order.** The plan says insert steps, then commit divisions and offsets. The code
   needs tilings and copy-outs first (`allocator.py:2902`: divisions run "on the tiled graph"),
   divisions next, offsets and clones after them (clones copy `work_slices`,
   `graph_editor.py:285-290`), and relayouts last (`lx_relayout.py:2076-2081` raises unless the
   address is already committed). Resolution: the ten-step order in Design.
2. **A `COPY_OUT` step cannot be inserted per edge or through `GraphEditor`.** `apply_tilings`
   takes specs only, and `_plan_tiling_propagation` decides copy-outs itself, per producer
   (`wsr/coarse_tile.py:590`). Resolution: lower tilings, then verify the `COPY_OUT` steps against
   `propagation.kind`. Accumulator, fill and combine ops have no step and need none.
3. **There is no canonical order.** The plan says commit inserts steps in canonical order, and
   `STEP_KINDS[StepKind.RELAYOUT].order` (2) follows `STEP_KINDS[StepKind.RESTICKIFY].order` (1).
   A relayout feeding a restickify is legal (`allocator.py:1513-1521`), though the only evidence
   for it is a mocked test. Resolution: list order of `DataEdge.steps` is authoritative and
   physical, and commit sorts nothing.
4. **Commit replaces more than `_push_allocation`.** It also replaces
   `_finalize_lx_relayout_allocation`, all of `_post_solve` and `_get_spill_reasons`. That includes
   `_stage_collective_outputs`, a post-solve residency change the plan does not mention. Resolution:
   port it unchanged, run it after `check_decisions` as a named exception, and judge `boxes_fit_lx`
   again at step 6, through `no_overlap(graph)`.
5. **`_push_allocation` is not deleted here.** The default stays on the old allocators until
   Stage 12, and Stage 13's list covers it only through their classes. Resolution: "replaces" holds
   on the new path. Stage 13 deletes the wrappers.
6. **An unpinned `RESTICKIFY` step is not lowered in Stage 10.** The plan implies it. Stick pins
   make it unreachable and any lowering untestable. Resolution: raise. Stage 11b changes the raise,
   and no stub is left behind.
7. **"`LxContextSwitchingPass` runs after commit, unchanged"** changes behaviour for SA, which has
   no post passes today (`allocator.py:5433-5435`). This is a real bug on main, reachable only with
   `LAYOUT_SOLVER=simulated_annealing`, and this plan assumes no upstream fix. Resolution: `commit`
   runs the pass for every family. SA plans gain brackets where a buffer straddles a risky extern
   kernel, which is a correctness gain and a listed parity exception.
8. **`Family` gains `dump_tables(encoding, solution)`.** The parent's protocol has only `encode`,
   `solve` and `read_back`, and `commit(graph)` sees neither the family nor its `Encoding`. The
   full `cost_expr_record` needs the family's menus and its chosen entries, which only the family
   can read out of its own native model. Resolution: the protocol gains the method, and `optimize`,
   which holds the accepted family, `Encoding` and `Solution`, writes the dump before it returns.
   A family whose tables carry `cost_expr=None` writes no line.

## Open questions

None.

## Out of scope

- Lifting stick pins, realizing unpinned restickifies, layouts decided after LX planning: Stage 11.
- The default flip and side-by-side CI comparison: Stage 12.
- Deleting `_push_allocation`, the allocators and the wrappers from this stage: Stage 13.
- The rules themselves, including whether `check_decisions` accepts a plan: Stage 3.
- `optimize`'s failure and fallback flow, and the greedy re-plan: Stage 9.
- `LxContextSwitchingPass`, `GraphEditor`'s clone mechanics, `materialize_lx_relayouts` and
  `coarse_tile`: reused unchanged.
- Fixing the output clone's position before its producer. It is preserved for parity. If it is worth
  changing, that is a separate torch-spyre PR with its own baseline.
