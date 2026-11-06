# LX allocator, Stage 11: Sticks in the solve

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0-10, plus torch-spyre#5682 Step 1, Step 2 items 1-5 and 7, and the issue's three
direct tests (see "Relation to #5682"). Needed by: step 12c of Stage 12. Stages 12 and 13 do not
depend on this stage, and it lands before or after them.

## Goal

With `config.lx_sticks_in_solve` on and a family selected, one solve chooses each op's stick from a
pruned form of `op.layouts`, and with it every restickify, jointly with division, residency and
placement. The stick is decided before `finalize_layouts`. The existing stick-layout passes then run
unchanged on the solved sticks, and the plan is committed afterwards. All five parent-plan
requirements hold.

What stays unchanged: the default path, `propagate_spyre_tensor_layouts`, the beam, `LX_PLANNING=0`,
graph-input layouts and `elide_proven_read_copies`. With the switch off, every Stage 0 recording is
identical. The switch stays off by default until step 12c, which Stage 12 owns.

This stage lands as two PRs, 11a and 11b, because their gates contradict each other (D2).

## Today

### The pipeline

`CustomPreSchedulingPasses.__init__` (`torch_spyre/_inductor/passes.py:508-581`) is one static list.
The beam ends the stick section at `:543`. The issue's 12 passes follow, unchanged, at `:544-574`,
then `_maybe_scratchpad_planning` (`:577`) and `elide_proven_read_copies` (`:580`). Gating lives in
the `_maybe_*` wrappers (`:351-356`). `_uuid` hashes their `@_runs` sources (`passes.py:164-171`),
but nothing calls `CustomPreSchedulingPasses.uuid()` for this pipeline, which runs from a
monkeypatch (`patches.py:213-229`). Source files reach the FX key only through Stage 9's
`optimizer_pass_sources()`.

### What the beam hands downstream

The beam's whole output is `op.committed_stl` (`optimize_restickify.py:852-856`), plus
`committed_stl` on each graph input's `InputBuffer` (`:740-741`). `reorder_nonstick_dims` rewrites
it (`nonstick_dim_order.py:903,1014,1039`), and `finalize_layouts` reads it and deletes `layouts`,
`restick_cost_fn` and `committed_stl` (`insert_restickify.py:621-634`).

`op.restick_cost_fn` is a complete edge oracle. `required_input_stls(out_stl)` gives what a
candidate needs on each input (`optimize_restickify.py:180,288,367`), and
`EdgeCostMap.layout(in_stl, target)` returns `None`, `INFEASIBLE`, or the restickify's target STL
(`:148-160`, `compute_restickify_needed`, `pass_utils.py:2619`). `AllSameNode` adds co-output
equality: two mutation writers of one buffer must agree exactly (`optimize_restickify.py:279-285`).

### Candidates, measured

A device-free compile of the snapshot (fake spyre tensors, stopped right after the beam, `/dev/vfio`
refused by an audit hook) gives the first real census of `op.layouts`. It covers 29 of the 32 models
in `probes-cpsat-graph/zoo.py`. Three sdpa models raised "No RuntimeContext". It used a `_C.so`
built 2026-10-18 from another checkout, FP16 inputs in default layout, and no Granite.

| measure | `op.layouts` as is | pruned (see Design) |
|---|---|---|
| ops with layouts | 305 | 305 |
| ops with more than one choice | 139 (46%) | 139 (46%) |
| choices per op: mean / max | 1.82 / 4 | 1.46 / 2 |
| input edges with more than one (source, required) pair | 249 of 405 | 249 of 405 |
| pairs per input edge, mean | 3.56 | 2.19 |
| pairs: compatible / restickify / infeasible | 744 / 644 / 52 | 522 / 330 / 36 |

- **The beam placed zero restickifies** on all 29 models, and took list position 0 on all 139 free
  ops. So the zoo alone says nothing about removing restickifies. `aten.clone` ops are not counted:
  they are `AnyInNode`, and one can still be a restickify by coordinates
  (`propagate_layouts.py:868-878`).
- **`op.layouts` is not a stick domain.** 42 ops have fewer distinct sticks than layouts.
- **Footprint moves a lot.** On 99 free ops the candidates' device sizes differ, by 64x at the
  median and the maximum: sparse layouts whose stick is constant 0
  (`_compatible_output_stick_exprs`, `propagate_layouts.py:351-380`).
- **Element arrangement and dtype never vary** within one op's candidates, on these FP16 graphs.
- **Free ops clump** into 35 connected regions, at most 4 per model. Matmul
  (`propagate_layouts.py:1615`), graph inputs, constants and `MultiOutput` have one candidate, so a
  matmul's choice is on its input edges.

**A restickify can be planned before its op exists.** A second device-free probe used SwiGLU through
`F.linear` (the shapes of Stage 0's `swiglu_transposed_128x512x1536`), where the beam places three
restickifies. Under the beam's stamps it built each as a detached `ComputedBuffer` from the
`Pointwise` that `lower_restickify` builds (`lowering.py:1665-1694`), plus a detached copy of the
consumer through `make_per_edge_inner_fn` (`insert_restickify.py:304`), both only in
`name_to_buffer`. `enumerate_work_division_candidates` gave each consumer 68 entries and each
restickify 25, equal entry for entry to the real ops' menus after `insert_restickify_padding`, and
the reads matched except for the new name. That is one model, and it compared menus and read deps
only.

### Restickify pricing, time and hazard

A restickify is an ordinary op with its own index in `graph.operations`, so its own tick
(`calculate_liveness`, `scratchpad/utils.py:77`). `predict_bundles` prices it through
`extract_op_features` inside its estimated bundle (`cost_model.py:3104`,
`scratchpad/allocator.py:3307-3319`). The transport law is `_transport_dma_excess_ns`
(`cost_model.py:2436`), and its geometry exists only for DL16 (`cost_features.py:637-638`). The tag
`"restickify"` comes from the read and write indices (`_hbm_pattern`, `cost_features.py:945`), so a
plain restickify, whose two indices are equal, gets `""`. Only a transposing copy gets the tag.

The joint path relies on the per-candidate residency gate, and `_restickify_read_is_core_local` is
the local-read proof, which also accepts a relayout's destination view (`allocator.py:1513-1521`).
CP-SAT's menu tilings skip restickifies (`allocator.py:4008-4009`). `build_tiling_space` has no such
test (`wsr/enumerate_tilings.py:409-423`, `:437-469`), so SA's exclusion is Stage 8's filter in
`SAFamily.encode`.

### Surprises

1. `elide_proven_read_copies` removes for_each_tile staging copies of graph inputs
   (`wsr/for_each_tile_lowering.py:2799,3272`). Sticks are one condition of its proof
   (`read_copy_elision.py:400-426`), not its purpose (D4).
2. With `AUTO_COARSE_TILING=1` the allocator hands CP-SAT no cost expression
   (`allocator.py:3275-3300`), so a restickify could not be priced there. SA keeps its priced
   expression (D5).
3. Passes can raise on a stick the beam would never choose (`nonstick_dim_order.py:802`,
   `padding.py:697`, `insert_restickify.py:477`, `enforce_indirect_access_layout.py:403`).
4. `reorder_nonstick_dims` can change more than dim order: flat-M projection rebuilds the STL
   (`nonstick_dim_order.py:298-306`) and a gather re-tile queues a real copy (`:897-902`). So a
   final layout is often not an element of `op.layouts`.
5. `dedup_and_promote_constants` needs no finalized layout (`dedup_constants.py:32-41`), yet it
   cannot run before `finalize_layouts`. It deletes a buffer name (`:179-182`) that a consumer's
   `EdgeCostMap.dep` still holds, and `finalize_layouts` calls `graph.get_buffer` on that name
   (`insert_restickify.py:713`).
6. Span floors fail silently. `_work_division_span_min_splits` is written only by
   `span_reduction_pass` (`work_division.py:2012`) and read with an empty default (`:203`). A solve
   ahead of that pass would have no floor and would not raise: the risk is a wrong plan, not an
   error. This is why #5682 Step 1 is a hard prerequisite.

## Design

### Where the solve goes

The solve has to precede `finalize_layouts`, because that is where `layouts` and `restick_cost_fn`
are deleted. The list keeps its order. One entry is added, and the old planning entry becomes commit
when a plan is held:

```python
split_multi_ops, propagate_spyre_tensor_layouts, validate_ops,
optimize_restickify_locations,     # unchanged: seed and fallback
_maybe_solve_sticks,               # NEW: holds a plan, or changes nothing
reorder_nonstick_dims, finalize_layouts, insert_restickify,
validate_no_restickify_on_mutation_targets, enforce_indirect_access_layout,
reorder_nonstick_dims_mutation, insert_post_mutation_restickify,
insert_restickify_padding, dedup_and_promote_constants,
_maybe_coarse_tile_span_overflow,  # returns at once when a plan is held
span_reduction,                    # no-op on a held plan (#5682 Step 1)
_distribute_work,                  # deferred on a held plan, in full otherwise
_maybe_scratchpad_planning,        # held plan: verify and commit. No plan: as today
elide_proven_read_copies,
```

Nothing moves for `LX_PLANNING=0` or the fallback path, so the pipeline never needs two orders, and
a pass #5682 Step 2 has moved past `_maybe_scratchpad_planning` is still after the solve.

### Pass by pass

`propagate_spyre_tensor_layouts`, `validate_ops` and the beam run unchanged before the solve.
"Reservation" is what the solve must know up front, so the pass stays inside the plan.

| pass | under this stage | reservation or change |
|---|---|---|
| `reorder_nonstick_dims` | After the solve, so "final" holds. Code unchanged | Dry-run preview per candidate |
| `finalize_layouts` | After the solve, code unchanged. Same `edge.layout` call the solve used | `verify_realized` asserts they agree |
| `insert_restickify` | After the solve. Loop metadata moves to a pure `restickify_loop_info` that `shadow_ops` shares. Records `restickify_realized` | A shadow op or a step. A row with `occurrence != 0` and a transferred advance is dropped (`insert_restickify.py:474-481`) |
| `validate_no_restickify_on_mutation_targets` | Stays as the check, and also becomes a banned row | Rows removed |
| `enforce_indirect_access_layout` | After the solve, unchanged | `padded_entry_output_stl` previews the gather pad. `_pad_output_for_stick_aligned_split` (`enforce_indirect_access_layout.py:71-98`) changes the buffer size and enables no split, so the preview needs only the size. A scatter whose alignment fails in preview loses that candidate (`align_tensors_pure`) |
| `reorder_nonstick_dims_mutation` | After the solve, unchanged | A copy on the beam's sticks is a shadow op or opts the graph out. A non-beam candidate that would queue one is dropped |
| `insert_post_mutation_restickify` | Unchanged. The stick was fixed in `propagate_layouts` (`:3450`), so the op has one candidate | #5682 Step 2 reservation, or the graph opts out |
| `insert_restickify_padding` | After the solve, unchanged | `restickify_padding_plan` per row. Unsupported cases drop the row |
| `dedup_and_promote_constants` | Stays after `finalize_layouts` (Surprise 5). Constants hold no LX (`ir.py:420`, `allocator.py:1162-1163`), so `rebind_graph` drops those nodes | None |
| `_maybe_coarse_tile_span_overflow` | Must be in the solve (#5682 Step 1). Until then `solve_decides_sticks` opts out on overflow groups. Returns at once on a held plan | A candidate with no legal menu is dropped |
| `span_reduction` | Floors inside the menus, per candidate (#5682 Step 1). No interim (Surprise 6) | Menu per candidate |
| `_distribute_work` | Deferred on a held plan ("The #5695 predicate") | Menu of one for pinned ops |
| `_maybe_scratchpad_planning` | Held plan: `verify_realized`, then `commit(graph)`. No plan: today's path with beam sticks | None |
| `elide_proven_read_copies` | Stays (D4). A solved stick can decline a proof that passes today (Risk R11) | None |

### Relation to #5682

Stage 11 is the issue's Step 3, and it depends on Step 1 and most of Step 2.

| issue item | here |
|---|---|
| Step 1, all three | Prerequisite, with the overflow item softly: the opt-out above is its interim. Stage 11 turns #5695's predicate true on a held plan, and evaluates floors per candidate |
| Step 2 padding, indirect access | Reservations assumed. **Generalized here** per row and candidate |
| Step 2 post-mutation, mutation reorder | Assumed done. Not stick-dependent, or the candidate is dropped |
| Step 2 mutation-target validation | The constraint is done here, as a banned row |
| Step 2 item 6 (`reorder_nonstick_dims`, `finalize_layouts`, `insert_restickify`) | **Done here, PR 11a, on the graph layer only.** The issue's "virtual op with its own buffer, split menu and price" is a shadow op |
| Step 2 item 7, dedup | Needs no finalized layout, but stays after `finalize_layouts` (Surprise 5), which breaks the issue's expectation |
| Direct tests of `reorder_nonstick_dims`, `reorder_nonstick_dims_mutation`, `insert_post_mutation_restickify` | Prerequisite, ahead of Step 2 as the issue says. Stage 0 leaves them to the issue, and 11a's preview rests on them |
| Step 3 | **Done here, PR 11b**, from a pruned domain. Stage 6's literal is made live, `CostModel.restickify_cost` is the term, and the beam stays |

**Which goes first.** Step 1, then Step 2 items 1-5 and 7, on `main`, in the issue's order. Then 11a
does item 6, which should not land in `allocator.py`, since Stage 13 would delete it, and 11b does
Step 3. What Stage 11 consumes from those items is span floors per candidate, `divide_deferred_op`
and the reservation functions. Step 2 moves passes behind the solve while this stage moves the solve
ahead of them: two routes to one order. An item landing after Stages 5-7 changes `CpSatLayoutSolver`
with no owner for the encoder port, and "Stages 5-7 port their reservations" holds only if those
items land first. So a half-landed Step 2 during Stages 5-10 is the worst case (Open question 1).

### The stick domain

A candidate is one element of `op.layouts`, an STL. It cannot be an int: factorized sticks span two
device dims, and sticks can be constant 0. So keys are the solve's domain, dims stay the rule
language, and each candidate carries its dims. The domain is `op.layouts` and nothing else: dim
order can move with a candidate's stick, but the solve does not choose dim order or padding.
`stick_domain` is built in this order:

1. Keep the beam's `committed_stl`, always.
2. Walk `op.layouts` in order. Keep one layout per distinct stick, where the stick is
   `device_coordinates(stl, write_dep, None)[-1]`.
3. Drop a non-beam layout whose `get_device_size_in_bytes` exceeds `STICK_FOOTPRINT_SLACK` (2) times
   the smallest. This removes the 64x sparse layouts.
4. Cap at `config.lx_stick_domain_limit` (default 2), beam first.
5. Drop a non-beam candidate whose preview raises, or queues a copy the beam's does not.

Rules 1-3 give the "pruned" column above, and the cap never bites on the zoo. Stage 3 has no check
for the offset-free stick, matmul outer-dim or element-arrangement rules
(`propagate_layouts.py:383`, `:1249`, `:1715`). They hold by construction, because every candidate
comes from `op.layouts`.

Stick pins stay, as a one-element domain, on:

- graph inputs, and ops tagged `_restickify_plan`;
- ops with `loop_info` or in a `prescribed_regions` region, and holders of
  `_read_copy_elision_record` (Decisions, "Loops");
- conv, exx2, topk, keep_by_index and layernormnorm ops (`propagate_layouts.py:2323-2345`): Stage 3
  has no stick table for them, so `check_report` could not judge a free one;
- every op when `config.auto_coarse_tiling` is on (D5).

**Growth.** Node tables gain a stick column, about 1.46x rows on average, and edge relations are
evaluated per pair, 2.19x. Relayout rows are built only for the beam's pair, which bounds the term
that dominates prepare time today. Projected, not measured: at default config enumeration (2-3% of
planning) grows 3%, model build and cost lowering (50.5 s of 174 s on the zoo) grows by 23 s, and
CP-SAT search has no direct data, but relayouts, the same shape of optional step, raised it 5.2x, so
the bracket is 1.5x to 5x. Zoo planning moves from 174 s to roughly 230-460 s. The 35 regions and
the beam's hint argue for the low end, but that is an argument, not a result.

**The binding limit is Granite.** Graphs 2 and 3 already search for 46.4 and 56.8 s against the 60 s
limit, so any factor above 1.1-1.3 hits it. That returns an incumbent at least as good as the hint,
but it is the failure the issue calls "only shows up as a worse plan".

### Graph-layer additions

No table indices reach the graph. Candidates and rows are named by `stick_key`, so a later
non-enumerating encoder can replace the rows with constraints. All of this is in `<lx>/sticks.py`
(see Changes), except the fields on the graph types.

```python
STICK_FOOTPRINT_SLACK = 2
EdgeKey = tuple[str, str, sympy.Expr, int]   # consumer, buffer, dep.index, occurrence

@dataclass(frozen=True)
class StickCandidate:
    key: str                       # stick_key(propagated), never a position
    propagated: SpyreTensorLayout  # the element of op.layouts; what commit writes back
    reordered: SpyreTensorLayout   # after the reorder preview: what finalize_layouts reads
    layout: FixedTiledLayout       # also after the gather-pad and input-pad previews
    dims: tuple[int, ...]          # stick_dims(layout) or (SPARSE_STICK,)
    beam: bool
    facts: PortFact                # facts_for(graph, out port, layout)
@dataclass(frozen=True)
class StickPairRow:
    src: str                              # StickCandidate.key of the producer
    dst: str                              # StickCandidate.key of the consumer
    read_layout: FixedTiledLayout         # what the in port reads under this pair
    restickify: FixedTiledLayout | None   # the step's target; None when the sticks agree
    crosses_frame: bool                   # a step, or a consumer that is itself a restickify
    step_bytes: int                       # padded footprint of the step, 0 without one
    facts: PortFact                       # facts_for(graph, in port, read_layout)
    @property
    def key(self) -> str: ...             # src and dst joined: the row of Sym.restick(e, row)
@dataclass(frozen=True)
class StickRelation:
    rows: tuple[StickPairRow, ...]        # absent pair means banned
    def row(self, src: str, dst: str) -> StickPairRow | None: ...
@dataclass(frozen=True)
class ShadowRow:
    key: EdgeKey
    target: FixedTiledLayout              # the realized op's layout, padded
    source_layout: FixedTiledLayout       # the source after any input pad

class OutPort:   # added
    candidates: tuple[StickCandidate, ...] = ()
class DataEdge:  # added; None when both ends are pinned, so Stage 6's path is untouched
    sticks: StickRelation | None = None
class SpyreOptimizationGraph:  # added
    restickify_realized: dict[EdgeKey, str]   # set by rebind_graph

def stick_key(stl: SpyreTensorLayout) -> str: ...   # device_size and stride_map, symbol-safe
def stick_domain(op: Operation, limit: int) -> tuple[SpyreTensorLayout, ...]: ...
def build_stick_candidates(ir: GraphLowering, limit: int
                           ) -> dict[str, tuple[StickCandidate, ...]]: ...
def build_stick_relations(ir: GraphLowering, candidates: Mapping[str, tuple[StickCandidate, ...]]
                          ) -> dict[EdgeKey, StickRelation]: ...
def predict_shadow_ops(ir: GraphLowering, candidates: Mapping[str, tuple[StickCandidate, ...]],
                       relations: Mapping[EdgeKey, StickRelation]
                       ) -> tuple[ShadowRow, ...] | None: ...
@contextlib.contextmanager
def tentative_layouts(ir: GraphLowering,
                      stamps: Mapping[str, SpyreTensorLayout]) -> Iterator[None]: ...
@contextlib.contextmanager
def shadow_ops(ir: GraphLowering, rows: Sequence[ShadowRow]) -> Iterator[None]: ...
def solve_decides_sticks(ir: GraphLowering) -> bool: ...
def write_solved_sticks(ir: GraphLowering, graph: SpyreOptimizationGraph) -> None: ...
def rebind_graph(ir: GraphLowering, graph: SpyreOptimizationGraph) -> None: ...
def verify_realized(ir: GraphLowering, graph: SpyreOptimizationGraph) -> None: ...
```

**`tentative_layouts`** is how everything reads a candidate, since the planner has more than 90
`device_layout` reads. It sets `op.layout = _fixed_tiled(op.layout, stl)` on the named ops and
graph-input buffers, and restores them in `finally`. The build runs under the beam's stamps, so
`_resolve_layout`'s assertion (`work_division.py:808-822`) and `mem_usage_by_buf`'s
`FixedTiledLayout` test (`scratchpad/utils.py:256-260`) hold. For one candidate, the op is stamped
with it. For one row, the source is stamped with `read_layout`. `_input_layout_overrides` is not
enough: it does not reach `_per_core_view_on_buf`. Two caches keyed without a layout must take
`stick_key`: `_prep_for`'s `prep_cache` (`work_division.py:1166-1173`) and the
`_per_core_view_on_buf` cache (`pass_utils.py:4162`).

**`build_stick_candidates`** dry-runs `reorder_nonstick_dims` once per candidate position `i`, each
op taking its `i`-th domain element or its beam element if shorter. The pass only writes
`committed_stl`, `nonstick_deferred` and `nonstick_reorder_log`, which are saved and restored. Its
per-buffer decisions depend on that buffer's own STL: `_try_flat_m_projection` reads the sibling
only for presence and size (`nonstick_dim_order.py:166-187,377-383`). So one run gives every op's
outcome for position `i`. An `Unsupported` at `i > 0` drops all non-beam candidates at that
position, and is logged. At `i = 0` it propagates, as today. A `"copy"` `_DeferredReorder` at
`i > 0` that position 0 lacks drops that candidate.

**`build_stick_relations`** walks each consumer and candidate:
`cost_fn.required_input_stls(c.reordered)` gives the target, and for each producer candidate
`edge.layout(p.reordered, target)` gives the row, the same call on the same STLs as
`finalize_layouts` (`insert_restickify.py:744`). `INFEASIBLE` is left out. So are rows where
`restickify_padding_plan` raises, where the edge reads a mutation target into its own mutation op,
and the advancing-occurrence case. For an `AnyInNode` consumer, `crosses_frame` comes from
`is_restickify_coords` (`pass_utils.py:771`). Co-output edges of `AllSameNode` become rows requiring
`src == dst`.

The key is `EdgeKey` because `edge_occurrences` counts per (name, `dep.index`)
(`insert_restickify.py:709-761`): `x[..., :8] + x[..., 8:]` has two rows at occurrence 0.

### Shadow ops

A graph built before `insert_restickify` has no op for a beam restickify, yet the base model holds
one: a menu, `in_buffer_`, a box and a cost term. `shadow_ops` makes the IR look as it will after
`insert_restickify` under the beam's sticks, for as long as the solve runs. In order, and undone in
`finally`:

1. For each row, a detached `ComputedBuffer` from `lower_restickify`'s `Pointwise`, with
   `layout=target`, the consumer's `origins`, and `loop_info` from `restickify_loop_info`. It enters
   `ir.name_to_buffer` and never goes through `realize()`, so no buffer name is consumed. The source
   is stamped `source_layout`.
2. For each consumer, a detached copy with `data=dataclasses.replace(op.data, inner_fn=...)`,
   through `make_per_edge_inner_fn` with the swaps `insert_restickify_on_node_inputs` would build.
   It replaces the consumer in `name_to_buffer`.
3. `ir.operations[:]` becomes the list with each shadow restickify just ahead of its consumer's
   copy.

Inside the context, Stages 2, 4, 5 and 6 and `project_transport_read_copies` all read an ordinary
IR. `is_restickify_op` holds, so `classify` gives a `RestickifyNode`, and Stage 2 writes
`Step(kind=StepKind.RESTICKIFY, pinned=True, realized_by=node)` with its pin. A detached
`ComputedBuffer` passes `_is_fusable_operation` (`fusion.py:165-178`), so it bundles as the real op
does. So 11a changes no encoder, and Stage 6's raise and Stage 10's `read_commit_plan` hold
unchanged there.

**Which ops get a shadow.** Where #5682 Step 2 has moved a pass past the solve, its ops are
reservations in both arms and need nothing here. Where a fixup op is still a node in the base model:

| op a preview predicts | handling |
|---|---|
| Beam restickify with both ends pinned, so every one at limit 1 | Shadow |
| `_insert_relayout_copy` (`enforce_indirect_access_layout.py:165`): a `"copy"` `_DeferredReorder` on a non-mutation consumer, or an unaligned scatter source. One op on the same path | Shadow |
| `_insert_mutation_relayout_copy` (`nonstick_dim_order.py:1147`), or `insert_post_mutation_restickify` on a graph-input target (`insert_restickify.py:930-970`): copy-in, a retargeted mutation, copy-back | Graph opts out |
| `RestickifyPadding.clones_input` (`padding.py:1016-1040`) | Graph opts out |
| A row that moves or rewrites a read-copy record (`insert_restickify.py:523-538`, `:555-566`) | Graph opts out |

`predict_shadow_ops` returns `None` for a graph that opts out. It takes the switch-off path, so its
plan is identical by construction, and the gate lists it by rule. A real op with no shadow fails
`verify_realized`, so a miss is an error, never a silent plan.

### Lifting the pins

`build_graph(ir, target, ...)` gains `stick_candidates=None`, after Stage 2's keywords. When it is
given, a node with two or more candidates gets no `"layout"` stick pin, and its `OutPort.candidates`
is filled. A node with one gets today's pin. `"graph input"` and `"pre-pass"` pins are never lifted.
`OutPort.layout` holds the beam's candidate before the solve, and `read_back` overwrites it.

| restickify | on the graph | first exists |
|---|---|---|
| Already in the IR at the build | `RestickifyNode`, pinned step | Stage 2 |
| Beam's, both ends pinned | Shadow `RestickifyNode`, pinned step | 11a |
| Solver's, on an edge with `sticks` | `Step(kind=StepKind.RESTICKIFY, pinned=False, to_stick=dim, target=row.restickify)` from `read_back`. No node, no menu | 11b |

At limit 1 no `sticks` exist, so nothing unpinned reaches an encoder or commit.

### `plan_sticks`

```python
def plan_sticks(ir: GraphLowering, config) -> SpyreOptimizationGraph | None: ...

@_runs(stick_domain, build_stick_candidates, build_stick_relations)
def _maybe_solve_sticks(graph: GraphLowering) -> None:
    if not solve_decides_sticks(graph):
        return
    if family_key(config) is None:    # the old path
        return
    plan = plan_sticks(graph, config)
    if plan is None:
        return
    write_solved_sticks(graph, plan)
    graph.spyre_held_plan = plan
```

`plan_sticks` is the first half of `plan_allocation`. It takes `select_target(config)`, the families
from Stage 9's `select_families(config)` and `SolveSettings.from_config(config)` as
`plan_allocation` does, and validates drain plans. Then,
under the beam's stamps, it builds candidates, relations and shadow rows. Inside `shadow_ops` it
calls `build_graph(ir, target, stick_candidates=...)` and `optimize(graph, family, fallback,
settings)`, which reads the plan back and checks it but never commits. It returns `None`, leaving
every `committed_stl` as the beam wrote it, on an infeasible outcome, an opt-out, or `SolveError`
from an empty `config.fallback_solver`. Stage 13 removes that last branch, since an empty value then
raises `ValueError` when the families are selected. If Stage 13 has already landed, the branch is
not written. A `RuleViolation` propagates. `sticks.py` joins `optimizer_pass_sources()`, which is
what puts it in the FX key.

`rebind_graph`, the first step of `verify_realized`, rebinds each node's `op` by buffer name,
because the passes in between replace ops (`pass_utils.py:2882`). It moves a shadow's node to the
real op and name through `ir.restickify_realized`, which `insert_restickify_on_node_inputs` writes,
and drops dedup's constants. Commit inserts nothing. `commit(graph)` gains one step ahead of
`read_commit_plan`: `divide_deferred_op` for each deferred op the plan holds no split for, as #5695
does after a solve.

From 11b, `read_commit_plan` accepts an unpinned `RESTICKIFY` step whose key is in
`restickify_realized`, and its op takes the step's division and address in the ordinary walk. An
unrealized unpinned step still raises.

### The #5695 predicate

Stage 9 leaves `solve_decides_work_division()` false on a family. This stage makes it true on a
family exactly when `V.graph.spyre_held_plan` is set. `_maybe_solve_sticks` runs ahead of
`_distribute_work`, so nothing has been deferred when it comes back empty.

| path | `_distribute_work` | a split for an op that needs one |
|---|---|---|
| Plan held, by the family or its SA fallback | Deferred | A pinned op's comes from Stage 2's `_committed_work_split` during the build, under the stamp and never written. `commit` divides the leftovers |
| No plan: infeasible, opted out, no fallback | In full, as today | Nothing is deferred |

SA's seed takes `configs[0]` (`sa_cooptimizer.py:1125`), not the committed division, so it needs no
split. Where the predicate lives, and whether `_work_distribution_op` exposes a computing half, are
unverified: #5695 is not in the snapshot.

### Landing before or after Stages 12 and 13

Nothing here reads which of them has landed. On the old path `family_key(config)` is `None`, and
otherwise it names a family. The gates select with `LAYOUT_SOLVER=ortools` or
`sa`, which mean the same at every point, against `main` at the branch point with the switch off.
The branch in `_maybe_scratchpad_planning` sits at the head of the wrapper, today's or Stage 13's. A
compile with no plan reaches `scratchpad_planning` with beam sticks, whichever path that holds by
then. This stage uses nothing Stage 13 deletes: `check_report` rather than the Stage 3 adapter,
Stage 6's `restickify_read_is_core_local` rather than the allocator method, and Stage 8's public
`DivisionSource` and `TableRelation` rather than the aliases.

### OR-Tools encoding

Stage 5 has the stick variable: `stick_var(enc, port)` holds a private candidate index, and
`stick_is(enc, port, dim)` is the OR of the candidates whose `dims` contain `dim`, so dim-keyed rule
data meets key-valued sticks there. Stage 6 binds the restickify literal in `encode_restickifies`,
and leaves the rest here. From commit 9 this stage adds:

- **Node table.** A free node's table gains a stick column. Each row is (stick key, tiling,
  division), legality is evaluated with `work_division_context_for_op` and the data of each flag in
  `node.restrictions` under the candidate's stamp, and footprint and cost tables key on the same
  row index. With `Rule.TILES_OFF_STICK` in `graph.target.rules`, a row's tiling is checked against
  that row's candidate `dims`. Under D5 a tiled node has one candidate, so this is Stage 5's check.
  Position numbers stay inside the encoder.
- **Stick relation.** `AddAllowedAssignments([stick_u, stick_v, restick_e], rows)`. A missing pair
  is banned.
- **Forced steps.** `encode_restickifies` keeps its raise for a forced restickify with no pin only
  where `edge.sticks` is `None`.
- **Edge match.** `build_edge_tables` gains a stick key on `pairs`. The division pair relation is
  built once per row, posted under that pair's literal. A pair with no step uses the view equality.
  A pair with a step matches on the step's write view.
- **Step.** An unpinned step has no split variable. It runs on the consumer's division projected
  onto its write, and a division that cannot be projected is a banned row. This stage owns
  `restick_resident_e`, with `restick_resident_e => restick_e`, and the step's optional
  `PlacementBox`, the one step with a box. A restickify then never takes a split of its own when its
  stick is free, which avoids a second menu per pair.
- **Co-output and hint.** `src == dst` rows give equality. Every `stick_v` is hinted to its beam
  key.
- **`read_back`.** Each free node's value picks its candidate and writes `OutPort.layout`. After the
  relayout decode, each chosen row with a `restickify` goes through `add_step` as an unpinned
  `RESTICKIFY` step, with `resident`, `offset` and `footprint` from its box. The step kind is in
  `graph.target.steps`, as for every step an encoder inserts.

### SA encoding

`AnnealModel` keeps its shape. `domains[node]` becomes a `_StickedDivisions(DivisionSource)` whose
entries are (stick key, `DivisionConfig`) from one inner source per candidate, the beam's first. So
a division flip can land on another stick, as the parent plan says. `DivisionConfig`'s key gains the
stick key. `relations[gate]` is a `TableRelation` over those entries, with the restickify forced by
the row. The seed is the beam's stick, so postmath's point stands: a valid seed gives a valid plan.
`SAFamily.encode` stops refusing an unpinned stick.

SA has no relayouts. A state whose row `crosses_frame` over a source sliced across cores with no
local proof is legal only with that source spilled: the packer treats the source as non-placeable in
that state, so nothing is penalized. A solved step's box joins `order` only in states where the row
has a step. `score` needs nothing new once `CostModel.objective` carries `Sym.stick` and
`Sym.restick`, since `priced_objective` wraps it.

### Requirement 1: `propagate_layouts` accepts a solved stick

No change to `propagate_layouts`. The seam is `committed_stl`, the same one the beam uses.
`write_solved_sticks` sets `op.committed_stl = candidate.propagated`, asserting membership in
`op.layouts`, and never writes a graph input. `reorder_nonstick_dims` then re-derives the `layout`
the preview computed.

### Requirement 2: codegen accepts layouts decided after LX planning

No codegen file changes. Codegen reads `op.get_layout()` and has never cared when it was set. The
real requirement is that planning splits in two: solve and read-back run before the stick passes,
commit after them, when `commit_divisions` has its `FixedTiledLayout`s. `verify_realized` is the
proof, and any failure raises `RuleViolation`. After `rebind_graph`:

1. The ops the passes inserted equal the shadow rows plus the solved `RESTICKIFY` steps, by
   `EdgeKey`, with equal target layouts.
2. Each op's final `FixedTiledLayout` equals its candidate's `layout`, and each shadow's its
   realized op's.
3. Each resident buffer's final per-core size is at most what the plan reserved.
4. `check_report(graph)` on the rebound graph, with span floors re-read from the final ops, reports
   no violation or error, as Stage 9 requires after a solve.

### Requirement 3: the cross-frame hazard holds

Stage 6 posts no joint hazard constraint: its gate is the rule, for a read view fixed per node. With
free sticks the read view changes per row, so the constraint is this stage's, tagged
`"cross_frame"`, the name of the cross-frame hazard's check in `ALWAYS`. For each row with
`crosses_frame`: `(stick_u == src) and (stick_v == dst) and resident_u and cores_u > 1` implies a
proven local read for the (division_u, division_v) pair, or, on the beam's pair only, a relayout.
The proof is Stage 6's `restickify_read_is_core_local` under stamps, as a table over the two
divisions. `crosses_frame` is needed because a clone can be a restickify without any step.

Non-beam pairs have no relayout option, so a relayout ahead of a restickify, which is legal
(`allocator.py:1513-1521`), arises only on the beam's pair. List order stays physical. A restickify
inside a nest with its value cut cannot arise on a free edge: ops with `loop_info` stay pinned, and
free sticks mean no menu tilings.

### Requirement 4: a restickify's compute time is accounted for

```python
def restickify_cost_features(src: FixedTiledLayout, dst: FixedTiledLayout,
                             read: MemoryDep, write: MemoryDep,
                             it_space: Mapping[sympy.Symbol, sympy.Expr],
                             work_slices: Mapping[sympy.Symbol, Any],
                             in_is_lx: Any, out_is_lx: Any) -> OpFeatures: ...

class CostModel:
    def restickify_cost(self, edge: DataEdge, row: StickPairRow) -> sympy.Expr: ...
```

**Price.** A shadow op needs none of this: it is a `ComputedBuffer` in the priced list, so the
objective stays equal. `restickify_cost_features` is for 11b's rows, which have no op. It factors
out what `_transport_read_geometry` and `_hbm_pattern` read from an op, sets
`transport_compute_read=False`, and derives `hbm_pattern` as `_hbm_pattern` does, so a plain
restickify gets `""`. A step in the plan is therefore priced by the existing law, with no new
calibration. An unpinned row is priced standalone, as `Sym.restick(e, row.key)` times
`predict_ops([features])`, which ignores bundle sharing and derates (Risk R5). Non-DL16 steps get
the byte charge only.

**Tick.** From 11b, `build_graph` gives a consumer one extra tick before it for each edge with any
restickify row, in `edge_occurrences` order, matching `insert_restickify_on_node_inputs`
(`insert_restickify.py:386-390`). So ticks do not depend on a decision, and an unused slot is empty
time. The source stays live through its step. A resident step's output lives from its slot to the
consumer. At limit 1 there are no such edges, and a shadow takes a tick as any node does.

### Requirement 5: layout-derived port facts become conditional

```python
def facts_for(graph: SpyreOptimizationGraph, port: Port, layout: FixedTiledLayout) -> PortFact:
    return derive_port_facts(port, layout) & graph.target.ops.ops[type(port.node)].facts
```

`port.facts` remains the value under the current layout. `facts_for` is a module function in
`nodes.py` beside `current_layout`, wrapping Stage 2's pure `derive_port_facts(port, layout)` with
the same `OpConfig.facts` mask Stage 2 applies to `port.facts`, so a fact the card's op config does
not allow is absent under every candidate and never becomes conditional. It is evaluated per
candidate on out ports and per row on in ports. `build_graph` fills each `StickCandidate.facts` and
`StickPairRow.facts` from it, with `dataclasses.replace`, when it attaches them to a port and an
edge, because only then is the node type known.

| fact or data | source | under this stage |
|---|---|---|
| `QFP8WT`, `STAGGERED_EA` | element arrangement | Per candidate and row. Uniform on the census, so normally unconditional |
| rule data for `QFP8WT` | device coords (`_qfp8wt_unsplittable_coords`) | Per candidate, as the menu is |
| `OVERLAPPING_READ` | `td.device_coords` (`work_division_constraints.py:1151-1172`) | Per row (D7) |
| `DIRECT_READ_SOURCE` | the saved record and the graph input's layout | Unconditional. Holders stay pinned |
| `splits_while_stuck` data, `aligned_ownership_split_domains`, `restickify_padding_blocked_vars` | layouts | Per candidate, through the menu |

Posting: `post(enc, subject, rule, expr, when=literal)`, with the candidate's or the row's
literal.
A fact with one value on all of a port's candidates is posted unconditionally, so a pinned or
uniform graph produces no new constraints. In SA these prune per entry of `_StickedDivisions` and
per relation entry.

### Padding as a pure function

```python
@dataclass(frozen=True)
class RestickifyPadding:
    output: FixedTiledLayout     # the step's padded output layout
    input: FixedTiledLayout      # the source layout after any input padding
    clones_input: bool           # a graph input needs a padded identity clone
    added_bytes: int

def restickify_padding_plan(src: FixedTiledLayout, dst: FixedTiledLayout,
                            read: MemoryDep, write: MemoryDep) -> RestickifyPadding: ...
```

`_pad_restickify_input` already decides before it mutates (`padding.py:988-1011`).
`_pad_restickify_output` is split the same way. The pass calls the plan function and applies it, so
the preview cannot drift. A pad on the source applies only if that row is chosen, so a source's
candidate reserves the largest pad among its rows: an upper bound where sticks are free, exact where
they are pinned.

### The beam stays

The beam keeps four jobs: it is the only stick choice under `LX_PLANNING=0`, the hint for CP-SAT,
the seed for SA, and the fallback when the solve returns nothing. So `_matmul_layouts`,
`_multi_arg_pointwise_layouts` and `_check_stick_all_same` stay, as Stage 13 keeps them.

| outcome of `plan_sticks` | effect |
|---|---|
| proven infeasible | Warning naming user hints. Returns `None`. Beam sticks stay, and everything runs as with the switch off |
| graph opts out | Logged and counted. Returns `None`, same route |
| out of time with no plan (`NO_PLAN`) | Stage 9's fallback to SA over the same free graph, seeded at the beam. With an empty `config.fallback_solver`, returns `None`, same route |
| `verify_realized` fails | `RuleViolation`. IR already mutated, so there is no retry |

`solve_decides_sticks` requires `config.lx_planning`, `config.lx_sticks_in_solve`,
`config.co_optimizing_lx_planning` (committed mode pins divisions that exist only after
`_distribute_work`), `not config.auto_coarse_tiling`, and empty `span_overflow_groups` under the
beam's stamps, the interim for #5682 Step 1. `_maybe_solve_sticks` adds the family test through
Stage 9's `family_key(config)`, the routing `plan_allocation` uses, so the two cannot disagree.

### Config

```python
lx_sticks_in_solve: bool = os.environ.get("SPYRE_LX_STICKS_IN_SOLVE", "0") == "1"
lx_stick_domain_limit: int = int(os.environ.get("SPYRE_LX_STICK_DOMAIN_LIMIT", "2"))
```

Neither is in `_cache_config_ignore_prefix` (`config.py:503`), so both key the FX graph cache.
`lx_stick_domain_limit=1` pins every stick while still building the graph before `finalize_layouts`:
that is 11a's mode.

## Changes

`<lx>` is `torch_spyre/_inductor/scratchpad/lx_allocator`.

| file | change |
|---|---|
| `<lx>/sticks.py` | New: everything in "Graph-layer additions" except the fields on the graph types |
| `<lx>/graph/nodes.py`, `<lx>/graph/optimization_graph.py` | `OutPort.candidates`, `DataEdge.sticks`, `facts_for`, `restickify_realized` |
| `<lx>/graph_builder.py` | `stick_candidates=None`, candidate and row facts through `facts_for`; tick gaps from 11b |
| `<lx>/optimizer.py` | `plan_sticks`, `sticks.py` in `optimizer_pass_sources()` |
| `<lx>/commit.py` | `commit` divides leftovers. `read_commit_plan` accepts realized steps |
| `<lx>/cost.py` | `CostModel.restickify_cost`, row terms in `objective` |
| `<lx>/family/ortools_encode.py`, `<lx>/family/edge_tables.py` | Everything under "OR-Tools encoding", conditional facts |
| `<lx>/family/sa/` | `_StickedDivisions`, relation over stick entries, step boxes, hazard as non-placeable source |
| the module holding `solve_decides_work_division` | True on a family with a held plan |
| `torch_spyre/_inductor/passes.py` | `_maybe_solve_sticks`. Branch at the head of `_maybe_scratchpad_planning`. Early return in `_maybe_coarse_tile_span_overflow` |
| `torch_spyre/_inductor/insert_restickify.py` | Record `restickify_realized`. Split out `restickify_loop_info` |
| `torch_spyre/_inductor/padding.py`, `cost_features.py` | `restickify_padding_plan`; `restickify_cost_features` |
| `torch_spyre/_inductor/work_division.py`, `pass_utils.py` | `_prep_for` and `_per_core_view_on_buf` cache keys take the stamped layout |
| `torch_spyre/_inductor/pass_counters.py`, `config.py` | Census counters, opt-out reasons; the two values |
| `docs/source/compiler/scratchpad_planning.md` | Pipeline order and stick domain |
| `tests/inductor/lx_equivalence.py` | `canonical_proto`, `StageRecord.canonical_sha`, `SCHEMA_VERSION` bump |

**PR 11a: the graph is built before `finalize_layouts`, sticks pinned.** Implements #5682 Step 2
item 6. Each commit leaves tests green; nothing is reachable until commit 8.

1. Stick census counters, off the path. Records domain sizes, pairs, regions.
2. `tentative_layouts`, plus layout in the two cache keys.
3. `restickify_padding_plan` and `restickify_loop_info`, with the passes calling them. No behaviour
   change.
4. `canonical_proto` in the harness.
5. `stick_key`, `stick_domain`, `StickCandidate`, `build_stick_candidates`.
6. `StickPairRow`, `StickRelation`, `build_stick_relations`.
7. `predict_shadow_ops`, `shadow_ops`, `build_graph(stick_candidates=)`. Under
   `lx_stick_domain_limit=1` each beam restickify is a shadow `RestickifyNode` with its own menu,
   plus `Step(kind=StepKind.RESTICKIFY, pinned=True, realized_by=node)`. No encoder or commit file
   changes.
8. `plan_sticks`, `write_solved_sticks`, `rebind_graph`, `verify_realized`, `_maybe_solve_sticks`,
   gated on both config values and refusing any limit but 1. The predicate turns true on a held
   plan, with `commit`'s leftover step.

**PR 11b: sticks are free.** Implements #5682 Step 3. Unpinned steps first exist here.

9. OR-Tools: stick column, stick relation, per-row edge match, optional step, hint, tick gaps. Stage
   6's raise is scoped to edges with no `sticks`. `read_back` decodes sticks and steps.
   `read_commit_plan` accepts a realized unpinned step.
10. Hazard under row literals; `crosses_frame`.
11. `facts_for` and conditional posting.
12. `restickify_cost_features`, `CostModel.restickify_cost` and row terms.
13. SA family.
14. `_maybe_solve_sticks` opens limit 2. Failure tests. Gate report added to the PR.

## Interfaces

**Consumes**

| symbol | signature | from Stage | used for |
|---|---|---|---|
| harness | `Recorder`, `proto_text`, `StageRecord`, `lx_fake_device`, `lx_zoo.MODELS`, `FIXED_GRAPHS`, features `beam_restickify`, `fixup_copy` | 0 | Gates, compile-path tests |
| `OutPort`, `InPort`, `Port`, `DataEdge`, `Pin`, `PortFact` | parent-plan members; `Port = InPort \| OutPort`; `DataEdge.steps`, `DataEdge.add_step(step)` | 1 | Extended here |
| `Step`, `StepKind.RESTICKIFY`, `apply` | `Step(kind=StepKind.RESTICKIFY, to_stick=None, realized_by=None, target=None, pinned=False, ...)`; `apply(step, layout)` returns `step.target` when set | 1 | Unpinned steps |
| `RestickifyNode` | Stage 1's class, filled by Stage 2 | 1, 2 | Shadow nodes |
| `stick_dims`, `SPARSE_STICK` | `(layout) -> tuple[int, ...]`; `-1` | 1 | `StickCandidate.dims` |
| `build_graph`, `select_target` | `build_graph(ir, target, *, divisions_committed=False, drain_plans=None, lx_relayout_plans=())`; `select_target(config) -> Target` | 2 | Gains `stick_candidates=None`. Runs before `finalize_layouts` under stamps |
| `Target`, `OpConfig` | `graph.target.ops.ops[type(node)].facts`, `graph.target.rules`, `graph.target.steps` | 1 | The facts mask in `facts_for`; `Rule.TILES_OFF_STICK`; `StepKind.RESTICKIFY` offered |
| stick pins | `Pin(out, "stick", stick_dims(layout) or (SPARSE_STICK,), source)`, sources `"layout"`, `"graph input"`, `"pre-pass"` | 2 | `"layout"` pins are not created for free nodes |
| `derive_port_facts` | `(port, layout) -> PortFact`, before the op config's mask | 2 | `facts_for` wraps and masks it |
| `classify`, `_committed_work_split` | a deferred op's split is computed, never written | 2 | Shadow nodes; pinned ops under stamps |
| `stick_tables`, `splits_while_stuck` | `(node, flag)`, for each flag in `node.restrictions` | 3 | Evaluated under stamps |
| `check_report`, `RuleViolation` | `(graph) -> CheckReport` | 3 | `verify_realized` |
| `ALWAYS`'s `cross_frame` check | its name string `"cross_frame"` tags provenance | 3, 6 | Hazard constraint |
| `Rule.TILES_OFF_STICK` | read from `graph.target.rules` | 3 | Per-row tiling check |
| `Sym.restick(e, row)`, `Sym.stick(n)` | Stage 4's rows | 4 | Row terms, stick binding |
| `CostModel.objective`, `.priced_objective` | `(graph, domains)`; `(graph, domains, *, linear_only)` | 4 | Row terms |
| `post`, in `ortools_encode.py` | `post(enc, subject, rule, expr, when)` | 5 | Conditional rules |
| `stick_var`, `stick_is`, `work_division_context_for_op` | `(enc, port) -> IntVar`, a private candidate index; `(enc, port, dim) -> bool \| IntVar`; as Stage 5 | 5 | Table column, legality under stamps |
| `encode_restickifies` literal | `Sym.restick(edge)`, folded on constants | 6 | The literal only, made live, and its raise scoped |
| `build_edge_tables`, `EdgeTables`, `Gate` | as Stage 6 | 6 | Stick key on `pairs` |
| `restickify_read_is_core_local` | `(ir, name, reader, *, plans=()) -> bool` | 6 | Hazard proof |
| `PlacementBox`, `Encoding.boxes`, `Solution`, `SolveStatus`, `OrToolsFamily.encode`, `.read_back` | as Stage 7 | 7 | Step box; decoding stick and the step's `target` |
| `AnnealModel`, `DivisionSource`, `EdgeRelation`, `TableRelation`, `SAFamily.encode`, `.read_back` | as Stage 8 | 8 | `_StickedDivisions` |
| `optimize`, `family_key`, `select_families`, `SolveSettings.from_config`, failure flow | `optimize(graph, family, fallback, settings)`, which reads back and checks but never commits; `family_key(config) -> str \| None`, `None` on the old path; `select_families(config) -> tuple[Family, Family \| None]` | 9 | `plan_sticks`, the old-path test |
| `solve_decides_work_division` rule, `optimizer_pass_sources()` | false on a family; planner sources in the FX key | 9 | Turned true on a held plan; lists `sticks.py` |
| `commit`, `read_commit_plan` | `commit(graph)`, runs the context-switching passes; `(graph) -> CommitPlan` | 10 | Called on the held plan. Unchanged in 11a except the leftover step. Accepts realized steps in 11b |
| `divide_deferred_op`, `solve_decides_work_division` | #5695 | #5682 Step 1 | Leftovers on a held plan |
| span floors inside menus | #5682 | #5682 Step 1 | Per candidate |
| reservations for fixup passes | #5682 | #5682 Step 2 | Ported by Stages 5-7 |
| three direct tests | #5682 | #5682 guards | Prerequisite |

**Provides**

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `config.lx_sticks_in_solve` | `bool`, default off; step 12c turns it on | 12, 13 | final |
| `solve_decides_work_division()` | true on a family only with a plan held, so only with the switch on | 12, 13 | final |
| `divide_deferred_op` in `commit`, the no-plan route | see "The #5695 predicate" | 13 | final |
| `config.lx_stick_domain_limit`, `canonical_proto` | `int`; `(model: object) -> str` | none | final |
| `StickCandidate`, `StickPairRow`, `StickRelation`, `ShadowRow`, `OutPort.candidates`, `DataEdge.sticks`, `restickify_realized` | as in Design | none | final |
| `facts_for` | `(graph, port, layout) -> PortFact`, masked by the node type's `OpConfig.facts` | none | final |
| `tentative_layouts`, `shadow_ops` | as in Design | none | final until a named follow-up has the graph own candidate layouts |
| `solve_decides_sticks` | `(ir) -> bool` | none | final; the span-overflow clause stays until a named follow-up |
| `stick_domain`, `build_stick_candidates`, `build_stick_relations`, `predict_shadow_ops`, `write_solved_sticks`, `rebind_graph`, `verify_realized`, `_maybe_solve_sticks` | as in Design | none | final |
| `plan_sticks`, `CostModel.restickify_cost` | `(ir, config) -> SpyreOptimizationGraph \| None`; as in Design | none | final |
| `restickify_cost_features`, `restickify_padding_plan`, `restickify_loop_info` | as in Design | none | final |
| `restick_resident`, the step's `PlacementBox`, the row hazard constraint | encoder internals | none | final |

## Tests

New tests, all under `tests/inductor/`. Compile-path files run on Stage 0's `fake` tier, one process
per compile.

- `test_stick_domain.py`: `test_beam_choice_always_kept` (rules 3 and 4 never remove
  `committed_stl`), `test_one_layout_per_distinct_stick` (the census's 42-op case),
  `test_sparse_layout_dropped_unless_beam` (the softmax pair), `test_limit_one_pins_everything`,
  `test_graph_input_and_matmul_output_have_one_candidate`,
  `test_loop_ops_and_read_copy_holders_stay_pinned`, `test_ops_with_no_stick_table_stay_pinned`,
  `test_candidate_that_raises_in_preview_is_dropped` (`_apply_qfp8wt_k_constraint`),
  `test_candidate_that_queues_a_copy_is_dropped` (a gather re-tile),
  `test_candidate_dims_match_stick_dims` (factorized and sparse),
  `test_preview_restores_committed_state`, `test_tentative_layouts_restores_on_exception`,
  `test_preview_equals_pass` (each preview equals what `reorder_nonstick_dims` produces with that
  stick committed).
- `test_stick_relation.py`: `test_rows_match_edge_cost_map` (every row agrees with `.cost` and
  `.layout`), `test_infeasible_pair_has_no_row`, `test_no_row_restickifies_a_mutation_target`,
  `test_co_output_rows_require_equality`, `test_clone_consumer_crosses_frame_without_a_step`,
  `test_rows_keyed_by_edge_key_match_finalize_layouts` (`x + x` and `x[..., :8] + x[..., 8:]`).
- `test_shadow_ops.py`, compile path, on the transposed SwiGLU: `test_shadow_menu_equals_real_op`,
  `test_shadow_reads_equal_insert_restickify`, `test_shadow_features_equal_real_op`
  (`extract_op_features` at one division, with bundle membership),
  `test_shadow_ops_restore_on_exception` (`operations`, `name_to_buffer`, no buffer name consumed),
  `test_shadow_builds_restickify_node_and_pinned_step`, `test_realized_op_without_a_shadow_raises`,
  `test_unshadowed_shapes_opt_out` (one per table row, and the counter names the reason).
- `test_canonical_proto.py`: `test_tick_shift_compares_equal` (models differing by tick offset and
  buffer names), `test_structure_change_compares_different` (an extra box, a changed size, a changed
  domain).
- `test_stick_encoding.py`, OR-Tools on synthetic graphs:
  `test_all_pinned_posts_no_stick_constraints`, `test_restick_literal_equals_row_for_every_pair`,
  `test_banned_pair_never_chosen`, `test_forced_restickify_raises_only_without_sticks`,
  `test_hazard_on_free_edge_forces_spill_or_proof`, `test_beam_hint_is_feasible`,
  `test_uniform_fact_posted_unconditionally`, `test_conditional_fact_posted_under_literal`,
  `test_fact_outside_op_config_is_never_posted` (a derived target whose `OpConfig.facts` drops
  `QFP8WT`),
  `test_solver_refuses_restickify_when_it_does_not_pay`, and
  `test_solver_takes_restickify_when_it_saves_spills`: two sticks, one needs a restickify and opens
  a split that keeps a buffer resident, and the solve takes it. This is the #5665 shape.
- `test_stick_encoding_sa.py`: `test_seed_uses_beam_sticks`, `test_every_visited_state_is_legal`
  (`check_decisions` on sampled states), `test_hazard_makes_source_nonplaceable`,
  `test_flip_changes_stick_and_forces_step`, `test_same_seed_same_plan`.
- `test_restickify_pricing.py`: `test_row_features_equal_real_op_features` (including
  `hbm_pattern`), `test_non_dl16_step_has_byte_charge_only`, `test_tick_gaps_match_insert_order`.
- `test_padding.py`, added: `test_padding_plan_matches_pass_output`,
  `test_padding_plan_matches_pass_input`, `test_padding_plan_raises_on_sliced_output`.
- `test_sticks_in_solve.py`, compile path: `test_verify_realized_catches_undeclared_restickify`, and
  a planted violation through `check_report`; `test_infeasible_keeps_beam_sticks`,
  `test_lx_planning_off_never_enters_the_solve`, `test_old_path_never_enters_the_solve`,
  `test_committed_mode_and_tiling_keep_sticks_pinned`, `test_held_plan_defers_distribute_work`,
  `test_no_plan_runs_distribute_work_in_full`, `test_commit_divides_leftover_deferred_ops`,
  `test_read_commit_plan_accepts_realized_step` (and raises on an unrealized one), and
  `test_forced_candidate_sweep`: over the zoo, force each free op to its non-beam candidate in turn,
  and run through kernel preparation. No pass may raise and `verify_realized` must pass.
- `test_solver_relayout_decision.py`, added: the new config names match no ignore prefix.

**Existing suites.** All parent-plan unit suites, serially. On every commit: `test_restickify`,
`test_padding`, `test_dedup_constants`, `test_frontend_work_bounds`, `test_indirect_access_*` and
the issue's direct tests.

**Device suites** after each PR, with the switch on for 11b: the parent plan's list and the
`test_inductor_ops_lx_*` shards. These run on a card, by hand: no CI job sets
`SPYRE_LX_STICKS_IN_SOLVE`.

**What proves the claim.** `test_solver_takes_restickify_when_it_saves_spills` and
`test_forced_candidate_sweep`, plus the predicted-runtime comparison below.

## Parity and gates

The parent plan gives this stage no gate of its own. Its rule is: "Any other difference is listed by
rule in its PR and either justified or fixed." Verification adds the Stage 0 harness, the device
suites after Stage 11, and planning time. The issue says plans change by design at this point, so
the gates take its guards.

**Precondition: probe P2, before Stage 5.** On today's main, device-free, no new code: for each of
the 35 free regions force the non-beam candidate after the beam, compile through `CostReport`, and
record `total_us`, whether a pass raises, and the count of ops added between the beam and planning
and how many end up LX-resident. This is the core of `test_forced_candidate_sweep`, plus a census.
It answers whether sticks pay at all and how many ops need a shadow or opt out, and it informs both
Open questions. It has not been run.

**Tiers.** `fake` zoo and the fixed set, with Stage 0's two restickify graphs, plus Granite for 11b.
The base arm is `main` at the branch point. `fake` inputs use default layouts, so `fake` proves
parity, not production plans. `CARD_ONLY` models stay on a card.

**PR 11a. Gate: nothing changes.**

- Compare, under `LAYOUT_SOLVER=ortools` and Stage 0's deterministic settings: base against 11a with
  `lx_sticks_in_solve=1`, `lx_stick_domain_limit=1`, and against 11a with the switch off.
- Corpus: as above, at default config. Stage 0's `beam_restickify` feature must have hits, or the
  gate is vacuous and fails.
- Pass, switch off: proto, decisions and committed plan identical, byte for byte.
- Pass, switch on, the hard gate: decisions, objective, inserted ops, addresses, divisions and spill
  reasons identical.
- Pass, switch on: `canonical_proto` identical (D8). It takes a clone of the model's proto, clears
  `name` on variables and constraints, collects the constants in the `start`, `end` and `size` of
  every interval `no_overlap_2d.x_intervals` names plus the bounds of `start_` and `end_` variables,
  replaces each with its rank, and applies the same map to linear constraints over those variables.
  Nothing else changes. Rewriting a cloned OR-Tools proto this way works on a toy model; on a real
  one it is unproven until commit 4. `StageRecord` gains `canonical_sha`, with a `SCHEMA_VERSION`
  bump, and `diff` compares it on switch-on arms.
- A graph that opts out is listed by rule, and passes because it takes the switch-off path.
- A difference is a bug. It is fixed, or listed by rule and justified.

**PR 11b. Gate: plans change, and each change is justified.**

- Switch off: same byte-identity pass as above.
- Switch on, limit 2, per graph on the same corpus:
  1. `check_report` and `verify_realized` report nothing on any plan.
  2. **Predicted runtime.** `CostReport.total_us` (`SPYRE_DUMP_COST=1`) against the 11a baseline,
     measured on the committed IR by `cost_model_pass`, so it does not trust the solver's pricing.
     Any increase over 0.1% needs a written reason in the PR. An unexplained increase blocks. The
     geomean must not rise.
  3. **Solve status.** On the fixed graph set, a graph that newly stops at the time limit or falls
     back fails. On Granite this is a report, and a limit stop is accepted where predicted total did
     not rise. With the hint, a stopped solve is never worse than the pinned plan on the solver's
     objective.
  4. **Size counters.** CP-SAT variables, constraints and table entries per op, bounded in the style
     of `test_frontend_work_bounds.py`. Bounds are set from 11b's first run and pinned.
  5. **Compile time.** Live `fake` compiles, interleaved, reading `pass_loop_ns` as Stage 9's Gate 3
     does. A report, not a gate: Stage 12 sets 12c's written limit. The parent's zoo replay cannot
     be used, since captures have no candidates.
- Family gate: all of the above with `LAYOUT_SOLVER` selecting each family, SA against its own 11a
  baseline.
- Device: the suites above pass, on a card. Predicted numbers justify default-off only.
- **The issue's weekly perf check** (`make tests TEST_TYPE=perf`) runs on 11b before step 12c, whose
  gate reads it.

## Risks

- **R1. Solve time.** Search growth is unmeasured, and Granite graphs 2 and 3 sit just inside the
  limit. Caught by gate item 3. Mitigation: domain limit, beam hint, relayouts on the beam's pair
  only, and `lx_stick_domain_limit=1` for large graphs if Granite cannot hold.
- **R2. Reservation undershoots.** A resident buffer ends up larger than planned. Caught by
  `verify_realized` step 3.
- **R3. Stamps leak or caches alias.** Caught by `test_tentative_layouts_restores_on_exception` and
  `test_preview_equals_pass`. A missed cache shows up as a `verify_realized` failure.
- **R4. A pass raises after the IR is mutated.** No rollback exists. Caught by preflight drops and
  `test_forced_candidate_sweep`.
- **R5. Standalone pricing misprices a step.** The solver can prefer a plan that the real model
  prices worse. Caught by gate item 2, measured on the IR. A relayout-before-restickify order is
  also unavailable on non-beam pairs, which would show the same way.
- **R6. SA gets worse.** A larger domain with the same budget. Caught by the family gate.
- **R7. Facts that vary on fp8 or fp32 graphs.** The census saw only FP16. Caught by a stick census
  on the fp8 and fp32 suites in commit 1, and by `test_conditional_fact_posted_under_literal`.
- **R8. No benefit.** The zoo gives the solve only footprint and split trades. P2 shows this first,
  and gate item 2 confirms it. If totals do not improve, the stage lands with the switch off and
  step 12c does not happen.
- **R9. Step 1 or 2 slips.** Stage 11 and 12c wait. Stages 12 and 13 do not. A two-solve form would
  avoid the wait, but it contradicts "one solve", so it is not planned.
- **R10. Stale `op` references at commit.** Caught by `rebind_graph`, which asserts every planned
  node resolves, is a realized shadow, or is a known dropped constant.
- **R11. A solved stick declines a direct read.** Holders of the record stay pinned, but the staged
  copy need not be, and its stick is one condition of the proof. A proof that passes today can then
  decline and the copy stays, while `project_transport_read_copies` priced it away: a plan change
  the solver cannot see. Caught by gate item 2, and listed as a reason when it fires.
- **R12. A shadow differs from the real op, or many graphs opt out.** The probe compared menus on
  one model. Caught by `test_shadow_ops.py`, the 11a gate, and `verify_realized` steps 1 and 2. P2
  counts opt-outs, and more shapes can follow one at a time under the 11a gate.

## Deviations from the parent plan

**D1. "Four things" with five bullets.** All five are required; each has its own section under
Design.

**D2. Stage 11 is not buildable as one PR, or without #5682 Steps 1-2.** The plan says lift the
pins. The solve runs at `passes.py:577`, after `finalize_layouts` has deleted `layouts` and
`restick_cost_fn` (`insert_restickify.py:625-627`) and after twelve passes have consumed the stick.
Span floors, padded sizes and inserted ops feed the solve from those passes. Resolution: the solve
moves ahead of `reorder_nonstick_dims`, Stage 11 takes the prerequisites in its header, and it
splits into 11a (plans identical) and 11b (plans change on purpose). One PR cannot satisfy both
gates, and it would lose the plan-equivalence proof.

**D3. "Pre-pass restickifies" stay pinned.** Read literally, every beam restickify would stay
pinned, and the solve could never remove one. Resolution: restickify ops in the IR at the build stay
pinned, and so does a beam restickify between two pinned ops, as a shadow. On an edge with free
sticks the beam's restickify is one row among others. The pre-pass stays as the source of candidates
and the seed.

**D4. `elide_proven_read_copies` stays.** The plan says it goes "because a restickify is never
present where the sticks agree". The pass removes staging copies of graph inputs, proved in
`read_copy_elision.py:275-528` and priced through `project_transport_read_copies`
(`allocator.py:3203-3216`). Sticks are one condition of its proof, not its purpose: a pointwise
consumer declines when source and copy stick symbols differ, and a matmul needs the source stick to
be the generated var (`read_copy_elision.py:400-426`). Choosing sticks in the solve does not replace
it. A restickify can carry the record (`insert_restickify.py:523-538`) or be the removed copy
(`:555-566`), which is where the parent's reason comes from, but most removals are not that branch,
and tests reach it only with stubbed proofs (`tests/inductor/test_coarse_tiling.py:5666-5704`).
`DIRECT_READ_SOURCE` depends on the pass. Resolution: it stays, no stage deletes it or makes it a
no-op, and `read_copy_elision` stays a live setting. Risk R11 lists the plan change this allows.

**D5. More stick pins stay than the plan lists.** The plan excepts only graph inputs and pre-pass
restickifies. (a) Loop ops: the Decisions table says loops are pinned hazards, and
`insert_restickify_on_node_inputs` holds per-read advance transfer the solve does not model. (b) All
ops under `AUTO_COARSE_TILING`: CP-SAT gets no cost expression (`allocator.py:3275-3300`), so a
restickify would be free. SA keeps its expression, but a fallback re-encodes the same graph, so both
families take the pin. Enumeration is about 88% of planning there, with a projected 1.74x on top.
(c) Conv, exx2, topk, keep_by_index and layernormnorm, which have stick rules in
`propagate_layouts.py` and no table in Stage 3. Resolution: pinned in this stage.

**D6. `Step.to_stick: int` is not enough.** Factorized and constant sticks have no single dim, and
`compute_restickify_target_layout` returns a full STL. Resolution: an unpinned `RESTICKIFY` step
carries Stage 1's `Step.target`, which the parent's `STEP_KINDS` table does not list.

**D7. `OVERLAPPING_READ` is layout-derived.** The Decisions row says in-port facts come "from the
read itself (overlap, direct-read provenance)". The code derives overlap from `td.device_coords`
(`work_division_constraints.py:1151-1172`). Resolution: per row, under `facts_for`.

**D8. Byte-identical proto in 11a.** Ticks come from the op list at the solve, and dedup has not run
there, so the numbers differ even when the order does not. A shadow's variables also carry another
name. Resolution: `canonical_proto`, with decision equality as the hard gate.

**D9. `plan_allocation` is split in two.** Stick passes must run between `optimize` and `commit`,
because commit needs final layouts, while the parent's `plan_allocation` runs them back to back.
Resolution: `plan_sticks` runs the first half from `_maybe_solve_sticks`, and
`_maybe_scratchpad_planning` calls `commit(graph)` on the held plan.

**D10. "One solve" holds, with one exception.** If `plan_sticks` returns `None` after a solve, the
later position runs a second solve with sticks pinned. This is a failure path only.

**D11. Stage order.** The parent places Stage 11 before Stages 12 and 13. Neither depends on it
here, only step 12c does. Stage numbers do not change.

## Open questions

**Q1. Ordering with #5682, to settle with mtsokol before Stage 5 starts.** Should Step 2 stop at
Step 1 plus the guards, with reservations living in Stage 11, or land fully before Stage 5? Stage 11
keeps its prerequisites as written, with item 6 reserved for 11a on the graph layer. Recommendation:
pick one now, whichever he can finish before Stage 5, because a half-landed Step 2 is the worst case
and each later item then needs its own encoder port. If Step 2 stops early, the one-op fixups become
shadows and the rest opt out, as Design says. If he does item 6 in `allocator.py`, 11a shrinks to a
port, but Stage 13 deletes that code.

**Q2. Should the order be 0-10, 12, 13, 11?** Recommendation: yes. Stage 11 needs external work and
has the hardest unknowns, nothing in its gates needs the old path, the Phase 3 exit then does not
wait on it, and #5682's Step 2 lands once. Stage numbers stay either way. If the owner keeps the
numbered order, nothing here changes, since the switch stays off through Stages 12 and 13.

## Out of scope

- Moving #5682 Step 1 and Step 2 items 1-5 and 7, and the issue's direct tests: the issue's roadmap.
- The harness, corpus and guards: Stage 0. Encodings for pinned sticks: Stage 6. Commit's step
  insertion: Stage 10. The default flip, comparison and step 12c: Stage 12. Deleting old solvers:
  Stage 13.
- Sticks under tiling, which would need a restickify rung on CP-SAT's ladder, and a non-enumerating
  stick encoding: future work.
- Choosing non-stick dim order or padding in the solve: "Stick layout optimization" and "Padding" in
  Future work.
- Direct reads as a solve decision.
