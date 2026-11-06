# LX allocator, Stage 13: Deletions

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0-10 and 12. Needed by: none (Phase 3 exit). Stage 11 may land before or
after this stage: nothing here needs it, and nothing deleted here is something it needs.

Paths under `torch_spyre/_inductor/` are shortened to `inductor/`, `inductor/scratchpad/` to
`scratchpad/`, and `scratchpad/lx_allocator/` to `lx_allocator/`. Line numbers are from the
snapshot. Where a line may have moved by Stage 13, the symbol name is the reference.

## Goal

When this PR merges, `scratchpad_planning` has one path: `plan_allocation`, with the `ortools`
or `sa` family. The old allocator classes, whose role `plan_allocation` takes over, the solver
base classes, the five placement solvers, `CpSatLayoutSolver`, `SaCoOptimizingSolver` and the
greedy re-plan are gone, along with the side-by-side machinery, every temporary item earlier
stages left for this stage, and config values, tests and docs that only served them. Family
plans do not change: the Stage 0 harness shows an empty diff on family cells between Stage 12's
head and every commit here, C12's gated change aside.

Stays unchanged: the beam, `_distribute_work`, `span_reduction`, `LX_PLANNING=0`,
`elide_proven_read_copies`, `LxContextSwitchingPass`, the native packer, and sticks pinned
unless Stage 11 and 12c have landed. `_matmul_layouts`, `_multi_arg_pointwise_layouts` and
`_check_stick_all_same` stay (D1).

## Today

**Selection.** `_maybe_scratchpad_planning` (`inductor/passes.py:425-432`) returns at once
when `config.lx_planning` is false. Otherwise it calls `anchor_lx_relayout_ownership` and
then `scratchpad_planning`. `select_allocator` (`scratchpad/allocator.py:5377`) maps config
to a class pair, and `scratchpad_planning` (`:5472`) catches `SolveError`, raised for any
status without a plan (`scratchpad/ilp_solver_ortools.py:871-875`, `:1434-1440`), and re-plans
with `ScratchpadAllocator(GreedyLayoutSolver, ...)` (`scratchpad/allocator.py:5495-5515`).

**Surprises that shape this plan.**

1. **A default compile without ortools raises today.** `_cpsat_solver_class` returns
   `GreedyLayoutSolver` (`scratchpad/allocator.py:5366`), which cannot choose divisions, so
   `select_allocator` raises `ValueError` unless `allow_exhaustive_search` is set (`:5436-5450`,
   pinned at `tests/inductor/test_scratchpad_use.py:1791`). Greedy is reached only with
   `CO_OPTIMIZING_LX_PLANNING=0`.
2. **The SA path runs no post-passes.** `select_allocator` builds the SA allocator with no
   `post_optimization_passes` (`scratchpad/allocator.py:5433-5435`), yet the residency guard
   for extern kernels is off whenever `enable_lx_context_switching` is on
   (`scratchpad/allocator.py:857`). This is reachable only with
   `LAYOUT_SOLVER=simulated_annealing`. Whether a buffer is clobbered needs a card, which I
   did not use.
3. **Only `collect_lx_relayout_plans` reaches completed-reduction routes.**
   `derive_completed_reduction_routes` has one caller, `scratchpad/lx_relayout.py:1733`.
   The collector's plans are used only by a solver with `supports_paired_buffers()`, and
   that is greedy alone (`tests/inductor/test_scratchpad_use.py:1884-1903`).
4. **Tests reach deep into the classes.** They call 44 distinct private names on them, and
   `tests/inductor/test_work_division.py:145` builds an allocator around a solver it never
   runs, only to reach those methods.
5. **A test resolves a dotted path to a method.** `TRANSFER_CERTIFICATES` names
   `...allocator.CoOptimizingAllocator._cap_relayout_groups`
   (`scratchpad/residency_rules.py:357`), and
   `tests/inductor/test_work_division.py:5143` calls `resolve_name` on it.
6. **`_runs` reaches no cache key at this snapshot.** `_uuid` hashes the files named by
   `@_runs` (`inductor/passes.py:164-171`), but nothing calls
   `CustomPreSchedulingPasses.uuid()`: the pipeline runs from a patch
   (`inductor/patches.py:213-229`). So neither `allocator.py` nor `lx_relayout.py` is hashed
   today. Stage 9 makes those sources reach the FX key.
7. **"Greedy" means three things.** `GreedyLayoutSolver`; the distributors inside
   `work_distribution` (`inductor/work_division.py:298`,
   `inductor/work_division_constraints.py:459`); and the parent's "greedy pins", a `Pin.source`
   value with no code in the snapshot. Only the first is deleted.

## Design

### Entry conditions

The PR does not open until all of these hold. The audit in step C0 checks each one.

- Stage 12's rollout step 6 holds. That includes `fallback_solver` non-empty and
  `lx_solver_completed_reduction_routes` on by default, the condition C12 rests on.
- No code under `torch_spyre/` outside the deleted set constructs an old class, except
  through the seams in the table below.
- `check_decisions` runs on every suite (parent: "From Stage 3").

Stage 11 is not a condition. `lx_sticks_in_solve` may hold either value, or not exist yet.

### Parent-named deletions, every importer

"Prod" lists production references under `torch_spyre/`. Test importers are under Tests.

| symbol | defined | prod importers and callers | docs and tools | after deletion |
|---|---|---|---|---|
| `ScratchpadAllocator` | `scratchpad/allocator.py:951` | `allocator.py:796` (field type), `:3085` (base), `:5377`, `:5465`, `:5474`, `:5511`; docstrings `lx_context_switching.py:363`, `pass_utils.py:4191`, `config.py:347` | `scratchpad_planning.md`, `co_optimizer.md`, `api/torch_spyre.rst`, `allocator-architecture.svg` | Pipeline gone; residency methods go to `LxRecordBuilder`. `_BufferResidencyChecks.allocator` retyped. Docstrings rewritten. |
| `CoOptimizingAllocator` | `allocator.py:3085` | `allocator.py:750`, `:5433`, `:5451`, `:5459`; string in `residency_rules.py:357`; docstrings `exhaustive_search.py:48,166`, `plan_solver.py:264,775`, `sa_cooptimizer.py:770` | `user_guide/examples/profile_ops.py:893-911` | As above. `:750` calls Stage 6's `_loop_carry_update_edge`. The certificate string changes in C2. `profile_ops.py` keeps Stage 12's `family_key(config)` and drops the allocator branch. |
| `MemoryPlanSolver` | `plan_solver.py:970` | Bases: `greedy_solver.py:31`, `firstfit_bestfit_solver.py:82`, `simulated_annealing.py:104`; typing: `simulated_annealing.py:70,99`, `allocator.py:960,990,1007,1090,3218,5345,5369` | `simulated_annealing_layout.md` | Gone last, when every subclass has gone. `excluded`, `record_exclusions` and `partition` stay in behaviour through `first_fit_place` only. |
| `CoreDivisionLayoutSolver` | `plan_solver.py:1156` | Bases: `exhaustive_search.py:34`, `ilp_solver_ortools.py:623`, `sa_cooptimizer.py:686`; `allocator.py:3088,3117,3219,5460` | `co_optimizer.md` | Gone with `MemoryPlanSolver`. |
| `CpSatLayoutSolver` | `ilp_solver_ortools.py:623` | `allocator.py:5354-5357` only | `scratchpad_planning.md`, `co_optimizer.md` | File `ilp_solver_ortools.py` deleted, helpers included (below). |
| `GreedyLayoutSolver` | `greedy_solver.py:31` | `allocator.py:87,5366,5370,5512`; `simulated_annealing.py:65,179` | `scratchpad_planning.md` | File deleted. |
| `FirstFitLayoutSolver` | `firstfit_bestfit_solver.py:82` | `allocator.py:91,3553,3632,5372`; `simulated_annealing.py:59,171`; `sa_cooptimizer.py:64,1285` | `random_buffers.py` | Class gone, algorithm stays as functions (see "First-fit"). |
| `BestFitLayoutSolver` | `firstfit_bestfit_solver.py:270` | `allocator.py:90,5371`; `simulated_annealing.py:58,175` | `api/torch_spyre.rst:752` | Gone. |
| `ExhaustiveSearchSolver` | `exhaustive_search.py:34` | `allocator.py:97,5452`; comment `sa_cooptimizer.py:1274` | `scratchpad_planning.md`, `api/torch_spyre.rst:756` | File deleted, with `prune` and `allow_exhaustive_search`. |
| `SimulatedAnnealingLayoutSolver` | `simulated_annealing.py:104` | `allocator.py:94,5373` | `simulated_annealing_layout.md`, three examples | File deleted after `SolverToPermutation` moves. |
| `SaCoOptimizingSolver` | `sa_cooptimizer.py:686` | `allocator.py:99,5434` | `sa_co_optimization.md` | File deleted (below). |
| `SolveError` re-plan | `allocator.py:5495-5515` | Class `plan_solver.py:43`, raised `ilp_solver_ortools.py:872,1436` and by Stage 9's `select_families` and `_fall_back`, doc `config.py:382`, docstring `allocator.py:2904` | `scratchpad_planning.md` | Handler, raises and class gone (see "`scratchpad_planning` after"). |
| `_matmul_layouts` | `propagate_layouts.py:1464` | `:2320`, `:3175` | `coarse_tiling_loops.md:1312` | **Stays** (D1). |
| `_multi_arg_pointwise_layouts` | `propagate_layouts.py:1896` | `:2314`, `:3212`; comments `constants.py:94`, `lowering.py:2203` | `coarse_tiling_loops.md`, `indirect_access.md:250` | **Stays** (D1). |
| `_check_stick_all_same` | `op_spec_validation.py:675` | `:672`, from `_check_stick_constraints`, run at `spyre_kernel.py:1485` | none | **Stays** (D1). |

`check_in_place_parent_is_read` (`plan_solver.py:903`) is not on the parent's list, but Stage 3
takes the rule over, so it needs an answer. It stays: `first_fit_place` keeps its input check
(`:953`), the oracle needs an independent statement (`tests/inductor/reference_perm_layout.py:383`),
and the native constructor mirrors it (`torch_spyre/csrc/perm_layout_native.cpp:158-164`). Only the
comment at `plan_solver.py:954-958`, which names `CpSatLayoutSolver._add_inplace_relaxation`, changes.

### Allocator methods, one disposition each

Groups are from an AST call graph of `allocator.py`.

| group | methods | disposition |
|---|---|---|
| Pipeline | both `__init__`, `_build_solver`, `plan_allocation`, `_run_passes`, base `_prepare_buffers`, both `_solve`, `_solver_generates_divisions`, `_solver_chooses_tilings` | Deleted. `CoOptimizingAllocator._solve`'s objective and cost dump (`:3218-3406`) already live in Stages 4, 6 and 10. |
| Placement-only prep | `_generate_buffers`, `_build_bound_buffers`, `_determine_in_place`, `_input_footprint`, `_planned_lx_buffer_names` | Deleted. Nothing but base `_prepare_buffers` reaches them. |
| Committed relayouts | `_append_lx_relayout_destinations`, `_allocated_lx_relayout_sources`, `_clear_lx_relayout_groups`, `_accepted_plans`, base `_finalize_lx_relayout_allocation` | Deleted; greedy was their only user. |
| Residency | `_residency_by_buf`, `_residency_reasons`, `_buffer_residency_reason`, `_input_residency_reason`, `_op_output_good_for_lx_reuse`, `_is_index_or_indirectly_accessed`, `_untiled_residency_bar`, `_read_count`, `_get_op_name` | Move verbatim to `LxRecordBuilder` in C2. This is the closure of `_BufferResidencyChecks`' calls, from an AST walk (D2). |
| Moved by Stages 2, 5, 6 | `_inplace_edge_ok`, `_op_inputs_good_for_lx_inplace`, `_enumerate_core_divisions`, `_tiling_candidates`, `_clone_divisions_and_matches`, `_cd_parent_relayouts`, `_cap_relayout_groups`, `_relayout_consumer_costs`, `_intern_view_group`, `_coarse_tile_loop_trip_count`, `_views_for_divs`, `_parent_residency_edges`, `_cd_parent_matches`, `_loop_carry_update_edge`, `_loop_carry_read_edges`, `_determine_in_place_division_invariant`, `_restickify_barrier`, `_restickify_read_is_core_local` | One-line delegates since those stages. Deleted with the class in C8, and callers use the moved function. |
| Record prep | `_prepare_buffers` (joint), `_division_map`, `_seed_aligned_division`, `_build_cd_bound_buffers`, `_division_space`, `_tile_reads`, `_eligible_clone_inputs`, `_relayout_copy_buffers`, `_direct_read_candidates_priced`, `_pricing_operations`, `_extract_op_features` | Deleted in C8. The new path has its own form through Stages 2, 4, 5 and 6. A method C0 shows `graph_builder`, `rules`, `edge_tables` or `commit` still calling becomes a module function in `allocator.py` first. |
| Commit | `_push_allocation`, `_set_one_allocation`, joint `_finalize_lx_relayout_allocation`, `_post_solve`, `_stage_collective_outputs` | Replaced by Stage 10's `commit(graph)`. Stage 10 left these as wrappers, deleted in C8. `_stage_collective_outputs` is not in the parent plan (D8). |
| Reporting | `_get_spill_reasons`, `_log_lx_pinning` | Stage 10 keeps the log output, because a debug log that names no spill reason is a silent loss. |

Module-level helpers stay in `allocator.py` under their names: `_lx_planning_size`
(`inductor/decompositions.py:507`), `_COST_PARAMS`
(`tools/cost_model/run_decode_matmul_sweep.py:90`, nine test sites), `DrainPlan`,
`validated_drain_plans`, `_hoisted_input_clone_entry`, `_assert_drain_plan_committed`,
`_extern_kernel_in_live_range`, `ScratchpadOptimizationPass`
(`scratchpad/lx_context_switching.py:51`), the twelve helpers `graph_builder.py` imports
(Stage 2, `division_pin_reasons` among them), `enumerate_op_divisions`, `tiling_candidates`
and `clone_divisions_and_matches` (Stage 5), and Stage 10's `push_one_placement`,
`commit_chosen_divisions`, `check_priced_footprint`, `accepted_lx_views` and
`collective_output_stages`, which `commit.py` keeps importing from here. Stage 10's wrappers
`commit_divisions`, `commit_lx_views` and `check_priced_footprints` go in C8, and their tests
call the extracted functions. Staying in the file is deliberate: 47 test sites use
`patch.object(allocator_module, ...)` and 28 more the dotted string, and a moved name would
leave a patch that passes and tests nothing.

### New private types

```python
# scratchpad/allocator.py
class LxRecordBuilder:
    """The allocator methods rules.py still reaches. Holds no solver and builds no record."""
    def __init__(self, *, decides_relayouts: bool) -> None: ...
```

It replaces the never-solving `ScratchpadAllocator` that `rules.py` uses as its host (Stage 3).
It has no `prepare`: `build_graph` walks the IR, so after Stage 12 nothing builds records. It
needs no options type either, because `_decides_lx_relayouts` (`allocator.py:905`) is the only
state the residency group reads. It stays after this stage.

`_BufferResidencyChecks` stays with `allocator` retyped to `LxRecordBuilder`. So do
`division_is_fixed`, `ncores` and `ncores_reasons`, the branch at `allocator.py:1400` and the
registry's `core_division` rule (`residency_rules.py:253`): committed mode reaches them through
`graph.divisions_committed`, and removing them would change committed plans. The docs table and
`test_registry_exception_keeps_later_guards` do not change. `planned_lx_buffers`,
`lx_relayout_plans`, `graph.lx_relayout_plans` and `build_graph`'s keyword of that name go with
C12. `get_ncores_for_buffers` stays, for `accepted_lx_views` (`allocator.py:3020`), collective
staging (`:3613`) and the branch at `:1400`.

### Other code that becomes dead

| code | why | action |
|---|---|---|
| `cooling_schedules.py` (298 lines) | Sole production importer is `simulated_annealing.py:53`. `sa_cooptimizer.py` has its own cool (`:118-119`, `:1919-1920`); line `:91` is only a comment. | Delete. |
| `simulated_annealing.py::SolverToPermutation` | Used only by Stage 8's `_seed_permutation` (today `sa_cooptimizer.py:66`, `:1284`). | Becomes `first_fit_order`. |
| `ilp_solver_ortools.py` helpers: `_PlacementUnit`, `_TilingGroups`, `_chose_a_tiling`, `_op_readers`, `_Slots`, `_gate_divisions`, `_LifetimeBufferWithCpVars`, `_CoreDivisionBufferWithCpVars` | Used only by `CpSatLayoutSolver`. Stages 4 to 7 copied what they need. | Deleted with the file. `lx_allocator/family/ortools_family.py` already guards `cp_model`. |
| `sa_cooptimizer.py` module helpers: `DivisionConfig`, `_DivisionSource`, `_MenuDivisions`, `_GeneratedDivisions`, `_EdgeRelation`, `_TableRelation`, `_ViewRelation`, `_table_relation`, `_canonical_key`, `_split_key`, `_one_axis_apart`, `_step_budget`, `_work_slices` | Stage 8 moved all but the last two to `family/sa/divisions.py`, leaving aliases. `_work_slices` is an alias of Stage 4's `utils.division_work_slices`. | The file goes in C9, and tests import the new names. `allocator.py:3189,3428,5001` and `tests/inductor/test_op_features.py:48` call `division_work_slices`. If `family/sa/` still imports `_step_budget` from here, it moves to `search.py`. |
| Exhaustive-only chain in `allocator.py`: `_enum_split_options`, `_check_and_add_matmul_options`, `_candidate_key`, `_output_profile`, `_from_output_profile`, `_reduction_bm_axes`, `_factored_bm_splits`, `_bm_axes_from_roles`, `_matmul_axis_parse`, `_output_axis_symbols`, `_output_stride_to_device_size`, `_seed_splits`, `_find_distinct_matmul_splits`, `_legal_split_options`, `_split_fits_sticks`, `DEFAULT_VARIANT_CAP`, `_FACTORED_B_FACTORS` | `prune` is set only at `allocator.py:5454` and read at `:3786`, `:3855`. `_division_map` computes `profiles` only for those branches. | Delete. `_split_option_is_legal` and `_hard_domain_violation` stay for `commit_divisions`. |
| `collect_lx_relayout_plans` (`lx_relayout.py:1532`), `anchor_lx_relayout_ownership` (`:1890`), `LifetimeBoundBuffer.paired_with`, `.lx_relayout_plans` | The plans have no consumer once greedy goes. The anchor still mutates owner order when `co_optimizing_lx_planning` is false (`:1902-1906`, `:1996`). | Deleted last, behind the gate in C12 (D5). |
| `utils.py::plot_buffers`, `quality_plot` | Only the three docs examples call them. | Delete with the examples. `utils.py:43` then has no import of `LifetimeBoundBuffer` left to justify. |
| `config.allow_exhaustive_search`, `layout_solver` values `greedy`, `bestfit`, `firstfit` | See "Config". | Remove. |

### Temporary items from earlier stages

| item | from | removed in |
|---|---|---|
| `lx_route_to_families`, its test pins; `_greedy_replan` | 9 | C4 |
| `LEGACY_LAYOUT_SOLVERS`, `warn_planner_selection` | 12 | C4 |
| `lx_check_decisions`, `lx_check_file`, cache prefix `lx_check` and the test pinning it; `_check_plan`, `_finish_plan_check`; `test_lx_check_hook.py`, `lx_check_summary.py`, `lx_check_known.json` | 3 | C4 |
| Reversed lane; the `lx_route_to_families` input and its export line, in `_test_matrix.yaml` and `lx_planner_compare.yaml` | 12, 9 | C4 |
| Hooks on `CpSatLayoutSolver._solve_and_record`, `MemoryPlanSolver.plan_layout`, `CoreDivisionLayoutSolver.plan_layout_and_core_divisions`, and any left on `_get_spill_reasons`; `decisions_from_buffers`, `sizes_from_buffers`; `replay_intree`, `replay_capture`, `blank_for_capture`; the `intree` tier | 0 | C4b |
| `plan_adapter.py`: `decode_todays_plan`, `check_todays_plan`, `check_buffers`, `Mismatch`, `PlanCheck`; `test_lx_plan_adapter.py` | 3 | C4b |
| `lx_sections.py`, `lx_shadow_plugin.py` with `sa_dual_run`, `test_lx_encoder_sections.py`, `test_lx_live_parity.py`, the `AGENTS.md` line | 5 to 8 | C4b |
| `test_lx_placement_parity.py`; `lx_objective_shadow.py`, `test_lx_cost_parity.py`; `commit_shadow` | 7, 4, 10 | C4b |
| `domains_from_buffers` | 4 | C4b, or into `lx_record_fixtures.py` if `solve_records` needs it |
| Stage 4's delegating `_build_score_fn`, `sym_*` properties, `_input_footprint`; Stage 5 and 6 delegates; `RelayoutCopyBuffer.cost_term` | 4, 5, 6 | with their classes, C6 to C10 |
| `SaCoOptimizingSolver` on `AnnealSearch`, aliases in `sa_cooptimizer.py` | 8 | C9 |
| `apply_chosen_tilings`, `planned_splits` wrappers | 10 | C8 |
| `compare_planners.py`, `test_compare_planners.py`, `test_planner_side_by_side.py`, the compare configs, `planner_compare_reasons.json`, label `lx_compare` with its `.github/` entries, `lx_planner_compare.yaml` with its forward lane and cron, the `layout_solver` input and the export step in `_test_matrix.yaml` | 12, 9 | C13 if 12c has landed. Otherwise they stay for 12c's gate, on `lx_sticks_in_solve` arms, and 12c's PR removes them |

Decided as final, so not removed: `graph.ir`, `OpNode.op`, `division_pin_reasons`,
`enumerate_op_divisions` and its two siblings, `settle_deferred_reads`,
`restickify_residency_bar`, the `_COST_PARAMS` alias, `replay_graph`,
`edge_tables_from_buffers`.

### Not dead

- **`work_division_constraints.py`.** `collect_work_division_constraints` runs from
  `span_reduction_pass` (`inductor/work_division.py:1963`), `work_distribution_pass` (`:2114`),
  `_cost_model_divide_op` (`:2976`) and `work_division_context_for_op` (`:1025`), before a graph
  exists or in `LX_PLANNING=0`. The file keeps its entry point, and its 16 rule functions
  (`:135-152`) stay as adapters over the bodies Stage 3 moved to `rules.py` (D6).
- **`residency_rules.py`.** It stays as the registry behind `_BufferResidencyChecks`, the
  generated docs table and four tests. `tools/lx_planning/generate_residency_rules.py` runs
  it with `runpy` "without loading the backend", so it must stay free of backend imports.
  Line 357's dotted path changes to Stage 6's function in `lx_allocator/family/edge_tables.py`.
- **`lx_relayout.py`.** Everything except the collector and anchor stays:
  `materialized_lx_relayouts` (`cost_features.py:1093`), `demote_lx_relayout_group` and
  friends (`scheduler.py:42`), `spyre_kernel.py:69`, `derive_completed_reduction_routes`
  (Stage 6), `LXRelayoutPlan` and `materialize_lx_relayouts` (Stage 10).
- **`coarse_tiling.py`.** `TileReads`, `prescribed_regions`, and Stage 10's
  `apply_tiling_choices` and `splits_from_divisions` stay, and `wsr/enumerate_tilings.py:92`,
  `wsr/tile_prediction.py:89` and `work_division.py:1344` import from it regardless.
- **`plan_solver.py`.** `TileAxis`, `TileSpec` and `CoreDivision` are imported by
  `work_division.py:81`, `wsr/enumerate_tilings.py:97` and `wsr/tile_prediction.py:90`, so
  the file stays. `cost_expr_record`, `solved_bindings` and `RelayoutCharge` stay for the
  cost dump (`cpsat_expr_printer.py:822`).
- **`perm_layout_native.cpp`.** No native code is deleted. `overlaps`, `top_or_inf` and
  `finalize` lose their production caller but keep coverage in `test_perm_layout_solver.py`.

### Buffer records

The Decisions table says "No `Buffer` class", but the parent's Stage 13 list omits
`LifetimeBoundBuffer`, `CoreDivisionBuffer` and `RelayoutCopyBuffer`. `LifetimeBoundBuffer`
stays: the first-fit functions and the native packer's duck-typed input read it (D4). The
other two have no production reader after C8, but `replay_graph`, `edge_tables_from_buffers`
and the capture loader read them. C13 deletes each type C0 shows none of those still needs.

### First-fit

Two callers need first-fit, and neither needs a solver object: the SA seed, which Stage 8 moved
from `sa_cooptimizer.py:1282-1286` to `AnnealSearch._seed_permutation` in
`lx_allocator/family/sa/search.py`, and collective staging, which calls `_build_gaps` and
`_pick_gap` directly (`allocator.py:3632`). Stage 7's `Packer` is its own first-fit over ints and
is unrelated. Postmath's point in PR #5 holds here: an annealer seeded from a valid first-fit
returns a valid plan, so this code must survive unchanged.

```python
# scratchpad/first_fit.py, renamed from firstfit_bestfit_solver.py
def first_fit_place(
    buffers: Sequence[LifetimeBoundBuffer], limit: int, alignment: int = 128
) -> list[LifetimeBoundBuffer]:
    """FirstFitLayoutSolver.plan_layout, unchanged in order and tie-breaks."""

def first_fit_order(
    buffers: list[LifetimeBoundBuffer], limit: int, alignment: int = 128
) -> list[int]:
    """SolverToPermutation(FirstFitLayoutSolver(deepcopy(buffers), ...)).permutation(buffers)."""

def free_gap(
    buffer: LifetimeBoundBuffer,
    placed: Sequence[LifetimeBoundBuffer],
    limit: int,
    alignment: int = 128,
) -> Optional[Gap]:
    """The first gap fitting buffer.size, as _pick_gap(_build_gaps(...)) answered."""
```

`first_fit_place` keeps `_check_in_place_relationships`, the `address is None` assertion, the
`_sort_key`, `_topological_sort` and the fallback rebuild with
`reuse_in_place_parents=False`. It inlines `MemoryPlanSolver.excluded` with
`tilings_from_menu()` false: `residency_reason`, then `untiled_bar`, then
`min_footprint > limit`, each written to `spill_reason`. Any reorder here changes every SA
plan, so the gate is exact.

### Staying, with reasons

- **`_distribute_work` stays.** `LX_PLANNING=0` has no other source of divisions
  (`passes.py:427`, `:574`). Pinned ops read their committed split through
  `_fixed_core_division` (`allocator.py:2370`, `:3671-3819`). The SA seed starts "every op at
  its committed division" (`sa_cooptimizer.py:1128-1131`), and the failure path that commits
  nothing relies on it. If #5695 has landed, C4 rewrites `solve_decides_work_division()`,
  whose contract names `select_allocator`. It keeps Stage 9's rule, false on every family,
  unless Stage 11 has landed, which makes it true with `lx_sticks_in_solve` on. The
  `divide_deferred_op` call in `_greedy_replan` goes with it: with the predicate false nothing
  is deferred, and with it true Stage 11 owns the SA seed, fallback and infeasible paths.
- **The beam stays.** The pre-pass is the source of pinned steps, with or without Stage 11,
  and issue #5682 keeps it for `LX_PLANNING=0` and solve failure. `op.layouts` and
  `restick_cost_fn` must therefore stay legal.
- **Stick rules stay** (D1). What changes is that K and N stop being derived twice:
  `_matmul_layouts` calls Stage 2's `matmul_axes`, and a new agreement test pins the
  generators to `stick_tables`.
- **`elide_proven_read_copies` stays.** No stage deletes it, `read_copy_elision` keeps its
  readers, and this stage does not touch the pass.

**After this stage, `LX_PLANNING=0` runs** what it runs today: the stick passes with the beam,
`span_reduction`, `_distribute_work`, then no planning and no `check_decisions`.
`validate_op_specs` is its only stick check, one more reason `_check_stick_all_same` stays.

**When ortools is missing.** The arch marker at `pyproject.toml:40` and `:59` stays.
`cpsat_expr_printer.py:40-46` and Stage 7's `family/ortools_family.py` guard the import, so
importing `FAMILIES` never fails. `select_families` promotes the fallback family with one
warning, which also removes the `ValueError` default compiles hit today.

**The debug greedy from PR #5 stays possible.** It needs a family slot, a plan with no solve,
and the same check. `_distribute_work` gives every op a division with no solve, Stage 7's
`Packer` packs plain ints, `FAMILIES` stays open, `check_decisions` is independent of the family,
no gate fails a family with an empty `ENCODES`, and `scratchpad_planning` names no family. The
value `"greedy"` is free for that family to take later.

### `scratchpad_planning` after

```python
def scratchpad_planning(graph: GraphLowering) -> None:
    """Plan and commit LX placement for a lowered graph."""
    # Imported locally: commit.py, which optimizer.py imports, imports this module.
    from torch_spyre._inductor.scratchpad.lx_allocator.optimizer import plan_allocation

    plan_allocation(graph, config)
```

```python
@_runs(anchor_lx_relayout_ownership, scratchpad_planning, plan_allocation,
       *optimizer_pass_sources())
def _maybe_scratchpad_planning(graph: GraphLowering) -> None:
    if not config.lx_planning:
        return
    anchor_lx_relayout_ownership(graph)  # C12 removes this line and the _runs entry
    scratchpad_planning(graph)
```

`optimizer_pass_sources()` is Stage 9's: every module under `lx_allocator/`, plus
`ilp_solver_ortools` and `sa_cooptimizer`, with Stage 10's `commit.py` and, if landed, Stage 11's
`sticks.py`. It already reaches the FX key. C4b, C9 and C10 drop `plan_adapter`, `sa_cooptimizer`
and `ilp_solver_ortools` from it, and C0 checks that it names nothing else this stage deletes.
Each commit changes hashed sources, so each moves the key.

`family_key(config)` stops returning `None`, because no old path remains. `select_families`
raises `ValueError` on an empty `fallback_solver`, before any solve, so `fallback` is `None` only
on a promoted family. `SolveError` goes from its body and from `_fall_back`, whose two raises
become `RuntimeError`: Stage 9's step 1 names the status, and step 4 names both statuses.
`INFEASIBLE` from the fallback still reaches `_return_infeasible`. SA reports `FEASIBLE` on
every solve, so on the default configuration either raise is a bug. A non-SA fallback with no
plan raises too, because no re-plan remains. With Stage 11 landed, `plan_sticks` loses its
`SolveError` branch in C4.

`LxContextSwitchingPass` runs inside `commit(graph)` for both families (surprise 2,
Consumes). Its import of `ScratchpadOptimizationPass` from `allocator.py` stays, and the
local-import comment at `allocator.py:5421` goes with `select_allocator`.

### Config

| value | readers after this stage | action |
|---|---|---|
| `layout_solver` | `family_key`, `tools/cost_model/run_decode_matmul_sweep.py:165`, `tools/lx_planning/lx_semantic_plan.py:234` | `Literal["ortools", "sa"]`, as below. Rewrite the comment at `config.py:347-363`. |
| `allow_exhaustive_search` | none | Remove (`config.py:369-377`), plus `api/torch_spyre.rst:754-761`. `config.py` warns at import if `ALLOW_EXHAUSTIVE_SEARCH` is set, since a stale variable is otherwise ignored silently. |
| `co_optimizing_lx_planning` | `plan_allocation`, `materialize_hinted_views.py:135`, `lx_relayout.py:1903` until C12 | Keep. False is committed mode, `graph.divisions_committed`. Comment rewritten. |
| `cpsat_time_limit_seconds` | `SolveSettings.from_config(config)` | Keep. Fix the "falls back to greedy" text at `config.py:379-384`. |
| `fallback_solver` | `select_families` | Keep, default `sa`. `""` now raises `ValueError`. |
| `lx_solver_completed_reduction_routes` | Stage 6's table | Keep, on. |
| `lx_sticks_in_solve` | Stage 11, if landed | Untouched. Its final value is Stage 12's 12c. |
| `lx_route_to_families`, `lx_check_decisions`, `lx_check_file` | none | Remove in C4. |
| `read_copy_elision`, `enable_uniqueness_check`, `_cpsat_warn_on_cost_expr`, `lx_solver_relayout`, `lx_solver_relayout_groups_per_edge`, `lx_solver_relayout_presolve_max_copies`, `auto_coarse_tiling`, `dump_cost_expr_file`, `lx_collective_output_staging`, `allow_all_ops_in_lx_planning`, `dxp_lx_frac_avail` | Stages 2 to 10, as Stage 12's settings section lists | Keep. |

The alias code is Stage 12's contract for retired values:

```python
# torch_spyre/_inductor/config.py; removed one tagged release after this stage
def layout_solver_key(value: str) -> str:
    """Map a retired value to its family key, warning. Any other value passes through."""
    key = {"cpsat": "ortools", "simulated_annealing": "sa"}.get(value, value)
    if key != value:
        warnings.warn(f"layout_solver={value!r} is retired, use {key!r}", stacklevel=2)
    return key

layout_solver: Literal["ortools", "sa"] = layout_solver_key(
    os.environ.get("LAYOUT_SOLVER", "ortools")
)  # type: ignore[assignment]
```

The stored value is normalized, so the FX key moves. `family_key` applies the same
function for a value set by `config.patch`. A key outside `FAMILIES` raises `ValueError` naming
the valid keys, as `allocator.py:5413` does, and `greedy`, `bestfit` and `firstfit` also name the
default. The map sits in a function because `install_config_module` turns a module-level dict
into a config entry, which would enter the FX key. One tagged release later a follow-up commit,
named in this PR, deletes the function, and `cpsat` then raises like any other value.

## Changes

| file | change |
|---|---|
| `torch_spyre/_inductor/scratchpad/allocator.py` | Classes out, `LxRecordBuilder` in, exhaustive chain out, `select_allocator` family out, `scratchpad_planning` rewritten |
| `lx_allocator/optimizer.py`, `rules.py`, `check.py`, `commit.py`, `family/sa/search.py` | `family_key`, `select_families` and `_fall_back` as above; host; `_parent_residency_edges` call, now Stage 6's moved function; `_loop_carry_update_edge` import; `_seed_permutation` |
| `lx_allocator/plan_adapter.py` | Deleted |
| `scratchpad/greedy_solver.py`, `exhaustive_search.py`, `simulated_annealing.py`, `cooling_schedules.py`, `ilp_solver_ortools.py`, `sa_cooptimizer.py` | Deleted |
| `scratchpad/firstfit_bestfit_solver.py` → `scratchpad/first_fit.py` | Class to functions, best-fit removed |
| `scratchpad/plan_solver.py` | `MemoryPlanSolver`, `CoreDivisionLayoutSolver`, `SolveError` out; comments fixed; later `paired_with`, `lx_relayout_plans` |
| `scratchpad/lx_relayout.py`, `coarse_tiling.py`, `residency_rules.py` | Collector and anchor removed (gated); Stage 10's wrappers out; certificate path |
| `scratchpad/utils.py`, `lx_context_switching.py` | Plot helpers out, docstring |
| `inductor/passes.py` | `_maybe_scratchpad_planning`, imports |
| `inductor/config.py` | Config values, `layout_solver_key`, comments |
| `inductor/work_division.py` | `solve_decides_work_division()`, if #5695 has landed |
| `inductor/propagate_layouts.py` | `_matmul_layouts` calls `matmul_axes`; comment at `:2053-2055` |
| `inductor/pass_utils.py:4191`, `scratchpad/coarse_tiling.py:463`, `scratchpad/lx_relayout.py:899` | Stale names in docstrings and comments |
| `tests/inductor/lx_equivalence.py`, `lx_corpus.py`, `lx_equivalence_golden.json`, `tools/lx_planning/lx_gate.sh` | Old arm out, golden on `fake` |
| `tests/inductor/*`, `tests/configs/torch_spyre_tests/inductor/*` | See Tests |
| `.github/workflows/lx_planner_compare.yaml`, `_test_matrix.yaml`, `.github/runner_overrides.yaml`, `.github/suite_test_flags.yaml`, `tools/lx_planning/compare_planners.py`, `planner_compare_reasons.json` | Per "Temporary items" |
| `AGENTS.md:126`, `docs/source/api/torch_spyre.rst`, `docs/source/compiler/{scratchpad_planning,co_optimizer,sa_co_optimization,cost_model,inductor_frontend,index}`, `docs/source/user_guide/debugging/failure_classes.md:163`, `docs/source/_static/images/lx/allocator-architecture.svg` | Rewritten for the new names, with Stage 12's kill-switch text removed |
| `docs/source/compiler/simulated_annealing_layout.md`, `docs/source/user_guide/examples/scratchpad/{toy_layout,random_buffers,inplace_annealing}.py` | Deleted, with `index.rst:68`, `examples/index.md:53-55` and links from `co_optimizer.md:669`, `sa_co_optimization.md:10,255`, `scratchpad_planning.md:636`, `native_packer_performance.md:17` |
| `docs/source/user_guide/examples/profile_ops.py` | Allocator branch out |

### Commits

Each commit leaves the full serial unit suite green, and each carries its own test changes,
because a test importing a deleted name fails at collection. Commits are signed off (`-s`).

| # | commit | why this order |
|---|---|---|
| C0 | Audit, not a commit. Scan the AST for every deleted name, every row of "Temporary items" and every allocator private method referenced from outside the deleted set, counting `lx_equivalence.py` and `lx_corpus.py` as importers, and paste the result in the PR. The script stays out of the repo. | Tells which of C2, C3, C8 and C13 are no-ops. A hit that no row explains stops the PR. |
| C1 | Retarget tests pinned to removed values to `ortools` and `sa` (`test_work_division_hint.py:1492,1593,1657,1735,2163`, `test_fallback_kernel.py:441`, the `solver_method` axes at `test_scratchpad_use.py:536,849,1200`). Tests only. Stage 12's soak dispatches this branch before 12b. | Both paths still exist, so any failure is a real gap in Stage 6, not deletion damage. |
| C2 | Add `LxRecordBuilder`. Move the residency group. Old classes delegate. `rules.py`'s host and `_UNBUILT_SOLVER` test sites move to the builder or the moved functions. `TRANSFER_CERTIFICATES` path in the same commit. | Old classes still work, so the suite is green and the harness diff is empty. |
| C3 | Extract `first_fit.py`. `FirstFitLayoutSolver` and `SolverToPermutation` become thin wrappers over it. `_seed_permutation` and `collective_output_stages` call functions. | Must precede C6, since `_seed_permutation` needs `SolverToPermutation` until then. |
| C4 | Remove the C4 rows of "Temporary items" and the `SolveError` handler. Rewrite `scratchpad_planning`, `_maybe_scratchpad_planning`, `family_key`, `select_families` and `_fall_back`, and `solve_decides_work_division()` if #5695 has landed. Delete `select_allocator`, `_cpsat_solver_class`, `_PLACEMENT_SOLVERS`. Narrow `layout_solver`, add `layout_solver_key`, remove `allow_exhaustive_search`. Rewrite or delete `TestSelectAllocator`, `TestSolverCapabilities`, `TestCpsatFallbackWarning`, `TestSolveErrorFallback`, `TestCpSatTimeoutFallback`, `TestCpSatAllocatorFallback` and the old-path rows of `test_planner_default.py`. Update `profile_ops.py`. List old-value plan changes in the PR. | From here old classes are unreachable in production, and tests build them directly. |
| C4b | Retire the old arm. `bless --tier fake` on the fixed set, recording decisions and predicted total, replaces `test_golden_intree`, and the other `intree` tests move to the fixed set. Move `Cell`'s default, `ZOO_CELLS` and `MATRIX_CELLS` to `ortools` and `sa`, and drop `ALLOW_EXHAUSTIVE_SEARCH` from `Cell.env`. Delete the C4b rows of "Temporary items", and `plan_adapter` from `optimizer_pass_sources()`. | Each runs an old solver as its reference, so it would fail at C6 to C10. Must precede C5. |
| C5 | Delete `exhaustive_search.py`, `prune`, and the exhaustive-only chain. | Top of the solver stack, nothing else depends on it. |
| C6 | Delete `simulated_annealing.py`, `cooling_schedules.py`, the doc examples, `plot_buffers`, `quality_plot`, and `test_simulated_annealing.py` with its config yaml. | Removes the last importer of greedy and best-fit outside `allocator.py`. |
| C7 | Delete `greedy_solver.py`, `BestFitLayoutSolver`, the two pattern classes and solver subclasses. | Depends on C6. |
| C8 | Delete `ScratchpadAllocator`, `CoOptimizingAllocator`, placement prep, the committed-relayout methods, the delegates and Stage 10's wrappers. `commit.py` imports Stage 6's `_loop_carry_update_edge`. | Must precede C9: their signatures name `CoreDivisionLayoutSolver`. |
| C9 | Delete `sa_cooptimizer.py` with its aliases and its `optimizer_pass_sources()` entry. Rewire `_work_slices` sites. Delete the wrappers from C3. | `SAFamily` has been the only production path since C4. |
| C10 | Delete `ilp_solver_ortools.py` and its `optimizer_pass_sources()` entry. Port `TestCpSatDivisionGate`, `SolveStatsTest`, `UniquenessCheckTest`, `TestCpSatUnallocatedReads`. | Same. |
| C11 | Delete `MemoryPlanSolver`, `CoreDivisionLayoutSolver`, `SolveError`. | Now has no subclass and no raiser. |
| C12 | Gated, and only with `lx_solver_completed_reduction_routes` on by default: delete `collect_lx_relayout_plans`, `anchor_lx_relayout_ownership`, `paired_with`, `lx_relayout_plans`. | Last rule-bearing code, kept apart so it can be dropped from the PR. |
| C13 | Delete buffer record types with no reader. Remove `compare_planners.py` and its row if 12c has landed. Rerun C0 over "Temporary items". | After every old solver is gone. |
| C14 | `_matmul_layouts` calls `matmul_axes`, keeping its `Unsupported` at `:1565`. Add the stick agreement test. | Independent, and small enough to review alone. |
| C15 | Docs, `AGENTS.md`, stale docstrings. Run `generate_residency_rules.py`. | Last, so links name what exists. |

## Interfaces

Consumes:

| symbol | signature | from Stage | used for |
|---|---|---|---|
| Equivalence harness | `lx_gate.sh`, `diff`, `bless --tier fake`, `Cell`, `ZOO_CELLS`, `FIXED_GRAPHS`, tiers `fake`, `unit-*`, `failure` | 0 | Every commit's gate. Old arm retired in C4b. |
| `decode_todays_plan`, `check_todays_plan`, `check_buffers`, `Mismatch`, `PlanCheck` | As Stage 3 | 3 | Deleted in C4b. |
| `check_decisions`, `RuleViolation`, `coverage` | `check_decisions(graph)`, `coverage(family, target) -> Coverage(gaps, unused)`, as Stage 3 | 3 | Run on every plan. Untouched. |
| `rules.stick_tables` | `(node, flag) -> list[StickTable]` | 3 | Agreement test. |
| Division-rule bodies | One body in `rules.py`, old `work_division_constraints.py` names as adapters | 3 | Rule audit. |
| Residency host | A never-solving allocator behind `rules.residency_reason` | 3 | Replaced in C2. |
| `matmul_axes` | `(op: ComputedBuffer) -> MatmulAxes`, `.k`, `.n`, pure, in `lx_allocator/graph_builder.py` | 2 | C14. |
| `build_graph` | `(ir, target, *, divisions_committed=False, drain_plans=None, lx_relayout_plans=())` | 2 | C12 drops the last keyword. |
| Moved candidate code | `lx_allocator/family/edge_tables.py`; allocator delegates | 5, 6 | Delegates go in C8. |
| `replay_graph`, `edge_tables_from_buffers` | `replay_graph(buffers, limit, alignment)`; final test helpers | 5, 6 | `solve_records` below. |
| `Family`, `Encoding`, `FAMILIES` | `FAMILIES: Mapping[str, type[Family]]`, keys `"ortools"`, `"sa"`, in `lx_allocator/family/registry.py` | 5, 7, 8 | Selection. |
| Guarded ortools import | `OrToolsFamily` raises `ImportError` when `cp_model` is `None`; importing `registry` never raises | 7 | Missing-ortools path. |
| `AnnealSearch._seed_permutation` | `(sizes: list[int]) -> list[int]`, in `lx_allocator/family/sa/search.py` | 8 | Calls `first_fit_order` after C3. |
| `family_key`, `select_families` | `family_key(config) -> str \| None`; `select_families(config) -> tuple[Family, Family \| None]`, in `lx_allocator/optimizer.py`, promoting the fallback when ortools is absent | 9 | `scratchpad_planning`, `plan_allocation`. `family_key`'s `None` return removed in C4. |
| `plan_allocation` | `plan_allocation(ir, config) -> SpyreOptimizationGraph` | 9 | Sole path. |
| `_fall_back`, `_return_infeasible` | Out of time retries the fallback; a proven infeasible graph commits nothing | 9 | Replaces the re-plan. |
| `optimizer_pass_sources()` | Planner sources in the FX key | 9, 10 | `_runs`. Loses the modules deleted here. |
| CI wiring | `_test_matrix.yaml` inputs `lx_route_to_families` and `layout_solver`, their export step, `lx_planner_compare.yaml` | 9, 12 | Per "Temporary items". |
| `lx_route_to_families`, `_greedy_replan` | As Stage 9 | 9 | Removed in C4. |
| `solve_decides_work_division()` | False on a family; with Stage 11 landed, true with `lx_sticks_in_solve` on | 9, or 11 if landed | C4. |
| Committed mode | `graph.divisions_committed`, a target derived with `SpyreCostModel(traffic_only=True)` | 1, 2, 4, 9 | Placement-only replacement. `division_is_fixed` branches stay. |
| `commit` | `commit(graph)`; replaces `_push_allocation`, including drains, hoisted LX prefetches, tilings, collective staging, spill reasons; runs `_context_switching_passes()` | 10 | Commit group gone. Surprise 2. |
| `cost_expr_record` writer | Full record from decoded graph values, `env.solver` and `env.allocator` keys included | 10 | Readers of the dump. |
| Stage 10's extractions | `push_one_placement`, `commit_chosen_divisions`, `check_priced_footprint`, `accepted_lx_views`, `collective_output_stages`, `apply_tiling_choices`, `splits_from_divisions` | 10 | Wrappers go in C8. |
| Completed-reduction relayouts | Stage 6 table calls `derive_completed_reduction_routes`; `lx_solver_completed_reduction_routes` on by default | 6, 12 | Retargeted device tests, C12. |
| Settings and retired-values tables, rollout step 6 | `layout_solver` default `"ortools"`, `fallback_solver` default `"sa"`; aliases as Stage 12's contract | 12 | Entry, Config, C4. |
| `LEGACY_LAYOUT_SOLVERS`, `warn_planner_selection` | As Stage 12 | 12 | Removed in C4. |
| `compare_planners.compare`, `Thresholds`, label `lx_compare`, lane cron | As Stage 12 | 12 | Removed by whichever of this stage and 12c lands last. |

Provides:

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `scratchpad_planning` | `(graph: GraphLowering) -> None` | none | final |
| `first_fit_place`, `first_fit_order`, `free_gap` | See Design | none | final |
| `LxRecordBuilder` | See Design | none | final |
| `config.layout_solver` | `Literal["ortools", "sa"]` | a debug family may extend it | final |
| `layout_solver_key` | `(value: str) -> str` | none | temporary, removed one tagged release after this stage |
| `lx_equivalence_golden.json`, `test_golden_fake` | `fake` compiles of the fixed set: decisions and predicted total | 11 if later, #5682 | final |
| `solve_records` | See Tests | none | final |

## Tests

Counts are from an AST scan: "direct" means a test function whose body names a deleted symbol.
All files are in `tests/inductor/` and are unit unless marked device.

| file | direct / total | disposition | is coverage replaced? |
|---|---|---|---|
| `test_simulated_annealing.py` | 6 / 18 | **Delete** with its yaml. | `CoolingScheduleTests` (10) pin dead code. `SimulatedAnnealingTests` and stress pin placement SA: best-seen, early stop and determinism map to Stage 8's `SAFamily` tests. `test_annealing_step_swap_handles_evicted_buffers` has no replacement and needs none. |
| `test_scratchpad_patterns.py` | class-level | Keep `TestVerifyPatterns` and `TestFirstFitPatterns` on `first_fit_place`. Drop Greedy (xfails 5 of 7) and BestFit. Port SA patterns. | SA classes port through `solve_records`. The two eviction xfails stay. |
| `test_scratchpad_solver.py` | 16 / 144 | Delete `TestExhaustiveSearchResidency`, `TestGreedyLayoutSolver`, `TestBestFitLayoutSolver`, `TestSimulatedAnnealingLayoutSolver`. Keep `TestFirstFitLayoutSolver`, `ScoreOrderingTests`, `TestTopologicalSort`, `TestCoreDivision`, printer, `TestInvScale`, `TestObjectiveCostParams`, drain tests. Rewrite `JointDivisionSolverTests` (5), `TestCpSatJointDivision` (21), `TestCpSatPlacementOnly`, `SolveStatsTest`, `UniquenessCheckTest`, `TestCpSatUnallocatedReads`, `TestCpSatDivisionGate` on `solve_records` with `OrToolsFamily`. | Rewrite required. The 35 shared `BaseLayoutSolverTests` keep their bodies and change only `solve()`. `TestAllocatorSpillReasons` moves to Stage 10's reasons. |
| `test_scratchpad_use.py` (device) | 25 / 39 | See below. | Rewrite. |
| `test_work_division.py` | 16 / 150 | `_UNBUILT_SOLVER` sites use `LxRecordBuilder(decides_relayouts=True)` or the moved function. `TestResidencyEdgeInversion` (8) ports with Stage 8. Tests of `prune=True` at `:3526`, `:3662` delete. | Kept by construction change, as methods move verbatim. |
| `test_work_division_hint.py` | 9 / 69 | Five greedy-pinned tests retargeted in C1. Plan-handoff tests at `:720-830` delete with C12. | See "Behaviour at risk". |
| `test_solver_relayout_decision.py` | 9 / 33 | `_cap_relayout_groups` calls Stage 6's function, `_relayout_copy_buffers` ports to `RelayoutGroup`, and the cache-prefix test loses `lx_check`. | Kept. Certificates name these tests, so keep the names. |
| `test_lx_equivalence.py`, `test_lx_equivalence_compile.py` | Stage 0 | C4b as in Commits. `test_golden_intree` becomes `test_golden_fake`. | Yes, `fake` replaces buffer-level replay. |
| `test_planner_default.py` | Stage 12 | `test_every_old_value_reaches_the_old_path`, `test_cache_key_differs_between_old_and_new`, `test_legacy_value_warns_once`, `test_value_without_replacement_names_the_default`, `test_exhaustive_flag_warns_only_on_a_family` delete in C4. | `test_scratchpad_entry.py` below. |
| `test_sa_cooptimizer.py` | 26 / 111 | Each class ports with Stage 8. The PR lists each deleted class with its Stage 8 counterpart. Class constructed directly at 20+ sites, so most of the file is affected even where not counted. `test_ceiling_is_higher_than_the_layout_only_annealer` (`:778`) deletes. | Must be shown class by class. A class without a counterpart is ported, not dropped. |
| `test_cooptimization_types.py` | 8 / 18 | `_NoOpSolver`, `NoOpSolverABCTest` delete. `EngineBindingTest` deletes. `SpillCostTest::test_matches_the_cpsat_engine_formula` moves to `SpyreCostModel.spill_cost`. Capture tests stay while `CoreDivisionBuffer` does. | Partly. |
| `test_coarse_tiling.py` | 6 / 435 | Allocator sites use the moved functions, `:9727` patches `op_inputs_good_for_lx_inplace`, and the `FirstFitLayoutSolver` patches at `:10200`, `:10223` become `free_gap` in C3. `_check_in_place_relationships` import stays. | Kept. |
| `test_solver_auto_coarse_tiling.py` (device) | 5 / 34 | `select_allocator()` at `:716` becomes Stage 5's `tiling_candidates`. | Kept. |
| `test_solver_relayout_e2e.py` (device) | 4 / 25 | `_Observed` retargets. `collect_lx_relayout_plans` at `:94` needs C12's answer. | Kept. |
| `test_cost_expr_dump.py`, `test_op_features.py`, `test_cost_model_decode.py`, `test_restickify_cost_model.py`, `test_scratchpad_utils.py` | 4/12, 3/22, 0/60, 2/21, 1/22 | Constructor changes. `test_cost_model_decode.py` reaches the class through a helper. | Kept. |
| `test_perm_layout_solver.py` | 1 / 81 | Only the exhaustive half of `test_loop_start_override_does_not_change_unrelated_preloop_plan` (`:911`) goes. | Oracle and native differential stay untouched. |
| `test_fallback_kernel.py` | 1 / 19 | `test_cpu_pointwise_chain_compiles_co_optimizing` moves to `ortools`. Its comment says the joint CP-SAT path lacks this guard, so it may fail. | Possibly a real gap. Stage 12's C1 dispatch finds out. |
| `reference_perm_layout.py`, `synthetic_cooptimization_graphs.py`, `cooptimization_capture_loader.py` | n/a | Unchanged while the buffer classes stay. | |
| `test_lx_inplace_layout.py`, `test_op_spec_validation.py` | 0 | Unchanged. They patch or exercise functions that stay (D1). | |

`test_scratchpad_use.py` by class. The `solver_method` axes become `ortools` and `sa` (`:536`,
`:849`, `:1200`), so `ParameterizedScratchpadUsage`'s 40 generated cases become 16.
`TestCloneAtGraphBoundaries` gains `sa`, because its axis would otherwise shrink to `ortools`
alone. `TestCpSatAllocatorFallback` now asserts ortools-absent compiles via SA and that the
default config no longer raises. `TestCpSatTimeoutFallback` counts `SAFamily` solves
instead of greedy plans, and keeps `hbm_with_lx < hbm_without_lx`. `TestSolveErrorFallback`
becomes the Stage 9 fallback and keeps its check that context switching survives it.
`TestSelectAllocator` keeps its bogus-value case and loses the rest.
`TestSolverCapabilities` and `TestCpsatFallbackWarning` go, the former replaced by
`coverage()` tests. `test_exhaustive_search_guard_error_text` and
`test_cpsat_co_optimization_guard_with_ortools_forced_absent` go, and the second is inverted
by this stage's new test. `TestInplaceEdgeGate` calls `inplace_edge_ok` (`:2033`).

**Behaviour at risk of quiet loss.** Each is called out in the PR description.

| behaviour | pinned by today | replacement |
|---|---|---|
| Completed-reduction relayout on device | `test_completed_reduction_relayout_device`, `test_unhinted_moe_down_route_preserves_the_chosen_split`, greedy only | Retarget to `ortools` with Stage 6's table, which needs `lx_solver_completed_reduction_routes` on. `completed_producer_cores` assertions stay. |
| Consumers sharing a relayout destination, grouped relayouts, read expansion | `test_lx_relayout_consumers_share_destination_view`, `test_grouped_lx_relayout_device`, `test_lx_relayout_read_expansion_device`, greedy only | Retarget. Under a priced solve a relayout is chosen, not guaranteed, so a test may need a graph where the copy pays. It must not be weakened to "compiles". |
| Atomic relayout group | `test_lx_relayout_allocation_is_atomic_in_one_greedy_solve` | Group atomicity is a model constraint. Needs a `check_decisions` case, or a Stage 6 test by name. |
| Relayout lifetime extension | `test_lx_relayout_copies_loop_lifetime_to_every_destination`, `test_lx_relayout_keeps_source_lifetime_for_later_original_reader` | Port to `RELAYOUT` steps. |
| Spill reasons in logs | `TestAllocatorSpillReasons` | Stage 10. |
| Placement-only SA quality | `test_annealing_never_worse_than_initial` | SA family's best-seen test. Plans change (D7). |
| Stick agreement across op kinds | `_check_stick_all_same` over reductions and dtype ops | Stays. It reads emitted OpSpecs, which no flag sees (D1). |

New tests:

| file | test | asserts |
|---|---|---|
| `test_scratchpad_entry.py` | `test_lx_planning_off_skips_plan_allocation` | With `lx_planning=False`, `_maybe_scratchpad_planning` touches nothing and never calls `plan_allocation`. Replaces `test_lx_anchor_pass_respects_planning_switch`. |
| | `test_each_layout_solver_value_reaches_its_family` | Each key of `FAMILIES` selects its family. |
| | `test_removed_layout_solver_values_raise` | `greedy`, `bestfit`, `firstfit`, `bogus` raise `ValueError` naming the valid keys. |
| | `test_retired_aliases_normalize_and_warn` | `cpsat` and `simulated_annealing` store `ortools` and `sa` from the env in a subprocess, warn, and reach the same family through `config.patch`. |
| | `test_stale_exhaustive_search_env_warns` | `ALLOW_EXHAUSTIVE_SEARCH=1` warns at import. |
| | `test_empty_fallback_solver_raises` | `fallback_solver=""` raises `ValueError`. |
| | `test_ortools_absent_promotes_the_fallback_once` | Inverts today's guard test. Default config compiles. Two resolutions log once, with the import error. |
| `test_first_fit.py` | `test_first_fit_order_golden` | Literal permutations recorded from `SolverToPermutation` at C3 on the synthetic graphs. |
| | `test_first_fit_excludes_barred_buffers` | The three `excluded` reasons, in order, written to `spill_reason`. |
| | `test_free_gap_matches_pick_gap` | Equal to the old composition, recorded at C3. |
| `test_stick_rule_agreement.py` | `test_matmul_layouts_satisfy_matmul_sticks` | `_matmul_layouts` output and required layouts fall inside `stick_tables(node, MATMUL_STICKS)`, including `M = 1` and `N = 1`. |
| | `test_pointwise_candidates_satisfy_pointwise_sticks` | Every `op.layouts` entry is allowed by `POINTWISE_STICKS`. |

Each new test file gets a config yaml under `tests/configs/torch_spyre_tests/inductor/`.

Record-level tests need one new seam, because their records never had a `GraphLowering`. It is
built on `replay_graph` and `edge_tables_from_buffers`:

```python
# tests/inductor/lx_record_fixtures.py
def solve_records(
    records: Sequence[LifetimeBoundBuffer],
    capacity: int,
    family: type[Family],
    alignment: int = 128,
) -> list[LifetimeBoundBuffer]:
    """Encode and solve hand-written records, and write addresses back onto them."""
```

**Suites run.** Unit, serially: every suite in the parent's Verification list that still
exists, plus `test_op_spec_validation`, `test_stick_rule_agreement`, `test_first_fit`,
`test_scratchpad_entry`, `test_lx_equivalence`. Device, on the final commit and on the C1
dispatch: `test_scratchpad_use`, `test_work_division`, `test_work_division_hint`,
`test_solver_relayout_e2e`, `test_solver_auto_coarse_tiling`, `test_lx_planning_suite`,
`test_lx_context_switching`, `test_fallback_kernel`, and the `test_inductor_ops_lx_*` shards.

**What proves the claim.** The name scan below finds nothing, the harness diff is empty on family
cells, and both suites pass with no skip added.

## Parity and gates

Parent: "**Phase 3 exit:** the optimizer is the default, and the replaced solvers and the
duplicated rules are gone." Verification: "**Every stage:** the Stage 0 harness, comparing
proto, decisions and committed plans between the stage's parent and the change."

- **Harness gate.** `lx_gate.sh` with HEAD's harness on both sides, base Stage 12's head, at
  Stage 0's deterministic settings. The gate compares family cells: `ortools` and `sa` ×
  co-optimizing and committed, at default and `AUTO_COARSE_TILING=1`, with the forced failure,
  at Stage 12's head against the same cells at each commit from C2 on. Per commit that is the
  fixed set, plus `intree` until C4b retires it. At the PR tip it is every tier left: `fake`,
  `matrix`, `unit-*` and `failure`; `zoo` stays a calibration at Stages 0, 9 and 12. Pass is
  byte-equal protos, decisions and committed plans. A difference is a bug in this PR, since
  nothing here is meant to change a family plan, and blocks the commit.
- **Old-value cells.** These are retired, not compared. At C4 `cpsat × 1` and
  `simulated_annealing × 1` alias onto arms Stage 12 held to identity (I1, I2), and
  `cpsat × 0` and `simulated_annealing × 0` become committed cells, under Stage 12's B1 and B2.
  The PR lists their plan changes against Stage 12's reversed lane on `total_us` bands, the SA
  committed change among them (D7). They block only if outside those bands.
- **C12 gate.** Runs only with `lx_solver_completed_reduction_routes` on by default. Family
  committed cells, same corpus. A plan difference is expected if the anchor mattered, so
  `hbm_traffic_bytes` takes Stage 12's B1 bands. If it fails them, or the flag is off, drop C12
  and keep the anchor and collector.
- **Name scan.** `git grep -nwE
  'ScratchpadAllocator|CoOptimizingAllocator|MemoryPlanSolver|CoreDivisionLayoutSolver|CpSatLayoutSolver|GreedyLayoutSolver|FirstFitLayoutSolver|BestFitLayoutSolver|ExhaustiveSearchSolver|SimulatedAnnealingLayoutSolver|SaCoOptimizingSolver|SolverToPermutation|select_allocator|allow_exhaustive_search|SolveError|lx_route_to_families|_greedy_replan|LEGACY_LAYOUT_SOLVERS|warn_planner_selection|lx_check_decisions|lx_check_file|decode_todays_plan|decisions_from_buffers|sizes_from_buffers|replay_intree|replay_capture|blank_for_capture'`
  across the repo returns nothing.
- **Dead-code scan.** The AST reachability script from C0 reports no new unreachable
  top-level name under `scratchpad/`.
- **Rule audit.** For each of the 16 constraints and each flag, the PR names one body (D6).
- **Exit wording.** "Duplicated rules are gone" cannot be met as written (D1, D6). It is met as
  one authoritative definition, with every other site a generator or validator tested against it.

After this PR the new path is the harness's reference, as mtsokol anticipated in PR #5.

## Risks

- **A retargeted test exposes a Stage 6 gap.** Most likely on the greedy-only relayout tests.
  Caught by Stage 12's dispatch of C1, before anything is deleted, and sent back to Stage 6.
- **A `patch` silently stops applying.** Keeping code in `allocator.py` avoids most cases, and
  a scan for `patch` targets in C0 and C2 catches the rest.
- **Local captures stop loading.** Pickles holding `RelayoutCopyBuffer` failed to load in #5676
  once the class went. The snapshot commits no pickle, and C13 cannot re-capture, so a type C0
  shows `load_captures` needing stays.
- **External users of `LAYOUT_SOLVER=greedy`.** Other repositories may set it, and I cannot
  see them from the snapshot. A compile then raises a clear `ValueError`.
- **Stage 11 lands first.** Any branch it keyed on the old path dies at C4. C0 lists them.
- **A dropped behaviour nobody listed.** The AST counts miss tests that reach a class
  through a helper. C0's scan works from production references, and the per-class port list
  for `test_sa_cooptimizer.py` is manual. That list is where I am least certain of coverage.

## Deviations from the parent plan

**D1. The three stick functions cannot be deleted.** The plan lists `_matmul_layouts`,
`_multi_arg_pointwise_layouts` and `_check_stick_all_same` as "held once in the flags". The
code shows otherwise:

- `_matmul_layouts` builds the output and required layouts and installs
  `FixedInOutNode` (`propagate_layouts.py:1608-1614`), and also runs for matmul mutations
  (`:3175`). `_multi_arg_pointwise_layouts` also does lane-0 index pruning, EA resolution,
  bool format resolution and in-place promotion (`:1909-2099`), none of which is a flag.
- The beam asserts every op with `layouts` has a cost function
  (`optimize_restickify.py:771`). Parent Stage 11 keeps the pre-pass, and #5682 keeps the
  beam for `LX_PLANNING=0` and solve failure.
- `_check_stick_all_same` runs on the emitted `OpSpec` at `after_simplification`, for
  `_pointwise_ops() | DTYPE_OPS | REDUCTION_OPS` (`op_spec_validation.py:668-672`). That is
  a different IR and a different stage, and it also runs under `LX_PLANNING=0`. Stage 3's
  `POINTWISE_STICKS` on `ReductionNode` tells the solve, not the emitted spec. Its sibling
  `_check_stick_matmul` (`:804`) duplicates `MATMUL_STICKS` the same way and the parent does
  not list it.

Resolution: all three stay. Flags are the authority for what the solve may choose,
`propagate_layouts` is the candidate generator, and `op_spec_validation` is the independent
last check. K and N are derived once (C14), and `test_stick_rule_agreement.py` keeps
generator and flags aligned. Removing them would need a replacement for `op.layouts` and the
beam, which is a stage of its own.

**D2. No stage owns the allocator's residency methods.** The plan says `plan_allocation` takes
over the classes' role, but `rules.py` reaches `_BufferResidencyChecks`, whose checks call
methods on the allocator (`allocator.py:815`, `:873`, `:940`). Resolution: `LxRecordBuilder`, a
small host keeping those bodies verbatim. Earlier stages moved everything else.

**D3. "First-fit lives on inside the backends" is two different things.** Parent Stage 8 calls
`NativePermutationLayoutSolver` the first-fit packer, but it stacks by permutation
(`perm_layout_native.cpp:17-23`). The priority-ordered gap search is
`FirstFitLayoutSolver`, which seeds SA. Resolution: `first_fit.py` as functions.

**D4. Buffer records are not on the list.** "No `Buffer` class" conflicts with first-fit and the
native packer, which read `LifetimeBoundBuffer`. Resolution: that type stays, the other two go
where C0 finds no reader, and no stage removes what is left.

**D5. Two functions not on the list become dead.** `collect_lx_relayout_plans` and
`anchor_lx_relayout_ownership`. If they stayed, `CO_OPTIMIZING_LX_PLANNING=0` would keep a
480-line oracle whose plans are thrown away. Resolution: C12, gated.

**D6. Division rules are shared, not taken over.** The parent's "takes over" reads as
replacement, but the pre-solve passes call `collect_work_division_constraints` before any graph
exists. Resolution: as Stage 3, one body in `rules.py`, the old names as adapters, and
`work_division_constraints.py` keeps its public entry point. The Rule audit gate covers all 16
rules, where the parent names only some.

**D7. Placement-only does not map at parity.** "The SA family with every division pinned" has a
different search from `SimulatedAnnealingLayoutSolver`: geometric cooling at
`sa_cooptimizer.py:118` against `SelfCalibratingReheatingSchedule`, and a different move set.
Plans for `LAYOUT_SOLVER=simulated_annealing CO_OPTIMIZING_LX_PLANNING=0` change at C4, when
that value stops meaning the old path. OR-Tools committed mode also differs from the old model
in objective, relayouts and residency branches. Resolution: accepted at Stage 12's bars. SA is
judged per corpus, median over five seeds, and is a listed plan change at C4.

**D8. `_stage_collective_outputs` is missing from the plan.** After the solve it places a spilled
matmul's output in leftover LX, while the drain clone stays in HBM, calling first-fit internals
(`allocator.py:3525-3646`). Resolution: Stage 10 keeps it as `collective_output_stages`, and C3
moves it to `free_gap`.

**D9. Stage 0's combination list shrinks.** "Every `layout_solver` × `co_optimizing_lx_planning`
combination" loses all five old values: three raise, and two alias for one tagged release. The
parent's "every stage" gate therefore holds on family cells, not on those combinations.

**D10. Stage 13 is absent from the parent's device-suite list.** This plan runs them anyway,
because it retargets device tests.

**D11. This stage does not follow Stage 11.** The parent orders Stage 11 first. Resolution: as
Stage 12, the old path goes with sticks pinned, and Stage 11 lands before or after this stage.

## Open questions

None.

## Out of scope

- The default flip and side-by-side runs: Stage 12.
- `elide_proven_read_copies`: stays, as Stage 11 decides.
- `commit(graph)` and how it carries drains, tilings and spill reasons: Stage 10.
- The debug greedy family itself: not staged. This plan only leaves room for it.
- Removing any remaining buffer records: not staged (D4).
- Unused native bindings (`overlaps`, `top_or_inf`, `finalize`): left in place.
- Moving `_distribute_work` or span floors into the solve: issue #5682 and PR #5695.
