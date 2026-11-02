# LX allocator, Stage 0: Harness and baseline

Parent plan: `plans/lx-allocator.md` at `5aff4a6`. Code base: torch-spyre `5268ba95`.
Depends on: none. Needed by: Stages 1-13.

## Goal

When this PR merges, torch-spyre has one equivalence harness that records every LX
solve and every committed plan under a deterministic solve, diffs two recordings, and
runs the regression guards from torch-spyre#5682 on a fixed graph set. Later stages call
it instead of building their own check. Compiles reach the planner on a fake device
wherever probe P1 shows they can, so those gates need no card, and the card calibrates
that device at Stages 0, 9 and 12. The in-tree graphs, the status and size guards and one
forced failure run in CI on every PR.
Everything lands under `tests/`, `tools/` and `docs/`. No file under `torch_spyre/`
changes, so plans, models and compile times stay exactly as they are.

## Today

Short paths: `ilp_solver_ortools.py`, `allocator.py`, `sa_cooptimizer.py`,
`simulated_annealing.py` and `cpsat_expr_printer.py` are under
`torch_spyre/_inductor/scratchpad/`. `config.py` and `passes.py` are under
`torch_spyre/_inductor/`.

**The solve.** `CpSatLayoutSolver._solve_and_record` (`ilp_solver_ortools.py:1325`) is the
one place the class
solves a plan. The only other `Solve` call is the optional uniqueness probe
(`ilp_solver_ortools.py:820`), off by default (`config.py:470`).
`_run` builds the parameters on one `CpSolver` before any solve: a 60 s wall-clock limit
(`ilp_solver_ortools.py:1376`, `config.py:385`), `config.get_cpu_count()` workers
(`:1399`), seed 0 (`:1405`) and the crashing subsolvers removed (`:1415`).

A solve is one of two shapes. With a cost expression there is one model
(`_minimize_cost_expr`, `:830`). Without one there is a ladder of up to five stages
(`:1457-1523`). The allocator withholds the expression under `auto_coarse_tiling`
(`allocator.py:3275-3299`), so tiling compiles always
run the ladder.

The ladder mutates one model. Each stage adds a lock whose right-hand side is the
previous stage's solved objective (`ilp_solver_ortools.py:1471`, `:1498`, `:1509`,
`:1521`). So the proto of stage n depends on the solver's answer at stage n-1, and an
identical proto at stage 2 needs an identical solve at stage 1.

`SolveError` is raised at two sites: `:872` (cost path) and `:1436` (ladder). Both format
`solver.WallTime()`, which raises `RuntimeError` if `Solve` never ran. A harness that
returned `UNKNOWN` without solving would, on the cost path, land in the `except
RuntimeError` at `:886` and quietly take the ladder instead of failing.

**The commit.** `ScratchpadAllocator.plan_allocation` (`allocator.py:1012`) is a fixed
template: prepare, solve, finalize relayouts, post-solve, spill reasons, push, post-passes
(`:1044-1052`). `_push_allocation` exists once (`:2036`) and `CoOptimizingAllocator`
inherits it. `_get_spill_reasons` (`:1135`, overridden at `:3648`) only reads, and it runs
immediately before the push. `scratchpad_planning` (`:5472`) catches `SolveError` and
replans with `ScratchpadAllocator(GreedyLayoutSolver, ...)` (`:5495-5515`).

**The cells.** `select_allocator` (`allocator.py:5377`) maps `layout_solver` ×
`co_optimizing_lx_planning` to an allocator and solver. `greedy`, `firstfit` and `bestfit`
with co-optimization need `ALLOW_EXHAUSTIVE_SEARCH=1` and run under
`ExhaustiveSearchSolver` (`:5436-5456`). The SA co-optimizing branch builds its allocator
without `post_optimization_passes` (`:5433`), so `LxContextSwitchingPass` does not run in
that cell. That breaks the invariant stated at `:5416-5420`. It is a bug on main, reachable
only with `LAYOUT_SOLVER=simulated_annealing`, and the baseline records it as it is.

**SA.** `SaCoOptimizingSolver` is seeded (`sa_cooptimizer.py:88`) and bounds steps, not
wall clock (`:94-98`). `SimulatedAnnealingLayoutSolver` defaults to `Random(0)`
(`simulated_annealing.py:215`). Both need `torch_spyre._C` for
`NativePermutationLayoutSolver`, but they do not start the runtime: a run over the
in-tree graphs ended with no `/dev/vfio` fd open.

**What already exists.**

- `tools/lx_planning/lx_semantic_plan.py` has `freeze` (`:39`), which refuses unknown
  types, and `graph_snapshot` (`:181`), which records final storage, dependencies, copies
  and placement. `capture_off_plan` itself does not fit, because `validate_contract`
  requires `lx_extended_ownership=False` (`:244`), and the default is on (`config.py:170`).
- `tests/inductor/cooptimization_capture_loader.py` and
  `tests/inductor/synthetic_cooptimization_graphs.py` give 18 buffer-level graphs: 7
  captures (softmax, mlp, swiglu, sdpa, rms_norm, simple_attn, flash_attention), 4 large
  captures, 7 synthetic. They have no cost expression, tiling or relayout copies.
- `tests/inductor/test_scratchpad_use.py:1476` (`_zero_solver_timeout`) already forces the
  `SolveError` fallback by zeroing the budget inside `CpSolver.Solve`.
- `TORCH_SPYRE_TIMING` (`torch_spyre/_inductor/timing_recorder.py`) writes per-pass
  records. `SPYRE_DUMP_COST` drives `cost_model_pass` (`torch_spyre/_inductor/passes.py:642`),
  whose result is `CustomPreSchedulingPasses.last_cost_report` (`passes.py:496`). It runs
  after the pass list, so a compile aborted at planning never produces it.
- The roadmap repo has `probes-cpsat-graph/zoo.py` (32 models),
  `deenum/timers.py` (solver-input capture, blanking at `:420`), `deenum/zoo_t.py` and
  `deenum/granite_t.py`. The earlier harness, `tests/inductor/cpsat_equivalence.py` from
  `9dab5027`, survives in `~/cpsat_equiv/base2_src/` with its drivers in
  `~/cpsat_equiv/scripts/`. It records CP-SAT solves only, and nothing after the solve.
- `~/cpsat_bench/swiglu_full/swiglu.py` compiles SwiGLU with no card: fake `spyre` tensors,
  `make_fx` then `compile_fx`, `TORCH_SPYRE_FRONTEND_ONLY=1`, `start_runtime` refused,
  `device_tensor_layout` faked, `/dev/vfio` refused by an audit hook and a watcher thread.
  193 of its 346 logs reach `COMPILE_DONE`, so it reaches the solver on SwiGLU. Stage 11's
  census on the same pattern, stopped after the beam, covered 29 of 32 zoo models:
  `sdpa_1x16x1024x64`, `sdpa_1x8x2048x64` and `sdpa_gqa_1x16x4x1024x64` raised "No
  RuntimeContext", and the beam placed no restickify on any of the 29. That proves SwiGLU
  only. The zoo through planning has not been run.

**Tests and the card.** `tests/inductor/conftest.py:56` is a session autouse fixture that
allocates on `spyre`, so plain pytest under `tests/inductor/` always opens the card. CI
test jobs run on card runners (`.github/workflows/_test_matrix.yaml:611-616`). With `import
torch` first and `--noconftest`, these eight suites passed here (664 passed, 5 skipped, 11
xfailed, about 95 s): `test_scratchpad_solver`, `test_solver_relayout_decision`,
`test_cooptimization_types`, `test_cost_expr_dump`, `test_sa_cooptimizer`,
`test_op_features`, `test_perm_layout_solver`, `test_scratchpad_patterns`. This host has a
card, so that does not prove they work without one. Commit 2 checks it by asserting no vfio
fd is open at the end.

**Measurements behind this plan.** All run on this snapshot from
`scratchpad/exp/stage-00/`, one worker, seed 0, no wall-clock limit. Zoo captures are the
October pickles from `main` 82edd8f6. They still load and solve here.

| question | result |
|---|---|
| Proto text, status, objective and decisions across `PYTHONHASHSEED=0,1,2,random`, separate processes | identical on 42 solves, 78 stages (the 36 in-tree solves plus 6 default zoo captures with cost expressions) |
| Same graph solved twice in one process | identical protos on all 18 in-tree graphs |
| Budget-stopped incumbent (`sdpa_4x8x256x64`, FEASIBLE at dtime 10) across hash seeds 0 and 7 | identical |
| `greedy`, `firstfit`, `bestfit` placement across hash seeds 0, 1, 2 | identical on 54 runs |
| SA co-optimizer and SA placement across hash seeds 0 and 1 | identical on 36 runs, 5 s, no card |
| `str(model.proto)` | under 4 ms per stage, 3 KB-517 KB of text. OR-Tools 9.15 has no `SerializeToString`, only text and `export_to_file` |
| Dtime 0 on a real solve | `UNKNOWN`, real `SolveError`, at ladder stages 0, 1 and 2 and on the cost path |
| Dtime 10 on three tiling captures | `SolveError` at ladder stage 3 (`division shape`). The October recording failed on the same three |
| Dtime 30 on the same captures, and on `sdpa_4x8` | all `OPTIMAL`, under 7 s wall each |
| `interleave_search=True` at one worker | different plan at equal objective on `mlp_swiglu_128`; no plan at dtime 10 on `sdpa_4x8`, where plain search finds one |

These do not cover the allocator itself: hash-seed stability of a live compile has not
been run. The one piece of evidence is that two zoo runs on device gave bit-identical plans
(`sa-vs-cpsat-REPORT.md:9`).

## Design

### Tiers

A solve runs from buffers. A commit needs a live `GraphLowering`, which a compile gives on
the card or on the fake device.

| tier | runs | corpus | needs | where it runs |
|---|---|---|---|---|
| `intree` | solvers over buffers | the 18 in-tree graphs, joint and placement-only, `cpsat` and `sa_legacy` arms: 72 solves | `_C` for `sa_legacy` | CI, every PR |
| `unit-host` | pytest plugin | the 8 buffer-only suites | `_C`, no runtime | CI runs the suites; the gate records them |
| `unit-device` | pytest plugin | `test_restickify_cost_model`, `test_coarse_tiling`, `test_work_division*`, `test_scratchpad_use`, `test_solver_auto_coarse_tiling` | card | gate |
| `fake` | fake compile | `lx_zoo.MODELS` minus `CARD_ONLY`, × `ZOO_CELLS` | `_C`, no card | gate; the fixed set in CI |
| `zoo` | card compile | all of `lx_zoo.MODELS`, × `ZOO_CELLS` | card | calibration at Stages 0, 9 and 12 |
| `matrix` | fake compile; card at calibration | `MATRIX_MODELS` × all 10 solver cells | `_C` | gate at Stages 0, 9, 10 |
| `failure` | fake compile; card at calibration | `mlp_swiglu_128x512x1536` with a forced failure, default and tiling | `_C` | one case in CI, both in the gate |
| `replay` | CP-SAT over pickles | pickles from this base's `fake` or `zoo` run, `cpsat` cells | no card | local fast loop only |
| `real` | card compile | Granite 4.0 micro, `max_new_tokens=2`, every planning call | card, weights, `hf-adapters` | gate, on the stages that ask for it |

`fake` proves parity, not production plans: fake inputs use default layouts, and both sides
of a diff share them. Gates that need a card only to reach `GraphLowering` use `fake`.
Numeric device suites stay on the card. A model the fake device cannot reach stays
card-only and is compared at calibration.

CI holds what the repo holds. `intree` runs against a committed golden (about 15 s for the
CP-SAT arm, 5 s for `sa_legacy`, measured here). `fake` compiles of the fixed set run in CI
as absolute checks: status, size, fallback. Base-versus-change diffs run through
`tools/lx_planning/lx_gate.sh` because CI has no base recording. Pickles are never
committed: they are large (7-28 MB per set), tied to one base, and regenerable.

**Golden.** `lx_equivalence_golden.json` holds the `intree` digests per case: proto shas,
status, objective and a decisions sha. A PR that changes the model runs `bless`, and the
change shows in its diff. Compile tiers have no golden: names move with every pass change.
`bless --tier fake` exists for the golden that replaces `intree` when `CpSatLayoutSolver`
goes.

**Zoo.** The zoo moves into torch-spyre as `tests/inductor/lx_zoo.py`, because a torch-spyre
PR cannot gate on a file in another repo. It has 35 models: the 32 unchanged, plus three.

- `flash_attention_1x8x256x64`, lifted from `test_flash_attention`
  (`tests/inductor/test_building_blocks.py:319`), since the issue's fixed set names flash
  attention and the zoo has none.
- `swiglu_transposed_128x512x1536`: SwiGLU through `F.linear` with weights stored
  transposed (#4208), as `_swiglu_case` builds them
  (`tests/inductor/test_solver_auto_coarse_tiling.py:514`).
- `gather_pool`: a gather feeding a windowed pool. No fixture for it exists in the repo, so
  its shapes are chosen in commit 1.

The last two are there because the rest of the zoo may have no restickify at all. Neither
is proven to produce one until probe P1 runs.

**Fixed graph set**, from the issue's guards, plus the two restickify graphs. The issue's
"one real model" is the `real` tier, which has no `fake` compile, so it is not a row here:

| issue name | solve level (`cooptimization_captures.json`) | compile level (`lx_zoo.MODELS`) |
|---|---|---|
| softmax | `softmax` | `softmax_256x1024` |
| mlp | `mlp` | `mlp_gelu_256x1024x4096` |
| swiglu | `swiglu` | `mlp_swiglu_128x512x1536` |
| sdpa | `sdpa` | `sdpa_1x8x256x64` |
| rms_norm | `rms_norm` | `rmsnorm_128x256` |
| flash attention | `flash_attention` | `flash_attention_1x8x256x64` |
| restickify, transposed | none | `swiglu_transposed_128x512x1536` |
| restickify, gather | none | `gather_pool` |

**Matrix models** are the eight from `same_plans.sh`: `softmax_256x1024`,
`rmsnorm_128x256`, `residual_rmsnorm_256x1024`, `pointwise_chain_512x1024`,
`mm_chain_256x512x1024x256`, `mlp_swiglu_128x512x1536`, `simple_attn_4x64x256x128`,
`sdpa_1x8x256x64`. Each cell runs at default config with `ALLOW_EXHAUSTIVE_SEARCH=1`.
A case over its 1200 s process timeout records `error: "timeout"`, on both sides.

### Fake device

`tests/inductor/lx_fake_device.py` lifts `swiglu.py`'s setup. It must load first in a fresh
process, because it sets `IS_INDUCTOR_SPAWNED_SUBPROCESS=1` and
`TORCH_SPYRE_FRONTEND_ONLY=1` before `import torch`.

- `guard_device()` installs the `/dev/vfio` audit hook and the fd watcher, which exits 99.
  An audit hook cannot be removed, so fake compiles always run in their own process.
- `install_fake_device()` refuses `start_runtime`, makes `device_tensor_layout` return
  `SpyreTensorLayout(list(shape), dtype)` for a fake `spyre` tensor, and folds device
  scalars on the host.
- `compile_fake(fn, args)` turns each zoo input into `torch.empty(*shape, dtype=dtype,
  device="spyre")` under `FakeTensorMode`, then runs `make_fx(fn, pre_dispatch=False)` and
  `compile_fx`.

**Probe P1 is the first commit.** It runs that setup over every model to the end of
`CustomPreSchedulingPasses`, records which models reach `_push_allocation` and why each
other one fails, counts `is_restickify_op` hits at the push, and compares final snapshots
with a card compile for the fixed set. The result fills `CARD_ONLY` and `FAKE_DIVERGES` in
`lx_zoo.py` and goes in the PR. P1 has not been run. If it reaches few models, the `fake`
tier shrinks and the rest stay on the card at calibration, and the schedule in Parity and
gates does not change.

### Hooks

All hooks are process-local monkeypatches installed by `install()` and restored by
`uninstall()`. None needs a production change.

| target | wrapper |
|---|---|
| `CpSatLayoutSolver._solve_and_record` (`ilp_solver_ortools.py:1325`) | On the first stage of a solve, read the production parameters into the record. Apply `SolveParams`, or dtime 0 if a `ForcedFailure` matches. Take `str(model.proto)` before the solve. Call the original. Record the stage. |
| `plan_layout` and `plan_layout_and_core_divisions`, for each class in `MemoryPlanSolver.__subclasses__()` (recursively) that defines them in `vars(cls)` | Outermost call only, tracked with a depth counter, because `ExhaustiveSearchSolver` calls an inner solver once per leaf. Run `on_solve_begin`; open a solve; write a capture if asked; call the original; close the solve with decisions read from the returned buffers. An exception closes the solve with `error` set and is re-raised unchanged. |
| `ScratchpadAllocator._solve` (`allocator.py:1090`), `CoOptimizingAllocator._solve` (`:3218`) | Remember the allocator and graph, so `on_solve_begin` can pass them. Both are `None` on a solver-only run. |
| `ScratchpadAllocator._push_allocation` (`allocator.py:2036`) | Before: observers, op names, `self._get_spill_reasons(graph, buffers)`. Call the original. After: `graph_snapshot(graph)`, build the `CommitRecord`, observers. |
| `insert_restickify_on_node_inputs` (`torch_spyre/_inductor/insert_restickify.py:350`) | Note which op names one call adds to `operations`. Those are the beam's restickifies. The patch replaces the global `insert_restickify` calls (`:586`), so the call from `torch_spyre/_inductor/enforce_indirect_access_layout.py:181`, through its own import, stays out. |
| `CustomPreSchedulingPasses.__call__` (`passes.py:583`), compile driver only | After it returns: read `last_cost_report.total_us`, take the final snapshot, raise `_PipelineDone`. |

`install(optional=...)` names targets that may be missing, because one harness runs on both
sides of a diff and a target can exist on only one of them. A missing target outside
`optional` raises. It defaults to `OPTIONAL_TARGETS`, so the CLI and the plugin get the same
list. That ships naming Stage 7's two targets under `torch_spyre._inductor.scratchpad`,
`lx_allocator.family.ortools_family.CpSatBackend._run_stages` and
`lx_allocator.family.placement.SolveSettings.from_config`.

The compile driver sets `torch._inductor.config.fx_graph_cache = False` and
`force_disable_caches = True` (as `probes-cpsat-graph/zoo.py:213-214`). A cache hit would
record nothing. It runs one cold process per model and cell, as `run_profile.sh` does. On
the card it unwraps `_PipelineDone` through dynamo's exception chain as `zoo.py:225-229`
does. It fails the case if no commit was recorded. Like the probes, a compile stops at its
first pipeline, so a model that partitions into several graphs records only the first. The
`real` tier records every planning call, which covers multi-graph compiles.

A recording hook must not change a test's outcome, with one known exception. Forcing the
parameters breaks two tests (four parametrizations in the October run) that assert
production's parameters: `test_relayout_solve_presolves_by_default` and
`test_relayout_solve_uses_available_parallel_search_workers`
(`tests/inductor/test_solver_relayout_decision.py:211`, `:236`). So recording runs are
never the pass/fail run. The gate runs each suite twice, as `record_tests.sh` did: once
plain for pass/fail, once with the plugin for recordings, deselecting `PLUGIN_DESELECT`.
A snapshot that fails in plugin mode (`test_coarse_tiling` calls `_push_allocation`
directly on stand-in graphs, `tests/inductor/test_coarse_tiling.py:9813`) records
`snapshot_error` and the test continues. In compile mode `snapshot_error` fails the case.

### What "byte-identical" takes

"Byte-identical" means `str(model.proto)` compares equal. That text is the only stable
form OR-Tools 9.15 offers, and it is cheap.

| source of nondeterminism | status | handling |
|---|---|---|
| Worker races, wall clock | real | `num_search_workers=1`, `max_time_in_seconds=inf`, `max_deterministic_time=30`, `random_seed=0`, set just before every `Solve` |
| Ladder locks depend on the previous answer | real, by design | Determinism above makes it reproducible. `diff` reports only the first differing stage of a solve as the cause and marks later ones as downstream |
| A budget that is too small | real, and worst | At dtime 10, three captures reach `UNKNOWN` at a later ladder stage. A compile would then record greedy plans on both sides and report parity on nothing. Default is 30, and `check_guards` flags any solve error outside the `failure` tier |
| `PYTHONHASHSEED`, Python sets | not observed in the solver or printer; not tested through the allocator | Every run pins `PYTHONHASHSEED=0`. The gate records one second time at `PYTHONHASHSEED=1` (G1). Any difference is a production bug to report, not something to hide |
| Sympy ordering | not observed. `Add`/`Mul` args sort deterministically, and the printer's one `free_symbols` iteration feeds a check that needs exactly one owner (`cpsat_expr_printer.py:872-879`) | Covered by the hash-seed runs |
| Sympy's global cache, process history | not observed in one process | `test_second_solve_in_process_is_identical`; compile tier uses a cold process per model |
| Global counters in names, solution hints | none: the printer's `_count` is per converter (`cpsat_expr_printer.py:198`), buffer names come from the compile, and neither the solver nor the printer adds a hint | Cold process per case. A hint would show in the proto text |
| Solve time in records | `solve_s`, `limit_s`, error text | Kept out of compared fields. Errors normalize `after N.NNs` and `0x...`, as the old `_error` did |
| Test order | real in plugin mode | Serial only. Case key is `nodeid#n`. No xdist |
| Environment | real | The manifest keys listed under Record format must match, or `diff` refuses to compare |

The gate pins `PYTHONHASHSEED=0`, `SPYRE_NUM_CPUS=1`, `SENCORES=32`, and leaves
`DXP_LX_FRAC_AVAIL` at its default. `SPYRE_NUM_CPUS=1` makes the production value recorded
in `production_params` stable across machines. It also matches what `validate_contract`
asks for.

**Why 30 and not 10.** The parent plan gives no number. The earlier harness used 10, and
in its October run 3 of 125 solves ended in `SolveError` (all at `division shape`) and 17
of 316 stages stopped FEASIBLE. At 30 those three reach `OPTIMAL` in under 7 s of wall
time. A stage that really hits 30 costs about 18 s, going by the 6 s seen at 10.
Calibration measures this on the current zoo, and `DEFAULT_DTIME` moves if a guard fires.

**A harness plan is not the production plan.** It proves a model or commit unchanged, and
says nothing about which plan a many-worker production portfolio picks on a given run. So
`production_params` is recorded before the override and compared, or the harness would hide
a change to production's own settings.

### Interleaved search stays off

`SolveParams.apply` sets `interleave_search = False` explicitly: with it on, the plan
changes (Deviations 1). On the family path `install` patches `SolveSettings.from_config` to
return `SolveSettings(DEFAULT_DTIME, workers=1, deterministic=True)`, because no config
route reaches one worker at a deterministic budget. That leaves interleaving unset, so the
family searches the same way. The class first exists at Stage 7, so the row lands there.

### Committed plan

`CommitRecord` is built after the push from `graph_snapshot`, which already handles
layouts, dependencies, the relayout registry and the op's `iteration_space_ownership`,
`op_it_space_splits` and `loop_info`. The full frozen snapshot is the comparison. The
named fields are a projection of it, for readable diffs. `inserted` is the ops present after
the push and absent before, with name, kind and reads, which covers the clone op of an LX
prefetch, output and drain clones and relayout ops. `relayouts` comes from `materialized_lx_relayouts(graph)`, with
source, destination, consumers and both addresses. `addresses` maps op name to
`layout.allocation["lx"]`, and `divisions` to frozen `op_it_space_splits`. `fallback` is
true when a `SolveError` closed the previous solve of this case. `restickifies` is under
Coverage.

`final_snapshot_sha` and `predicted_total_us` come from the end of the pipeline, in a
`FinalRecord`. That covers `LxContextSwitchingPass` and `elide_proven_read_copies`.
`SPYRE_DUMP_COST=1` is set in compile mode. It only reads the IR, as the comment at
`passes.py:630-634` states.

### Decisions are recorded without menu indices as the key

Later stages decode plans onto a graph that has no menu. So a `BufferDecision` carries the
division as data: splits by symbol name, reduction symbols and tile splits. The menu index
is a separate field. A family comparison drops it and the decision's relayouts, which the
family path leaves empty, with `ignore={"menu_index", "decision_relayouts"}`. `CommitRecord`
still compares relayouts.

### Forced failure

A `ForcedFailure(solve_index, stage_index)` sets `max_deterministic_time = 0.0` on the
matching stage and lets OR-Tools run. The solver returns `UNKNOWN` for real, `WallTime()`
works, and production raises its own `SolveError` from its own site. This is
`_zero_solver_timeout`'s approach with the budget that the harness uses. Default config
reaches `ilp_solver_ortools.py:872`, and `AUTO_COARSE_TILING=1` reaches `:1436`, so the two
`failure` cases cover both raise sites. `scratchpad_planning` then falls back, and the
`CommitRecord` shows `fallback=True`, `allocator="ScratchpadAllocator"` and
`solver="GreedyLayoutSolver"`. No SA failure is forced, because SA does not raise
`SolveError`.

### Zoo captures

`capture_dir` writes one pickle per outermost CP-SAT solve, at solver entry, in the format
`timers.py` wrote (`:432-443`), plus `schema` and `source_sha`. `residency_edges`,
`division_space` and `tile_reads` are blanked first, as `timers.py:420`. CP-SAT reads none
of the three, so replay is faithful, and `test_capture_replay_matches_live` proves it.
Import `torch` before unpickling.

A capture is a local fast loop for today's CP-SAT solver, nothing more.

- **SA cannot replay.** With `division_space` blanked it runs from menus
  (`sa_cooptimizer.py:1123`), with no `residency_edges` it uses pair tables (`:1081`), and
  a capture from a CP-SAT compile holds relayout copies SA never sees.
- **It is no route to the new encoder.** Replay stops at the buffers, and the graph holds
  live ops, so it does not pickle. `fake` compiles serve Stages 5 to 8 on the zoo. The
  in-repo graphs stay their CI arm, through Stage 5's `replay_graph` and Stage 6's
  `edge_tables_from_buffers`.

`replay` refuses a pickle whose `source_sha` is not the base being compared. Old pickles
load and solve happily (the October ones do), so staleness will not announce itself.

### Where recordings live, and rebases

Nothing but the `intree` golden goes in git. A recording is a directory, keyed by tree:

```text
$LX_EQUIV_ROOT/                     # default ~/lx_equiv
  src/<sha12>-h<harness8>/          # frozen tree plus HEAD's harness, _C.so symlinked
  <sha12>-h<harness8>[-dirty<hash8>]/
    MANIFEST.json                   # identity and environment
    DONE                            # written last
    <tier>/<cell>/
      solves/<key>.json             # SolveRecord
      commits/<key>.json            # CommitRecord
      finals/<key>.json             # FinalRecord
      protos/<sha256>.txt.gz        # content-addressed
      exprs/<sha256>.srepr.gz
      snapshots/<sha256>.json.gz
      provenance/<sha256>.json.gz   # provenance_record rows, one proto stage
      timing/<key>.<pid>.json       # never compared
    captures/<cell>/<model>.<n>.pkl
    logs/
```

`<key>` is the sanitized case plus `#<n>`, the solve's position in the case. The October
recording took 11 MB for 125 solves.

`tools/lx_planning/lx_gate.sh [--base COMMIT] [--tiers ...] [--record-only]` is the one
command, and it replaces `freeze.sh`, `gate.sh` and `rec_pair.sh`, which hard-code one
worktree:

1. `base` is `--base`, which must be an ancestor of HEAD, or else
   `git merge-base origin/main HEAD`.
2. Freeze the base into `src/<sha12>-h<harness8>` with `git archive`, then copy HEAD's
   `tests/inductor/lx_*` over it, so one harness measures both sides. Symlink `_C.so`, or
   stop and say to build it if `torch_spyre/csrc` differs between base and HEAD.
3. If that recording's `DONE` is missing, record the base. A rebase changes the sha, so the
   baseline is re-recorded by construction, and a baseline is never rewritten.
4. Record HEAD, frozen the same way, with `base_sha` set in its manifest.
5. Plain-run the unit suites, then `diff`, `guards` and `timing`, and write `report.md`
   for the PR, with recording time per tier.

Card tiers run under `flock ~/.spyre-card.lock`, as the old scripts did. `fake` and
`intree` take no lock. `--record-only` records a commit and stops, so a nightly cron on a
host that keeps `$LX_EQUIV_ROOT` can record `main` by sha, `lx_gate.sh` finds the base
already recorded, and a rebase costs one recording. CI pods keep nothing, so this is not a
workflow, and which host runs it is not settled.

### Record format

Everything is JSON with sorted keys, `schema: 1`. The format is the contract later stages
read. A stage that adds a field bumps `SCHEMA_VERSION` in the same commit. Both sides of a
diff run HEAD's harness, so they share the schema.

```python
@dataclass
class StageRecord:
    proto_sha: str                # sha256 of str(model.proto)
    status: str                   # solver.StatusName
    objective: float | None       # None when status has no incumbent
    objective_used: bool
    variables: int                # compared via proto_sha; kept for the size guard
    constraints: int
    wall_s: float                 # never compared
    provenance_sha: str | None    # provenance/<sha>; None until Stage 5

@dataclass
class BufferDecision:
    address: int | None
    spill_reason: str | None
    menu_index: int | None        # today's chosen_division
    splits: dict[str, int]        # symbol name -> factor; output and reduction axes
    reduction: list[str]          # sorted names of the reduction axes in splits
    tile_splits: list[tuple[str, int]]   # outermost first
    relayouts: dict[str, list]    # parent -> [group, src index, dst index, dest address]

@dataclass
class SolveSizes:
    buffers: int                  # len(decisions) on a solve with no buffers
    menu_entries: int = 0         # sum of len(core_divisions)
    pair_entries: int = 0         # sum of len(pairs) over cd_parent_matches
    relayout_copies: int = 0

@dataclass
class SolveRecord:
    schema: int
    case: str
    index: int
    family: str                   # today: "cpsat" | "sa_legacy" | "anneal" | "greedy"
                                  # | "firstfit" | "bestfit" | "exhaustive";
                                  # the families add "ortools" and "sa"
    solver: str                   # class name
    stages: list[StageRecord]     # empty for a family with no proto
    production_params: dict[str, object]
    cost_expr_sha: str | None     # exprs/<sha>.srepr.gz
    objective_ns: float | None    # plan_solver._evaluate(expr, solved_bindings(buffers))
    score_ns: float | None        # solver.score_ns()
    off_expression_ns: dict[str, float]
    bindings: dict[str, float]    # solved_bindings, by symbol name
    sizes: SolveSizes
    decisions: dict[str, BufferDecision]
    error: str | None             # type and normalized message

@dataclass
class CommitRecord:
    schema: int
    case: str
    index: int
    allocator: str
    solver: str
    fallback: bool
    inserted: list[dict[str, object]]
    relayouts: list[dict[str, object]]
    addresses: dict[str, int]
    divisions: dict[str, object]
    spill_reasons: dict[str, str]  # as _get_spill_reasons returned it
    restickifies: list[dict[str, object]]  # name, source, origin, swaps_stick,
                                  # source_resident, after_relayout; see Coverage
    snapshot_sha: str | None
    snapshot_error: str | None

@dataclass
class FinalRecord:
    schema: int
    case: str
    final_snapshot_sha: str | None
    predicted_total_us: float | None
    timing_path: str | None       # never compared
```

`MANIFEST.json` holds `schema`, `tree_sha`, `dirty_hash`, `base_sha`, `harness_digest`
(sha256 of `tests/inductor/lx_*`), `params`, `python`, `torch`, `ortools`, `sympy`,
`machine`, `PYTHONHASHSEED` and `settings`: the keys `validate_contract` requires, without
the two a `Cell` sets, plus `lx_solver_relayout`, `dxp_lx_frac_avail` and
`enable_lx_context_switching`.

`diff` requires `python`, `torch`, `ortools`, `sympy`, `machine`, `PYTHONHASHSEED`,
`params` and `settings` to match, and `base_sha`, when set, to equal the base's `tree_sha`.
`schema` and `harness_digest` match by construction, so they are not compared. Cell values
live in the directory path, so a family cell can be compared with an old one.

### Harness API

`tests/inductor/lx_equivalence.py` holds the recorder, hooks, diff and pytest plugin, and
imports nothing from `lx_corpus`. Its top level is standard library only, so Stage 12's
`compare_planners.py` can use the readers with no torch. It imports solver and allocator
modules lazily, inside `install`, after `import torch`, because `torch_spyre` imported first
fails torch's backend autoload under pytest. `sympy` is imported where it is used.

```python
SCHEMA_VERSION: int = 1
DEFAULT_DTIME: float = 30.0            # LX_EQUIV_DTIME overrides
OPTIONAL_TARGETS: tuple[str, ...]      # dotted hook targets absent on one side of a diff

@dataclass(frozen=True)
class SolveParams:
    num_workers: int = 1
    random_seed: int = 0
    max_deterministic_time: float = DEFAULT_DTIME

    def apply(self, parameters: "sat_parameters_pb2.SatParameters") -> dict[str, object]:
        """Set the deterministic parameters, return the ones replaced.

        Sets num_search_workers, random_seed, max_deterministic_time,
        max_time_in_seconds = inf and interleave_search = False."""

@dataclass(frozen=True)
class ForcedFailure:
    solve_index: int = 0               # nth outermost solve in the case
    stage_index: int = 0               # nth model solve within it

class Recorder:
    out: Path
    def begin_case(self, case: str) -> None: ...
    def begin_solve(self, *, family: str, solver: str) -> None: ...
    def record_stage(
        self, proto: str, *, status: str, objective: float | None, objective_used: bool,
        provenance: Sequence[Mapping[str, object]] | None = None,  # provenance_record rows
    ) -> None: ...
    def end_solve(
        self, decisions: Mapping[str, BufferDecision], *,
        sizes: SolveSizes | None = None,  # None: SolveSizes(buffers=len(decisions))
        cost_expr: "sympy.Expr | None" = None, bindings: Mapping[str, float] | None = None,
        score_ns: float | None = None, off_expression_ns: Mapping[str, float] | None = None,
        production_params: Mapping[str, object] | None = None, error: str | None = None,
    ) -> SolveRecord: ...
    def record_commit(self, record: CommitRecord) -> None: ...
    def record_final(self, record: FinalRecord) -> None: ...

def proto_text(model: object) -> str:
    """str(model.proto), for anything with a .proto: cp_model.CpModel today, the
    parent plan's CpModel from Stage 5. The one definition of the bytes compared."""

def provenance_record(p: "Provenance") -> dict[str, object]:
    """One JSON row: constraint, rule, subject name, outcome. Reads Stage 5's
    Provenance by attribute, so this module imports nothing from Stage 5."""

def decisions_from_buffers(
    buffers: Sequence["LifetimeBoundBuffer"],
) -> dict[str, BufferDecision]: ...

def sizes_from_buffers(buffers: Sequence["LifetimeBoundBuffer"]) -> SolveSizes: ...

def snapshot_commit(
    graph: "GraphLowering", *, case: str, index: int, allocator: str, solver: str,
    ops_before: Sequence[str], spill_reasons: Mapping[str, str], fallback: bool,
    snapshot_dir: Path,
) -> CommitRecord:
    """Build a CommitRecord from the live graph, so a commit that is not
    _push_allocation can use it. The beam's names come from the installed Recorder."""

class Observer(Protocol):
    def on_solve_begin(self, allocator, solver, ir, buffers, cost_expr) -> None: ...
    def on_solve_end(self, solver, buffers, cost_expr, record: SolveRecord) -> None: ...
    def on_commit_begin(self, graph, allocator, buffers, accepted) -> None: ...
    def on_commit_end(self, graph, allocator, record: CommitRecord) -> None: ...

def add_observer(observer: Observer) -> Callable[[], None]:
    """Register an observer, return its remover. Methods may be missing. An
    observer's exception is stored in Recorder.observer_errors, never raised."""

def install(
    out: str | Path, *, tier: str, cell: str, params: SolveParams = SolveParams(),
    force_failure: ForcedFailure | None = None, capture_dir: str | Path | None = None,
    allow_snapshot_errors: bool = False, optional: Collection[str] = OPTIONAL_TARGETS,
) -> Recorder:
    """Patch the targets. A missing target raises AttributeError unless it is in
    optional, so a stage that removes one moves the hook in the same PR."""

def uninstall() -> None: ...

@contextmanager
def recording(out: str | Path, **install_kwargs) -> Iterator[Recorder]: ...

@dataclass(frozen=True)
class Difference:
    case: str
    index: int
    kind: str      # "missing" "stage_count" "proto" "status" "objective"
                   # "objective_used" "cost_expr" "objective_ns" "production_params"
                   # "decisions" "error" "commit" "snapshot" "predicted_total"
    detail: str
    first_cause: bool = True    # False for a difference downstream of an earlier stage
    rule: str | None = None     # from provenance when either side has it

def diff(
    base: str | Path, change: str | Path, *, ignore: Collection[str] = (),
    allow_env: Collection[str] = (),
) -> list[Difference]:
    """Compare two tier/cell directories exactly. Raises ManifestMismatch on a
    required key not in allow_env. ignore names kinds, plus "menu_index",
    "decision_relayouts" and "solver_names" (family, solver, allocator)."""

class ManifestMismatch(Exception):
    """Two recordings differ in identity or environment, so a diff means nothing."""

@dataclass(frozen=True)
class GuardViolation:
    case: str
    guard: str      # "fallback" "solve_error" "limit" "size" "observer" "snapshot"
    detail: str

def check_guards(
    root: str | Path, *, tier: str, fixed_set_only: bool = False
) -> list[GuardViolation]: ...

@dataclass(frozen=True)
class CostRow:
    case: str
    base: float | None
    change: float | None

def cost_report(base: str | Path, change: str | Path) -> list[CostRow]:
    """Per-case cost, objective_ns where priced, else the first ladder stage's
    objective, which is HBM traffic. Not exact; a report for non-parity stages."""

def timing_report(base: str | Path, change: str | Path) -> str:
    """Per pass and stage inclusive time, from alternating runs. Never a gate."""

def corpus_coverage(root: str | Path) -> dict[str, int]:
    """Feature name -> number of cases that hit it. See Coverage."""

def load_solves(path: str | Path) -> dict[str, SolveRecord]: ...
def load_commits(path: str | Path) -> dict[str, CommitRecord]: ...
def load_cost_expr(path: str | Path, sha: str) -> "sympy.Expr": ...
def blank_for_capture(buffers: Sequence["LifetimeBoundBuffer"]) -> list: ...
```

`tests/inductor/lx_corpus.py` holds the corpus and the CLI.

```python
@dataclass(frozen=True)
class Cell:
    layout_solver: str = "cpsat"
    co_optimizing: bool = True
    auto_coarse_tiling: bool = False

    @property
    def slug(self) -> str: ...
    def env(self) -> dict[str, str]:
        """LAYOUT_SOLVER, CO_OPTIMIZING_LX_PLANNING, AUTO_COARSE_TILING,
        ALLOW_EXHAUSTIVE_SEARCH."""

DEFAULT_CELL: Cell
TILING_CELL: Cell
SA_CELL: Cell
SA_TILING_CELL: Cell
ZOO_CELLS: tuple[Cell, ...]       # the four above
MATRIX_CELLS: tuple[Cell, ...]    # 5 solvers x co_optimizing, 10 cells
FIXED_GRAPHS: Mapping[str, tuple[str | None, str]]  # name -> (capture, zoo model)
MATRIX_MODELS: tuple[str, ...]
HOST_SUITES: tuple[str, ...]
DEVICE_SUITES: tuple[str, ...]
PLUGIN_DESELECT: tuple[str, ...]
KNOWN_UNSTABLE: Mapping[str, str]             # case -> torch-spyre issue, empty today
SIZE_BOUNDS: Mapping[str, Mapping[str, float]]

@dataclass(frozen=True)
class InTreeCase:
    case: str
    buffers: Sequence["CoreDivisionBuffer"]
    joint: bool

def intree_cases() -> Iterator[InTreeCase]: ...
def replay_intree(case: InTreeCase, *, arm: str) -> None: ...  # "cpsat" | "sa_legacy"
def replay_capture(path: str | Path, *, case: str) -> None: ...
def compile_model(
    name: str, *, device: str = "fake",  # "fake" | "card"; runs in the calling process
    force_failure: ForcedFailure | None = None, capture_dir: Path | None = None,
) -> None: ...
def record_compile(
    name: str, cell: Cell, out: Path, *, device: str = "fake",
    observers: Sequence[str] = (),   # "module:factory", each gives an Observer
    force_failure: ForcedFailure | None = None,
) -> Path:
    """Run compile_model in a cold subprocess, as pytest and the CLI do. Returns the
    tier/cell directory."""
```

`tests/inductor/lx_fake_device.py` and `lx_zoo.py` hold the rest.

```python
# lx_fake_device.py
def guard_device() -> None: ...
def install_fake_device() -> None: ...
def compile_fake(fn: Callable, args: Sequence[torch.Tensor]) -> None: ...

# lx_zoo.py
MODELS: Mapping[str, Callable[[], tuple[Callable, tuple[torch.Tensor, ...]]]]
CARD_ONLY: Mapping[str, str]       # model -> why the fake device cannot reach it
FAKE_DIVERGES: Mapping[str, str]   # model -> why its fake snapshot differs from the card
```

```text
python3 tests/inductor/lx_corpus.py record  --tier T --out DIR [--cells ...] [--models ...]
                                            [--device D] [--observer M:F ...] [--timing-only]
python3 tests/inductor/lx_corpus.py diff    BASE CHANGE [--ignore KIND ...] [--allow-env K ...]
                                            [--report F]
python3 tests/inductor/lx_corpus.py guards  DIR --tier T
python3 tests/inductor/lx_corpus.py timing  BASE_TREE CHANGE_TREE [--rounds N] [--tier T]
python3 tests/inductor/lx_corpus.py check-golden [--tier T]
python3 tests/inductor/lx_corpus.py bless   [--tier T]
python3 tests/inductor/lx_fake_device.py census --out DIR [--models ...] [--card]   # P1
python3 -m pytest -p lx_equivalence ...     # with LX_EQUIV_DIR and PYTHONPATH=.:tests/inductor
```

The plugin imports `torch` in `pytest_configure`, ahead of collection, so it is safe with
`--noconftest`, which is how `unit-host` runs. `--timing-only` installs no hooks, because
`srepr`, `_evaluate` and snapshots would overlap the timing regions, and keeps only
`TORCH_SPYRE_TIMING` output. `timing` alternates cold processes of the two trees (ABAB) on
live compiles.
Environment: `LX_EQUIV_ROOT` (default `~/lx_equiv`), `LX_EQUIV_DIR` with
`LX_EQUIV_TIER` and `LX_EQUIV_CELL` (plugin output), `LX_EQUIV_DTIME` (budget, recorded
in the manifest).

### Regression guards from #5682

Each guard in the issue maps to one piece of this harness.

| guard in the issue | here |
|---|---|
| Fixed graph set | `FIXED_GRAPHS`, at both levels, and the `real` tier for the real model |
| Plan-equivalence check | `diff` over `intree`, `unit-*`, `fake`, `matrix`. Stronger than the issue asks: proto and commit, not only split, residency, address and objective |
| Solve-status check | `check_guards`: `fallback` and `solve_error` on any tier but `failure`; `limit` on any status other than `OPTIMAL`, a failure on the fixed set and a report elsewhere. A budget stop elsewhere still shows up as a `status` difference when it changes |
| Solver-size counters | `check_guards` `size`, per buffer, as `test_frontend_work_bounds.py` bounds per op |
| Compile-time report | `timing_report` over `TORCH_SPYRE_TIMING=1` records. Never a gate |
| Predicted-runtime comparison | `predicted_total_us` from `SPYRE_DUMP_COST=1`. Exact in `diff` at parity. `cost_report` for stages that change plans on purpose |

Size bounds for `intree` come from the measured maxima over the 18 graphs at the last
ladder stage: 10.3 variables, 14.1 constraints and 24.8 menu entries per buffer. The
bounds are 12, 16 and 30. Compile bounds are written into `SIZE_BOUNDS` at calibration.
Four October captures suggest 10-11 variables and 18-20 constraints per buffer on real
graphs. `SIZE_BOUNDS`, `DEFAULT_DTIME` and `PLUGIN_DESELECT` are refreshed at each card
calibration, and a stage that moves a bound on purpose updates it in its own PR.

**Coverage.** `corpus_coverage(root)` lists the features the corpus must contain and how
many cases hit each: tiled solve with copies, copies of several parents, execution chain,
placement-only buffers, cost expression, multi-parent in-place child, LX prefetch, output
clone, fired relayout, drain clone, collective stage, budget stop and fallback. A feature
with no hit is listed in the PR. Drain clones and collective stages may only be reached by
the device suites.

Four of the five restickify features are hard, because that handling is what the later stages
contest and the zoo may not touch it. `relayout_before_restickify` is report-only: the only
evidence for it is a mocked test. Each PR reports hits per feature. They come from
`CommitRecord.restickifies`, which `snapshot_commit` builds on the live graph, because
`is_restickify_op` (`torch_spyre/_inductor/padding.py:563`) needs the op and the
`GraphLowering`, not a frozen snapshot. There is one row for each op that it or
`_is_synthetic_restickify` (`torch_spyre/_inductor/enforce_indirect_access_layout.py:229`)
accepts. `source` is the buffer `_restickify_input` (`padding.py:478`) returns.

| feature | a row counts when |
|---|---|
| `restickify_op` | `swaps_stick`: `is_restickify_op` holds |
| `beam_restickify` | `origin="beam"`: the `insert_restickify_on_node_inputs` hook saw it added |
| `fixup_copy` | `origin="fixup"`: synthetic, and added elsewhere (`insert_restickify.py:866`, `torch_spyre/_inductor/enforce_indirect_access_layout.py:165`, `torch_spyre/_inductor/nonstick_dim_order.py:1147`). It may not swap the stick |
| `restickify_reads_resident_source` | `swaps_stick` and `source_resident`: the source has `layout.allocation["lx"]` |
| `relayout_before_restickify` | `swaps_stick` and `after_relayout`: `materialized_lx_relayout_for_destination(graph, source)` is not `None` (`torch_spyre/_inductor/scratchpad/lx_relayout.py:343`) |

## Changes

| file | change |
|---|---|
| `tests/inductor/lx_equivalence.py` | new: recorder, hooks, diff, guards, reports, pytest plugin |
| `tests/inductor/lx_corpus.py` | new: cells, tiers, compile driver, replay, CLI |
| `tests/inductor/lx_zoo.py` | new: 35 models, 32 lifted from `probes-cpsat-graph/zoo.py`; `CARD_ONLY`, `FAKE_DIVERGES` |
| `tests/inductor/lx_fake_device.py` | new: fake device, lifted from `swiglu.py`; P1 census |
| `tests/inductor/lx_equivalence_golden.json` | new: `intree` digests per case |
| `tests/inductor/test_lx_equivalence.py` | new: host tests of the harness and CI guards |
| `tests/inductor/test_lx_equivalence_compile.py` | new: compile tests, on `fake` in subprocesses, plus one calibration test on the card |
| `tests/configs/torch_spyre_tests/inductor/test_lx_equivalence_config.yaml` | new: labels `[unit, regression, trunk]`, both files `unlisted_test_mode: mandatory_success` |
| `tools/lx_planning/lx_gate.sh` | new: freeze, record, diff, report |
| `docs/source/compiler/scratchpad_planning.md` | add a short "LX equivalence harness" section after "Matched OFF-plan comparison" |

New modules follow `AGENTS.md`: the 14-line license header, `import regex`, not `re`, 88
columns. `lx_*.py` names avoid pytest collection.

Commits, each leaving the suite green:

1. **Probe P1.** `lx_zoo.py` and `lx_fake_device.py` with its census command. Run P1, fill
   `CARD_ONLY` and `FAKE_DIVERGES`, record the result in the PR. `test_zoo_names_are_stable`.
2. **Recorder and solve hooks.** `lx_equivalence.py` through `diff`, `check_guards`
   without commit data, and the plugin. `lx_corpus.py` with the `intree` tier. Host tests,
   config yaml, `make check-all-configs`.
3. **Golden.** `lx_equivalence_golden.json`, `bless`, `check-golden`, size bounds for
   `intree`.
4. **Commit hooks.** `snapshot_commit`, observers, the compile driver and
   `test_lx_equivalence_compile.py`.
5. **Tiers.** `unit-*`, `fake`, `zoo`, `matrix`, `failure`, `replay`, `real`, captures,
   `corpus_coverage`, `cost_report`, `timing_report`.
6. **Gate and calibration.** `lx_gate.sh`, docs, then one card calibration: compile
   `SIZE_BOUNDS`, `DEFAULT_DTIME` if the zoo calls for it, `PLUGIN_DESELECT` as measured,
   recording time per tier.

## Interfaces

**Consumes.** Stage 0 depends on no other stage. The last row is read only once Stage 5
lands.

| symbol | signature | from Stage | used for |
|---|---|---|---|
| `CpSatLayoutSolver._solve_and_record` | `(self, solver, model, *, objective=False) -> int` | existing | stage hook; gone at Stage 13 |
| `MemoryPlanSolver.plan_layout` and `CoreDivisionLayoutSolver.plan_layout_and_core_divisions` | `(self, log_lx_usage=False)`, `(self, cost_expr=None)` | existing | solve hook |
| `ScratchpadAllocator._push_allocation` | `(self, graph, buffers, accepted_lx_relayouts)` | existing | commit hook; retargeted at Stage 10 |
| `ScratchpadAllocator._get_spill_reasons` | `(self, graph, allocation) -> dict` | existing | spill reasons |
| `CustomPreSchedulingPasses.__call__`, `.last_cost_report` | `(self, graph)`, `CostReport \| None` | existing | abort point, predicted total |
| `solved_bindings`, `_evaluate` | `(buffers) -> dict`, `(expr, bindings) -> float \| None` | existing | `objective_ns` |
| `freeze`, `graph_snapshot` | `tools/lx_planning/lx_semantic_plan.py` | existing | commit snapshot |
| `materialized_lx_relayouts`, `materialized_lx_relayout_for_destination` | `(graph) -> dict`, `(graph, destination_name) -> LXRelayoutPlan \| None` | existing | `CommitRecord.relayouts`, `.restickifies` |
| `load_captures`, `synthetic_graphs` | test helpers | existing | `intree` tier |
| `is_restickify_op`, `_restickify_input`, `_is_synthetic_restickify` | `(op, graph) -> bool`, `(op, graph) -> (dep, buf, layout)`, `(buf) -> bool` | existing | `CommitRecord.restickifies` |
| `insert_restickify_on_node_inputs` | `(op, resticks_needed, operations) -> None` | existing | beam hook |
| `Provenance` | `.constraint`, `.rule`, `.subject`, `.outcome` | Stage 5 | `provenance_record`, read by attribute |

**Provides.**

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `install`, `uninstall`, `recording` | see Harness API | 7-10, 12, 13 | final |
| `OPTIONAL_TARGETS` | dotted targets; ships with Stage 7's two, and a later stage adding a hook adds its target | 7, 8, 10, 13 | final |
| `SolveParams`, `DEFAULT_DTIME` | `SolveParams(num_workers=1, random_seed=0, max_deterministic_time=30.0)`, `30.0` | 7, 9, 12 | final |
| `ForcedFailure` | `ForcedFailure(solve_index=0, stage_index=0)` | 7, 9 | final |
| `Recorder.begin_solve`, `.record_stage`, `.end_solve`, `.record_commit` | see Harness API | 5, 7-11 | final |
| `proto_text` | `(model: object) -> str`; `str(model.proto)` | 5, 7, 11 | final |
| `provenance_record` | `(p: Provenance) -> dict[str, object]` | 5, 7 | final |
| `decisions_from_buffers`, `sizes_from_buffers` | see Harness API | none; the old-solver hooks call them | temporary, removed in Stage 13 |
| `BufferDecision` | see Record format | 7, 8, 9 | final |
| `snapshot_commit` | see Harness API | 10 | final |
| `add_observer`, `Observer` | see Harness API | 2, 4, 5, 8 | final |
| `diff`, `Difference`, `ManifestMismatch` | see Harness API | 1-13, through `lx_gate.sh` | final |
| `check_guards`, `GuardViolation` | see Harness API | 2, 12; 1-13 through `lx_gate.sh` | final |
| `cost_report`, `CostRow` | see Harness API | none yet; for a stage that changes plans on purpose | final |
| `timing_report`, `timing` CLI | see Harness API | 5, 8, 12 | final |
| `load_solves`, `load_commits`, `load_cost_expr` | see Harness API; `load_solves` and `load_commits` import no torch. Stage 4 evaluates its objective at `SolveRecord.bindings` and compares it with `objective_ns` | 4, 12 | final |
| `Cell`, `DEFAULT_CELL`, `TILING_CELL`, `SA_CELL`, `SA_TILING_CELL`, `ZOO_CELLS`, `MATRIX_CELLS`, `FIXED_GRAPHS`, `MATRIX_MODELS` | see Harness API | 2, 3, 5, 6, 8-13 | final; Stage 9 adds the `ortools` and `sa` cells, Stage 13 retires `cpsat` and `simulated_annealing` |
| `corpus_coverage` | see Harness API | 2, 3, 5, 6, 8, 9, 10 | final |
| `intree_cases`, `compile_model`, `record_compile`, `lx_zoo.MODELS`, `CARD_ONLY`, `FAKE_DIVERGES` | see Harness API | 2, 4, 5, 8-12 | final; Stage 10 adds `compile_model(stop_at="codegen")` |
| `guard_device`, `install_fake_device`, `compile_fake` | see Harness API | 11; the rest through `record_compile` | final |
| `replay_intree`, `replay_capture`, `blank_for_capture` | see Harness API | 4 | temporary, removed in Stage 13 |
| `lx_corpus.py` CLI, `lx_gate.sh` | see Harness API | 1-13 | final |
| Record format, `SCHEMA_VERSION` | see Record format | 8-11 change it; 1-13 read it through `diff` | final; a stage that changes it bumps the version in that commit |

**What later stages must do to use it.**

- **Stages 2, 4, 5 and 8** register an `Observer`. `on_solve_begin` is their point with the
  allocator and `GraphLowering` in hand, and `record_compile(observers=...)` runs one on a
  `fake` compile. `allocator` and `ir` are `None` on a solver-only run.
- **Stage 5** passes `[provenance_record(p) for p in model.provenance]` to `record_stage`
  when it records its own model. `diff` aligns top-level proto blocks with
  `difflib.SequenceMatcher` and fills `Difference.rule` from the side that has provenance,
  skipping rows with no constraint. That is what "listed by rule in its PR" reads.
- **Stage 7** adds the `CpSatBackend._run_stages` and `SolveSettings.from_config` rows to
  `install` in `lx_equivalence.py`, the first applying `SolveParams` and `ForcedFailure` and
  calling `record_stage`, and adds `buffer_decision(decision, alignment)`, which turns a
  `NodeDecision` into a `BufferDecision` in bytes with `menu_index=None`, `relayouts={}`.
- **Stage 8** records through `Recorder` with `family="sa"` and no stages.
- **Stages 7 and 8** leave `sizes` to its default. The size guard then bounds a stage's
  variables and constraints per decision, and skips the menu bound.
- **Stage 9** adds `decisions_from_graph` over Stage 3's `decisions`, the `ortools` and `sa`
  cells, and the `install` rows that open and close family solves on `AnnealBackend.solve`,
  both `read_back`s and `SpyreOptimizer._solve_with`.
- **Stage 10** retargets the commit hook at `SpyreOptimizer.commit` and calls
  `snapshot_commit` with `allocator="SpyreOptimizer"`.
- **Stages 8, 9 and 10** state that their gate runs HEAD's harness over the base tree.
- **A stage that removes a hook target** moves the hook in the same PR and adds the old
  target to `OPTIONAL_TARGETS`.
- **Stage 13** counts `lx_equivalence.py` and `lx_corpus.py` as importers of the old
  solvers and allocators, removes `decisions_from_buffers`, `sizes_from_buffers`, the
  replay helpers and the old-solver hooks, and replaces `test_golden_intree` with `bless
  --tier fake` on the fixed set, recording decisions and predicted total.

## Tests

`tests/inductor/test_lx_equivalence.py`. Host logic, runs under plain pytest in CI and
under `--noconftest` with no card.

| test | asserts |
|---|---|
| `test_params_are_applied_before_each_solve` | a recording spy on `CpSolver.Solve` sees one worker, seed 0, infinite wall clock, dtime 30, interleave off on every stage |
| `test_production_params_are_recorded_first` | `production_params` holds the 60 s limit and config's worker count, not the forced values |
| `test_ladder_records_every_stage` | a no-cost joint solve records 3 stages, placement-only 1 |
| `test_second_solve_in_process_is_identical` | two recordings of `intree` in one process, `diff` empty |
| `test_hash_seed_does_not_change_recording` | subprocess `record --tier intree` at `PYTHONHASHSEED=0` and `1`, `diff` empty |
| `test_golden_intree` | `intree` matches `lx_equivalence_golden.json`; decisions are compared on x86_64 only, protos everywhere |
| `test_diff_detects[kind]` | each `Difference.kind` fires on a record mutated in that one field, and only that kind; the test that shows `diff` can fail |
| `test_diff_marks_downstream_stages` | a change at stage 0 gives one `first_cause=True` and later stages `False` |
| `test_diff_ignores_time_fields` | changing `wall_s`, `timing_path` and `after N.NNs` gives no difference |
| `test_diff_refuses_other_environment[key]` | each required manifest key raises `ManifestMismatch`, unless in `allow_env` |
| `test_diff_ignores_harness_identity` | a changed `schema` or `harness_digest` alone does not refuse |
| `test_diff_checks_base_sha` | a change recorded against another base is refused; an ancestor given by `--base` is accepted |
| `test_diff_ignore_menu_index` | a reordered menu with the same splits differs by default, not with `ignore={"menu_index"}`; empty decision relayouts likewise with `"decision_relayouts"` |
| `test_decisions_carry_splits_not_only_index` | `BufferDecision.splits` matches `core_divisions[chosen].splits` by name |
| `test_forced_failure_raises_solve_error[stage]` | `ForcedFailure` at stages 0, 1, 2, and on the cost path with a small `cost_expr`, raises `SolveError` from production code, the record has `UNKNOWN` and `error` |
| `test_solve_error_is_reraised_unchanged` | the wrapper does not swallow or wrap the exception |
| `test_exhaustive_records_one_solve` | `ExhaustiveSearchSolver` over greedy gives one `SolveRecord` |
| `test_sa_legacy_arm_records_decisions_without_stages` | `family="sa_legacy"`, `stages=[]`, `score_ns` set |
| `test_provenance_record_fields` | a stand-in with Stage 5's attributes gives one JSON row; a row with no constraint is skipped in alignment |
| `test_uninstall_restores_originals` | each patched attribute `is` the original afterwards |
| `test_install_raises_on_missing_target` | with `optional=()`, deleting a target raises `AttributeError`; the same target in `optional` installs the rest |
| `test_readers_import_no_torch` | a subprocess imports `lx_equivalence` and calls `load_solves`, with `torch` absent from `sys.modules` |
| `test_end_solve_defaults_sizes` | `sizes=None` records `buffers=len(decisions)`, and the size guard still reads variables and constraints |
| `test_observer_errors_do_not_escape` | a raising observer leaves the solve intact and shows in `check_guards` |
| `test_guard_fires[guard]` | each guard fires on a crafted record |
| `test_intree_guards` | every `intree` stage is `OPTIMAL`, no error, and per-buffer variables, constraints and menu entries are within bounds |
| `test_capture_blanks_faketensor_fields` | `blank_for_capture` empties the three fields and leaves the input untouched |
| `test_replay_refuses_other_base` | a pickle with another `source_sha` is refused |
| `test_case_key_is_stable` | sanitizing keeps keys unique and path-safe |

`tests/inductor/test_lx_equivalence_compile.py`. Each compile is a `fake` subprocess
through `record_compile`, except the last row.

| test | asserts |
|---|---|
| `test_fake_compile_opens_no_device` | a `fake` compile ends with no vfio fd, and `start_runtime` raises |
| `test_commit_record_has_clones_and_addresses` | `softmax_256x1024` at default: `inserted` holds the LX prefetch's clone op, `addresses` non-empty, `snapshot_error` None |
| `test_commit_matches_graph_snapshot` | `snapshot_sha` equals a direct `graph_snapshot` taken by an observer |
| `test_on_solve_begin_sees_allocator_and_ir` | an observer receives the allocator and the `GraphLowering` |
| `test_spill_reasons_are_recorded` | a spilled buffer's reason is the one `_get_spill_reasons` gave |
| `test_restickify_rows_match_the_graph` | on a fixed-set model P1 showed has a restickify: each row's `swaps_stick` and `source` equal what an observer reads from the live graph, and `origin` is `"beam"` exactly for names the hook saw |
| `test_fixed_set_status_and_size` | each compile-level `FIXED_GRAPHS` model at default: no fallback, all `OPTIMAL`, sizes in bounds. A `CARD_ONLY` model skips with its reason |
| `test_forced_failure_falls_back_to_greedy` | with `ForcedFailure()`: `fallback=True`, `ScratchpadAllocator`, `GreedyLayoutSolver`, commit recorded |
| `test_capture_replay_matches_live` | for `softmax_256x1024` and `sdpa_1x8x256x64`, replay protos and decisions equal the live compile's |
| `test_compile_twice_is_identical` | two cold compiles of `softmax_256x1024` at `PYTHONHASHSEED=0` and `1`, `diff` empty |
| `test_predicted_total_is_recorded` | `FinalRecord.predicted_total_us` is positive |
| `test_fake_compile_matches_device[model]` | card, skipped unless `LX_EQUIV_CALIBRATE=1`, which only `lx_gate.sh` sets: for each fixed-set model, the final snapshot of a `fake` compile equals a card compile's; a model in `FAKE_DIVERGES` is a strict xfail citing the reason |

`tests/inductor/test_lx_semantic_plan.py` and every other existing suite run unchanged. No
existing test is edited.

**Device suites.** `test_fake_compile_matches_device`, and the gate's `unit-device`, `zoo`
and `real` tiers.

**What proves this stage's claim.** G0 to G7 below, which show the harness is stable,
sensitive and complete, on an unchanged compiler.

## Parity and gates

The parent plan's gate for this stage is a description: "A device-free equivalence harness
solves deterministically: one worker, seed 0, a deterministic time budget and no
wall-clock limit", recording "the model proto, status, objective and decisions" and, after
`_push_allocation`, "the committed plan: inserted clones, relayout ops, addresses,
divisions and spill reasons", with "No production code changes in this stage." It is
met as follows, with "device-free" narrowed by Deviations 2.

| gate | compared | corpus | pass |
|---|---|---|---|
| G0 probe | P1 recorded | all models | every model reached, or in `CARD_ONLY` with a reason |
| G1 stability | the Stage 0 tree recorded at `PYTHONHASHSEED=0` against the same tree at `1`, in fresh processes, `--allow-env PYTHONHASHSEED` | all tiers except `zoo` and `real` | `diff` empty |
| G2 no production change | the Stage 0 branch against its merge-base | all tiers | `diff` empty, and `git diff --stat` shows no path under `torch_spyre/` |
| G3 sensitivity | `test_diff_detects[*]`, `test_guard_fires[*]`, and one real mutation: a throwaway commit changing one coefficient in `spill_cost`, run through `lx_gate.sh` | `intree`, `fake` subset | gate fails, reporting `proto` |
| G4 guards | `check_guards` | `intree` (hard), fixed set (hard), the rest (report) | no `fallback`, no `solve_error`, no `limit` on the fixed set |
| G5 coverage | `corpus_coverage` | all tiers | `restickify_op`, `beam_restickify`, `fixup_copy` and `restickify_reads_resident_source` each have a case; `relayout_before_restickify` is report-only; any other feature has a case, or the PR says why not |
| G6 suites | plain pytest on the unit suites from the parent plan's Verification, plus the new tests; `pre-commit run --files <changed>`; `make check-all-configs` | | pass |
| G7 calibration | one card run: `zoo`, `matrix`, `failure`, `test_fake_compile_matches_device`, bounds refreshed | fixed set; all of `zoo` recorded | equal snapshots, or listed in `FAKE_DIVERGES` with a reason |

**A difference at Stage 0** is a harness bug or production nondeterminism. A harness bug
gets fixed. A real hash-seed dependence in production cannot be fixed here, since this
stage does not touch production. It gets a torch-spyre issue and an entry in
`KNOWN_UNSTABLE` in `lx_corpus.py` with that issue's number, and G1 passes with the case
listed. Every run keeps `PYTHONHASHSEED=0`, so base and change stay comparable.

**For later stages**, a gate run is `lx_gate.sh` with the tiers below, using HEAD's harness
on both sides. A commit-level gate means `intree` and the fixed set. A PR-tip gate means the
row for that stage.

| stage | tiers at the PR tip | why |
|---|---|---|
| 0 | all | calibration |
| 1 | `intree`, import test | nothing imports the package |
| 2 | `intree`, fixed set, oracle on `fake` zoo | one extraction |
| 3, 4, 5, 6, 7, 8 | `intree`, `fake` at both configs, `unit-*`, `failure` | shared code moved |
| 9, 10 | all, including `matrix` | first routing and commit; Stage 9 also calibrates on the card |
| 11 | `fake`, restickify graphs, Granite for 11b | |
| 12 | its own lanes | calibrates on the card |
| 13 | fixed set per commit, all at the PR tip | |

A non-empty `diff` fails. Stages allowed to differ list each `Difference` by rule in the PR
and pass the accepted kinds through `--ignore` or a reviewed report. After a rebase, the
recording is redone only at the last rebase before merge.

**What this gate does not show.** It shows an unchanged compiler is measured as unchanged,
and `fake` shows parity on default-layout inputs, not production plans. Results are not
known until the gate runs: no zoo plan, commit snapshot, compile hash-seed result, `fake`
compile of the zoo or matrix cell has been recorded on this snapshot, and total recording
time is unmeasured.

## Risks

- **`graph_snapshot` fails on a real graph at the push point.** It was written for
  codegen time and has never run here, and `freeze` refuses unknown types by design.
  Caught on the first compile test. A change to `lx_semantic_plan.py` is allowed since it
  is under `tools/`.
- **The two frontends build different graphs.** `fake` goes through `make_fx` and
  `compile_fx`, the card through `torch.compile`. Node order or names may differ, and a
  fake input's default layout may differ from an upload's. Caught by
  `test_fake_compile_matches_device`. A model that differs goes in `FAKE_DIVERGES`, and it
  is still valid for parity, since both sides of a diff share the fake.
- **P1 reaches few models.** The three sdpa models already fail at the beam, and nothing
  past the beam has been run on the zoo. Then `CARD_ONLY` grows and more gates fall back to
  calibration runs.
- **A restickify feature has no case.** Neither new graph is proven to produce a
  restickify. A hard feature with no case stops the PR at G5 until a graph is found.
- **The allocator depends on hash order.** The solver-level result says nothing about it.
  Caught by G1 and `test_compile_twice_is_identical`.
- **Golden churn.** A PR changing the model, or an OR-Tools bump that moves a tied
  optimum, has to run `bless`. Decisions are compared on x86_64 only, where CI runs.
- **Exhaustive matrix cells run away.** `ExhaustiveSearchSolver` is a DFS, and whether the
  eight models finish is unknown, because the old `same_plans` results were not kept. The
  1200 s timeout bounds a case, and a timeout is recorded the same way on both sides, so a
  cell that always times out proves nothing. `corpus_coverage` lists such cells, and the
  model list is cut down for them if needed.
- **Plugin recordings differ from what tests assert.** A test patching `_solve_and_record`
  or `CpSolver.Solve` bypasses or overrides the hook, on both sides alike. Plain runs stay
  the pass/fail signal.
- **#5695 lands first.** It should not touch any hook target. It defers `_distribute_work`
  and changes the fallback path, so the `failure` tier and a re-recorded base will show
  whether it held its "plans should be identical" claim. It only changes `op_it_space_splits`
  timing, which is read after the commit.
- **The nightly host does not exist.** Without `--record-only` on a schedule, every gate
  records its own base, and the cost is unmeasured. The merge queue is not re-measured
  either way.

## Deviations from the parent plan

1. **One worker does not interleave.** The Determinism row says "a fixed seed and
   interleaved search". Stage 0 says one worker, and these are different searches: at one
   worker, `interleave_search=True` picked a different plan at equal objective on
   `mlp_swiglu_128` and found no plan at dtime 10 on `sdpa_4x8`, where plain search finds
   one. **Resolution:** the harness runs one worker without interleaving. `from_config`
   gives at least two interleaved workers under determinism, and `install` patches it to
   one worker on the family path.

2. **"Device-free" means a fake device, and only where it reaches.** Stage 0 says a
   device-free harness records committed plans. `_push_allocation` needs a live
   `GraphLowering`, which `swiglu.py` shows a fake device can reach on SwiGLU. **Resolution:**
   the `fake` tier, with `CARD_ONLY` models compared at card calibration. Inputs use default
   layouts, so `fake` proves parity, not production plans. `unit-device`, `real` and
   numeric device suites stay on the card.

3. **SA cannot replay zoo captures.** The corpus reads as if one set of captures serves both
   families. Blanking removes what SA reads (`sa_cooptimizer.py:1081`, `:1123`), and a
   capture from a CP-SAT compile holds buffers SA is never given (`allocator.py:5049-5058`).
   **Resolution:** SA runs in its own `fake` and `zoo` cells, and in `intree`.

4. **Planning time is not measured from captures in one process.** Verification says to
   replay zoo captures "on base and change, in one process and interleaved". A capture
   holds buffers, so the new encoder cannot replay one, and two source trees cannot be
   imported in one process (the earlier A/B alternated, `~/cpsat_equiv/scripts/build_ab.sh`).
   **Resolution:** `lx_corpus.py timing` alternates cold processes (ABAB) on live compiles.

5. **"Every combination" runs over a subset of models.** The corpus lists "every
   `layout_solver` × `co_optimizing_lx_planning` combination" without saying over what.
   Ten cells over 35 models at two configs is 700 compiles, including exhaustive DFS cells.
   **Resolution:** all 10 cells over 8 models at default config, and the full zoo only on
   the `cpsat` and `simulated_annealing` co-optimizing cells, at default and tiling.

6. **Not every stage runs the whole corpus, and not after every rebase.** Verification says
   every stage compares against the harness and baselines are re-recorded after every
   rebase. `matrix` alone is bounded only by a 1200 s timeout per case, and Stage 4 and
   Stage 13 want a diff per commit. **Resolution:** the tier table in Parity and gates, and
   one re-recording at the last rebase before merge.

## Open questions

None.

## Out of scope

- `check_decisions`, the Phase 1 report and the buffers-to-graph adapter: Stage 3.
- `Model.provenance` content and rule attribution beyond the reserved field: Stage 5.
- The `CpSatBackend._run_stages` and `SolveSettings.from_config` hooks and
  `buffer_decision`: Stage 7.
- `decisions_from_graph`: Stage 9.
- `test_lx_live_parity.py`: Stage 5.
- `SPYRE_DUMP_COST` as a gate on plans that change on purpose: Stage 11 and Stage 12.
- Side-by-side CI on models: Stage 12.
- The issue's direct tests for `reorder_nonstick_dims`, `reorder_nonstick_dims_mutation`
  and `insert_post_mutation_restickify`: a prerequisite of Stage 11. Its weekly perf run:
  Stages 11 and 12.
- Any fix to production nondeterminism that G1 finds. It gets an issue, not a change here.
- Greedy as an encoder-then-solver family. Nothing here rules it out: `family` is a free
  string, and a family with no stages records like SA.
