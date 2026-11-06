# LX allocator, Stage 12: Default flip

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0 to 10; step 12c also on Stage 11. Needed by: Stage 13.

## Goal

A compile with no planner settings runs `plan_allocation` with `OrToolsFamily`, with sticks
still pinned (`lx_sticks_in_solve` off). The old path stays selectable by `LAYOUT_SOLVER=cpsat`,
the kill switch. The flip does not depend on Stage 11: before it, the new path must produce
byte-identical plans on joint arms, and it must stay within fixed bands on committed arms. This
stage also owns three one-line defaults: the SA fallback, completed-reduction routes, and, as the
separate step 12c, `lx_sticks_in_solve`. Nothing is deleted. Rules, families and commit do not
change.

## Today

**Selection.** `select_allocator` maps config to a solver
(`torch_spyre/_inductor/scratchpad/allocator.py:5377`). `layout_solver` defaults to `"cpsat"`
(`torch_spyre/_inductor/config.py:365`). With `co_optimizing_lx_planning` on (`config.py:41`) the
result is a `CoOptimizingAllocator`; off, a `ScratchpadAllocator` (`allocator.py:5431-5469`).
`_maybe_scratchpad_planning` skips this when `lx_planning` is off
(`torch_spyre/_inductor/passes.py:427`). Tooling reads `layout_solver` too:
`docs/source/user_guide/examples/profile_ops.py:911-916` exits unless the allocator is a
`CoOptimizingAllocator`, so it stops working at the flip.

**Facts that shape this stage.**

1. **Planner source is not in the FX key at this snapshot.** Nothing calls
   `CustomPreSchedulingPasses.uuid()`; the pipeline runs from a monkeypatch
   (`torch_spyre/_inductor/patches.py:213-229`), not a config slot. Config values are in the key
   (`config.py:498-503`). Stage 9 fixes the source half through `optimizer_pass_sources()`, at the
   price of one recompile for every user when Stage 9 lands.
2. **Without ortools a default compile raises `ValueError`** (`allocator.py:5437`, pinned at
   `tests/inductor/test_scratchpad_use.py:1791-1809`). `pyproject.toml:35-39` is stale: the greedy
   fallback exists only with `CO_OPTIMIZING_LX_PLANNING=0`. `ALLOW_EXHAUSTIVE_SEARCH=1` gives the
   exhaustive DFS. `docs/source/api/torch_spyre.rst:755-760` is correct.
3. **The SA co-optimizer gets no `LxContextSwitchingPass`** (`allocator.py:5433`). This is a real bug
   on main, reachable only with `LAYOUT_SOLVER=simulated_annealing`, not from default config. The
   exposed case is a buffer live across a single-output, risky `FallbackKernel`. Whether it clobbers
   on a card is unverified. The plan does not assume an upstream fix: families run the pass, and old
   SA arms are compared with it patched in, as Stage 9 does.
4. **`SPYRE_LX_PLANNER_RELAYOUT=0` does not stop solver relayout copies** (`allocator.py:5056-5061`).
   `docs/source/api/torch_spyre.rst:715` suggests otherwise.
5. **Literal `"cpsat"` pins in tests.** There are 39, and `config.patch` beats the environment, so
   `LAYOUT_SOLVER=ortools` in a job does not reach them. `SPYRE_LX_ROUTE_TO_FAMILIES=1` does. Tests
   that build old classes directly keep testing old code whatever the default is.

**CI.** All card suites go through `.github/workflows/_test_matrix.yaml`, one `spyre_pf_x1` job per
config with a 60-minute limit (`:611-616`).

| workflow | trigger | runs |
|---|---|---|
| `torch_spyre_tests.yaml` | PR, push, merge queue | `regression` |
| `runtests_nightly.yaml` | cron `0 2 * * *` | `regression`, model-ops `capability` tier |
| `integration-tests.yaml` | dispatch | `integration`, and `perf` from an orchestrator outside this repo |
| `test_multi_spyre` job | `regression`, `trunk`, `integration`, `suite_distributed` only (`:1179-1183`) | two-card suites |

- No workflow here compiles a full model. `tests/configs/model_ops_tests/` are per-op capability
  suites, xfail by config (`.github/suite_test_flags.yaml:6-8`). Models run in hf-adapters-dev
  (`Makefile:109`).
- `REPORT_NAME` is the config's basename with `overwrite: true` (`_test_matrix.yaml:693`, `:734`), so
  a shadow tier inside the nightly run would overwrite the nightly's reports. The lane must be its
  own workflow.
- Cards are self-hosted pools. The `/.github/` owners (`CODEOWNERS:57`) approve their use, and
  `.github/runner_overrides.yaml` records that a 4-card pool was given up because it blocked PR
  checks. The weekly perf orchestrator, hf-adapters-dev and spyre-frameworks are owned outside this
  repo.

**Conventions.** There is no changelog. The PR template asks about user-facing changes.
Environment variables are documented in `docs/source/api/torch_spyre.rst:700-845` and
`AGENTS.md:118-134`.

## Design

### The flip

```python
# torch_spyre/_inductor/config.py
layout_solver: Literal[...] = os.environ.get("LAYOUT_SOLVER", "ortools")  # type: ignore[assignment]
```

The default must be a new value, not a re-pointed `"cpsat"`: the kill switch needs the old string to
keep meaning the old path, and the cache key needs the default to change. PR 12b moves this one
value and adds no config field. `lx_route_to_families` stays off by default, or the kill switch
would stop working.

### Settings that select the path

| setting (env) | default | after the flip | after Stage 13 |
|---|---|---|---|
| `layout_solver` (`LAYOUT_SOLVER`) | `cpsat` to `ortools` | family key: Stage 9's `family_key(config)` names it, and `plan_allocation` builds `FAMILIES[key].from_config(config)` through `select_families`. Old value: `family_key` is `None`, the old path, with a warning | `Literal["ortools", "sa"]`, see below |
| `target` | `"spyre"` | unchanged: `select_target` reads it on the new path, and it is in the FX key | stays |
| `co_optimizing_lx_planning` (`CO_OPTIMIZING_LX_PLANNING`) | `1` | `0` is committed mode, `graph.divisions_committed`. Pass-level readers (`lx_relayout.py:1903`, `materialize_hinted_views.py:135`) unchanged | stays |
| `lx_planning` (`LX_PLANNING`) | `1` | `0` skips the optimizer; layouts come from `propagate_layouts` and the beam | stays |
| `allow_exhaustive_search` (`ALLOW_EXHAUSTIVE_SEARCH`) | `0` | old `greedy`, `bestfit`, `firstfit` only. On a family: no effect, warns if set | removed |
| `auto_coarse_tiling` (`AUTO_COARSE_TILING`) | `0` | `select_target` keeps `Optimization.COARSE_TILING` only when it is on, for both families. Stays off | stays |
| `lx_planner_relayout` (`SPYRE_LX_PLANNER_RELAYOUT`) | `1` | same as today, and still does not gate solver relayouts | stays |
| `lx_solver_relayout` (`SPYRE_LX_SOLVER_RELAYOUT`) | `1` | `0`: `OrToolsFamily.encode` offers no relayouts (Stage 6) | stays |
| `read_copy_elision` (`SPYRE_READ_COPY_ELISION`) | `1` | unchanged: `elide_proven_read_copies` runs on both paths and `division_pin_reasons` reads the flag | stays |
| `lx_route_to_families` (`SPYRE_LX_ROUTE_TO_FAMILIES`) | `0` | test selector; off, or `cpsat` stops being the old path | removed |
| `fallback_solver` (`FALLBACK_SOLVER`) | `""` to `sa` in 12a | `""` keeps the greedy re-plan | non-empty, a Stage 13 entry condition |
| `lx_solver_completed_reduction_routes` (`SPYRE_LX_SOLVER_COMPLETED_ROUTES`) | `0` to `1` in step 5 | off through the identity gates | on, a condition for Stage 13's C12 |
| `lx_sticks_in_solve` (`SPYRE_LX_STICKS_IN_SOLVE`) | `0` to `1` in 12c | off through the flip | follows 12c |

The other planner settings keep their meaning on a family through the stage named:
`cpsat_time_limit_seconds` and `num_cpus` through `SolveSettings.from_config` (Stage 7; at least 2
interleaved workers under `deterministic`); `enable_uniqueness_check`,
`lx_solver_relayout_presolve_max_copies` (Stage 7); `lx_solver_relayout_groups_per_edge`,
`lx_extended_ownership`, `lx_same_tile_group_relayout` (Stage 6); `_cpsat_warn_on_cost_expr` (Stages
4, 7, 8, 9); `allow_all_ops_in_lx_planning`, `dxp_lx_frac_avail` (Stages 2, 3);
`enable_lx_context_switching`, `lx_collective_output_staging`, `dump_cost_expr_file` (Stage 10).
`ktir_emitter`, `sencores`, `hbm_pool_planning` and the hint flags are unaffected.

### Warnings

```python
# scratchpad/allocator.py, temporary
LEGACY_LAYOUT_SOLVERS: Mapping[str, str | None] = MappingProxyType(
    {"cpsat": "ortools", "simulated_annealing": "sa",
     "greedy": None, "bestfit": None, "firstfit": None}
)

def warn_planner_selection() -> None:
    """Warn once per process about a legacy value, or allow_exhaustive_search on a family."""
```

It uses `warn_once` (`torch_spyre/_inductor/logging_utils.py:88`) and is called from
`scratchpad_planning` when `allocator is None`. The legacy message reads: `layout_solver='cpsat'
selects the legacy LX planner, which is deprecated and will be removed. Set it to 'ortools' or unset
LAYOUT_SOLVER. If you set this to work around a planner problem, please open an issue.` A value with
no replacement names the default. An unknown value still raises `ValueError`, now listing the
accepted values. `LAYOUT_SOLVER=""` still raises, so a broken pipeline is not hidden.

Missing ortools needs nothing here: Stage 9's `select_families` promotes the fallback family with a
warning. The FX key still says `ortools`, as today's says `cpsat` on a machine that degrades.

### Retired values, the contract for Stage 13

Stage 12 retires nothing. The alias code lands in Stage 13.

| value or variable | Stage 13 behaviour |
|---|---|
| `cpsat`, `simulated_annealing` | Alias to `ortools`, `sa` for one tagged release with the warning, then `ValueError`. Normalize the stored value so the key moves |
| `greedy` | `ValueError` naming the default. The word stays reserved for the debug greedy family |
| `bestfit`, `firstfit` | `ValueError` naming the default |
| `ALLOW_EXHAUSTIVE_SEARCH` | `config.py` warns at import if set, since a stale env var is otherwise ignored silently |

### Cache behaviour

| what | reaction to the flip |
|---|---|
| FX cache, env unset | Key moves with `layout_solver`. Each graph recompiles once |
| FX cache, `LAYOUT_SOLVER=cpsat` | Old-path entries only, never a new-path entry, because the values differ. An entry that misses recompiles on the old path |
| `torch.compiler.save_cache_artifacts` bundles | Same key logic. A pre-flip bundle misses under the default and cannot serve a wrong plan |
| Kernel disk cache (`SPYRE_KERNEL_CACHE`) | Unaffected. `compute_specs_hash` hashes the specs (`torch_spyre/execution/kernel_cache.py:236`) |

Comparison runs disable caches regardless: a repeated arm would otherwise time a cache hit.

### What "side by side on CI models" is

| leg | corpus | where | decides |
|---|---|---|---|
| Compare configs | `FIXED_GRAPHS`, then the 35 `lx_zoo.MODELS` at default and tiling, on `fake` compiles | three configs, label `lx_compare`, both arms | plans, traffic, time, status |
| Card calibration | Stage 0 `zoo` and `matrix` tiers on the card, once at the 12b tip | by hand with `lx_gate.sh` | that `fake` parity holds on real uploads |
| Shadow lane | 33 configs below | `lx_planner_compare.yaml`, new path; baseline is that day's `tests-nightly` | correctness on the device |
| Real model | Granite tier `real`, and hf-adapters-dev | outside CI, by hand | plans on the largest graphs |

`fake` uses default layouts, so it proves parity, not production plans. A model the `fake` tier cannot
reach stays card-only and is listed. Whether `spyre_pf_x0`'s image can run `fake` is unverified, so
the compare configs stay on the default runner.

The `lx_compare` label goes on 33 configs that compile through the planner: the 18
`test_inductor_ops_lx_*` shards plus `test_scratchpad_use`, `test_work_division`,
`test_work_division_hint`, `test_solver_relayout`, `test_solver_auto_coarse_tiling`,
`test_lx_planning_suite`, `test_hbm_pool_planning`, `test_fallback_kernel`,
`test_lx_context_switching`, `test_lx_extern_kernel_clobber`, `test_coarse_tiling`,
`test_for_each_tile_e2e`, `test_sdpa_tiling`, `test_sliding_window_attention`,
`test_frontend_work_bounds`. Distributed suites run by one dispatch with
`test_type=suite_distributed`. Model-ops suites are left out: their verdicts are capability
records, so a planner failure would not fail the lane.

**Arm pairs.** Joint: `cpsat` with `ortools`, `simulated_annealing` with `sa`, at default and tiling.
Committed: the same pairs with co-optimization off. Old SA arms run with `LxContextSwitchingPass`
patched in, as in Stage 9.

**Plans and traffic.** Stage 0's deterministic settings on both sides; `install` patches
`SolveSettings.from_config` to one worker. Each arm is `lx_corpus.py record`, then `diff`.
`compare_planners.py` reads the recordings for `hbm_traffic_bytes` (Stage 9) and the predicted
`total_us` (`cost_model_pass.py:168`). The solver's objective is compared on committed arms only. On
12c arms it is not comparable, because it gains restickify terms, so those use the two report
metrics.

**Planning time.** Stage 9's Gate 3 protocol: one process per model, `A B B A`, three rounds, caches
off, `TORCH_SPYRE_FRONTEND_ONLY=1`. It reads `pass_loop_ns`, the `inclusive_ns` of
`stage:CustomPreSchedulingPasses:pass_loop` (`passes.py:614`), so work moved between passes is still
counted. The two `A` runs give the A/A spread. The stall watcher kills a run after 300 s of silence,
so the three configs get `-v -s` in `.github/suite_test_flags.yaml` and print one line per model.
Recording time per config is unmeasured; a config that passes the 60-minute limit splits by model.

**Known cost.** The new path adds `select_target`, `build_graph`, `check_decisions` and encoding.
If #5695 has landed, `solve_decides_work_division()` is false on a family with sticks pinned, so
`_distribute_work` runs in full there and is deferred on the old arm. That delta is stated in the
time gate's reasons.

### `tools/lx_planning/compare_planners.py`

Pure stdlib over Stage 0's readers, so its tests run on `spyre_pf_x0`.

```python
@dataclasses.dataclass(frozen=True)
class Thresholds:
    tolerance: float = 0.001
    cap: float = 0.04
    max_aa: float = 1.10      # starting values, rewritten after calibration
    total_time: float = 1.10
    median_time: float = 1.05
    p90_time: float = 1.15
    listed_time: float = 1.25
    min_seconds: float = 0.2

def compare(
    old: Path, new: Path, *, joint: bool,
    thresholds: Thresholds = Thresholds(),
    reasons: Mapping[str, str] = MappingProxyType({}),
) -> Report: ...

def main(argv: Sequence[str] | None = None) -> int: ...  # 0 pass, 1 fail, 2 invalid
```

`Report` holds entries (`case`, `metric`, `old`, `new`, verdict `ok`, `listed`, `fail` or `invalid`,
`reason`) and renders markdown. `reasons` is `tools/lx_planning/planner_compare_reasons.json`,
`"<case>/<metric>"` to a cause and an issue link. A reason matching no entry fails. A case missing
from one arm fails. `tests/inductor/test_planner_side_by_side.py` runs the recordings, calls `compare`
and attaches the markdown with `record_property`, so the JUnit file is the artifact.

### CI wiring

Stage 9 lands the two `_test_matrix.yaml` inputs and a dispatch-only `lx_planner_compare.yaml`. Stage 12
adds, in the workflow:

- **Forward lane, cron `0 8 * * *`:** `test_type: lx_compare`, `SPYRE_LX_ROUTE_TO_FAMILIES=1`, which
  moves unpinned and `cpsat`-pinned tests alike. Before the flip this is the only new-path coverage.
  After it, normal CI covers unpinned tests and this lane still covers the pinned ones.
- **Reversed lane, weekly, added in 12b:** the same configs with `LAYOUT_SOLVER=cpsat` and routing
  off, so the kill switch is exercised.

The workflow passes no ClickHouse secrets and is not in `push-to-clickhouse.yaml`'s list, so shadow
results never reach production rows. The `/.github/` owners pick the hours.

### Rollout

| step | what | leaves when |
|---|---|---|
| 0 | Entry: Stages 0-10 merged, Phase 2 exit met, no open `RuleViolation` issue, Stage 9's Gate 4 forced-failure report exists | all true |
| 1 | PR 12a: cron, compare configs, compare tool, `fallback_solver` default `"sa"`. Default compiles keep their plans | merged |
| 2 | Calibration, 7 nights. The `A` runs give old-vs-old. Each `Thresholds` time value and `max_aa` becomes the largest nightly value of that statistic, rounded up to the next 1%, committed before the soak | committed |
| 3 | Soak, 7 consecutive nights passing every gate below. A new-path failure resets the count; an invalid night neither counts nor resets. Dispatches, once each: Stage 13's C1 retarget branch, so `greedy`-pinned tests meet the new path now; `LAYOUT_SOLVER=simulated_annealing` with routing on; `suite_distributed`. By hand: card calibration, Granite. hf-adapters-dev and spyre-frameworks run `LAYOUT_SOLVER=ortools` and report | all gates hold; a named owner signs off in 12b |
| 4 | PR 12b. Its CI is `regression` on the new default. Dispatch `integration-tests.yaml` with `test_type=regression` on the branch. Do not set the `integration-test` label first, because it skips native CI (`torch_spyre_tests.yaml:115-127`) | merge queue passes |
| 5 | By the seventh post-flip night: one zoo comparison and `test_completed_reduction_relayout_device` (`test_work_division_hint.py:2187`) with `lx_solver_completed_reduction_routes=1`, under the committed-arm bands with each difference listed by rule. Then a one-line PR turns it on | merged |
| 6 | Stage 13 may start: 14 consecutive forward nights after step 5; two green reversed lanes; one tag holding the flip and a working kill switch; `fallback_solver` non-empty; routes on; no open issue resolved by using the kill switch | all true |
| 12c | PR turning `lx_sticks_in_solve` on, and removing the compare tooling if Stage 13 has already landed. Requires 11b merged and the weekly perf run. Independent of steps 5 and 6, and may come after Stage 13. Stage numbers do not change | gate below |

**Kill switch.** `LAYOUT_SOLVER=cpsat` or `simulated_annealing` before import, or `config.patch` at
runtime, with routing off. It is a config value, so it needs no cache flush. A repository-level revert
of the flip commit is one default plus one test pin. `RuleViolation` does not fall back to the old
path, and a failed family solve falls back to SA, never to old code. After 12c,
`SPYRE_LX_STICKS_IN_SOLVE=0` is the kill switch for sticks alone.

### Documentation

- `docs/source/compiler/scratchpad_planning.md`: status block (`:5-20`), Solvers (`:532-537`), a new
  "Selecting the planner" section with the value table, kill switch and removal condition.
- `docs/source/api/torch_spyre.rst`: `LAYOUT_SOLVER` (`:750`), `ALLOW_EXHAUSTIVE_SEARCH` (`:754`),
  `SPYRE_LX_SOLVER_RELAYOUT` (`:826`), and the correction to `SPYRE_LX_PLANNER_RELAYOUT` (`:715`).
- `docs/source/compiler/co_optimizer.md:99,107,166,205,214,323-328`, `sa_co_optimization.md:15`,
  `simulated_annealing_layout.md:25`: legacy status.
- `docs/source/user_guide/debugging/failure_classes.md:163`: add `LAYOUT_SOLVER=cpsat` as a triage step.
- `AGENTS.md:126`, `pyproject.toml:35-39,54`, the comment at `config.py:347-367`.
- `profile_ops.py:911-916`: accept a compile that went through `plan_allocation`.
- `torch_spyre/profiler/_ffdc.py:308`: add `LAYOUT_SOLVER`, `CO_OPTIMIZING_LX_PLANNING` and
  `AUTO_COARSE_TILING` to `_ENV_KEYS`.

## Changes

| file | change | PR |
|---|---|---|
| `tools/lx_planning/compare_planners.py` | new | 12a |
| `tests/inductor/test_compare_planners.py`, config | new, `[unit, regression, trunk]`, `spyre_pf_x0` | 12a |
| `tests/inductor/test_planner_side_by_side.py` | new | 12a |
| `tests/configs/.../test_planner_compare_{guards,zoo_default,zoo_tiling}_config.yaml` | new, `[lx_compare]` only | 12a |
| 33 existing configs | add `lx_compare` | 12a |
| `.github/runner_overrides.yaml`, `.github/suite_test_flags.yaml` | as above | 12a |
| `.github/workflows/lx_planner_compare.yaml` | forward cron | 12a |
| `torch_spyre/_inductor/config.py` | `fallback_solver` default | 12a |
| `torch_spyre/_inductor/config.py` | `layout_solver` default, comment | 12b |
| `torch_spyre/_inductor/scratchpad/allocator.py` | `LEGACY_LAYOUT_SOLVERS`, `warn_planner_selection`, error text | 12b |
| `tests/inductor/test_planner_default.py`, config | new, `[unit, regression, integration, trunk]` | 12b |
| `tests/inductor/test_frontend_work_bounds.py` | `:162` pin to `ortools`, recalibrated if it breaches (`:90-92`) | 12b |
| `tools/lx_planning/planner_compare_reasons.json` | new | 12b |
| `.github/workflows/lx_planner_compare.yaml` | reversed lane | 12b |
| `torch_spyre/profiler/_ffdc.py`, `profile_ops.py`, docs above, `AGENTS.md`, `pyproject.toml` | as above | 12b |
| `torch_spyre/_inductor/config.py` | `lx_solver_completed_reduction_routes` default | step 5 |
| `torch_spyre/_inductor/config.py`, docs | `lx_sticks_in_solve` default | 12c |
| compare tool, its tests, configs, label and `lx_planner_compare.yaml` | removed here only if Stage 13 landed first, otherwise by Stage 13's C13 | 12c |

**PR 12a, no plan change on a default compile:** (1) compare tool and tests; (2) side-by-side test and
three configs; (3) labels, run `make check-all-configs`; (4) cron; (5) fallback default.

**PR 12b:** (1) flip the default, re-pin `test_frontend_work_bounds.py`, add
`test_default_layout_solver_is_ortools` and the reasons file: this commit is the revert target; (2)
warnings, with tests, after the flip so the default never warns; (3) `_ffdc.py`, `profile_ops.py`;
(4) docs; (5) reversed lane.

## Interfaces

**Consumes**

| symbol | signature | from Stage | used for |
|---|---|---|---|
| `lx_corpus.py record`, `diff`, `guards`; `lx_gate.sh` | Stage 0 CLI | 0 | plan comparison, guards, calibration run |
| `recording`, `compile_model`, `load_solves`, `load_commits`, `check_guards`, `timing_report`, `Cell`, `FIXED_GRAPHS`, `lx_zoo.MODELS` (35) | Stage 0 API | 0 | side-by-side test |
| `fake` tier | compile tier | 0 | compare configs |
| `install` patching `SolveSettings.from_config` | `SolveSettings(DEFAULT_DTIME, workers=1, deterministic=True)` | 0, 7 | deterministic arms |
| `check_decisions`, `RuleViolation` | as in Stage 3 | 3 | correctness |
| `FAMILIES` | keys `"ortools"`, `"sa"` in `lx_allocator/family/registry.py`, mapping to `OrToolsFamily` and `SAFamily`; `FAMILIES[key].from_config(config)` builds one | 7, 8 | default value |
| `SolveSettings.from_config` | `time_limit` from `cpsat_time_limit_seconds`, `workers` from `get_cpu_count()`, at least 2 interleaved under `deterministic` | 7 | settings test |
| Family config reads | `Optimization.COARSE_TILING` in `graph.target.optimizations`, `lx_solver_relayout`, relayout caps, `ktir_emitter` | 2, 5, 6 | settings test |
| `select_target`, `config.target` | `(config) -> Target`; `TARGETS[config.target]` narrowed by `sencores`, LX size and `auto_coarse_tiling`; `config.target` defaults to `"spyre"` and is in the FX key | 2 | settings test, cache behaviour |
| `lx_solver_completed_reduction_routes` | `bool`, env `SPYRE_LX_SOLVER_COMPLETED_ROUTES`, default off | 6 | step 5 |
| `plan_allocation`, `optimize` | `plan_allocation(ir, config) -> SpyreOptimizationGraph`, reached from `scratchpad_planning` when `family_key(config)` is not `None`; `optimize(graph, family, fallback, settings)` | 9 | routing |
| `family_key`, `select_families` | `(config) -> str \| None`, `None` sends the compile to the old path, and `lx_route_to_families` routes the old values; `(config) -> tuple[Family, Family \| None]` from `FAMILIES[...].from_config(config)`, promoting the fallback when ortools is absent | 9 | routing, warnings |
| `lx_route_to_families`, `fallback_solver` | `bool`, env `SPYRE_LX_ROUTE_TO_FAMILIES`; `str`, env `FALLBACK_SOLVER`, defaults to `""` | 9 | lanes, 12a |
| `PlanOutcome`, `last_outcome` | Stage 9's record of each `plan_allocation` call; `.status` holds `SolveStatus.name`; `.fell_back`; `last_outcome() -> PlanOutcome \| None`, the latest call in the process | 9 | guards |
| `hbm_traffic_bytes`, `outcome` in the commit record; `decisions_from_graph`; `ortools` and `sa` cells | Stage 9's harness extensions | 9 | arm comparison |
| Committed mode | `graph.divisions_committed`, `graph.target.cost_model` with `traffic_only=True` | 1, 2, 4, 9 | committed arms |
| SA baseline patch | `LxContextSwitchingPass` in `select_allocator`'s SA branch, test-side | 9 | old SA arms |
| Gate 3 protocol, Gate 4 report | as in Stage 9 | 9 | time gate, 12a |
| CI wiring | `_test_matrix.yaml` inputs `lx_route_to_families` and `layout_solver`, exporting `SPYRE_LX_ROUTE_TO_FAMILIES` and `LAYOUT_SOLVER`, unset when empty; dispatch-only `lx_planner_compare.yaml` | 9 | lanes |
| `optimizer_pass_sources()` | planner sources in the FX key | 9, 10 | cache behaviour |
| `commit` | `commit(graph)`, runs `post_optimization_passes` | 10 | context switching on families |
| `cost_expr_record` writer | full record from decoded graph values | 10 | `SPYRE_DUMP_COST_EXPR_FILE` |
| Old path intact | `ScratchpadAllocator`, `CoOptimizingAllocator`, `_push_allocation` keep working through Stage 12 | 9, 10 | kill switch |
| `lx_sticks_in_solve`, `solve_decides_work_division()` | true on a family only with the switch on | 11 | 12c only |

**Provides**

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `config.layout_solver` default | `"ortools"` | 13 | final |
| `config.fallback_solver` default | `"sa"` | 13 | final |
| `config.lx_solver_completed_reduction_routes` default | on | 13 | final |
| `config.lx_sticks_in_solve` default | on, step 12c | parent's Goal | final |
| `LEGACY_LAYOUT_SOLVERS`, `warn_planner_selection` | above | 13 | temporary, removed in Stage 13 |
| Settings and retired-values tables | above | 13 | contract |
| Stage 13 entry condition | rollout step 6 | 13 | contract |
| `compare_planners.compare`, `Thresholds`, label `lx_compare`, lane cron | above | 12c, 13 | temporary, removed by whichever of Stage 13 and 12c lands last |

## Tests

**`tests/inductor/test_compare_planners.py`** (no card, loaded by path as `test_eval_model.py:43-45`).
`test_identical_recordings_pass`; `test_joint_plan_difference_fails_even_with_lower_traffic`;
`test_listed_rule_difference_passes_on_joint_arm`; `test_traffic_bands` (+0.09% ok, +1% needs a reason,
+4.1% fails with one); `test_geomean_above_one_fails`; `test_committed_objective_worse_when_both_optimal_fails`;
`test_time_uses_median_of_rounds`; `test_time_ignores_graphs_below_min_seconds`;
`test_noisy_aa_is_invalid_not_failed` (exit code 2); `test_per_graph_time_is_listed_not_failed`;
`test_unforced_fallback_fails`; `test_missing_case_fails`; `test_stale_reason_fails`.

**`tests/inductor/test_planner_side_by_side.py`** (label `lx_compare`). Classes `TestGuards`,
`TestZooDefault`, `TestZooTiling`, each with `test_joint_plans_identical`, `test_committed_bands`,
`test_planning_time`, `test_guards`, `test_decisions_check_clean`, reading one class cache as
`test_frontend_work_bounds.py:179-184` does, so gates pass in any order (`AGENTS.md:97`). `TestGuards`
adds `test_results_match_reference`: each arm's executed result against the fp32 CPU reference, at
the tolerance the case carries in `test_scratchpad_use.py`, on the card.

**`tests/inductor/test_planner_default.py`** (12b)

| test | asserts |
|---|---|
| `test_default_layout_solver_is_ortools` | with the env unset, a config loaded by path (`test_async_compile.py:52-61`) has `ortools` |
| `test_cache_key_differs_between_old_and_new` | key config for the four values is pairwise different (`test_solver_relayout_decision.py:286`) |
| `test_every_old_value_reaches_the_old_path` | routing off: `family_key(config)` is `None`, `plan_allocation` is not called and `select_allocator` gives the old class |
| `test_family_values_reach_plan_allocation` | `ortools`, `sa`, co-opt on and off, each running `FAMILIES[key].from_config(config)` through `select_families` |
| `test_legacy_value_warns_once`, `test_value_without_replacement_names_the_default` | message and dedup |
| `test_default_does_not_warn`, `test_exhaustive_flag_warns_only_on_a_family` | |
| `test_invalid_value_lists_the_choices` | `bogus` raises `ValueError` |
| `test_lx_planning_off_skips_the_optimizer` | `plan_allocation` not called |
| `test_solver_relayout_off_offers_no_relayout`, `test_planner_relayout_off_keeps_solver_relayouts` | today's behaviour, kept on purpose |
| `test_auto_coarse_tiling_reaches_the_target`, `test_time_limit_and_workers_reach_settings` | settings follow config: `Optimization.COARSE_TILING` in `graph.target.optimizations` |

**Existing suites.** The parent's unit list, serially, and its device list, in the lane. PR 12b's CI
runs all of `regression` on the new default.

**What proves the claim.** The compare configs prove plans match. The lane and PR 12b's CI prove
correctness. The reversed lane proves the old path stayed selectable.

## Parity and gates

Parent gate: "Run the old and new paths side by side on CI models, comparing traffic, planning time
and correctness, then make the new optimizer the default."

With `lx_sticks_in_solve` off the search space is today's, so the flip is an identity gate. PR 12b
may not merge until every line holds over the soak.

**Plans.**

| id | check | pass |
|---|---|---|
| I1 | Joint arms, OR-Tools, default and tiling, `FIXED_GRAPHS` and zoo on `fake`, and `zoo` on the card once | `diff` empty: proto, decisions, commit, snapshot, status, predicted total. A non-identical plan is a failure even if traffic falls |
| I2 | Joint arms, SA | Same, except differences carrying a rule on Stage 8's listed rule differences, which go in the reasons file |
| B1 | OR-Tools committed | Per solve, traffic objective no worse when both are `OPTIMAL`. `hbm_traffic_bytes` up to 1.001 ok, up to 1.04 with a reason, above that fail. Geomean at most 1.000 |
| B2 | SA committed | Per corpus, median over five seeds. Accepted as a listed plan change at Stage 13 C4 |

`total_us` is reported on B1 and B2, with any increase over 0.1% listed. The bands come from the 0.1%
"worse" line in `sa-vs-cpsat-REPORT.md` and the printer's documented 4% bound
(`cpsat_expr_printer.py:55-57`). A larger loss cannot be blamed on the encoding.

**Planning time.** `pass_loop_ns`, median of three rounds per graph, graphs of at least 0.2 s. The
values below are starting points that rest on few observations. Calibration replaces them before the
soak (step 2).

| id | check | start | source |
|---|---|---|---|
| P0 | A/A total ratio per corpus; beyond it the night is invalid | 1.10 | zoo r1/r2: 174.4/158.5, driven by one search |
| P1 | `sum(new)/sum(old)` | 1.10 | parent: "within run-to-run noise" |
| P2 | median per-graph ratio | 1.05 | per-pass spread 3.68% median (issue 5682) |
| P3 | p90 per-graph ratio | 1.15 | per-pass spread 12.9% at p90 |
| P4 | above 1.25 and at least 1 s | listed, not failed | up to ~25% per model across processes; a per-graph bound would flake |

Since joint protos are identical, search time is not the new path's. The time gate judges what the
new path adds around the solve, and a delta over P1 is reported by stage (`build_graph`, encode,
check, `_distribute_work`).

**Status.**

| id | check | pass |
|---|---|---|
| S1 | `check_guards` on the fixed set | no `fallback`, `solve_error` or `limit`, on either arm |
| S2 | Every graph | `last_outcome().fell_back` false unforced; `status` equal, as part of I1 |
| S3 | Granite, by hand | Plans identical at deterministic settings. Graphs 2 and 3 searched 46.4 s and 56.8 s of a 60 s limit, and that search has varied 31.9 s to 17.3 s on another graph, so a one-shot production status proves nothing. At one worker and a fixed seed, identical protos give equal deterministic time, and that is the check |
| S4 | Stage 0's size bounds | pass on the new arm |

**Correctness.**

| id | check | pass |
|---|---|---|
| C1 | Lane, 33 configs | Green. A failure counts if the same test passed in that day's `tests-nightly`. A test failing on both goes to its owner and does not reset the soak |
| C2 | `check_decisions`, new arms only | zero reports and zero `RuleViolation`. Old arms may hold known reports |
| C3 | `test_results_match_reference` | each arm matches the reference on every fixed-set model |
| C4 | `suite_distributed` and Stage 13 retarget dispatches | green once |
| C5 | Forced failure, Stage 0's `failure` tier | SA fallback plan passes `check_decisions` |
| C6 | Reversed lane | green after the flip |

**12c gate, switch on against switch off, on `ortools`.** Plans differ here by design, so this is a
quality gate, on the zoo and Stage 0's restickify graphs.

| id | check | pass |
|---|---|---|
| T1 | `total_us` per graph | up to 1.001 ok, up to 1.04 with a reason, above that fail |
| T2 | `hbm_traffic_bytes` per graph | same bands; above tolerance is `listed` if `total_us` fell by more than 0.1% |
| T3 | geomean of each | at most 1.000 |
| T4 | Status | `check_guards` as S1; a status drop is compared against repeated switch-off runs or read from deterministic time, not one shot |
| T5 | Time | the owner's written budget (Open questions) |
| T6 | Device perf | two switch-off runs give the per-op A/A spread, since the references hold no noise figure and the report format is outside this repo. Geomean new/old within that spread, and any op worse than twice it gets a reason. `clickhouse_env: dev` |
| T7 | Correctness | C1-C3, and Granite by hand |

**A difference.** A `listed` entry goes in the reasons file with a cause and an issue link, reviewed
in the PR. A `fail` is fixed in the stage that owns the rule, the soak count resets, and the comparison
reruns. Thresholds do not move after calibration to make a run pass.

## Risks

- **Pinned tests hide the new path after the flip.** PR CI runs `cpsat` pins on the old path. Caught
  by the forward lane staying on until Stage 13.
- **The kill switch rots.** Caught by the reversed lane.
- **A `fake` input hides a production difference.** Caught by the card calibration and C3.
- **Identity fails on a graph the owning stage's gate passed, because main moved.** Caught nightly,
  and by Stage 5's `test_lx_live_parity.py` in PR CI. Fixed in the family's `encode`, per
  `AGENTS.md`.
- **Old arm drifts on main.** The nightly gate compares against the live old path, so drift shows up
  as an I1 failure with the cause in `diff`.
- **Time-limit squeeze on large graphs under 12c.** Growth of 5.6% on Granite graph 3 hits the limit
  and the plan silently worsens. Caught by T4.
- **Downstream pins.** spyre-frameworks or hf-adapters may set `LAYOUT_SOLVER=cpsat` and stay on the
  old path until Stage 13 breaks them. I cannot see those repos. Caught by the warning and by asking
  at step 3.
- **Suite timeouts after the flip.** Caught by the lane, under the same 60-minute limit.
- **Perf history shifts at 12c.** The PR notes it, so a dashboard reader can find the step.

## Deviations from the parent plan

1. **Three PRs, not one.** The comparison has to run on `main` before the default moves, and scheduled
   workflows run only from the default branch. So 12a (lane), 12b (flip) and 12c (sticks) land apart.
2. **The flip does not wait for Stage 11.** The parent orders Stage 11 first. The flip runs with
   `lx_sticks_in_solve` off, and turning it on is 12c, gated on 11b and the weekly perf run. The parent's
   Goal is reached at 12c, which may follow Stage 13.
3. **"CI models" do not exist as such.** No workflow here runs a full model. Resolution: the four legs
   above. The real-model leg is by hand, so that part of the gate rests on a person's report.
4. **Planning time is a gate, while issue 5682 says "a report, not a gate".** Follows the parent, at
   aggregate level only, with an A/A control and per-graph time listed.

## Open questions

1. **Without ortools, should a default compile degrade to SA or keep raising?** Planned: degrade through
   Stage 9's `select_families`, since a `ValueError` on s390x helps nobody. If "keep raising",
   `select_families` raises instead of promoting. Stage 12 changes only its docs.
2. **What planning-time budget applies to 12c?** Stage 11 projects 174 s to 230-460 s on the zoo, so
   P1's bound cannot hold there. Recommendation: one written limit once 11b's numbers exist. Only T5
   changes.
3. **Are 7 nights before the flip, 14 after and one tag the right soak?** Recommendation: yes as a floor.
   I have no flake-rate data. Longer changes only steps 3 and 6.
4. **Will the `/.github/` owners accept 36 nightly card jobs plus a weekly lane?** Recommendation: run it
   through step 6, then keep the parent's device list and the 3 compare configs for 12c. If refused,
   run by dispatch twice a week and scale the soak; the gates do not change.

## Out of scope

- Deleting anything, and the alias code: Stage 13.
- Sticks in the solve itself: Stage 11. Stage 12 owns only the default.
- Routing to `plan_allocation`, the failure flow in `optimize`, committed mode, the
  `_test_matrix.yaml` input and the cache-key fix: Stage 9.
- `commit(graph)`, post passes and the cost dump: Stage 10.
- Making `SPYRE_LX_PLANNER_RELAYOUT=0` disable solver relayouts: not planned; the docs are corrected.
- Turning on `AUTO_COARSE_TILING` by default, and the debug greedy family: not in this plan.
