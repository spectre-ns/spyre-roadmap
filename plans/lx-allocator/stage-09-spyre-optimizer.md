# LX allocator, Stage 9: `plan_allocation` and `optimize`

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0, 1, 2, 3, 4, 5, 6, 7, 8. Needed by: Stages 10, 11, 12, 13.

All `file:line` citations are under `torch_spyre/_inductor/` unless they start with
`tests/`, `docs/` or `torch/` (torch 2.13), or name a file at the repo root.

## Goal

When this PR merges, a compile with `config.layout_solver` set to a `FAMILIES` key runs
the whole new path end to end: `plan_allocation` selects the target and builds the graph,
`optimize` encodes, solves, reads back and runs `check_report`, and `plan_allocation`
commits through a temporary bridge into today's commit code. The Failure decision of the
parent plan holds on that path, and the Phase 2 exit gate has passed for the OR-Tools and
SA families. The planner's source files reach the FX
cache key, so every user recompiles once. The default stays `cpsat`, and with any existing
`layout_solver` value the old path's plans are unchanged: same allocator, same CP model,
same plans, same `SolveError` re-plan. No old code is deleted.

## Today

### Where planning is entered

- `CustomPreSchedulingPasses.passes` lists `_distribute_work`
  (`passes.py:574`), `_maybe_scratchpad_planning` (`passes.py:577`) and
  `elide_proven_read_copies` (`passes.py:580`), in that order.
- Each pass runs inside `_flat_operations` (`passes.py:446-476`). If the graph holds
  `LoopNode`s, `graph.operations` is an `OpsView` and rebinding it raises, so the
  planner must edit in place.
- `_maybe_scratchpad_planning` (`passes.py:425-432`) returns when `config.lx_planning`
  is off, otherwise calls `anchor_lx_relayout_ownership(graph)` and passes the result to
  `scratchpad_planning(graph, lx_relayout_plans=plans)`.
- `anchor_lx_relayout_ownership` (`scratchpad/lx_relayout.py:1890`) returns `None` when
  `co_optimizing_lx_planning` is on (`:1901-1906`). With it off it may commit a
  different owner order through `commit_tensor_work_division` (`:2003`).
- `_runs(...)` (`passes.py:359-366`) names the source files `_uuid` hashes
  (`passes.py:164-171`), and `_uuid` hashes only the file that defines each listed object.
  None of this reaches the FX cache key today. `CustomPreSchedulingPasses` runs from a
  patch of `GraphLowering._update_scheduler` (`patches.py:213-229`), not from a config
  slot, so nothing calls its `uuid()` (`passes.py:698`). Only the five config-slot
  pipelines are hashed (`patches.py:182-186`). The docstring at `passes.py:492` and
  `docs/source/compiler/coarse_tiling_loops.md:1873` say otherwise. Config values do reach
  the key (`config.py:498-503`).
- `scratchpad_planning` (`scratchpad/allocator.py:5472`) calls `select_allocator()`
  (`:5377`), then `allocator.plan_allocation` inside `try/except SolveError`
  (`:5493-5515`).
- `select_allocator` picks `CoOptimizingAllocator` when `co_optimizing_lx_planning` is
  on (`:5431-5463`), otherwise `ScratchpadAllocator` (`:5465`). An unknown
  `layout_solver` raises `ValueError` (`:5410-5414`).
- `ScratchpadAllocator.plan_allocation` is a template method
  (`scratchpad/allocator.py:1012-1052`):
  pre-passes, `_prepare_buffers`, `_build_solver`, `_solve`,
  `_finalize_lx_relayout_allocation`, `_post_solve`, `_get_spill_reasons`,
  `_push_allocation`, `_log_lx_pinning`, post-passes.

### Failure and fallback flows today

| # | trigger | where | what happens |
|---|---|---|---|
| 1 | Priced solve returns neither `OPTIMAL` nor `FEASIBLE` (`INFEASIBLE`, `UNKNOWN`, `MODEL_INVALID`) | `scratchpad/ilp_solver_ortools.py:871-875` | `SolveError` |
| 2 | Any ladder rung returns neither | `ilp_solver_ortools.py:1433-1440` | `SolveError`, even when an earlier rung found a plan |
| 3 | `SolveError` reaches `scratchpad_planning` | `allocator.py:5495-5515` | Logged at INFO. A fresh `ScratchpadAllocator(GreedyLayoutSolver)` replans with the committed work divisions and keeps the failed allocator's post-passes. It recollects relayout plans (no `lx_relayout_plans` argument) |
| 4 | `FEASIBLE` at the time limit | `ilp_solver_ortools.py:871`, `:1435` | A plan. Nothing distinguishes it except `last_solve_stats` (`:1347-1354`) |
| 5 | Cost expression cannot be built (`ValueError`, `RuntimeError`, `TypeError`, which covers `Unsupported`) | `allocator.py:3321-3336` | Warning with traceback, `cost_expr = None`, ladder. Re-raises when `_cpsat_warn_on_cost_expr` is false |
| 6 | Cost expression cannot be lowered | `ilp_solver_ortools.py:886-900` | Warning, `_minimize_cost_expr` returns `None`, ladder. Same re-raise |
| 7 | `auto_coarse_tiling` with a `linear_cost_only` solver | `allocator.py:3275-3299` | No failure: the cost expression is withheld and the ladder ranks plans |
| 8 | Ladder runs | `ilp_solver_ortools.py:1449-1455` | Every relayout copy is pinned out, because the ladder has no relayout price |
| 9 | Ortools missing | `allocator.py:5345-5366`, `ilp_solver_ortools.py:663-669` | `warn_once`, `GreedyLayoutSolver`. With co-optimization on, `ValueError` unless `allow_exhaustive_search` (`allocator.py:5436-5450`) |
| 10 | SA has no usable cost expression | `scratchpad/sa_cooptimizer.py:789-793` | INFO, memory-only objective |
| 11 | SA search | `sa_cooptimizer.py:778-805`, `:1897-1934` | Never returns "no plan". Best-seen from the seed state. Its budget is steps, not wall-clock (`:90-100`) |
| 12 | Fixed division fails a hard domain | `allocator.py:2376-2386`, `:3785`; `tests/inductor/test_work_division.py:3518` | `Unsupported`, before any solve. The comment at `allocator.py:3687-3689` says this must not be bypassed |
| 13 | No legal candidate for an op | `allocator.py:3801-3802` | `Unsupported` |
| 14 | Post-solve commit cannot state the plan | `allocator.py:2907-2909`, `:2939`, `:2973`, `:2990`, `:3029`; `scratchpad/coarse_tiling.py:698`, `:784` | `Unsupported`. Deliberately not `SolveError`: tilings are already applied, so a re-plan would see a graph it did not plan (`allocator.py:2902-2905`, `tests/inductor/test_work_division.py:3603`) |
| 15 | Relayout record disagrees with the solve | `allocator.py:3480-3500`; `scratchpad/lx_relayout.py:2015-2029`, `:2044-2081` | `AssertionError` or `RuntimeError`, compile fails |
| 16 | `_justify` cannot squeeze | `ilp_solver_ortools.py:1880-1887` | Keeps the solver's offsets. Not a failure |

### What `_push_allocation` and the tail read

`_push_allocation` (`allocator.py:2036-2158`) reads `b.name`, `b.address` and `b.lx_view`
from each record, plus `self._validated_drain_plans` and
`self._collective_output_stages`. It skips records whose name starts with
`__spyre_lx_relayout__:` (`:2066`), and calls `materialize_lx_relayouts` last (`:2158`).
On the joint path it runs after this tail, and each step reads more:

| step | reads from records |
|---|---|
| `_finalize_lx_relayout_allocation` (`:3453-3502`) | `chosen_relayouts`, `chosen_division`, `address`, the copy's record by `relayout_copy_name` |
| `apply_chosen_tilings` (`scratchpad/coarse_tiling.py:772`, `chosen_tilings` `:754`) | `name`, `chosen_division`, `core_divisions[i].tiling` |
| `check_priced_footprints` (`:2947`) | `chosen_division`, `tiling`, `size`, `per_core_size` |
| `commit_divisions` (`:2886`), `planned_splits` (`coarse_tiling.py:819`) | `chosen_division`, `splits`, `output_splits`, `label`, and the set of planned names |
| `commit_lx_views` (`:2999`) | `address`, `isinstance(RelayoutCopyBuffer)`. It writes `lx_view` from the ownership judge on the live graph |
| `_stage_collective_outputs` (`:3525`) | `address`, `chosen_division`, `per_core_size`, `size`, for every placed record, relayout copies included |
| `_assert_drain_plan_committed` (`:691`) | Both records' `core_divisions` and `chosen_division`, as an index pair in `match_pairs` |

Nothing in the tail reads `uses`, `in_place_parents` or `boundary`.

### Surprises

- **The joint SA path has no `LxContextSwitchingPass`.** `select_allocator` builds
  `CoOptimizingAllocator(layout_planning=SaCoOptimizingSolver, size=size)`
  (`allocator.py:5432-5435`) without `post_optimization_passes`. The other two branches
  pass them (`:5451-5463`). `enable_lx_context_switching` defaults on (`config.py:56`),
  which also switches off the extern-kernel residency guard (`allocator.py:857-862`), so
  the invariant stated at `allocator.py:5416-5420` is broken. This is a real bug on main,
  reachable only with `LAYOUT_SOLVER=simulated_annealing`, not from default config. The
  exposed case is a buffer live across a single-output, risky `FallbackKernel`. Whether it
  clobbers on a card is unverified. `tests/inductor/test_scratchpad_use.py:1587` covers
  only the cpsat branch.

- **Placement-only CP-SAT and SA take no relayouts.** `supports_paired_buffers()` is
  false for both (`ilp_solver_ortools.py:632`, `scratchpad/simulated_annealing.py:114`),
  so `_prepare_buffers` drops the plans (`allocator.py:1069-1075`). Greedy takes them
  (`scratchpad/greedy_solver.py:33`), which means today's re-plan can fire planner
  relayouts that the SA fallback cannot.

## Design

### Entry point and selection

`scratchpad_planning` stays the single entry. It is the only old function that changes
behavior, and only when a family is configured.

```python
def scratchpad_planning(graph, allocator=None, *, lx_relayout_plans=None) -> None:
    if allocator is None:
        # Imported locally: optimizer.py imports this module through its bridge.
        from torch_spyre._inductor.scratchpad.lx_allocator.optimizer import (
            family_key, plan_allocation)

        if family_key(config) is not None:
            try:
                plan_allocation(graph, config)
            except SolveError as error:  # no plan left, or ortools absent and no fallback
                _greedy_replan(graph, error, _context_switching_passes())
            return
        allocator = select_allocator()
    try:
        allocator.plan_allocation(graph, lx_relayout_plans=lx_relayout_plans)
    except SolveError as error:
        _greedy_replan(graph, error, allocator.post_optimization_passes)
```

`_greedy_replan(graph, error, post_passes)` is today's except body moved verbatim
(`allocator.py:5496-5515`), including its INFO line, which
`tests/inductor/test_scratchpad_use.py:1613` asserts. `_context_switching_passes()` is the
list at `allocator.py:5416-5428`, extracted so `select_allocator` and the new path share
it. `lx_relayout_plans` is dropped on the new path: `anchor_lx_relayout_ownership`
still runs and any ownership it commits stays, but neither `CpSatLayoutSolver` nor
`SimulatedAnnealingLayoutSolver` consumes its plans today, so the committed baselines do
not either.

```python
def family_key(config) -> str | None:
    """The FAMILIES key this config routes to, or None for the old path."""

def select_families(config) -> tuple[Family, Family | None]:
    """The primary family and the fallback, built fresh for one compile."""
```

- `family_key` starts from `config.layout_solver`. With `config.lx_route_to_families` on,
  `"cpsat"` maps to `"ortools"` and `"simulated_annealing"` to `"sa"`. `"greedy"`,
  `"firstfit"` and `"bestfit"` are never remapped, since no family replaces them yet.
- A key not in `FAMILIES` gives `None`, and the old path runs. No other value of
  `layout_solver` is touched.
- `select_families` builds `FAMILIES[family_key(config)].from_config(config)` and
  `FAMILIES[config.fallback_solver].from_config(config)`, so `auto_coarse_tiling`,
  `lx_placement` and the rest reach them. An empty `fallback_solver` gives `None`.
- `ImportError` from building the primary (ortools absent, see `ilp_solver_ortools.py:663`)
  does a `warn_once` and promotes the fallback family to primary with no fallback, so a
  platform without ortools keeps LX planning. With no fallback configured it raises
  `SolveError`, and `scratchpad_planning` sends the compile to the greedy re-plan.
- `config.fallback_solver` not in `FAMILIES` and not empty raises `ValueError`, like
  an unknown `layout_solver`.
- There is no exhaustive-search or `allow_exhaustive_search` path for a family.
- Every family runs `_context_switching_passes()` after commit.

New config values, both in the FX graph cache key because they change the plan:

```python
fallback_solver: str = os.environ.get("FALLBACK_SOLVER", "")
lx_route_to_families: bool = os.environ.get("SPYRE_LX_ROUTE_TO_FAMILIES", "0") == "1"
```

- `fallback_solver` is the family re-encoded when the configured family ends with no
  plan: a `FAMILIES` key, or `""` to keep the greedy re-plan as the only fallback. It
  defaults to `""`, and Stage 12 flips it to `"sa"` (see Parity and gates).
- `lx_route_to_families` routes the legacy `"cpsat"` and `"simulated_annealing"` values
  to their families, so a whole suite can run on the new path without editing the tests
  that patch `layout_solver`. Stage 12's lanes use it, and Stage 13 removes it.

Neither name matches a prefix in `_cache_config_ignore_prefix` (`config.py:503-519`), so
the test pinning those prefixes stays green unchanged.

### Planner sources in the FX cache key

Without this, a planner fix keeps serving stale plans from a persistent cache. Two changes
in `passes.py` make the planner's sources reach the key:

```python
class CustomPostFusionPasses(_SpyreNodePassPipeline):
    def uuid(self) -> Any | None:
        # CustomPreSchedulingPasses runs from a patch, not a config slot, so Inductor
        # never asks it for a uuid. This hashed slot carries it into the key.
        return (super().uuid(), CustomPreSchedulingPasses().uuid())

@_runs(anchor_lx_relayout_ownership, scratchpad_planning, plan_allocation,
       *optimizer_pass_sources())
def _maybe_scratchpad_planning(graph: GraphLowering) -> None: ...
```

```python
def optimizer_pass_sources() -> tuple[Any, ...]:
    """One class or function per planner module that `_runs` does not already name."""
```

- It lives in `lx_allocator/optimizer.py` and names every module under `lx_allocator/`,
  plus `ilp_solver_ortools` and `sa_cooptimizer`. Stage 10 adds `commit.py`, and Stage 13
  reuses it.
- The post-fusion slot is hashed on torch 2.13 (`torch/_inductor/codecache.py:1534`), the
  pinned version (`pyproject.toml:15`).
- **Cost:** one recompile for every user when Stage 9 lands, and one after each later
  change to these files. The statements cited in Today become true.
- The decorator imports the family modules at the first compile (`patches.py:167`), so
  ortools now loads on non-cpsat compiles too. Every import of it is guarded, so nothing
  fails. The docstring at `allocator.py:5352` changes, and the import time is unmeasured.

### `plan_allocation` and `optimize`

New module `scratchpad/lx_allocator/optimizer.py`. It imports no solver at module level.
There is no optimizer object: each call builds its families and graph and keeps nothing.

```python
@dataclass
class PlanOutcome:
    """How one planning call ended, for logs, the timing record and the harness."""
    family: str                  # FAMILIES key whose solve decided the outcome
    status: str                  # that Solution.status.name
    first_status: str | None     # the primary's status name when fell_back
    fell_back: bool
    decided: bool                # read_back ran and the check passed, so commit follows

def plan_allocation(ir: GraphLowering, config) -> SpyreOptimizationGraph: ...
def optimize(graph: SpyreOptimizationGraph, family: Family, fallback: Family | None,
             settings: SolveSettings) -> PlanOutcome: ...
def last_outcome() -> PlanOutcome | None: ...
```

`commit(graph)` is Stage 10's. Until then `plan_allocation` calls the temporary bridge in
its place.

```python
def plan_allocation(ir, config):
    family, fallback = select_families(config)
    committed = not config.co_optimizing_lx_planning
    target = select_target(config)                          # Stage 2's function
    if committed:
        target = replace(target, cost_model=SpyreCostModel(traffic_only=True))
    for candidate in (family, fallback):
        if candidate is not None and (gaps := coverage(candidate, target).gaps):
            raise ValueError(f"{_key_of(candidate)} cannot encode {gaps}")
    drain_plans = validated_drain_plans(ir, division_is_fixed=committed)
    graph = build_graph(ir, target, drain_plans=drain_plans,
                        divisions_committed=committed)      # Stage 2's function
    outcome = optimize(graph, family, fallback, SolveSettings.from_config(config))
    _record_outcome(outcome)
    if outcome.decided:
        _commit_via_bridge(ir, graph, drain_plans, divisions_committed=committed)
    return graph

def optimize(graph, family, fallback, settings):
    if not graph.nodes:                     # as ilp_solver_ortools.py:746, sa:786
        return PlanOutcome(_key_of(family), "OPTIMAL", None, False, False)
    solved = _solve_with(family, graph, settings)
    if solved.solution.status is SolveStatus.INFEASIBLE:
        return _return_infeasible(graph, solved)
    if not solved.solution.ok:
        solved = _fall_back(graph, fallback, settings, solved)
        if solved.solution.status is SolveStatus.INFEASIBLE:
            return _return_infeasible(graph, solved)
    with timing_recorder.stage("lx:read_back", family=solved.key):
        solved.family.read_back(solved.solution, graph)
    with timing_recorder.stage("lx:check", family=solved.key):
        report = check_report(graph)
    if report.violations or report.errors:
        raise RuleViolation([*report.violations, *report.errors])
    return PlanOutcome(solved.key, solved.solution.status.name, solved.first_status,
                       solved.first_status is not None, True)
```

- `coverage(family, target)` runs at config time, before the graph is built, for the
  primary and the fallback. A non-empty `gaps` is a family that claims to be selectable
  and cannot honor something the target uses, so it raises `ValueError` naming the family
  and the flags. `unused` (a chosen step kind or an optimization the family never uses,
  such as SA's `RELAYOUT`) does not raise.
- `check_report(graph)` is the full form of `check_decisions(graph)`: it also returns the
  rules that raised while being judged.
- `_key_of(family)` is the `FAMILIES` key of `type(family)`.
- `last_outcome()` returns the outcome of the most recent `plan_allocation` in this
  process, as `last_cost_report` does for the cost report. Logs, the timing record, the
  harness and #5682's Solve-status guard read it.

`_solve_with(family, graph, settings, first_status=None)` runs one attempt and returns a
private `_Solved(key, family, encoding, solution, first_status)`:

1. Copy `graph.pins`, so greedy pins the family's `encode` adds can be dropped on a retry.
2. `encoding = family.encode(graph)` under `lx:encode`. `encode` reads
   `graph.target.cost_model`. This runs inside the pass, where `V.graph` is active, as
   pricing needs (`cost_features.py:1139-1143`).
3. `solution = family.solve(encoding, settings)` under `lx:solve`, with the status name in
   the event's `meta`.
4. `status is SolveStatus.INVALID` raises `RuntimeError` (see Deviations).

Strong exception guarantee, which everything below relies on: **nothing touches the IR
before `_commit_via_bridge`.** `build_graph`, `encode`, `solve` and `read_back` only read
the IR. This is what makes a fallback and a `RuleViolation` safe, as the comment at
`allocator.py:5496-5500` requires of today's re-plan.

`optimize` takes no IR and never commits, so unit tests drive it on a hand-built Stage 1
graph with fake families.

### The Failure decision, flow by flow

| today (# above) | new path | owner of the behavior |
|---|---|---|
| 1 `INFEASIBLE` | `_return_infeasible` | Stage 9, with `SolveStatus` from Stages 7, 8 |
| 1 `UNKNOWN`, which Stage 7 reports as `NO_PLAN` | `_fall_back` | Stage 9 |
| 1 `MODEL_INVALID`, reported as `INVALID` | `RuntimeError`, compile fails | Stage 9 (Deviations) |
| 2, a later ladder stage fails | `NO_PLAN`, then `_fall_back`. The earlier stage's plan is lost, as today | Stages 7, 9 |
| Pack mode runs out of attempts | `NO_PACKING`, then `_fall_back` | Stages 7, 9 |
| 3 greedy re-plan | Stays behind `_fall_back`: an empty `fallback_solver`, or a fallback that also finds no plan, raises `SolveError`, which reaches `_greedy_replan`. Stage 13 removes it | Stages 9, 13 |
| 4 `FEASIBLE` at the limit | A plan. `solution.ok` is true, so it takes the normal path: `read_back`, check, commit. No fallback, no warning. `outcome.status == "FEASIBLE"`, logged at INFO | Stage 9 |
| 5, 6 objective unusable | Handled inside `encode`. Same degrade as today (ladder, copies pinned out, warning) and same re-raise. `optimize` sees an ordinary solve | Stages 4, 6, 7 (Consumes) |
| 7, 8 ladder under tiling | Ordinary solve as far as `optimize` is concerned | Stages 4, 6, 7 |
| 9 ortools missing | Fallback family becomes primary, else the greedy re-plan | Stage 9 |
| 10 SA without a score | Handled in `SAFamily.score` | Stage 8 |
| 11 SA never fails | `Solution` with `SolveStatus.FEASIBLE`. Seed trouble is handled in seeding, per postmath in PR #5: Stage 9 adds nothing after SA except flow 3 | Stage 8 |
| 12 illegal fixed division | `Unsupported`, same message, from `_return_infeasible` if `encode` did not already raise it | Stages 5, 9 |
| 13 no legal candidate | `Unsupported` from `encode`, propagates | Stage 5 |
| 14, 15 post-solve | Unchanged: they run in the bridge and propagate. No fallback, because the IR has been edited | Stage 9 now, Stage 10 after |
| 16 `_justify` | Inside `OrToolsFamily.solve` | Stage 7 |

`scratchpad_planning` catches only `SolveError`. `SolveError` derives from `Exception`
(`plan_solver.py:43`) and `RuleViolation` must not derive from it, so a failed check
cannot turn into a re-plan.

**`_return_infeasible(graph, solved)`:**

1. `_illegal_fixed_divisions(graph)`: for each node with a `"split"` pin, test
   `_split_option_is_legal(node.op, division_from_work_split(node, pin.value.split).splits)`
   (`allocator.py:2389`). Any failure raises `Unsupported(f"{node.name}: fixed split
   violates hard domain.")`, the message of `_legal_fixed_division`. The graph we would
   otherwise return unchanged carries that same division to codegen, so this keeps flow
   12's protection whichever stage's table pruned it. Skipped for a node with no `op`, as
   on a hand-built test graph.
2. `hints = [p for p in graph.pins if not p.internal]`. Empty raises `Unsupported`
   naming the family, status and pin sources. The parent says internal pins must never
   be infeasible, so with no user hint this is a bug, not a user problem (Deviations).
3. Otherwise emit the warning below and return an outcome with `decided=False`.
   `read_back` is not called, `graph` holds no decisions, `plan_allocation` does not
   commit, the IR is untouched and the post-passes do not run, since nothing is in LX for
   them to protect.

The compile then continues as if `LX_PLANNING=0` had run for this graph: every op keeps
the division `_distribute_work` committed, and every buffer is in HBM. That path is
supported (`passes.py:427`).

Warning text, through `warnings.warn` with a new `LxPlanningWarning(UserWarning)` in
`torch_spyre/ops/warning_types.py`, next to `FallbackWarning` and `RetileWarning`. A
user wrote these hints, so this goes to them, not to a logger they must enable:

```text
LX planning: no plan satisfies these user hints, so nothing was placed in LX and
this graph runs from HBM: <what>=<value> on <target> (<source>); ...
Remove or relax a hint to get LX residency.
```

`<target>` is `pin.target.name` for a node, and `<src> -> <dst>` for an edge or port.
Each hint is listed, however many there are. Naming the smallest conflicting set is
future work in the parent plan.

**`_fall_back(graph, fallback, settings, first)`:**

1. If `fallback is None`, log and raise `SolveError(first.solution.status.name)`.
2. Restore `graph.pins` from the copy taken in `_solve_with`. Greedy pins belong to the
   family whose `encode` added them, and the fallback must see the same graph.
3. `logger.warning` once for this graph, then `_solve_with(fallback, graph, settings,
   first.solution.status.name)`.
4. `INFEASIBLE` goes to `_return_infeasible`. Any other result with `ok` false raises
   `SolveError`, so the greedy re-plan is the last resort until Stage 13.

Stage 13 removes `SolveError`. Both raises then become `RuntimeError`, and an empty
`fallback_solver` raises `ValueError` in `select_families`.

### When a graph falls back

- **Who re-encodes:** the family named by `config.fallback_solver`. Its `encode` builds a
  new `Encoding` over the same graph, which holds no decisions because `read_back` never
  ran. The failed `Encoding` is discarded.
- **Time budget:** the same `settings`, so the worst case is the primary's budget
  (`cpsat_time_limit_seconds`, default 60 s) plus SA's encode and search. Stage 9 does not
  depend on SA honoring `time_limit`. No measurement exists on a graph where CP-SAT truly
  found no plan.
- **What the plan loses:** SA does not decide relayouts (`sa_cooptimizer.py:716`), and
  `sa-vs-cpsat-REPORT.md` puts that gap at 1.117x geomean on the zoo. The greedy re-plan
  has planner relayouts but fixed divisions. Neither dominates, and Gate 4 reports both.
- **What the user sees:** `logger.warning` on logger `scratchpad.optimizer`:

  ```text
  LX solve with <family> found no plan after <seconds> s (<status>); re-planning with
  <fallback>. Raise CPSAT_TIME_LIMIT_SECONDS to give it longer.
  ```

  This is deliberately louder than today's INFO: #5682 notes that a slow solve only shows
  up as a worse plan. `outcome.fell_back` and `first_status` record it, and #5682's
  Solve-status guard reads `last_outcome()`.

### `check_report` and `RuleViolation` in production

- On the new path, `check_report`, the full form of `check_decisions`, runs on every plan,
  from every family, primary or fallback, with no switch to disable it. It runs after
  `read_back` and before any IR edit, under the `lx:check` timing stage.
- Any violation, or any rule that raised while being judged, raises
  `RuleViolation([*report.violations, *report.errors])`. The compile fails with that
  exception, which names each rule and subject. Nothing retries it: a plan that breaks a
  rule means an encoding bug, and retrying with another family would hide it. The IR is
  untouched, and `_run_pass_loop`'s `finally` still expands the loop nodes
  (`passes.py:615-622`).
- On the old path, and on the greedy re-plan, Stage 3's reporting mode stays, under
  `lx_check_decisions`, and never fails a compile.
- `_stage_collective_outputs` runs inside the bridge, after the check. It gives a spilled
  matmul output a free LX gap and moves no chosen allocation (`allocator.py:3538`). The
  bridge copies those addresses onto the graph and calls Stage 3's `no_overlap(graph)`, as
  `_finish_plan_check` does on the old path, and raises `RuleViolation` on any result
  (Deviations). `residency` is not re-judged, since a staged output is exempt from the bars
  by design.

### Temporary commit bridge

New module `scratchpad/lx_allocator/optimizer_bridge.py`. It turns decisions on the graph into the
records the old tail consumes, then runs the old tail unmodified. It reads no menu index.
Every record carries exactly one division, which keeps the bridge independent of the
family's tables.

```python
def records_from_graph(graph: SpyreOptimizationGraph) -> list[CoreDivisionBuffer]: ...

class _CommitBridge(CoOptimizingAllocator):
    def __init__(self, size: int, post_optimization_passes, drain_plans,
                 *, divisions_committed: bool) -> None: ...
    def commit_graph(self, ir, graph) -> None: ...

def _commit_via_bridge(ir, graph, drain_plans, *, divisions_committed: bool) -> None: ...
```

`_commit_via_bridge` builds a `_CommitBridge` with `_context_switching_passes()` and calls
`commit_graph`. `_CommitBridge.commit_graph` is `CoOptimizingAllocator.plan_allocation`
from `_finalize_lx_relayout_allocation` onward, in the same order:

```python
allocation = records_from_graph(graph)
accepted = self._finalize_lx_relayout_allocation(allocation, ir)
self._post_solve(ir, allocation, accepted)
self._judge_staged_outputs(graph, allocation)   # no_overlap over late addresses
reasons = self._get_spill_reasons(ir, allocation)
self._push_allocation(ir, allocation, accepted)
self._log_lx_pinning(ir, reasons)
self._run_passes(self.post_optimization_passes, ir)
```

`__init__` passes `SaCoOptimizingSolver` as `layout_planning`, only because the
constructor asks it for two capability flags (`allocator.py:3120-3121`); no solver is
built and the tail reads neither flag. It sets `self._validated_drain_plans = drain_plans`,
the object computed once in `plan_allocation`, which is the single-object rule at
`allocator.py:3135-3138`.

How each record is made, in `graph.nodes` order, so the clones `_push_allocation` inserts
come in today's order (Stage 2 builds nodes in record order):

| record field | source |
|---|---|
| `name` | `node.name` |
| `size` | `node.size` |
| `uses` | `[]`. The tail never reads it, and `LifetimeBoundBuffer.__post_init__` allows it (`plan_solver.py:120-125`) |
| `address` | `node.offset * graph.target.alignment` if `node.resident`, else `None`. The graph holds alignment units, and the tail reads bytes |
| `core_divisions` | `[division_from_work_split(node, node.work_split)]`. A resident input takes one `CoreDivision` from its view, below. A spilled input takes `[]` |
| `chosen_division` | `0`, or `None` for a spilled input |
| `lx_view` | `node.prefetch_view` for a resident input, else `None` |
| `spill_reason` | `node.spill_reason` |
| `chosen_relayouts` | Fired `RELAYOUT` steps on the node's in-edges, below |

An LX prefetch has no `work_split`, but the tail still needs a division on its record.
`_stage_collective_outputs` abandons staging at any placed record with `chosen_division`
`None` (`allocator.py:3558-3563`), and otherwise takes the span from `per_core_size`. So a
resident input gets `CoreDivision(splits={Symbol(f"d{dim}"): f for dim, f in
view.work_slice_dims if f > 1})`, whose `output_partition` is the view's split product. No
reader of an input's record uses those keys (`chosen_tilings` and `planned_splits` look
ops up by name, and an input is not an op). `lx_view` is what `_push_allocation` reads
(`:2080`, `:2085`), but `commit_lx_views` overwrites it on every resident record (`:3030`),
so the view comes from the ownership judge on the live graph, as today.

For each `edge` in `graph.data_edges` and each step on it with `step.kind is
StepKind.RELAYOUT` (`read_back` leaves only fired steps):

```python
group = _intern_view_group(groups.setdefault(src.name, {}), step.to_view)  # :2869
candidate = RelayoutCandidate(
    parent=src.name, consumer=dst.name, source_division=0, consumer_division=0,
    group=group, source_view=step.source_view, destination_view=step.to_view,
    cost_ns=0.0, source_footprint_bytes=step.source_footprint_bytes,
    destination_footprint_bytes=step.destination_footprint_bytes)
address = step.offset * graph.target.alignment
records[dst.name].chosen_relayouts[src.name] = ChosenRelayout(candidate, address)
```

Then one `RelayoutCopyBuffer` per group, through the existing `build_relayout_copy`
(`plan_solver.py:856`) with `consumer_ticks` from `graph.step`, `address` as above and
`chosen_division=0`, appended after the nodes sorted by `(parent, group)` as
`_relayout_copy_buffers` does (`allocator.py:4638`, `:4490`). The copies must be in the
allocation: `_stage_collective_outputs` packs around every placed record, so omitting one
would let a stage land on a live copy, and `commit_lx_views` skips them by type.

The single-entry convention makes the index checks hold without change:
`_finalize_lx_relayout_allocation` compares 0 with 0, and `_assert_drain_plan_committed`
tests `(0, 0)` against `match_pairs` over the two one-entry lists, a real test of the
chosen pair.

Decisions the bridge ignores, and why:

- `DataEdge.in_place`: equal offsets already express it, and the tail reads no in-place flag.
- `COPY_OUT` steps: `apply_tilings` re-derives cuts and copy ops itself
  (`coarse_tiling.py:667`), as it does for today's solver output.
- Pinned `RESTICKIFY` steps: they already exist in the IR. The `RestickifyNode` a step
  realizes is an ordinary node and gets a record.
- A `RESTICKIFY` step with `pinned` false raises `NotImplementedError`. None exists before
  Stage 11, and it must not pass silently.

**Removal in Stage 10.** Delete `optimizer_bridge.py`, `_commit_via_bridge` and
`tests/inductor/test_optimizer_bridge.py`, and call Stage 10's `commit(graph)` at the one
call site in `plan_allocation`. Stage 10 inherits this list, all of which the bridge runs
today and which `commit` must replace: tilings, footprint check, divisions, LX views,
collective output staging with its `no_overlap` check, input hoisting, drains,
`materialize_lx_relayouts` last, spill-reason logging. The post passes move into
`commit`, which runs `_context_switching_passes()`. `optimize` and the failure flow do not
change, and `plan_allocation` changes only at that call. `commit` takes only the graph, so
Stage 10 decides how the IR and the drain plans reach it.

### Committed mode

`co_optimizing_lx_planning=False` with a family selected means every division is pinned
to what `_distribute_work` committed. It is the same `plan_allocation`, not a second
path. It passes `divisions_committed=True`, `build_graph` sets
`graph.divisions_committed`, and Stage 2 is the sole pin writer: every running node with no
pin gets a `"split"` pin with source `"committed"`. LX prefetches stay unpinned, because a
prefetch's view follows its consumers.

Three things differ from the joint path, each needed for the baseline to mean anything:

1. **Objective.** `plan_allocation` builds on a target derived with
   `replace(target, cost_model=SpyreCostModel(traffic_only=True))`, so `encode` reads
   that cost model from `graph.target`. `priced_objective` returns `None`, and
   `ladder` is `residency` alone, priced as `plan_layout` prices it: `spill_cost` with
   `boundary_known=False` and sizes from `traffic_sizes`, with no cut traffic. This is not
   the ladder flows 5 and 6 degrade to, which keeps `cut_traffic`. Today's placement-only
   CP-SAT minimizes exactly this, so a priced objective could not pass a gate on traffic
   except by luck (Deviations).
2. **Relayouts.** Pinned out, as whenever `priced_objective` returns `None`
   (`ilp_solver_ortools.py:1450-1455`), which also matches the baselines (see Surprises).
3. **Residency branches.** `graph.divisions_committed` selects each rule's
   `division_is_fixed=True` branch: the loop-carry exception (`allocator.py:832`, and the
   rule text at `scratchpad/residency_rules.py:120-126`), the restickify barrier
   (`:847-855`), the core-division mismatch (`:932-936`) and the prefix gather
   (`:896-915`). `validated_drain_plans` returns `{}` (`allocator.py:464`). These branches
   stay through Stage 13.

The bridge also differs. With `divisions_committed`, `_post_solve` runs only
`commit_lx_views` and `log_solver_decisions`. It skips `apply_chosen_tilings` (no solver
tilings), `_stage_collective_outputs` (the placement path does not run it) and, most
importantly, `commit_divisions`: `commit_iteration_space_ownership`
(`pass_utils.py:2082`) rebuilds ownership in canonical order, which would discard the
owner order `anchor_lx_relayout_ownership` committed.

**Precondition, before Stage 7.** This design and its bar in Gate 2 rest on two unmeasured
things: how often committed mode lands on another tied optimum, and how the SA committed
arm compares. Probe P3, which has not been run, settles both before the design is fixed.
On today's main, monkeypatch `CoOptimizingAllocator._division_map` to pin every op at its
committed division, with the ladder and copies pinned out. Compare with
`ScratchpadAllocator` + `CpSatLayoutSolver` on the objective and on committed traffic.
Repeat for SA joint with pinned menus against `SimulatedAnnealingLayoutSolver`, over five
seeds.

### Cost dump

With `config.dump_cost_expr_file` set, the new path does not call `cost_expr_record`:
`solved_bindings` binds `division_*` to the record's index (`plan_solver.py:557`), which
is 0 on bridge records, so `RelayoutCharge` would evaluate wrongly. It writes a reduced
line through `emit_json_line` (family, status, objective, solve seconds), so a reader can
tell the file is not stale. Stage 10 restores the full record from decoded graph values.
`SPYRE_DUMP_COST=1` is unaffected, since it reads the final IR.

### Room for a debug greedy family

Nothing here rules it out, and `test_debug_greedy_shaped_family_runs_the_normal_path`
keeps it that way. It needs only a `FAMILIES` entry and a `layout_solver` value, and can
be `fallback_solver` too. Neither `plan_allocation`, `optimize` nor the bridge names a
family, and `optimize` reads only `solution.status` and `solution.ok`. A greedy family's
`encode` may do almost nothing. A plan with committed divisions and nothing resident
passes the check by construction.

### `#5695` landing first

That PR adds `solve_decides_work_division()`, which must agree with `select_allocator()`
and skips `_distribute_work` for solver-decided ops. Stage 9 needs committed divisions
for pins, committed mode, an infeasible graph left unchanged, and the greedy re-plan. So
in this PR the predicate returns false whenever `family_key(config)` is not `None`,
which keeps `_distribute_work` running on the new path. Rebasing also keeps the
`divide_deferred_op` call #5695 adds to the except body, inside `_greedy_replan`. Stage 11
turns the predicate true on a family. Until then the new path pays for `_distribute_work`
where the old one does not, and Gate 3's measure counts that.

### CI wiring

A small PR lands ahead of this stage's main PR, so that Gate 1 runs by dispatch. It does
nothing until a caller sets its inputs.

- `.github/workflows/_test_matrix.yaml` gains optional string inputs `lx_route_to_families`
  and `layout_solver`, default `''`. The jobs `test`, `test_multi_spyre`, `test_retry` and
  `test_multi_spyre_retry` get one step after "Derive report paths":

  ```yaml
  - name: Export LX planner selection
    env:
      ROUTE: ${{ inputs.lx_route_to_families }}
      SOLVER: ${{ inputs.layout_solver }}
    run: |
      if [ -n "$ROUTE" ]; then echo "SPYRE_LX_ROUTE_TO_FAMILIES=$ROUTE" >> "$GITHUB_ENV"; fi
      if [ -n "$SOLVER" ]; then echo "LAYOUT_SOLVER=$SOLVER" >> "$GITHUB_ENV"; fi
  ```

  Each stays unset when empty: an empty `LAYOUT_SOLVER` raises `ValueError` in
  `select_allocator` (`allocator.py:5410-5414`), so a plain `env:` entry would not do.
- `.github/workflows/lx_planner_compare.yaml` is new and dispatch-only. It calls
  `_test_matrix.yaml` with inputs `ref`, `test_type`, `lx_route_to_families` (default `'1'`)
  and `layout_solver`, and passes no ClickHouse secrets. Stage 12 adds the cron.

## Changes

| file | change |
|---|---|
| `.github/workflows/_test_matrix.yaml` | Separate PR, first: the two inputs and the export step |
| `.github/workflows/lx_planner_compare.yaml` | Separate PR, first: new, dispatch-only |
| `torch_spyre/_inductor/scratchpad/lx_allocator/optimizer.py` | New: `plan_allocation`, `optimize`, `PlanOutcome`, `last_outcome`, `family_key`, `select_families`, `optimizer_pass_sources` |
| `torch_spyre/_inductor/scratchpad/lx_allocator/optimizer_bridge.py` | New, temporary: `records_from_graph`, `_CommitBridge`, `_commit_via_bridge` |
| `torch_spyre/_inductor/scratchpad/allocator.py` | `scratchpad_planning` routes; `_greedy_replan`, `_context_switching_passes` extracted; docstring at `:5352` |
| `torch_spyre/_inductor/passes.py` | `CustomPostFusionPasses.uuid`; `_runs` lists `plan_allocation` and `optimizer_pass_sources()`; comment at `:429` updated |
| `torch_spyre/_inductor/config.py` | `fallback_solver`, `lx_route_to_families` |
| `torch_spyre/ops/warning_types.py` | `LxPlanningWarning` |
| `tests/inductor/test_spyre_optimizer.py` | New, device-free |
| `tests/inductor/test_optimizer_bridge.py` | New, device-free, temporary |
| `tests/inductor/test_spyre_optimizer_e2e.py` | New, device suite |
| `tests/inductor/lx_traffic.py` | New: `committed_hbm_traffic` |
| `tests/configs/torch_spyre_tests/inductor/test_{spyre_optimizer,optimizer_bridge,spyre_optimizer_e2e}_config.yaml` | New shards, per AGENTS.md "Tests must be selected" |
| `tests/inductor/test_scratchpad_use.py`, `test_solver_auto_coarse_tiling.py` and others | Old-path spies patch `lx_route_to_families=False` |
| `tests/inductor/lx_equivalence.py` | `hbm_traffic_bytes` and `outcome` in the committed-plan record, `SCHEMA_VERSION` bump, `decisions_from_graph`, `install` rows that open and close family solves, SA baseline patch |
| `tests/inductor/lx_corpus.py` | `ortools` and `sa` cells, the arm comparison |
| `docs/source/compiler/scratchpad_planning.md`, `co_optimizer.md` | Optimizer path and failure flow, and a link to it |
| `AGENTS.md` | `FALLBACK_SOLVER`, `SPYRE_LX_ROUTE_TO_FAMILIES` in the env table |

The CI wiring is its own PR, merged first. Then the main PR's commits, each leaving the
suite green. Nothing calls the new code until commit 4.

1. **Add the optimizer's config and `LxPlanningWarning`.** Inert. Cache-key tests.
2. **Add `optimize` and the failure flow.** Driven in tests by fake families and a
   hand-built Stage 1 graph; `optimize` never commits.
3. **Add the commit bridge.** Tested on mock graphs in the style of
   `tests/inductor/test_work_division.py:3590`.
4. **Add `plan_allocation` and route `scratchpad_planning` to it.** Adds `family_key`,
   `select_families`, `_greedy_replan`, `_context_switching_passes`. An empty Stage 0 diff
   proves the default.
5. **Put the planner's sources in the FX key.**
6. **Add committed mode.** The bridge branch, `divisions_committed` and the derived
   target `plan_allocation` builds.
7. **Extend the harness.** Traffic, record fields with a `SCHEMA_VERSION` bump,
   `decisions_from_graph`, cells, the SA baseline patch and the arm comparison. `install`
   gains three rows, all in `OPTIONAL_TARGETS`: `SAFamily.solve` calls
   `begin_solve(family="sa", solver="SAFamily")`; `OrToolsFamily.read_back` and
   `SAFamily.read_back` then call `end_solve(decisions_from_graph(graph), ...)`; and
   `_solve_with` closes the solve with its status when `not solution.ok`.
   Without these an `ortools` or `sa` cell records nothing and `diff` compares nothing.
8. **Pin old-path spies with `lx_route_to_families=False`,** from the first run on each arm.
9. **Add the device e2e suite.**
10. **Document the optimizer path.**

## Interfaces

### Consumes

| symbol | signature | from Stage | used for |
|---|---|---|---|
| Harness, corpus, regression guards | `install`, `ForcedFailure`, `Recorder`, `SCHEMA_VERSION`, `Cell`, `FIXED_GRAPHS`, `lx_zoo.MODELS`, `CARD_ONLY`; tiers `intree`, `fake`, `zoo`, `matrix`, `unit-*`, `failure`; `lx_gate.sh` | 0 | Every comparison. Extended, as stated in Changes |
| `install` patching `SolveSettings.from_config` | `SolveSettings(DEFAULT_DTIME, workers=1, deterministic=True)` | 0, 7 | Deterministic arms |
| Family recording | `OrToolsFamily.solve` harness rows, per ladder stage; `buffer_decision(decision, alignment) -> BufferDecision` | 7 | `decisions_from_graph`, forced failures |
| `SpyreOptimizationGraph` | `.nodes`, `.data_edges`, `.pins`, `.target`, `.divisions_committed`, `.pins_on(target, what)`, `.step(node)` | 1, 2 | Everywhere |
| `Target`, `select_target` | `Target` frozen, with `.alignment` and `.cost_model`; `select_target(config) -> Target` | 1, 2, 4 | `plan_allocation`, bridge addresses |
| `OpNode` | `.name` equal to `op.get_name()`; `.op`; `.size`; `.work_split`; `.resident`; `.offset` in alignment units | 1 | Bridge |
| `OpNode.spill_reason`, `InputNode.prefetch_view` | `str \| None`, `PerCoreView \| None`, written by `read_back` | 1, filled by 7, 8 | Bridge |
| `DataEdge`, `Step`, `StepKind` | `.src: OutPort`, `.dst: InPort`, `.steps`, ports have `.node`; `Step.kind`, `.offset` in alignment units, `.pinned`; `Step(kind=StepKind.RESTICKIFY, pinned=True, realized_by=node)` | 1 | Bridge |
| `RELAYOUT` step fields | `.to_view`, `.source_view`, `.source_footprint_bytes`, `.destination_footprint_bytes`; the footprints stay in bytes | 1, filled by 7, 8 | Plans and copies in the bridge |
| `Pin`, `WorkSplitPin` | `.target`, `.what`, `.value`, `.source`, `.internal`; a `"split"` pin holds `WorkSplitPin(split, reason, seeds)`; `"committed"` is internal, `"hint"` is not | 1, 2 | Warning, `_illegal_fixed_divisions` |
| `division_from_work_split` | `(node, split) -> CoreDivision`, keeping symbol keys, `reduction_syms` and tile levels in order | 2 | Bridge, `_illegal_fixed_divisions` |
| `build_graph` | `build_graph(ir, target, *, divisions_committed=False, drain_plans=None, lx_relayout_plans=())`. Makes nodes through `target.ops` and stores `graph.target`. Reads the IR only. Sole writer of committed pins, and sets `graph.divisions_committed` | 2 | `plan_allocation` |
| `check_report`, `check_decisions`, `CheckReport` | `check_report(graph) -> CheckReport`, the full form of `check_decisions(graph)`; `.violations`, `.errors` | 3 | `optimize` |
| `no_overlap` | `(graph) -> list[Violation]` | 3 | Bridge, staged outputs |
| `RuleViolation` | `RuleViolation(violations)`, subclass of `Exception`, not of `SolveError` | 3 | `optimize` |
| `coverage`, `Coverage` | `(family, target) -> Coverage(gaps, unused)`; only `gaps` refuses | 3 | `plan_allocation`, at config time |
| `decisions`, `NodeDecision` | `(graph) -> dict[str, NodeDecision]` | 3 | `decisions_from_graph` |
| `decode_todays_plan` | `(graph, allocation, accepted_relayouts) -> list[Mismatch]` | 3 | Baseline arms |
| `SpyreCostModel` | `(params=ALLOCATOR_COST_PARAMS, *, traffic_only=False)`; `select_target` attaches `COST_MODELS.get(target.name)`, a `SpyreCostModel()` on `SPYRE`. With `traffic_only=True`, `priced_objective` returns `None` and `ladder` is `residency` alone, with no `cut_traffic`, as `plan_layout` prices it | 4 | Committed mode's derived target |
| Objective degrade | `encode` handles an unusable objective internally, as flows 5, 6: ladder, relayouts pinned out, re-raise when `_cpsat_warn_on_cost_expr` is false | 4, 6, 7 | `optimize` stays one path |
| `Family`, `Encoding` | `encode(graph) -> Encoding`, reading `graph.target.cost_model`; `solve(encoding, settings) -> Solution`; `read_back(solution, graph)`. `encode` does not mutate the IR, and added pins go through `graph.pin`. `encode` raises `Unsupported` for no legal division, as `allocator.py:3801` | 5, 7 | `optimize`, flow 13 |

| `FAMILIES`, `from_config` | `FAMILIES: Mapping[str, type[Family]]`, keys `"ortools"`, `"sa"`; `FAMILIES[key].from_config(config)` builds one; `OrToolsFamily` raises `ImportError` without ortools | 7, 8 | Selection |
| `SolveStatus`, `Solution` | `.status: SolveStatus` (`OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `NO_PLAN`, `NO_PACKING`, `INVALID`); `.ok: bool`, holds a plan; `.objective` | 7 | `optimize` |
| Solve contract | A later ladder stage that fails gives `NO_PLAN` and loses the earlier plan. A roof-induced infeasibility in packing mode is `NO_PACKING`. Only a model proven infeasible as posted reports `INFEASIBLE` | 7 | The flow table |
| `SolveSettings` | `SolveSettings(time_limit, workers, deterministic)`; `from_config(config)`: `cpsat_time_limit_seconds`, `get_cpu_count()` workers, at least 2 interleaved under `deterministic` | 7 | `plan_allocation` |
| `SAFamily` | `SAFamily.from_config(config)` is `SAFamily(seed=0)`. `status` is `SolveStatus.FEASIBLE`, never `ok=False` from search. Seed trouble is handled in seeding | 8 | Fallback, SA arm |
| SA level 4 | Zero check reports at default and `AUTO_COARSE_TILING=1`; the tiling filter is a listed rule difference | 8 | Entry condition |

### Provides

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `plan_allocation` | `(ir, config) -> SpyreOptimizationGraph`, in `lx_allocator/optimizer.py`: `select_families`, `select_target`, `coverage`, `build_graph`, `optimize`, then commit when `outcome.decided` | 10, 11, 12, 13 | final; Stage 10 replaces the bridge call with `commit(graph)` |
| `optimize` | `(graph, family, fallback, settings) -> PlanOutcome`; never touches the IR | 10, 11 | final |
| `_context_switching_passes` | `() -> list[ScratchpadOptimizationPass]`, run by the bridge, and by `commit` from Stage 10 | 10, 13 | final |
| `PlanOutcome`, `last_outcome` | Fields above; `.status` holds `SolveStatus.name`; `last_outcome() -> PlanOutcome \| None`, the latest `plan_allocation` in the process | 11, 12; recorded by the harness | final |
| `_fall_back`, `_return_infeasible` | `(graph, fallback, settings, first) -> _Solved`; `(graph, solved) -> PlanOutcome` | 13 | final; Stage 13 changes `_fall_back`'s raises |
| `solve_decides_work_division()` rule | false wherever `family_key(config)` is not `None` | 11, 13 | final; Stage 11 turns it true on a held plan |
| `family_key` | `(config) -> str \| None`, in `lx_allocator/optimizer.py`; `None` sends the compile to the old path | 11, 12, 13 | final; `None` return removed in Stage 13 |
| `select_families` | `(config) -> tuple[Family, Family \| None]` from `FAMILIES[...].from_config(config)`; promotes the fallback when ortools is absent | 11, 12, 13 | final |
| Committed mode | `co_optimizing_lx_planning` off gives `divisions_committed=True` and a target derived with `SpyreCostModel(traffic_only=True)` | 10, 12, 13 | final |
| `optimizer_pass_sources` | `() -> tuple[Any, ...]`, planner sources for `_runs` | 10-13 | final; Stages 10 and 11 add `commit.py` and `sticks.py`, Stage 13 drops the modules it deletes |
| `config.fallback_solver` | `str`, env `FALLBACK_SOLVER`, default `""`, a `FAMILIES` key or empty | 12, 13 | final; Stage 12 flips it to `"sa"`, and non-empty is a Stage 13 entry condition |
| `config.lx_route_to_families` | `bool`, env `SPYRE_LX_ROUTE_TO_FAMILIES` | 12, 13 | temporary, removed in Stage 13 |
| `LxPlanningWarning` | `UserWarning` subclass | 12 | final |
| `_greedy_replan` | `(graph, error, post_passes) -> None` | 13 | temporary, removed in Stage 13 |
| `records_from_graph`, `_CommitBridge`, `_commit_via_bridge` | Above | 10 | temporary, removed in Stage 10 |
| `committed_hbm_traffic` | `tests/inductor/lx_traffic.py`, `(graph: GraphLowering) -> int` | 10, 11, 12 | final |
| `hbm_traffic_bytes`, `outcome` | Committed-plan record fields, with a `SCHEMA_VERSION` bump | 10, 11, 12 | final |
| `decisions_from_graph` | `(graph) -> dict[str, BufferDecision]`, in `lx_equivalence.py`, over Stage 3's `decisions` and Stage 7's `buffer_decision` | 10, 12 | final |
| `ortools` and `sa` cells | `Cell(layout_solver="ortools")`, `Cell(layout_solver="sa")` in `lx_corpus.py` | 10, 12, 13 | final |
| SA baseline patch | `LxContextSwitchingPass` in `select_allocator`'s SA branch, test-side in `lx_equivalence.py` | 12 | temporary, removed in Stage 13 |
| Gate 3 protocol, Gate 4 report | Parity and gates | 12 | final |
| CI wiring | `_test_matrix.yaml` inputs `lx_route_to_families` and `layout_solver`, exporting `SPYRE_LX_ROUTE_TO_FAMILIES` and `LAYOUT_SOLVER`, unset when empty; dispatch-only `lx_planner_compare.yaml` | 12, 13 | temporary; Stage 13 removes `lx_route_to_families`, and the rest goes with the lane, whichever of Stage 13 and 12c lands last |
| Timing stages `lx:encode`, `lx:solve`, `lx:read_back`, `lx:check` | `timing_recorder.stage` names, with `family` | 12 | final |

## Tests

### `tests/inductor/test_spyre_optimizer.py` (device-free)

A fake family returns scripted `Solution`s and records calls. Graphs are hand-built with
Stage 1 types. Tests that need commit drive `plan_allocation` with the bridge patched.

| test | asserts |
|---|---|
| `test_plan_runs_encode_solve_read_back_check_commit_in_order` | Call order; bridge called once |
| `test_feasible_at_time_limit_takes_the_normal_path` | `FEASIBLE`: `read_back`, check and commit run; the fallback family never encodes; no warning |
| `test_infeasible_warns_with_every_user_hint` | `LxPlanningWarning` text contains each hint's target and value |
| `test_infeasible_decodes_and_commits_nothing` | `read_back` not called, nodes hold no decisions, `outcome.decided` false, bridge not called, post-passes not run |
| `test_infeasible_does_not_try_the_fallback` | Fallback family untouched |
| `test_infeasible_without_user_hints_is_unsupported` | Raises `Unsupported` naming pin sources |
| `test_infeasible_with_illegal_fixed_division_keeps_the_message` | `Unsupported`, "fixed split violates hard domain", read from `pin.value.split` |
| `test_no_plan_reencodes_with_the_fallback[NO_PLAN, NO_PACKING]` | The fallback family encodes the same graph object and solves with the same `settings` |
| `test_fallback_starts_from_the_original_pins` | A pin the primary's `encode` added is gone when the fallback encodes |
| `test_fallback_plan_takes_the_normal_path` | Read back through the fallback family, checked, committed; `outcome.fell_back`, and `first_status` as a `SolveStatus.name` |
| `test_no_plan_left_raises_solve_error[fallback fails, no fallback]` | `SolveError` |
| `test_fallback_infeasible_warns` | Goes through `_return_infeasible` |
| `test_invalid_raises` | `INVALID`: `RuntimeError`, no fallback |

| `test_failed_check_raises_rule_violation_and_commits_nothing` | Bridge not called, for primary and fallback plans |
| `test_rule_error_raises_rule_violation` | A rule that raised, in `report.errors`, fails the check |
| `test_rule_violation_is_not_caught_by_scratchpad_planning` | Propagates, greedy not built |
| `test_solve_error_reaches_greedy_replan` | `scratchpad_planning` with a `plan_allocation` that raises `SolveError`; `_context_switching_passes()` passed through, as `test_fallback_keeps_lx_context_switching` does |
| `test_coverage_gap_raises_before_build[primary, fallback]` | `plan_allocation` raises `ValueError` naming family and flag; `build_graph` not called |
| `test_empty_graph_does_not_solve` | Nothing encoded |
| `test_debug_greedy_shaped_family_runs_the_normal_path` | A registered family whose `encode` returns a bare `Encoding` and whose `Solution` is `FEASIBLE` with nothing resident goes through check and commit, as primary and as fallback |
| `test_family_key_routes_family_keys` | Replaces Stage 7's `test_select_allocator_rejects_ortools_until_routed`: `"ortools"` reaches `plan_allocation` |
| `test_family_key_is_none_for_old_values` | Every value of `layout_solver` not in `FAMILIES` |
| `test_routing_maps_only_the_two_legacy_values` | `greedy`, `firstfit`, `bestfit` stay on the old path |
| `test_family_is_built_from_config` | `FAMILIES[key].from_config(config)`: `auto_coarse_tiling` reaches the family |
| `test_ortools_absent_promotes_the_fallback` | `ImportError`: fallback becomes primary, warning once |
| `test_ortools_absent_without_fallback_reaches_greedy` | `SolveError` from `select_families` reaches `_greedy_replan`, with the context-switching pass |
| `test_unknown_fallback_solver_is_rejected` | `ValueError` |

| `test_context_switching_runs_for_every_family` | Both keys carry `LxContextSwitchingPass` |
| `test_optimizer_config_is_in_the_cache_key` | Both settings change `save_config_portable`'s output |
| `test_pre_scheduling_sources_are_in_the_cache_key` | Changing the value `CustomPreSchedulingPasses.uuid()` returns changes `CustomPostFusionPasses.uuid()` |
| `test_every_planner_module_is_hashed` | Every `.py` under `lx_allocator/`, `ilp_solver_ortools.py` and `sa_cooptimizer.py` are among the files `_uuid` hashes |
| `test_committed_mode_pins_divisions_and_prices_traffic` | `divisions_committed=True` reaches `build_graph`, and `graph.target.cost_model` is `SpyreCostModel(traffic_only=True)` |
| `test_last_outcome_is_the_latest_call` | `last_outcome()` after two `plan_allocation` calls is the second's outcome |

### `tests/inductor/test_optimizer_bridge.py` (device-free, removed in Stage 10)

| test | asserts |
|---|---|
| `test_one_record_per_node_in_graph_order` | Names and order, a `RestickifyNode` included |
| `test_record_division_round_trips` | `per_core_size(0)`, `output_splits`, tiling levels in order |
| `test_record_addresses_are_bytes` | `address` and the relayout address are `offset * alignment` |
| `test_resident_input_carries_view_and_partition` | `lx_view`, `chosen_division == 0`, `output_partition` is the view's split product |
| `test_resident_input_does_not_stop_staging` | Real `_stage_collective_outputs` still stages an output with a resident input in the allocation |
| `test_spilled_input_has_no_division` | `chosen_division is None` |
| `test_fired_relayouts_fold_into_one_plan` | Two consumers, one view: one `LXRelayoutPlan`, one copy |
| `test_two_slot_spellings_are_one_group` | `same_partition` views intern together |
| `test_copy_is_seen_by_collective_staging` | A stage is not placed over a live copy |
| `test_staged_output_overlap_raises_rule_violation` | A staged address overlapping a placed node fails `no_overlap` |
| `test_finalize_index_checks_hold_on_single_entry_records` | Real `_finalize_lx_relayout_allocation`, no assertion |
| `test_drain_check_uses_the_chosen_pair` | `_assert_drain_plan_committed` passes on a match, raises on a mismatch |
| `test_committed_bridge_keeps_anchored_ownership` | `iteration_space_ownership` object unchanged |
| `test_unpinned_restickify_is_rejected` | `NotImplementedError` |

| `test_tail_reads_no_uses` | Records with `uses=[]` survive the tail on a fixture graph |

### `tests/inductor/test_spyre_optimizer_e2e.py` (device suite)

Parameterized over each family and over `co_optimizing_lx_planning` on and off, on the
softmax and MLP cases of `test_scratchpad_use.py`:

- `test_family_compiles_and_matches_cpu`: numerics match, `assert_uses_lx`.
- `test_forced_timeout_falls_back`: `fallback_solver="sa"`; `_zero_solver_timeout`
  patches `CpSolver.Solve`, as `tests/inductor/test_scratchpad_use.py:1477` does; numerics
  match, HBM traffic falls against `lx_planning=False`, `outcome.fell_back`, warning logged.
- `test_forced_timeout_without_fallback_uses_greedy`: the default `fallback_solver=""`;
  greedy called, as `:1495-1509` counts it.
- `test_ortools_absent_uses_the_fallback`: `fallback_solver="sa"`, as `:1411-1422`,
  without `allow_exhaustive_search`.
- `test_fallback_kernel_is_bracketed_on_sa`: an opaque call between a write and a read of
  an LX-resident buffer, using the op in `tests/inductor/test_lx_context_switching.py:85`,
  gets dump and restore clones on the SA family.

### Existing suites

Unit, serial, on every commit: `test_scratchpad_solver`, `test_solver_relayout_decision`,
`test_cooptimization_types`, `test_cost_expr_dump`, `test_restickify_cost_model`,
`test_sa_cooptimizer`, `test_coarse_tiling`, `test_work_division*`,
`test_scratchpad_patterns`, `test_perm_layout_solver`, `test_scratchpad_use`,
`test_solver_auto_coarse_tiling`, and the graph-layer tests. Run `make check-all-configs`.

Device suites: `test_scratchpad_use`, `test_work_division`, `test_solver_relayout_e2e`,
`test_solver_auto_coarse_tiling`, `test_lx_planning_suite`, the `test_inductor_ops_lx_*`
shards, and the new e2e file.

**What proves the claim.** `test_spyre_optimizer.py`, one test per row of the flow table,
plus the forced-timeout e2e tests, prove the failure flow. An empty Stage 0 diff on commit
4 proves the default path. The two source-key tests prove the FX key. The gate below proves
the Phase 2 exit.

## Parity and gates

The parent's gate: "**Phase 2 exit**, with each new family selected in turn: the unit and
device suites pass; HBM traffic is no worse than today's on the suites and the zoo:
`CoOptimizingAllocator` for OR-Tools, `SaCoOptimizingSolver` for SA, including committed
mode (every division pinned); planning time on the zoo stays within run-to-run noise of
today's."

### Tiers and entry conditions

Every gate runs HEAD's harness over the base tree, through `lx_gate.sh`.

- **Commit level:** `intree` and the fixed set.
- **PR tip:** all tiers, including `matrix`, with `zoo` once on the card as calibration.
  Numeric device suites stay on the card. Fake inputs use default layouts, so `fake` proves
  parity, not production plans. A model `fake` cannot reach stays card-only and is listed.
  The PR reports hits per coverage feature, including the restickify features.
- **Entry conditions:** Stage 8's level 4 reports are zero at default and at
  `AUTO_COARSE_TILING=1`, because any report is fatal on this path. P3 has run (Committed
  mode).

### Default path

Stage 0 harness, parent of this PR against this PR, default config and every
`layout_solver` × `co_optimizing_lx_planning` combination of the old values. Empty diff in
proto, decisions and committed plans. Any difference blocks.

### Arms

Old and new paths live in one tree behind config, so every comparison is between arms of
one build, and deeptools differences cannot enter.

| arm | env for the new path | baseline env | today's classes |
|---|---|---|---|
| OR-Tools, joint | `LAYOUT_SOLVER=ortools` | `LAYOUT_SOLVER=cpsat` | `CoOptimizingAllocator` + `CpSatLayoutSolver` |
| OR-Tools, joint, tiling | same, `AUTO_COARSE_TILING=1` | same | same |
| OR-Tools, committed | same, `CO_OPTIMIZING_LX_PLANNING=0` | same | `ScratchpadAllocator` + `CpSatLayoutSolver` |
| SA, joint | `LAYOUT_SOLVER=sa` | `LAYOUT_SOLVER=simulated_annealing` | `CoOptimizingAllocator` + `SaCoOptimizingSolver` |
| SA, joint, tiling | same, `AUTO_COARSE_TILING=1` | same | same |
| SA, committed | same, `CO_OPTIMIZING_LX_PLANNING=0` | same | `ScratchpadAllocator` + `SimulatedAnnealingLayoutSolver` |

Joint SA baselines are run with `LxContextSwitchingPass` patched into the SA branch of
`select_allocator` (Deviations).

### Gate 1: suites

- **Device access:** the Spyre card, by dispatch of `lx_planner_compare.yaml`, one job per
  config. Total run time on the card pool is unmeasured.
- **How selected:** input `lx_route_to_families=1` for the OR-Tools arm, plus
  `layout_solver=simulated_annealing` for the SA arm, so that each test keeps the solver
  flavor it asks for. Setting `LAYOUT_SOLVER` alone would miss the sites that patch it (28
  to `cpsat`, 7 to `simulated_annealing`, by grep over `tests/`).
- **What runs:** the unit list and device list above, for both joint arms, tiling off.
  `test_scratchpad_use`, `test_work_division` and `test_lx_planning_suite` also run
  with `CO_OPTIMIZING_LX_PLANNING=0`, by hand on the card, since the dispatch exports only
  the two values above. Tests that patch tiling or co-optimization themselves cover the rest.
- **How judged:** JUnit from the arm against JUnit from a dispatch with both inputs empty,
  on the same commit, test id by test id. Pass means every test has the same outcome,
  including xfail and skip. A strict XPASS is a difference.
- **Exclusions:** a test that spies on an old class patches `lx_route_to_families=False` in
  commit 8. A grep finds 10 such `patch.object` sites in `test_scratchpad_use.py` and 6 in
  `test_solver_auto_coarse_tiling.py`. The full list goes in the PR, and each exclusion
  needs an equivalent test on the new path or a line saying why none applies.

### Gate 2: HBM traffic

**Metric.** `committed_hbm_traffic(graph)` runs at the end of the pre-scheduling pass list
on the final `GraphLowering`, so both arms are judged by the same code from what was
actually committed, not by anything a solver reports:

```python
total = 0
for op in all_ops(graph.operations):
    trips = math.prod(concrete loop_info.loop_count, or 1 when untiled)
    for name in {written buffers of op} | {distinct read names of op}:
        if "lx" not in layout_of(name).allocation:
            total += trips * math.prod(device_size[:-1]) * 128
```

The byte size is the one `compile_and_collect_mem_usage` uses
(`tests/inductor/test_scratchpad_use.py:315`). `measure_hbm_transfers` counts each HBM
buffer once however often it is read, so it is not reused. Context-switch dumps, copy-outs
and LX prefetches are charged. Symbolic trip counts count as 1 and are flagged. The harness
stores the value as `hbm_traffic_bytes` in the committed-plan record.

**Corpus.** Suites: the `unit-*` tiers. Zoo: the `fake` tier, every model of
`lx_zoo.MODELS` not in `CARD_ONLY`. Both need a live `GraphLowering`, which is why captures
cannot serve here. Granite is reported if available and does not gate.

**Settings.** Stage 0's deterministic settings on both sides: `install` patches
`SolveSettings.from_config`, so OR-Tools gets one worker, seed 0 and the same deterministic
budget, with no wall-clock limit. SA uses its fixed seed, and seeds 0 to 4, set test-side,
on the committed arm.

**Pass rule.** Records pair by (test id or model, solve index). An unpaired solve fails.

| arm | pass |
|---|---|
| Joint, both families, with and without tiling | Identical plans (clones, relayouts, addresses, divisions). A non-identical plan is a failure even if traffic falls. On SA at tiling, a difference from Stage 8's listed tiling filter goes in the PR instead |
| OR-Tools committed | Per solve, the traffic objective is no worse when both sides are `OPTIMAL`. `hbm_traffic_bytes` against baseline: up to 1.001 ok, up to 1.04 with a reason, above that fail. Geomean at most 1.000 |
| SA committed | Per corpus, total `hbm_traffic_bytes`, median over five seeds, no higher than the baseline's. A changed plan is accepted as a listed plan change at Stage 13 C4 |

A failure is fixed, or justified by rule in the PR, as the parent's Stages preamble
requires, and a justified one needs the owner's sign-off. A `status` other than `OPTIMAL`
on either side is listed, because it means the comparison ran into a budget. The bands are
Stage 12's, so Phase 2 exit and the flip read one number.

**Report, not gating.** Predicted total from `SPYRE_DUMP_COST=1`
(`last_cost_report.total_us`), read from the final IR for both arms, with any increase over
0.1% listed, the threshold in `sa-vs-cpsat-REPORT.md`.

**Expected on committed arms.** Neither is measured. `#5676`'s 80/80 "same plans" result
predates its priced pinned menus and does not carry over. SA is the weakest arm: the
baseline is a different annealer (5,000-step clamp, `sa_cooptimizer.py:90-93`), and the
family with pinned divisions runs only reorder moves, at up to 50,000 steps. If it loses,
the fix belongs in `SAFamily.solve`'s budget under pinned divisions, and this gate blocks
until then.

### Gate 3: planning time

The parent's Verification says to replay captures. That cannot time `build_graph`,
`check_report` or the bridge, so this gate uses live `fake` compiles (Deviations).

- **Run:** on the zoo, in one process per model, interleaved `A B B A` over
  `torch.compiler.reset()`, three rounds, caches off, `TORCH_SPYRE_FRONTEND_ONLY=1`,
  `TORCH_SPYRE_TIMING=1`. Separate processes vary by up to 25% per model (feasibility
  report), which is why the arms share a process.
- **Measure:** `pass_loop_ns`, the `inclusive_ns` of
  `stage:CustomPreSchedulingPasses:pass_loop` (`passes.py:614`), plus the `lx:*` stages on
  the new path. If #5695 has landed, the new path runs `_distribute_work` and the old one
  does not, so a single pass's event would miss that work.
- **Noise:** defined by the data, from the two `A` runs of each round: the per-model A/A
  spread and the total A/A spread.
- **Pass:** median total on the arm ≤ median baseline total × (1 + total A/A spread).
- **Listed, not failed:** a model whose median is over its own baseline by more than the
  90th percentile of per-model A/A spread. A threshold on a quantile of the null flakes on
  a no-op change. Models under 0.2 s are excluded per model, as the feasibility study did.
- **Reported separately:** search time, since Stages 5 to 8 give the same model, and
  non-search time (`build`, `encode`, `read_back`, `check`, bridge), which is what this PR
  adds. If the gate fails, the Stage 3 check is the first place to look.
- Committed arms are measured too, since a joint-shaped model may cost more than the
  placement model.

### Gate 4: failure flow on real graphs

Not in the parent's exit, but it is what this stage adds, and Stage 0's corpus has one
forced failure for it.

- `ForcedFailure` on the `fake` zoo, default config, `FALLBACK_SOLVER=sa`: every
  model compiles, `check_report` reports nothing, `outcome.fell_back` is true.
- Traffic of the SA fallback against the greedy re-plan (`fallback_solver=""`) per
  model. **Report only** (Open questions), attached to the PR. Stage 12 needs it.
- Reused #5682 guards: plan equivalence is Gate 2's joint rule, and the Solve-status check
  fails any unforced run with `outcome.fell_back`.

## Risks

- **A plan differs only through the bridge.** Record order or copy-group numbering changes
  clone order or names. Caught by the plan comparison in Gate 2 on identical protos, and
  group numbering by `test_two_slot_spellings_are_one_group`. A pure renumbering of
  relayout clones on one source is a listed difference, not a failure.

- **A node exists with no matching record.** Stage 2 may build a node for an op that
  today's allocator leaves out of the solve. `commit_divisions` would then commit a split
  today's path never touched. Caught by divisions in the committed-plan diff.

- **Check false positive in production.** A compile that worked now raises. Bounded by the
  Phase 1 exit and Gate 1, and the new path is opt-in until Stage 12.


- **A user's deterministic run differs between arms.** Today it means one worker
  (`ilp_solver_ortools.py:1399-1403`), and `SolveSettings.from_config` gives at least 2
  interleaved. `install` patches that away, so no gate sees it. Stage 7 chose this.

- **`#5695` lands after this PR.** Its predicate must learn `family_key`. Caught by
  its agreement test, once extended.

## Deviations from the parent plan

1. **Infeasible with no user hint.**
   Plan: every proven-infeasible model warns and returns. Code: fixed-division conflicts
   raise `Unsupported`, and must not be bypassed (`allocator.py:3687-3689`, `:2386`).
   Resolution: `_return_infeasible` raises `Unsupported` for an illegal fixed division, and
   for any infeasible model with no user hint. With a user hint it follows the plan.
2. **`INVALID`.**
   Plan: silent. Code: `MODEL_INVALID` today is a `SolveError` and a silent re-plan
   (`ilp_solver_ortools.py:871`). It means `encode` posted a malformed model, and SA
   would hide that.
   Resolution: `RuntimeError`, same reasoning as `RuleViolation`.
3. **Committed-mode baseline.**
   Plan: lists `CoOptimizingAllocator` and `SaCoOptimizingSolver`, "including committed
   mode". Code: `select_allocator` uses `ScratchpadAllocator` there (`allocator.py:5465`),
   and `SaCoOptimizingSolver.plan_layout` raises (`sa_cooptimizer.py:767-776`).
   Resolution: baselines are `ScratchpadAllocator` with `CpSatLayoutSolver` and with
   `SimulatedAnnealingLayoutSolver`.
4. **"Placement-only is the same model with every division pinned" is not enough.**
   Code: placement-only minimizes traffic with no relayouts (`ilp_solver_ortools.py:687`,
   `:632`), and residency branches on `division_is_fixed` (`allocator.py:832`, `:847`,
   `:896-915`, `:932`). A priced objective could not reach a traffic gate.
   Resolution: committed mode builds on a target derived with
   `SpyreCostModel(traffic_only=True)`, pins relayouts out and takes the fixed-division
   branches, through `graph.divisions_committed`. Stage 13's claim holds
   with that reading.
5. **"No worse" cannot hold per solve on committed arms.**
   Plan: "HBM traffic is no worse than today's", "including committed mode". Code: the arms
   solve different models, the solver minimizes `spill_cost` (a differential,
   `ilp_solver_ortools.py:366`) and the metric counts bytes, so an equal-objective tie can
   raise bytes. SA committed compares two different annealers.
   Resolution: Gate 2's bands for OR-Tools, and a five-seed corpus median for SA.
6. **SA family gets `LxContextSwitchingPass`; today's joint SA does not.**
   Plan: "`LxContextSwitchingPass` stays a separate pass" after commit. Code:
   `allocator.py:5432-5435` omits it while `enable_lx_context_switching` turns the guard
   off, a bug on main.
   Resolution: the pass runs for every family. Joint SA baselines are run with it patched
   in, so the extra dumps are not read as a traffic regression. The old path is left
   unchanged, as this stage must leave it, and the plan assumes no upstream fix.
7. **Selecting a family with `layout_solver` alone cannot run the suites.**
   Plan: "with each new family selected in turn". Code: tests patch `layout_solver`
   (35 sites to the two solvers).
   Resolution: temporary `lx_route_to_families`. Until Stage 13 removes it,
   `plan_allocation` reads `FAMILIES[family_key(config)]`, which is
   `config.layout_solver` except for the two routed legacy values.
8. **Planning time cannot use replayed captures.**
   Plan's Verification: replay zoo captures. Code: `build_graph` and the bridge need a live
   `GraphLowering`.
   Resolution: live `fake` compiles, interleaved in one process.
9. **`_stage_collective_outputs` runs after the check.**
   Plan: "a single solver-free check judges every plan". Code: it gives a spilled buffer an
   address and moves no chosen one, after the solve (`allocator.py:3525-3646`).
   Resolution: kept as is for Stage 9, because moving it needs a graph decision, which is
   Stage 10's commit work. `no_overlap` judges the staged addresses in the bridge.
10. **SA does not become the fallback here.**
    Plan: "SA by default", replacing the greedy re-plan "only once Stage 8's parity holds".
    Code: SA has no relayouts, and its seed is not the committed division, so it has no
    floor against greedy on a timed-out graph.
    Resolution: `fallback_solver` defaults to `""`, and Stage 12 flips it after Gate 4's
    report exists. The parent's entry point reads `FAMILIES[config.fallback_solver]`
    unconditionally; here an empty value gives no fallback family, and a solve with no
    plan raises `SolveError` to the greedy re-plan.

## Open questions

1. **Is Gate 4's comparison of the SA fallback with the greedy re-plan a report or a gate?**
   SA is not guaranteed at least as good as greedy, as mtsokol asked in PR #5: greedy has
   planner relayouts and SA does not. Recommendation: a report, since a fallback is rare and
   SA searches divisions. If it is a gate, Stage 12's flip also needs SA fallback traffic no
   worse than greedy, and `fallback_solver` may stay `""` longer.

## Out of scope

- `commit(graph)`, and porting the bridge's tail: Stage 10.
- The full per-term `cost_expr_record` on the new path: Stage 10, once decisions lower from
  the graph.
- Lifting stick pins, and making `solve_decides_work_division()` true on a family: Stage 11.
- Flipping the default and `fallback_solver`, the cron lanes and the compare tool:
  Stage 12.
- Deleting `ScratchpadAllocator`, `CoOptimizingAllocator`, the solvers, the `SolveError`
  re-plan, `_greedy_replan` and `lx_route_to_families`: Stage 13.
- Encoding, solving, `read_back`, ladder behavior: Stages 5 to 8.
- The debug greedy family itself: not staged. This plan keeps room for it.
- Fixing `select_allocator`'s SA branch on the old path: not in this plan (Deviation 6).
- Explanations naming the smallest conflicting set of hints: future work in the parent.
