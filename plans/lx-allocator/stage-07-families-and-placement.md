# LX allocator, Stage 7: Families and placement

Parent plan: `plans/lx-allocator.md` at `c8ebd86`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0, 1, 2, 3, 4, 5, 6. Needed by: Stages 8, 9, 10, 11, 12, 13.

## Goal

When this PR merges, the OR-Tools family exists as one object: `OrToolsFamily` encodes,
solves and reads back, and satisfies the `Family` protocol. Its `encode` records every
placement box in the `Encoding`. Its `solve` places them in one of two modes, set by its
`placement` field, `no_overlap` (what today does) or `pack` (new), runs the objective stages
and returns a `Solution`. Its `read_back` decodes that solution onto the graph. `FAMILIES`
maps a `config.layout_solver` value to the family. In `no_overlap` mode the whole model is
compared with today's for the first time, and every solve in the sequence must be
byte-identical. Nothing in production calls the family yet. `select_allocator`,
`scratchpad_planning` and every existing solver are unchanged, and `config.layout_solver`
still defaults to `"cpsat"`.

## Today

All citations are under `torch_spyre/_inductor/scratchpad/` unless they name another path.

**The plan's `no_overlap` mode is today's mode.** `CpSatLayoutSolver` places natively with
one global `AddNoOverlap2D` (`ilp_solver_ortools.py:1762`). The `pack` mode does not exist
anywhere. There is no `AddCumulative` or `add_cumulative` in `torch_spyre/`, `tests/` or
`tools/`. `_C.pack_boxes`, named in the sketch in `sketch/`, is not on main:
`torch_spyre/csrc/module.cpp` and `perm_layout_native.h` register only
`NativePermutationLayoutSolver`. That class and `FirstFitLayoutSolver` evict what does not
fit, so neither reports how far a placement overflowed, which is what lowering a roof needs
(`firstfit_bestfit_solver.py:82-96`).

**What placement adds to the proto, in posting order:**

| entry | where | notes |
|---|---|---|
| `off_{name}`, domain `[0, max(0, M-1)]` | `_LifetimeBufferWithCpVars.__post_init__`, `ilp_solver_ortools.py:301` | created right after `in_buffer_{name}` and before `merge_*`, for every buffer |
| `offset(src) == offset(dst)` under the merge literal | `_add_inplace_relaxation:1639` | first constraint of each edge's group; implications and `constrain_merge` follow |
| `add_at_most_one` on incoming and outgoing merges | `:1656-1658` | in-place rule, not placement, but it sits between the two groups below |
| `offset + eff_size <= M` under `in_buffer` | `:1660-1662` | one loop over all buffers |
| `start_{name}`, `[t, t+1]`, `== t + sum(ins)` | `_add_no_overlap_2d:1718-1721` | only with an in-place parent; `t` is `_Slots.write_slot(start_time)` |
| `length_{name}`, `[0, _Slots.horizon(bufs)]` | `:1725-1731` | only when `sb.end` or `sb.start` is a variable |
| optional interval `x_{name}` | `:1735-1739` | present iff `in_buffer` |
| `top_{name}`, `[0, max(0, M-1) + buffer.size]`, `== offset + eff_size` | `:1747-1752` | `buffer.size` is the scaled total, not the per-core footprint |
| optional interval `y_{name}` | `:1753-1761` | |
| `AddNoOverlap2D(x, y)` | `:1762` | |

`_run` calls these before the residency gates (`:1369-1373`): tiling, lifetime ends, in-place
relaxation with placement, then `_add_core_division`, then the relayout coupling. So the
no-overlap constraint is in the middle of the model, not the end. Solver units are alignment
units: `_capacity_units = limit // alignment` (`:679`), sizes go through
`ceil_div(size, alignment)` in `_wrap` (`:727`), and offsets are scaled back to bytes in
`_plan_layout_generic:773`. Time is in `_Slots` (`:229-245`): op `t` writes in slot `2t` and a
cut's copy runs in slot `2t+1`.

**In-place sharing.** A fired merge pins one offset and shortens the child by the single
handoff tick, never the parent, so the parent's box still protects the bytes above a smaller
child (`:1677-1699`). `_check_in_place_relationships` guarantees the overlap is exactly one tick
(`plan_solver.py:947`). A child whose whole life is the handoff becomes a zero-length box that the
propagator ignores (`:1696-1699`). In the joint model a merge requires equal footprints
(`constrain_merge:604`).

**A box's lifetime can be a variable.** `sb.end` is an `IntVar` when a tiling can move it, and
`sb.start` is one for an input clone, an LX prefetch, that a nest may hoist (`_add_lifetime_ends:1231-1272`). An in-place
merge pins the parent's `end` exactly (`:1649-1652`).

**Search parameters** are set once in `_run` and kept for the whole solve sequence:

- `max_time_in_seconds = cpsat_time_limit_seconds`, unset when that is 0 (`:1376`). It applies per
  solve, so a five-stage ladder can spend five budgets (`config.py:379-384`).
- `cp_model_presolve = False` only above `lx_solver_relayout_presolve_max_copies`, default 0, which
  means never (`:1383-1398`).
- `num_search_workers` is 1 when `torch.are_deterministic_algorithms_enabled()`, otherwise
  `config.get_cpu_count()` (`:1399-1403`). Pinned by
  `tests/inductor/test_solver_relayout_decision.py:232`.
- `random_seed = 0` always (`:1405`).
- `ignore_subsolvers` drops the three `variables_shaving*` subsolvers, which can segfault on a
  `NoOverlap2D` under OR-Tools 9.15 (`:160-164`, `:1406-1415`).

Today sets neither `max_deterministic_time` nor `interleave_search`, and posts no hints: there is
no `add_hint` in `scratchpad/` or `tests/inductor/`.

**The objective is not always one solve.** With a cost expression, `_minimize_cost_expr` solves once
(`:1447`, `:830-900`), with no objective when the cost is constant (`:866-870`). Otherwise `_run` runs up
to five stages, each locking the previous optimum (`:1458-1523`): residency, cut tiebreak, parallelism,
division shape, tile count. CP-SAT with `AUTO_COARSE_TILING=1` always takes this path, because the
allocator withholds the expression (`allocator.py:3275-3295`), and so does a failed build or lowering
(`allocator.py:3336`, `:886-900`).

**After the solve.** `_extract` builds placement units from fired merges by union-find, coarsens each to
one rectangle (union of lifetimes, largest footprint), and slides them down with `_justify`; if that does
not fit it keeps the solver's offsets (`:1811-1887`, `:1926-1967`). It then writes `chosen_division` and
`address` and folds fired relayouts into `ChosenRelayout` (`:1889-1922`). `_plan_layout_generic` sets
`spill_reason` to the forced reason, else `untiled_bar` when the op stayed untiled, else
`_SOLVER_CHOSE_SPILL` (`:774-782`). `merge_vars`, `cut`, `via_full` and `joined` are never read back:
commit re-derives cuts and copy-outs from the chosen tilings (`wsr/coarse_tile.py:590`). The
`bottom_justify` and `time_limit_seconds` constructor arguments have no caller, so `_justify` always runs.

**Outcome handling.** A stage with no plan raises `SolveError`, which `scratchpad_planning` turns into a
greedy re-plan (`allocator.py:5495-5515`). That happens even when an earlier ladder stage held a plan.

**Config and the cache key.** `layout_solver` is a `Literal` of five strings read from `LAYOUT_SOLVER`
with no validation (`config.py:365-367`). Only `select_allocator` reads it for routing (`allocator.py:5406`,
`:5432`), and an unknown value raises `ValueError("Invalid layout_solver config option ...")`
(`:5409-5414`). The path into the FX graph cache key is:
`register_backend_for_device(..., device_custom_config=config)` (`torch_spyre/_inductor/__init__.py:276-281`);
torch stores it in `custom_backend_codegen_configs` (`torch/_inductor/codegen/common.py:430`, torch 2.13.0);
`FxGraphHashDetails` calls `save_config_portable(ignore_private_configs=False, readonly_values=True)`
(`torch/_inductor/codecache.py:1549`), which drops only `_cache_config_ignore_prefix`
(`config.py:503-519`). So every other module-level setting, private ones included, is in the key. The
deterministic-algorithms flag is hashed separately (`codecache.py:1451`). `num_cpus` is deliberately out of
the key. `tests/inductor/test_solver_relayout_decision.py:320` pins what each ignored prefix matches.

**OR-Tools can be absent.** The base install has no `ortools` on ppc64le or s390x
(`requirements/run.txt:29`); ppc64le gets a from-source build out of band (`pyproject.toml:37-39`).
`ilp_solver_ortools.py:120-132` guards the import, and `_cpsat_solver_class` then returns greedy
(`allocator.py:5345-5366`). With co-optimization on by default, `select_allocator` then raises `ValueError`
(`allocator.py:5436-5450`).

## Design

### Modules

All paths are under `torch_spyre/_inductor/scratchpad/lx_allocator/family/`. `base.py`, `placement.py`
and `packer.py` import no solver. `registry.py` imports only the guarded family module.

| module | holds |
|---|---|
| `base.py` | Stage 5's `Encoding`, which gains `boxes` and `capacity`; `Family`, the protocol |
| `placement.py` | `PlacementBox`, `SolveSettings`, `SolveStatus`, `Solution` |
| `packer.py` | `SolvedBox`, `PlacementUnit`, `placement_units`, `justify`, `Packer`, `Packed` |
| `ortools_family.py` | Stage 5's `OrToolsFamily`, which gains its fields, `solve` and `read_back`; `CpSatSolution`, `_NativePlacer`, `_PackPlacer`; guards `cp_model` as `ilp_solver_ortools.py:120-132` does |
| `ortools_encode.py` | Stage 5's and 6's encoding functions, which call the placer; `OrToolsState`, which gains the placement fields |
| `registry.py` | `FAMILIES` |

### `PlacementBox` and the encoding

```python
PlacementKey = Hashable  # opaque to placement: an OpNode, or a relayout group's name

@dataclass(eq=False)
class PlacementBox:
    """One thing that may sit in LX. Fields are ints or the family's variables."""
    key: PlacementKey
    name: str                      # today's buffer name, used in variable names
    start: int | Any               # first slot held, before any in-place shortening
    end: int | Any                 # exclusive end slot
    size: int | Any                # per-core footprint in alignment units
    max_size: int                  # OpVars.units: bounds ``top_{name}``
    present: bool | Any            # residency
    ties: list[tuple[Any, PlacementKey]] = field(default_factory=list)
    # (merge_* literal, parent key): while it holds, this box shares the parent's offset
    # and starts one slot later. Today's incoming edges, in ``in_place_parents`` order.
    span: tuple[int | Any, int | Any] | None = None  # (start, end) as placed
```

There is one box per node output, a `RestickifyNode` included, and one per relayout group, however many edges
the group serves. A step has no box of its own, and a relayout step's belongs to its group; the one exception is
Stage 11's unpinned restickify step. `start` and `end` hold a variable when today's do. No field holds a menu index,
so the box survives lazy divisions.

Stage 7 adds to `Encoding`: `boxes: list[PlacementBox]`, default empty, and `capacity: int`, LX units per core,
equal to `graph.target.lx_bytes // graph.target.alignment` for today's `limit // alignment`. The OR-Tools
placement state goes on Stage 5's `OrToolsState`, reached as `enc.ortools`, not on `Encoding`:

```python
horizon: int = 0                       # _Slots.horizon over every box
offsets: dict[PlacementKey, Any] = field(default_factory=dict)   # native mode only
roof: Any | None = None                # pack mode only
placer: _Placer | None = None
```

`OpVars.offset` goes: the offset variable belongs to the placer, and pack mode has none.

Sizes, offsets and capacity are all in alignment units. `ceil_div(ceil_div(size, a), d)` equals
`ceil_div(ceil_div(size, d), a)`, so a box's size is `ceil_div(footprint_bytes, alignment)` under any
division. `OpNode.offset`, `Step.offset` and `Step.footprint` hold alignment units too. Anything that needs
bytes multiplies by `graph.target.alignment` at the edge. A relayout step's `source_footprint_bytes` and
`destination_footprint_bytes` stay in bytes, unrounded, because `LXRelayoutPlan` needs them.

### Native posting stays in encode order

A `solve` that posts offsets and the no-overlap cannot give a byte-identical proto, because today's `off_`
variables sit among the buffer variables and the no-overlap sits before the residency gates. Canonicalizing the
diff would not settle it: shuffling the order the buffers are visited changed decisions on 21 of 30 tied toy
models at one worker, seed 0. So the family posts at today's positions during `encode`, through a placer:

```python
class _Placer(Protocol):
    def declare(self, enc: Encoding, box: PlacementBox) -> None: ...
    def tie(self, enc: Encoding, parent: PlacementBox, child: PlacementBox,
            literal: Any) -> None: ...
    def place(self, enc: Encoding) -> None: ...
```

`OrToolsFamily.encode` sets `enc.ortools.placer` from its `placement` field before the first encoding function runs, so
the functions in `ortools_encode.py` never see the mode. Stage 5's `encode` sequence is unchanged. The calls land
at three points:

1. `declare(box)` from `add_unit`, between `in_buffer_{name}` and the `merge_*` variables. `add_unit` creates the box.
   Today's line 301.
2. `tie(parent, child, lit)` from `encode_in_place`, as the first post for each edge, after appending to
   `child.ties`. Line 1639. It replaces Stage 6's `_post_shared_offset`.
3. `place(enc)` as the body of `encode_placement`, after filling each box's `start` and `end` from
   `OpVars.start` and `.end` and setting `enc.ortools.horizon`. Lines 1660-1664. It replaces Stage 6's `_encode_capacity`.

Boxes follow `add_unit` order, which is today's `working` order: LX prefetches and computed buffers in
`_generate_buffers` order, then relayout copies last (`allocator.py:4490`).

Placement constraints keep Stage 5's invariant that every constraint has one `Provenance`. The placer records
`start = len(enc.model.proto.constraints)` and calls `tag(enc, subject, rule, start)` after posting, with the
`ALWAYS` check that states the rule, as its name string: `"in_place_conditions"` in `tie`, and `"boxes_fit_lx"`
(resident boxes fit LX and never overlap) in `place` with subject `"graph"`.

`_x_interval(enc, box)` is shared by both placers and ports lines 1708-1739. It creates `start_{name}` when
`box.ties` is non-empty, creates `length_{name}` when `box.start` or `box.end` is a variable, posts the optional
interval `x_{name}`, and sets `box.span`. It asserts that a box with a variable `start` has no ties.

```python
class _NativePlacer:  # tag calls left out
    def declare(self, enc, box):
        enc.ortools.offsets[box.key] = enc.model.new_int_var(
            0, max(0, enc.capacity - 1), f"off_{box.name}")

    def tie(self, enc, parent, child, literal):
        enc.model.add(enc.ortools.offsets[parent.key] == enc.ortools.offsets[child.key]
                      ).OnlyEnforceIf(literal)

    def place(self, enc):
        cp, top_bound = enc.model, max(0, enc.capacity - 1)
        for box in enc.boxes:
            offset = enc.ortools.offsets[box.key]
            cp.add(offset + box.size <= enc.capacity).OnlyEnforceIf(box.present)
        xs, ys = [], []
        for box in enc.boxes:
            offset = enc.ortools.offsets[box.key]
            xs.append(_x_interval(enc, box))
            top = cp.new_int_var(0, top_bound + box.max_size, f"top_{box.name}")
            cp.add(top == offset + box.size)
            ys.append(cp.new_optional_interval_var(
                offset, box.size, top, box.present, f"y_{box.name}"))
        cp.add_no_overlap_2d(xs, ys)
```

### Pack mode

```python
class _PackPlacer:
    def declare(self, enc, box): pass
    def tie(self, enc, parent, child, literal): pass  # box.ties already records it

    def place(self, enc):
        intervals = [_x_interval(enc, box) for box in enc.boxes]
        enc.ortools.roof = enc.model.new_int_var(0, enc.capacity, "roof")
        enc.model.add_cumulative(intervals, [b.size for b in enc.boxes], enc.ortools.roof)
```

The pack model is the native model without `off_*`, `top_*`, `y_*`, the tie equalities, the fit constraints and the
no-overlap, plus `roof` and the cumulative. A test asserts exactly that difference. `add_cumulative` accepts optional
intervals with variable sizes, a variable capacity and zero-length intervals in OR-Tools 9.15 (checked).

The cumulative is a relaxation of the 2-D constraint, since a load under the roof need not be packable.
`OrToolsFamily.solve` closes the gap:

```python
def _solve_packed(self, enc, settings) -> CpSatSolution:
    limit = enc.capacity
    for attempt in range(self.attempts):
        cp = enc.model.clone()
        if attempt:
            cp.add(enc.ortools.roof <= limit)
        solution = self._run_stages(enc, cp, settings)
        if not solution.ok:
            return solution if attempt == 0 else self._no_packing(solution)
        packed = Packer().pack(placement_units(self._solved_boxes(enc, solution)))
        overflow = packed.top - enc.capacity
        if overflow <= 0:
            solution.offsets = packed.offsets
            return solution
        limit = min(limit, packed.peak - overflow)
        if limit < 0:
            break
    return self._no_packing(solution)
```

Four details:

- **Each attempt runs on a clone.** Ladder stages add locks, and a lock from an attempt under a higher roof would make
  the next attempt infeasible. `cp_model.CpModel.clone` keeps variable indices, so stage expressions and `solver.Value` work on
  the clone through the original variables, as `_is_unique_solution:806-810` already relies on.
- **The roof falls from the achieved peak, not from the old limit.** `limit -= overflow` from capacity leaves the roof
  above a peak that was already below capacity, so the re-solve returns the same plan. `peak - overflow` always binds.
- **A later failure is not infeasibility.** INFEASIBLE after a lowered roof says nothing about the rules. It maps to
  `NO_PACKING`, so `optimize` retries with the fallback family instead of warning about user hints.
- **`attempts` is an `OrToolsFamily` field, default 4.** It is not a config value, because it is a search budget, and
  the mode is already in the cache key.

**An in-place pair in pack mode.** The cumulative needs nothing new: the shortened child (`start + sum(ins)`) makes the
handoff tick count the parent once. After the solve, `placement_units` joins fired ties into one unit: union of lifetimes,
largest footprint, one offset for every member. `constrain_merge` keeps footprints equal and the lifetimes abut, so the unit
is exact.

**A box's lifetime in pack mode.** `_solved_boxes` reads `box.span` through `solution.value`, so a tiled `end` and a hoisted
LX prefetch's `start` are the solved values. The in-place end equality (`:1649-1652`) still holds because Stage 6 posts it in
both modes. A zero-length in-place child is covered by its unit, and any other box always has length at least 1.

On the 12 tight toy instances I tried, pack was valid every time and took at most 3 attempts. It equalled the native
optimum on 8 and was worse on 4, the worst 19 against 9. The sketch's table reports equal objectives and a 5.7x to 18x
speedup on a synthetic decoder stack. Neither set is a capture, so this plan claims validity for pack mode, not optimality.

### `Packer`, `placement_units`, `justify`

```python
@dataclass(frozen=True)
class SolvedBox:
    key: PlacementKey
    start: int
    end: int
    size: int
    offset: int | None             # the solver's offset; None when packing
    tied_to: tuple[PlacementKey, ...]  # parents it overwrites in place

@dataclass
class PlacementUnit:
    members: list[PlacementKey]
    footprint: int
    start: int
    end: int
    offset: int = 0

def placement_units(boxes: Sequence[SolvedBox]) -> list[PlacementUnit]: ...
def justify(units: list[PlacementUnit], capacity: int) -> dict[PlacementKey, int] | None: ...

@dataclass(frozen=True)
class Packed:
    offsets: dict[PlacementKey, int]
    top: int    # highest unit reached, may exceed capacity
    peak: int   # largest load at any slot

class Packer:
    ORDERS = (by_start, by_size, by_area)
    def pack(self, units: Sequence[PlacementUnit]) -> Packed: ...
```

- `placement_units` is copied from `_extract:1837-1879`, keeping the union direction and component order, so ties in the
  justify sort break as today.
- `justify` is copied verbatim from `_justify:1926-1967`. The old copies are deleted in Stage 13.
- `Packer.pack` runs first-fit over each order on a stable sort and keeps the lowest `top`, the earliest order winning a
  tie. The orders are start then largest first, size descending then start, and area descending then start. `peak` is the
  sweep over unit starts.
- These are pure ints. They import no solver and are usable by any family. Stage 8 keeps `NativePermutationLayoutSolver`,
  because its parity gate is today's SA plans.

### Running the stages

Stage 4 owns the terms and Stage 6's `encode_objective` lowers them into `enc.ortools.objective_stages`, a
`list[SolveStage]`. Stage 7's `solve` runs them. `_run_stages` reproduces
`_run:1458-1523` with one rule: before each stage after the first it locks the previous stage (`expr <= round(value)`
when minimizing, `>=` when maximizing), then sets the objective and solves. I checked the rule against each of today's
conditions (`:1470`, `:1497`, `:1509`, `:1520`): a lock is posted exactly when another stage follows.

```python
def _run_stages(self, enc, cp, settings) -> CpSatSolution:
    solver = self._configured_solver(enc, settings)
    for index, stage in enumerate(enc.ortools.objective_stages):
        ...  # lock the previous stage, set this one's objective
        status = self._solve_stage(solver, cp, objective=stage.name == "cost")
        ...  # a status with no plan ends the run

def _solve_stage(self, solver, cp, *, objective: bool = False) -> int:
    """The one solve call. Port of _solve_and_record:1346-1355, same signature."""
```

Setting the first stage's objective again is harmless: a repeated `minimize` left the proto text unchanged (checked). A
stage with `expr=None` solves for any plan. In `no_overlap` mode `cp` is `enc.model` itself, with no clone, so each proto is
exactly what today solves, and an `Encoding` is single-use, as a solver instance is today. `_is_unique_solution` is copied and
runs only after a `"cost"` stage that returned OPTIMAL, when the family's `uniqueness_check` field is set, so `solve` does not
read config.

### Under the harness

Stage 0's stage hook sits on `CpSatLayoutSolver._solve_and_record`, which the family never calls, so this stage adds a row to
`install` in `tests/inductor/lx_equivalence.py`. The record format does not change.

| target | wrapper |
|---|---|
| `OrToolsFamily._run_stages` | Open the solve with `begin_solve(family="ortools", solver="OrToolsFamily")`. At each stage, through its one solve call `_solve_stage`, apply `SolveParams`, or dtime 0 if a `ForcedFailure` matches, take `proto_text` before the solve, and call `record_stage` after it with `provenance_record` of each `Provenance`. |
| `SolveSettings.from_config` | Return `SolveSettings(DEFAULT_DTIME, workers=1, deterministic=True)`. |

The parity plugin closes the solve with `end_solve`, building each `BufferDecision` with `buffer_decision(decision, alignment)`
over Stage 3's `decisions(graph)`. A `ForcedFailure` gives `UNKNOWN`, which `_run_stages` returns as `NO_PLAN`.

### `SolveSettings`

```python
@dataclass(frozen=True)
class SolveSettings:
    time_limit: float = 60.0  # per solve; seconds, or deterministic time when
                              # deterministic; 0 means no limit
    workers: int = 1
    deterministic: bool = False

    @classmethod
    def from_config(cls, config) -> "SolveSettings":
        deterministic = torch.are_deterministic_algorithms_enabled()
        workers = config.get_cpu_count()
        return cls(config.cpsat_time_limit_seconds,
                   max(2, workers) if deterministic else workers, deterministic)
```

| parameter | today | default | deterministic, 1 worker | deterministic, 2+ workers |
|---|---|---|---|---|
| `max_time_in_seconds` | limit if nonzero | same | unset | unset |
| `max_deterministic_time` | unset | unset | limit if nonzero | limit if nonzero |
| `num_search_workers` | 1 or CPU count | `workers` | 1 | `workers` |
| `interleave_search` | unset | unset | unset | True |
| `random_seed` | 0 | 0 | 0 | 0 |
| `ignore_subsolvers` | `_CRASHING_SUBSOLVERS` | same | same | same |
| `cp_model_presolve` | False above threshold | same | same | same |

The default column equals today's non-deterministic parameters field for field, and a test compares them. The field stays
`num_search_workers`, since existing tests and the harness read that one. The presolve
threshold is an `OrToolsFamily` field read from config, and `solve` compares it with
`len(enc.ortools.relayout_literals)`, so `SolveSettings` stays free of CP-SAT terms.

**Why one worker does not interleave.** In OR-Tools 9.15 on toy models, `interleave_search` with one worker gave different
results across identical runs on all 6 instances tested, while one worker without it repeated exactly, and two or more
workers with it repeated exactly and agreed across 2, 4 and 8 workers. The agreement across worker counts held only where the
solve reached OPTIMAL: my attempts to cut it with a budget overshot (budget 0.5, deterministic time 1.6 to 3.5), so it is
unverified under a binding budget. `from_config` uses at least 2 interleaved workers for that reason. Stage 0's `install` gives
the "deterministic, 1 worker" column, which matches `SolveParams`.

### `Solution`

```python
class SolveStatus(Enum):
    OPTIMAL = auto()
    FEASIBLE = auto()     # a plan, optimality unproven
    INFEASIBLE = auto()   # the rules and pins admit no plan
    NO_PLAN = auto()      # the budget ran out with no plan, or a later stage failed
    NO_PACKING = auto()   # no packing fit within the attempts
    INVALID = auto()      # the solver rejected the model: an encoding bug

@dataclass
class Solution:
    status: SolveStatus
    objective: float | None = None        # the first objective stage's value
    offsets: dict[PlacementKey, int] = field(default_factory=dict)  # alignment units
    stats: dict[str, Any] = field(default_factory=dict)             # last_solve_stats keys

    @property
    def ok(self) -> bool:
        return self.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)

@dataclass
class CpSatSolution(Solution):
    encoding: Encoding | None = None   # what read_back decodes
    def value(self, x) -> int: ...   # ints pass through, else solver.Value
    def truth(self, x) -> bool: ...  # bools pass through, else solver.BooleanValue
```

`value` and `truth` are on `CpSatSolution` only, so the base class asks nothing of a family without CP variables.
`read_back(solution, graph)` takes no encoding, so the solution carries the one it solved, and the family holds no
state between calls.

Mapping from CP-SAT: OPTIMAL, FEASIBLE, INFEASIBLE and MODEL_INVALID map by name, and UNKNOWN becomes `NO_PLAN`. Only
the first stage can return `INFEASIBLE`. At a later stage it can only mean a lock failed, so it becomes `NO_PLAN`. A later stage that fails gives
`NO_PLAN` and loses the earlier stage's plan, as today, and goes to the fallback. A family's `solve` returns these statuses
and never raises for them.

### `Family`, `OrToolsFamily` and config

```python
class Family(Protocol):
    ENCODES: ClassVar[Restriction]
    RULES: ClassVar[Rule]
    STEPS: ClassVar[frozenset[StepKind]]
    OPTIMIZATIONS: ClassVar[Optimization]
    @classmethod
    def from_config(cls, config) -> "Family": ...
    def encode(self, graph: SpyreOptimizationGraph) -> Encoding: ...
    def solve(self, encoding: Encoding, settings: SolveSettings) -> Solution: ...
    def read_back(self, solution: Solution, graph: SpyreOptimizationGraph) -> None: ...

Placement = Literal["no_overlap", "pack"]

@dataclass(frozen=True)
class OrToolsFamily:
    placement: Placement = "no_overlap"
    attempts: int = 4
    presolve_max_copies: int = 0
    uniqueness_check: bool = False

    @classmethod
    def from_config(cls, config) -> "OrToolsFamily":
        return cls(placement=config.lx_placement,
                   presolve_max_copies=config.lx_solver_relayout_presolve_max_copies,
                   uniqueness_check=config.enable_uniqueness_check)

    def encode(self, graph: SpyreOptimizationGraph) -> Encoding: ...   # Stages 5, 6
    def solve(self, encoding: Encoding, settings: SolveSettings) -> CpSatSolution: ...
    def read_back(self, solution: Solution, graph: SpyreOptimizationGraph) -> None: ...
```

`ENCODES`, `RULES`, `STEPS` and `OPTIMIZATIONS` are the ones Stages 5 and 6 declare; placement adds no flag.
Coarse tiling is not a family field: `encode` reads `Optimization.COARSE_TILING` from
`graph.target.optimizations`, which `select_target` narrows by config. The cost model is not a field either:
`encode` reads `graph.target.cost_model`.

`OrToolsFamily.__post_init__` raises `ValueError` on an unknown placement, and, as in Stage 5, `ImportError` with
the message at `ilp_solver_ortools.py:664-669` when `cp_model` is `None`. The field defaults are config's defaults.
`from_config(config)` reads `lx_placement`, `lx_solver_relayout_presolve_max_copies` and `enable_uniqueness_check`,
so callers build a family with `FAMILIES[key].from_config(config)` and tests pass fields.

In `registry.py`:

```python
FAMILIES: Mapping[str, type[Family]] = {"ortools": OrToolsFamily}
```

Config changes in `config.py`. `lx_placement` says how the `"ortools"` family places LX boxes.

```python
layout_solver: Literal["greedy", "bestfit", "firstfit", "cpsat", "simulated_annealing",
                       "ortools"] = ...
lx_placement: Literal["no_overlap", "pack"] = os.environ.get(
    "SPYRE_LX_PLACEMENT", "no_overlap")
```

Both are ordinary settings, so both reach the cache key by the path traced above, and `lx_placement` matches no ignored prefix.
The family's source files are not in the key; Stage 9 adds them. `select_allocator` is not touched: `"ortools"` reaches its existing
`ValueError` until Stage 9 routes it. `registry.py` is imported lazily, as `_cpsat_solver_class` imports `ilp_solver_ortools`.

### `OrToolsFamily.read_back`

```python
def read_back(self, solution: Solution, graph: SpyreOptimizationGraph) -> None:
```

It returns at once when `solution.ok` is false, so a failed solve leaves the graph as built. Here `enc` is
`solution.encoding`, `value` and `truth` are `solution.value` and `solution.truth`, `v` is `enc.ops[node]`, and the
decode helpers are Stage 6's functions in `ortools_encode.py`.

| decision | read from | today | lands on |
|---|---|---|---|
| division | `work_split_from_division(node, v.table.entries[value(v.division)])` | `record_division:616` | `node.work_split`, non-input nodes |
| prefetch view | `prefetch_view(enc, node, value)` | `:616` | `InputNode.prefetch_view`; `work_split` stays `None` |
| residency | `truth(v.in_buffer)` | `_extract:1827` | `node.resident` |
| offset | `solution.offsets[node]` | `_extract:1889-1895` | `node.offset`, units, `None` when spilled |
| spill reason | `v.table.residency`, else `v.table.untiled_bar` if the entry is untiled, else `_SOLVER_CHOSE_SPILL`; `None` when resident | `:774-782` | `node.spill_reason` |
| in place | `truth(in_place_literal(enc, edge))`, `False` when `None` | used only for units | `edge.in_place` |
| relayout | `fired_relayouts(enc, value)` | `:1901-1922` | one relayout step per group on each of `fired.gate.edges` |
| copy-out | `truth(copy_out_literal(enc, src))` and `truth(via_full_literal(enc, edge))` | not read today | one copy-out step per producer, on each outside edge |
| nest boundary | `not truth(joined_literal(enc, before))`, for each adjacent pair of running ops | not read today | `graph.cuts.add(graph.step(before))` |
| stick | nothing | none | unchanged in Phase 2; Stage 11 adds it |

Tricky cases:

- **Relayouts** have one decoder, `fired_relayouts`, which resolves each fired literal to its one row and raises otherwise. It
  guards PR #5676's confirmed wrong-plan bug, so `read_back` never touches a served literal or a menu index. One
  `Step(kind=StepKind.RELAYOUT, ...)` is built per `fired.group.name` and shared by every edge it serves, with
  `to_view = row.destination_view`, `reduction_split = row.reduction_split`, `source_view = row.source_view`, both byte
  footprints from the row, `resident=True`, and the box's offset and size in units. No index reaches the graph.
- **A `RestickifyNode`** decodes as any node. Its pinned restickify step gets nothing.
- **Step order.** List order of `edge.steps` is physical, and nothing sorts it. A copy-out step goes through `insert_step` at the
  front, by `STEP_KINDS[StepKind.COPY_OUT].order`. A relayout step that reads the producer goes through `insert_step` before a
  pinned restickify step on that edge, which is legal and is what the cross-frame hazard wants. Otherwise it goes through
  `add_step`.
- **Copy-outs.** `add_step` and `insert_step` record the shared instance in `Step.edges`. `Step.footprint` is the tile's size in
  units. It has no offset, since `STEP_KINDS[StepKind.COPY_OUT].in_lx` is false.
- **`graph.cuts`** gets the tick of `before` for every adjacent pair of running ops that does not share a non-empty nest.
  `joined_literal` is `False` across a segment break, so that includes two untiled ops, matching `nests`, where the id steps at
  every cut.
- **A cut with no data edge.** A tiled graph output, or a tiled buffer read only by an extern kernel, is cut unconditionally
  (`:982`, `:1013-1015`) and has no edge to carry a step. `read_back` adds no step. Its traffic is priced by node, and
  Stage 10's `verify_copy_outs` skips such producers.
- **Solved lifetimes** are not written to the graph. `check_decisions` computes the same span from the decisions. A test compares
  `graph.lifetime` in slots with `box.span` for every resident box on tiled solves.

### Room for a debug greedy family and for Stage 8

The owner and mtsokol agreed on a debug greedy solve that follows the same encode-then-solve pattern, and its `encode` may do
nothing. Nothing here rules that out:

- `boxes` and `provenance` may be empty, and a family may solve from the graph alone. Its `Encoding.model` would need a third
  member in the union (`None` for a family with no native model), which that family's PR adds.
- `Solution` carries a status and offsets and has no `value`.
- `Family` names no CP type in its methods, and `FAMILIES` takes a new entry.
- `Packer`, `placement_units` and `PlacementBox` work on plain ints, so a greedy family can pack with them, on hosts without OR-Tools.
- Its config value cannot be `"greedy"` until Stage 13, because `GreedyLayoutSolver` holds that string. `"debug_greedy"` is free. This
  stage adds neither the value nor the family.

A test-only `_NullFamily` proves the protocol: its `encode` records nothing, its `solve` returns FEASIBLE, and the round trip
through `FAMILIES[key].from_config(config)` works. For Stage 8, `SAFamily.encode` returns an `Encoding` whose `model` is an `AnnealModel`, with
integer `PlacementBox`es, and it registers `"sa"`. SA also has no use for `_Placer`, which is private to the OR-Tools family.

### Old code to new home

| old | new home | moved or copied |
|---|---|---|
| `_add_no_overlap_2d`, the fit loop, the tie equality | `_NativePlacer`, `_x_interval` | copied; Stage 13 deletes |
| `_justify`, `_PlacementUnit`, the union-find in `_extract` | `justify`, `PlacementUnit`, `placement_units` | copied; Stage 13 deletes |
| `_solve_and_record`, `_is_unique_solution`, parameter setup, ladder loop and locks | `OrToolsFamily.solve` and its helpers | copied; Stage 13 deletes |
| `_extract`, spill reasons in `_plan_layout_generic` | `OrToolsFamily.read_back` | copied; Stage 13 deletes |
| Stage 6's `_post_shared_offset`, `_encode_capacity` | `_Placer.tie`, `_Placer.place` | moved; deleted here |

## Changes

| file | change |
|---|---|
| `torch_spyre/_inductor/config.py` | `"ortools"` in the `layout_solver` literal; `lx_placement`; comments |
| `scratchpad/lx_allocator/family/placement.py`, `packer.py`, `registry.py` | new, see Design |
| `lx_allocator/family/base.py` | `Family`; `Encoding.boxes`, `.capacity` |
| `lx_allocator/family/ortools_family.py` | `OrToolsFamily` fields, `from_config` and `__post_init__`; `encode` sets `enc.ortools.placer`; `solve`; `read_back`, with `solution` typed `Solution`; `CpSatSolution`, `_NativePlacer`, `_PackPlacer`, `_x_interval` |
| `lx_allocator/family/ortools_encode.py` | `OrToolsState.horizon`, `.offsets`, `.roof`, `.placer`; `add_unit`, `encode_in_place` and `encode_placement` call the placer; `OpVars.offset`, `_post_shared_offset` and `_encode_capacity` removed |
| `tests/inductor/lx_equivalence.py` | `install` rows for `OrToolsFamily._run_stages` and `SolveSettings.from_config`; `buffer_decision` |
| `tests/inductor/lx_sections.py`, `lx_shadow_plugin.py`, `test_lx_live_parity.py` | drop set empty; the plugin also solves and decodes; whole proto on the fixed set |
| `tests/inductor/test_lx_families.py`, `test_lx_packer.py`, `test_lx_ortools_solve.py`, `test_lx_read_back.py`, `test_lx_placement_parity.py` | new |
| `tests/configs/torch_spyre_tests/inductor/test_lx_*_config.yaml` | one per new test file, `unlisted_test_mode: mandatory_success` |

Commits, each leaving the suite green:

1. Config: the `"ortools"` value and `lx_placement`, with cache-key tests.
2. `placement.py`: boxes, settings, statuses, `Solution`; the `Family` protocol, and `Encoding.boxes` and `capacity`, tested with `_NullFamily`.
3. `packer.py` with its pure tests, including the differential against `_justify`.
4. The `placement` field, box recording, `_NativePlacer` and the three calls, with provenance tags. Empty the `project_today` drop set. The whole proto of the first solve compares byte-identical.
5. `OrToolsFamily.solve` running stages in native mode with parameters, and the harness rows. Ladder cases compare byte-identical, solve by solve.
6. The other `OrToolsFamily` fields and `registry.py`.
7. `read_back` and `buffer_decision`, with their tests.
8. `_PackPlacer` and `_solve_packed` with their tests.
9. `test_lx_placement_parity.py`, the plugin's solve and decode, `test_lx_live_parity.py` on the whole proto, and shard configs.

## Interfaces

**Consumes**

| symbol | signature | from Stage | used for |
|---|---|---|---|
| harness | `install`, `SolveParams`, `ForcedFailure`, `DEFAULT_DTIME`, `Recorder.begin_solve`, `.record_stage`, `.end_solve`, `proto_text`, `provenance_record(p) -> dict`, `BufferDecision`, tiers `intree`, `fake`, `unit-*`, `failure` | 0 | gate, harness rows |
| `SpyreOptimizationGraph` | `.nodes` in op order, `.data_edges`, `.cuts: set[int]`, `.target`, `.step(node) -> int`, `.lifetime(node) -> tuple[int, int]` | 1 | `encode`, `read_back`, span test |
| `Target` | `.lx_bytes`, `.alignment`, `.optimizations`, `.cost_model`; `SPYRE`, `TARGETS` | 1 | `Encoding.capacity`, bytes at the edge, coverage test |
| `Restriction`, `Rule`, `Optimization`, `StepKind` | flag sets and the enum | 1 | `Family` class attributes |
| decision fields | `OpNode.work_split`, `.resident`, `.offset`; `InputNode.prefetch_view`; `DataEdge.in_place`, `.steps`, `.add_step(step)`, `.insert_step(index, step)` | 1 | `read_back` targets |
| `Step`, `STEP_KINDS` | `Step(kind=StepKind.RELAYOUT, to_view, reduction_split, source_view, source_footprint_bytes, destination_footprint_bytes)`; a restickify step's `pinned`, `realized_by`; `Step.resident`, `.offset`, `.footprint`, `.edges`; `STEP_KINDS[kind].order`, `.in_lx` | 1 | `read_back` |
| `work_split_from_division` | `(node, division) -> WorkSplit` | 2 | `read_back` |
| `build_graph`, `select_target` | `build_graph(ir, target)` with keyword defaults; `select_target(config) -> Target` | 2 | corpus runs |
| `check_decisions`, `decisions`, `NodeDecision` | `(graph) -> list[Violation]`; `(graph) -> dict[str, NodeDecision]` | 3 | gate; pack proof |
| `coverage`, `Coverage` | `(family, target) -> Coverage(gaps, unused)`; a forced step kind or a rule the family lacks is a gap, a chosen step kind or an optimization is only unused | 3 | coverage test |
| `decode_todays_plan` | `(graph, allocation, accepted_relayouts) -> list[Mismatch]` | 3 | gate |
| `ALWAYS` checks `in_place_conditions`, `boxes_fit_lx`; `RuleTag` | provenance rules, each tagged by its name string | 3 | `_Placer` |
| `OrToolsFamily` | `encode(graph) -> Encoding`; `ENCODES`, `RULES`, `STEPS`, `OPTIMIZATIONS`; `read_back` raising `NotImplementedError` | 5 | extended |
| `Encoding`, `OrToolsState`, `Provenance` | `model`, `provenance`, `pruned`, `symbols`, `ortools`; `OrToolsState.ops`, `.units` | 5 | extended |
| encoding functions | `add_unit(enc, table, *, size_units, boundary) -> OpVars`; `encode_placement(enc)`; `post`, `symbol`, `tag(enc, subject, rule, start)`; in `ortools_encode.py`, each taking `enc` first | 5 | hooks |
| `OpVars`, `DivisionTable` | `table`, `name`, `units`, `in_buffer`, `division`, `eff_size`; `entries`, `residency`, `untiled_bar` | 5 | boxes, `read_back` |
| `build_today`, `project_today`, `replay_graph`, `lx_shadow_plugin`, `test_lx_live_parity.py` | Stage 5 helpers and live arm | 5 | gate |
| `SolveStage`, `OrToolsState.objective_stages` | `SolveStage(name, expr, maximize=False)`; `name == "cost"` for the priced stage; `expr=None` for a constant cost; ladder names `residency`, `cut tiebreak`, `parallelism`, `division shape`, `tile count` | 6 | `_run_stages` |
| `encode_in_place`, `OpVars.end`, `.start`, slot arithmetic | in-place loop over each node's incoming edges, `add_at_most_one` before `encode_placement`; `_Slots` values as `_encode_lifetimes` uses them | 6 | `tie`, boxes |
| `_post_shared_offset`, `_encode_capacity` | Stage 6's parity helpers | 6 | replaced here |
| `fired_relayouts`, `FiredRelayout` | `(enc, value) -> list[FiredRelayout]`; `gate`, `group`, `row`; raises unless exactly one row | 6 | `read_back` |
| decode helpers | `joined_literal(enc, before)`, `in_place_literal(enc, edge)`, `copy_out_literal(enc, node)`, `via_full_literal(enc, edge)`, `prefetch_view(enc, node, value) -> PerCoreView`, `decision_variables(enc)` | 6 | `read_back`, uniqueness probe |
| `OrToolsState.relayout_literals` | `dict[str, literal]`, group name to residency | 6 | presolve switch |
| `edge_tables_from_buffers` | test helper | 6 | replay arm |

**Provides**

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `PlacementBox` | see Design | 8, 11 | final |
| `Encoding.boxes`, `Encoding.capacity` | `list[PlacementBox]`, `int` in alignment units | 8, 11 | final |
| `OrToolsState.horizon`, `.offsets`, `.roof`, `.placer` | OR-Tools placement state, reached as `enc.ortools` | none outside Stage 7 (Stage 11 adds step boxes through `Encoding.boxes`) | final |
| `Family` | protocol: `ENCODES`, `RULES`, `STEPS`, `OPTIMIZATIONS`; classmethod `from_config(config) -> Family`; `encode(graph) -> Encoding`, `solve(encoding, settings) -> Solution`, `read_back(solution, graph) -> None` | 8, 9, 13 | final |
| `OrToolsFamily` | frozen dataclass `(placement="no_overlap", attempts=4, presolve_max_copies=0, uniqueness_check=False)`, config's defaults; `from_config(config)` reads `lx_placement`, `lx_solver_relayout_presolve_max_copies`, `enable_uniqueness_check`; `solve(encoding, settings) -> CpSatSolution` | 9, 13 | final |
| `OrToolsFamily.read_back` | `read_back(solution, graph) -> None`; a no-op when `not solution.ok` | 9, 10, 11 | final |
| `FAMILIES` | `Mapping[str, type[Family]]`, `{"ortools": OrToolsFamily}`; `FAMILIES[key].from_config(config)` builds one | 8, 9, 12, 13 | final |
| `SolveSettings` | `time_limit`, `workers`, `deterministic`, `from_config(config)`; at least 2 workers when deterministic | 8, 9, 12 | final |
| `SolveStatus`, `Solution`, `CpSatSolution` | see Design; `CpSatSolution.encoding`, `.value(x)`, `.truth(x)` | 8-12 | final |
| `Packer`, `placement_units`, `justify` | see Design | any family that packs | final |
| `OpNode.offset`, `Step.offset`, `Step.footprint` | alignment units; bytes multiply by `graph.target.alignment` | 8, 9, 10 | final |
| `OpNode.spill_reason` values | Stage 1's field, filled by `read_back` | 8, 9, 10 | final |
| `config.layout_solver = "ortools"`, `config.lx_placement` | strings | 9, 12 | final |
| `Solution.stats` | keys of today's `last_solve_stats` | 10 (cost dump) | final |
| `OrToolsFamily._run_stages`, `._solve_stage` | harness targets | 9 | final |
| `buffer_decision` | `(decision: NodeDecision, alignment: int) -> BufferDecision`, in `lx_equivalence.py`; `menu_index=None`, `relayouts={}` | 8, 9 | final |

## Tests

`tests/inductor/test_lx_families.py`

- `test_families_maps_ortools`: the registry, and `OrToolsFamily.from_config(config)` reads each setting from config.
- `test_select_allocator_rejects_ortools_until_routed`: `pytest.raises(ValueError, match="Invalid layout_solver")`, tagged `@expects_raise`. Stage 9 turns it positive.
- `test_family_settings_enter_cache_key[layout_solver, lx_placement]`: patching either changes `save_config_portable(ignore_private_configs=False, readonly_values=True)`, using the helper at `test_solver_relayout_decision.py:286`.
- `test_solve_settings_from_config[deterministic, num_cpus]`: `(False, 16)` gives 16, `(True, 16)` gives 16, `(True, 1)` gives 2, and the time limit follows config.
- `test_solution_ok_by_status`: parametrized over `SolveStatus`.
- `test_null_family_round_trip`: `_NullFamily`, registered with `monkeypatch`, satisfies `Family`, encodes nothing, solves to FEASIBLE and decodes. This is the debug-greedy room.
- `test_family_holds_no_state`: two encodings from one `OrToolsFamily` stay independent, and `read_back` decodes the encoding its solution carries.
- `test_coverage_every_family_and_target`: for every pair in `FAMILIES` × `TARGETS`, `coverage(family, target).gaps` is empty, or the case is an explicit skip naming the gap.
- `test_family_modules_import_without_ortools`: in a subprocess with `sys.modules["ortools"] = None`, `registry` imports and `OrToolsFamily()` raises `ImportError` with today's message.
- `test_placement_and_packer_import_no_solver`: importing `base`, `placement` and `packer` leaves `ortools` out of `sys.modules`.

`tests/inductor/test_lx_packer.py`

- `test_first_fit_takes_lowest_gap`, `test_touching_boxes_do_not_conflict`, `test_best_of_three_orders`.
- `test_overflow_above_peak`: `(start, end, size)` of `(1,5,2), (2,4,4), (4,7,2), (5,6,5)` has peak 7 and every order reaches 8.
- `test_tied_group_packs_as_one`: an in-place pair takes one offset and spans the union of lifetimes. `test_placement_units_chain`: a three-link chain is one unit.
- `test_justify_matches_cpsat_justify`: differential against `CpSatLayoutSolver._justify` over seeded random units, including a `None` result. Stage 13 removes it.

`tests/inductor/test_lx_ortools_solve.py`, on hand-built encodings

- `test_default_parameters_equal_today`: capture `SatParameters` through a patched `CpSolver.Solve` from `CpSatLayoutSolver` and `OrToolsFamily.solve` on one model, and assert equal text.
- `test_harness_settings_match_solve_params`: `SolveSettings(DEFAULT_DTIME, workers=1, deterministic=True)` and `SolveParams.apply` agree as parsed values, where unset and `False`, and unset and `inf`, are equal.
- `test_forced_failure_is_no_plan[stage]`: a `ForcedFailure` at a stage under `install` gives `NO_PLAN`, recorded through `record_stage`.
- `test_placer_constraints_are_tagged[no_overlap, pack]`: every constraint has one `Provenance`, with the `"boxes_fit_lx"` or `"in_place_conditions"` check on placement's.
- `test_deterministic_one_worker_parameters`, `test_deterministic_many_workers_parameters`, `test_zero_time_limit_sets_no_limit`, `test_presolve_switch_above_threshold`.
- `test_deterministic_runs_repeat[1, 4]`: two runs give identical offsets and values. `test_interleaved_agrees_across_workers`: workers 2 and 4 agree on a model solved to OPTIMAL.
- `test_ladder_locks_like_today`: lock constraints and solve count equal today's on a tiled model.
- `test_single_stage_posts_no_lock`, `test_constant_cost_solves_without_objective`.
- `test_status_mapping`: INFEASIBLE at the first stage, UNKNOWN to `NO_PLAN`, a forced failure at a later stage to `NO_PLAN`.
- `test_offsets_keep_solver_values_when_justify_cannot_fit`.
- `test_pack_model_differs_only_by_placement`: the two protos differ by exactly the entries listed in Design.
- `test_pack_in_place_pair_shares_offset`: a fired in-place overwrite gives both boxes one offset.
- `test_pack_handoff_tick_counts_parent_once`: a pair that exactly fills capacity stays resident.
- `test_pack_uses_solved_lifetime`: a box with a variable `end` is packed against its solved end, and the same box with an earlier solved end shares space with a later one.
- `test_pack_roof_falls_and_refits`: on a model that overflows once, the second attempt fits and `enc.ortools.roof` was constrained below the first peak.
- `test_pack_can_lose_to_native`: the overflow instance at capacity 7 keeps everything resident natively and spills under pack, and both are valid. This documents the gap.
- `test_pack_exhausted_is_no_packing` (`attempts=1` on that instance), `test_pack_infeasible_after_lowering_is_no_packing`, `test_pack_infeasible_first_attempt_is_infeasible`.
- `test_pack_locks_do_not_leak_between_attempts`: a two-stage ladder that overflows once still finds a plan.
- `test_pack_repeatable`: two deterministic runs give equal offsets.

`tests/inductor/test_lx_read_back.py`, on hand-built graphs

- `test_no_plan_leaves_graph_untouched`, `test_division_to_work_split`, `test_prefetch_view_decoded`, `test_in_place_edge_decoded`.
- `test_resident_offset_in_units`, `test_spilled_offset_none`, `test_spill_reason_order[forced, untiled_bar, solver_chose]`.
- `test_relayout_only_through_fired_relayouts`: with `fired_relayouts` patched to return nothing, no relayout step appears although a served literal holds.
- `test_relayout_fields_from_row`: `to_view`, `reduction_split`, `source_view`, both footprints; no index on the step.
- `test_relayout_shared_across_served_edges`: one object, on two edges, with the group's offset.
- `test_relayout_inserted_before_pinned_restickify`, `test_steps_keep_list_order`: nothing sorts `steps`.
- `test_copy_out_shared_per_producer`, `test_cut_graph_output_leaves_no_step`, `test_cuts_decoded[joined, not_joined, segment_break]`: `graph.cuts` holds the tick before each cut and no other.
- `test_restickify_node_decodes_as_a_node`: the pinned step's fields stay unset.
- `test_buffer_decision`: decoded units scale to `address`, and fields map from `NodeDecision`.

`tests/inductor/test_lx_placement_parity.py` is the CI replay arm. It covers only what `replay_graph` and `edge_tables_from_buffers` can reach, namely `load_captures`, `synthetic_graphs()` and hand-built cases. It tests emission, solve and decode, not the table or facts, because those nodes have no IR.

- `test_solve_sequence_matches`, `test_protos_byte_identical`, `test_parameters_match`, `test_responses_match`, `test_decoded_plan_matches`: items 1 to 5 below.
- `test_native_plan_passes_check`, `test_pack_plan_passes_check`: item 6, partial where there is no IR.
- `test_solved_spans_match_graph_lifetime`: `box.span` against `graph.lifetime`, on tiled entries.
- `test_pack_objective_not_below_native`: on entries where both modes reach OPTIMAL.

`tests/inductor/test_lx_live_parity.py`, Stage 5's file, gains the whole-proto comparison and items 1 to 6 on `fake` compiles of the fixed set, at default and tiling.

Existing suites, run serially: `test_scratchpad_solver`, `test_solver_relayout_decision`, `test_cooptimization_types`, `test_cost_expr_dump`, `test_restickify_cost_model`, `test_sa_cooptimizer`, `test_coarse_tiling`, `test_work_division*`, `test_scratchpad_patterns`, `test_perm_layout_solver`, `test_scratchpad_use`, `test_solver_auto_coarse_tiling`, and the graph-layer tests. The parent plan lists no device suite after Stage 7, and this stage changes no production path, so `test_scratchpad_use` and the like are expected to pass as they stand. `make check-all-configs` must pass. `pre-commit run --files <changed>`.

**What proves the parent's claim.** "Places natively, or by packing": the parity suite for native, and the tests above plus `test_pack_plan_passes_check` for packing. "An in-place pair packs as one box": `test_tied_group_packs_as_one` and `test_pack_in_place_pair_shares_offset`. "`SolveSettings` carries the time limit, workers and opt-in determinism": the parameter tests.

## Parity and gates

The parent's gate: "Through Phase 1, and in Phase 2 wherever the rules agree with today's, the CP model stays byte-identical and the committed plans stay identical. Any other difference is listed by rule in its PR and either justified or fixed." Verification adds "the Stage 0 harness, comparing proto, decisions and committed plans between the stage's parent and the change."

Placement does not exist before this stage, so the proto diff finishes here: the `project_today` drop set becomes empty and the diff covers the whole model, where "byte-identical" means `proto_text`-identical.

Compared at each solve between `CpSatLayoutSolver` and `OrToolsFamily(placement="no_overlap")`, both under `install`, so `SolveSettings.from_config(config)` gives `SolveSettings(DEFAULT_DTIME, workers=1, deterministic=True)`:

1. **The sequence of solves.** The same count, stage by stage, including locks in ladder cases.
2. **Proto text per solve.** `proto_text`, recorded through `record_stage` on both sides.
3. **Parameters per solve.** Compared as parsed values, not text: `SolveParams` sets `interleave_search: false` and `max_time_in_seconds: inf` explicitly, where the deterministic column leaves them unset.
4. **Responses.** The solution vector is equal. Given 2 and 3, a deterministic solve implies this. It is checked because it is cheap and catches a non-determinism that would invalidate the rest.
5. **Decoded plan.** After `read_back`, `decisions(graph)` against `decisions` of a graph filled by `decode_todays_plan(graph, allocation, accepted_relayouts)`; fired relayouts against `chosen_relayouts` on the buffers, by destination view and address. `in_place`, copy-out steps and `graph.cuts` have no counterpart in today's output, so 4 and `check_decisions` cover them.
6. **`check_decisions(graph) == []`** on every plan, in both placement modes.

A pass is zero differences in 1 to 6. The PR-tip tiers are `intree`, the `fake` zoo at default and `AUTO_COARSE_TILING=1`, `unit-*` and `failure`, where a forced failure must return `NO_PLAN`. The `fake` zoo runs as live compiles: `lx_shadow_plugin` wraps `CoOptimizingAllocator._solve`, runs `build_graph(ir, select_target(config))`, encode, solve and `read_back`, and takes the allocator's accepted plans from `_finalize_lx_relayout_allocation`. Captures cannot reach the graph layer, so they serve only the replay arm. No card is needed. A model the `fake` tier cannot reach stays card-only and is listed in the PR. Fake inputs use default layouts, so this proves parity, not production plans. The PR reports hits per coverage feature, including the restickify features.

The gate covers joint solves. The placement-only allocator has no family counterpart here, since its model differs by design, and `greedy`, `bestfit`, `firstfit` and `simulated_annealing` have none either.

Committed plans cannot be compared here, because nothing commits a family's plan until Stage 9's bridge. Today's committed plans do not change, since no production path changed, and the Stage 0 diff shows that. Any difference in 1 to 6 is either a bug fixed in this PR or, where a rule intentionally differs, listed by rule with its Stage 5 or 6 reason.

Pack mode has no parity gate. It must pass item 6. The PR also reports its objective against native on every entry where both reach OPTIMAL, with the assertion `pack >= native`. I have no measurement of pack on real graphs.

## Risks

- **Box or hook order differs from today's.** LX prefetches come first and relayout copies last, and `declare` must precede the `merge_*` variables. A wrong order shifts every index. Caught by proto text.
- **Justify differs in a tie.** Two zero-length boxes share a sort key, so component order matters. Caught by the differential test and item 5.
- **Interleaved search is less deterministic than measured.** Toy models only, and a binding budget is untested. `num_cpus` is out of the cache key, so a plan that varies with worker count would vary by machine under one key. Caught, if at all, by `test_interleaved_agrees_across_workers`. It does not affect the gate, which runs one worker.
- **Pack compile time.** Worst case is `attempts × stages × time_limit`. Pack is off by default, and no default path can reach it.
- **Mutable `Encoding`.** A second `solve` on one encoding in native mode sees old locks. Stage 9's fallback re-encodes into a new encoding, so it is safe. `OrToolsFamily.solve` asserts first use.
- **The crash workaround.** `ignore_subsolvers` must stay on in every mode, including interleaved, or google/or-tools#5532 returns. Pinned by the parameter tests.
- **FakeTensor captures.** The replay arm must blank `residency_edges`, `division_space` and `tile_reads` before any deepcopy. Boxes hold references to nodes, so an `Encoding` must not be deep-copied. `enc.model.clone()` copies only the proto.
- **Drift on main.** The old solver stays live until Stage 13, and `test_lx_live_parity.py` is what fails in CI when its model changes without `ortools_encode.py`.

## Deviations from the parent plan

1. **Native placement posts during encode, not in `solve`.**
   - Plan: the family "records the boxes to place" when it encodes, and `OrToolsFamily.solve` "places in one of two modes, set by its `placement` field".
   - Code: `off_` is created among the buffer variables (`ilp_solver_ortools.py:301`), ties sit inside the in-place loop (`:1639`), and the no-overlap precedes the residency gates (`:1371-1372`). Posting at solve time cannot be byte-identical.
   - Resolution: a `_Placer`, chosen by `encode` from the `placement` field, called from three points in `encode`. The mode stays inside the family, and `boxes` stays the record. Once Stage 12 ends byte parity, the calls can move into `solve` with no API change.
2. **One `solve` can run several solves.**
   - Plan: `solve(encoding, settings)` runs one solve of one objective.
   - Code: up to five locked solves (`:1458-1523`).
   - Resolution: `OrToolsFamily._run_stages` runs `enc.ortools.objective_stages` and returns one `Solution`.
3. **One-worker determinism does not interleave.**
   - Plan: "budgets by deterministic time with a fixed seed and interleaved search"; Stage 0 solves on one worker.
   - Code: OR-Tools 9.15 with `interleave_search` at one worker did not repeat on any of 6 toy instances; at two or more workers it did.
   - Resolution: interleave when `workers >= 2`. `from_config` uses at least 2. Stage 0's `install` selects plain one-worker search.
4. **`OpNode.spill_reason`.**
   - Plan: `OpNode` has no spill reason.
   - Code: Stage 0 records spill reasons, and `CoOptimizingAllocator._get_spill_reasons` surfaces the solver's (`allocator.py:3648-3669`). Only `read_back` knows the solver chose to spill.
   - Resolution: Stage 1 declares the field, as plain data, and `read_back` fills it.
5. **A cut with no data edge has no copy-out step.**
   - Plan: a copy-out is a step on a data edge, "forced when a cut lies between a data edge's ends."
   - Code: a tiled graph output, or a buffer read only by an extern kernel, is cut with no modelled consumer (`:982`, `:1013-1015`), and its traffic is priced (`:1311-1314`).
   - Resolution: `read_back` adds no step, as in Design.
6. **More failure outcomes.**
   - Plan: the Failure row has infeasible, out of time, and failed check.
   - Code: pack mode can exhaust its attempts with a valid model, and a later ladder stage can fail (`:1435-1439`).
   - Resolution: `NO_PACKING` and `NO_PLAN`, both with `ok == False`, which Stage 9 sends to the fallback. `INVALID` exists for `MODEL_INVALID`, which today becomes a `SolveError`.
7. **`"ortools"` is selectable but not runnable in this stage.**
   - Plan: "chosen by a `config.layout_solver` value."
   - Code: `select_allocator` is the only router, and `plan_allocation` arrives in Stage 9.
   - Resolution: the value exists, is in the cache key, and `FAMILIES` resolves it. `select_allocator` raises its existing `ValueError` until Stage 9.

## Open questions

None.

## Out of scope

- Routing `"ortools"` through `scratchpad_planning`, `plan_allocation` and `optimize`, failure and fallback flow, the bridge to `_push_allocation`, the harness's `ortools` cells and `decisions_from_graph`: Stage 9.
- Objective terms: Stage 4. Lowering them, `encode_objective`, `SolveStage`: Stage 6.
- `AnnealModel`, `SAFamily` and the `"sa"` value: Stage 8.
- The debug greedy family and its config value: not staged. This stage only keeps it possible.
- Committing plans from the graph, including the cut with no data edge: Stage 10.
- Stick variables in `read_back` and layout decoding: Stage 11.
- Changing the default `layout_solver`, and any default for `lx_placement` other than `no_overlap`: Stage 12.
- Deleting `CpSatLayoutSolver`, `_justify`, `_extract` and the ladder's old home: Stage 13.
- Returning an earlier ladder stage's plan when a later stage fails. Worth revisiting after Stage 12.
- Hints on pack re-solves: not planned. Today has none, and they would change nothing in native mode.
- The CPMpy family and a MIP family: future work.
