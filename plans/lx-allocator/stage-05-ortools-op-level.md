# LX allocator, Stage 5: `OrToolsEncoder`, op level

Parent plan: `plans/lx-allocator.md` at `5aff4a6`. Code base: torch-spyre `5268ba95`.
Depends on: Stages 0, 1, 2, 3, 4. Needed by: Stages 6, 7, 8, 9, 11, 13.

## Goal

When this PR merges, `OrToolsEncoder.encode(graph, restrictions)` turns a
`SpyreOptimizationGraph` into a `CpModel` holding the op-level variables and constraints
of today's joint CP-SAT model for every op, created through the same `cp_model` calls in
the same relative order, with each constraint tagged with its rule and each pruned table
entry recorded. `Encoder`, `Model`, `CpModel`, `DivisionTable` and `build_division_tables`
exist and have tests. Nothing in a compile calls them yet: `CpSatLayoutSolver` still
builds and solves every production model, and the allocator changes are extractions that
leave protos, decisions and plans unchanged. `test_lx_live_parity.py` holds the encoder
to `CpSatLayoutSolver` in CI from here on. The Stage 5 model is not a complete model and
is never solved in production.

## Today

### Which code is op level

Short names in citations, so the tables stay readable. Under
`torch_spyre/_inductor/scratchpad/`: `ilp_solver_ortools.py`, `allocator.py`,
`plan_solver.py`, `residency_rules.py`, `cpsat_expr_printer.py`, and `scratchpad/utils.py`.
Under `torch_spyre/_inductor/`: `work_division.py`, `work_division_constraints.py`,
`wsr/enumerate_tilings.py`. A bare `lines N-M` means `ilp_solver_ortools.py`.

| lines | what | owner |
|---|---|---|
| `ilp_solver_ortools.py:298-301` | `in_buffer_*`, `off_*` | **Stage 5** |
| `ilp_solver_ortools.py:308-311` | `merge_*`, created mid-wrapper | Stage 6 (hook here) |
| `ilp_solver_ortools.py:431-518` | division index, `eff_size`, `core_cost`, `cores`, split and tile columns, one-entry constants | **Stage 5** |
| `ilp_solver_ortools.py:520-525` | `_division_is` literal, first used by Stage 6 and the printer | **Stage 5** |
| `ilp_solver_ortools.py:716-738` | `_wrap`: scales `size` to alignment units | **Stage 5** |
| `ilp_solver_ortools.py:837-862` | `sym_map`, `buffer_map` | **Stage 5** |
| `ilp_solver_ortools.py:864-900` | objective lowering and solve | Stage 4 terms and printer, Stage 6 lowering, Stage 7 solve |
| `ilp_solver_ortools.py:944-973` | nest-id interning, `loop_id_*`, `tiled_*` columns | **Stage 5** |
| `ilp_solver_ortools.py:975-1172` | groups, `apart`, `via_full`, cuts, cut guard, nest ends and starts | Stage 6 |
| `ilp_solver_ortools.py:1191-1202` | `tile_count_*` column | **Stage 5** |
| `ilp_solver_ortools.py:1375-1523` | solver parameters, lexicographic ladder | Stage 7 runs it, Stage 6 lowers it |
| `ilp_solver_ortools.py:1795-1804` | forced residency pins, untiled bar | **Stage 5** |
| `allocator.py:3694-3713` | core budget, prescribed regions, readers for tiling | **Stage 5** |
| `allocator.py:3724-3785` | the reasons an op keeps a fixed division | Stage 2 (pins) |
| `allocator.py:3821-3881` | fp8 seed alignment, run from a pin's `seeds` | **Stage 5** |
| `allocator.py:3786-3796` | pruned menus for `ExhaustiveSearchSolver` only (`prune=True` at `allocator.py:5454`) | not OR-Tools, stays |
| `allocator.py:3797-3816` | enumerate, empty-menu error, core-budget assert | **Stage 5** |
| `allocator.py:3883-4030` | `_enumerate_core_divisions`, `_tiling_candidates` | **Stage 5** |
| `allocator.py:4187-4194` | `tileable`, per-buffer residency verdicts | **Stage 5** calls Stage 3's rule |
| `allocator.py:4226-4267`, `4710-4828` | clone eligibility, clone menus, `matches` | Stage 6 |
| `allocator.py:4467-4471` | `untiled_bar` | **Stage 5** calls Stage 3's rule |
| `plan_solver.py:1101-1128` | `excluded`, `record_exclusions` | **Stage 5** (table verdict) |

The rest of the model (gates, merges, lifetimes, coupling, placement, extraction) and the
rest of `_build_cd_bound_buffers` are Stage 6's and Stage 7's.

### Creation order today

Variables and constraints are appended, so the order below is the proto. `V` is a
variable and `C` a constraint.

1. `_wrap` runs once per buffer (`ilp_solver_ortools.py:764`) in list order: eligible
   input clones, the LX prefetches (`allocator.py:4250`), then op outputs in `mem_usage` order, which is
   `graph.operations` order (`allocator.py:4269`,
   `torch_spyre/_inductor/scratchpad/utils.py:242`), then relayout copies
   (`allocator.py:4490`). Per buffer:
   - `V in_buffer_n`, `V off_n` in `[0, max(0, M-1)]`, then `V merge_p_n` per in-place
     parent (lines 298-311).
   - With one table entry: `new_constant(0)`, then `new_constant(1)`, `new_constant(0)`,
     and no constraint (lines 462, 477).
   - With more: `V div_n`, `V eff_size_n`, `V core_cost_n`, `V occ_n`; per split key in
     first-seen order `V split_n_key` in `[1, sencores]` then `C element`; per tile key
     `V tile_n_{o|r}dim` then `C element`; then `C element` on `eff_size`, `cores`,
     `core_cost`, in that order (lines 480-516).
2. `_add_tiling`, only when some entry anywhere is tiled. For every divided buffer,
   copies included: `V loop_id_n`, `C element`, `V tiled_n`, `C` reified `!=`, `C` reified
   `==` (lines 961-973). This is contiguous, and all of it precedes `_add_tiling_groups`.
3. Groups, per-edge relations, cut guard, nest ends and starts (lines 975-1027).
4. `_add_lifetime_ends`, then `_add_inplace_relaxation`, which runs through
   `_add_no_overlap_2d` (lines 1370-1371).
5. `_add_core_division`: one `in_buffer == 0` per forced buffer in buffer order
   (line 1796), then per buffer with an `untiled_bar` either `in_buffer == 0` or
   `add_implication(in_buffer, tiled)` (lines 1797-1804), then every `constrain_residency`
   (line 1806).
6. `_constrain_relayout_copies` (line 1373).
7. `_tile_count_terms`: `V tile_count_n`, `C element` per buffer whose tile count varies
   (line 1427). It always runs, even under a cost expression.
8. The objective (lines 1442-1523).

So op-level content sits in four separated ranges of the proto, which this plan calls
**S1** (step 1 minus `merge_*` and copies), **S2** (step 2 minus copies), **S3** (the
first two loops of step 5, minus copies) and **S4** (step 7).

### How the table is built today

`enumerate_work_division_candidates(op, max_cores, tiling=None)`
(`torch_spyre/_inductor/work_division.py:1051`) is the cross product of each axis's
`factor_domain`, filtered by `WorkDivisionContext.is_legal` (`work_division.py:901`).
Production calls it only from `CoOptimizingAllocator._enumerate_core_divisions`:

- Only a `ComputedBuffer` whose `data` is `Pointwise` or `Reduction` is enumerated.
  Anything else keeps `_fixed_core_division(op)` (`allocator.py:3895-3898`).
- The untiled call comes first (`allocator.py:3900`). If it raises `Unsupported` the op
  takes its fixed division and **no tiling is tried**, even with tiling on.
- `_tiling_candidates` runs after that call, so its own `Unsupported`
  (`allocator.py:2860`) cannot fire for such an op.
- It returns `[TileSpec()]` unless `config.auto_coarse_tiling` and the solver's
  `tilings_from_menu()`, and also for an op with `loop_info`, an op in a prescribed
  region, or a restickify (`allocator.py:4001-4009`). Otherwise it returns
  `enumerate_tile_options(op, readers=...)`, untiled first
  (`wsr/enumerate_tilings.py:480`), filtered by `is_clean` and then
  `_drop_read_distance_violations`, which can remove the untiled option.
- Each non-empty tiling resolves its loop vars through `try_resolve_tile_axis_loop_vars`
  and calls `enumerate_work_division_candidates(op, max_cores, tiling=tiling)`, on the
  per-tile frame. A failed resolve or an `Unsupported` skips that tiling
  (`allocator.py:3920-3940`).
- Each candidate goes through `_core_division(op, candidate, tiling, tile_splits)`, and
  duplicates are removed on sorted splits, `reduction_syms` and tiling, first seen wins
  (`allocator.py:3941-3955`). Table order is this order.
- An empty result falls back to the fixed division with `enumerated` false
  (`allocator.py:3956-3958`).

The table is not always enumerated. A fixed reason gives one entry
(`allocator.py:3738-3785`). A clone's menu is derived from its consumers' menus and views
(`allocator.py:4730-4828`), so it needs them first, though it is created first. A relayout
copy has one entry (`plan_solver.py:896`). The direct-read rejection loop can replace a
menu after buffers are built (`allocator.py:3145-3178`).

### Surprises

- **Flag rules are already inside the table.** All sixteen functions in
  `collect_work_division_constraints` (`work_division_constraints.py:135-152`) feed
  `is_legal`. In Phase 2, sticks are pinned, so nothing conditional reaches the proto:
  today's model has no stick variable at all.
- **Residency is per buffer, not per division.** `BUFFER_RESIDENCY_RULES` give one reason
  per buffer (`residency_rules.py:296`). The only division-dependent ban is `untiled_bar`.
  It is posted as an implication on `tiled`, not as a table constraint on the index.
- **Today's model has constants as proto variables.** `new_constant` interns by value and
  appends a variable on first use. `only_enforce_if(True)` and `add_bool_or([x, True])`
  materialize one the same way, and `add(True)` too (experiment on OR-Tools 9.15.6755).
  A single stray literal renumbers every later variable.
- **Every constraint name is empty except intervals.** `new_optional_interval_var` writes
  `x_n` and `y_n` into the constraint's name. `with_name` also writes to the proto.
- **Unsized ops are in the model.** `mem_usage` gives them `size == -1`
  (`scratchpad/utils.py:261`), and `ceil_div(-1, 128)` is 0, so they have zero units.
- **Tests patch these names.** `allocator.enumerate_work_division_candidates` and
  `allocator._fixed_core_division` (`tests/inductor/test_work_division.py:3635`),
  `_tiling_candidates` (`tests/inductor/test_solver_auto_coarse_tiling.py:887`),
  `_division_map` (`tests/inductor/test_scratchpad_use.py:2263`,
  `tests/inductor/test_work_division_hint.py:823`) and `_enumerate_core_divisions`
  (`tests/inductor/test_work_division.py:3482`). The earlier attempt moved this code and
  had to move about 40 patches (feasibility report, Phase 1, deviation 4).

## Design

### Files

New package `torch_spyre/_inductor/scratchpad/lx_allocator/family/`:

- `base.py`: `Model`, `Encoder`, `Provenance`, `Outcome`, `Pruned`. No solver import.
- `division_table.py`: `DivisionTable`, `TableRecorder`, `build_division_tables`,
  `stick_of`, `read_stick`. No solver import. Stage 8 uses it too.
- `ortools_encoder.py`: `CpModel`, `OpVars`, `OrToolsEncoder`. It imports `cp_model` under
  the same guard as `ilp_solver_ortools.py:120-132`, and `OrToolsEncoder.__init__` raises
  the same `ImportError` when it is missing.

### `base.py`

`Model` and `Encoder` are fully defined here. Stage 7 adds attributes, not a new base.
`RuleTag` is Stage 3's, imported from `rules.py`.

```python
class Outcome(Enum):
    POSTED = auto()    # reached the proto; `constraint` is its index
    PRUNED = auto()    # decided by pins: applied to the table, nothing posted
    VACUOUS = auto()   # decided by pins: condition false or already satisfied
    VIOLATED = auto()  # decided by pins: false; an empty bool_or was posted

@dataclass(frozen=True)
class Provenance:
    subject: Any                    # OpNode | Port | DataEdge | str, never a solver object
    rule: RuleTag
    outcome: Outcome
    constraint: Optional[int] = None
    pin: Optional[Pin] = None       # the pin behind it, for the explanations work

@dataclass(frozen=True)
class Pruned:
    subject: OpNode
    rule: RuleTag
    tiling: TileSpec                # the frame the rule applied in
    entries: int                    # table entries removed
    axis: Any = None                # set on an axis-domain record
    factors: tuple[int, ...] = ()   # the factors removed from that axis's domain
    sample: tuple[tuple[tuple[str, int], ...], ...] = ()  # at most PRUNED_SAMPLE entries

class Model(ABC):
    def __init__(self) -> None:
        self.provenance: list[Provenance] = []
        self.pruned: list[Pruned] = []
        self.symbols: dict[str, Any] = {}  # Stage 7 adds boxes and capacity, with defaults

class Encoder(ABC):
    ENCODES: ClassVar[Restriction] = Restriction.NONE

    def __init__(self, cost_model: Optional[CostModel] = None) -> None:
        self.cost_model = cost_model

    @abstractmethod
    def encode(self, graph: SpyreOptimizationGraph,
               restrictions: Optional[RestrictionOverrides] = None) -> Model: ...

    @abstractmethod
    def read_back(self, solution: Any, graph: SpyreOptimizationGraph) -> None: ...
```

`solution` is `Any` because `Solution` first exists in Stage 7, which tightens it. `Model`
has no abstract method: its subclasses differ only in payload.

`CpModel` wraps the OR-Tools object. Stages 6 and 7 read `ops` and `units`.

```python
class CpModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.cp = cp_model.CpModel()
        self.ops: dict[OpNode, OpVars] = {}
        self.units: list[OpVars] = []       # every _add_unit result, in creation order
        self.columns: dict[str, tuple[OpVars, list[int]]] = {}  # today's buffer_map
        self.tile_terms: list[cp_model.IntVar] = []             # for the ladder

    proto: "cp_model_helper.CpModelProto"   # property: self.cp.proto

    def tag(self, subject, rule: RuleTag, start: int) -> None: ...
```

### `division_table.py`

```python
PRUNED_SAMPLE = 8

@dataclass
class DivisionTable:
    name: str                         # today's buffer name
    node: Optional[OpNode]            # None on a relayout group (Stage 6)
    entries: list[CoreDivision]       # table order is private to the families
    enumerated: bool                  # the whole legal space (today's _DivisionMap)
    residency: Optional[str] = None   # why the node may not reside under any entry
    untiled_bar: Optional[str] = None # why it may not reside while untiled
    pruned: list[Pruned] = field(default_factory=list)
    decided: list[Provenance] = field(default_factory=list)  # rules pins decided
    conditional: list[SplitRule | StickTable] = field(default_factory=list)  # free sticks

    split_keys: list      # property: dict.fromkeys over entries' splits, first seen
    tile_keys: list[tuple[int, bool]]  # property: (host_dim, is_reduction), first seen
    tileable: bool        # property: not an InputNode and some entry is tiled

    def per_core_units(self, size_units: int) -> list[int]: ...
    def cores_used(self) -> list[int]: ...
    def core_cost(self) -> list[int]: ...

class TableRecorder:
    def __init__(self, node: OpNode) -> None: ...
    def reject(self, splits, check: str, sources: Sequence[str], tiling: TileSpec): ...
    def drop_factors(self, axis, factors, sources: Sequence[str], tiling: TileSpec): ...
    def skip_tiling(self, tiling: TileSpec, reason: str) -> None: ...
    def records(self) -> list[Pruned]: ...

def build_division_tables(
    graph: SpyreOptimizationGraph, *, coarse_tiling: bool, max_cores: int,
    restrictions: Optional[RestrictionOverrides] = None,
) -> dict[OpNode, DivisionTable]: ...
```

`per_core_units` is `ceil_div(size_units, cd.output_partition *
cd.tiling.output_tile_count)`, the arithmetic of `_wrap` and `per_core_size`
(`plan_solver.py:395-401`). It reads `CoreDivision`s because the entries of an LX prefetch or a
copy have no `Axis`, so Stage 1's `per_core_units` cannot take them, and
`test_per_core_units_match_the_graph_helper` holds the two equal on op tables.
`cores_used` is `cd.cores_used`, `core_cost` is the sum of squared split factors
(`ilp_solver_ortools.py:446-448`).

Tables cover running nodes only, a `RestickifyNode` and an `OpaqueNode` included, because
each is a buffer today (`allocator.py:4269`). A `RestickifyNode`'s pinned step gets
nothing. LX prefetches are Stage 6's: `prefetch_menu(input_node, tables)` reads these tables.

`build_division_tables` works in this order. It takes no solver and is safe to call again
after new pins are added, which Stage 6's `settle_deferred_reads` relies on.

1. When `coarse_tiling`, compute prescribed ops and readers from `graph.ir.operations`,
   exactly as `allocator.py:3701-3713`.
2. For each running node in `graph.nodes`:
   - with a `split` pin, `enumerated` is false and the table has one entry. If
     `pin.value.seeds` is set, the entry is what `seed_aligned_division` returns for the
     ops `graph.node(s).op`. Otherwise, or on `None`, it is
     `division_from_work_split(node, pin.value.split)`, passed through
     `_legal_fixed_division(op, [division], pin.value.reason)`, so an illegal pin raises
     `Unsupported` as today. A `committed` pin skips that check, since only the joint path
     runs it. Two `split` pins that disagree raise `Unsupported`. A `Provenance` with
     `pin=` goes in `decided`.
   - with an `opaque` pin, the one entry is `_fixed_core_division(op)` and `enumerated` is
     false, as `allocator.py:3895-3898`.
   - otherwise `enumerate_op_divisions(op, max_cores, tilings=lambda:
     tiling_candidates(...), recorder=..., skip_constraints=...)`.
   - no entries raises `Unsupported(f"{name}: no legal core-division candidates.")`, and
     the budget assert stays, both as `allocator.py:3801-3816`.
3. Apply rules (see "Sticks and flags"). Pruning a table to empty raises the same
   `Unsupported`, with the rule added to the message.
4. Set `residency`, in the order of `MemoryPlanSolver.excluded`: Stage 3's
   `residency_reason(node, graph, decides_relayouts=True, tileable=table.tileable)`, else
   the `min footprint ... > LX capacity` reason computed in bytes from
   `plan_solver.py:1108-1111`. Set `untiled_bar` from `untiled_bar(node, graph)`, only when
   `table.tileable`, as `allocator.py:4467-4471`. Tables come back in `graph.nodes` order.

Stage 7's `read_back` reads `table.residency` and `table.untiled_bar` for
`node.spill_reason`. They are the joint family's verdict. Stage 8 asks `residency_reason`
itself, with `decides_relayouts=False`, and does not read these fields.

### Extractions from `allocator.py`

These stay in `allocator.py` as module functions, and the methods become delegates.
Nothing moves to another module, so every patch listed under "Surprises" keeps working.
`division_table.py` imports them inside the function, which avoids a cycle.

```python
def enumerate_op_divisions(
    op, max_cores, tilings: Callable[[], list[TileSpec]],  # after the untiled call
    *, recorder=None, skip_constraints: frozenset[str] = frozenset(),
) -> tuple[list[CoreDivision], bool]: ...

def tiling_candidates(
    op, max_cores, *, tilings_from_menu: bool, prescribed_ops: Collection[str],
    readers: Sequence[Operation], recorder=None,
) -> list[TileSpec]: ...

def seed_aligned_division(
    op, seeds: list[Operation], menu: Callable[[], list[CoreDivision]], max_cores,
    prep_cache: dict,
) -> Optional[CoreDivision]: ...
```

`CoOptimizingAllocator._enumerate_core_divisions(op, max_cores)` keeps its signature and
calls `enumerate_op_divisions(op, max_cores, lambda: self._tiling_candidates(op,
max_cores))`, so `patch.object(CoOptimizingAllocator, "_tiling_candidates", ...)` still
applies. `_seed_aligned_division` keeps its signature and passes its `self.prune` branch
(`allocator.py:3855-3863`) as `menu`. `build_division_tables` passes the enumeration, and
one `prep_cache` per call. Stage 13 deletes the delegates with the class; the functions
stay in `allocator.py`.

### Recording prunes without changing the enumeration

All changes are additive and default to today's behaviour.

```python
# work_division_constraints.py
class ConstraintResult:
    ...
    blocked_by: dict[Symbol, list[str]]   # dim -> constraint function names
    allowed_by: dict[Symbol, list[str]]

def collect_work_division_constraints(ctx, skip=frozenset()) -> ConstraintResult: ...

# work_division.py
class WorkDivisionContext:
    def first_violation(self, splits) -> Optional[str]: ...
    def violation_sources(self, splits, check: str) -> list[str]: ...
    def removed_factors(self, v) -> tuple[list[int], list[str]]: ...

def work_division_context_for_op(op, max_cores=None, tiling=None, skip_constraints=frozenset()): ...

def enumerate_work_division_candidates(
    op, max_cores, tiling=None, *, recorder=None, skip_constraints=frozenset()
) -> list[dict[Symbol, int]]: ...
```

- `WorkDivisionContext.first_violation` runs `is_legal`'s six checks in their order and
  returns the failing one's tag: `factor_domain`, `core_budget`, `one_reduction`,
  `span_cap`, `split_domains` or `span_floor`. `is_legal` becomes a test that it returns
  `None`. It is private to this recording.
- With `recorder=None` the function runs today's comprehension. With a recorder it runs
  the same product and reports rejects, and only rejected combinations pay for attribution.
  An entry that breaks two rules is charged to the first in `is_legal` order, so `entries`
  totals are a partition of the rejects, not a per-rule census.
- Before the product, the recorder receives `removed_factors` for each axis: divisors that
  `factor_domain` dropped through `allowed_splits` or a span floor. Those never become
  entries, so they are axis-domain records (`axis`, `factors`, `entries=0`).
- `skip_constraints` exists because the parent plan lets a test remove a flag through
  `restrictions`. The rule is inside enumeration, so a removal has no effect unless the
  constraint function is skipped. `build_division_tables` computes the skip set per node:
  the names in `CONSTRAINT_RULES` whose `Restriction` is on the node's type and absent
  from `restrictions_for(node, restrictions)`. It is empty in production.

Tags map to rules through `CHECK_RULES` (`factor_domain` to `Always.SPLIT_DIVIDES`,
`core_budget` to `Always.CORE_BUDGET`, `one_reduction` to `Always.ONE_REDUCTION_SPLIT`,
`span_cap` to `Always.SPAN_CAP`, `span_floor` to `Always.SPAN_FLOOR`). `split_domains`
maps through Stage 3's `CONSTRAINT_RULES`, which names all sixteen functions, so every
prune has a rule.

`PRUNED_SAMPLE` bounds the entries stored per record (Deviation 8). `tiling_candidates`
reports `is_clean` and read-distance drops, and `enumerate_op_divisions` reports skipped
tilings, through `skip_tiling`.

### `OrToolsEncoder`

```python
class OrToolsEncoder(Encoder):
    ENCODES = (Restriction.WHOLE_STICKS | Restriction.PADDED_REDUCTION_UNSPLIT
               | Restriction.MATMUL_STICKS | Restriction.POINTWISE_STICKS
               | Restriction.INDIRECT_ACCESS | Restriction.ENCODED_K_UNSPLIT
               | Restriction.WINDOW_UNSPLIT | Restriction.CONV_SPATIAL_UNSPLIT
               | Restriction.TOPK_SPLITS | Restriction.KEEP_BY_INDEX_SPLITS
               | Restriction.PLAIN_REDUCTION_UNSPLIT)

    def __init__(self, cost_model=None, *, coarse_tiling: bool = False) -> None: ...

    def encode(self, graph, restrictions=None) -> CpModel:
        self._begin(graph, restrictions)
        self._build_tables()                 # sets self._tables; no variables
        self._edges = self._settle_tables()  # hook; Stage 6: settle_deferred_reads
        self._encode_op_vars()               # S1: _add_unit per LX prefetch and op
        self._encode_relayout_copies()       # hook; Stage 6: _add_unit per group
        self._encode_loop_columns()          # S2
        self._encode_tiling()                # hook; Stage 6
        self._encode_in_place()              # hook; Stage 6, Stage 7's tie inside
        self._encode_placement()             # hook; Stage 7, Stage 6's capacity stub
        self._post_residency()               # S3
        self._encode_restickifies()          # hook; Stage 6: no proto in Phase 2
        self._encode_read_gates()            # hook; Stage 6
        self._encode_tile_counts()           # S4
        self._bind_symbols()
        self._encode_sticks()                # free sticks only; nothing in Phase 2
        self._encode_objective()             # hook; Stage 6, and Stage 7 runs the stages
        return self._m

    def read_back(self, solution, graph) -> None:
        raise NotImplementedError("Stage 7")
```

All eleven flags count as encoded: in Phase 2 the table obeys them through enumeration, and
step 3 of `build_division_tables` verifies it. The sequence is today's order, not the parent
plan's op-then-edge-then-graph order (Deviation 2). Each hook is an empty method in this PR,
with a comment giving the lines it reproduces, and `_settle_tables` returns `None`.
`_settle_tables` runs before any variable because a settled menu changes `div_n`.

| hook | reproduces | owner |
|---|---|---|
| `_settle_tables()` | `allocator.py:3145-3183` | Stage 6 |
| `_encode_prefetches()` | the LX prefetches' `_wrap` calls, called first by `_encode_op_vars` | Stage 6 |
| `_declare_in_place_vars(v)` | lines 308-311, called by `_add_unit` after `off_` | Stage 6 |
| `_encode_relayout_copies()` | the copies' `_wrap` calls | Stage 6 |
| `_encode_tiling()` | lines 975-1172, then `_add_lifetime_ends` | Stage 6 |
| `_encode_in_place()` | lines 1633-1658 | Stage 6 |
| `_encode_placement()` | lines 1660-1664 and `_add_no_overlap_2d` | Stage 7 |
| `_encode_restickifies()` | nothing today: the joint path posts no restickify constraint | Stage 6 |
| `_encode_read_gates()` | line 1806, then `_constrain_relayout_copies` | Stage 6 |
| `_encode_objective()` | lines 864-900 and the ladder's lowering | Stage 6 |

`OpVars` is one unit's bundle and implements Stage 4's `DivisionOwner`, which is what the
printer asks of a wrapper.

```python
@dataclass(eq=False)
class OpVars:
    table: DivisionTable
    name: str
    units: int
    boundary: BufferType
    in_buffer: cp_model.IntVar
    offset: cp_model.IntVar             # Stage 7 moves this behind its placer
    division: Optional[cp_model.IntVar] = None  # interned constant 0 on a one-entry table
    # Defaulted, and set after the in-place hook: eff_size, core_cost, cores (a variable or,
    # on a one-entry table, an int), cores_used, cp_core_divs, cp_core_divs_raw,
    # cp_tile_counts, cp_tile_counts_raw. Set by _encode_loop_columns: loop_id, tiled.

    selector_name: str       # property: division_symbol(name).name
    split_symbols: Mapping   # property: {key: Sym.split(name, key)} over table.split_keys
    tile_symbols: Mapping    # property: {key: Sym.tile(name, *key)} over table.tile_keys
    menu_size: int           # property: len(table.entries)

    def splits_at(self, i: int) -> Mapping: ...   # table.entries[i].splits
    def division_is(self, i: int) -> cp_model.IntVar: ...
```

`division_is` is a port of lines 520-525 with a per-instance dict. For a one-entry table it
returns the interned constants 1 and 0 as lines 477-478 do. Those are proto constants on
purpose: Stage 6 passes them to `add_bool_or` (line 1602), and byte identity needs them.
They are distinct from Python-level folding of pinned sticks, below.

- `_begin` creates `self._m`, reads `config.sencores`, and sets `self._capacity_units` to
  `graph.capacity // graph.alignment`, because `Model.capacity` is Stage 7's.
- `_add_unit(table, *, size_units, boundary) -> OpVars` is today's `_wrap` for one buffer:
  "Creation order today", step 1, with today's name strings. It calls
  `_declare_in_place_vars(v)` after `off_`, appends to `m.units`, and sets
  `m.ops[table.node]` when the table has a node. `size_units` is `ceil_div(size,
  alignment)`. Stage 7 adds its `declare` call and box here.
- `_encode_op_vars` copies each table's `pruned` and `decided` onto the model, after
  `_settle_tables` because that can rebuild, then calls `_encode_prefetches`, then
  `_add_unit` per table, with `boundary` as `allocator.py:4475-4477`.
- `_encode_loop_columns` is lines 944-973: it interns nests by `level_counts` over
  `m.units` in order, returns if there are none, then creates columns for every unit.
- `_post_residency` first posts `in_buffer == 0` for each unit with `table.residency`,
  then for each unit with `untiled_bar`: `add_implication(in_buffer, tiled)` if
  `table.tileable`, else `in_buffer == 0`. Today's branch is `sb.nest_end is None`.
  `position` (lines 1134-1141) puts a buffer in a run exactly when it is an op output
  with some tiled entry, which is `table.tileable`.
- `_encode_tile_counts` skips a table whose counts are all equal, and fills `tile_terms`.
- `_bind_symbols` binds `Sym.resident(v.name)` to `in_buffer` through `symbol` and calls
  Stage 4's `bind_division_owner(m.symbols, m.columns, v)` per unit, which is lines 843-862.
  `cost_expr_to_cpsat(m.cp, m.symbols, m.columns, expr, strict=True)` then works.
- `_cost_domains() -> CostDomains` has one `NodeDomain(v.name, tuple(v.table.entries))` per
  unit. `readers` stays empty because a listing is a gate, so Stage 6 fills it when it
  extends this method with the `RelayoutOffer`s.
- `decision_variables() -> list` is `in_buffer` then `division` per unit, as lines 406-411
  and 619-620, without the offset. Stage 6 adds `merge_*` and served literals.
- `_menus` is `{node: table.entries}`, for Stage 6.

Variable names stay today's literal strings. Names in `symbols` come from Stage 4's `Sym`,
except `division_<n>`, named by `plan_solver.division_symbol` inside the encoder. It is the
one exposed index, which `RelayoutCharge` needs as today (`plan_solver.py:487`), and it stays
inside `Model.symbols`. Pins name divisions by value, for the non-enumerating direction.

### `post` and provenance

```python
def post(self, subject, rule, expr, when=True) -> Optional[cp_model.Constraint]: ...
def post_in(self, subject, rule, x, values, when=True) -> Optional[...]: ...
def post_table(self, subject, rule, xs, rows, when=True) -> Optional[...]: ...
def symbol(self, name: str, variable: Any) -> None: ...
```

- `expr` is a `bool`, a bounded linear expression, or a callable `cp -> Constraint`. The
  callable covers calls whose proto differs from a linear constraint:
  `add_implication` gives `bool_and` under an enforcement literal, and `add_element` has no
  expression form. It is a callable so that nothing is created when the rule folds.
- `when` is `True`, `False`, a literal, or a list of those. Booleans in a list fold out.
  `when` false, `expr` true, or an empty fold: **nothing touches the proto**.
  `Provenance(outcome=VACUOUS)`.
- `expr` false under `True`: `cp.add(False)`, which adds an empty `bool_or` and no
  variable (experiment). Under a literal: `add_bool_or([lit.negated()])`.
  `outcome=VIOLATED`.
- Otherwise: build, then `only_enforce_if(when)` unless `when is True`, then record
  `ct.index` with `outcome=POSTED`.
- `post` never calls `with_name`. Reading `.index` leaves `str(proto)` unchanged and
  `with_name` changes it (experiment). Provenance lives only on the `Model`.
- `post_in` uses `add_linear_expression_in_domain(x, Domain.from_values(values))`. An int
  `x` folds to `post(subject, rule, x in values, when)`.
- `post_table` folds pinned columns out of the rows rather than calling `new_constant`.
  All columns pinned folds to `post(..., tuple(xs) in rows, when)`. No row left folds to
  false.
- `CpModel.tag` tags constraints created outside `post` from index `start` to the end, for
  Stage 7's placers and the printer's helper constraints.

The invariant from this stage on: every index in `proto.constraints` has exactly one
`Provenance`, and every `Provenance.constraint` is in range. Structural constraints count:
column ties use the tag `Always.TABLE_COLUMN`.

### Sticks and flags

A stick is either a pinned value or a variable. Stage 2 pins every stick through Phase 2, so
all of this folds, and no call reaches `cp_model` for a real graph until Stage 11. A pinned
value is `stick_dims(layout) or (SPARSE_STICK,)`: dims stay the rule language, and QFP8WT
holds two.

```python
def stick_of(port: Port, graph) -> Optional[tuple[int, ...]]: ...     # division_table.py
def read_stick(edge: DataEdge, graph) -> Optional[tuple[int, ...]]: ...
def _stick_candidates(self, port: Port) -> list[tuple[int, ...]]: ... # ortools_encoder.py
def _stick_var(self, port: Port) -> cp_model.IntVar: ...
def _stick_is(self, port: Port, dim: int) -> "bool | IntVar": ...
```

`stick_of` reads the `stick` pin on an out port. For an in port it is the stick of
`port.current_layout()`, which walks the pinned steps, provided the producer is pinned, and
stops ahead of the step a `RestickifyNode` realizes. `read_stick(edge, graph)` stays the stick
of `edge.read_layout()`, for edges. `None` means free. `_encode_sticks` creates `_stick_var`, a private
candidate index, and one literal per candidate, only for free ports, after S4, so they
cannot disturb S1 to S4. `_stick_candidates` gives each host dim as a one-dim candidate
here, and Stage 11 replaces it with `StickCandidate.dims`. `_stick_is(port, dim)` is `dim in
stick_of(port, graph)` when pinned, else the OR of the candidates whose dims contain `dim`.

Rule application, in step 3 of `build_division_tables`. Split rows apply to enumerated
tables only: a pinned entry stays, as today, and `check_decisions` judges it. Rows are per
frame, so step 3 loops over the tilings in the table, in first-seen order, and each row
touches only that tiling's entries. For each tiling, the rows are
`splits_while_stuck(node, flag, tiling)` for each member of `restrictions_for(node,
restrictions)`, then `fact_splits(node, tiling)` and `always_splits(node, tiling)`. Each
`SplitRule` is compared on `cd.splits.get(rule.axis.symbol, 1)`:

- `rule.port is None`, or pinned with `is_stuck(rule)`: prune this tiling's entries whose
  factor is outside `factors`. One `Provenance(outcome=PRUNED)` in `decided`, one `Pruned`
  record with this `tiling` if any entry went.
- pinned and not stuck: one `Provenance(outcome=VACUOUS)`. No effect.
- free: added to `table.conditional`.

Then, once per node, each `StickTable` from `stick_tables(node, flag)`:

- all columns pinned: a row holds if `row[c] in stick_of(ports[c], graph)` for each column.
  Some row holds gives `VACUOUS`. None gives `VIOLATED` in `decided`, and the encoder posts
  it through `post(node, flag, False)`. A pinned stick breaking a stick rule is a bug that
  Stage 3's `check_decisions` would already have reported, and failing loudly is correct.
- any free column: added to `table.conditional`.

`_encode_sticks` walks `table.conditional`. A `SplitRule` becomes `post_in(node, rule.rule,
column, rule.factors, when=self._stick_is(rule.port, rule.dim))`. `column` is
`cp_core_divs[rule.axis.symbol]`. A key with no column means the axis is never split, so
`x` is 1 and `post_in` folds. A column spans every tiling, so `_encode_sticks` asserts that
a table with a conditional row has one tiling; Stage 11 pins sticks under
`AUTO_COARSE_TILING`, so the two do not meet. A `StickTable` becomes `post_table` over the
candidate indices, with pinned columns folded and each row expanded to the candidates that
contain its dim.

**Effect on the proto.** With every stick pinned, a flag adds one or two `Provenance`
entries and no variable, no constraint, and no constant. Pruning runs before any variable
exists, so a prune that removes nothing changes nothing. A prune that removes something
changes `div_n`'s domain and is a real rule disagreement, which the gate reports by rule.

### Ported rules

| old | new | how | deleted by |
|---|---|---|---|
| `_LifetimeBufferWithCpVars.__post_init__` (298-301), `_CoreDivisionBufferWithCpVars.__post_init__`, `_division_is` | `_add_unit`, `OpVars` | copied | Stage 13 |
| `_add_tiling` (944-973), `_add_core_division` (1795-1804), `_tile_count_terms` | `_encode_loop_columns`, `_post_residency`, `_encode_tile_counts` | copied | Stage 13 |
| `decision_variables` (406-411, 619-620) | `OrToolsEncoder.decision_variables` | copied | Stage 13 |
| `_minimize_cost_expr` (843-862) | `_bind_symbols` | reused: Stage 4's `bind_division_owner` | Stage 13 |
| `excluded` | `build_division_tables` step 4 | copied | Stage 13 |
| `_enumerate_core_divisions`, `_tiling_candidates`, `_seed_aligned_division` | module functions in `allocator.py` | moved, methods delegate | Stage 13 deletes the delegates |
| `sym_core_divs` and `sym_tile_counts` key order | `split_keys`, `tile_keys` | copied, without the `division_space` branch (`plan_solver.py:466`), which only SA reaches | Stage 13 |

## Changes

| file | change |
|---|---|
| `torch_spyre/_inductor/scratchpad/lx_allocator/family/`: `__init__.py`, `base.py`, `division_table.py`, `ortools_encoder.py` | new |
| `torch_spyre/_inductor/scratchpad/allocator.py` | three extractions, methods delegate |
| `torch_spyre/_inductor/work_division.py` | `WorkDivisionContext.first_violation`, recorder, `skip_constraints` |
| `torch_spyre/_inductor/work_division_constraints.py` | `blocked_by`, `allowed_by`, `skip` |
| `AGENTS.md` | one line: a change to how `CpSatLayoutSolver` posts its model must change `OrToolsEncoder` in the same PR |
| `tests/inductor/lx_sections.py` | new test helper: build today's model, project, diff |
| `tests/inductor/lx_replay_graph.py` | new test helper: a graph and tables from buffers |
| `tests/inductor/lx_shadow_plugin.py` | new pytest plugin and `Observer`: live comparison at each solve |
| `tests/inductor/test_lx_family_base.py`, `test_lx_division_table.py`, `test_lx_ortools_encoder_op.py`, `test_lx_encoder_sections.py`, `test_lx_live_parity.py`, `test_work_division_recording.py` | new |
| `tests/configs/torch_spyre_tests/inductor/test_lx_*_config.yaml`, `test_work_division_recording_config.yaml` | one per new test file, labels `[unit, regression, trunk]`, `unlisted_test_mode: mandatory_success` |

Each new file carries the 14-line license header. Tests use `import regex`, never `re`.

Commits:

1. `base.py` and the `CpModel` shell with `post`, `post_in`, `post_table`, `symbol`, `tag`,
   with unit tests on a bare encoder. Nothing calls them.
2. The three allocator extractions. Stage 0 harness diff must be empty.
3. Recording in `work_division.py` and `work_division_constraints.py`, with
   `test_work_division_recording.py`. Stage 0 harness diff must be empty.
4. `division_table.py`, tested on hand-built graphs and on mock ops as
   `tests/inductor/test_work_division.py` builds them.
5. `OrToolsEncoder`: S1 to S4, residency, symbols, cost domains, hooks.
6. Sticks and flags, with the free-stick path solved in a unit test.
7. `lx_sections.py`, `lx_replay_graph.py`, `test_lx_encoder_sections.py`,
   `lx_shadow_plugin.py`, `test_lx_live_parity.py`, the `AGENTS.md` line, and the gate.
   This commit is probe P4: run the live arm on the `fake` zoo at both configs before Stage
   6's design freezes. A hand port from a graph built without the allocator has never been
   compared on a real compile, so the first section diff shows whether node order, prefetch
   order and menus line up, which Stage 6's and Stage 7's whole-proto claims depend on.
   P4 has not been run.

## Interfaces

Consumes.

| symbol | signature | from Stage | used for |
|---|---|---|---|
| equivalence harness | `lx_gate.sh`, `lx_corpus.py timing`, tiers `intree`, `fake`, `unit-*`, `failure`; `record_compile(name, cell, out, *, device, observers)`, `Observer.on_solve_begin(allocator, solver, ir, buffers, cost_expr)`, `Recorder.begin_solve`, `.record_stage(..., provenance=)`, `proto_text`, `provenance_record`, `corpus_coverage`, `FIXED_GRAPHS`, `DEFAULT_CELL`, `TILING_CELL`, `CARD_ONLY` | 0 | gates |
| `SpyreOptimizationGraph` | `nodes`, `alignment`, `capacity` (bytes), `pins_on(target, what)`, `pin(target, what, value, source)` | 1 | everything |
| graph facts | `graph.ir`, `graph.node(name)` | 2 | steps 1 and 2 |
| node order | `graph.nodes`: leaf `InputNode`s, then ops in `graph.operations` order, one node per op including unsized ones | 2 | S1 order |
| `build_graph` | keyword defaults | 2 | live arm |
| `OpNode` | `name`, `size` (total device bytes; unsized may be `-1` or `0`), `op`, `runs`, `graph_output`, `deferred_direct_read`, `in_ports`, `out` | 1, filled by 2 | tables |
| `InputNode`, `RestickifyNode`, `OpaqueNode` | node types; `RestickifyNode` and `OpaqueNode` run | 1 | tables |
| `Port`, `InPort.edge`, `OutPort.edges`, `DataEdge.read_layout()`, `Pin`, `WorkSplitPin`, `Restriction`, `PortFact`, `restrictions_for`, `RestrictionOverrides` | as Stage 1 | 1 | sticks, pins, flags |
| `Axis` | `Axis.symbol` keys `CoreDivision.splits` | 1 | rule prune |
| `stick_dims`, `SPARSE_STICK` | `(layout) -> tuple[int, ...]`; `-1` | 1 | `read_stick` |
| `split` pin | `Pin(node, "split", WorkSplitPin(split, reason, seeds=()), source)`, on every op where `_division_map` has a reason (nine strings over the eight branches at `allocator.py:3738-3770`), with `seeds` on fp8 consumers | 2 | table |
| `opaque`, `stick` pins | `Pin(node, "opaque", type_name, "hazard")`; on `node.out`, value `stick_dims(layout) or (SPARSE_STICK,)` | 2 | table, sticks |
| `division_from_work_split`, `work_split_from_division` | `(node, split) -> CoreDivision`, `(node, division) -> WorkSplit` | 2 | pinned entries, live arm |
| `SplitRule`, `StickTable`, `is_stuck` | fields `rule`, `axis`, `factors`, `port`, `dim`; `rule`, `ports`, `rows`; `(rule) -> bool` | 3 | rules |
| `splits_while_stuck`, `fact_splits`, `always_splits` | `(node, flag, tiling=None)`, `(node, tiling=None)`, both `-> list[SplitRule]` | 3 | rules |
| `stick_tables` | `(node, flag) -> list[StickTable]` | 3 | rules |
| `residency_reason`, `untiled_bar` | `(node, graph, *, decides_relayouts, tileable) -> str \| None`, called with `decides_relayouts=True, tileable=table.tileable`; `(node, graph) -> str \| None` | 3 | `residency`, `untiled_bar` |
| `Always`, `RuleTag` | `SPLIT_DIVIDES`, `CORE_BUDGET`, `ONE_REDUCTION_SPLIT`, `SPAN_CAP`, `SPAN_FLOOR`, `RESIDENCY`, `TABLE_COLUMN` | 3 | rule tags |
| `CONSTRAINT_RULES` | `dict[str, RuleTag]`, keyed by constraint function name | 3 | `CHECK_RULES`, `skip_constraints` |
| `coverage` | `coverage(graph, encoder, overrides=None) -> list[str]`, reads `encoder.ENCODES` | 3 | test |
| `CostModel` | type only, stored on the encoder | 4 | constructor |
| `Sym` | `resident`, `split`, `tile`, in `graph/symbols.py` | 4 | `symbols`, `OpVars` |
| `DivisionOwner`, `bind_division_owner` | protocol, implemented by `OpVars`; `(sym_map, buffer_map, owner) -> None` | 4 | `_bind_symbols` |
| `NodeDomain`, `CostDomains`, `domains_from_buffers` | `NodeDomain(name, candidates, readers=(), extra_tile_keys=())`, `CostDomains(nodes, relayouts=())`, `(buffers) -> CostDomains` | 4 | `_cost_domains`, test |
| `cost_expr_to_cpsat` | `(model, sym_map, buffer_map, cost_expr, *, strict=False)`, called with `strict=True` | 4 | test |

`plan_solver.division_symbol` (`plan_solver.py:479`) is existing code and stays.

Provides.

| symbol | signature | for Stages | stability |
|---|---|---|---|
| `Model` | `provenance`, `pruned`, `symbols` | 6-9, 13 | final; Stage 7 adds `boxes`, `capacity` |
| `Encoder` | `__init__(cost_model=None)`, `encode(graph, restrictions=None) -> Model`, `read_back(solution, graph)`, `ENCODES` | 6-9, 13 | final; Stage 7 tightens `solution` |
| `Provenance`, `Outcome`, `Pruned`, `PRUNED_SAMPLE` | above | 0, 6, 7, 8 | final |
| `CpModel` | `cp`, `proto`, `ops`, `units`, `columns`, `tile_terms`, `tag(subject, rule, start)` | 6, 7 | final |
| `OrToolsEncoder` | `__init__(cost_model=None, *, coarse_tiling=False)`, `encode`, `ENCODES` (all eleven flags), `self._m` | 6, 7, 11 | final; Stage 7 adds `placer` |
| `OrToolsEncoder.read_back` | raises `NotImplementedError` | 7 | temporary, replaced in Stage 7 |
| `post`, `post_in`, `post_table`, `symbol` | above | 6, 7, 11 | final |
| `_build_tables`, `_menus`, `self._tables` | `() -> None`, no variables; `dict[OpNode, list[CoreDivision]]`, `dict[OpNode, DivisionTable]` | 6 | final |
| `_add_unit` | `(table, *, size_units, boundary) -> OpVars` | 6, 7 | final; Stage 7 adds its box and `declare` |
| hooks | see the hook table | 6, 7 | Stages 6 and 7 fill them |
| `_cost_domains` | `() -> CostDomains`, `readers` empty | 6 | final; Stage 6 extends it |
| `decision_variables` | `() -> list` | 6, 7 | final; Stage 6 extends it |
| `OpVars` | above, implements `DivisionOwner` | 6, 7 | final; Stage 6 adds fields with defaults, Stage 7 removes `offset` |
| provenance invariant | every constraint index tagged once | 6, 7, 9 | final |
| `DivisionTable` | above; `node=None` on a relayout group | 6, 7, 8 | final |
| `build_division_tables` | above; tables for running nodes only, safe to call again after new pins | 6, 8 | final |
| `stick_of`, `read_stick` | `(port, graph)`, `(edge, graph) -> Optional[tuple[int, ...]]` | 6, 11 | final; Stage 11 extends |
| `_stick_var`, `_stick_is`, `work_division_context_for_op` | `(port) -> IntVar`; `(port, dim) -> bool \| IntVar`; `(op, max_cores=None, tiling=None, skip_constraints=frozenset())` | 11 | final |
| `enumerate_op_divisions`, `tiling_candidates`, `seed_aligned_division` | in `allocator.py`, called by `build_division_tables` | none | final; they stay there |
| `tests/inductor/lx_sections.py` | `build_today`, `project_today`, `section_diff` | 6, 7 | temporary, removed in Stage 13 with the solver |
| `tests/inductor/lx_replay_graph.py` | `replay_graph`, `replay_table`, `ReplayEncoder` | 6, 7, 13 | final |
| `lx_shadow_plugin`, `test_lx_live_parity.py`, the `AGENTS.md` line | live arm | 6, 7, 8 | temporary, removed in Stage 13 |

## Tests

`tests/inductor/test_lx_family_base.py`
- `test_base_and_table_import_no_solver`: in a subprocess, importing `base` and
  `division_table` leaves `ortools` out of `sys.modules`.
- `test_encoder_is_abstract`, `test_model_fields_start_empty`.

`tests/inductor/test_work_division_recording.py`
- `test_first_violation_agrees_with_is_legal`: on mock contexts, each of the six checks.
- `test_recorder_does_not_change_candidates`: list and order equal.
- `test_recorder_charges_the_first_failing_check`: an entry failing two checks counts once.
- `test_removed_factors_names_the_constraint`: an `allowed_splits` removal names its
  function, a span floor names `span_floor`.
- `test_skip_constraints_relaxes_only_that_rule`, `test_collect_result_unchanged_by_sources`.

`tests/inductor/test_lx_division_table.py`
- `test_table_equals_enumerate_core_divisions`: same entries, same order, same `enumerated`,
  on each mock op shape from `tests/inductor/test_work_division.py`.
- `test_untiled_enumeration_failure_never_tries_tilings`,
  `test_tiling_candidates_called_after_untiled_enumeration`.
- `test_split_pin_gives_one_entry_not_enumerated`, `test_opaque_pin_gives_the_committed_division`.
- `test_seeds_take_the_aligned_division`: a `None` keeps the pin's split.
- `test_illegal_pin_raises_unsupported`, `test_committed_pin_skips_the_legal_check`,
  `test_conflicting_split_pins_raise`, `test_no_candidates_raises_unsupported`,
  `test_over_budget_asserts`.
- `test_tables_cover_running_nodes_only`: none for an `InputNode`, one for a `RestickifyNode`
  and an `OpaqueNode`.
- `test_residency_matches_excluded`: verdict then min footprint, with today's strings.
- `test_untiled_bar_only_when_tileable`, `test_rebuild_after_new_pin`.
- `test_rules_prune_per_tiling`: a row from one tiling's frame removes entries of that
  tiling only, and its `Pruned` names the tiling.
- `test_rules_leave_a_pinned_entry`, `test_fact_and_always_rows_apply`,
  `test_pruned_sample_is_bounded`.
- `test_per_core_units_match_the_graph_helper`: equal to Stage 1's on op tables.

`tests/inductor/test_lx_ortools_encoder_op.py`
- `test_encode_runs_in_order`: a subclass recording each step sees the sequence above, with
  no variable before `_settle_tables` returns.
- `test_encodes_lists_all_eleven`, `test_coverage_reports_nothing`: for each node type.
- `test_one_entry_table_has_no_columns`, `test_new_constant_order_is_zero_then_one`,
  `test_unsized_node_has_zero_units`.
- `test_prefetches_hook_runs_first`, `test_add_unit_calls_the_in_place_hook_after_off`.
- `test_loop_columns_only_when_some_entry_is_tiled`, `test_tile_count_only_where_counts_vary`.
- `test_division_is_is_cached_and_two_way`, `test_decision_variables_leave_out_offsets`.
- `test_opvars_is_a_division_owner`, `test_symbols_lower_through_the_printer`.
- `test_cost_domains_match_buffers`: equal `domains_from_buffers` but for `readers`.
- `test_post_folds_without_touching_the_proto`: for each of `when=False`, `expr=True`,
  `str(proto)` unchanged, and no constant variable appears.
- `test_post_false_adds_no_variable`, `test_post_records_index_and_writes_no_name`.
- `test_every_constraint_has_one_provenance`: on the corpus graphs.
- `test_pinned_flags_add_nothing_to_the_proto`: with and without flags.
- `test_pinned_stick_prunes_and_records`: a tightened `factors` removes entries, a `Pruned`
  record names the flag, and `div_n`'s domain shrinks.
- `test_free_stick_posts_under_its_literal`: solved with one worker. With the stick on
  `dim` an illegal split is infeasible, with the stick elsewhere it is feasible.
- `test_two_dim_stick_is_true_on_either_dim`: pinned QFP8WT, and a free two-dim candidate.
- `test_stick_table_with_a_pinned_column_makes_no_constant`,
  `test_violated_stick_table_posts_empty_bool_or`.
- `test_flag_override_relaxes_the_table`: the parent plan's `ReductionNode` example.

`tests/inductor/test_lx_encoder_sections.py`
- `test_sections_match_today[<case>]`: parametrized over `load_captures` for both paths in
  `tests/inductor/cooptimization_capture_loader.py` and over `synthetic_graphs()`.
- `test_sections_match_today_hand_built[<case>]`: tiled menus, `untiled_bar`, a relayout
  copy, in-place edges, and a forced buffer. The in-repo corpus has no tiled entry.
- `test_section_diff_names_the_rule`: an edited encoder output makes `section_diff` report
  the first differing constraint with its `Provenance`.

`tests/inductor/test_lx_live_parity.py`
- `test_live_parity[<fixed graph>-<cell>]`: a `fake` `record_compile` with the shadow
  observer, for `FIXED_GRAPHS` minus `CARD_ONLY`, at `DEFAULT_CELL` and `TILING_CELL`. At
  least one solve is seen, and its report has no section diff and no menu mismatch.
- `test_shadow_report_flags_a_changed_menu`: so the test above can fail.

The parent plan's unit suites run serially. Its device suites run the old paths here,
because no compile reaches the encoder.

What proves the claim: Gate 2 below, supported by
`test_every_constraint_has_one_provenance` and `test_pinned_flags_add_nothing_to_the_proto`.
Housekeeping is `pre-commit run --files <changed>`.

## Parity and gates

The parent plan's gate: "Proven by: an empty proto diff against today's model wherever the
rules agree." A whole-proto diff is impossible here, because today's model has edge and
placement content that does not exist until Stages 6 and 7 (Deviation 1). The gate below is
what "empty" means until then.

**Gate 1, production unchanged.** `lx_gate.sh` between this PR's parent and this PR: proto,
status, objective, decisions and committed plan all identical. At the PR tip the tiers are
`intree`, `fake` at both configs, `unit-*` and `failure`. At a commit they are `intree` and
the fixed set. Any difference blocks the PR, because no production path should change. The
PR reports `corpus_coverage` hits per feature, restickify features included. `fake` proves
parity on default-layout inputs, not production plans.

**Gate 2, section diff.** `build_today(buffers, limit, alignment)` subclasses
`CpSatLayoutSolver`, overrides the phase methods to record `(len(variables),
len(constraints))` checkpoints, and raises from `_tile_count_terms` to stop before any
objective or solve. `project_today` keeps:

- variables: each non-copy buffer's `_wrap` range without `merge_*`; the `loop_id_*` and
  `tiled_*` of non-copies (from `_add_tiling` entry to `_add_tiling_groups` entry); the
  `tile_count_*` range.
- constraints: the same `_wrap` and tiling ranges, the `len(forced) + #untiled_bar`
  constraints at `_add_core_division` entry (each loop iteration posts exactly one), and
  the tile-count range. Any kept-range constraint touching a dropped variable is dropped.

It then renumbers `vars`, `literals` and `enforcement_literal` in the constraints' text to
the kept order. Pass means `[str(v) for v in model.proto.variables]` and the constraint
strings are equal to the projection, lists in order, for every solve. Those are the only
reference fields in kept ranges. Comparison is on text because the 9.15 `CpModelProto`
wrapper has no `SerializeToString`, so "byte-identical" here means `proto_text`-identical.
Stage 6 reuses `build_today` and `project_today` with a smaller drop set and writes no
second comparator. At Stage 7 the drop set is empty.

LX prefetch variables stay in the kept ranges, because interned constants and nest ids depend
on them: a prefetch's `_wrap` can create the first `new_constant`, and nest ids are first-seen
over every unit. Prefetch menus are Stage 6's, so both arms use `ReplayEncoder(prefetches, *,
tables=None, **kwargs)`, in `lx_replay_graph.py`, which fills `_encode_prefetches` from
today's buffers. Stage 6's `prefetch_menu` replaces that override. `replay_table(buffer, limit)`
makes a `DivisionTable` from a buffer, with `residency` as `MemoryPlanSolver.excluded` gives.

- **Replay arm**, in the unit suite. `replay_graph(buffers, limit, alignment)` gives
  `.graph`, `.tables` and `.prefetches`: one node per non-copy buffer, an `InputNode` where
  `first_use_is_read`, with `op=None`, and a `replay_table` for each. `ReplayEncoder` takes
  `tables`, so `build_division_tables` does not run. It covers `load_captures`,
  `synthetic_graphs()` and the hand-built cases. This tests emission, not the table.
- **Live arm**, `lx_shadow_plugin`, a `pytest -p` plugin that is also an `Observer` factory
  for `record_compile`. At `on_solve_begin` on a compile's joint solve it runs `build_graph(ir)` and
  `ReplayEncoder(prefetches).encode`, with `prefetches` keyed by the graph's `InputNode`s in
  `graph.nodes` order, so prefetch order is compared. It then checks `table.entries ==
  buffer.core_divisions` for each op (with the rule's `Pruned` records on a mismatch) and
  the section diff against `build_today(solver.buffers, solver.limit, solver.alignment)`.
  It writes one report row per solve, and passes `model.provenance` to `record_stage` under
  an `LX_EQUIV_DIR` of its own, because a section model is not comparable with a base
  recording. It runs on the `fake` zoo at both configs and on `unit-*`. This tests the
  table, pins, flags and residency.
- **CI arm**, `test_lx_live_parity.py`: the live arm on `fake` compiles of the fixed set at
  both configs, labels `[unit, regression, trunk]`. It is what fails when a later main
  commit changes `CpSatLayoutSolver` and not the encoder. Stages 6, 7 and 8 extend it, and
  Stage 13 removes it.

A prototype of the replay arm passes on this snapshot, with a stand-in encoder reading
buffers directly (not a graph), using `limit = 1 << 21`:

| corpus | solves | variables kept / today | constraints kept / today | equal |
|---|---|---|---|---|
| in-repo captures (153 in-place parents, 54 residency reasons, 11 one-entry tables) | 11 | 2448 / 3040 | 1581 / 4116 | 11 / 11 |
| synthetic graphs | 7 | 644 / 736 | 371 / 835 | 7 / 7 |
| hand-built: tiled, untiled bar, copy, in-place, forced | 3 | 62 / 98 | 47 / 128 | 3 / 3 |

The stand-in also kept the provenance invariant and wrote no constraint names on the three
hand-built cases. No real tiled menu has been through this comparison, since the in-repo
corpus has zero tiled entries. The live arm is the first comparison from a real graph.

**Gate 3, rules agree.** On the live arm, rule application (step 3 of
`build_division_tables`) writes no `Pruned` record, and no `VIOLATED` appears. A difference
is listed by rule in the PR and justified or fixed.

**Gate 4, parent's verification items.** Device suites pass on the card. Planning time:
`lx_corpus.py timing BASE_TREE CHANGE_TREE --tier fake` alternates cold processes on live
compiles and shows no growth beyond noise; only the delegates are on the production path.
`build_today` against `encode` time is reported, not gated. The guards from #5682 live in
Stage 0 and are reused as they are.

## Risks

- **A later change creates a constant early.** One `new_constant` or `only_enforce_if(True)`
  before today's first shifts every index. Gate 2 fails at the first differing variable, and
  `test_post_folds_without_touching_the_proto` covers `post`.
- **Stage 2 omits a `split` pin.** `build_division_tables` then enumerates a fixed op.
  The live arm's `table.entries` check names the op.
- **A `PrefetchMenu` differs from today's.** Both arms take LX prefetches from buffers, so Stage 5
  cannot see it. Stage 6's gate does, once `prefetch_menu` replaces the override.
- **A main commit changes the old model after this merges.** A #5682 Step 2 item is one.
  `test_lx_live_parity.py` fails on it, and the `AGENTS.md` line says the PR must port the
  change to the encoder.
- **Enumeration runs twice once Stage 9 calls the encoder.** With tiling on, enumeration is
  about 88% of planning, and step 3 adds a `frame_of` per tiling, whose cost is unmeasured.
  Stage 9 must not run `_prepare_buffers` and the encoder's tables side by side in
  production. The planning-time check after Stage 9 catches it.
- **Flag data disagrees with `factor_domain`.** `whole_stick_factors` and the stick-adjusted
  basis are two derivations of one rule, for instance on symbolic dims (`work_division.py:890`).
  Gate 3 reports it by rule.
- **#5695 lands first.** It changes when `_fixed_core_division` has a value and adds
  `divide_deferred_op` on pinned paths. The extractions conflict textually in
  `_enumerate_core_divisions`. Rebase and re-record per the parent plan; the
  `test_table_equals_enumerate_core_divisions` test and Gate 1 catch semantic drift. Stage
  2's pin values must come after the same call.
- **OR-Tools behaviour differs off the pinned version.** Interning and `.index` were checked
  on 9.15.6755 only. `pyproject.toml` allows `ortools>=9.0`. The gate runs on the pinned
  version, and baselines are per environment.
- **A private rename breaks `build_today`.** It raises or skips a checkpoint, and
  `test_sections_match_today` fails.

## Deviations from the parent plan

1. **Plan: "an empty proto diff against today's model."** Code: S1 to S4 are about 80% of
   the variables and under 40% of the constraints on the in-repo captures, and the rest
   needs Stages 6 and 7 (`ilp_solver_ortools.py:975-1172`, `:1633-1806`). Resolution: Gate 2's
   section diff, whose drop set is empty at Stage 7.
2. **Plan: "Residency rules are applied while the table is built", and the sequence diagram
   has op level first.** Code: forced pins come after the no-overlap (line 1796 against line
   1664), and `tile_count` after the relayout coupling (line 1427). Byte identity requires
   those positions. Resolution: the verdict is computed while the table is built, and the
   constraint is emitted at today's position, so op level sits in four ranges.
3. **Plan: "A division is banned together with residency rather than removed."** Code:
   no residency rule looks at a division, and the single division-dependent ban is
   `untiled_bar`, posted as an implication on `tiled`. Resolution: post exactly that.
   `post_in` over the index is not used, because it would change the proto.
4. **Plan: "Each node type's flags are posted."** Code: all sixteen constraint functions
   and `is_legal`'s span cap and floors already sit inside the table
   (`work_division_constraints.py:135-152`). Resolution: enumeration keeps deciding the
   table, and in Phase 2 flags verify rather than prune.
5. **The deferred direct-read menu is not op level.** `_prepare_buffers`
   (`allocator.py:3145-3183`) pins a restickify's menu after pricing, using buffers including
   relayout copies (`allocator.py:3208-3215`). `config.read_copy_elision` is on by default.
   Resolution: `_settle_tables` runs between tables and variables, and Stage 6 owns the
   loop. Until then the live arm adds `Pin(node, "split", WorkSplitPin(split, "unproved or
   unpriced direct read"), "hazard")` from the allocator's final `divisions`, so the loop
   itself is not covered by Stage 5's gate.
6. **Plan: "every `layout_solver` × `co_optimizing_lx_planning` combination" and byte
   identity.** Code: `plan_layout` uses `_LifetimeBufferWithCpVars`, with a constant footprint
   and no division (`ilp_solver_ortools.py:304`, `allocator.py:1542`). "Every division pinned"
   (Stage 13) would add the one-entry constants, so the protos cannot be identical.
   Resolution: Gate 2 covers the joint path only. Gate 1 covers the rest. Committed mode has
   its own bar in Stage 9. Also not covered is `ExhaustiveSearchSolver`'s `prune=True`, which
   never reaches CP-SAT.
7. **Plan: "replay the zoo captures on base and change, in one process and interleaved."**
   Code: two trees cannot share a process, and a capture cannot reach the graph layer.
   Resolution: `lx_corpus.py timing`, alternating cold processes on live compiles.
8. **Plan: "the table entries each rule pruned are recorded."** The rejected combinations
   per op are unmeasured (menus reach 164 on the zoo, and the raw product is larger), and
   tiling multiplies them. Resolution: counts for all, entries up to `PRUNED_SAMPLE`, and
   axis-domain records for factors that never became entries.

## Open questions

None.

## Out of scope

- Restickify, relayout copies, cuts, nests, in-place overwrite, prefetch menus and reads, the deferred-read
  loop, cross-frame hazard, the partial-reduction cut guard, `_encode_objective` and its
  lowering: Stage 6.
- `boxes`, `capacity`, capacity and no-overlap constraints, `ModelAbstractFactory`,
  `OrToolsFactory`, `Backend`, `CpSatBackend`, `Solution`, `SolveSettings`, `FACTORIES`, config
  values, `read_back`, running the objective stages: Stage 7.
- `AnnealModel`, `SAEncoder`: Stage 8. It reuses `build_division_tables`.
- `CostModel`, `Sym`, objective terms, printer changes: Stage 4.
- Calling the encoder from a compile and the failure flow: Stage 9.
- Lifting stick pins and rebuilding tables over candidate layouts: Stage 11.
- Deleting old code: Stage 13.
- Pin derivation for fixed ops: Stage 2. Rule data, residency rules, `check_decisions`: Stage 3.
- A greedy family. Nothing here rules one out: it could use `build_division_tables` and an
  `Encoder` whose `Model` carries no payload.
