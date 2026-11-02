# LX allocator, Stage 1: Graph types

Parent plan: `plans/lx-allocator.md` at `5aff4a6`. Code base: torch-spyre `5268ba95`.
Depends on: Stage 0 (the `intree` tier, for the gate only).
Needed by: Stages 2 to 13.

## Goal

When this PR merges, torch-spyre has a new, unused package holding the graph layer's data
types: `SpyreOptimizationGraph`, ports, node types with frozen flags, edges, pins, the three
built transforms, `WorkSplit`, `Axis`, `Restriction` and `PortFact`. Every field Stages 2 to 10
read or fill is declared here, so none of them edits a class to add one. Every type can be built
and tested without a device, a live FX graph or a solver, and a test proves the layer imports no
solver. What a flag requires is Stage 3. No production module imports the package, so no compile
path, CP model or committed plan changes.

## Today

**Layouts.** `FixedTiledLayout` is at `torch_spyre/_inductor/ir.py:89`. It extends upstream
`FixedLayout` with `device_layout`, `allocation` and `lx_view` (`ir.py:105-108`).
`unallocated_copy` copies the geometry without allocation or per-core ownership (`ir.py:110`).
`SpyreTensorLayout` is C++, typed at `torch_spyre/_C.pyi:186`. `scratchpad/__init__.py` is the
licence header only.

**Both layouts can be built with no device.** The pattern exists at
`tests/inductor/test_cost_model_decode.py:169-177`. I ran it: construction, `pickle` of a
`SpyreTensorLayout` and `copy.deepcopy` of a `FixedTiledLayout` all work and never start the
runtime. It needs the built `_C` extension, because `ir.py:32` imports it at module level, as
does `torch_spyre/_inductor/constants.py:17`.

**Three traps in layout equality and identity.**
- `FixedTiledLayout == FixedTiledLayout` ignores `device_layout` and `lx_view`. It inherits
  `Layout.__eq__` from upstream (`torch._inductor.ir`, checked on the installed torch 2.13.0),
  which compares only device, dtype, size, stride, offset and `is_pinned`. Two layouts of one
  buffer stuck on different dims compare equal (run: `a == b` was `True` with `stl_a == stl_b`
  `False`). The allocator compares `device_layout` directly (`scratchpad/allocator.py:1719`).
- `SpyreTensorLayout.__eq__` ignores `valid_elements` (`csrc/spyre_tensor_impl.h:236-242`).
- `SpyreTensorLayout` discards its `dim_map` at construction, so host-to-device dim identity
  is not on the object (`wsr/loop_tiling_common.py:215-216`).

**Stick layouts after a restickify.** `compute_restickify_target_layout`
(`pass_utils.py:2352`) needs the read's coordinates and a `PropArg`, so it cannot run without a
live dep. Its integer core is two pure functions, `restickify_device_size`
(`pass_utils.py:2271`) and `restickify_stride_map` (`pass_utils.py:2290`). It returns
`SpyreTensorLayout(device_size, stride_map, stl.device_dtype)`, which resets the element
arrangement (`pass_utils.py:2453`). The op's layout is assigned from the plan entry
(`insert_restickify.py:286`), then `insert_restickify_padding` may grow a device dim of it
(`padding.py:650`), and that pass runs before LX planning (`passes.py:558-577`). So a
pre-inserted restickify's real layout is not always the pure swap.

**A restickify op is a full solver record today.** It is in `mem_usage` like any op
(`allocator.py:4269`) and gets a `CoreDivisionBuffer` and a menu through `_division_map`
(`allocator.py:3737-3800`), so it has its own tick, size, division, residency and box. Its
division variable exists only when the menu has more than one entry
(`ilp_solver_ortools.py:454-479`).

**Ticks.** `calculate_liveness` (`scratchpad/utils.py:77`) gives each buffer the strictly
increasing indices into `graph.operations` where it is touched. `LifetimeBoundBuffer` turns
them into `start_time = min(uses[0], start override)` and
`end_time = max(uses[-1] + 1, end override)` (`plan_solver.py:161-170`), and `read_count` is
`len(uses)` less the producing write (`:146-159`). Overrides come from
`counted_loop_lifetime_overrides` (`utils.py:143`) and from drains (`allocator.py:4162-4166`).
Graph inputs have no op: `uses[0]` is their first reader, and an input with no uses gets no
record (`allocator.py:1573-1575`). `CoreDivisionBuffer.op_position` is the same index
(`allocator.py:4174`, `:4474`).

**Sizes.** `mem_usage_by_buf` sizes a buffer with `get_device_size_in_bytes` and marks anything
without a `FixedTiledLayout` as `-1` (`utils.py:226`, `:256-271`). The co-optimizing records
carry the total size (`allocator.py:4287`). The per-core figure is
`ceil_div(size, output_partition * output_tile_count)` in `CoreDivisionBuffer.per_core_size`
(`plan_solver.py:395-401`) and in SA's copy, clamped to 0 (`sa_cooptimizer.py:1142-1165`). CP-SAT
divides by alignment first (`ilp_solver_ortools.py:727`, `:435`). Reduction splits and reduction
tile levels do not shrink the buffer (`plan_solver.py:238-249`, `:312-315`). The placement path
uses `dev_size // num_cores` (`utils.py:274`, `allocator.py:1742`).

**Whole sticks.** `adjust_it_space_for_sticks` (`work_division.py:406`) finds, per loop symbol,
the largest `elems_per_stick` among tensors stuck on it (`:438-484`) and converts the extent to
a stick count by ceiling division (`:487-493`). A QFP8WT tensor counts both of its last two
device coordinates (`:443-461`, `_qfp8wt_stick_coords` at `:396`).
`WorkDivisionContext.factor_domain` takes divisors of the symbolic granularity, the stick count
or the element count, in that order (`:889-899`), through `_legal_split_factors` (`:188`).

**What a division holds.** `CoreDivision` (`plan_solver.py:279`) has `splits` keyed by iteration
symbol, `reduction_syms`, `tiling: TileSpec` and `tile_splits`, one `(symbol, trip count)` per
level, outermost first. `_core_division` drops factors of 1 and classifies reduction symbols by
`write.index.coeff(s) == 0` (`work_division.py:1109-1143`). `WorkDivisionContext.reduction_vars`
uses a different test, absence from the output's device coordinates (`work_division.py:1018-1024`),
and `coordinate_mask_blocked_vars` reads that one. `RelayoutCandidate.source_division` and
`consumer_division` are menu indices (`scratchpad/lx_relayout.py:143-144`), which is what these
types must avoid.

**Package inits, measured.** I ran the snapshot's Python with a `_C.so` from another checkout
(see Risks). With `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, `import torch_spyre._inductor.scratchpad`
loads 16 `torch_spyre*` modules, no `ortools`, and no Spyre compiler module, though
`_inductor/__init__.py:15-24` pulls `constants` (hence `_C`), `patches`, `config`, `propagate_hints`
and `profiler._ffdc`. On a default install, `import torch` loads 73 through `_autoload_impl`
(`torch_spyre/__init__.py:288`, `:298`, `:319`, `:336`), including `allocator`, `sa_cooptimizer`,
`simulated_annealing` and `plan_solver`. `ortools` arrives only through the lazy import at
`allocator.py:5354`. So a `sys.modules` check for solver modules means nothing unless autoload is
off.

## Design

### Module layout

```text
scratchpad/lx_allocator/
  graph/   flags axes transforms nodes optimization_graph symbols   # Stage 1; Sym from 4
  division_convert.py graph_builder.py                              # Stage 2
  rules.py check.py plan_adapter.py                                 # Stage 3
  cost.py                                                           # Stage 4
  family/  base division_table ortools_encoder                      # Stage 5
           edge_tables                                              # Stage 6
           placement packer ortools_family registry                 # Stage 7
           sa/                                                      # Stage 8
  optimizer.py optimizer_bridge.py                                  # Stage 9
  commit.py                                                         # Stage 10
```

`pyproject.toml:112` finds new packages automatically. The "no solver" boundary is the directory
`graph/` and nothing else. `graph_builder.py`, `rules.py` and the rest sit outside it because
they legitimately import `allocator` or `plan_solver`. Both `__init__.py` files hold the licence
header only and re-export nothing.

| module | contents | imports |
|---|---|---|
| `graph/flags.py` | `Restriction`, `PortFact` | stdlib |
| `graph/axes.py` | `Axis`, `WorkSplit`, `TileLevel`, `per_core_units` | sympy |
| `graph/transforms.py` | `Transform`, `CopyOut`, `Restickify`, `Relayout`, `stick_dims`, `moved_stick`, `SPARSE_STICK` | `axes`, `_C`, `ir`, `pass_utils` |
| `graph/nodes.py` | ports, `OpNode` and subtypes, `DataEdge`, `ExecutionEdge`, `Pin`, pin value types, `restrictions_for` | `flags`, `axes`, `transforms`, sympy |
| `graph/optimization_graph.py` | `SpyreOptimizationGraph` | `nodes`, `transforms` |

The import order is `flags`, `axes` < `transforms` < `nodes` < `optimization_graph`. Files follow
AGENTS.md: the 14-line licence header, 88 columns, `from __future__ import annotations`.
`PerCoreView`, `FixedTiledLayout` where used only as a type, `GraphLowering`, `Dep`, `DrainPlan`,
`LXRelayoutPlan`, and `OpNode` and `DataEdge` in `transforms.py` are `TYPE_CHECKING` imports.

### `flags.py`

`PortFact` is the parent plan's. `Restriction` is the parent plan's six members in order, with the
`INDIRECT_ACCESS` comment ending at "partial" (there is no padding exception), followed by five
more that Stage 3's rows tag:

```python
    WINDOW_UNSPLIT = auto()          # pooling and conv kernel windows stay on one core
    CONV_SPATIAL_UNSPLIT = auto()    # strided conv keeps its output image dims unsplit
    TOPK_SPLITS = auto()             # topk's legal split domains
    KEEP_BY_INDEX_SPLITS = auto()    # keep_by_index's legal split domains
    PLAIN_REDUCTION_UNSPLIT = auto() # plain reductions' K split domains
```

### `axes.py`

```python
@dataclass(frozen=True)
class Axis:
    """One loop axis of one op. A value, not an index into anything."""
    symbol: sympy.Symbol             # the op's iteration symbol, the CoreDivision.splits key
    extent: int                      # elements, concretize_expr(it_space[symbol])
    reduction: bool = False          # write.index.coeff(symbol) == 0, as _reduction_syms
    granularity: int | None = None   # SymbolMeta divisor basis; goes with issue #5890
    off_output_coords: bool = False  # absent from the output's device coordinates
    symbolic: bool = False           # extent is a shape hint; goes with issue #5890

    @property
    def name(self) -> str: ...       # str(self.symbol)

TileLevel = tuple[Axis, int]         # one tile level: the axis and its trip count

@dataclass(frozen=True)
class WorkSplit:
    """A chosen core division plus coarse tiling, as values."""
    splits: Mapping[Axis, int] = field(default_factory=dict)  # cores along each axis
    tiling: tuple[TileLevel, ...] = ()                        # outermost first

    def __post_init__(self) -> None: ...  # factors >= 1; drops factors of 1
    def __hash__(self) -> int: ...        # hash((frozenset(splits.items()), tiling))
    # properties: cores_used, output_splits, reduction_splits, output_partition,
    # tile_count, output_tile_count, as CoreDivision and TileSpec define them

def per_core_units(
    size: int, splits: Mapping[Axis, int], tiling: Sequence[TileLevel], alignment: int
) -> int: ...
```

- **How these stay free of menu indices.** `WorkSplit` holds factors keyed by `Axis`, with no
  position or `chosen_division`. A table row exists only inside an encoder, and `read_back` stores
  the row's value on `node.work_split`. A later non-enumerating encoder builds the same value from
  per-axis variables. `Relayout.reduction_split` is a factor, `Restickify.to_stick` a buffer dim.
- **Conversions live outside `graph/`.** `TileSpec` and `CoreDivision` sit in `plan_solver.py`, on
  the wrong side of the boundary. Stage 2's `division_convert.py` converts in both directions.
  `splits` stays a plain `dict` so it deep-copies and pickles; treat it as read-only.
- `per_core_units` is `max(0, ceil_div(size, alignment * partition * output_tiles))`, where
  `partition` and `output_tiles` skip axes with `reduction` set. `ceil_div` is a private two-liner
  here for the same boundary reason.

### `transforms.py`

```python
SPARSE_STICK = -1   # the dim_order sparse-stick marker (propagate_layouts.py:254)

def stick_dims(layout: FixedTiledLayout) -> tuple[int, ...]:
    """Host dims `layout` is stuck on. One dim normally, two for QFP8WT, none for a sparse
    or size-1 stick or when no single dim matches. The only reader of a layout's stick."""

def moved_stick(layout: FixedTiledLayout, to_stick: int) -> SpyreTensorLayout:
    """The layout's device layout with the stick moved to host dim `to_stick`."""

@dataclass(eq=False, kw_only=True)
class Transform(ABC):
    resident: bool | None = None     # decision
    offset: int | None = None        # decision, alignment units
    footprint: int = 0               # per-core alignment units, decoded with the step
    pinned: bool = False             # present before the solve
    edges: list[DataEdge] = field(default_factory=list, init=False, repr=False)

    ORDER: ClassVar[int]             # where a decoder appends this kind; never a sort key
    forced: ClassVar[bool]
    in_lx: ClassVar[bool] = True

    @abstractmethod
    def apply(self, layout: FixedTiledLayout) -> FixedTiledLayout: ...

@dataclass(eq=False, kw_only=True)
class CopyOut(Transform):            # ORDER = 0, forced = True, in_lx = False
    def apply(self, layout): ...

@dataclass(eq=False, kw_only=True)
class Restickify(Transform):         # ORDER = 1, forced = True
    to_stick: int                    # host dim of the buffer, or SPARSE_STICK
    realized_by: OpNode | None = None        # the RestickifyNode of a pinned step
    target: FixedTiledLayout | None = None   # an unpinned step's layout, from Stage 11
    def apply(self, layout): ...

@dataclass(eq=False, kw_only=True)
class Relayout(Transform):           # ORDER = 2, forced = False
    to_view: PerCoreView
    reduction_split: int = 1
    source_view: PerCoreView | None = None
    source_footprint_bytes: int = 0         # unrounded, what LXRelayoutPlan takes
    destination_footprint_bytes: int = 0    # unrounded
    def apply(self, layout): ...
```

`Transform` instances compare by identity. One instance may sit on several edges: a copy-out is
shared by every reader outside the nest, and a relayout group by its consumers.

**Units.** `OpNode.offset`, `Transform.offset` and `Transform.footprint` are alignment units, as
the solvers work. Anything needing bytes multiplies by `graph.alignment` at the edge. The two
`Relayout` byte fields are the exception, because `LXRelayoutPlan` needs an unrounded figure that
a unit count has lost.

**A pinned `Restickify` is the edge's view of a node.** A pre-inserted restickify op is a
`RestickifyNode`. The edge into it carries `Restickify(pinned=True, realized_by=node)`. Division,
residency, box and tick live on the node. The step has no footprint, variables or box of its own,
and its `resident`, `offset`, `footprint` stay at their defaults.

**How each `apply` works.** None touches `V.graph`, an op or a solver, and none mutates its
argument, which can be the object the IR holds. `apply` needs only `_C`, `ir.py`, and for
`Restickify` two pure functions from `pass_utils`.

| step | result |
|---|---|
| `CopyOut` | `layout.unallocated_copy()`. Same geometry. The buffer is in HBM, so it has no allocation and no `lx_view`. |
| `Relayout` | `layout.unallocated_copy()`, with `lx_view = self.to_view`. The device layout is unchanged, the same shape as today's copy layout (`lx_relayout.py:2126`). |
| `Restickify`, `realized_by` set | `realized_by.out.layout`, which holds padding that `to_stick` alone cannot reproduce. |
| `Restickify`, `target` set | `target`. Unpinned steps exist only from Stage 11. |
| `Restickify`, neither | A new `FixedTiledLayout` from the input's host fields and `moved_stick(layout, self.to_stick)`, built as `_fixed_tiled` does (`insert_restickify.py:171`). |

Dropping `lx_view` on a restickify is correct: `PerCoreView` is keyed by device dim
(`core_mapping.py:1061`), and a restickify changes the device dims.

**`stick_dims`.** It reads `stride_map[-1]`. An entry of `-1` is a sparse or size-1 stick, and
gives `()`. Otherwise it is the host stride of the stick dim, and the answer is the unique host
dim with that stride and size above 1. For `ElementArrangement.QFP8WT` it does the same for
`stride_map[-2:]`, matching `_qfp8wt_stick_coords`. Strides distinguish same-size dims, which is
what defeats the size-based match in `_resize_device_layout` (`ir.py:172-184`). Results on real
layouts: `[128, 256]` stuck on 1 gives `(1,)`, stuck on 0 gives `(0,)`, and a stick on a size-1
dim gives `()`. A stick pin's value is `stick_dims(layout) or (SPARSE_STICK,)`.

**`moved_stick`.**
1. Raise `ValueError` unless the arrangement is `STANDARD`, `stick_dims` gives exactly one dim
   `old`, and `to_stick` is a different host dim of size above 1.
2. Find `new_j`, the outer device dim with `device_size == host[to_stick]` and
   `stride_map == stride[to_stick]`.
3. Find `old_j`, the outer device dim other than `new_j` with
   `device_size == ceil(host[old] / eps)` and `stride_map == min(eps, host[old]) * stride[old]`,
   the tile-count dim as `_resize_device_layout` describes it (`ir.py:380-384`). Matching the pair
   matters: `[3, 5]` stuck on 1 has `device_size [1, 3, 64]` and `stride_map [5, 5, 1]`, so stride
   alone collides.
4. Raise `ValueError` if either match is not unique. A factorized layout lands here.
5. Return `SpyreTensorLayout(restickify_device_size(...), restickify_stride_map(...),
   stl.device_dtype)`, the same three calls as `compute_restickify_target_layout`.

The result equals a from-scratch `SpyreTensorLayout(size, stride, dtype, order)` on the cases I
ran: `[128, 256]` both ways, `[4, 128, 256]`, a permuted write and a short old stick. It differs
where the new stick is shorter than a stick: `[3, 5]` moved to dim 0 gives `stride_map [320, 1, 5]`
against `[15, 1, 5]`, from `restickify_stride_map`, and the test pins that. Phases 1 and 2 never
reach this branch.

### `nodes.py`: ports

```python
@dataclass(eq=False, kw_only=True)
class Port(ABC):
    name: str
    node: OpNode = field(repr=False)
    axis_map: dict[Axis, int] = field(default_factory=dict)  # axis -> outermost host dim
    facts: PortFact = PortFact.NONE
    dep: Dep | None = None                       # the read or write; None on a hand-built port
    coords: tuple[sympy.Expr, ...] | None = None         # host coordinates
    device_coords: tuple[sympy.Expr, ...] | None = None  # under the read layout

    @abstractmethod
    def current_layout(self) -> FixedTiledLayout | None: ...
    def dims_of(self, axis: Axis) -> tuple[int, ...]: ...  # every host dim the axis reaches
    def stuck_axes(self) -> tuple[Axis, ...]: ...          # dims_of meets stick_dims

@dataclass(eq=False, kw_only=True)
class InPort(Port):
    operand: int | None = None        # positional by default, set by the builder
    index: sympy.Expr | None = None   # the read's index expression
    edge: DataEdge | None = None      # None: unbound, as a gather's `values`
    read_layout: FixedTiledLayout | None = None  # only under _input_layout_overrides
    direct_read: DirectReadStick | None = None
    def current_layout(self): ...     # self.edge.read_layout(), else None; on a RestickifyNode,
                                      # the layout ahead of the step it realizes

@dataclass(eq=False, kw_only=True)
class OutPort(Port):
    layout: FixedTiledLayout | None = None
    index: sympy.Expr | None = None   # the write's index expression
    edges: list[DataEdge] = field(default_factory=list)
    def current_layout(self): ...     # self.layout
```

`dims_of` reads `coords`: the dims whose coordinate has `axis.symbol` free, or the `axis_map` entry
when `coords` is `None`. An in port stores a layout only where `_input_layout_overrides` makes an
op read a graph input in a layout other than the committed one (`insert_restickify.py:940`).
Otherwise `current_layout` derives it from the edge. On a `RestickifyNode` it applies only the
steps ahead of the `Restickify` that node realizes, because the node reads the source. Treat a
returned layout as read-only: it can be the IR's own object.

### `nodes.py`: node types

```python
_FROZEN_ON_TYPE = frozenset({"restrictions", "PORTS", "VARIADIC", "runs"})

class _NodeType(type):
    def __setattr__(cls, name, value): ...  # raises AttributeError on _FROZEN_ON_TYPE
    def __delattr__(cls, name): ...         # same

@dataclass(eq=False, kw_only=True)
class OpNode(metaclass=_NodeType):
    name: str                      # the op's buffer name
    axes: tuple[Axis, ...] = ()
    size: int = 0                  # total device bytes; -1 unsized, as mem_usage_by_buf
    op: Any = None                 # the live IR op; None on hand-built nodes
    tick: int | None = None        # graph.operations index; assigned by add_node if None
    uses: tuple[int, ...] = ()     # calculate_liveness's entry
    lifetime_start_override: int | None = None
    lifetime_end_override: int | None = None
    graph_output: bool = False
    mutation_target: OpNode | None = None
    deferred_direct_read: bool = False
    span_floors: dict[Axis, int] = field(default_factory=dict)
    tile_local_axes: frozenset[Axis] = frozenset()
    prev: ExecutionEdge | None = field(default=None, repr=False)
    next: ExecutionEdge | None = field(default=None, repr=False)
    work_split: WorkSplit | None = None   # decision; stays None on an InputNode
    resident: bool | None = None          # decision
    offset: int | None = None             # decision, alignment units
    spill_reason: str | None = None       # why it left LX, written by read_back
    operands: InitVar[int | None] = None  # variadic types only
    layout: InitVar[FixedTiledLayout | None] = None
    in_ports: dict[str, InPort] = field(init=False, default_factory=dict)
    out: OutPort = field(init=False)

    PORTS: ClassVar[tuple[str, ...]] = ()
    VARIADIC: ClassVar[bool] = False
    runs: ClassVar[bool] = True
    restrictions: ClassVar[Restriction] = Restriction.WHOLE_STICKS

    def __init_subclass__(cls, **kwargs) -> None: ...  # restrictions must be a Restriction
    def __post_init__(self, operands, layout) -> None: ...
    def __setattr__(self, name, value) -> None: ...    # raises on _FROZEN_ON_TYPE
    @property
    def first_use_is_read(self) -> bool: ...           # not self.runs
    @property
    def read_count(self) -> int: ...                   # LifetimeBoundBuffer.read_count
    def port(self, name: str) -> InPort: ...
    def add_in_port(self, name: str, *, operand: int | None = None) -> InPort: ...
    def ports(self) -> tuple[Port, ...]: ...           # in ports, then out
    def padded_reductions(self) -> list[tuple[InPort, Axis]]: ...
    def whole_stick_factors(self, axis: Axis) -> tuple[int, ...]: ...
    def broadcast_dims(self, port: InPort) -> list[int]: ...
    def same_index_read(self, port: InPort) -> bool: ...  # False here
    def footprint(
        self, splits: Mapping[Axis, int], tiling: Sequence[TileLevel], alignment: int
    ) -> int: ...

class PointwiseNode(OpNode):
    PORTS = ("in0",)
    VARIADIC = True
    restrictions = OpNode.restrictions | Restriction.POINTWISE_STICKS
    def same_index_read(self, port): ...

@dataclass(eq=False, kw_only=True)
class MatmulNode(OpNode):
    k: Axis | None = None    # None on the path where the builder raises Unsupported
    n: Axis | None = None    # None when N = 1: there is no loop symbol
    reduction_type: str = ""
    PORTS = ("x", "y")
    restrictions = (OpNode.restrictions | Restriction.MATMUL_STICKS
                    | Restriction.PADDED_REDUCTION_UNSPLIT
                    | Restriction.ENCODED_K_UNSPLIT)

@dataclass(eq=False, kw_only=True)
class ReductionNode(OpNode):
    reduction_type: str = ""
    conv_params: dict[str, Any] | None = None
    PORTS = ("in0",)
    VARIADIC = True
    restrictions = (OpNode.restrictions | Restriction.PADDED_REDUCTION_UNSPLIT
                    | Restriction.POINTWISE_STICKS | Restriction.WINDOW_UNSPLIT
                    | Restriction.CONV_SPATIAL_UNSPLIT | Restriction.TOPK_SPLITS
                    | Restriction.KEEP_BY_INDEX_SPLITS
                    | Restriction.PLAIN_REDUCTION_UNSPLIT)

@dataclass(eq=False, kw_only=True)
class IndirectNode(OpNode):
    scatter: bool = False
    PORTS = ("table", "index", "values")
    restrictions = (OpNode.restrictions | Restriction.INDIRECT_ACCESS
                    | Restriction.POINTWISE_STICKS)

@dataclass(eq=False, kw_only=True)
class InputNode(OpNode):
    prefetch_view: PerCoreView | None = None   # decision: the view of an LX prefetch
    runs = False
    restrictions = Restriction.NONE

class OpaqueNode(OpNode):
    """Runs but is not a Spyre kernel: a tick and HBM reads and writes, its committed
    division only, never resident."""
    PORTS = ("in0",)
    VARIADIC = True
    restrictions = Restriction.NONE

class RestickifyNode(OpNode):
    """A stick-swapping copy fixed before the solve. It reads one stick and writes another
    by design, so it has no POINTWISE_STICKS. A direct OpNode subclass: node types are leaves."""
    PORTS = ("in0",)
    VARIADIC = True
    restrictions = OpNode.restrictions

RestrictionOverrides = Mapping[type[OpNode], Restriction]

def restrictions_for(
    node: OpNode, overrides: RestrictionOverrides | None = None
) -> Restriction:
    """`overrides[type(node)]` if present, else the type's frozen flags."""
```

**Construction.** `__post_init__` raises `TypeError` for `OpNode` itself, which is how it stays
abstract. It builds one `InPort` per name, with `operand` equal to the position, and one `OutPort`
named `out` holding `layout`. A variadic type gets `in0 .. in{max(operands, 1) - 1}`. A fixed type
rejects a non-`None` `operands`. Unbound ports are legal: a gather leaves `values` unbound.
`port(name)` raises `KeyError` naming the node and its ports. `add_in_port` adds a port no tuple
names, such as a matmul's `target` or a further `table1`, and raises `ValueError` on a duplicate.
On a self-matmul `identify_matmul_inputs` returns one dep for both ports
(`pass_utils.py:1288-1289`): there is one edge whose `dst` is `x`, and the builder sets `y.edge` to
it, so `y.edge` is never `None`.

**How `restrictions` is frozen.**
- `OpNode.__setattr__` raises `AttributeError` for the frozen names, as the parent plan requires.
- The metaclass `_NodeType` does the same for assignment and deletion on the class, so a test cannot
  mutate a type and leak into the next, which AGENTS.md forbids. `mock.patch.object` raises by
  design, and a test must use the mapping instead.
- A subclass sets its flags in the class body, which does not go through `__setattr__`.
  `__init_subclass__` rejects a value that is not a `Restriction`.

**The test-only override.** `restrictions_for` is the only way code reads flags. Stage 3's
`check_decisions` and `coverage`, the encoders' `encode`, and `optimize` all call it and pass their
`restrictions` argument through. The lookup uses the exact type, which is sound because every node
type is a leaf.

**Facts.** All answer for the untiled frame, from the layouts the ports see now. Stage 3's rows
need a tiling and take theirs from its own frame, so these methods do not delegate to `rules.py`
(that would be an import cycle). A Stage 3 test holds the two to one rule.

`whole_stick_factors(axis)` gives the ascending divisors of the axis's whole-stick basis, the domain
that applies while the axis is stuck. Whether something is stuck is Stage 3's guard.
1. If `axis.granularity` is set, that is the basis, as `factor_domain` does first.
2. Otherwise let `stuck` be the ports with `axis` in `stuck_axes()`. This is the set
   `adjust_it_space_for_sticks` scans, including the output and QFP8WT's two dims.
3. If `stuck` is empty, nothing is stuck now and the question is hypothetical, so use every port
   that maps the axis and has a layout. If there are none, return the divisors of `extent`.
4. `eps` is the largest `elems_per_stick()` among them, and the basis is `ceil(extent / eps)`.

It agrees with `_legal_split_factors(v, basis)` with no allowed splits or span floor.

`padded_reductions()` returns `(port, axis)` for each bound in port and each `axis` in its
`axis_map` with `axis.off_output_coords` and `axis.extent % eps != 0`, where `eps` is that port's
own `elems_per_stick()`. It ignores whether the port is stuck there, because Stage 3 posts the rule
under the stick. `coordinate_mask_blocked_vars` (`work_division_constraints.py:351`) tests one
`eps`, the maximum across tensors stuck on the symbol. Per-port results give the same answer: every
`eps` in use is a power of two, so a multiple of the largest is a multiple of the rest.

`broadcast_dims(port)` gives the host dims of the read layout that no axis reaches through
`dims_of`. It returns `[]` when the layout is `None`, and raises `ValueError` on an unbound port.

`same_index_read(port)` is `False` on the base class. On `PointwiseNode` it is true when
`self.out.index` is set, the port is bound, and every bound in port of this node reading the same
`src` has `index == self.out.index`. That is `get_op_pointwise_inputs` (`utils.py:740-749`),
including its rule that one bad read bars the buffer. The builder leaves `out.index` as `None` in
the two cases that function and `_op_inputs_good_for_lx_inplace` refuse: no single `MemoryDep` write
(`utils.py:737-739`), or no origin target (`allocator.py:1427-1429`). The full gate is
`DataEdge.in_place_candidate`, which Stage 2 fills.

`footprint(splits, tiling, alignment)` is `per_core_units(self.size, splits, tiling, alignment)`,
in alignment units.
- `ceil_div(ceil_div(s, a), d) == ceil_div(s, a * d)` for positive integers, so CP-SAT's order and
  SA's order give the same number.
- It ignores reduction splits and reduction tile levels, as `output_partition` and
  `output_tile_count` do. The sketch's `footprint` divides by every tile level and uses floats. Do
  not port it.
- The placement path's `dev // n` agrees with this ceiling when `dev % 128 == 0`, `n <= 32` and
  `n` is the output partition. `_op_num_cores` folds in K splits, but a K-split writer is rejected
  (`utils.py:652-662`). Size in sticks holds for 128-byte sticks
  (`csrc/spyre_tensor_impl.cpp:295-310`); I did not check every format.

`read_count` is `max(0, len(uses) - (0 if first_use_is_read else 1))`, so it needs `uses` filled.

### `nodes.py`: edges and pins

```python
@dataclass(eq=False, kw_only=True)
class DataEdge:
    src: OutPort
    dst: InPort
    in_place: bool | None = None         # decision
    transforms: list[Transform] = field(default_factory=list)  # physical order
    mutation: bool = False               # a write through a mutation layout, not a read
    in_place_candidate: bool = False     # the division-invariant in-place gates hold

    def add_step(self, step: Transform) -> None: ...   # append; step.edges.append(self)
    def insert_step(self, index: int, step: Transform) -> None: ...
    def read_layout(self) -> FixedTiledLayout | None: ...

@dataclass(eq=False, kw_only=True)
class ExecutionEdge:
    before: OpNode
    after: OpNode
    cut: bool | None = None   # true unless both ops share a non-empty nest

INTERNAL_PIN_SOURCES = frozenset(
    {"graph input", "pre-pass", "layout", "hazard", "committed", "greedy"})
PIN_SOURCES = INTERNAL_PIN_SOURCES | {"hint"}

@dataclass(frozen=True)
class WorkSplitPin:
    split: WorkSplit             # factors by Axis, never a menu index
    reason: str                  # for logs and the infeasible warning
    seeds: tuple[str, ...] = ()  # fp8 matmul seeds

@dataclass(frozen=True)
class DimOrderPin:
    indirect_device_pos: int
    stride_map: tuple[int, ...]

@dataclass(frozen=True)
class DirectReadStick:
    axis: Axis
    eps: int

@dataclass(frozen=True, eq=False)
class Pin:
    target: OpNode | Port | DataEdge
    what: str       # see the table below
    value: Any      # a value, never a table index
    source: str     # one of PIN_SOURCES; __post_init__ raises ValueError otherwise

    @property
    def internal(self) -> bool: ...   # source in INTERNAL_PIN_SOURCES
```

| `what` | target | `value` |
|---|---|---|
| `stick` | `node.out` | `tuple[int, ...]`, `stick_dims(layout) or (SPARSE_STICK,)` |
| `restickify` | the edge into the `RestickifyNode` | `to_stick` |
| `split` | node | `WorkSplitPin` |
| `tiling` | node | `()`, a `tuple[TileLevel, ...]` |
| `loop_carry` | node | `LoopCarryRecord` |
| `drain` | storage node | `DrainPlan` |
| `opaque` | node | the op's type name |
| `dim_order` | the `table` port | `DimOrderPin` |

`what` is an open string so the table can grow, but `source` is checked. Sources are facts, and
reasons go in `WorkSplitPin.reason`. Stick pins from the committed layout use source `"layout"`,
so Stage 11 lifts exactly those; graph-input and pre-pass stick pins stay.

`read_layout()` starts from `dst.read_layout` when set, else `src.layout`, returns `None` if that is
`None`, and applies `transforms` in list order. With no steps it returns that start layout itself.
It is recomputed on each call, so no cache can go stale.

**Step order.** The list order of `DataEdge.transforms` is authoritative and physical. Nothing
sorts it. `ORDER` only places a step a decoder appends: `add_step` appends, and `insert_step` puts
a step at a position, so a relayout that reads the producer can go before a pinned restickify.
Both record the edge in `step.edges`.

### `optimization_graph.py`

```python
@dataclass(eq=False, kw_only=True)
class SpyreOptimizationGraph:
    capacity: int                  # LX bytes per core
    alignment: int = 128           # as MemoryPlanSolver's default
    max_cores: int = 1             # config.sencores
    divisions_committed: bool = False
    ir: GraphLowering | None = None
    drain_plans: Mapping[str, DrainPlan] = field(default_factory=dict)
    lx_relayout_plans: Sequence[LXRelayoutPlan] = ()
    nodes: list[OpNode] = field(default_factory=list)
    data_edges: list[DataEdge] = field(default_factory=list)
    execution_edges: list[ExecutionEdge] = field(default_factory=list)
    pins: list[Pin] = field(default_factory=list)

    def add_node(self, node: OpNode) -> OpNode: ...
    def node(self, name: str) -> OpNode: ...       # by buffer name
    def connect(self, out: OutPort, in_port: InPort) -> DataEdge: ...
    def pin(self, target: OpNode | Port | DataEdge, what: str, value: Any,
            source: str) -> Pin: ...
    def pins_on(self, target: OpNode | Port | DataEdge, what: str) -> list[Pin]: ...
    def step(self, node: OpNode) -> int: ...
    def read_ticks(self, node: OpNode) -> list[int]: ...
    def lifetime(self, target: OpNode | Relayout) -> tuple[int, int]: ...
    def steps(self) -> list[Transform]: ...
```

- `add_node` appends and raises `ValueError` on a repeated name. A running node also gets `tick` if
  it has none (the last running tick plus 1, or 0), raises `ValueError` unless ticks strictly
  increase, and is linked to the last running node by a new `ExecutionEdge`. An `InputNode` joins no
  chain, gets no tick, and keeps `prev`/`next` as `None`, per "Leaf `InputNode`s with downstream
  edges only". The sketch linked inputs into the chain; that does not carry over.
- `connect` raises `ValueError` if `in_port.edge` is set or either end is not in the graph. It sets
  `in_port.edge`, appends to `out.edges` and `data_edges`, and returns the edge. Two ports may read
  one out port.
- `pins_on` matches `target` by identity.
- `steps()` lists each `Transform` once, first seen across `data_edges`.

**`step(node)`** returns `node.tick` and raises `ValueError` on an `InputNode`. It is the
`graph.operations` index, matching `calculate_liveness` and `op_position`. It is a stored fact
because inputs sit in `nodes` without a tick, so a list position would be wrong.

**`read_ticks(node)`** is the sorted distinct `step(edge.dst.node)` over `node.out.edges`, skipping
`mutation` edges, which are writes. A pre-pass restickify is a node, so a producer's life ends at
it, which is its real reader today. On a built graph the ticks equal `uses` less the producing
write, and a test holds that.

**`lifetime(node)`** returns `(start, end)`, end exclusive.
- `ticks` is `node.uses`, or on a hand-built node with none, the node's own tick plus `read_ticks`.
- `start = ticks[0]` and `end = ticks[-1] + 1`. An `InputNode` with no reads raises `ValueError`,
  because such an input has no record today.
- Then `start = min(start, lifetime_start_override)` and `end = max(end, lifetime_end_override)`
  where set.

This is `LifetimeBoundBuffer.start_time` and `end_time`, with the same override semantics. A test
compares them directly.

**`lifetime(step)`** takes a `Relayout` only: from the earliest `step(edge.dst.node)` across
`step.edges` to the latest plus 1, matching `build_relayout_copy` (`plan_solver.py:893-895`). A
tiled op starting at its nest is a solve decision (Stage 6), and the placement path's half-ticks
(`allocator.py:1906`) are the encoder's. `lifetime` returns plain ticks.

## Changes

| file | change |
|---|---|
| `torch_spyre/_inductor/scratchpad/lx_allocator/__init__.py`, `graph/__init__.py` | new, licence only |
| `.../lx_allocator/graph/{flags,axes,transforms,nodes,optimization_graph}.py` | new |
| `tests/inductor/lx_graph_fixtures.py` | new: layout and small-graph builders, reused by Stages 2 and 3 |
| `tests/inductor/test_lx_graph_types.py` | new |
| `tests/configs/torch_spyre_tests/inductor/test_lx_graph_types_config.yaml` | new, same shape as `test_perm_layout_solver_config.yaml`: labels `[unit, regression, trunk]`, `unlisted_test_mode: mandatory_success` |

Ordered commits, each signed off (`git commit -s`), each leaving the suite green:

1. Package skeleton, `flags.py`, the test file with its config, and the import-boundary tests. The
   boundary lands first so every later commit is held to it.
2. `axes.py` and its tests.
3. `transforms.py`, `lx_graph_fixtures.py` and their tests.
4. `nodes.py` (ports, node types, edges, pins, `restrictions_for`) and its tests.
5. `optimization_graph.py`, with step, lifetime, connect and pin tests.

Unused code is exercised only by these tests until Stage 2. Run `make check-all-configs` and
`pre-commit run --files <changed>` before each push.

## Interfaces

Consumes:

| symbol | signature | from Stage | used for |
|---|---|---|---|
| harness `intree` tier | as Stage 0 names it | Stage 0 | the gate; no production call |
| `FixedTiledLayout` | `(device, dtype, size, stride, device_layout, offset=0)`, `.unallocated_copy()` | existing, `ir.py:89` | every `apply` |
| `SpyreTensorLayout` | `_C.pyi:186` | existing | `moved_stick`, tests |
| `restickify_device_size` | `(old_device_size, old_sd_outer_dim, old_sd_host_size, new_sd_outer_dim, new_sd_host_size, stick_size) -> list` | existing, `pass_utils.py:2271` | `moved_stick` |
| `restickify_stride_map` | `(old_stride_map, old_sd_outer_dim, old_sd_host_stride, new_sd_outer_dim, new_sd_host_stride, stick_size) -> list` | existing, `pass_utils.py:2290` | `moved_stick` |
| `PerCoreView` | `core_mapping.py:1061` | existing, type only | `Relayout`, `InputNode.prefetch_view` |

Provides. All paths are under `torch_spyre._inductor.scratchpad.lx_allocator.graph`. Every
dataclass except `Axis`, `WorkSplit`, `Pin` and the three pin value types is keyword-only, so the
signatures below list field names, not positions. "Filled by" names who writes a field; the field
itself is declared here.

| symbol | signature | for Stages | stability | filled by |
|---|---|---|---|---|
| `flags.Restriction` | `Flag`: `NONE` and the eleven members in Design | 2, 3, 5, 8, 9 | final | |
| `flags.PortFact` | `Flag`: `NONE`, `QFP8WT`, `STAGGERED_EA`, `OVERLAPPING_READ`, `DIRECT_READ_SOURCE` | 2, 3, 5, 11 | final | 2 |
| `axes.Axis` | `Axis(symbol, extent, reduction=False, granularity=None, off_output_coords=False, symbolic=False)`, `.name` | 2, 3, 5 | final; `granularity`, `symbolic` go with #5890 | 2 |
| `axes.TileLevel` | `tuple[Axis, int]` | 2, 3, 10 | final | |
| `axes.WorkSplit` | `WorkSplit(splits={}, tiling=())`, hashable; `.cores_used`, `.output_splits`, `.reduction_splits`, `.output_partition`, `.tile_count`, `.output_tile_count` | 2, 3, 5, 7, 8, 10 | final | |
| `axes.per_core_units` | `(size, splits, tiling, alignment) -> int` | 5 | final | |
| `transforms.SPARSE_STICK` | `-1` | 2, 3, 5, 11 | final | |
| `transforms.stick_dims` | `(layout) -> tuple[int, ...]`, the only layout stick reader | 2, 3, 5, 11 | final | |
| `transforms.moved_stick` | `(layout, to_stick) -> SpyreTensorLayout` | none; `Restickify.apply` calls it | final | |
| `transforms.Transform` | `resident, offset, footprint, pinned, edges`; `ORDER`, `forced`, `in_lx`; `apply(layout)` | 6-10 | final | 3, 7, 8 |
| `transforms.CopyOut` | `CopyOut(**transform_fields)` | 3, 6-8, 10 | final | 3, 7, 8 |
| `transforms.Restickify` | `Restickify(to_stick, realized_by=None, target=None, **transform_fields)` | 2, 3, 5-11 | final | 2, 11 |
| `transforms.Relayout` | `Relayout(to_view, reduction_split=1, source_view=None, source_footprint_bytes=0, destination_footprint_bytes=0, **transform_fields)` | 3, 6-10 | final | 3, 7, 8 |
| `nodes.Port` | `name, node, axis_map, facts, dep, coords, device_coords`; `current_layout()`, `dims_of(axis)`, `stuck_axes()` | 2, 3, 5, 11 | final | 2 |
| `nodes.InPort` | adds `operand, index, edge, read_layout, direct_read` | 2, 3, 5, 9, 11 | final | 2 |
| `nodes.OutPort` | adds `layout: FixedTiledLayout \| None, index, edges` | 2, 3, 5, 9, 11 | final | 2 |
| `nodes.OpNode` facts | `name, axes, size, op, tick, uses, lifetime_start_override, lifetime_end_override, graph_output, mutation_target, deferred_direct_read, span_floors, tile_local_axes`; `first_use_is_read`, `read_count` | 2-10 | final | 2 |
| `nodes.OpNode` decisions | `work_split, resident, offset, spill_reason`, in alignment units for `offset` | 3, 5-10 | final | 3, 7, 8 |
| `nodes.OpNode` structure | `prev, next, in_ports, out`; `port`, `add_in_port`, `ports`, `padded_reductions`, `whole_stick_factors`, `broadcast_dims`, `same_index_read`, `footprint`; `PORTS`, `VARIADIC`, `runs`, `restrictions` | 2-10 | final | |
| `nodes.PointwiseNode`, `OpaqueNode`, `RestickifyNode` | `Cls(name=..., operands=..., layout=..., ...)` | 2, 3, 5, 6, 8, 10, 11 | final | 2 |
| `nodes.MatmulNode` | adds `k: Axis \| None, n: Axis \| None, reduction_type: str` | 2, 3 | final | 2 |
| `nodes.ReductionNode` | adds `reduction_type: str, conv_params: dict[str, Any] \| None` | 2, 3, 5 | final | 2 |
| `nodes.IndirectNode` | adds `scatter: bool` | 2, 3 | final | 2 |
| `nodes.InputNode` | adds `prefetch_view: PerCoreView \| None` | 2, 3, 5-10 | final | 3, 7, 8 |
| `nodes.restrictions_for` | `(node, overrides=None) -> Restriction` | 3, 5 | final | |
| `nodes.RestrictionOverrides` | `Mapping[type[OpNode], Restriction]` | 3, 5, 9 | final | |
| `nodes.DataEdge` | `src, dst, in_place, transforms, mutation, in_place_candidate`; `add_step`, `insert_step`, `read_layout` | 2, 3, 5-11 | final | 2, 3, 7, 8 |
| `nodes.ExecutionEdge` | `before, after, cut` | 2, 3, 6-8 | final | 3, 7, 8 |
| `nodes.Pin` | `target, what, value, source`, `.internal` | 2, 3, 5, 6, 8, 9, 11 | final | 2 |
| `nodes.WorkSplitPin`, `DimOrderPin`, `DirectReadStick` | Design | 2, 3, 5, 6, 9 | final | 2 |
| `nodes.PIN_SOURCES`, `INTERNAL_PIN_SOURCES` | frozensets of the seven source strings | none; `Pin` validates against them | final | |
| `optimization_graph.SpyreOptimizationGraph` | `capacity, alignment, max_cores, divisions_committed, ir, drain_plans, lx_relayout_plans, nodes, data_edges, execution_edges, pins`; `add_node`, `node`, `connect`, `pin`, `pins_on`, `step`, `read_ticks`, `lifetime`, `steps` | 2-11, 13 | final | 2 |
| `tests/inductor/lx_graph_fixtures.py` | `fixed_layout(size, stick, dtype=torch.float16)`, `two_op_graph()` | 2, 3 | final | |

Rules other stages must follow:
- `Axis` is constructed by `symbol=`. `reduction` is `write.index.coeff(symbol) == 0` and drives
  `footprint`; `off_output_coords` is `symbol not in coord_vars` and drives the padded-reduction
  rule. They can differ, as they do today.
- Never compare `FixedTiledLayout`s with `==`. Compare `.device_layout`.
- A module under `graph/` may not import a solver at runtime, and that includes `plan_solver`.
- A `RestickifyNode` is judged and encoded as a node. Its pinned step gets nothing from `read_back`.
- Steps reach an edge through `add_step` or `insert_step`, never by assigning to `transforms`, or
  `Transform.edges` goes stale.

## Tests

All are in `tests/inductor/test_lx_graph_types.py`, none needs a device, and all use the
`_cache_fixed_layout` pattern. `lx_graph_fixtures.py` provides `fixed_layout(size, stick,
dtype=torch.float16)` and `two_op_graph()`, an input, a matmul and a pointwise node.

`TestFlags`:
- `test_restriction_members`: the eleven members, in order.
- `test_flags_per_type`: each of the seven node types' `restrictions` equals Design exactly.
- `test_setting_restrictions_on_instance_raises`, `test_setting_restrictions_on_type_raises`: for
  `PORTS` and `runs` too; `del` and `mock.patch.object` also raise.
- `test_subclass_with_non_flag_restrictions_rejected`.
- `test_override_replaces_flags_for_that_type_only`: the parent plan's example; `None` and `{}`
  give the type's flags.

`TestAxesAndWorkSplit`:
- `test_worksplit_drops_factor_one`, `test_tiling_order_matters`, `test_rejects_factor_below_one`,
  `test_worksplit_deepcopy_and_pickle`.
- `test_matches_core_division`: for a grid of splits and tilings, `cores_used`, `output_partition`,
  `tile_count` and `output_tile_count` equal a real `CoreDivision` and `TileSpec` of the same values.
- `test_decisions_need_no_table`: a `WorkSplit` from values alone drives `footprint` and a pin.

`TestTransforms`:
- `test_stick_dims`: either dim, same-size dims, a permuted write, a size-1 and a sparse stick; each
  against the stick symbol from `device_coordinates`.
- `test_stick_dims_qfp8wt`: two dims, matching `_qfp8wt_stick_coords`.
- `test_copyout_keeps_geometry_drops_allocation`, `test_relayout_sets_view_only`.
- `test_realized_restickify_returns_node_layout`: a padded layout comes back unchanged.
- `test_unpinned_restickify_target`: `target` wins over `moved_stick`.
- `test_unpinned_restickify_matches_from_scratch`: on the cases in Design.
- `test_moved_stick_short_new_stick_keeps_todays_arithmetic`: the `[320, 1, 5]` result.
- `test_moved_stick_rejects`: same dim, size-1 target, non-`STANDARD` arrangement, unresolved stick;
  each raises `ValueError`.
- `test_apply_never_mutates_input`: for all three steps, the input's `allocation` and `lx_view` are
  unchanged.
- `test_layout_eq_ignores_device_layout`: documents the trap. If upstream fixes it, this tells us.
- `test_order_and_class_flags`: `ORDER`, `forced`, `in_lx` per the Transforms diagram.

`TestPorts`:
- `test_ports_by_name`: `port("x")`, `port("table")`, `in0`; `InputNode` has none; an unknown name
  raises `KeyError` naming the node.
- `test_variadic_ports`: `PointwiseNode(operands=3)`, `ReductionNode(operands=2)`,
  `RestickifyNode`, `OpaqueNode` get `in0..`.
- `test_fixed_type_rejects_operands`, and `test_add_in_port`: a matmul `target` and an indirect
  `table1` bind; a duplicate raises.
- `test_gather_leaves_values_unbound`, `test_port_facts`, `test_abstract_opnode_rejected`.
- `test_in_port_layout_follows_edge`: a step on the edge changes `current_layout()` and
  `stuck_axes()`, and a `RestickifyNode`'s `in0` still reads the layout ahead of its own step.
- `test_read_layout_override`: `InPort.read_layout` replaces `src.layout` as the start, and steps
  still apply.
- `test_dims_of_flattened_read`: one axis reaches two dims from `coords`; `axis_map` keeps the
  outermost; `stuck_axes` sees either.

`TestNodeFacts`:
- `test_whole_stick_factors_matches_today`: over fixtures of real `TensorDep`s (fp16 and fp32 both
  stuck; fp32 stuck with fp16 not), `whole_stick_factors` equals
  `_legal_split_factors(v, adjust_it_space_for_sticks(...)[0][v])`.
- `test_padded_reductions_match_coordinate_mask`: agrees with `coordinate_mask_blocked_vars` on the
  same fixture, and uses `off_output_coords`, not `reduction`.
- `test_broadcast_dims`, `test_whole_stick_factors_granularity`: the symbolic basis wins.
- `test_same_index_read_truth_table`: same index; offset read; two reads of one buffer with one
  offset (both false); `out.index` unset; matmul always false.
- `test_footprint_matches_per_core_size`: over a grid including reduction splits and reduction tile
  levels, `footprint` equals both `ceil_div(CoreDivisionBuffer(size).per_core_size(0), 128)` and
  `CoreDivisionBuffer(ceil_div(size, 128)).per_core_size(0)`.
- `test_footprint_clamps_unsized`: size `-1` gives 0.
- `test_read_count_matches`: `read_count` and `first_use_is_read` equal a real `LifetimeBoundBuffer`.

`TestGraph`:
- `test_add_node_chains_running_nodes_only`, `test_step_rejects_input`, `test_node_by_name`.
- `test_explicit_ticks`: `step` returns them; a non-increasing tick raises.
- `test_connect`: double binding raises; two ports may read one out port.
- `test_lifetime_matches_lifetime_bound_buffer`: over a table of `uses` and both overrides,
  `lifetime(node)` equals a real `LifetimeBoundBuffer`'s `(start_time, end_time)`; inputs use
  `first_use_is_read=True`.
- `test_read_ticks_match_uses`: with a double read and a `mutation` edge.
- `test_unused_input_has_no_lifetime`.
- `test_restickify_node_and_step`: ticks 0, 1, 2 give producer `(0, 2)` and node `(1, 3)`; the
  step's `read_layout` is the node's layout, and it holds no footprint.
- `test_relayout_step_lifetime_matches_copy`: a step shared by two edges equals
  `build_relayout_copy`'s start and end.
- `test_read_layout_after_steps`: no steps returns `src.layout`; `[Restickify, Relayout]` and
  `[Relayout, Restickify]` built with `insert_step` give the expected `device_layout` and `lx_view`,
  in list order.
- `test_shared_step_listed_once`: `steps()` deduplicates and `edges` holds both.
- `test_pins_by_source`: all seven sources give the right `internal`; an unknown one raises;
  `pins_on` filters by target and `what`; a port target works.
- `test_small_graph_deepcopy_and_pickle`: `two_op_graph()` with `op=None` round-trips.

`TestGraphLayerImportsNoSolver`:
- `test_no_solver_import_in_source`: walks every `.py` under `graph/` with `ast`, including
  function-local imports, skipping only `if TYPE_CHECKING:` bodies. Relative imports are resolved.
  It fails on the roots `ortools` and `cpmpy`; on the scratchpad modules `ilp_solver_ortools`,
  `cpsat_expr_printer`, `plan_solver`, `allocator`, `greedy_solver`, `firstfit_bestfit_solver`,
  `exhaustive_search`, `simulated_annealing`, `sa_cooptimizer`, `cooling_schedules`; and on any
  `...lx_allocator.*` module outside `graph`. It also asserts the walk found at least the five
  modules, so a moved directory cannot make it pass vacuously.
- `test_no_solver_in_sys_modules`: a subprocess with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` imports torch
  and then each module under `graph/`, and asserts none of the same names is in `sys.modules`. This
  catches a solver arriving transitively. On a prototype of these modules the list came back empty.
- `test_package_inits_are_empty`: both `__init__.py` files parse to no statements.

**Existing suites.** The parent plan's unit list runs unchanged. No device suite is needed, since
nothing calls the new code.

**What proves the claim.** `TestGraphLayerImportsNoSolver` proves "the graph layer imports no
solver". "Flags frozen per type, overrides, ports by name, port facts, lifetimes, an edge's read
layout after its steps, and pins by source" map one to one to the test classes above.

## Parity and gates

The parent plan's gates:

> Each stage is its own PR on `main` and leaves the full suite green. Through Phase 1, and in
> Phase 2 wherever the rules agree with today's, the CP model stays byte-identical and the
> committed plans stay identical.

> **Every stage:** the Stage 0 harness, comparing proto, decisions and committed plans between
> the stage's parent and the change.

**How they are met.** This stage runs the `intree` tier and the import test, on the PR tip. Pass
means an empty diff in proto, status, objective, decisions and committed plan, and any difference
is a bug in this PR. It does not run the zoo, `matrix` or a card: nothing imports the package, so
those would prove nothing that `grep -r lx_allocator torch_spyre` outside the package does not.

Agreement with today's code is proven per function by the `matches` tests. They run on hand-built
fixtures, not real compiles. Real-graph agreement of `lifetime`, sizes and read sets is a Stage 2
gate.

## Risks

- **`==` on layouts passes silently.** Any later "same layout" check with `==` is true for
  differently stuck layouts. Caught by `test_layout_eq_ignores_device_layout`, the rule under
  Interfaces, and review. A wrong use would show as a missing restickify in Stage 3's check.
- **`stick_dims` trusts `stride_map`.** It returns `()` when no unique host dim matches, which reads
  as "not stuck" and widens a domain instead of failing. Unknown how often. Caught by Stage 3's
  comparison with `WorkDivisionContext` on the corpus.
- **`read_ticks` and `uses` diverge.** `lifetime` prefers `uses`, so a divergence would hide in
  `read_ticks`. Earlier work matched read sets 744/762, with the misses being multi-output plumbing
  (feasibility §3c). Caught by Stage 2's comparison with the records.
- **`deepcopy` of a big graph.** On a 400-node chain of the prototype, `copy.deepcopy` raised
  `RecursionError` through the port and edge links. A built graph also holds live ops in `op` and
  `graph.ir`, so it must not be deep-copied or pickled at all. Stage 0 records decisions, not graphs,
  and SA's `copy.deepcopy(ff_bufs)` (`sa_cooptimizer.py:1285`) must not be ported as a graph copy.
- **`read_layout` cost.** It builds a layout per step per call. Enumeration is the main planning
  cost, so Stage 5's timing run must include it. The fix is memoizing in the encoder, not the graph.
- **My import measurements used a foreign `_C.so`.** The snapshot has no extension, so I used the
  Oct 18 build from `/home/senuser/torch-spyre` with the snapshot's Python. Module lists come from
  Python imports and should hold. `_C` behaviour (layout construction, `stride_map`) is from that
  older build. The tests re-check all of it on the real build.

## Deviations from the parent plan

1. **`PORTS` as fixed tuples.** A pointwise node's arity varies, a fused reduction reads several
   operands (`wsr/loop_tiling_common.py:450-451`), and a scatter destination or matmul read-back fits
   no named port. Resolution: `VARIADIC` with `operands`, `PORTS = ("in0",)` on the variadic types,
   and `add_in_port` for extra ports.
2. **`MatmulNode.k`, `.n: Axis`.** With `n_size == 1` there is no loop symbol
   (`propagate_layouts.py:1539-1541`), and `k` cannot be built on the builder's `Unsupported` path.
   Resolution: both `Axis | None`. Stage 3's `MATMUL_STICKS` then requires a sparse stick on `y`, as
   `_matmul_layouts` does.
3. **`OutPort.layout: FixedTiledLayout` as always present.** Mutation aliases, host buffers and
   multi-output ops have none (`utils.py:256-271`; `allocator.py:1646-1648`). Earlier work counted 62
   such buffers in 762 (feasibility §3c). Resolution: `FixedTiledLayout | None`, and `size` keeps the
   `-1` sentinel.
4. **"An in port holds no layout of its own".** `_input_layout_overrides` makes an op read a graph
   input in another layout (`insert_restickify.py:940`). Resolution: `InPort.read_layout`, set only
   there.
5. **"Plain data", and `OpNode`'s member list.** Rules and encoders need the live op
   (`work_division_context_for_op`, `TileReads`), so the graph holds `OpNode.op` and `graph.ir`. It
   also holds the facts in Provides that the diagrams lack: `tick`, `uses`, overrides, `graph_output`,
   port coordinates, `OutPort.index` for `same_index_read`, `spill_reason` and the rest. None is a
   solver type. Inputs stay off the execution chain and raise in `step`.
6. **Two more node types, and a pre-pass restickify as a step alone.** Extern kernels and CPU buffers
   run and hold a tick but fit no parent type: `OpaqueNode`. A restickify op has its own division,
   residency and box, so a step alone cannot keep the model byte-identical: `RestickifyNode` plus
   `Restickify(pinned=True, realized_by=node)`. Because it is a node, `lifetime` takes `OpNode` and
   `Relayout` only.
7. **Canonical `ORDER`.** The plan gives `Restickify` ORDER 1 and `Relayout` ORDER 2. Both orders are
   legal in today's code: restickify then relayout (`lx_relayout.py:661-690`, with an end-to-end
   test) and relayout then restickify (`allocator.py:1513-1521`, whose only test is mocked). A fixed
   order cannot express that. Resolution: list order is physical, and `ORDER` only places a step a
   decoder appends.
8. **Six flags, and their spread.** Stage 3's rules for windows, conv, topk, keep_by_index and plain
   reductions have no member, and `_check_stick_all_same` covers reductions and gathers. Resolution:
   five more members, all on `ReductionNode`, and `POINTWISE_STICKS` on `ReductionNode` and
   `IndirectNode`. The `INDIRECT_ACCESS` comment loses its padding clause: no such exception exists.
9. **Additions for shared and decoded steps.** `add_step`, `insert_step`, `Transform.edges` and
   `steps()` record which edges share a step. `Relayout` gains `source_view` and its two byte
   footprints, because commit needs them and a menu record must stay in the encoder.
   `InputNode.prefetch_view` holds an LX prefetch's decision, since an input has no axes for a `WorkSplit`.
10. **Two more pin sources.** The Pins row names hints, graph-input arrival, the pre-pass, hazards
    and greedy pins. A committed layout's stick and a committed division are none of these.
    Resolution: sources `layout` and `committed`, both internal, and reasons in
    `WorkSplitPin.reason`, never in `source`.

## Open questions

Both must be settled before this PR merges, because every later parity gate depends on them. The
plan is written against the recommendation.

1. **Does a pre-inserted restickify stay a `RestickifyNode` plus a pinned step?** Recommendation:
   yes. It is the only form that keeps byte parity with no flag exemption, and the parent's "pinned
   step" still holds on the edge. If the answer is a step alone, `Restickify` needs the node's
   division, tick, size and residency, and Stages 2, 3, 5, 6, 7, 8 and 10 change.
2. **Is relayout-before-restickify a legal step order, with list order physical?** Recommendation:
   yes. It is legal in the code, and the only test of it is mocked. If the answer is no,
   `read_layout` sorts by `ORDER`, and Stage 6 must express the cross-frame hazard another way.

## Out of scope

- What any flag or fact requires, `Always`, `splits_while_stuck`, `stick_tables`, `check_decisions`,
  `coverage`, `RuleViolation`, the residency verdict and the adapter from today's plans: Stage 3.
- Deriving every "filled by 2" field from real ops, `classify`, and `WorkSplit` conversion: Stage 2.
- A per-tile frame for these facts: Stage 3's rows.
- `Sym` (in `graph/symbols.py`), the objective and everything priced: Stage 4.
- Encoders, models, solving, placement boxes and `read_back`: Stages 5 to 8.
- `completed_producer_cores`: derived at commit, Stage 10.
- `OutPort.candidates`, `DataEdge.sticks`, `Port.facts_for`, `graph.restickify_realized` and tick
  slots in `step`: Stage 11, in its own PR, since they use its types.
- `Transpose`: not built in this plan. It is in the diagram only to mark its place.
