"""LX allocator design skeleton: the code behind the design page.

One module so every snippet on the page is consistent with the others. It runs
as a smoke test with OR-Tools: `python3 lx_design.py`. Bodies that need the
real IR or cost model raise NotImplementedError and say where that code lives
on main today. It follows the decisions closed in the page's threads on
2026-10-28.
"""

from __future__ import annotations

import itertools
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Callable, ClassVar, Iterator

import sympy
from ortools.sat.python import cp_model

FACTORS = (1, 2, 4, 8, 16, 32)  # stand-in for WorkDivisionContext.factor_domain
MAX_CORES = 32
TILE_COUNTS = (2, 4)


# region Layout, Axis, WorkSplit
@dataclass(frozen=True)
class Axis:
    name: str
    extent: int  # element count
    reduction: bool = False


@dataclass(frozen=True)
class WorkSplit:
    """Decision: core division plus coarse tiling."""

    splits: dict[str, int]  # loop axis -> cores along it
    tiling: dict[str, int] = field(default_factory=dict)  # loop axis -> tile count


@dataclass(frozen=True)
class Layout:
    """Decision: the layout of a node's output, one per node (there are no
    ports). What a node needs from each input is a restriction; where it
    differs from the producer's layout, a transform on the edge converts.
    Sticks name buffer dimensions, so the two ends of an edge compare directly."""

    stick: int  # the stuck buffer dimension
    axis_order: tuple[int, ...] | None = None
    view: object | None = None  # canonical per-core view
# endregion


# region Restriction
class Restriction(Flag):
    """What is special about an op type's decisions, as flags a node type ORs
    together and freezes: MatmulNode.restrictions = OpNode.restrictions |
    MATMUL_STICKS | PADDED_REDUCTION_UNSPLIT. A flag only names the rule. Each
    family's encoder holds its encoding, check_decisions holds its check, and
    the data both read comes from the node's facts and the attributes its node
    type adds. Rules every op obeys are not flags (see Always)."""

    NONE = 0
    WHOLE_STICKS = auto()  # a stuck axis splits in whole sticks (main: factor_domain)
    PADDED_REDUCTION_UNSPLIT = auto()  # a padded reduction stays on one core while stuck (main: coordinate_mask_blocked_vars)
    MATMUL_STICKS = auto()  # x read stuck on K, y on N, the output on N (main: _matmul_layouts)
    POINTWISE_STICKS = auto()  # inputs that index the output's stick axis stick on it; broadcast dims exempt


RULE_TEXT = {
    Restriction.WHOLE_STICKS: "stuck axis splits in whole sticks",
    Restriction.PADDED_REDUCTION_UNSPLIT: "a padded reduction stays on one core while stuck",
    Restriction.MATMUL_STICKS: "matmul stickification",
    Restriction.POINTWISE_STICKS: "pointwise stickification",
}


class Always:
    """Rules every op obeys, so they are not flags: the encoder always applies
    them and check_decisions always checks them. Each name tags what it posts."""

    SPLIT_DIVIDES = "split divides the axis"
    CORE_BUDGET = "within the core budget"
    ONE_REDUCTION_SPLIT = "at most one reduction axis split"
    TILES = "tiles on output axes, never the stick dim"
    CUT_TILE = "a copied-out tile that stays resident has no reduction split"
    RESTICKIFY = "a restickify on an input only where the read stick differs"
    PINNED = "a pinned restickify on an input stays"
    COPY_OUT = "a copy-out it reads lives in HBM"
    IN_PLACE = "in place: same index, adjacent lifetimes, same layout and footprint"
    TILE_ALIGNED = "tiles line up with every input inside the nest"
    NO_OVERLAP = "resident boxes fit LX and never overlap"


# main's residency rules (scratchpad/residency_rules.py), also for every op: the
# encoder applies them while it builds the table, banning a division with
# residency rather than removing it
RESIDENCY_RULES: dict[str, Callable[[OpNode, WorkSplit | None], bool]] = {
    "empty tensor (main: residency rule 'empty')": lambda node, split: node.size == 0,
}


def restrictions_for(node: OpNode, overrides: dict | None = None) -> Restriction:
    """The flags frozen on the node's type, or a test's override for that type."""
    return (overrides or {}).get(type(node), type(node).restrictions)


# ---- what a flag asks of a node: data from its facts, read by encoder and check --
def splits_while_stuck(node: OpNode, flag: Restriction) -> list[tuple[int | None, int, str, tuple[int, ...]]]:
    """(operand, or None for the output; dim; axis; factors): while that is
    stuck on dim, the axis splits in one of factors."""
    if flag is Restriction.WHOLE_STICKS:
        out = [(i, d, a, node.whole_stick_factors(a, node.operand(i).src.elems_per_stick))
               for i, r in enumerate(node.reads) for a, d in r.items()]
        return out + [(None, d, a, node.whole_stick_factors(a, node.elems_per_stick))
                      for a, d in node.writes.items()]
    if flag is Restriction.PADDED_REDUCTION_UNSPLIT:
        return [(i, node.reads[i][a], a, (1,)) for i, a in node.padded_reductions()]
    return []


def stick_tables(node: OpNode, flag: Restriction) -> list[tuple[str, tuple, frozenset]]:
    """(rule, columns, rows): the sticks on these columns (operands, None for
    the output) are one of rows."""
    if flag is Restriction.MATMUL_STICKS:  # the axes come from MatmulNode's attributes
        return [(f"{label} sticks on {axis}", (col,), frozenset({(dims[axis],)}))
                for label, col, dims, axis in (("x", node.x, node.reads[node.x], node.k),
                                               ("y", node.y, node.reads[node.y], node.n),
                                               ("out", None, node.writes, node.n))]
    if flag is Restriction.POINTWISE_STICKS:
        def choices(i, axis):
            return ([node.reads[i][axis]] if axis in node.reads[i] else []) + node.broadcast_dims(i)
        rows = frozenset((*combo, out_dim) for axis, out_dim in node.writes.items()
                         for combo in itertools.product(*(choices(i, axis) for i in range(len(node.reads)))))
        return [(RULE_TEXT[flag], (*range(len(node.reads)), None), rows)]
    return []


# ---- helpers the always-on rules share ----------------------------------------------
def stuck(node: OpNode, operand: int | None) -> int:
    """The stuck buffer dim of what the node reads on an operand (after the
    edge's steps), or of its output when operand is None."""
    return node.layout.stick if operand is None else node.operand(operand).read_layout().stick


def nests(graph: SpyreOptimizationGraph) -> dict[OpNode, int]:
    """Nest ids from the decoded cuts: the id steps at every cut."""
    out, n = {graph.nodes[0]: 0}, 0
    for x in graph.execution_edges:
        n += bool(x.cut)
        out[x.after] = n
    return out


def legal_tiles(node: OpNode) -> list[tuple[str, int]]:
    """main: _tiling_candidates. One output axis, never a reduction axis or the
    innermost dim."""
    inner = len(node.shape) - 1
    return [(a.name, t) for a in node.axes if a.name in node.writes and not a.reduction
            and node.writes[a.name] != inner for t in TILE_COUNTS]


def tile_allowed(node: OpNode, splits: dict[str, int], tile: tuple[str, int] | None) -> bool:
    return tile is None or (tile in legal_tiles(node)
                            and node.axis(tile[0]).extent % (splits.get(tile[0], 1) * tile[1]) == 0)


def in_place_candidate(e: DataEdge, graph: SpyreOptimizationGraph) -> bool:
    """main: in_place_parents. The consumer reads the input at its own index,
    over a buffer of the same size, and is its last reader."""
    return (e.src.runs and e.dst.runs and e.dst.same_index_read(e.operand)
            and graph.lifetime(e.src)[1] == graph.step(e.dst) + 1 and e.src.size == e.dst.size)


def tile_aligned(e: DataEdge, axis: str) -> bool:
    """main: ilp_solver_ortools.py:985-992. What the consumer reads lines up
    with what the producer wrote along the tiled axis."""
    return axis in e.src.writes and e.dst.reads[e.operand].get(axis) == e.src.writes[axis]


def live_boxes(graph: SpyreOptimizationGraph) -> list[tuple[int, int, int, list]]:
    """(start, end, LX units, members) for everything decoded as resident in
    LX: node outputs, an in-place chain as one box, and resident steps."""
    root = {n: n for n in graph.nodes}

    def find(n):
        while root[n] is not n:
            n = root[n]
        return n
    for e in graph.data_edges:
        if e.in_place:
            root[find(e.dst)] = find(e.src)
    groups: dict[OpNode, list[OpNode]] = {}
    for n in graph.nodes:
        if n.resident:
            groups.setdefault(find(n), []).append(n)
    out = []
    for members in groups.values():
        spans = [graph.lifetime(n) for n in members]
        out.append((min(s for s, _ in spans), max(e for _, e in spans),
                    max(footprint(n, graph.alignment) for n in members), members))
    for e in graph.data_edges:
        for t in e.transforms or []:
            if t.in_lx and t.resident:
                at = graph.step(e.dst)
                out.append((at, at + 1, t.footprint, [t]))
    return out


def footprint(node: OpNode, alignment: int) -> int:
    """Per-core LX units of one tile of the node's output, from its decisions."""
    ws = node.work_split or WorkSplit({})
    return node.footprint(ws.splits, ws.tiling, alignment)


# ---- the check: every always-on rule, then each flag, on decoded decisions ---------
def _always(node: OpNode, graph: SpyreOptimizationGraph) -> Iterator[str]:
    ws, nest = node.work_split, nests(graph)
    if any(fn(node, ws) and node.resident for fn in RESIDENCY_RULES.values()):
        yield next(name for name, fn in RESIDENCY_RULES.items() if fn(node, ws))
    if not node.runs:
        return
    if any(node.axis(a).extent % f for a, f in ws.splits.items()):
        yield Always.SPLIT_DIVIDES
    if math.prod(ws.splits.values()) > MAX_CORES:
        yield Always.CORE_BUDGET
    if sum(ws.splits.get(a.name, 1) > 1 for a in node.axes if a.reduction) > 1:
        yield Always.ONE_REDUCTION_SPLIT
    if ws.tiling and (len(ws.tiling) != 1 or not tile_allowed(node, ws.splits, *ws.tiling.items())):
        yield Always.TILES
    leaves = not node.outputs or any(nest[e.dst] != nest[node] for e in node.outputs)
    split_r = any(ws.splits.get(a.name, 1) > 1 for a in node.axes if a.reduction)
    if ws.tiling and leaves and node.resident and split_r:
        yield Always.CUT_TILE
    for e in node.inputs:
        steps = e.transforms or []
        resticks = [t for t in steps if isinstance(t, Restickify)]
        if len(resticks) > 1 or any(t.to_stick == e.src.layout.stick for t in resticks):
            yield Always.RESTICKIFY
        if any(not any(isinstance(t, Restickify) and t.to_stick == p.value for t in steps)
               for p in graph.pins_on(e, "restickify")):
            yield Always.PINNED
        if any(isinstance(t, CopyOut) and t.resident for t in steps):
            yield Always.COPY_OUT
        if e.src.runs and e.src.work_split.tiling and nest[e.src] == nest[node]:
            (axis, _), = e.src.work_split.tiling.items()
            if not tile_aligned(e, axis):
                yield Always.TILE_ALIGNED
    over = [e for e in node.inputs if e.in_place]
    if len(over) > 1 or not all(
            in_place_candidate(e, graph) and e.src.resident and node.resident and not e.transforms
            and e.src.layout.stick == node.layout.stick
            and footprint(e.src, graph.alignment) == footprint(node, graph.alignment) for e in over):
        yield Always.IN_PLACE


def _flag_holds(node: OpNode, flag: Restriction) -> bool:
    splits = node.work_split.splits
    return (all(splits.get(axis, 1) in factors for operand, dim, axis, factors
                in splits_while_stuck(node, flag) if stuck(node, operand) == dim)
            and all(tuple(stuck(node, c) for c in cols) in rows for _, cols, rows in stick_tables(node, flag)))


def _no_overlap(graph: SpyreOptimizationGraph) -> bool:
    cap = graph.capacity // graph.alignment
    boxes = live_boxes(graph)
    for _s, _e, size, members in boxes:
        offs = {m.offset for m in members}
        if len(offs) != 1 or None in offs or offs.pop() + size > cap:
            return False
    for (sa, ea, za, ma), (sb, eb, zb, mb) in itertools.combinations(boxes, 2):
        oa, ob = ma[0].offset, mb[0].offset
        if sa < eb and sb < ea and oa < ob + zb and ob < oa + za:
            return False
    return True


def check_decisions(graph: SpyreOptimizationGraph, overrides: dict | None = None) -> list[str]:
    """Every always-on rule, then each flag on the node's type, on the
    decisions the graph holds now (decoded, or edited by hand). No solver."""
    out = []
    for node in graph.nodes:
        out += [f"{node.name}: {rule}" for rule in dict.fromkeys(_always(node, graph))]
        if node.runs:
            out += [f"{node.name}: {RULE_TEXT[f]}" for f in restrictions_for(node, overrides)
                    if not _flag_holds(node, f)]
    if not _no_overlap(graph):
        out.append(f"graph: {Always.NO_OVERLAP}")
    return out


def coverage(graph: SpyreOptimizationGraph, encoder: OrToolsEncoder, overrides: dict | None = None) -> list[str]:
    """Flags on the graph's node types that the encoder cannot encode."""
    missing = Restriction.NONE
    for node in graph.nodes:
        missing |= restrictions_for(node, overrides) & ~encoder.ENCODES
    return [f"{f.name}: no encoding in {type(encoder).__name__}" for f in missing]
# endregion


# region OpNode
@dataclass(eq=False)
class OpNode:
    """One op and the memory it writes. There is no Buffer class: the node
    holds its output's size and, once decoded, its residency and LX offset.
    Data, facts read off the data, decisions, and the restriction flags
    frozen for its op type, which the encoders read and the node never
    evaluates. Subtypes add the attributes their flags read. No solver types."""

    name: str
    axes: tuple[Axis, ...]
    reads: tuple[dict[str, int], ...]  # per operand: loop axis -> buffer dim it indexes
    writes: dict[str, int]  # loop axis -> output buffer dim
    shape: tuple[int, ...]  # output buffer
    size: int  # output bytes
    elems_per_stick: int = 64  # output dtype (fp16)
    inputs: list[DataEdge] = field(default_factory=list, repr=False)
    outputs: list[DataEdge] = field(default_factory=list, repr=False)
    prev: ExecutionEdge | None = field(default=None, repr=False)
    next: ExecutionEdge | None = field(default=None, repr=False)
    # decisions: decoded from the solution only after a solve returns a plan
    work_split: WorkSplit | None = None
    layout: Layout | None = None
    resident: bool | None = None
    offset: int | None = None  # decoded from the family's placement

    runs: ClassVar[bool] = True  # an op executes (False: a graph input)
    restrictions: ClassVar[Restriction] = Restriction.WHOLE_STICKS  # subtypes OR in theirs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.restrictions, Restriction), f"{cls.__name__}.restrictions must be a Restriction"

    def __setattr__(self, name, value):
        if name == "restrictions":  # frozen for the op type; a test overrides per type instead
            raise AttributeError(f"{type(self).__name__}.restrictions is frozen for its op type")
        super().__setattr__(name, value)

    def operand(self, i: int) -> DataEdge:
        return next(e for e in self.inputs if e.operand == i)

    def axis(self, name: str) -> Axis:
        return next(a for a in self.axes if a.name == name)

    # ---- facts: plain Python over the node's data, no decisions ------------
    def padded(self, axis: str, elems_per_stick: int) -> bool:
        """The axis is not a whole number of sticks at that dtype, so its last
        stick is partly padding."""
        return self.axis(axis).extent % elems_per_stick != 0

    def padded_reductions(self) -> list[tuple[int, str]]:
        """(operand, axis) pairs where a reduction axis ends in a padded stick
        of the operand's dtype. main's coordinate_mask_blocked_vars reads the same data."""
        return [(i, a) for i, r in enumerate(self.reads) for a in r
                if self.axis(a).reduction and self.padded(a, self.operand(i).src.elems_per_stick)]

    def whole_stick_factors(self, axis: str, elems_per_stick: int) -> tuple[int, ...]:
        """Core counts along `axis` that give every core whole sticks while it
        is stuck. main: factor_domain counts stick axes in sticks."""
        sticks = math.ceil(self.axis(axis).extent / elems_per_stick)
        return tuple(f for f in FACTORS if sticks % f == 0)

    def broadcast_dims(self, i: int) -> list[int]:
        """Buffer dims of operand i that no loop axis indexes (size-1 dims read
        by broadcast); a stick there is main's constant stick."""
        return [d for d in range(len(self.operand(i).src.shape)) if d not in self.reads[i].values()]

    def same_index_read(self, i: int) -> bool:
        """Operand i is read at the output's own index over a buffer of the
        same shape: the first condition for running in place over it."""
        return self.reads[i] == self.writes and self.operand(i).src.shape == self.shape

    def footprint(self, splits: dict[str, int], tiling: dict[str, int], alignment: int) -> int:
        """Per-core LX units of one tile of the output under a split and a
        tiling: its share of each written axis, divided by the tile count."""
        part = math.prod(splits.get(a, 1) for a in self.writes)
        tiles = math.prod(tiling.values()) if tiling else 1
        return math.ceil(self.size / part / tiles / alignment)
# endregion


# region PointwiseNode, MatmulNode, ReductionNode, InputNode
class PointwiseNode(OpNode):
    """Elementwise over its inputs, broadcasting size-1 dims."""

    restrictions = OpNode.restrictions | Restriction.POINTWISE_STICKS


@dataclass(eq=False)
class MatmulNode(OpNode):
    """out[M, N] = x[M, K] @ y[K, N]. Its own attributes say which operands are
    x and y and which loop axes are K and N; MATMUL_STICKS reads them."""

    x: int = 0
    y: int = 1
    k: str = "K"
    n: str = "N"

    restrictions = OpNode.restrictions | Restriction.MATMUL_STICKS | Restriction.PADDED_REDUCTION_UNSPLIT


class ReductionNode(OpNode):
    """Reduces one or more axes of its input."""

    restrictions = OpNode.restrictions | Restriction.PADDED_REDUCTION_UNSPLIT


class InputNode(OpNode):
    """A graph input or weight: a leaf on the execution and dependency graphs,
    with downstream edges only and no op to run. How it arrives (its stick) is
    a pin with source "graph input"."""

    runs = False
    restrictions = Restriction.NONE  # it runs nothing
# endregion


# region DataEdge
@dataclass(eq=False)
class DataEdge:
    """dst reads src as its operand-th input (an index, not a port). Two
    decisions: whether dst runs in place over src, and the transform steps
    that turn src's layout into what dst reads, in canonical order."""

    src: OpNode
    dst: OpNode
    operand: int
    in_place: bool | None = None  # decision
    transforms: list[Transform] | None = None  # decision

    def read_layout(self) -> Layout:
        """What dst reads: src's layout through the edge's steps."""
        layout = self.src.layout
        for t in self.transforms or []:
            layout = t.apply(layout)
        return layout
# endregion


# region ExecutionEdge
@dataclass(eq=False)
class ExecutionEdge:
    """Consecutive ops. A cut ends a coarse-tiling nest. Cuts are derived: one
    falls wherever the two ops' tiles don't match, and two ops share a nest
    only if every data edge between them lines up tile by tile."""

    before: OpNode
    after: OpNode
    cut: bool | None = None  # derived; decoded by read_back
# endregion


# region Transform
@dataclass(eq=False, kw_only=True)
class Transform(ABC):
    """One step on a data edge. It produces a buffer with its own memory, a
    footprint, residency and LX offset, placed like a node's output. Kinds
    differ in whether they are forced and whether they live in LX."""

    resident: bool | None = None  # decision
    offset: int | None = None  # decoded from the family's placement
    footprint: int = 0  # per-core LX units, decoded with the step
    pinned: bool = False  # present before the solve (a pre-inserted restickify)

    ORDER: ClassVar[int]  # canonical position in an edge's step list
    forced: ClassVar[bool]  # present whenever its trigger holds, never chosen
    in_lx: ClassVar[bool] = True  # False: lives in HBM and is never placed
    elidable: ClassVar[bool] = False  # a forced step the solver may drop once proven safe; none of these kinds

    @abstractmethod
    def apply(self, layout: Layout) -> Layout: ...
# endregion


# region CopyOut, Restickify, Relayout
@dataclass(eq=False, kw_only=True)
class CopyOut(Transform):
    """Forced when a cut lies between the ends (main: a cut tile). Its buffer
    lives in HBM, filled tile by tile through a mutation layout, so it is never
    placed in LX. One per producer, shared by every reader outside the nest,
    and priced as main's _cut_traffic_terms."""

    ORDER, forced, in_lx = 0, True, False

    def apply(self, layout):
        return layout


@dataclass(eq=False, kw_only=True)
class Restickify(Transform):
    """Forced where what dst reads is stuck differently from src's output. A
    restickify inserted before the graph is built becomes a pinned step. It
    needs no elision: where the sticks agree it is simply absent."""

    to_stick: int
    ORDER, forced = 1, True

    def apply(self, layout):
        return Layout(self.to_stick, layout.axis_order, layout.view)


@dataclass(eq=False, kw_only=True)
class Relayout(Transform):
    """Moves data between cores' LX into another per-core view. Only from a
    resident source (a spilled one reloads from HBM in the consumer's
    configuration); inside a tile when both ends share the nest, priced per
    tile times the trip count; never across a cut. Options are keyed by the
    producer's reduction split as well as the two layouts, so a
    completed-reduction route reads only the last core of each K group
    (main: derive_completed_reduction_routes)."""

    to_view: object
    reduction_split: int = 1
    ORDER, forced = 2, False

    def apply(self, layout):
        return Layout(layout.stick, layout.axis_order, self.to_view)
# endregion


# region Pin
@dataclass(frozen=True)
class Pin:
    """A constraint fixed before the solve that narrows its space; never stored
    as a decision and never dropped. Internal pins (how a graph input arrives,
    a pre-pass, a hazard such as a user-written loop, or the encoder pinning
    greedily) must never make the model infeasible. A user hint can: the solve
    is then returned infeasible with a warning that names the hints."""

    target: OpNode | DataEdge
    what: str  # "stick", "split" or "restickify"
    value: object
    source: str  # "graph input", "pre-pass", "hazard", "greedy" or "hint"

    INTERNAL: ClassVar[frozenset[str]] = frozenset({"graph input", "pre-pass", "hazard", "greedy"})

    @property
    def internal(self) -> bool:
        return self.source in self.INTERNAL

    @property
    def where(self) -> str:
        t = self.target
        return t.name if isinstance(t, OpNode) else f"{t.src.name} -> {t.dst.name}"
# endregion


# region SpyreOptimizationGraph
@dataclass(eq=False)
class SpyreOptimizationGraph:
    """Op nodes in execution order, data and execution edges, and pins. The
    facts come from here; decisions are decoded onto it only after a solve
    returns a plan, and commit lowers from it."""

    nodes: list[OpNode] = field(default_factory=list)
    data_edges: list[DataEdge] = field(default_factory=list)
    execution_edges: list[ExecutionEdge] = field(default_factory=list)
    pins: list[Pin] = field(default_factory=list)
    alignment: int = 128
    capacity: int = 2 * 1024 * 1024  # LX bytes per core

    def add_node(self, node: OpNode) -> OpNode:
        if self.nodes:
            x = ExecutionEdge(self.nodes[-1], node)
            self.nodes[-1].next, node.prev = x, x
            self.execution_edges.append(x)
        self.nodes.append(node)
        return node

    def add_read(self, src: OpNode, dst: OpNode, operand: int) -> DataEdge:
        e = DataEdge(src, dst, operand)
        src.outputs.append(e)
        dst.inputs.append(e)
        self.data_edges.append(e)
        return e

    def pin(self, target, what: str, value, source: str) -> Pin:
        p = Pin(target, what, value, source)
        self.pins.append(p)
        return p

    def pins_on(self, target, what: str) -> list[Pin]:
        return [p for p in self.pins if p.target is target and p.what == what]

    def step(self, node: OpNode) -> int:
        """The node's position in execution order."""
        if getattr(self, "_order_len", None) != len(self.nodes):  # rebuilt only when nodes change
            self._order = {n: i for i, n in enumerate(self.nodes)}
            self._order_len = len(self.nodes)
        return self._order[node]

    def lifetime(self, node: OpNode) -> tuple[int, int]:
        """Execution steps the node's output is live: its op through its last reader."""
        return self.step(node), max([self.step(e.dst) for e in node.outputs], default=self.step(node)) + 1
# endregion




# region CostModel, Sym
class Sym:
    """The names the cost model and the encoder share. They couple by name: the
    encoder registers a variable under each, and a name with no variable fails
    when the encoder encodes the objective."""

    @staticmethod
    def resident(n: OpNode) -> str:
        return f"resident[{n.name}]"

    @staticmethod
    def cores(n: OpNode) -> str:
        return f"cores[{n.name}]"

    @staticmethod
    def split(n: OpNode, axis: str) -> str:
        return f"split[{n.name}][{axis}]"

    @staticmethod
    def copy(n: OpNode) -> str:
        return f"copy[{n.name}]"

    @staticmethod
    def copy_resident(n: OpNode) -> str:
        return f"copy_resident[{n.name}]"

    @staticmethod
    def _edge(e: DataEdge) -> str:
        return f"{e.src.name}->{e.dst.name}.{e.operand}"

    @staticmethod
    def reads_full(e: DataEdge) -> str:
        return f"reads_full[{Sym._edge(e)}]"

    @staticmethod
    def restick(e: DataEdge) -> str:
        return f"restick[{Sym._edge(e)}]"

    @staticmethod
    def restick_resident(e: DataEdge) -> str:
        return f"restick_resident[{Sym._edge(e)}]"


class CostModel:
    """The objective as a sympy expression over named symbols, built above the
    line from the graph's data. The family's encoder consumes it and
    encodes it into its own variables (a CP-SAT expression here; a CPMpy
    expression, or lambdify for annealing, later). Bytes of HBM traffic,
    priced as main does: spill_cost for residency, _cut_traffic_terms for
    copy-outs. More cores only break ties."""

    K_EXCHANGE = 64 * 1024  # bytes per extra core on a matmul's K, standing in for main's cost model

    def objective(self, graph: SpyreOptimizationGraph) -> sympy.Expr:
        S, terms = sympy.Symbol, []
        for n in graph.nodes:
            terms.append(self.spill_cost(n) * (1 - S(Sym.resident(n))))
            if not n.runs:
                continue
            terms.append(-S(Sym.cores(n)))
            if isinstance(n, MatmulNode):
                terms.append(self.K_EXCHANGE * (S(Sym.split(n, "K")) - 1))
            terms += self.cut_traffic(n)
        for e in graph.data_edges:  # the shuffle once; a spilled step is written and read back
            r, rr = S(Sym.restick(e)), S(Sym.restick_resident(e))
            terms.append(e.src.size * r + 2 * e.src.size * (r - rr))
        return sympy.Add(*terms)

    @staticmethod
    def spill_cost(n: OpNode) -> int:
        """main: _LifetimeBufferWithCpVars.spill_cost. The reads residency would
        serve plus the producer's write; a graph input's first read is its
        clone-in, and a graph output's write-out happens either way."""
        reads = len(n.outputs) - (0 if n.runs else 1)
        intermediate = n.runs and bool(n.outputs)
        return (max(reads, 0) + int(intermediate)) * n.size

    @staticmethod
    def cut_traffic(n: OpNode) -> list[sympy.Expr]:
        """main: _cut_traffic_terms. A spilled cut tile is written and read back
        by the copy; a resident one costs the copy's write (unless it is the
        graph output's own) and every outside reader's read."""
        S = sympy.Symbol
        cut, rc = S(Sym.copy(n)), S(Sym.copy_resident(n))
        terms = [2 * n.size * (cut - rc), int(bool(n.outputs)) * n.size * rc]
        return terms + [n.size * S(Sym.reads_full(e)) for e in n.outputs]
# endregion




# region Backend, PlacementBox, Packer
@dataclass(frozen=True)
class SolveSettings:
    time_limit_s: float = 30.0  # tuned per use
    workers: int = 8
    deterministic: bool = False  # opt-in: budget by deterministic time with a fixed seed
    deterministic_budget: float = 20.0


class Solution(ABC):
    status: str
    ok: bool  # holds a plan (optimal, or an incumbent at the limit)
    objective: float | None
    offsets: dict  # LX offset per placed box: an OpNode, or (DataEdge, "restickify")

    @abstractmethod
    def value(self, x) -> int: ...


class Backend(ABC):
    """Private to its family. Sees the model, objective included, never the
    graph, and places the model's boxes its own way."""

    @abstractmethod
    def solve(self, model: Model, settings: SolveSettings) -> Solution: ...


class CpSatSolution(Solution):
    def __init__(self, solver: cp_model.CpSolver, status: int, offsets: dict | None = None):
        self.solver, self.status = solver, solver.StatusName(status)
        self.ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        self.objective = solver.ObjectiveValue() if self.ok else None
        self.offsets = offsets or {}

    def value(self, x):
        return int(x) if isinstance(x, (bool, int)) else self.solver.Value(x)


class CpSatBackend(Backend):
    """CP-SAT, private to the OR-Tools family. It places the model's boxes
    natively (an offset per box under a 2-D no-overlap) or by packing (a
    cumulative under a roof, then the Packer; if no packing fits, the roof
    comes down by the overflow and it solves again)."""

    def __init__(self, placement: str = "no_overlap", attempts: int = 4):
        assert placement in ("no_overlap", "pack")
        self.placement, self.attempts = placement, attempts

    def solve(self, model, settings):
        cp, cap = model.cp, model.capacity
        boxes = [b for b in model.boxes if b.present is not False]
        xs = [self._interval(cp, b.start, b.end - b.start, b.end, b.present) for b in boxes]
        if self.placement == "no_overlap":
            off = {id(b): cp.NewIntVar(0, cap, "") for b in boxes}
            key = {b.key: off[id(b)] for b in boxes}
            ys = []
            for b in boxes:
                o = off[id(b)]
                top = o + b.size if isinstance(b.size, int) else cp.NewIntVar(0, 2 * cap + 10**7, "")
                ys.append(self._interval(cp, o, b.size, top, b.present))
                c = cp.Add(o + b.size <= cap)
                if b.present is not True:
                    c.OnlyEnforceIf(b.present)
                for lit, other in b.ties:  # in place: the input's offset
                    cp.Add(o == key[other]).OnlyEnforceIf(lit)
            cp.AddNoOverlap2D(xs, ys)
            solver, status = self._run(cp, settings)
            s = CpSatSolution(solver, status)
            if s.ok:
                s.offsets = {b.key: solver.Value(off[id(b)]) for b in boxes if s.value(b.present)}
            return s
        roof, limit = cp.NewIntVar(0, cap, "roof"), cap
        cp.AddCumulative(xs, [b.size for b in boxes], roof)
        for _ in range(self.attempts):
            solver, status = self._run(cp, settings)
            s = CpSatSolution(solver, status)
            if not s.ok:
                return s
            items = [(b.key, s.value(b.start), b.end, s.value(b.size), [k for lit, k in b.ties if s.value(lit)])
                     for b in boxes if s.value(b.present)]
            s.offsets, top = Packer().pack(items)
            if top <= cap:
                return s
            limit -= top - cap
            cp.Add(roof <= limit)
        s.ok, s.status, s.offsets = False, "NO_PACKING", {}
        return s

    @staticmethod
    def _interval(cp, start, size, end, present):
        if isinstance(start, int) and isinstance(size, int):
            return (cp.NewFixedSizeIntervalVar(start, size, "") if present is True
                    else cp.NewOptionalFixedSizeIntervalVar(start, size, present, ""))
        return (cp.NewIntervalVar(start, size, end, "") if present is True
                else cp.NewOptionalIntervalVar(start, size, end, present, ""))

    @staticmethod
    def _run(cp, settings):
        solver = cp_model.CpSolver()
        p = solver.parameters
        p.num_workers = settings.workers
        if settings.deterministic:
            p.max_deterministic_time = settings.deterministic_budget
            p.random_seed, p.interleave_search = 0, True
        else:
            p.max_time_in_seconds = settings.time_limit_s
        return solver, solver.Solve(cp)


@dataclass(eq=False)
class PlacementBox:
    """One thing that may sit in LX, as the encoder records it in the model: a
    node's output or a transform step, with when it is live, how many LX
    units it takes and whether it is resident. The backend places it."""

    key: object  # an OpNode, or (DataEdge, "restickify")
    start: object  # a step, or a step plus an in-place literal
    end: int
    size: object  # LX units: int or variable
    present: object  # literal or bool
    ties: list = field(default_factory=list)  # (literal, key): the same offset as key while it holds


class Packer:
    """Offsets for the boxes a solve made resident: lowest offset first, in a
    few orders, keeping the best. A tied group (an op in place over its input)
    packs as one box. A detail of the backends that pack: CP-SAT with
    placement="pack", annealing, and MIP solvers."""

    ORDERS = {
        "start": lambda b: (b[0], -b[2]),
        "size": lambda b: (-b[2], b[0]),
        "area": lambda b: (-(b[1] - b[0]) * b[2], b[0]),
    }

    def pack(self, items: list[tuple]) -> tuple[dict, int]:
        """items: (key, start, end, size, keys tied to it). Returns the offset
        per key and the LX units the best packing reaches."""
        root = {it[0]: it[0] for it in items}

        def find(k):
            while root[k] != k:
                k = root[k]
            return k
        for k, *_rest, tied in items:
            for other in tied:
                root[find(k)] = find(other)
        groups: dict = {}
        for k, start, end, size, _ in items:
            g = groups.setdefault(find(k), [start, end, size, []])
            g[0], g[1], g[2] = min(g[0], start), max(g[1], end), max(g[2], size)
            g[3].append(k)
        boxes = [tuple(g) for g in groups.values()]
        best = None
        for key in self.ORDERS.values():
            placed, top = self._first_fit(sorted(boxes, key=key))
            if best is None or top < best[1]:
                best = (placed, top)
        if best is None:
            return {}, 0
        return {k: off for b, off in best[0] for k in b[3]}, best[1]

    @staticmethod
    def _first_fit(order):
        placed, top = [], 0
        for b in order:
            busy = sorted((o, o + p[2]) for p, o in placed if b[0] < p[1] and p[0] < b[1])
            y = 0
            for lo, hi in busy:
                if y + b[2] <= lo:
                    break
                y = max(y, hi)
            placed.append((b, y))
            top = max(top, y + b[2])
        return placed, top
# endregion


# region Model
class Model(ABC):
    """A family's encoded model: its constraints, the rule behind each, the
    symbol table the cost model's names resolve through, and the boxes its
    backend places."""

    provenance: list[tuple[str, str, int | None]]  # (subject, rule, constraint index; None if folded)
    pruned: list[tuple[str, str, int]]  # (node, rule, table entries it removed)
    symbols: dict[str, object]
    boxes: list[PlacementBox]
    capacity: int  # LX units per core


class OrToolsModel(Model):
    def __init__(self, capacity: int) -> None:
        self.cp = cp_model.CpModel()
        self.provenance, self.pruned, self.symbols, self.boxes = [], [], {}, []
        self.capacity = capacity
# endregion


# region Encoder
class Encoder(ABC):
    """Public: encode and read_back. The model is the same whatever backend
    runs it, and read_back decodes only a solution that holds a plan, so a
    failed solve leaves the graph exactly as built."""

    @abstractmethod
    def encode(self, graph: SpyreOptimizationGraph, restrictions: dict | None = None) -> Model:
        """restrictions: a test's override of a node type's flags; by default each type's own."""

    @abstractmethod
    def read_back(self, solution: Solution, graph: SpyreOptimizationGraph) -> None: ...
# endregion




# region OrToolsEncoder
class OrToolsEncoder(Encoder):
    """CP-SAT. Encodes the always-on rules at every level (op, edge, op-op,
    graph) and the flags frozen on each node type (ENCODES says which it
    knows): each op's work-division table enumerated as main does (lazy is on
    the roadmap), pruned by what holds unconditionally and by the residency
    rules, conditional rules posted under their literal. It records the boxes
    the backend places, may pin decisions greedily, and consumes the cost
    model, encoding its objective into its own variables. Its variables are
    private; read_back decodes the solution onto the graph, offsets included."""

    def __init__(self, coarse_tiling: bool = False, cost_model: CostModel | None = None):
        self.coarse_tiling = coarse_tiling  # main: config.auto_coarse_tiling, off by default
        self.cost_model = cost_model or CostModel()

    # ---- public ----------------------------------------------------------
    def encode(self, graph, restrictions=None):
        self.m, self.align = OrToolsModel(graph.capacity // graph.alignment), graph.alignment
        self.v: dict[OpNode, dict] = {}
        self.e: dict[DataEdge, dict] = {}
        self.x: dict[ExecutionEdge, object] = {}
        self._loops: dict[tuple, int] = {}  # interned loop nests; 0 is untiled
        self.g, self.rs = graph, restrictions  # restrictions: a test's override of a node type's flags
        for node in graph.nodes:  # op: sticks, pinned ones folded
            self._stick(node)
        for e in graph.data_edges:  # what each op reads on each input
            self._read_stick(e)
        for node in graph.nodes:  # op: the table, stick tables, residency rules
            self._table(node)
        for e in graph.data_edges:  # an op's inputs: restickify, its buffer, in place
            self._edge(e)
        self._nests(graph)  # op-op: cuts, tile alignment, copy-outs
        self._placement(graph)  # graph: the boxes the backend places
        self._objective(graph)  # the cost model, in this model's variables
        return self.m

    def read_back(self, solution, graph):
        if not solution.ok:
            return  # nothing to decode: the graph stays as built
        s = solution
        for node, v in self.v.items():
            node.work_split = v["read"](s)
            node.layout = Layout(s.value(v["stick"]))
            node.resident = bool(s.value(v["resident"]))
            node.offset = s.offsets.get(node) if node.resident else None
        copies = {}
        for e, d in self.e.items():
            steps = []
            fp = s.value(self.v[e.src]["fp"])
            if s.value(self.v[e.src]["copy"]) and s.value(d["outside"]):
                steps.append(copies.setdefault(e.src, CopyOut(footprint=fp)))
            if s.value(d["restick"]):
                resident = bool(s.value(d["step_resident"]))
                steps.append(Restickify(to_stick=s.value(d["read_stick"]), footprint=fp,
                                        pinned=bool(graph.pins_on(e, "restickify")), resident=resident,
                                        offset=s.offsets.get((e, "restickify")) if resident else None))
            e.transforms = sorted(steps, key=lambda t: t.ORDER)
            e.in_place = bool(s.value(d["in_place"]))
        for x in graph.execution_edges:
            x.cut = not s.value(self.x[x])

    ENCODES: ClassVar[Restriction] = (Restriction.WHOLE_STICKS | Restriction.PADDED_REDUCTION_UNSPLIT
                                      | Restriction.MATMUL_STICKS | Restriction.POINTWISE_STICKS)

    # ---- op level -------------------------------------------------------------
    def _stick(self, node):
        cp = self.m.cp
        pins = self.g.pins_on(node, "stick")
        if pins:  # a pin narrows the space; folded into a constant here
            stick = pins[0].value
            self.m.provenance.append((node.name, f"pin ({pins[0].source}): stick {stick}", None))
        else:
            stick = cp.NewIntVarFromDomain(cp_model.Domain.FromValues(range(len(node.shape))),
                                           f"{node.name}.stick")
        self.v[node] = {"stick": stick, "resident": cp.NewBoolVar(f"{node.name}.resident")}
        self.symbol(Sym.resident(node), self.v[node]["resident"])

    def _read_stick(self, e):
        pins = self.g.pins_on(e, "restickify")
        if pins:
            r = pins[0].value
            self.m.provenance.append((Sym._edge(e), f"pin ({pins[0].source}): restickify to {r}", None))
        else:
            r = self.m.cp.NewIntVarFromDomain(cp_model.Domain.FromValues(range(len(e.src.shape))),
                                             f"{Sym._edge(e)}.read_stick")
        self.e[e] = {"read_stick": r}

    def _stick_of(self, node, operand):
        return self.v[node]["stick"] if operand is None else self.e[node.operand(operand)]["read_stick"]

    def _table(self, node):
        v = self.v[node]
        flags = restrictions_for(node, self.rs)
        if not node.runs:
            v.update(split={a.name: 1 for a in node.axes}, cores=1, loop=0, index=0,
                     fp=node.footprint({}, {}, self.align), read=lambda s: None)
            self._residency(node, [None])
            return
        names = [a.name for a in node.axes]
        tiles = [None] + (legal_tiles(node) if self.coarse_tiling else [])
        menu = [{"split": dict(zip(names, c)), "tile": t}
                for c in itertools.product(FACTORS, repeat=len(names)) for t in tiles]

        def prune(rule, keep):
            nonlocal menu
            kept = [m for m in menu if keep(m)]
            if not kept:
                self.post(node.name, rule, False)  # nothing left: infeasible, and named
            elif len(kept) < len(menu):
                self.m.pruned.append((node.name, rule, len(menu) - len(kept)))
                menu = kept

        for p in self.g.pins_on(node, "split"):  # a pin narrows the table; never dropped
            prune(f"pin ({p.source}): split {p.value}",
                  lambda m, v=p.value: all(m["split"].get(a) == f for a, f in v.items()))
        prune(Always.SPLIT_DIVIDES, lambda m: all(node.axis(a).extent % f == 0 for a, f in m["split"].items()))
        prune(Always.CORE_BUDGET, lambda m: math.prod(m["split"].values()) <= MAX_CORES)
        prune(Always.ONE_REDUCTION_SPLIT, lambda m: sum(m["split"][a.name] > 1 for a in node.axes if a.reduction) <= 1)
        later = []
        for flag in flags:  # the node type's flags: splits while something is stuck
            for operand, dim, axis, factors in splits_while_stuck(node, flag):
                lit = self.eq(self._stick_of(node, operand), dim)
                if lit is True:
                    prune(RULE_TEXT[flag], lambda m, a=axis, f=factors: m["split"][a] in f)
                elif lit is not False:
                    later.append((RULE_TEXT[flag], axis, factors, lit))
        prune(Always.TILES, lambda m: tile_allowed(node, m["split"], m["tile"]))
        for flag in flags:  # and stick tables
            for rule, cols, rows in stick_tables(node, flag):
                self.post_table(node.name, rule, [self._stick_of(node, c) for c in cols], rows)

        index, column = self._index(node, menu)
        v["index"], v["menu"] = index, menu
        v["split"] = {a: column([m["split"][a] for m in menu], f"split[{a}]") for a in names}
        for rule, axis, factors, lit in later:
            self.post_in(node.name, rule, v["split"][axis], factors, lit)
        v["cores"] = column([math.prod(m["split"].values()) for m in menu], "cores")
        v["loop"] = column([self.loop_id(m["tile"]) for m in menu], "loop")
        v["fp"] = column([node.footprint(m["split"], dict([m["tile"]]) if m["tile"] else {}, self.align)
                          for m in menu], "footprint")
        v["read"] = lambda s, menu=menu, index=index: WorkSplit(
            dict(menu[s.value(index)]["split"]),
            dict([menu[s.value(index)]["tile"]]) if menu[s.value(index)]["tile"] else {})
        self.symbol(Sym.cores(node), v["cores"])
        for a in names:
            self.symbol(Sym.split(node, a), v["split"][a])
        self._residency(node, [WorkSplit(m["split"], dict([m["tile"]]) if m["tile"] else {}) for m in menu])

    def _residency(self, node, entries):
        """main's residency rules, applied while the table is built: a
        division under which the node may not be resident is banned with
        residency, not removed from the table."""
        v = self.v[node]
        for name, blocks in RESIDENCY_RULES.items():
            blocked = [i for i, ws in enumerate(entries) if blocks(node, ws)]
            if len(blocked) == len(entries):
                self.post(node.name, name, v["resident"] == 0)
            elif blocked:
                allowed = [i for i in range(len(entries)) if i not in blocked]
                self.post_in(node.name, name, v["index"], allowed, v["resident"])

    def _index(self, node, menu):
        cp = self.m.cp
        index = 0 if len(menu) == 1 else cp.NewIntVar(0, len(menu) - 1, f"{node.name}.division")

        def column(values, name):
            if len(set(values)) == 1:  # main: one-entry elements made presolve slow
                return values[0]
            x = cp.NewIntVarFromDomain(cp_model.Domain.FromValues(sorted(set(values))), f"{node.name}.{name}")
            cp.AddElement(index, values, x)
            return x
        return index, column

    # ---- edge level -------------------------------------------------------------
    def _edge(self, e):
        d = self.e[e]
        src, dst = self.v[e.src], self.v[e.dst]
        name = Sym._edge(e)
        d["restick"] = restick = self.neq(src["stick"], d["read_stick"])
        if self.g.pins_on(e, "restickify"):
            self.post(name, Always.PINNED, restick if isinstance(restick, bool) else restick == 1)
        self.symbol(Sym.restick(e), restick)
        if restick is False:
            d["step_resident"] = False
        else:
            d["step_resident"] = sr = self.m.cp.NewBoolVar(f"{name}.restick.resident")
            if restick is not True:
                self.m.cp.AddImplication(sr, restick)
        self.symbol(Sym.restick_resident(e), d["step_resident"])
        d["in_place"] = False
        if in_place_candidate(e, self.g):
            d["in_place"] = ip = self.m.cp.NewBoolVar(f"{name}.in_place")
            rule = Always.IN_PLACE
            for x in (src["resident"], dst["resident"]):
                self.post(name, rule, x == 1, ip)
            self.post(name, rule, (not restick) if isinstance(restick, bool) else restick == 0, ip)
            self.post(name, rule, src["fp"] == dst["fp"], ip)  # a plain bool when both are constants
            self.post(name, rule, src["stick"] == dst["stick"], ip)
        if e is e.dst.inputs[-1]:  # at most one in-place input per op
            ips = [self.e[x]["in_place"] for x in e.dst.inputs if self.e[x]["in_place"] is not False]
            if len(ips) > 1:
                self.post(e.dst.name, Always.IN_PLACE, sum(ips) <= 1)

    # ---- op-op level: nests, cuts, copy-outs ---------------------------------------
    def _nests(self, graph):
        """Cuts, nests and copy-outs, all derived from the tilings chosen.
        joined: both ops run the same non-empty nest. A running group id steps
        at every cut, so two ops share a nest exactly when their ids match.
        main: _add_tiling_groups, _cut_traffic_terms, _add_partial_reduction_cut_guard."""
        cp, v = self.m.cp, self.v
        group = {graph.nodes[0]: 0}
        for x in graph.execution_edges:
            a, b = v[x.before]["loop"], v[x.after]["loop"]
            self.x[x] = j = self.all([self.eq(a, b), self.neq(a, 0)])
            if isinstance(j, bool) and isinstance(group[x.before], int):
                group[x.after] = group[x.before] + (0 if j else 1)
            else:
                g = cp.NewIntVar(0, len(graph.nodes), f"{x.after.name}.nest")
                cp.Add(g == group[x.before] + (1 - j if not isinstance(j, bool) else 0 if j else 1))
                group[x.after] = g
        for e in graph.data_edges:
            self.e[e]["outside"] = out = self.neq(group[e.src], group[e.dst])
            if e.src.runs and e.dst.runs and out is not True:
                for (axis, count), lid in self._loops.items():
                    if not tile_aligned(e, axis):
                        self.post(Sym._edge(e), Always.TILE_ALIGNED, v[e.src]["loop"] != lid,
                                 True if out is False else out.Not())
        for node in graph.nodes:
            res = v[node]["resident"]
            tiled = self.neq(v[node]["loop"], 0)
            leaves = [self.e[e]["outside"] for e in node.outputs] + ([True] if not node.outputs else [])
            v[node]["copy"] = c = self.all([tiled, self.any(leaves)])
            self.symbol(Sym.copy(node), c)
            self.symbol(Sym.copy_resident(node), self.all([c, res]))
            for e in node.outputs:
                self.symbol(Sym.reads_full(e), self.all([c, self.e[e]["outside"], res]))
            if c is not False:
                for a in node.axes:
                    s = v[node]["split"][a.name]
                    if a.reduction and not (isinstance(s, int) and s == 1):
                        self.post(node.name, Always.CUT_TILE, s == 1, self.all([c, res]))

    # ---- the objective --------------------------------------------------------------
    def _objective(self, graph):
        """The cost model's objective, encoded into this model's variables:
        each symbol resolved through the names registered while encoding
        (main: cpsat_expr_printer's sym_map). An unknown name fails here."""
        self.m.cp.Minimize(to_cpsat(self.cost_model.objective(graph), self.m.symbols))

    # ---- graph level: the boxes to place -------------------------------------------
    def _placement(self, graph):
        """No overlap, as the boxes the family's backend places: each node's
        output and each restickify step's, with when it is live, its LX units
        and whether it is resident. An op in place over its input starts a
        step later and shares the input's offset."""
        v = self.v
        for node in graph.nodes:
            t, end = graph.lifetime(node)
            edges = [e for e in node.inputs if self.e[e]["in_place"] is not False]
            start = t + self.any([self.e[e]["in_place"] for e in edges]) if edges else t  # one literal: affine
            self.m.boxes.append(PlacementBox(node, start, end, v[node]["fp"], v[node]["resident"],
                                            [(self.e[e]["in_place"], e.src) for e in edges]))
        for e in graph.data_edges:
            if self.e[e]["step_resident"] is not False:
                t = graph.step(e.dst)
                self.m.boxes.append(PlacementBox((e, "restickify"), t, t + 1, v[e.src]["fp"],
                                                self.e[e]["step_resident"]))
        self.m.provenance.append(("graph", Always.NO_OVERLAP, None))  # placed by the backend

    # ---- vocabulary -------------------------------------------------------------
    def symbol(self, name: str, x) -> None:
        self.m.symbols[name] = x

    def post(self, subject, rule, expr, when=True):
        """Post `expr` under `when`, tagged with the rule. Either may already
        be a plain bool (a pinned stick, a constant column): then it folds."""
        if when is False or expr is True:
            return
        cp = self.m.cp
        if expr is False:
            c = cp.AddBoolOr([when.Not()]) if when is not True else cp.Add(cp.NewConstant(0) == 1)
        else:
            c = cp.Add(expr)
            if when is not True:
                c.OnlyEnforceIf(when)
        self.m.provenance.append((subject, rule, c.Index()))

    def post_in(self, subject, rule, x, values, when=True):
        if isinstance(x, int):
            return self.post(subject, rule, x in values, when)
        if when is False:
            return
        c = self.m.cp.AddLinearExpressionInDomain(x, cp_model.Domain.FromValues(values))
        if when is not True:
            c.OnlyEnforceIf(when)
        self.m.provenance.append((subject, rule, c.Index()))

    def post_table(self, subject, rule, xs, rows, when=True):
        if all(isinstance(x, int) for x in xs):
            return self.post(subject, rule, tuple(xs) in rows, when)
        xs = [self.m.cp.NewConstant(x) if isinstance(x, int) else x for x in xs]
        c = self.m.cp.AddAllowedAssignments(xs, sorted(rows))
        if when is not True:
            c.OnlyEnforceIf(when)
        self.m.provenance.append((subject, rule, c.Index()))

    def loop_id(self, tile) -> int:
        """Intern a loop nest by its tiled axis and count; 0 is untiled."""
        return 0 if tile is None else self._loops.setdefault(tile, len(self._loops) + 1)

    def eq(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        lit = self.m.cp.NewBoolVar("")
        self.m.cp.Add(a == b).OnlyEnforceIf(lit)
        self.m.cp.Add(a != b).OnlyEnforceIf(lit.Not())
        return lit

    def neq(self, a, b):
        lit = self.eq(a, b)
        return (not lit) if isinstance(lit, bool) else lit.Not()

    def all(self, lits):
        if any(l is False for l in lits):
            return False
        lits = [l for l in lits if l is not True]
        if not lits:
            return True
        if len(lits) == 1:
            return lits[0]
        both = self.m.cp.NewBoolVar("")
        self.m.cp.AddBoolAnd(lits).OnlyEnforceIf(both)
        self.m.cp.AddBoolOr([l.Not() for l in lits]).OnlyEnforceIf(both.Not())
        return both

    def any(self, lits):
        if any(l is True for l in lits):
            return True
        lits = [l for l in lits if l is not False]
        if not lits:
            return False
        if len(lits) == 1:
            return lits[0]
        either = self.m.cp.NewBoolVar("")
        self.m.cp.AddBoolOr(lits).OnlyEnforceIf(either)
        self.m.cp.AddBoolAnd([l.Not() for l in lits]).OnlyEnforceIf(either.Not())
        return either


def to_cpsat(expr: sympy.Expr, symbols: dict[str, object]):
    """A linear sympy expression as a CP-SAT one, each symbol resolved by name
    (main: cpsat_expr_printer's sym_map). An unknown name fails here."""
    out = 0
    for term, coeff in sympy.expand(expr).as_coefficients_dict().items():
        if term == 1:
            out += int(coeff)
        elif not term.is_Symbol:
            raise NotImplementedError(f"nonlinear cost term {term}: main's printer handles more")
        elif term.name not in symbols:
            raise KeyError(f"cost symbol {term.name} has no variable in the model")
        else:
            out += int(coeff) * symbols[term.name]
    return out
# endregion


# region CpmpyEncoder, SAEncoder
class CpmpyEncoder(Encoder):
    """An expansion axis after OR-Tools ships, and possibly its replacement:
    the same always-on rules and node-type flags, posted as CPMpy expressions.
    CPMpy has no 2-D no-overlap global, and a MIP solver behind it could not
    use one well, so this family packs."""

    def encode(self, graph, restrictions=None):
        raise NotImplementedError("expansion axis")

    def read_back(self, solution, graph):
        raise NotImplementedError("expansion axis")


class SAEncoder(Encoder):
    """An expansion axis: moves over the graph's decisions, each rule scored as
    a penalty with the same checks check_decisions uses, and packs.
    The fallback for a solve with no plan does not need it: it anneals the
    OR-Tools model."""

    def encode(self, graph, restrictions=None):
        raise NotImplementedError("expansion axis")

    def read_back(self, solution, graph):
        raise NotImplementedError("expansion axis")
# endregion


# region ModelAbstractFactory, OrToolsFactory, CpmpyFactory, SAFactory
class ModelAbstractFactory(ABC):
    """One family as one object that encodes and solves. It creates the
    family's encoder and keeps its backend private, so a model can only reach
    a backend that can solve it. Chosen from config, so the family is in the
    FX graph cache key."""

    @abstractmethod
    def encoder(self, cost_model: CostModel | None = None) -> Encoder: ...

    @abstractmethod
    def solve(self, model: Model, settings: SolveSettings) -> Solution: ...

    def fallback(self, model: Model, settings: SolveSettings) -> Solution:
        """Simulated annealing over the same model, for a solve that ran out of
        time with no plan; it packs the boxes (branch: scratchpad/backend.py
        Annealing, _C.pack_boxes). A model proven infeasible is not annealed:
        the optimizer returns it infeasible."""
        raise NotImplementedError("simulated annealing over the encoded model")


class OrToolsFactory(ModelAbstractFactory):
    """The OR-Tools encoder and a private CP-SAT backend that places boxes
    natively or by packing."""

    def __init__(self, placement: str = "no_overlap", **options):
        self.options = options  # coarse_tiling: with placement, the family's configuration
        self._backend = CpSatBackend(placement)

    def encoder(self, cost_model=None):
        return OrToolsEncoder(cost_model=cost_model, **self.options)

    def solve(self, model, settings):
        return self._backend.solve(model, settings)


class CpmpyFactory(ModelAbstractFactory):
    def encoder(self, cost_model=None):
        return CpmpyEncoder()

    def solve(self, model, settings):
        raise NotImplementedError("expansion axis")


class SAFactory(ModelAbstractFactory):
    def encoder(self, cost_model=None):
        return SAEncoder()

    def solve(self, model, settings):
        raise NotImplementedError("expansion axis")


FACTORIES: dict[str, type[ModelAbstractFactory]] = {
    "ortools": OrToolsFactory, "cpmpy": CpmpyFactory, "sa": SAFactory,
}  # chosen from config, so the choice is in the FX graph cache key
# endregion


# region SpyreOptimizer
class RuleViolation(Exception):
    pass


class SpyreOptimizer:
    """Builds the graph, coordinates the solve, checks and commits. No solver
    types, and no placement: the family places. LxContextSwitchingPass stays
    a separate pass after commit."""

    def __init__(self, family: str = "ortools", settings: SolveSettings = SolveSettings(),
                 cost_model: CostModel | None = None, **options):
        self.family = FACTORIES[family](**options)
        self.settings, self.cost_model = settings, cost_model or CostModel()

    def plan_allocation(self, ir) -> SpyreOptimizationGraph:
        return self.optimize(self.build_graph(ir))

    def optimize(self, graph: SpyreOptimizationGraph, restrictions: dict | None = None) -> SpyreOptimizationGraph:
        """Solve and commit. A model proven infeasible is returned as it is,
        nothing decoded or committed, with a warning naming the user hints:
        internal pins are never infeasible, and no pin is dropped."""
        encoder = self.family.encoder(self.cost_model)  # it encodes the objective
        self.model = encoder.encode(graph, restrictions)
        self.solution = self.family.solve(self.model, self.settings)
        if self.solution.status == "INFEASIBLE":
            hints = [f"{p.what} {p.value} on {p.where}" for p in graph.pins if not p.internal]
            warnings.warn(f"no LX plan satisfies the restrictions and pins; user hints: {hints or 'none'}",
                          stacklevel=2)
            return graph
        if not self.solution.ok:  # out of time with no plan, or no packing fit
            self.solution = self.family.fallback(self.model, self.settings)
        encoder.read_back(self.solution, graph)
        failures = check_decisions(graph, restrictions)  # no solver involved
        if failures:
            raise RuleViolation(failures)
        self.commit(graph)
        return graph

    def build_graph(self, ir) -> SpyreOptimizationGraph:
        raise NotImplementedError("one node per op, axis-to-dim maps from index expressions, pins")

    def commit(self, graph) -> None:
        raise NotImplementedError("insert each edge's steps in order (pinned ones exist already), "
                                  "commit divisions and offsets")
# endregion


# region Scenarios
def _input(g: SpyreOptimizationGraph, name: str, dims: tuple[int, ...]) -> InputNode:
    """A graph input arriving stuck on its last dim: a pin, not a decision."""
    n = g.add_node(InputNode(name, axes=tuple(Axis(f"d{i}", e) for i, e in enumerate(dims)), reads=(),
                             writes={f"d{i}": i for i in range(len(dims))}, shape=dims,
                             size=math.prod(dims) * 2))
    g.pin(n, "stick", len(dims) - 1, "graph input")
    return n


def swiglu_mlp(tokens=128, hidden=4096, inter=12800) -> SpyreOptimizationGraph:
    """down(silu(x @ W_gate) * (x @ W_up)) at an 8B-class size, fp16.
    Weights arrive as nn.Linear weights [out, in], so each is read transposed:
    its natural stick is `in` (K) but a matmul's y must stick on N. The
    restickify pre-pass has already inserted those three restickifies
    (issue #4208), so they arrive as pinned steps."""
    g = SpyreOptimizationGraph()

    def mm(name, M, K, N):
        return g.add_node(MatmulNode(
            name, axes=(Axis("M", M), Axis("K", K, reduction=True), Axis("N", N)),
            reads=({"M": 0, "K": 1}, {"K": 1, "N": 0}),  # y = weight[out=N, in=K]
            writes={"M": 0, "N": 1}, shape=(M, N), size=M * N * 2))
    x = _input(g, "x", (tokens, hidden))
    w_gate, w_up, w_down = _input(g, "W_gate", (inter, hidden)), _input(g, "W_up", (inter, hidden)), \
        _input(g, "W_down", (hidden, inter))
    gate, up = mm("gate", tokens, hidden, inter), mm("up", tokens, hidden, inter)
    h = g.add_node(PointwiseNode(
        "silu_mul", axes=(Axis("M", tokens), Axis("N", inter)),
        reads=({"M": 0, "N": 1}, {"M": 0, "N": 1}), writes={"M": 0, "N": 1},
        shape=(tokens, inter), size=tokens * inter * 2))
    down = mm("down", tokens, inter, hidden)
    for src, dst, i in [(x, gate, 0), (w_gate, gate, 1), (x, up, 0), (w_up, up, 1),
                        (gate, h, 0), (up, h, 1), (h, down, 0), (w_down, down, 1)]:
        e = g.add_read(src, dst, i)
        if src in (w_gate, w_up, w_down):
            g.pin(e, "restickify", 0, "pre-pass")  # stick on N, the weight's dim 0
    return g


def sdpa_decode_core(heads=32, kv_len=100, head_dim=128) -> SpyreOptimizationGraph:
    """scores = q @ k^T, max over keys, exp(scores - max), sum over keys,
    probs @ v for one decode token, fp16, with a KV length that is not a whole
    number of 64-element sticks. The division by the sum is left out. The max
    is kept as [h, 1], so its stick can sit on the size-1 dim (main's constant
    stick) and broadcast into the subtraction. The padded key stick is the case
    superdsc._get_coordinate_mask masks (#3248): it cannot be split across
    cores where it is reduced. No pre-pass here: the solver places the
    restickify k needs."""
    g = SpyreOptimizationGraph()
    q, k, v = _input(g, "q", (heads, head_dim)), _input(g, "k", (heads, kv_len, head_dim)), \
        _input(g, "v", (heads, kv_len, head_dim))
    scores = g.add_node(MatmulNode(
        "scores", axes=(Axis("h", heads), Axis("K", head_dim, reduction=True), Axis("N", kv_len)),
        reads=({"h": 0, "K": 1}, {"h": 0, "N": 1, "K": 2}), writes={"h": 0, "N": 1},
        shape=(heads, kv_len), size=heads * kv_len * 2))
    row_max = g.add_node(ReductionNode(
        "max", axes=(Axis("h", heads), Axis("L", kv_len, reduction=True)),
        reads=({"h": 0, "L": 1},), writes={"h": 0}, shape=(heads, 1), size=heads * 2))
    sub_exp = g.add_node(PointwiseNode(
        "sub_exp", axes=(Axis("h", heads), Axis("L", kv_len)),
        reads=({"h": 0, "L": 1}, {"h": 0}), writes={"h": 0, "L": 1},
        shape=(heads, kv_len), size=heads * kv_len * 2))
    total = g.add_node(ReductionNode(
        "sum", axes=(Axis("h", heads), Axis("L", kv_len, reduction=True)),
        reads=({"h": 0, "L": 1},), writes={"h": 0}, shape=(heads, 1), size=heads * 2))
    pv = g.add_node(MatmulNode(
        "probs_v", axes=(Axis("h", heads), Axis("K", kv_len, reduction=True), Axis("N", head_dim)),
        reads=({"h": 0, "K": 1}, {"h": 0, "K": 1, "N": 2}), writes={"h": 0, "N": 1},
        shape=(heads, head_dim), size=heads * head_dim * 2))
    for src, dst, i in [(q, scores, 0), (k, scores, 1), (scores, row_max, 0),
                        (scores, sub_exp, 0), (row_max, sub_exp, 1), (sub_exp, total, 0),
                        (sub_exp, pv, 0), (v, pv, 1)]:
        g.add_read(src, dst, i)
    return g


class Sketch(SpyreOptimizer):
    """The optimizer with an empty commit (no IR here)."""

    def commit(self, graph):
        pass


def plan(g: SpyreOptimizationGraph, title: str, coarse_tiling: bool = False,
         placement: str = "no_overlap") -> SpyreOptimizationGraph:
    opt = Sketch(settings=SolveSettings(deterministic=True, workers=2), coarse_tiling=coarse_tiling,
                 placement=placement)
    gaps = coverage(g, opt.family.encoder(opt.cost_model))
    assert not gaps, gaps
    opt.optimize(g)
    m, s = opt.model, opt.solution
    print(f"\n== {title}")
    how = "a native 2-D no-overlap" if placement == "no_overlap" else "packing after the solve"
    print(f"  {s.status}, objective {int(s.objective)} (bytes of HBM traffic, less one per core); "
          f"placed by {how}")
    for n in g.nodes:
        if not n.runs:
            continue
        pruned = sum(k for node, _, k in m.pruned if node == n.name)
        posted = sum(1 for node, _, _ in m.provenance if node == n.name)
        tiles = f" tiles={n.work_split.tiling}" if n.work_split.tiling else ""
        place = f"LX offset {n.offset}" if n.resident else "HBM"
        print(f"  {n.name:9} {pruned:5} table entries pruned, {posted} constraints posted")
        print(f"            split={n.work_split.splits}{tiles} stick={n.layout.stick} {place}")
    for e in g.data_edges:
        if e.transforms or e.in_place:
            steps = [f"{type(t).__name__}{'(pinned)' if t.pinned else ''}"
                     f"{' in LX' if t.in_lx and t.resident else ''}" for t in e.transforms]
            print(f"  {e.src.name} -> {e.dst.name}[{e.operand}]: {steps}{' in place' if e.in_place else ''}")
    joined = [f"{x.before.name}+{x.after.name}" for x in g.execution_edges if not x.cut]
    if joined:
        print("  same nest:", joined)
    outputs = [n.name for n in g.nodes if n.runs and not n.outputs and n.work_split.tiling]
    if outputs:
        print("  tiled graph outputs, copied out to HBM:", outputs)
    print("  check:", check_decisions(g) or "all restrictions hold")
    return g
# endregion


if __name__ == "__main__":
    plan(swiglu_mlp(), "SwiGLU MLP, 128 tokens, hidden 4096, intermediate 12800")
    plan(sdpa_decode_core(), "SDPA decode core, 32 heads, KV length 100, head dim 128")
    plan(swiglu_mlp(tokens=4096), "SwiGLU MLP prefill, 4096 tokens, AUTO_COARSE_TILING=1",
         coarse_tiling=True, placement="pack")
