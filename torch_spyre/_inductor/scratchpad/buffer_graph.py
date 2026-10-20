# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The CP-SAT layout model's input as nodes, edges and op order.

Every buffer the allocator hands the solver -- placement-only, core-division or
relayout copy -- becomes one :class:`CoreDivisionBoundBuffer`. What the buffer
classes used to tell apart is data here: a decision made before the solve is a
filled-in field on the node, and the solver finds every field left ``None``.
How buffers depend on each other is a :class:`BufferGraph` of edges, and where
ops sit in time is its :class:`Timeline`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import sympy

from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    CoreDivisionBuffer,
    LifetimeBoundBuffer,
    RelayoutCopyBuffer,
    ceil_div,
    division_symbol,
)

if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.lx_relayout import RelayoutCandidate

# (producer division, consumer division) index pairs.
Pairs = tuple[tuple[int, int], ...]


def spill_cost(buffer: LifetimeBoundBuffer, size: Optional[int] = None) -> int:
    """Differential HBM traffic a spill adds over residency, in units of
    ``size`` (default ``buffer.size``).

    That is the reads residency would serve from LX plus the producer's write,
    which residency turns into a free LX write. A graph input has no producer
    write, and a graph output's write-out happens either way, so neither pays
    it; a plain :class:`LifetimeBoundBuffer` tracks no boundary, and
    ``first_use_is_read`` marks inputs instead. An input's first read is the
    clone-in that residency cannot avoid, so it is not counted as served.
    """
    boundary = getattr(buffer, "boundary", None)
    is_intermediate = (
        boundary == BufferType.Intermediate
        if boundary is not None
        else not buffer.first_use_is_read
    )
    reads_served = buffer.read_count - (1 if buffer.first_use_is_read else 0)
    return (reads_served + (1 if is_intermediate else 0)) * (
        buffer.size if size is None else size
    )


@dataclass
class CoreDivisionBoundBuffer:
    """A buffer bound to the core divisions it may take.

    ``size`` is the total footprint across cores, in alignment units.
    ``core_divisions`` is never empty: a buffer whose division was fixed
    upstream has the one undivided candidate. ``division``, ``resident`` and
    ``offset`` are the decisions, each fixed before the solve or ``None`` for
    the solver to fill in.
    """

    name: str
    size: int
    core_divisions: tuple[CoreDivision, ...]
    start_time: int
    end_time: int
    spill_cost: int
    boundary: BufferType = BufferType.Intermediate
    # The cost model's unknowns for this buffer's splits and tile counts.
    sym_core_divs: dict = field(default_factory=dict)
    sym_tile_counts: dict = field(default_factory=dict)
    division: Optional[int] = None
    resident: Optional[bool] = None
    offset: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.core_divisions, f"{self.name} has no core division"
        if len(self.core_divisions) == 1 and self.division is None:
            self.division = 0
        assert self.division is None or 0 <= self.division < len(
            self.core_divisions
        ), f"{self.name}: division {self.division} is not a candidate"

    @property
    def sym_is_lx(self) -> sympy.Symbol:
        return sympy.Symbol(f"is_lx_{self.name}", integer=True, nonnegative=True)

    @property
    def sym_division(self) -> sympy.Symbol:
        return division_symbol(self.name)

    @property
    def per_core_sizes(self) -> list[int]:
        """Per-core LX footprint under each division. A coarse tiling keeps one
        tile resident at a time, so it shrinks the footprint too."""
        return [
            ceil_div(self.size, cd.output_partition * cd.tiling.output_tile_count)
            for cd in self.core_divisions
        ]

    @property
    def cores_used(self) -> list[int]:
        return [cd.cores_used for cd in self.core_divisions]

    @property
    def shape_costs(self) -> list[int]:
        """Summed squared split factors per division: for a fixed core count,
        smallest when the split is spread over more axes (2x2 before 4x1)."""
        return [sum(s**2 for s in cd.splits.values()) for cd in self.core_divisions]

    @property
    def tileable(self) -> bool:
        return any(not cd.tiling.is_untiled for cd in self.core_divisions)


@dataclass(frozen=True)
class RelayoutRead:
    """The read on a :class:`SliceEdge` can go through relayout copy
    ``copy`` under one of ``candidates``' division pairs."""

    copy: str
    candidates: tuple["RelayoutCandidate", ...]

    @property
    def group(self) -> int:
        return self.candidates[0].group

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return [(c.source_division, c.consumer_division) for c in self.candidates]


@dataclass(frozen=True)
class SliceEdge:
    """``consumer`` reads ``producer``. While the producer is resident the two
    divisions must be one of ``pairs``, which slice it alike, or the read must
    go through a resident copy in ``relayouts``. No pairs means no division of
    the consumer reads the producer in place."""

    producer: str
    consumer: str
    pairs: Pairs
    relayouts: tuple[RelayoutRead, ...] = ()


@dataclass(frozen=True)
class InPlaceEdge:
    """``child`` may take over ``parent``'s storage at the tick they share.

    ``pairs`` are the division pairs that slice the storage alike, which with
    equal per-core footprints is what lets the child reuse it exactly. ``None``
    when the child's division was fixed upstream: then
    ``_check_in_place_relationships`` has already checked that it fits.
    """

    parent: str
    child: str
    pairs: Optional[Pairs]


@dataclass(frozen=True)
class CopyEdge:
    """``copy`` is a relayout destination of ``source``, priced only under
    the source divisions in ``priced``."""

    source: str
    copy: str
    priced: tuple[int, ...]


@dataclass(frozen=True)
class Timeline:
    """Op order: the tick at which each op output is written."""

    produced_at: Mapping[str, int]

    def runs(self, names: Iterable[str]) -> list[list[str]]:
        """The op outputs among ``names`` in op order, split wherever two are
        not written on consecutive ticks. ``names`` not written by an op (graph
        inputs, relayout copies) are dropped."""
        ordered = sorted(
            (n for n in names if n in self.produced_at),
            key=lambda n: self.produced_at[n],
        )
        runs: list[list[str]] = []
        for name in ordered:
            if runs and self.produced_at[name] == self.produced_at[runs[-1][-1]] + 1:
                runs[-1].append(name)
            else:
                runs.append([name])
        return runs


@dataclass
class BufferGraph:
    """Nodes, the edges between them, and the op order they live in. Edges are
    listed in the order the solver adds their constraints."""

    nodes: dict[str, CoreDivisionBoundBuffer]
    slices: list[SliceEdge]
    in_place: list[InPlaceEdge]
    copies: dict[str, CopyEdge]
    timeline: Timeline

    def __post_init__(self) -> None:
        self._consumers: dict[str, list[SliceEdge]] = {}
        for edge in self.slices:
            self._consumers.setdefault(edge.producer, []).append(edge)

    def consumers_of(self, producer: str) -> list[SliceEdge]:
        return self._consumers.get(producer, [])


def _chooses_division(buffer: LifetimeBoundBuffer) -> bool:
    return isinstance(buffer, CoreDivisionBuffer) and bool(buffer.core_divisions)


def _node(buffer: LifetimeBoundBuffer, alignment: int) -> CoreDivisionBoundBuffer:
    units = ceil_div(buffer.size, alignment)
    common = dict(
        name=buffer.name,
        size=units,
        start_time=buffer.start_time,
        end_time=buffer.end_time,
        spill_cost=spill_cost(buffer, units),
        boundary=getattr(buffer, "boundary", BufferType.Intermediate),
    )
    if not _chooses_division(buffer):
        return CoreDivisionBoundBuffer(core_divisions=(CoreDivision(),), **common)
    assert isinstance(buffer, CoreDivisionBuffer)
    return CoreDivisionBoundBuffer(
        core_divisions=tuple(buffer.core_divisions),
        sym_core_divs=buffer.sym_core_divs,
        sym_tile_counts=buffer.sym_tile_counts,
        **common,
    )


def build_buffer_graph(
    buffers: Sequence[LifetimeBoundBuffer],
    alignment: int,
    barred: Iterable[str] = (),
) -> BufferGraph:
    """The graph of ``buffers``, sized in ``alignment`` units, with every
    decision that is already known fixed on its node: the division of a buffer
    with one candidate, and non-residency for the ``barred`` buffers and for
    every relayout copy that can never fire.

    Edges need both ends in this solve. Slicing and relayout edges also need
    both to choose a division: one fixed upstream slices however it was fixed,
    which the allocator has already reconciled.
    """
    nodes = {b.name: _node(b, alignment) for b in buffers}
    for name in barred:
        nodes[name].resident = False
    order = {b.name: i for i, b in enumerate(buffers)}
    joint = {b.name for b in buffers if _chooses_division(b)}
    copy_of_group = {
        b.group_key: b.name for b in buffers if isinstance(b, RelayoutCopyBuffer)
    }

    slices: list[SliceEdge] = []
    for consumer in buffers:
        if consumer.name not in joint:
            continue
        assert isinstance(consumer, CoreDivisionBuffer)
        for producer in consumer.parents:
            if producer not in joint:
                continue
            by_group: dict[tuple[str, int], list["RelayoutCandidate"]] = {}
            for c in consumer.cd_parent_relayouts.get(producer, ()):
                if c.group_key in copy_of_group:
                    by_group.setdefault(c.group_key, []).append(c)
            slices.append(
                SliceEdge(
                    producer=producer,
                    consumer=consumer.name,
                    pairs=tuple(consumer.cd_parent_matches.get(producer, ())),
                    relayouts=tuple(
                        RelayoutRead(copy_of_group[key], tuple(cands))
                        for key, cands in sorted(by_group.items())
                    ),
                )
            )
    slices.sort(key=lambda e: order[e.producer])

    read_copies = {read.copy for e in slices for read in e.relayouts}
    copies: dict[str, CopyEdge] = {}
    for b in buffers:
        if not isinstance(b, RelayoutCopyBuffer):
            continue
        if b.relayout_parent in joint and b.name in read_copies:
            copies[b.name] = CopyEdge(
                source=b.relayout_parent,
                copy=b.name,
                priced=tuple(sorted(b.cost_by_source_division)),
            )
        else:
            nodes[b.name].resident = False

    in_place = [
        InPlaceEdge(
            parent=parent,
            child=child.name,
            pairs=tuple(child.cd_parent_matches.get(parent, ()))
            if isinstance(child, CoreDivisionBuffer) and child.name in joint
            else None,
        )
        for child in buffers
        for parent in child.in_place_parents
        if parent in nodes
    ]

    timeline = Timeline(
        produced_at={
            b.name: b.uses[0]
            for b in buffers
            if b.uses
            and not b.first_use_is_read
            and not isinstance(b, RelayoutCopyBuffer)
        }
    )
    return BufferGraph(nodes, slices, in_place, copies, timeline)
