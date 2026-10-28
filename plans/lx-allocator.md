# LX allocator: one optimizer over op nodes, restriction flags and solver families

## Context

`plans/cpsat-graph-buffer-refactor.md` models the CP-SAT solve as a graph of buffer
nodes, dependency edges and execution edges, and outlines a Phase 3 in which every
solver moves onto that one problem definition. torch-spyre/torch-spyre#5676 (branch
`scratchpad-refactor-experimental`, a draft marked "DO NOT MERGE") built that plan end
to end. Its prototype includes:

- `buffer_graph.py`: `ResourceNode`, `DependencyEdge`, `ExecutionEdge`, `Transform`,
  and `SpyreOptimizationGraph.solve(model, cost_expr, strategy)`;
- `encoding_strategy.py` and `lazy_strategy.py`: `EncodingStrategy`, with the
  `Enumerated`, `PairTable` and `Lazy` strategies;
- `graph_builder.py`: `GraphBuilder`;
- `backend.py`: the recorded-model backends `CpSat`, `Mip`, `BruteForce` and
  `Annealing`, selected by `SPYRE_LX_BACKEND`;
- `pricing.py`: `cost_expr`.

The open `plans/cpsat-graph-staged-conversion.md` (this repo's #4, decisions of
2026-10-23) explains why torch-spyre/torch-spyre#5676 cannot land as it is:

- it is too large to review;
- default compiles plan 21–30% slower;
- the greedy fallback is gone;
- completed-reduction relayouts are lost;
- it never ported torch-spyre/torch-spyre#5246.

It instead lands the structure on `main` in 12 PRs that keep the CP model
byte-identical.

This plan is the target that path leads to. It settles the earlier plan's Phase 3. Its
design (the LX allocator design, 34 questions closed on 2026-10-28) keeps the graph but
changes what sits on it and what solves it:

- nodes are ops, subclassed by op kind;
- what is special about an op type is a set of `Restriction` flags frozen on its node
  type;
- the encoding strategy is replaced by a solver family whose encoder applies the rules
  and whose private backend solves and places;
- legality is judged after the solve by `check_decisions`, with no solver involved.

The goal is one optimizer that decides work division, sticks, restickify, relayout,
copy-out, in place, coarse tiling and LX placement in one solve. The optimizer then
checks the plan without a solver and commits it. It replaces the placement-only
solvers, the duplicated stick rules (`propagate_layouts`, `optimize_restickify`,
`op_spec_validation`) and the `SolveError` → greedy re-plan path.

The design is written out as a runnable sketch in **`probes/lx-allocator/`**
(`lx_design.py`, about 1,500 lines, OR-Tools 9.15, sympy 1.14). It plans three real
cases end to end:

- a SwiGLU MLP whose transposed `nn.Linear` weights arrive with the restickifies the
  pre-pass inserted (torch-spyre/torch-spyre#4208);
- SDPA decode with a padded KV stick that cannot be split where it is reduced
  (torch-spyre/torch-spyre#3248);
- 4096-token prefill with `AUTO_COARSE_TILING=1`, placed by packing.

Each case solves to OPTIMAL and passes `check_decisions`. The design page (UML,
sequence, encodings, a 50-row map from current `main` symbols to the design, and the
decision log) is a private claude.ai artifact:
<https://claude.ai/artifact/UNVmhkwDhbnHpfazJ1VmKj>.

Work lands on torch-spyre `main` in small PRs, under the gates of the staged
conversion. The branch is a source to lift from where `main` has not moved past it,
never a base. File references to `main` are at torch-spyre `6487062` (2026-10-27), and
references to the branch are at `4999f73`.

## What changes relative to the graph refactor

| `cpsat-graph-buffer-refactor` / torch-spyre/torch-spyre#5676 | this plan |
|---|---|
| `ResourceNode`: one buffer an op needs, with its menu | `OpNode`, subclassed by op kind (`PointwiseNode`, `MatmulNode`, `ReductionNode`), plus `InputNode` for graph inputs and weights. A node is an op and the memory it writes (size, residency, offset), with one `Layout(stick)` for its output. It holds no menu, and there is no `Buffer` class. |
| `DependencyEdge` (with `in_place`) | `DataEdge`: an operand index, `in_place`, and the transform steps on the edge in canonical order (`CopyOut`, `Restickify`, `Relayout`). Each step holds its own memory. Data edges carry no restrictions. |
| `RelayoutEdge`, `CloneRead` on edges (`graph_builder.py`) | `Relayout` is a step on the edge, keyed by the producer's reduction split. Clone reads become constraints between the clone's view and the consumer's. |
| `EncodingStrategy` with per-axis `Fixed` / `Table` / `Lazy` / `Pruned` | The family's encoder. Op-specific rules are `Restriction` flags on node types, and rules every op obeys are built into the encoder. Pins fold into constants, and the enumerated table is pruned by what holds unconditionally. The design has no separate strategy or lookup object. |
| `graph.solve(model, cost_expr, strategy)` | `SpyreOptimizer.optimize(graph)`. The graph has no `solve` and holds no solver types. |
| `Backend` protocol, chosen by `SPYRE_LX_BACKEND` | `ModelAbstractFactory`, chosen from config (so it is in the FX cache key), returns one object per family that encodes and solves. The backend is private to its family. `Annealing` becomes the fallback over the same model, `Mip` packs, and `BruteForce` stays as a test oracle. |
| `pricing.cost_expr` | `CostModel.objective(graph)`: a sympy expression over named symbols. The encoder consumes it and encodes it by symbol name, and an unknown name fails. |
| `_PlacementUnit`, `justify`, `AddNoOverlap2D`, `_C.pack_boxes` | The encoder records `PlacementBox`es. The backend places them, either natively (2-D no-overlap) or by packing: a cumulative constraint under a roof, then a first-fit packer, lowering the roof and re-solving on overflow. |
| `SolveError` | A model proven infeasible is returned with a warning that names the user hints, and nothing is decoded or committed. Out of time with no plan → anneal the same model. A failed check raises `RuleViolation`. |
| Phase 2: lazy encodings | Sticks, restickify, in place and copy-out are variables already. Lazy work division is a later item, worth doing once coarse tiling, work division and relayout combine. |
| Future work: restickification in the solve | Stage 10 here. |

## Relation to the staged conversion (#4)

The staged conversion's Stages 0–1 (harness and gates, then the graph) fit this design.
Stage 1 here reshapes the graph's types. From Stage 2 on, the two plans make different
calls:

| staged conversion (2026-10-23) | this design (2026-10-28) |
|---|---|
| `PlacementStrategy`, `CoreDivisionStrategy` and `CoOptimizingStrategy` over an abstract `EncodingFactory` | No strategies. The family's encoder applies the always-on rules and each node type's flags. Joint division and placement is the only mode, and placement-only is the same model with every division pinned. |
| SA stays separate behind `SaConversionStrategy` until a probe shows it can solve the unified model | SA anneals the same model as the fallback when a solve runs out of time with no plan. That rests on the staged conversion's *Probe: SA on the unified model* holding. If it does not hold, the fallback stays the conversion. |
| Greedy, first-fit, best-fit, placement SA and exhaustive search stay outside, and greedy is the fallback | The placement-only solvers go, and the first-fit packer lives on inside the backends that pack. A model proven infeasible is returned with a warning naming the user hints, and nothing is re-planned. |
| `CpSatLayoutSolver` stays the only CP-SAT entry point | `SpyreOptimizer` over a family chosen from config (also never from an environment variable). |
| No new exception types in the solve path | `RuleViolation` when `check_decisions` fails on a decoded plan. That check runs after the solve and indicates a bug, not a user error. |
| Every PR keeps the CP model byte-identical | The same through Phase 1, and in Phase 2 wherever the rules agree. Each remaining difference is listed by rule. |

If this design is adopted, the staged conversion's Stages 0–1 and 3 stay as written. Its
Stages 2 and 4–7 are replaced by Phases 2–3 here, and its Stage 6 conversion stays only
until the SA probe holds.

The Stage 0 equivalence harness, its corpus, the same-plans check and the verification
suites carry over unchanged (see *Verification*).

## Design

The interfaces, as the sketch defines them. Bodies are in `probes/lx-allocator/lx_design.py`,
and each class has a runnable example in `lx_examples.py`.

```python
# ---- graph layer: no solver types --------------------------------------------
class Restriction(Flag):            # what is special about an op type
    NONE = 0
    WHOLE_STICKS = auto()           # a stuck axis splits in whole sticks (main: factor_domain)
    PADDED_REDUCTION_UNSPLIT = auto()  # a padded reduction stays on one core while stuck
                                    #   (main: coordinate_mask_blocked_vars)
    MATMUL_STICKS = auto()          # x stuck on K, y on N, the output on N (main: _matmul_layouts)
    POINTWISE_STICKS = auto()       # inputs that index the output's stick axis stick on it

class OpNode:                       # name, axes, reads, writes, shape, size; decisions:
    restrictions: ClassVar[Restriction] = Restriction.WHOLE_STICKS   # work_split, layout,
    runs: ClassVar[bool] = True                                      # resident, offset
class PointwiseNode(OpNode): restrictions = OpNode.restrictions | Restriction.POINTWISE_STICKS
class MatmulNode(OpNode):           # x, y, k, n: the attributes MATMUL_STICKS reads
    restrictions = OpNode.restrictions | Restriction.MATMUL_STICKS | Restriction.PADDED_REDUCTION_UNSPLIT
class ReductionNode(OpNode): restrictions = OpNode.restrictions | Restriction.PADDED_REDUCTION_UNSPLIT
class InputNode(OpNode): runs = False; restrictions = Restriction.NONE

class DataEdge: src, dst, operand, transforms, in_place
class ExecutionEdge: before, after, cut          # cut derived where tiles don't match
class Transform: resident, offset, footprint, pinned   # CopyOut (HBM), Restickify, Relayout
class Pin: target, what, value, source           # graph input | pre-pass | hazard | greedy | hint
class SpyreOptimizationGraph: nodes, data_edges, execution_edges, pins, alignment, capacity
class CostModel:
    def objective(self, graph) -> sympy.Expr: ...  # spill_cost, cut traffic, restickify, K-split

def check_decisions(graph, overrides=None) -> list[str]: ...   # every rule, no solver
def coverage(graph, encoder, overrides=None) -> list[str]: ...  # flags a family cannot encode

# ---- family layer -------------------------------------------------------------
class Encoder(ABC):
    def encode(self, graph, restrictions=None) -> Model: ...
    def read_back(self, solution, graph) -> None: ...
class ModelAbstractFactory(ABC):
    def encoder(self, cost_model=None) -> Encoder: ...
    def solve(self, model, settings: SolveSettings) -> Solution: ...
    def fallback(self, model, settings) -> Solution: ...   # anneal the same model
FACTORIES = {"ortools": OrToolsFactory, "cpmpy": CpmpyFactory, "sa": SAFactory}

class SpyreOptimizer:
    def optimize(self, graph):
        model = encoder.encode(graph); solution = family.solve(model, settings)
        # INFEASIBLE -> warnings.warn(user hints), return; no plan -> family.fallback
        encoder.read_back(solution, graph); check_decisions(graph); self.commit(graph)
```

Rules every op obeys are not flags. `Always` names them; the encoder always applies
them and `check_decisions` always checks them:

- the split divides the axis;
- the core budget holds;
- at most one reduction axis is split;
- tiles are on output axes, never the stick dim;
- the cut-tile guard: a copied-out tile that stays resident has no reduction split;
- main's residency rules;
- restickify only where the read stick differs;
- pinned steps stay;
- a copy-out lives in HBM;
- in-place conditions;
- tile alignment inside a nest;
- resident boxes fit LX and never overlap.

A flag only names a rule. `splits_while_stuck(node, flag)` and
`stick_tables(node, flag)` turn it into data from the node's facts, which both the
encoder and the check read. A test can override a type's flags
(`{ReductionNode: ReductionNode.restrictions & ~Restriction.PADDED_REDUCTION_UNSPLIT}`).
Setting `restrictions` on an instance raises.

## Design rules

These were settled in the design threads and in chat (2026-10-28).

| topic | decision |
|---|---|
| Graph inputs | Leaf `InputNode`s with downstream edges only. How they arrive is a pin. |
| Layout | One `Layout` per node (its output), with no ports. What a consumer needs is a restriction, and a mismatch is a step on the edge. |
| Memory | No `Buffer` class. Nodes and steps carry footprint, residency and offset, and placement works over both. |
| Restrictions | A `Flag` class. Node types OR flags together and freeze them, and subtypes add the attributes the flags read. Always-on rules are built into the encoder and the check. Data edges have none. A new flag needs its data, an encoding in each family, and a check. |
| Who applies rules | The family's encoder, at op, edge, op-op and graph level. There is no separate lookup or strategy. |
| Residency rules | Always on, applied while building the table. A division is banned together with residency, not removed. |
| Work division | Enumerated as on `main` ships first. Lazy work division comes later. |
| Cuts | Derived where adjacent tiles don't match. Tile alignment is required on every edge inside a run (as `ilp_solver_ortools.py:985-992`). |
| Copy-out | The buffer lives in HBM, filled tile by tile through its mutation layout, and is never placed in LX. Priced as `_cut_traffic_terms`. |
| Restickify | A transform. It is forced where sticks differ, so it needs no elision. One inserted before the graph is built is a pinned step. |
| Relayout | Keyed by the producer's reduction split, which restores the completed-reduction routes. Only from a resident source (a spilled one reloads in the consumer's configuration). Inside a tile it is priced per tile times the trip count. Never across a cut. |
| In place | A variable on the edge: same-index read, adjacent lifetimes, same layout and footprint, at most one per op. The pair packs as one box. |
| Objective | A sympy expression from the cost model, consumed by the encoder by symbol name. Cores only break ties. |
| Placement | A backend detail. CP-SAT places natively or by packing; annealing and MIP solvers pack. |
| Families | One object encodes and solves, with a private backend. OR-Tools comes first; CPMpy and SA are expansion axes, and CPMpy may later replace OR-Tools. |
| Pins | Constraints, never decisions, never dropped. They come from hints, the pre-pass, hazards, or greedy pinning to cut search. Internal pins must never be infeasible. |
| Failure | Proven infeasible → returned with a warning naming the user hints. Out of time with no plan → anneal the same model. |
| Decisions | They live in the solution. `read_back` decodes them onto the graph only from a plan, and commit lowers from the graph. |
| Determinism | Opt-in: `SolveSettings(deterministic=True)` uses deterministic time, seed 0 and interleaved search. Otherwise speed wins. |
| Loops | Not modelled. User-written loops are opaque hazards, pinned and never re-tiled. Drains stay with whatever inserts them. |
| After commit | `LxContextSwitchingPass` stays a separate pass. |
| Diagnostics | Conflict explanation (MUS/MCS) and `AxisRotation` are out of the first implementation. |

## Phase 1: the design's types, no change to the solve

Each stage is its own PR on `main` and leaves the full suite green. The owner commits
each stage. Stage 0 is the staged conversion's Stage 0 (harness, corpus, same-plans
check), unchanged.

### Stage 1. Node and edge types

- This takes the place of the staged conversion's Stage 1, or reshapes its types if
  that stage lands first.
- `ResourceNode` becomes `OpNode` with `PointwiseNode`, `MatmulNode` (`x`, `y`, `k`,
  `n`), `ReductionNode` and `InputNode`. A node holds its output's size, layout,
  residency and offset.
- `DependencyEdge` becomes `DataEdge` (operand index, `in_place`, step list).
- `RelayoutEdge` and the copy edges become `Relayout`, `CopyOut` and `Restickify`
  steps.
- `Pin` gains `source`.
- The existing strategies keep encoding through these types.
- Verification: the Stage 0 proto diff is empty.

### Stage 2. Builder

The builder is the staged conversion's Stage 1 builder, or #5676's `GraphBuilder`
lifted where `main` has not moved past it.

- The builder builds one node per op, choosing its subtype from the op kind (the
  predicates codegen uses, as `is_restickify_op` does). Each node gets axis-to-dim maps
  from the index expressions (`TensorDep`).
- The builder adds pins for graph-input arrival, the restickifies the pre-pass inserted
  (source `pre-pass`), work-division hints, hazards (`PrescribedRegion`, `DrainPlan`),
  and every node's stick as `propagate_layouts` fixed it.
- Pinning sticks keeps Phases 1–2 at parity, since commit then never changes a layout.
  Stage 10 lifts those pins.
- Verification: the proto diff is empty.

### Stage 3. Restrictions and the check

- Add `Restriction`, `Always`, `RESIDENCY_RULES`, `splits_while_stuck`,
  `stick_tables`, `check_decisions` and `coverage`. Their sources on `main`:
  - `work_division_constraints.py` (`coordinate_mask_blocked_vars`, `:350`);
  - `_matmul_layouts` and `_multi_arg_pointwise_layouts` (`propagate_layouts.py:1298`,
    `:1534`);
  - `_check_stick_all_same` (`op_spec_validation.py:675`);
  - `WorkDivisionContext.is_legal` (`work_division.py:818`);
  - `ResidencyChecks` (`residency_rules.py`);
  - `check_in_place_parent_is_read` (`plan_solver.py:903`).
- Run `check_decisions` on every plan the current solve returns. It reports and never
  fails the compile.
- **Exit for Phase 1:** across the unit and device suites, `check_decisions` reports
  nothing, or each report traces to a real bug that gets an issue.

## Phase 2: the OR-Tools family, behind config

### Stage 4. Cost model

- `CostModel.objective(graph)` ports `main`'s objective:
  - `spill_cost` (`ilp_solver_ortools.py:358`), with inputs and outputs discounted;
  - `_cut_traffic_terms` (`:1267`) for copy-outs;
  - the co-optimizing `cost_expr`, with its per-bundle terms and relayout charges as
    `cost_expr_record` (`plan_solver.py:574`) dumps them;
  - cores, only to break ties.
- #5676's `pricing.py` is the reference for the shape of the port.
- `Sym` names the symbols, and the existing printer (`cpsat_expr_printer.py`) encodes
  the expression by name.
- Verification: on the Stage 0 corpus, the objective equals today's, solve for solve.

### Stage 5. `OrToolsEncoder`, op level

- Builds what `CpSatLayoutSolver` builds today, and #5676's `EnumeratedStrategy` and
  `PairTableStrategy` or #4's `CoreDivisionStrategy` if either lands first:
  - the enumerated table (`enumerate_work_division_candidates`), with coarse tiling in
    the same table and residency rules applied while building it;
  - sticks as variables, or as constants where pinned;
  - each node type's flags (conditional splits under their stick literal, stick tables
    as table constraints).
- Every constraint it posts is tagged with its rule in the model's provenance.
- Verification: the proto diff against today's model is empty where the rules agree. Every remaining difference is listed by rule and either justified in the PR
  or fixed.

### Stage 6. Edge and op-op level

- Restickify is forced where the read stick differs, or pinned.
- **The restickify cross-frame hazard becomes an always-on rule.** A restickify must not
  read a core-sliced resident source unless a relayout comes first (`main`:
  `_restickify_barrier`, `allocator.py:1403`). The sketch does not encode this yet.
- Relayout options are keyed by (source layout, producer reduction split, destination
  layout), with completed-reduction routes (`derive_completed_reduction_routes`,
  `lx_relayout.py:680`). The sketch declares `Relayout` but does not post it yet.
- Copy-outs are derived from cuts.
- Tile alignment is enforced inside a run.
- In place is a variable on the edge.
- Clone reads become constraints.

### Stage 7. Families and placement

- Add `ModelAbstractFactory` and `OrToolsFactory`, registered in `FACTORIES` and
  selected by a new `config.layout_solver` value, so the choice enters the FX cache key.
  This replaces `SPYRE_LX_BACKEND` as the selector.
- The private `CpSatBackend` places natively (2-D no-overlap) or by packing. The packer
  is `_C.pack_boxes`.
- `SolveSettings` makes determinism opt-in.
- `Mip` becomes a family that packs. That is not needed for the switch-over, and it
  stays out of scope until its linearization solves attention graphs (see #4's *Out of
  scope*).
- `BruteForce` stays as the oracle for encoder unit tests.

### Stage 8. `SpyreOptimizer`

- `optimize` runs encode → solve, then `read_back` → `check_decisions` → commit.
- A model proven infeasible returns with a warning naming the user hints.
- A solve out of time with no plan runs `fallback`: the branch's `Annealing` over the
  same recorded model, which packs, once the SA probe holds. Until then the fallback is
  the staged conversion's `SaConversionStrategy`.
- A failed check raises `RuleViolation`.
- `graph.solve` goes.
- **Exit for Phase 2**, with the new family selected:
  - the unit and device suites pass;
  - HBM traffic (the objective) is no worse than `CoOptimizingAllocator`'s on the
    suites and the zoo captures;
  - planning time on the zoo replay stays within run-to-run noise of `main`'s (torch-spyre/torch-spyre#5676
    planned 21–30% slower by default).

## Phase 3: commit, sticks in the solve, switch-over

### Stage 9. Commit

- `SpyreOptimizer.commit` inserts each edge's steps in canonical order (pinned steps
  exist already), then commits divisions and offsets through `GraphEditor`. This
  replaces `_push_allocation` (`allocator.py:1019`).
- `LxContextSwitchingPass` runs after commit, unchanged.

### Stage 10. Sticks in the solve

- Lift the Stage 2 stick pins, except graph inputs and pre-pass restickifies. The
  solver then chooses each node's stick and places the restickifies it needs, as the
  sketch does for `k` in SDPA decode.
- This is the earlier plan's "restickification in the solve", and it needs:
  - `propagate_layouts` to accept a stick decided by the solve;
  - codegen to accept layouts decided after LX planning;
  - the cross-frame hazard from Stage 6 kept;
  - a restickify's compute time accounted for.
- The restickify pre-pass stays as the source of pinned steps.
- `elide_proven_read_copies` (`read_copy_elision.py:607`) goes, because a restickify is
  never present where the sticks agree.

### Stage 11. Shadow comparison and default flip

- Run the old and new paths side by side on CI models, comparing traffic, planning
  time and correctness, then flip the default.

### Stage 12. Deletions

- `GreedyLayoutSolver`, `FirstFitLayoutSolver`, `BestFitLayoutSolver`,
  `ExhaustiveSearchSolver`.
- `MemoryPlanSolver`, `CoreDivisionLayoutSolver` and `CpSatLayoutSolver`.
- The `SolveError` → greedy re-plan in `scratchpad_planning` (`allocator.py:5127`).
- The stick rules now held once in the flags: `_matmul_layouts`,
  `_multi_arg_pointwise_layouts` and `_check_stick_all_same`.
- **Exit for Phase 3:** the optimizer is the default, and the replaced solvers and
  duplicated rules are deleted.

## Later (not planned)

- **Lazy work division.** The branch's `LazyStrategy` becomes a second encoding in the
  OR-Tools family. It is worth doing once coarse tiling, work division and relayout
  combine and the enumerated table grows too large.
- **Conflict explanation (MUS, MCS, step-wise).** This would name which user hints make
  a model infeasible, rather than listing all of them.
- **`AxisRotation`** and the dim-order conversions from the earlier plan's future work.
- **CPMpy and SA families.** They read the same flags (`coverage()` names any flag a
  family cannot encode), and CPMpy may later replace OR-Tools.
- **Multi-level tiles.** `main`'s `TileSpec`s are multi-level, outermost first, while
  the sketch holds one level.
- **Solve budget per use**, for example a quick compile against an AOT build.

## Verification

- **Stages 1–2, and Stage 5 where the rules agree:** the Stage 0 equivalence harness
  from `cpsat-graph-buffer-refactor` must show an empty proto diff. That means 1 worker,
  `random_seed=0`, no wall-clock limit, the same corpus, and the 89 zoo captures replayed
  without a device.
- **From Stage 3:** `check_decisions` runs on every plan in every suite.
- **Unit and device suites:** the same lists as that plan's *Verification*, serial,
  base against change, with baselines re-recorded on each stage's parent commit. Add the
  sketch's examples (`run_examples.py`) as unit tests of the graph layer and the check.
- **After Stages 5, 8 and 10:** replay the zoo captures with `timers.py` on base and
  change, in one process and interleaved, and flag planning-time growth beyond noise.
- `pre-commit run --files <changed>` (ruff, mypy).

## Known gaps in the sketch

The sketch is the design, not the implementation.

- `build_graph`, `commit`, the annealing fallback, and the CPMpy and SA encoders raise
  `NotImplementedError`.
- `Relayout` is declared but not encoded.
- Clone reads are not modelled.
- The restickify cross-frame hazard is not encoded (Stage 6).
- Lifetimes are fixed: each runs from the node's step to its last reader. On `main`,
  torch-spyre/torch-spyre#5246 makes a tiled op's live range start at its nest's start (`_add_nest_starts`,
  `ilp_solver_ortools.py:1078`), which is a solve decision. Stage 6 must carry that over.
- Tiles have one level.
- Placement-only parity (every division pinned) is untested. torch-spyre/torch-spyre#5676 priced the committed
  mode like the open mode and changed plans that way, so the Phase 2 parity gate must
  include the committed mode.
- `to_cpsat` handles the cost terms the sketch uses; `main`'s printer handles more.

## Probes

**`probes/lx-allocator/`** holds the sketch and what it measures. It needs no
torch-spyre checkout and no card, only `pip install ortools sympy`.

| probe | file | answers |
|---|---|---|
| The design, runnable | `lx_design.py` | Plans the three cases (MLP decode, SDPA decode, MLP prefill with coarse tiling and packing) deterministically, and checks each with `check_decisions` |
| Examples per class | `lx_examples.py`, `run_examples.py` | One small runnable example for each class on the design page: flag freezing and overrides, a narrower encoder caught by `coverage`, the infeasible-hint warning, both placement modes |
| Placement under pressure | `stack.py` | A decoder stack (14 ops per layer) under shrinking LX capacity, planned with CP-SAT placing natively and by packing; objective, time, spills, peak and check |

Typical runs:

```bash
cd probes/lx-allocator
python3 lx_design.py                 # the three cases (output: run_output.txt)
python3 run_examples.py Restriction  # one class's examples
python3 stack.py 8 120 8 1,0.1,0.04,0.02   # layers, time limit s, workers, capacity scales
```

`stack.py` results on 8 CP-SAT workers (`stack_output.txt`):

| layers | LX capacity | native 2-D no-overlap | packing | same objective | check |
|---|---|---|---|---|---|
| 8 | ×1 | 12.8 s | 1.2 s | yes | ok |
| 8 | ×0.1 | 5.8 s | 1.0 s | yes | ok |
| 8 | ×0.04 | 7.9 s | 1.3 s | yes | ok |
| 8 | ×0.02 (16–17 ops spill) | 13.7 s | 2.4 s | yes | ok |
| 16 | ×1 | 43.8 s | 3.6 s | yes | ok |
| 16 | ×0.02 (33 ops spill) | 299.3 s | 16.4 s | yes | ok |

Every run ended OPTIMAL. Packing reached the same optimum 5.7 to 18 times faster, and
the gap widens as the graph grows and LX tightens. The times are for the whole
`optimize`: encode, solve, place and check. This
is why placement is a backend detail: a family can pick packing without changing the
model. The graphs are synthetic (32 tokens, hidden 4096, intermediate 12800, KV length
100), not captures.
