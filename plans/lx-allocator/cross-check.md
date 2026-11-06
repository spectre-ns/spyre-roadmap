# LX allocator stage plans: cross-check record

This records the cross-check of the 14 stage plans against the parent plan
(`plans/lx-allocator.md` at `5aff4a6`) and torch-spyre main at `5268ba95`. The fixes are
already in the plans. What remains here is what the owner must decide, what the parent plan
needs, and what was found on main.

## Re-aligned to parent c8ebd86

On 2026-11-06 the stage plans were re-aligned to the parent plan as merged at `c8ebd86` (merge
`7dbc103`). The counts and findings below come from the check against `5aff4a6` and were not
re-run. "Decisions taken" and the tables of open items now use the `c8ebd86` names. The
re-alignment settled or changed these:

- **Restriction overrides are gone.** Each node's `restrictions` is set by `OpFactory.make` from
  its type's `OpConfig` on the target. `encode`, `check_decisions`, `coverage` and `optimize` take
  no override argument, and a test builds with a derived target instead.
- **`ExecutionEdge` is gone.** `graph.nodes` is in op order, and `graph.cuts` holds the tick of
  the op before each cut. Stage 6's decode helpers and Stage 3's nest helper read `graph.cuts`.
- **Transforms became one `Step` with a `StepKind`.** `DataEdge.transforms` is `DataEdge.steps`
  and `ORDER` is `STEP_KINDS[kind].order`. The pre-pass restickify and step-order decisions stand
  under the new names, and open decisions 1 and 2 stay open, written to their recommendation.
- **`Always` is split.** The arithmetic rules are `ALWAYS` checks. Four card rules
  (`ONE_REDUCTION_SPLIT`, `TILES_OFF_STICK`, `RESIDENT_TILE_UNREDUCED`, `COPY_OUT_IN_HBM`) are
  `Rule` flags read from `graph.target.rules`, so Stage 6's partial-reduction cut guard is posted
  under `Rule.RESIDENT_TILE_UNREDUCED`, a new Stage 6 deviation. The parent-change request for
  `Always` now targets `ALWAYS`, and Stage 3 confirms which rules are `Rule` flags.
- **The card is a `Target`.** Sizes, step kinds, rules, optimizations and the cost model come from
  `graph.target`, chosen by `select_target(config)`. `Rule` and `Optimization` are defined in
  Stage 1, which widens the "Stages 0 to 1" request. `config.target` joins the FX graph cache key
  beside the Stage 9 cache-key fix. Coarse tiling is `Optimization.COARSE_TILING`, and an
  optimization that is off gets no variables.
- **Families replace factories, encoders and backends.** One `Family` object encodes, solves and
  reads back into an `Encoding`, registered in `FAMILIES`. The "Placement" request now reads
  family-owned, and the fallback family comes from `config.fallback_solver`.
- **No `SpyreOptimizer`.** `plan_allocation`, `optimize` and `commit` are functions.

A second pass reconciled names the stages had re-aligned separately. Each provider and every
consumer now agree on these:

1. **Building a family.** `FAMILIES[key].from_config(config)`; every family has a
   `from_config(cls, config)` classmethod, and Stage 7's `Family` protocol lists it.
   `OrToolsFamily`'s field defaults are config's defaults, not config reads.
2. **Fallback setting.** `config.fallback_solver`, env `FALLBACK_SOLVER`. Empty means no
   fallback family, and the greedy re-plan stays until Stage 13.
3. **Families are stateless.** `solve` returns a `Solution` that carries its `encoding`
   (`CpSatSolution.encoding`, `AnnealSolution.encoding`), and `read_back(solution, graph)` reads
   it. `SAFamily.score` takes the encoding.
4. **Always-on rule tags** are strings equal to the check names in Stage 3's `ALWAYS`
   (`"tiles_line_up"`, `"cross_frame"`, `"in_place_conditions"`, `"boxes_fit_lx"`, ...), plus
   `TABLE_COLUMN`. No stage tags with another spelling.
5. **The partial-reduction cut guard** is `Rule.RESIDENT_TILE_UNREDUCED` everywhere.
6. **Cost model across the `graph/` boundary.** `SPYRE.cost_model` stays `None` in
   `graph/target.py`. Stage 4's `cost.py` holds the `CostModel` protocol and
   `COST_MODELS = {"spyre": SpyreCostModel()}`, and `select_target` attaches
   `COST_MODELS.get(target.name)` with `replace`. Committed mode is
   `replace(target, cost_model=SpyreCostModel(traffic_only=True))`. Stage 1's open question on
   this is closed, and Stages 1 and 4 list it as a deviation.
7. **`OpNode.footprint(splits, tiling, alignment)`** keeps `alignment`; callers pass
   `graph.target.alignment`, and the result is in alignment units.
8. **`config.target`** is added and provided by Stage 2, with `select_target`.
9. **Encoding functions.** Stage 5's public names in `family/ortools_encode.py` (`add_unit`,
   `menus`, `cost_domains`, `encode_objective`, `stick_var`, `stick_is`, `post`, `tag`, ...) are
   the only spelling, each taking `enc` first. `joined_literal(enc, before)` takes two arguments.
10. **OR-Tools private state** lives in `enc.ortools: OrToolsState`. Stage 6's `edges`,
    `copies`, `relayout_literals`, `incoming`, `objective_stages` and Stage 7's `horizon`,
    `offsets`, `roof`, `placer` are its fields. `Encoding` keeps only the shared fields and
    `ortools`; Stage 8 keeps `AnnealModel` as `model`.
11. **Coverage.** `coverage(family, target) -> Coverage(gaps, unused)`. A gap is a restriction or
    `Rule` the target uses that the family cannot encode, or a forced step kind it cannot
    insert. A missing chosen step kind (`RELAYOUT`) or `Optimization` is only unused. So
    `coverage(SAFamily, SPYRE)` has no gaps, which closes Stage 8's open question.
12. **Old-path routing.** Stage 9's `family_key(config) -> str | None` and
    `select_families(config)` are the providers. Stage 11's old-path test is
    `family_key(config) is None`, and Stage 12 names `family_key`, `select_families`,
    `PlanOutcome` and `last_outcome()`.
13. **Port facts.** `derive_port_facts(port, layout)` (Stage 2, unmasked) and
    `facts_for(graph, port, layout)` (Stage 11, masked by the node type's `OpConfig.facts`).
14. **`Family.dump_tables(encoding, solution)`** is Stage 10's extension of Stage 7's protocol.

Unchanged: open decisions 1 to 8, the findings on main, the goals that need rewording and the
probes. No other request in "Changes the parent plan needs" is in `c8ebd86`.

## Result

The sequence is buildable on paper, with no dependency cycle between stages. It reaches the
parent's Phase 3 exit only if that exit is reworded (see "Goals that need rewording").
Nothing has been built or run on real models, except the small probes each plan names.

Four reviewers read the plans independently. Each had one job.

| review | scope | result |
|---|---|---|
| Interfaces | Every Consumes row against its provider | 61 findings: 14 blockers, 27 high, 15 medium, 5 low |
| Gates | Whether each stage can land and each gate can be evaluated | 41 findings: 8 critical, 13 high, 16 medium, 4 low |
| Coverage | Every parent class, bullet, decision and verification line | 52 findings: 4 critical, 12 high, 25 medium, 11 low |
| Code claims | Whether the plans' statements about today's code are true | 40 claims checked: 30 confirmed, 6 partly, 4 refuted. 2 mis-citations in about 900 |

The plans were written in parallel, so most findings had one cause: later stages extended
Stage 1's types on their own, under different names. After the fixes, one pass reconciled
all 14 plans against each other, checking 295 Consumes rows and correcting 117 Provides rows.

A final independent read then rechecked the 26 blockers and criticals and followed a pinned
restickify, a tiled graph with a copy-out and a prefetched input through the stages. It
closed 25 of the 26 and found 21 new issues (5 high, 6 medium, 10 low), including a false
constraint on pinned restickifies and no harness recording of family solves. All are fixed
in the plans, but those last fixes have had no further independent review.

## Decisions taken

These settle conflicts between plans. "Owner" marks the ones the owner may want to reverse.

| topic | decision | owner? |
|---|---|---|
| Pre-pass restickify | `RestickifyNode` in the graph, plus `Step(kind=StepKind.RESTICKIFY, pinned=True, realized_by=node)` on the edge. The node holds the division, residency, box and tick. | yes |
| Step order | List order of `DataEdge.steps` is physical and authoritative. `STEP_KINDS[kind].order` only places a step a decoder appends. | yes |
| Objective | Stage 4 owns the terms, Stage 6 lowers them, Stage 7 runs them. Today's priced expression and lexicographic ladder both stay. | |
| Sticks and the flip | Stage 12 flips with sticks pinned and does not depend on Stage 11. Step 12c turns sticks on later. | yes |
| Rules SA breaks under tiling | Stage 3 states `in_place_conditions` and `tiles_line_up` in physical form. OR-Tools stays stricter by encoding. SA filters tilings of `for_each_tile` regions and restickifies. | yes |
| Same-nest relayouts on menu tilings | Not built in this plan. Moves to Future work. | yes |
| Committed mode | Traffic objective, relayouts out, `division_is_fixed` branches stay. Stage 9's bands replace zero tolerance, and SA is judged per corpus on a five-seed median. | yes |
| SA replacing the greedy re-plan | `config.fallback_solver` starts empty. Stage 12 flips it after Stage 9's Gate 4 report exists. | yes |
| Device-free tier | Stage 0 adds a `fake` compile tier, and gates that only need `GraphLowering` use it. Card calibration runs at Stages 0, 9 and 12. | |
| Cache key | The fix moves to Stage 9, because Stages 9 to 11 rely on it. | |
| Final `layout_solver` values | `"ortools"` and `"sa"`, the `FAMILIES` keys. `cpsat` and `simulated_annealing` alias for one tagged release, then raise. | |
| `elide_proven_read_copies` | Stays. No stage deletes it. | |
| Stick rule functions | `_matmul_layouts`, `_multi_arg_pointwise_layouts` and `_check_stick_all_same` stay. | |
| `is_output_stick_aligned_for_entry` | Does not exist. No padding exception and no builder fact. | |
| LX prefetch | Input clones only. Output and collective clones keep their name. Code identifiers and `merge_*` proto names stay. | |

## Owner decisions still open

The plans are written against the recommendation in each case.

**Decide first, because they change what lands early:**

1. **Restickify representation** (Stage 1). Must be settled before Stage 1 merges.
   Recommendation: `RestickifyNode` plus a pinned step.
2. **Relayout before restickify as a legal order** (Stage 1). Must be settled before Stage 1
   merges. Recommendation: yes, with list order physical. The code allows it, but its only
   test is mocked.
3. **Ordering with #5682** (Stage 11). Settle with mtsokol before Stage 5 starts: does Step 2
   stop at Step 1 plus the guards, or land in full before Stage 5? A half-landed Step 2 during
   Stages 5 to 10 is the worst case, because each later item then needs its own encoder port.
4. **The stage order 0-10, 12, 13, 11** (Stage 11). Recommendation: yes. Stage 11 needs
   external work and has the hardest unknowns, and nothing in its gates needs the old path.

**Decide before the stage they affect:**

5. **Fix SA's missing `LxContextSwitchingPass` on main now?** See the next section.
   Recommendation: a small separate PR before Stage 0 records its baseline. It removes the
   baseline patch in Stages 9 and 12 and the exception in Stage 10.
6. **Is Stage 9's Gate 4 a report or a gate?** Recommendation: a report. SA has no guaranteed
   floor against greedy.
7. **Without ortools, degrade to SA or keep raising?** (Stage 12) Recommendation: degrade.
8. **Stage 12's numbers:** the planning-time budget for 12c, the soak length, and whether the
   `.github` owners accept 36 nightly card jobs.

## Found on main

These exist today, independently of this plan.

- **The joint SA path runs no `LxContextSwitchingPass`.** `select_allocator` gives `[]` for SA
  joint and `['LxContextSwitchingPass']` for the other paths (`allocator.py:5432-5435`), yet
  `enable_lx_context_switching` turns the extern guard off. A buffer live across a
  single-output risky `FallbackKernel` is exposed. A default compile cannot reach this; it
  needs `LAYOUT_SOLVER=simulated_annealing`. Whether an LX buffer is actually clobbered needs
  a card to confirm.
- **A default compile without ortools raises `ValueError`.** This was run with ortools blocked
  at import. `pyproject.toml:35-39` promises a fallback, while
  `docs/source/api/torch_spyre.rst:755-760` documents the error, so the two disagree.
- **Planner source is not in the FX cache key.** `CustomPreSchedulingPasses.uuid()` is never
  called. `allocator.py`, `lx_relayout.py`, `ilp_solver_ortools.py` and `sa_cooptimizer.py`
  are in no hashed pipeline, so persistent caches can keep stale plans. The fix costs every
  user one recompile.
- **Stale comments.** The padding comments at `work_division_constraints.py:1038-1039`,
  `pass_utils.py:1673` and `enforce_indirect_access_layout.py:75-79` are the likely source of
  the false `is_output_stick_aligned_for_entry` claim. `sa_cooptimizer.py:275`, `:798` and
  `:1129` wrongly call SA's seed the committed division.

## Goals that need rewording

The parent's Goal and Phase 3 exit cannot be met as worded. The work still delivers one entry
point, a check on every plan, a family seam and a place for sticks.

| parent goal | where it lands | suggested wording |
|---|---|---|
| Each rule exists once | True for division and residency rules. For sticks, `propagate_layouts`, `op_spec_validation` and the beam's cost function stay beside the new `stick_tables`. | One authority for each rule, with generators and validators tested against it |
| One solver family | Two by design, one per solve. A first-fit seed, collective staging and the beam remain. | One family per solve, chosen by config |
| A single check judges every plan | Yes for the solve's output, and `_stage_collective_outputs` is now covered too. Loop-carry views are judged at commit. | Unchanged |
| One solve decides sticks and coarse tiling | Never both. All sticks stay pinned under `AUTO_COARSE_TILING`, and inputs, fixups and loops stay pinned everywhere. | One solve decides sticks at default config |
| Placement-only is the same model with every division pinned | It also needs the traffic objective, relayouts out and the fixed-division branches. | One solve path, with committed mode as a mode of it |
| No `Buffer` class | A small `LxRecordBuilder` remains, and `CoreDivisionBuffer` likely survives. | Name a follow-up |

## Changes the parent plan needs

The parent has only taken the review answers in `5aff4a6`. These come from the plans'
Deviations. Each plan's Deviations section has the evidence.

| area | change |
|---|---|
| Stages 0 to 1 | Stage 0 has a card tier and a `fake` tier, and is not fully device-free. Stage 1 owns `Restriction`, `PortFact`, `Rule` and `Optimization`. |
| `Restriction` | Add `WINDOW_UNSPLIT`, `CONV_SPATIAL_UNSPLIT`, `TOPK_SPLITS`, `KEEP_BY_INDEX_SPLITS`, `PLAIN_REDUCTION_UNSPLIT`. `POINTWISE_STICKS` also covers reductions and indirect ops. |
| `INDIRECT_ACCESS`, Stage 3, Padding | Remove the padded-output exception. Padding enlarges a buffer and enables no split. |
| `ALWAYS` | Add the rules Stage 3 and Stage 6 found, including the residency slicing gate, which the parent's list omits. "The split divides the axis" holds for the stick basis on a stuck axis. |
| Port facts | QFP8WT reaches only the out port and operand 1, and table reads are outside these rules. `OVERLAPPING_READ` depends on the layout. |
| Node types and ports | Add `RestickifyNode` and `OpaqueNode`. `PORTS` is variadic for pointwise and reduction, with a synthetic `table` port, overflow ports and a matmul `target`. `MatmulNode.k` and `.n` may be `None`. |
| Layout | `OutPort.layout` is optional for extern, CPU and mutation-alias ops. `InPort.read_layout` exists for `_input_layout_overrides`. |
| Graph | Build from `GraphLowering`, with the allocator's records as the test oracle. `OpNode.op` and `graph.ir` are live references, and the list of node facts grows. |
| Pins | Add the `layout` and `committed` sources, wider hazards, and loop facts on the graph. |
| Objective | A priced expression plus a ladder, with `domains` passed in. Each family's `encode` is strict on unknown names, and production SA warns as today. |
| Stage 5 gate | A section diff that reaches the whole proto at Stage 7. Residency verdicts are computed with the table and posted at today's position. |
| Placement | Family-owned (`OrToolsFamily.solve`, by its `placement`), but called during `encode` while byte parity holds. |
| Relayout, Stage 6 | The joint path needs no new hazard constraint. Completed-reduction routes go behind a flag that Stage 12 turns on. A tiled op's lifetime starts at its write slot. |
| Copy-out | A cut with no data edge has no step, and traffic is priced by node. |
| Failure | Add no-plan outcomes and `MODEL_INVALID`. An infeasible model with no user hint raises `Unsupported`. |
| Determinism | Interleave only at two or more workers. |
| Stage 8 | Every enumerated op generates at any config. The packer stacks by permutation, and FirstFit seeds it. An edge with no pair spills rather than cuts. |
| After commit | The context pass runs for every family, which changes SA. |
| Stage 10 | The commit order, post-solve work and collective staging stay as an exception. Old wrappers stay until Stage 13. |
| Stage 11 | Five requirements, not four. #5682 prerequisites, 11a and 11b, pre-pass means ops already in the IR, `elide_proven_read_copies` stays, more pins stay. |
| Stage 12 | PRs 12a, 12b and 12c, what "CI models" means, an aggregate time gate, and sticks flipped separately. |
| Stage 13 | The stick functions stay, first-fit moves, dead functions and adapters are listed, and collective staging stays. |
| Verification | Planning time is measured on live compiles, because captures do not reach the new encoder. Device runs are added at Stages 3 and 13. |
| Silent today | The debug greedy family from the PR #5 discussion, and SA fallback through seeding. Neither is planned or ruled out. |

## Probes to run first

None of these has been run. Each answers a question cheaply, before the stage that depends on it.

| probe | when | question |
|---|---|---|
| P1 | First commit of Stage 0 | Can every zoo model compile device-free through planning, and do final snapshots match a card? The device-free compile is proven for SwiGLU only (`~/cpsat_bench/swiglu_full`). The gate schedule depends on this. |
| P2 | Before Stage 5 | Do sticks pay at all? The beam placed zero restickifies on 29 zoo models. Forcing the non-beam candidates on main answers this with no new code. |
| P3 | Before Stage 7 | Do committed mode's bands hold on main, with every op pinned at its committed division? |
| P4 | Stage 5, commit 7 | Does a graph built without the allocator line up with today's model on a real compile? |

## Limits of this check

- No plan was compiled or executed. Interface matching is a reading of 14 documents.
- Planning-time figures are projections, and Stage 11's solve growth has no direct measurement.
- The restickify shapes are close to unexercised on the corpus, so Stage 0 adds two graphs for
  them. Relayout before restickify has been seen only in a mocked test.
- The plans run from about 520 to 1,100 lines each, above the 150 to 400 lines first intended.
