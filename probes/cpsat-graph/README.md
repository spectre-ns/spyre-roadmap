# Probes for the CP-SAT graph refactor

Measurement code behind `plans/cpsat-graph-buffer-refactor.md` (Phase 2.0 and
Phase 3a). Nothing here changes torch-spyre: every hook is a process-local
monkeypatch. Outputs land in `deenum/runs/` and are not committed.

Run everything with the torch-spyre checkout first on `PYTHONPATH`. As a
script, Python otherwise resolves `torch_spyre` from site-packages, which may be
a different checkout. The shell drivers do this through `TORCH_SPYRE` (default
`~/torch-spyre`). Import `torch` before unpickling a capture.

## `deenum/`: lazy vs eager enumeration (Phase 2.0)

| file | what it does |
|---|---|
| `timers.py` | Phase timers over the co-optimizing allocator and CP-SAT; per-edge view-key census; optional solver-input capture (`*.pkl`, live compiler objects blanked) with per-candidate view and tile keys (`edge_raw`). `DEENUM_HASHJOIN=1` swaps `match_pairs` for a hash join on the exact view key and checks it against the original. |
| `zoo_t.py`, `granite_t.py` | Compile one zoo model / Granite under the timers (needs the Spyre card). |
| `run_profile.sh`, `run_profile_hj.sh` | All 32 zoo models in default and `AUTO_COARSE_TILING=1`, then Granite; eager, or eager + hash join. Waits for the card between units. |
| `lazy_probe.py` | Device-free. Rebuilds one reduced layout model per capture three ways: `E` (menu index + pair tables, today), `L1` (per-axis split / tile vars, lazy view and tile gates, per-node menu legality), `L2` (no menu). Checks each edge's inferred per-axis relation is exact against the allocator's final table. Arms via `ARMS=`; `INTERLEAVE=1` for deterministic multi-worker search. |
| `report_profile.py`, `report_probe.py` | Summaries. |
| `zoo_cprof.py` | cProfile one allocator method during a compile. |
| `zoo_ms.py` | Record every `movement_supported` decision (both views' digit keys, core counts, result), to test a closed form for relayout legality (Phase 2d). |
| `ms_check.py` | The closed form for relayout legality, checked against recorded decisions (`ms_check.py <ms.jsonl> ...`). |
| `ms_synth.py` | The same check on synthetic digit-form views, invalid geometries included (`ms_synth.py <out.json> [random_pairs] [per_reason] [seed]`; the plan's 306K pairs are seeds 0 and 1). |

`../zoo.py` defines the 32 zoo graphs (`zoo.py <model|list> <out_dir>`). Granite runs
`ibm-granite/granite-4.0-micro` generation offline from `HF_HOME` (default `~`), with
the adapters from `HF_ADAPTERS` (default `~/hf-adapters`); `run_profile.sh` runs it
after the zoo.

`timers.py` wraps hot helpers (`same_partition`, `_per_core_view_from_prep`, …) in a
counting wrapper that costs ~0.95 µs per call: under tiling that is ~59 s of the zoo's
match-table time. The hooks target torch-spyre `main` 82edd8f6; record the commit with
each run and re-point them when the solver's private methods change.

```bash
TS=${TORCH_SPYRE:-$HOME/torch-spyre}
bash deenum/run_profile.sh                      # eager profile + captures
PYTHONPATH=$TS python3 deenum/lazy_probe.py deenum/runs/caps_default out.jsonl 1,16 30
ARMS=E,L1,L2 PYTHONPATH=$TS python3 deenum/lazy_probe.py deenum/runs/caps_tiling out.jsonl 16,1 30
python3 deenum/report_probe.py out.jsonl
python3 deenum/ms_check.py ms.jsonl
PYTHONPATH=$TS python3 deenum/ms_synth.py synth_s0.json 150000 3000 0
```

## `sa_ilp/`: SA as a generic backend (Phase 3a)

| file | what it does |
|---|---|
| `sa_solver.py` | `SaSolver`: the `pywraplp.Solver` modeling API (`IntVar`/`BoolVar`/`Add`/`Minimize`/`Solve`/…) plus `Add(...).OnlyEnforceIf`, `NewOptionalIntervalVar` and `AddNoOverlap2D`, backed by simulated annealing. Linear rows and global constraints (2-D no-overlap: overlap area, incremental, with a repair move) are scored by violation with an adaptive penalty. |
| `cpsat_backend.py` | The same interface on `cp_model`; the exact reference for the layout model (`pywraplp` has no intervals). |
| `problems.py`, `run_probe.py` | Generic ILPs (knapsack, set cover, generalized assignment, bin packing, random ILP) plus LX residency from captures, written once against the shared API; solved by `pywraplp` SCIP / CP-SAT and by `SaSolver`. |
| `lx_layout.py` | LX layout (residency + offsets + 2-D no-overlap) from captures on CP-SAT and SA through the same builder, with an independent feasibility check of SA's answer. |

```bash
PYTHONPATH=$TS python3 sa_ilp/run_probe.py ilp.jsonl 10 60
PYTHONPATH=$TS python3 sa_ilp/lx_layout.py lx.jsonl "deenum/runs/caps_default/*.pkl" 10 30
PYTHONPATH=$TS python3 sa_ilp/lx_layout.py lx.jsonl "deenum/runs/caps_granite_default/*.pkl" 10 30
```

The report scripts summarize; the plan's 2.0 ratio table is the median L/E time to
OPTIMAL over pairs where both arms proved optimality and E took ≥ 0.2 s.

`PROBE_CAPS` overrides where `problems.py` looks for captures.
