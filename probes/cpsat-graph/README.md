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

```bash
bash deenum/run_profile.sh                      # eager profile + captures
PYTHONPATH=~/torch-spyre python3 deenum/lazy_probe.py deenum/runs/caps_default out.jsonl 1,16 30
python3 deenum/report_probe.py out.jsonl
```

## `sa_ilp/`: SA as a generic backend (Phase 3a)

| file | what it does |
|---|---|
| `sa_solver.py` | `SaSolver`: the `pywraplp.Solver` modeling API (`IntVar`/`BoolVar`/`Add`/`Minimize`/`Solve`/…) plus `Add(...).OnlyEnforceIf`, `NewOptionalIntervalVar` and `AddNoOverlap2D`, backed by simulated annealing. Linear rows and global constraints (2-D no-overlap: overlap area, incremental, with a repair move) are scored by violation with an adaptive penalty. |
| `cpsat_backend.py` | The same interface on `cp_model`; the exact reference for the layout model (`pywraplp` has no intervals). |
| `problems.py`, `run_probe.py` | Generic ILPs (knapsack, set cover, generalized assignment, bin packing, random ILP) plus LX residency from captures, written once against the shared API; solved by `pywraplp` SCIP / CP-SAT and by `SaSolver`. |
| `lx_layout.py` | LX layout (residency + offsets + 2-D no-overlap) from captures on CP-SAT and SA through the same builder, with an independent feasibility check of SA's answer. |

```bash
python3 sa_ilp/run_probe.py ilp.jsonl 10 60
PYTHONPATH=~/torch-spyre python3 sa_ilp/lx_layout.py lx.jsonl "deenum/runs/caps_default/*.pkl" 10 60
```

`PROBE_CAPS` overrides where `problems.py` looks for captures.
