# Probes for the LX allocator

This folder holds the runnable design behind `plans/lx-allocator.md` and what it
measures. Nothing here touches torch-spyre or needs the Spyre card. Install
`pip install ortools sympy` (written against OR-Tools 9.15 and sympy 1.14, on
Python 3.13).

| file | what it does |
|---|---|
| `lx_design.py` | The design as one module: the graph layer, `Restriction` flags and the always-on rules, `check_decisions`, `CostModel`, `OrToolsEncoder`, `CpSatBackend` (native 2-D no-overlap or packing), the families and `SpyreOptimizer`. Run it to plan three cases (SwiGLU MLP decode with pinned weight restickifies, SDPA decode with a padded KV stick, MLP prefill with `AUTO_COARSE_TILING=1` placed by packing) deterministically and check each one. Bodies that need the real IR raise `NotImplementedError` and say where that code lives on `main`. |
| `run_output.txt` | What `python3 lx_design.py` prints. A change that alters a plan shows up as a diff. |
| `lx_examples.py` | One small example for each class on the design page, as `# example: <Class>` blocks. |
| `run_examples.py` | Runs each block on its own after `from lx_design import *`. Pass words to run only the blocks whose title contains them. |
| `stack.py` | Placement under LX pressure. It builds a decoder stack (14 ops per layer, with k/v cache and weights as `InputNode`s), shrinks LX capacity, and plans the stack twice: CP-SAT placing natively, then packing. For each run it prints the objective, wall time, spills, peak LX units against capacity, and the result of `check_decisions`. |
| `stack_output.txt` | `stack.py` at 8 and 16 layers. |

```bash
python3 lx_design.py | diff - run_output.txt
python3 run_examples.py                       # every class
python3 stack.py 8 120 8 1,0.1,0.04,0.02      # layers, time limit s, workers, capacity scales [tokens]
```
