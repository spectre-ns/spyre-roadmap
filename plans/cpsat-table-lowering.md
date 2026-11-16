# CP-SAT table lowering, rebuilt bottom-up behind a config flag

Reference result: torch-spyre#6315 (`table-lowering-experimental`, tip `c26f6171`, draft).
Code base: torch-spyre `cab4721d`, the PR's base. Line numbers are at `c26f6171`.

Short paths: `config.py` and `cost_model.py` are under `torch_spyre/_inductor/`.
`cpsat_table_lowering.py`, `cost_expr_numpy.py`, `cpsat_expr_printer.py`,
`ilp_solver_ortools.py`, `plan_solver.py` and `sa_cooptimizer.py` are under
`torch_spyre/_inductor/scratchpad/`.

## Goal

When the last PR of this plan merges, torch-spyre has the table lowering of #6315 behind
`config.cpsat_table_lowering`, off by default. With the flag on, the CP-SAT cost objective
is built from exact element tables, linear sums and `lin_min`/`lin_max`, and the printer
takes an objective the tables cannot lower. With the flag off, every solve goes through
the printer as it does on `main`. Table values are computed with numpy functions, and the
code arrives in layers, each with its own tests.

## Today

#6315 lowers the objective with tables in `cpsat_table_lowering.py` (1148 lines, plus 311
lines of tests). The result is what we want. Three things about how it is built are not:

1. **It evaluates through `sympy.lambdify`.** `_TableLowering._compile` generates and
   execs Python source for every node it tables. Uncommitted edits in `~/torch-spyre` add
   a numpy dispatch table (`_NUMPY_OPS`, `_numpy_value`, `_piecewise`), but the decode
   delivery price, an implemented function, still goes through lambdify.
2. **It is on the default path.** Its one change outside the module swaps the import at
   `ilp_solver_ortools.py:134`, so every CP-SAT solve takes the tables.
3. **It is one module built top-down.** Pure helpers, CP-SAT primitives and the four
   rules arrived together, and tests reach most of it only through
   `lower_cost_with_tables`.

## Decisions (2026-11-16)

- **numpy first, lambdify as the fallback.** A node type in the dispatch table is
  evaluated with its numpy function. Implemented functions are called through `_imp_`
  directly. Only a node type outside the table goes to lambdify.
- **A stack of PRs**, one per stage, off torch-spyre `main`. #6315 stays open as the
  reference and closes when the stack's tip matches it.
- **Off by default.** Stages 1 to 7 add files nothing imports. Stage 8 adds the flag.

## Stages

Every stage is one PR with its tests. Branches: `table-lowering-N-<slug>-experimental`,
each on the one before.

| # | PR adds | Needs ortools |
|---|---|---|
| 1 | numpy values of a cost expression: `cost_expr_numpy.py` | no |
| 2 | pure helpers: ladders, rungs, fold, scaling | no |
| 3 | inputs, indexes, one table per node, linear sums | yes |
| 4 | rule 1 and parts: Min/Max by rung, ranks, pairs, functions of a list | yes |
| 5 | rule 2: fold of Min/Max factors over shared inputs | yes |
| 6 | rule 3 in the objective: linear Min/Max where the table is larger | yes |
| 7 | rule 4: switched products and literal gates | yes |
| 8 | `config.cpsat_table_lowering`, the solver's choice, printer fallback, docs | yes |

The rules are the four in #6315's module docstring.

### Stage 1: numpy values (`cost_expr_numpy.py`)

Start from the uncommitted edits, moved out of `_TableLowering` into a module with no
CP-SAT in it.

- `NUMPY_OPS: dict[type, Callable]`: Add, Mul, Pow, Min, Max, floor, ceiling, log, exp,
  Abs, the six relationals, And, Or, Not, ITE, KroneckerDelta, `RelayoutCharge`.
- `NumpyValues.value(node, env, memo)`: memo, then symbols from `env`, numbers,
  Boolean atoms, Piecewise (`np.select`, every arm computed, NaN when no arm holds),
  `NUMPY_OPS`, an implemented function through a vectorized `_imp_`, and last the
  lambdify fallback, cached per node and counted in `lambdified`. None for an undefined
  function with no `_imp_`.
- `NumpyValues.on_grid(node, columns, shape)`: the node at every joint assignment,
  row-major; None with a reason when a value is not finite.

The table covers everything the cost model builds today: Add, Mul, Pow (exponent -1 and
floats), Min, Max, floor, log, Piecewise ending in `True`, Eq/Le/Ge/Lt/Gt, And, Or,
numbers, `decode_delivery_price_*` (`cost_model.py:1426-1449`) and `RelayoutCharge`
(`plan_solver.py:487`). ITE and `False` are latent: they need auto tiling or cost
parameters other than production's.

Tests, `tests/inductor/test_cost_expr_numpy.py`:

- every `NUMPY_OPS` entry against exact sympy (`float(expr.xreplace(point))`) at every
  grid point, and a check that no entry lacks a case;
- the cost model's own forms: `floor(1024/split)**-0.25`, `Min(c*log(a), Max(0, 3.0 -
  c*log(b)))`, `1/split >= Rational(5, 128)` as a condition, an Or of Ge, nested
  Piecewise, ITE with `False`, `x**0.0`;
- Piecewise: an infinite value in an arm not taken is discarded; no arm true is refused;
- `RelayoutCharge` equals `is_lx * prices[division]`, 0 past the table, and agrees with
  its `_imp_`;
- the real decode delivery price through `_imp_` with `lambdified == 0`; an undefined
  function gives None; `sympy.atan` goes through lambdify once and is cached;
- `on_grid`: row-major order over axes of three sizes, two symbols on one axis, a
  constant broadcast;
- guard: objectives built by the cost model (the builders `test_op_features.py` and
  `test_cost_model_replication.py` already use) evaluate with `lambdified == 0` and agree
  with `sympy.lambdify(modules="math")`, the annealer's evaluator
  (`sa_cooptimizer.py:849`).

### Stage 2: pure helpers (`cpsat_table_lowering.py`, first slice)

`COST_SCALE` and the bounds, `_same_value_key`, `clamped_ladder`, `rungs_on`,
`FoldFactor`, `FoldStep`, `fold_reachable`, as in #6315. `_scaled` and `_switch_scale`
become module functions (static methods in #6315) so they are tested before the class
exists.

Tests, `tests/inductor/test_cpsat_table_lowering.py`: the four ladder and fold tests from
#6315, plus:

- values equal to 12 digits share a rung; a value off the ladder gives None;
- the Min of rungs is the rung of the Min (seeded random operands);
- a constant operand seeds the fold's start state; operands sharing an input join one
  group;
- `_scaled` rounds once;
- `_switch_scale` is 1 for whole values and otherwise keeps `p * w` under 2^60;
- the pair `q*P <= p*w`, `q*P >= p*w - (q-1)` admits only `P = (p*w) // q`, negatives
  included.

### Stage 3: inputs, indexes, tables, linear sums

`_Index`, `_Indexed`, `_Linear` (constant and tables), and `_TableLowering` with
`_map_inputs`, `_map_variable`, `_position_of`, `_keys_of`, `_joint_index`, `_aux`,
`_element`, `_evaluate` (a cache over `NumpyValues.on_grid`), `indexed` for a constant, a
symbol and one table over a node's inputs, `emit` and `_emit_sum`, `_bounds` and `_build`
for tables, `lower`, `_drop_unpriced`, and `lower_cost_with_tables`.

After this stage a sum whose terms each fit one table under the cap is lowered exactly.
Anything else returns None with the model untouched.

Tests: `_buffers`, `_element_sizes`, `_assert_exact` from #6315, and its tests for a sum
of buffers, a Piecewise cost, an unmapped function, a relayout charge (exactness only
here) and the untouched model. New:

- a division's symbols share one index; two menu lengths on one division are refused;
- a non-contiguous domain gets a position; a domain past the cap is no input;
- the joint index agrees with `on_grid`'s order;
- an affine list adds no element; a constant table adds no variable;
- an entry past `_MAX_ENTRY` is refused;
- a relayout of an undivided buffer costs 0.

### Stage 4: rule 1 and parts

`_from_parts`, `_extremum` (without the fold factor), `_rank`, `_pair`, `_product`,
`_function_of`, `_same_index_sum`, and the choice in `_index_node` between one table and
the parts.

Tests: #6315's ladder-rung test. New: equal values share a rank before pairing; a pair
past the cap is refused; a function of a short list reuses its index; the smaller form in
entries wins, one case each way.

### Stage 5: rule 2

`_fold_factor`, `_fold_too_long`, `_fold`, and the fold carried through `_function_of`.

Tests: #6315's four-Max product at cap 200. New: no fold when crossing the lists fits.

### Stage 6: rule 3 in the objective

`_table_entries`, `_linear_entries`, `_prefers_linear`, `_emit_extremum`, `_emit_product`
for numbers times one node, `_operand`, and extrema in `_Linear`, `_bounds`, `_build`.

Tests: #6315's Min of literal sums. New: a near tie keeps the table (`_LINEAR_MARGIN`); a
negative coefficient turns Min into Max; each Min/Max operand is a variable of its own
(the presolve out-of-memory guard).

### Stage 7: rule 4

`_Linear.switched`, `_emit_switched`, `_split_gate`, `_emit_gated`, the wide path of
`_emit_product`, `_switched`, `_value_literals_of`, and switched terms in `_bounds`.

Tests: #6315's two switched-product tests and three gate tests, with the table-size
assertions the gates make true. New: whole factor values take one equality per value,
fractional ones the floor pair; two wide factors are refused with the model untouched; a
product too wide for its factor's precision is refused.

### Stage 8: the flag

- `config.py`, beside `cpsat_time_limit_seconds` (`:407`), in the style of `ktir_emitter`
  (`:137`):
  `cpsat_table_lowering: bool = os.environ.get("CPSAT_TABLE_LOWERING", "0") == "1"` and
  `cpsat_table_cap: int` from `CPSAT_TABLE_CAP`, default 8192. Both names are public, so
  both enter the FX graph cache key: plans can differ between lowerings.
- `cpsat_table_lowering.py`: `cost_expr_to_cpsat`, the tables or else the printer,
  reading `config.cpsat_table_cap` when called. It logs why the tables declined.
- `ilp_solver_ortools.py`: line 134 stays as on `main`. `_minimize_cost_expr` (`:865`)
  reads the flag when called and imports the table module only when it is on.
- Docs: rows in the env var table of `docs/source/api/torch_spyre.rst`; the file table
  and objective paragraph in `docs/source/compiler/scratchpad_planning.md` (`:300`,
  `:636-648`); `docs/source/compiler/co_optimizer.md` (`:296`, `:424-431`).

Tests, with `config.patch` as the suites do:

- #6315's test that an untableable expression goes to the printer;
- flag off: a `CpSatLayoutSolver` solve never calls the table module (patched to raise);
- flag on: the same solve never calls the printer, ends `OPTIMAL`, and its plan's exact
  cost is no higher than the printer's plan, within one `COST_SCALE` unit per table.

Known difference, as in #6315: the tables' objective is `COST_SCALE` (1024) times the
cost, and the printer's is in cost units. Only the uniqueness probe reads it
(`ilp_solver_ortools.py:825-827`, off by default); its "within 1" becomes within 1/1024.

## End state against #6315

`cpsat_table_lowering.py` at the stack's tip is `c26f6171`'s file except for:

- evaluation moved to `cost_expr_numpy.py` (`_compile`, `_NUMPY_FUNCTIONS` and
  `_relayout_charge` leave the module);
- `TABLE_CAP` replaced by `config.cpsat_table_cap`;
- `_scaled` and `_switch_scale` at module level.

`ilp_solver_ortools.py` differs from `main` only inside `_minimize_cost_expr`.

## Verification

Each PR:

- `python3 -m pytest tests/inductor/test_cost_expr_numpy.py
  tests/inductor/test_cpsat_table_lowering.py`
- `make precommit`, `mypy torch_spyre tests` and `make check-all-configs`. Both test files
  are listed in `tests/configs/torch_spyre_tests/inductor/test_cpsat_table_lowering_config.yaml`.
- `git commit -s`, the Apache header on new files, `feat(scratchpad): ...` subjects.
- Stages 1 to 7: `git diff main --stat` shows new files only.

After Stage 8, posted on that PR:

1. **Same file.** `git diff c26f6171 <tip>` on `cpsat_table_lowering.py` shows only the
   three differences above.
2. **Same models.** Replay the 50 captured objectives (`~/pr6301-review/replay.py`,
   results in `~/pr6301-review/tableonly/`) at the tip and at `c26f6171`: every capture
   `OPTIMAL`, with equal `entries`, `objective` and `exact`.
3. **Lowering time.** `lower_s` per capture, numpy against lambdify, from the same replay.
4. **Flag on.** The planner suites with `CPSAT_TABLE_LOWERING=1`:
   `test_scratchpad_solver.py`, `test_scratchpad_use.py`, `test_cost_model_replication.py`
   and `test_solver_relayout_decision.py`. #6315's CI ran every shard with the tables on
   the default path and all passed; its one red job was a card fault
   (`DdrInitRetryLimitExceeded`).
5. **Flag off.** The same suites with no flag, unchanged.

## Next step

Stage 1 starts in a new torch-spyre worktree off `upstream/main`. The uncommitted edits in
`~/torch-spyre` are read, not moved. PRs to torch-spyre open on the owner's go.
