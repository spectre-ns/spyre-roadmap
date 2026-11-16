# CP-SAT table lowering, rebuilt bottom-up behind a config flag

Reference result: torch-spyre#6315 (`table-lowering-experimental`, tip `c26f6171`, draft).
Code base: torch-spyre `cab4721d`, the PR's base. Line numbers are at that commit.

Short paths: `config.py` and `cost_model.py` are under `torch_spyre/_inductor/`.
`cpsat_table_lowering.py`, `cost_expr_numpy.py`, `cpsat_expr_printer.py`,
`ilp_solver_ortools.py`, `plan_solver.py` and `sa_cooptimizer.py` are under
`torch_spyre/_inductor/scratchpad/`.

## Goal

When the last PR of this plan merges, torch-spyre has a table lowering of the CP-SAT cost
objective behind `config.cpsat_table_lowering`, off by default, that reproduces #6315's
results. With the flag on, the objective is built from exact element tables, linear sums
and `lin_min`/`lin_max`, and the printer takes an objective the tables cannot lower. With
the flag off, every solve goes through the printer as it does on `main`. Table values are
computed with numpy functions, and the code arrives in layers, each with its own tests.

## Today

#6315 lowers the objective with tables, and its results are what we want: on 50 captured
objectives every solve is `OPTIMAL` and the objective is exact to 1.2e-7. How it is built
is not what we want to land:

1. **It evaluates through `sympy.lambdify`.** Python source is generated and exec'd for
   each node it tables.
2. **It is on the default path.** It replaces the printer's import at
   `ilp_solver_ortools.py:134`, so every CP-SAT solve takes the tables.
3. **It is one module built top-down.** Helpers, CP-SAT primitives and rules arrived
   together, and tests reach most of it only through the entry point.

This plan does not port that code. It states what each layer does, and the implementation
is free to differ.

## Decisions (2026-11-16)

- **numpy first, lambdify as the backup.** A generic evaluator finds each node's numpy
  function by name. A node with no numpy function is evaluated through lambdify.
- **A stack of PRs**, one per stage, off torch-spyre `main`. #6315 stays open as the
  reference result and closes when the flag reproduces it.
- **Off by default.** Stages 1 to 7 add files nothing imports. Stage 8 adds the flag.
- **Behaviour, not code.** The stages below say what each layer does. #6315 is cited for
  its results only.

## Estimates

Measured on the 50 replayed objectives: 45 captured compiles (model blocks at the default
config and with auto tiling, plus tests) and 5 synthetic products. Two sources: a count of
what each top-level term of the objective reads (2026-11-16), and #6315's lowering at a
cap of 8192 (replay of 2026-11-15).

**Cost between buffers is rare.** Of 6663 terms:

| A term reads | Terms | Share |
|---|---|---|
| no division (residency literals only) | 1670 | 25.1% |
| one buffer's division | 4851 | 72.8% |
| two | 38 | 0.6% |
| three or more | 104 | 1.6% |

Each of the 45 captured compiles has 1 to 4 terms that read two or more divisions, the
*wide* terms.
The widest reads 4 to 8 divisions on a model block, and 35 and 49 on the two large tests.
One table over its inputs would hold from 3.6e4 to 7.5e18 entries on a model block, so it
is never one table.

**How that simplifies.**

- *Narrow terms, 98%.* Each is a list over one buffer's menu, or a coefficient on
  literals. 81% read exactly one literal, 18% none, and 57 terms read four or more. Terms
  on one division under one gate add into one list in numpy, before anything reaches the
  model. Their entries are about the sum over divisions of menu size times gated groups,
  which is linear in the number of buffers. No table spans two buffers, and on every
  model block the largest table is the largest menu.
- *Wide terms, 2%.* Rules 1, 2 and 4 exist for these 1 to 4 terms. By the estimate below
  they take a fifth to two fifths of the entries at the default config and half to seven
  tenths with auto tiling.

| Objective | Divisions | Largest menu | Terms | Wide | Narrow entries (est.) | Entries, #6315 | Lower (s) | Solve (s) |
|---|---|---|---|---|---|---|---|---|
| default, decode block | 12 | 20-26 | 165-169 | 2 | 249-318 | 396-525 | 0.61-0.68 | 0.08-0.10 |
| default, prefill block | 14 | 55-65 | 187-189 | 1 | 1056-1217 | 1304-1535 | 0.72-0.74 | 0.40-0.50 |
| auto tiling, decode block | 12 | 30-41 | 101-105 | 3 | 332-429 | 622-870 | 0.63-0.92 | 0.14-0.17 |
| auto tiling, prefill block | 14 | 380-450 | 165-167 | 3 | 5849-6689 | 16908-21998 | 0.89-1.27 | 1.74-2.38 |
| flash attention test | 41 | 338 | 232 | 4 | 28986 | 49227 | 2.01 | 8.89 |
| prefill batch GQA test | 91 | 662 | 401 | 3 | 58399 | 115667 | 5.94 | 13.49 |

The narrow estimate is one list per division and gate, sized by the menu. It ignores lists
that turn out affine and cost nothing, so it runs high.

**What it says about the stages.**

- Stage 3 lowers 6465 of the 6521 narrow terms as one table each; the other 56 read
  four or more literals. It lowers none of the 50 objectives end to end: each has at
  least one term whose inputs pass the cap, and the lowering is all or nothing.
- Entry count is decided in Stages 4 to 7, on 1 to 4 terms per objective.
- Lowering time matters most where the model is smallest. At the default config a decode
  block takes 0.61 to 0.68 s to lower and 0.08 to 0.10 s to solve. How much of that is
  lambdify is not measured. Stage 1 is where it would be saved, and the last PR reports it.

## How a table is read

- An *input* is one CP-SAT variable the cost depends on: a buffer's division, a residency
  literal, or another integer variable in the symbol map. A buffer's split and tile
  symbols are columns over its division.
- A node is *indexed* when its value is `values[index]`: `values` is a float list
  computed in numpy before the solve, and `index` is a CP-SAT expression that the inputs
  determine. The solver never chooses an index.
- *Entries* are the table entries a form puts in the model. The cap bounds one table.
- Lists are float from the leaves up. CP-SAT holds positions only, until a list is
  collapsed into a linear term (next section).

The four rules that keep lists short:

1. A Min/Max of indexed operands is indexed by rung. Its ladder holds only the values it
   can return, each operand maps to a rung, and the Min/Max of the rungs is the rung of
   the Min/Max.
2. A product of Min/Max factors over shared inputs is indexed by the tuples of rungs that
   can occur together, found by a chain of small tables, one per group of shared inputs.
3. A node takes the cheaper, in entries, of one table over its inputs and the form built
   from its parts. In the objective a Min/Max stays linear when that needs under half the
   entries of its table.
4. A product whose factor cannot share a table with the rest is switched on the factor's
   value: one literal per value holds the product at that value times the rest. A 0/1
   literal times a term is the two-value case.

## How a table maps onto CP-SAT

A list is computed in numpy before the solve. CP-SAT returns the right entry because a
list and its index are made together, by one rule in one order. Every operation that makes
a new list also makes the constraint that computes the new index.

The only lookup is `add_element(index, table, target)`, which enforces
`target == table[index]`. The rest is making `index` the position numpy used.

**Inputs.**

- A buffer's division is a variable in `0..n-1`, and each of the buffer's symbols has `n`
  values: position `d` holds the symbol's value at division `d`. The index is the
  division itself.
- A residency literal has the list `[0, 1]` and is its own index.
- Any other integer variable has its domain's values, ascending. Its index is `var - lo`
  when the domain has no gaps. With gaps it is a position `pos` under
  `add_element(pos, domain_values, var)`; the values are distinct, so one position fits.

**Each way a list is made.** Lists are built from the leaves of the expression upward: a
node's list is made from its arguments' lists.

| Form | The list, in numpy | Its index, in CP-SAT |
|---|---|---|
| a function of one indexed node | `f(values)`, elementwise | the same index |
| terms on one index | the lists added elementwise | the same index |
| one table over inputs of sizes `n1..nm` | the node on the row-major grid: assignment `(i1..im)` at position `i1*s1 + ... + im*sm`, `s_k = n_(k+1) * ... * n_m`, so the last input varies fastest | `joint == index_1*s1 + ... + index_m*sm` |
| rank: equal values share a position | `distinct`, the values ascending without repeats | `add_element(index, rank_of, rank)`, where `rank_of[i]` is the position of `values[i]` in `distinct` |
| product of lists `a` and `b` of sizes `na`, `nb` | the outer product: `a[i] * b[j]` at position `i*nb + j` | `pair == rank_a*nb + rank_b` |
| Min/Max by rung | the ladder, ascending | per operand `add_element(index_o, rungs_o, rung_o)`, then `rank == min(rung_o)` or `max(rung_o)` |
| fold | one value per reachable state | per step `position == state*size + group`, then `add_element(position, next_state, state')` |

**Why the right entry comes back.**

- Every index is fixed by an equality, an element or a min/max over indexes that are
  fixed themselves. Once the inputs have values, each index has one feasible value, and
  it is the position numpy used.
- The strides of a joint table, and `nb` of a pair, are computed once and used for both
  the numpy list and the linear equality. The inputs of a table are taken in one fixed
  order for both.
- A ladder is ascending, so position keeps order: the Min of the rungs is the rung of the
  Min. A value clamped to the bound can never be the result, so sharing the bound's rung
  changes nothing.

**Collapsing float lists into linear terms.** Every list is float: the leaf lists over
the menus, and each list made from them. CP-SAT cannot hold a float, so below a linear
term it holds positions only (divisions, ranks, rungs, pair and fold positions), and no
cost value. A *linear term* is where a list must become a CP-SAT integer: a term of the
objective's sum, a term of an operand of a Min/Max kept linear, or a term of the sum
inside a switched product. There the list is collapsed, in this order:

1. Every float multiplier above the list is applied to it in numpy: the term's
   coefficient, the constant factors of a product, and a coefficient pushed through a
   linear Min/Max, where a negative one turns Min into Max.
2. Lists on one index in the same sum are added.
3. The list is rounded once, `table[i] = rint(COST_SCALE * values[i])`. An entry past
   2^50 is refused, so a sum of thousands of entries stays inside int64.
4. It is read by `add_element(index, table, target)`, and the linear term is `target`.
   A list is affine when its rounded entries are, so `table[0] + step * index` equals
   the table entry for entry.

No integer made by a collapse is multiplied by a float afterwards, with the one exception
below. A unit is 1/`COST_SCALE` of the cost. Each collapsed list is within half a unit of
its float value, a sum of T lists and a constant is within (T + 1) / 2 units, and a
linear Min/Max is within the largest bound among its operands.

The exception is rule 4, where a factor too wide to pair multiplies a sum that is already
collapsed:

- a 0/1 literal: the product is the sum or 0, and nothing is added;
- whole values `v`: the product is `v * w` exactly, so the sum's bound times `|v|`;
- fractional values: `floor(p * w / q)`, which adds `|w| / (2q) + 1` units, with `q`
  from 2^16 to 2^24.

Collapsing this form too means pairing every list of the sum with the factor's values. It
is exact to half a unit per list and costs the factor's number of values times the
entries. #6315 measured that form at 26 to 27 s against 5 to 6 s for the switch on the 14B
and 24B auto-tiling prefill objectives. The plan keeps the switch. The largest error
measured over the 50 objectives is 1.2e-7 of the cost.

**Worked example**, each form checked with CP-SAT at every assignment. Buffer a has
`split_a = [1, 2, 4]` over `division_a` in `0..2`. Buffer b has `split_b = [1, 3]` over
`division_b` in `0..1`.

1. *One table over both*, `10 / (split_a * split_b) ** 0.5`. Sizes (3, 2) give strides
   (2, 1) and `joint == 2*division_a + division_b`. The float list is
   `[10, 5.7735, 7.0711, 4.0825, 5, 2.8868]`. Collapsed, it is
   `[10240, 5912, 7241, 4180, 5120, 2956]`. At `division_a = 1`, `division_b = 1` the
   joint index is 3 and the entry is 4180: 10 / 6 ** 0.5 is 4.0825, and times 1024 that
   is 4180.46. The worst entry is 0.46 units off.
2. *Max by rung*, `Max(split_a / 2, split_b)`. The operand lists are `[0.5, 1, 2]` and
   `[1, 3]`. A Max is never under the largest operand minimum, 1, so 0.5 is clamped to 1.
   The ladder is `[1, 2, 3]`, the rungs are `[0, 0, 1]` for a and `[0, 2]` for b. At
   `division_a = 2`, `division_b = 0` the rungs are 1 and 0, their max is 1, and the
   entry is 2, which is Max(2, 1).
3. *Rank*. A menu `tile_a = [1, 2, 1]` gives `8 / tile_a = [8, 4, 8]`. `distinct` is
   `[4, 8]` and `rank_of` is `[1, 0, 1]`: divisions 0 and 2 share rank 1, and
   `distinct[rank_of[d]]` is the value at every division `d`.

4. *Collapse order*, at `split_a = 2`. For `7 * split_a ** 0.5` the entry is 10137, the
   rounding of 7 * 1448.15; rounding first and multiplying after gives 7 * 1448 = 10136.
   For `split_a ** 0.5 + split_a ** 0.7` the one entry is 3112, the rounding of
   1448.15 + 1663.49; collapsing each list alone gives 1448 + 1663 = 3111.

The *exact* tests of every stage are the check on this mapping: they solve at every
assignment of the inputs, so an entry read from the wrong position fails.

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

In the tests below, *exact* means: the lowered objective, solved at every assignment of
the inputs, equals `COST_SCALE` times the expression evaluated by sympy, within the bound
stated under "Collapsing float lists into linear terms".

### Stage 1: numpy values (`cost_expr_numpy.py`)

One recursive function, `numpy_value(expr, env)`:

1. a symbol is its array in `env`, a number is a float, `True` and `False` are
   themselves;
2. the arguments are evaluated first;
3. the numpy function is found by the node's class name: a table of overrides first,
   then numpy's own function of that name;
4. with no function it raises `NotImplementedError`.

A caller that gets `NotImplementedError` evaluates that node through `sympy.lambdify`
instead. This is the only use of lambdify. It is cached per node and counted.

Overrides, for names numpy does not share: Add, Mul, Pow, Abs; Min and Max
(`np.minimum`, `np.maximum`); ceiling; the six relationals; And, Or, Not; ITE
(`np.where`); Piecewise (`np.select` over its pairs, every arm computed, NaN when no arm
holds); `RelayoutCharge` (`is_lx * prices[division]`, 0 past the table); an implemented
function through its `_imp_`, vectorized.

Two rules for the lookup, both checked on numpy 2.5.1 and sympy 1.14:

- The by-name lookup takes ufuncs only. `np.floor`, `np.log`, `np.exp` and `np.sign` are
  ufuncs and mean what sympy means. `np.sinc` has sympy's name and another meaning
  (`sinc(1)` is 0.84 in sympy and 0 in numpy), and it is not a ufunc.
- An n-ary node folds its arguments pairwise. `np.add.reduce` over a list that mixes a
  constant with arrays raises.

`values_on_grid(expr, columns, shape)` is `numpy_value` at every joint assignment of the
node's inputs, row-major, with the lambdify backup. It refuses a value that is not finite.

The overrides cover everything the cost model builds today: Add, Mul, Pow (exponent -1 and
floats), Min, Max, floor, log, Piecewise ending in `True`, Eq/Le/Ge/Lt/Gt, And, Or,
numbers, `decode_delivery_price_*` (`cost_model.py:1426-1449`) and `RelayoutCharge`
(`plan_solver.py:487`). ITE and `False` are latent: they need auto tiling or cost
parameters other than production's.

Tests, `tests/inductor/test_cost_expr_numpy.py`:

- every override, and every by-name function the cost model uses, against exact sympy
  (`float(expr.xreplace(point))`) at every grid point;
- the cost model's own forms: `floor(1024/split)**-0.25`, `Min(c*log(a), Max(0, 3.0 -
  c*log(b)))`, `1/split >= Rational(5, 128)` as a condition, an Or of Ge, nested
  Piecewise, ITE with `False`, `x**0.0`;
- Piecewise: an infinite value in an arm not taken is discarded; no arm true is refused;
- `RelayoutCharge` agrees with its `_imp_`; the real decode delivery price is evaluated
  with no lambdify; an undefined function with no `_imp_` is refused;
- a name numpy lacks (`sympy.sec`) raises `NotImplementedError` and is then evaluated
  through lambdify, once; `sympy.sinc` takes that path too and gives sympy's value;
- `values_on_grid`: row-major order over axes of three sizes, two symbols on one axis, a
  constant broadcast;
- guard: objectives built by the cost model (the builders `test_op_features.py` and
  `test_cost_model_replication.py` already use) never reach lambdify, and agree with
  `sympy.lambdify(modules="math")`, the annealer's evaluator (`sa_cooptimizer.py:849`).

### Stage 2: pure helpers (`cpsat_table_lowering.py`, first slice)

Functions of numbers and lists, with no CP-SAT model:

- the ladder of a Min/Max: the values it can return, ascending, and the bound past which
  an operand's value is clamped;
- rungs: each of a list's clamped values as its position on a ladder, or None when one is
  not on it;
- the reachable fold: for Min/Max factors given as rungs over their inputs, the tuples of
  rungs that occur together and the steps that compute which one occurs, or None past
  the cap;
- scaling: one rounding of a float list onto `COST_SCALE`;
- the switch scale: for a factor with fractional values, the power of two `q` that holds
  each value as `p/q` with every product inside int64.

Tests, `tests/inductor/test_cpsat_table_lowering.py`:

- a Min's ladder drops the values above its smallest operand maximum; a Max mirrors it;
- values equal to 12 digits share a rung; a value off the ladder gives None;
- the Min of rungs is the rung of the Min (seeded random operands);
- the fold numbers only the tuples that occur together; a constant operand seeds its
  start state; operands sharing an input join one group; a step past the cap is refused;
- scaling rounds once;
- the switch scale is 1 for whole values and otherwise keeps `p * w` under 2^60;
- the pair `q*P <= p*w`, `q*P >= p*w - (q-1)` admits only `P = (p*w) // q`, negatives
  included.

### Stage 3: inputs, indexes, tables, linear sums

- Inputs are mapped from the printer's two maps. A buffer's symbols share its division as
  their index. Any other variable is indexed by its domain, through a position lookup
  when the domain has gaps. A domain past the cap is not an input.
- A node whose inputs fit the cap is one table over them, read through a joint index in
  the order `values_on_grid` uses.
- The objective is a constant plus one list per index. Lists are collapsed here and
  nowhere below: the coefficient is applied and terms on one index are added in float,
  then the list is rounded once. A list affine in its index is a linear term with no
  table, and a constant list is a number.
- The whole objective is planned before anything is added to the model.
- `lower_cost_with_tables(model, sym_map, buffer_map, cost_expr, cap)` returns the
  objective, or None with the model untouched.

After this stage a sum whose terms each fit one table is lowered exactly. Anything else
returns None.

Tests:

- a sum of single-buffer terms is exact, with no table across two buffers;
- a coefficient is applied before the rounding: `7 * split ** 0.5` at split 2 is 10137,
  not 7 * 1448;
- terms on one index are added before the rounding: `split ** 0.5 + split ** 0.7` at
  split 2 is one entry, 3112, not 1448 + 1663;
- one table over two inputs of different sizes, for an expression that is not symmetric
  in them, is exact: a list and an index in different orders fail it;
- a variable with gaps in its domain reads the entry of its value;
- a Piecewise cost, a function with no numpy name and a relayout charge are exact;
- two menu lengths on one division are refused; a variable with gaps in its domain gets a
  position; a domain past the cap is no input;
- an affine list adds no table; a constant list adds no variable;
- an entry too large for int64 sums is refused;
- a relayout of a buffer this solve does not divide costs 0;
- an expression it cannot lower leaves the model's variables and constraints unchanged.

### Stage 4: rule 1 and parts

- A Min/Max of indexed operands is indexed by rung (rule 1).
- Equal values share one rank, and a product of two short lists is one table over the
  pair of their ranks.
- A function of an indexed node keeps that node's index.
- A node takes the cheaper of one table over its inputs and the form built from its parts
  (rule 3).

Tests:

- a Max over three buffers is exact with no table as large as the joint table, and each
  operand lands on the rung of its clamped value;
- a menu with repeated values reads the same entry through its rank as through its
  division;
- a product of two lists of different sizes, not symmetric in them, is exact;
- equal values share a rank before pairing; a pair past the cap is refused;
- a function of a short list adds no index;
- the smaller form in entries wins, one case each way.

### Stage 5: rule 2

Min/Max factors of one product, over shared inputs, become one indexed value over the
reachable tuples of their rungs, when crossing their lists would pass the cap.

Tests: a product of four Max factors over three shared buffers is exact under a cap their
crossed ladders pass; no fold when the crossed lists fit.

### Stage 6: rule 3 in the objective

- A Min/Max too wide to index, or whose table is over twice its linear form, stays a
  `lin_min`/`lin_max` over linear operands.
- A coefficient on the Min/Max is applied to its operands' lists before they are
  collapsed. A negative one turns Min into Max.
- Each operand sits behind a variable of its own. With a long sum of literals inside
  `lin_max`, CP-SAT's presolve ran out of memory (`bad_alloc` past 40 GB on a Qwen3 0.6B
  prefill objective).

Tests: a Min of two sums of residency literals stays linear and is exact, with no table
over its inputs; a near tie keeps the table; a negative coefficient gives a `lin_max`;
a fractional coefficient on a linear Min is within the largest operand's bound; each
operand is a single variable.

### Stage 7: rule 4

- A product whose factor cannot pair with the rest is switched on the factor's value.
  Whole values take one equality per value. Fractional values take the floor pair of
  Stage 2. This is the one form where a float multiplies a sum already collapsed, with
  the bound stated under "Collapsing float lists into linear terms".
- A 0/1 literal, one minus it, or a multiple of either, times a term is the two-value
  case: the term is lowered with the literal fixed on, and no table reads the literal. A
  relayout charge is its literal times a price by division.
- A gate whose rest is constant is a coefficient on the literal.

Tests: a fractional factor times a sum of lists with fractional entries is within the
rule 4 bound; a factor times a Min too wide to pair is exact, also inside a Min; a relayout
charge and a gated term are exact with no table larger than the menu; a constant rest adds
no table; two wide factors are refused with the model untouched; a product too wide for
its factor's precision is refused.

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

- an expression the tables cannot lower goes to the printer;
- flag off: a `CpSatLayoutSolver` solve never calls the table module (patched to raise);
- flag on: the same solve never calls the printer, ends `OPTIMAL`, and its plan's exact
  cost is no higher than the printer's plan, within one `COST_SCALE` unit per table.

Known difference: the tables' objective is `COST_SCALE` (1024) times the cost, and the
printer's is in cost units. Only the uniqueness probe reads it
(`ilp_solver_ortools.py:825-827`, off by default); its "within 1" becomes within 1/1024.

## Verification

Each PR:

- `python3 -m pytest tests/inductor/test_cost_expr_numpy.py
  tests/inductor/test_cpsat_table_lowering.py`
- `make precommit`, `mypy torch_spyre tests` and `make check-all-configs`. Both test files
  are listed in `tests/configs/torch_spyre_tests/inductor/test_cpsat_table_lowering_config.yaml`.
- `git commit -s`, the Apache header on new files, `feat(scratchpad): ...` subjects.
- Stages 1 to 7: `git diff main --stat` shows new files only.

After Stage 8, posted on that PR:

1. **Same results as #6315.** Replay the 50 captured objectives (`~/pr6301-review/replay.py`,
   #6315's results in `~/pr6301-review/tableonly/`). Every capture is `OPTIMAL` at the
   exact cost #6315 reaches, and no objective is further from exact than 1.2e-7. A
   different plan at that cost is a tie.
2. **Size and time, reported.** Entries, largest table, lowering time and solve time per
   capture, beside the table in Estimates. A capture with more entries or a slower solve
   than #6315 is explained in the PR.
3. **Flag on.** The planner suites with `CPSAT_TABLE_LOWERING=1`:
   `test_scratchpad_solver.py`, `test_scratchpad_use.py`, `test_cost_model_replication.py`
   and `test_solver_relayout_decision.py`. #6315's CI ran every shard with the tables on
   the default path and all passed; its one red job was a card fault
   (`DdrInitRetryLimitExceeded`).
4. **Flag off.** The same suites with no flag, unchanged.

## Next step

Stage 1 starts in a new torch-spyre worktree off `upstream/main`. PRs to torch-spyre open
on the owner's go.
