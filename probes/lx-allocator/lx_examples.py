# Small, runnable examples for each class on the design page. Each block runs on
# its own after `from lx_design import *`; the page shows the block and its output.

# example: SpyreOptimizer
class SdpaOptimizer(SpyreOptimizer):
    """build_graph and commit stand in for the IR side."""

    def build_graph(self, ir):
        return sdpa_decode_core(kv_len=100)

    def commit(self, graph):
        for e in graph.data_edges:
            for t in e.transforms:
                print("insert", type(t).__name__, "on", e.src.name, "->", e.dst.name)

opt = SdpaOptimizer(settings=SolveSettings(deterministic=True, workers=2))
graph = opt.plan_allocation(ir=None)
print("solution:", opt.solution.status, "| objective", int(opt.solution.objective))
node = {n.name: n for n in graph.nodes}
print("sum split:", node["sum"].work_split.splits)
node["sum"].work_split = WorkSplit({"h": 8, "L": 4})  # split the padded key axis by hand
node["max"].layout = Layout(0)  # stick max's output on h instead
print("check_decisions after two hand edits (no solver involved):")
for failure in check_decisions(graph):
    print("  ", failure)
g = sdpa_decode_core()  # a user hint that cannot hold: 8 cores on L, which 100 does not divide
g.pin(next(n for n in g.nodes if n.name == "sum"), "split", {"h": 4, "L": 8}, "hint")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    SdpaOptimizer(settings=SolveSettings(deterministic=True, workers=2)).optimize(g)
print("with the hint:", caught[0].message)

# example: CostModel
g = sdpa_decode_core()
objective = CostModel().objective(g)
print(len(objective.free_symbols), "symbols; the coefficients that mention scores:")
scores, k_edge = g.nodes[3], g.data_edges[1]
for name in (Sym.resident(scores), Sym.cores(scores), Sym.split(scores, "K"), Sym.copy(scores),
             Sym.restick(k_edge), Sym.restick_resident(k_edge)):
    print(f"  {int(objective.coeff(sympy.Symbol(name))):>8}  {name}")
model = OrToolsEncoder(cost_model=CostModel()).encode(g)  # the encoder consumes the cost model
print("every name has a variable in the model:", all(s.name in model.symbols for s in objective.free_symbols))
print("encoded into CP-SAT's variables:", len(model.cp.Proto().objective.vars), "terms in the objective")

class WithTypo(CostModel):
    def objective(self, graph):
        return super().objective(graph) + sympy.Symbol("resident[nowhere]")

try:
    OrToolsEncoder(cost_model=WithTypo()).encode(g)
except KeyError as e:
    print("an unknown name fails while the encoder encodes it:", e)

# example: SpyreOptimizationGraph
g = swiglu_mlp()
print(len(g.nodes), "nodes in execution order:", [n.name for n in g.nodes])
down = g.nodes[-1]
print("down reads:", [(e.src.name, e.operand) for e in down.inputs])
print(len(g.data_edges), "data edges,", len(g.execution_edges), "execution edges,", len(g.pins), "pins")
print("x is live for steps", g.lifetime(g.nodes[0]), "(until up, its last reader, has run)")

# example: OpNode
g = sdpa_decode_core()
node = {n.name: n for n in g.nodes}
print("scores reads k [h, L, d] as:", node["scores"].reads[1])  # loop axis -> buffer dim
print("sum: padded reductions", node["sum"].padded_reductions(),
      "| L splits in whole sticks:", node["sum"].whole_stick_factors("L", 64))
print("sub_exp reads max [h, 1] with broadcast dims:", node["sub_exp"].broadcast_dims(1))
print("sub_exp reads scores at its own index:", node["sub_exp"].same_index_read(0))
print("probs_v LX units per core, split 32 ways on h:", node["probs_v"].footprint({"h": 32}, {}, g.alignment))

# example: PointwiseNode, MatmulNode, ReductionNode, InputNode
for cls in (OpNode, PointwiseNode, MatmulNode, ReductionNode, InputNode):
    print(f"{cls.__name__:13} {cls.restrictions}")
print("MatmulNode adds x, y, k, n, which MATMUL_STICKS reads:",
      {f: getattr(MatmulNode, f) for f in ("x", "y", "k", "n")})
mm = sdpa_decode_core().nodes[3]
try:
    mm.restrictions = Restriction.NONE
except AttributeError as e:
    print("frozen per op type:", e)

# example: DataEdge
g = sdpa_decode_core()
Sketch(settings=SolveSettings(deterministic=True, workers=2)).optimize(g)
for e in g.data_edges:
    steps = [type(t).__name__ for t in e.transforms]
    print(f"{e.src.name:>7} -> {e.dst.name}[{e.operand}]  stick {e.src.layout.stick} -> reads "
          f"{e.read_layout().stick}  {steps}{'  in place' if e.in_place else ''}")

# example: ExecutionEdge
g = swiglu_mlp(tokens=4096)  # prefill
Sketch(settings=SolveSettings(deterministic=True, workers=2), coarse_tiling=True).optimize(g)
for x in g.execution_edges:
    print(f"{x.before.name:>8} | {x.after.name:<8} cut={x.cut}")
print("tilings:", {n.name: n.work_split.tiling for n in g.nodes if n.runs})
print("(inputs never run a nest; gate, up, silu_mul and down share one, tiled 4 ways on M)")

# example: Pin
g = swiglu_mlp()
g.pin(g.nodes[-1], "split", {"M": 32}, "hint")  # a user hint on down
for p in g.pins:
    print(f"  {p.source:12} {'internal' if p.internal else 'user':8} {p.what:10} {str(p.value):10} on {p.where}")
model = OrToolsEncoder().encode(g)
print("folded into constants by the encoder:",
      sum(1 for _, rule, i in model.provenance if rule.startswith("pin") and i is None))

# example: Transform, CopyOut, Restickify, Relayout
for kind in sorted([Relayout, Restickify, CopyOut], key=lambda k: k.ORDER):
    print(f"{kind.ORDER} {kind.__name__:11} forced={kind.forced!s:5} in_lx={kind.in_lx!s:5} "
          f"elidable={kind.elidable}")
step = Restickify(to_stick=1)
print("k [h, L, d] stuck on d, after Restickify(to_stick=1):", step.apply(Layout(2)))
print("a step carries its own memory:", {f: getattr(step, f) for f in ("resident", "offset", "footprint")})

# example: Restriction
g = sdpa_decode_core()
node = {n.name: n for n in g.nodes}
flags = restrictions_for(node["sum"])
print("sum's flags:", flags, "| has PADDED_REDUCTION_UNSPLIT:", Restriction.PADDED_REDUCTION_UNSPLIT in flags)
print("what it asks of sum (operand, dim, axis, factors):",
      splits_while_stuck(node["sum"], Restriction.PADDED_REDUCTION_UNSPLIT))
print("always on, not flags:", [v for k, v in vars(Always).items() if k.isupper()][:4], "...")
plain = len(OrToolsEncoder().encode(g).provenance)
test = len(OrToolsEncoder().encode(g, {ReductionNode: ReductionNode.restrictions & ~Restriction.PADDED_REDUCTION_UNSPLIT}).provenance)
print(f"constraints posted: {plain}; with a test dropping PADDED_REDUCTION_UNSPLIT from ReductionNode: {test}")
class Narrow(OrToolsEncoder):
    ENCODES = Restriction.WHOLE_STICKS | Restriction.PADDED_REDUCTION_UNSPLIT
print("coverage:", coverage(g, OrToolsEncoder()) or "every flag in use has an encoding",
      "| a narrower encoder:", coverage(g, Narrow()))

# example: Encoder
g = sdpa_decode_core()
enc = OrToolsEncoder()
model = enc.encode(g)
enc.read_back(CpSatSolution(cp_model.CpSolver(), cp_model.UNKNOWN), g)  # a solve that found nothing
print("after a solve with no plan:", {n.name: n.work_split for n in g.nodes if n.runs})
solution = OrToolsFactory().solve(model, SolveSettings(deterministic=True, workers=2))
enc.read_back(solution, g)
print("after a plan:", solution.status, {n.name: n.work_split.splits["h"] for n in g.nodes if n.runs})
print("the public contract:", sorted(Encoder.__abstractmethods__))

# example: OrToolsEncoder
g = sdpa_decode_core()
model = OrToolsEncoder(cost_model=CostModel()).encode(g)  # always-on rules plus each node type's flags
proto = model.cp.Proto()
print(len(proto.variables), "variables,", len(proto.constraints), "constraints,",
      len(proto.objective.vars), "objective terms from the cost model")
removed = {}
for _, rule, n in model.pruned:
    removed[rule] = removed.get(rule, 0) + n
for rule, n in sorted(removed.items()):
    print(f"{n:5} table entries pruned by {rule}")
posted = sorted({rule for _, rule, i in model.provenance if i is not None})
print(len(posted), "kinds of constraint posted, op to graph level:")
for rule in posted:
    print("  ", rule)

# example: CpmpyEncoder, SAEncoder
g = sdpa_decode_core()
for cls in (CpmpyEncoder, SAEncoder):
    try:
        cls().encode(g)
    except NotImplementedError as e:
        print(f"{cls.__name__:12} {e}: would encode the same restriction classes")

# example: ModelAbstractFactory, OrToolsFactory, CpmpyFactory, SAFactory
family = FACTORIES["ortools"](placement="pack", coarse_tiling=True)
enc = family.encoder(CostModel())
print("encoder:", type(enc).__name__, "| consumes:", type(enc.cost_model).__name__, "| coarse tiling:", enc.coarse_tiling)
print("its backend is private:", [a for a in vars(family) if "backend" in a], "| places by", family._backend.placement)
try:
    family.fallback(None, SolveSettings())
except NotImplementedError as e:
    print("a solve with no plan falls back to:", e)
print("families:", sorted(FACTORIES))

# example: Model
g = swiglu_mlp()
model = OrToolsEncoder().encode(g)
count = {}
for _, rule, i in model.provenance:
    key = "pins, folded into constants" if rule.startswith("pin") else rule + ("" if i is not None else " (placed by the backend)")
    count[key] = count.get(key, 0) + 1
for rule, n in sorted(count.items()):
    print(f"{n:3}  {rule}")
print(len(model.symbols), "names in the symbol table")

# example: Backend
for placement in ("no_overlap", "pack"):  # CP-SAT can do either; annealing and MIP solvers pack
    g = swiglu_mlp(tokens=4096)  # prefill, tiled 4 ways on M
    opt = Sketch(settings=SolveSettings(deterministic=True, workers=2), coarse_tiling=True, placement=placement)
    opt.optimize(g)
    print(f"{placement:10} objective {int(opt.solution.objective)}, offsets "
          f"{ {n.name: n.offset for n in g.nodes if n.resident} }")
boxes = [("a", 0, 3, 4, []), ("b", 1, 4, 2, []), ("c", 3, 5, 4, ["a"])]  # (key, start, end, units, tied to)
print("its packer alone:", Packer().pack(boxes), "(c in place over a shares its offset)")
g = swiglu_mlp()
plans = set()
for run in range(3):
    enc = OrToolsEncoder()
    s = CpSatBackend().solve(enc.encode(g), SolveSettings(deterministic=True, workers=2))  # objective already in the model
    enc.read_back(s, g)
    plans.add(tuple((n.name, tuple(n.work_split.splits.items()), n.resident) for n in g.nodes if n.runs))
print("3 deterministic solves of the 128-token MLP gave", len(plans), "plan;", s.status, int(s.objective))
