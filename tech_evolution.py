import itertools
import math
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

# ============================================================
# 1. Boolean function representation
# ============================================================

@dataclass
class BoolFunction:
    """
    Represents a multi-output Boolean function.
    truth_table: list of tuples of output bits, indexed by input pattern index.
                 Input patterns are ordered lexicographically on bits (0..2^n_inputs-1).
    """
    n_inputs: int
    n_outputs: int
    truth_table: List[Tuple[int, ...]]  # length 2^n_inputs, each element is length n_outputs

    def distance_to(self, other: "BoolFunction") -> int:
        """Hamming distance between two multi-output functions on all input patterns."""
        assert self.n_inputs == other.n_inputs
        assert self.n_outputs == other.n_outputs
        
        if len(self.truth_table) != len(other.truth_table):
            # This should not happen if n_inputs match, but safety check against empty tables
            return max(len(self.truth_table), len(other.truth_table)) * self.n_outputs

        dist = 0
        for row_self, row_other in zip(self.truth_table, other.truth_table):
            for a, b in zip(row_self, row_other):
                if a != b:
                    dist += 1
        return dist

    def is_equal(self, other: "BoolFunction") -> bool:
        return (
            self.n_inputs == other.n_inputs
            and self.n_outputs == other.n_outputs
            and self.truth_table == other.truth_table
        )

    @staticmethod
    def from_lambda(n_inputs: int, n_outputs: int, fn):
        """
        Helper: build from a Python function fn(inputs) -> tuple of bits.
        inputs is a list/tuple of length n_inputs with 0/1.
        """
        tt = []
        for bits in itertools.product([0, 1], repeat=n_inputs):
            out = fn(bits)
            if n_outputs == 1:
                # allow single int
                if not isinstance(out, (tuple, list)):
                    out = (out,)
            tt.append(tuple(int(b) for b in out))
        return BoolFunction(n_inputs, n_outputs, tt)


# ============================================================
# 2. Circuit components / technologies
# ============================================================

@dataclass
class Component:
    """
    Encapsulated component (technology) with an interface and a BoolFunction.
    cost: sum of primitive costs (roughly number of gates).
    name: human-readable label.
    """
    name: str
    function: BoolFunction
    cost: int
    is_primitive: bool = False


# ============================================================
# 3. Goals ("needs")
# ============================================================

@dataclass
class Goal:
    """
    A goal is a target Boolean function we want to approximate.
    """
    name: str
    target: BoolFunction
    # best current approximations: list of components (could keep several non-comparable ones)
    best_components: List[Component] = field(default_factory=list)


# ============================================================
# 4. Library of technologies
# ============================================================

class TechnologyLibrary:
    def __init__(self):
        self.primitives: List[Component] = []
        self.constants: List[Component] = []
        self.technologies: List[Component] = []  # non-primitive, discovered tech
        self.goals: List[Goal] = []
        self.next_id: int = 0

    def add_primitive(self, component: Component):
        component.is_primitive = True
        self.primitives.append(component)

    def add_constant(self, component: Component):
        # treat constants as primitives for selection probabilities
        component.is_primitive = False
        self.constants.append(component)

    def add_goal(self, goal: Goal):
        self.goals.append(goal)

    def add_technology(self, comp: Component):
        self.technologies.append(comp)

    def new_name(self, prefix="tech") -> str:
        self.next_id += 1
        return f"{prefix}_{self.next_id}"


# ============================================================
# 5. Combining components into a new circuit
#    (simplified but faithful to the paper's spirit)
# ============================================================

@dataclass
class CompositeWiring:
    """
    Describes how N components are combined to create a new function.
    """
    # For each component i (0..N-1):
    #   input_map[i] is a list of length n_inputs_of_c[i].
    #   Each element is either:
    #     - An integer >= 0: index of a global input
    #     - A tuple (j, out_idx): output 'out_idx' of component 'j' (where j < i).
    #       (We enforce strictly acyclic/feed-forward wiring by only allowing connections from previous components)
    
    component_input_maps: List[List[Union[int, Tuple[int, int]]]]
    
    # Which outputs of which components are exposed as global outputs?
    # List of (component_index, output_index)
    exposed_outputs: List[Tuple[int, int]]


def build_composite_wiring(components: List[Component], prob_new_input: float = 0.2) -> CompositeWiring:
    """
    Builds a random wiring for a list of components C0, C1, ... CN-1.
    Rules:
    - C0's inputs are always new global inputs.
    - For C_k (k > 0), each input is either:
      - A new global input (prob p)
      - Wired to an output of some C_j where j < k (prob 1-p)
    - Outputs are randomly selected subset of all available outputs.
    """
    input_maps = []
    global_inputs_count = 0
    
    # Track available outputs for wiring: list of (comp_idx, out_idx)
    available_outputs: List[Tuple[int, int]] = []

    for comp_idx, comp in enumerate(components):
        my_map = []
        for _ in range(comp.function.n_inputs):
            # Decide connection source
            # If it's the very first component, it MUST consume global inputs (or constants if we had them)
            # effectively acting as the "input layer".
            # For simplicity, let's say C0 always takes global inputs.
            # Later components can take global inputs or previous outputs.
            
            use_global = False
            if comp_idx == 0:
                use_global = True
            elif not available_outputs:
                use_global = True
            elif random.random() < prob_new_input:
                use_global = True
            
            if use_global:
                my_map.append(global_inputs_count)
                global_inputs_count += 1
            else:
                # Wire to previous
                src = random.choice(available_outputs)
                my_map.append(src)
        
        input_maps.append(my_map)
        
        # Add my outputs to available pool
        for out_i in range(comp.function.n_outputs):
            available_outputs.append((comp_idx, out_i))

    # Select global outputs
    # We can pick from ANY component's outputs (available_outputs has all of them now)
    if not available_outputs:
        exposed = []
    else:
        n_exposed = random.randint(1, len(available_outputs))
        exposed = random.sample(available_outputs, n_exposed)
        # sort for determinism/tidiness
        exposed.sort()

    return CompositeWiring(
        component_input_maps=input_maps,
        exposed_outputs=exposed
    )


def realize_composite_function(
    components: List[Component],
    wiring: CompositeWiring
) -> BoolFunction:
    """
    Simulates the N-component circuit.
    """
    # Determine number of global inputs
    # It's simply the max index found in the wiring maps + 1
    max_in = -1
    for cmap in wiring.component_input_maps:
        for signal in cmap:
            if isinstance(signal, int):
                if signal > max_in:
                    max_in = signal
    n_global_inputs = max_in + 1

    # Simulate
    # truth_table has 2^n_global_inputs rows
    # each row is tuple of outputs
    
    tt = []
    n_outputs = len(wiring.exposed_outputs)
    
    # Check for excessive input size BEFORE processing
    # (Though caller usually checks this, we do it implicitly by range size)
    # 2^20 is ~1M, manageable but slow. 2^18 is safer limit.
    
    for global_bits in itertools.product([0, 1], repeat=n_global_inputs):
        # We need to store outputs of each component for this row
        # comp_outputs[i] = tuple of bits
        comp_outputs: List[Tuple[int, ...]] = []
        
        for comp_idx, comp in enumerate(components):
            # gather inputs
            input_vals = []
            for signal in wiring.component_input_maps[comp_idx]:
                if isinstance(signal, int):
                    # Global input
                    input_vals.append(global_bits[signal])
                else:
                    # Wiring from previous: (src_comp_idx, src_out_idx)
                    src_c, src_o = signal
                    val = comp_outputs[src_c][src_o]
                    input_vals.append(val)
            
            # evaluate component
            row_idx = bits_to_index(input_vals)
            out_vals = comp.function.truth_table[row_idx]
            comp_outputs.append(out_vals)
        
        # Collect global outputs
        row_out = []
        for (c_idx, o_idx) in wiring.exposed_outputs:
            row_out.append(comp_outputs[c_idx][o_idx])
        
        tt.append(tuple(row_out))

    return BoolFunction(n_global_inputs, n_outputs, tt)


def bits_to_index(bits: List[int]) -> int:
    """Convert bit list (MSB-first) to integer index. Here we take bit[0] as MSB."""
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    return idx


# ============================================================
# 6. Goal evaluation and encapsulation
# ============================================================

def evaluate_and_maybe_add(
    lib: TechnologyLibrary,
    new_comp: Component,
    cost_weight: float = 1.0
) -> List[Goal]:
    """
    Evaluate new_comp against all goals. If it improves any goal, add as a technology.
    Returns list of goals that were improved.
    """
    improved_goals = []

    for goal in lib.goals:
        # Only compare if input/output dimensions match
        if (
            goal.target.n_inputs != new_comp.function.n_inputs
            or goal.target.n_outputs != new_comp.function.n_outputs
        ):
            continue

        new_dist = goal.target.distance_to(new_comp.function)
        
        should_add = False
        if not goal.best_components:
            # No current approximation, accept
            goal.best_components.append(new_comp)
            should_add = True
        else:
            # Compute distances for existing best
            current_dists = [goal.target.distance_to(c.function) for c in goal.best_components]
            min_current = min(current_dists)

            if new_dist < min_current:
                # Strictly better: replace all (for simplicity)
                goal.best_components = [new_comp]
                should_add = True
            elif new_dist == min_current:
                # Possibly add if cheaper AND not identical to an existing one
                same_fn_cheaper = False
                for c in goal.best_components:
                    if new_comp.function.is_equal(c.function) and new_comp.cost < c.cost:
                        same_fn_cheaper = True
                        break
                if same_fn_cheaper:
                    # replace cheaper functionally equivalent
                    goal.best_components = [
                        new_comp if (c.function.is_equal(new_comp.function) and c.cost > new_comp.cost) else c
                        for c in goal.best_components
                    ]
                    should_add = True
        
        if should_add:
            improved_goals.append(goal)

    if improved_goals:
        lib.add_technology(new_comp)

    return improved_goals


# ============================================================
# 7. Selection of components for combination
# ============================================================

def select_component(lib: TechnologyLibrary) -> Component:
    """
    Select a component according to probabilities:
    - 0.5: primitives
    - 0.015: constants 0/1
    - 0.485: existing technologies
    (simplified; if a bucket is empty, fall back to others)
    """
    r = random.random()
    # primitives
    if r < 0.5 and lib.primitives:
        return random.choice(lib.primitives)
    # constants
    if r < 0.515 and lib.constants:
        return random.choice(lib.constants)
    # technologies
    if lib.technologies:
        return random.choice(lib.technologies)
    # fallback: primitives or constants if technologies empty
    if lib.primitives:
        return random.choice(lib.primitives)
    if lib.constants:
        return random.choice(lib.constants)
    raise RuntimeError("No components available!")


# ============================================================
# 8. Example setup: NAND primitive and some basic goals
# ============================================================

def make_nand_primitive(lib: TechnologyLibrary):
    def nand_fn(bits):
        x, y = bits
        return (1 - (x & y),)

    nand = Component(
        name="NAND",
        function=BoolFunction.from_lambda(2, 1, nand_fn),
        cost=1,
        is_primitive=True,
    )
    lib.add_primitive(nand)


def make_constants(lib: TechnologyLibrary):
    # Constant 0 and 1 as single-output, zero-input functions.
    zero = Component(
        name="CONST_0",
        function=BoolFunction.from_lambda(0, 1, lambda bits: (0,)),
        cost=1,
    )
    one = Component(
        name="CONST_1",
        function=BoolFunction.from_lambda(0, 1, lambda bits: (1,)),
        cost=1,
    )
    lib.add_constant(zero)
    lib.add_constant(one)



def make_basic_goals(lib: TechnologyLibrary):
    # ==========================
    # 1. Basic Logical Functions
    # ==========================

    # NOT: 1 -> 1
    lib.add_goal(Goal("NOT", BoolFunction.from_lambda(1, 1, lambda b: (1 - b[0],))))

    # IMPLY: 2 -> 1 (not A or B)
    lib.add_goal(Goal("IMPLY", BoolFunction.from_lambda(2, 1, lambda b: (int((not b[0]) or b[1]),))))

    # ==========================
    # 2. N-way Logic Gates (2 <= n <= 8)
    # ==========================
    
    # We want up to n=8.
    # Note: Truth table size 2^n. For n=8, 256 rows. Very fast.
    for n in range(2, 9): # 2..8
        # N-way XOR
        def make_xor_n(n_val):
            def fn(bits):
                res = 0
                for b in bits:
                    res ^= b
                return (res,)
            return fn
        lib.add_goal(Goal(f"XOR_{n}", BoolFunction.from_lambda(n, 1, make_xor_n(n))))

        # N-way OR
        def make_or_n(n_val):
            def fn(bits):
                res = 0
                for b in bits:
                    res |= b
                return (res,)
            return fn
        lib.add_goal(Goal(f"OR_{n}", BoolFunction.from_lambda(n, 1, make_or_n(n))))

        # N-way AND
        def make_and_n(n_val):
            def fn(bits):
                res = 1
                for b in bits:
                    res &= b
                return (res,)
            return fn
        lib.add_goal(Goal(f"AND_{n}", BoolFunction.from_lambda(n, 1, make_and_n(n))))


    # ==========================
    # 3. M-Bitwise Operations (2 <= m <= 7)
    # ==========================
    # Inputs: 2*m. 
    # For m=7, inputs=14 -> 16k rows. Still manageable.
    
    def make_bitwise_goal(m: int, name_prefix: str, op):
        def fn(bits):
            res = []
            for i in range(m):
                val = op(bits[i], bits[m + i])
                res.append(val)
            return tuple(res)
        return Goal(f"{name_prefix}_{m}", BoolFunction.from_lambda(2 * m, m, fn))

    for m in range(2, 8): # 2..7
        lib.add_goal(make_bitwise_goal(m, "BITWISE_XOR", lambda x, y: x ^ y))
        lib.add_goal(make_bitwise_goal(m, "BITWISE_OR", lambda x, y: x | y))
        lib.add_goal(make_bitwise_goal(m, "BITWISE_AND", lambda x, y: x & y))


    # ==========================
    # 4. Arithmetic
    # ==========================

    # Full Adder (3 -> 2)
    def full_adder(bits):
        a, b, cin = bits
        s = a ^ b ^ cin
        cout = (a & b) | (a & cin) | (b & cin)
        return (s, cout)
    lib.add_goal(Goal("FULL_ADDER", BoolFunction.from_lambda(3, 2, full_adder)))

    # K-Bit Adder (1 <= k <= 8)
    # Inputs: 2*k. (We won't assume Cin for k-bit block adder to keep inputs lower? 
    # Or standard k-bit adder has NO Cin input usually, just A+B? 
    # The prompt says "k-bit-adder 2k k+1 addition", so NO carry-in input.)
    # For k=8, inputs=16 -> 65k rows. Getting heavy but okay.
    
    def make_adder_goal(k: int):
        def adder_fn(bits):
            # A is bits[0:k], B is bits[k:2k]
            a_val = 0
            b_val = 0
            for i in range(k):
                if bits[i]: a_val += (1 << i)
                if bits[k + i]: b_val += (1 << i)
            
            total = a_val + b_val
            
            # Outputs: S0..Sk-1, Cout
            out = []
            for i in range(k):
                out.append((total >> i) & 1)
            out.append((total >> k) & 1)
            return tuple(out)
        
        return Goal(f"ADDER_{k}BIT", BoolFunction.from_lambda(2 * k, k + 1, adder_fn))

    for k in range(1, 9): # 1..8
        lib.add_goal(make_adder_goal(k))


    # ==========================
    # 5. Comparators (1 <= k <= 8)
    # ==========================
    # Inputs: 2*k. Max 16 inputs.

    # K-Bit Equality
    def make_eq_goal(k: int):
        def eq_fn(bits):
            # A: 0..k-1, B: k..2k-1
            equal = 1
            for i in range(k):
                if bits[i] != bits[k+i]:
                    equal = 0
                    break
            return (equal,)
        return Goal(f"EQUAL_{k}BIT", BoolFunction.from_lambda(2 * k, 1, eq_fn))

    # K-Bit Less Than
    def make_less_goal(k: int):
        def less_fn(bits):
            a_val = 0
            b_val = 0
            for i in range(k):
                if bits[i]: a_val += (1 << i)
                if bits[k + i]: b_val += (1 << i)
            return (1 if a_val < b_val else 0,)
        return Goal(f"LESS_{k}BIT", BoolFunction.from_lambda(2 * k, 1, less_fn))

    for k in range(1, 9): # 1..8
        lib.add_goal(make_eq_goal(k))
        lib.add_goal(make_less_goal(k))


# ============================================================
# 9. Main evolutionary loop
# ============================================================

def run_evolution(
    steps: int = 10000,
    max_components_per_step: int = 12,
    verbose_every: int = 1000,
    seed: Optional[int] = None,
) -> TechnologyLibrary:
    if seed is not None:
        random.seed(seed)

    lib = TechnologyLibrary()
    make_nand_primitive(lib)
    make_constants(lib)
    make_basic_goals(lib)

    # Metrics tracking
    history_steps = []
    goal_distances = {g.name: [] for g in lib.goals}
    
    # We want to track usage of technologies. 
    # Store a list of dictionaries: [{tech_name: count, ...}, ...]
    tech_usage_history = [] 

    for step in range(1, steps + 1):
        # pick between 2 and max_components_per_step components
        n_components = random.randint(2, max_components_per_step)
        
        components = []
        for _ in range(n_components):
            components.append(select_component(lib))

        # build wiring and composite function
        wiring = build_composite_wiring(components, prob_new_input=0.2)

        # to keep things computationally manageable, skip circuits with too many inputs
        # Determine number of global inputs from wiring
        max_in = -1
        for cmap in wiring.component_input_maps:
            for signal in cmap:
                if isinstance(signal, int):
                    if signal > max_in:
                        max_in = signal
        n_global_inputs_guess = max_in + 1
        
        if n_global_inputs_guess > 18:  # limit (tune as you like)
            continue

        new_fn = realize_composite_function(components, wiring)

        # cost is sum of costs of components + 1 (for this encapsulation)
        new_cost = sum(c.cost for c in components) + 1
        new_name = lib.new_name()

        new_comp = Component(
            name=new_name,
            function=new_fn,
            cost=new_cost,
            is_primitive=False,
        )

        improved_goals = evaluate_and_maybe_add(lib, new_comp)
        if improved_goals:
             goal_names = ", ".join(g.name for g in improved_goals)
             print(f"Step {step}: Discovered new technology: {new_comp.name} (Cost: {new_comp.cost}) for goals: {goal_names}")

        # Record metrics periodically
        if verbose_every is not None and step % verbose_every == 0:
            history_steps.append(step)
            
            # 1. Goal distances
            for goal in lib.goals:
                if goal.best_components:
                    # min distance
                    d = min(goal.target.distance_to(c.function) for c in goal.best_components)
                    goal_distances[goal.name].append(d)
                else:
                    # If no approximation, use max possible distance (roughly 2^inputs * outputs)
                    # or just None/Nan. Let's use None for plotting gaps.
                    goal_distances[goal.name].append(None)

            # 2. Tech usage (fine-grained)
            # Count how many times each specific technology appears in the "best_components" of all goals.
            # This shows which techs are "keystones".
            current_usage = {}
            for goal in lib.goals:
                for c in goal.best_components:
                    # We only care about non-primitives? Or all? 
                    # User asked for "all technology". Let's track everything including primitives.
                    name = c.name
                    current_usage[name] = current_usage.get(name, 0) + 1
            
            tech_usage_history.append(current_usage)

            print(f"Step {step}: techs={len(lib.technologies)}, improved={bool(improved_goals)}")
            # Optional: print details... (omitted to save screen space if requested)

    # Plotting
    plot_fitness(history_steps, goal_distances)
    plot_usage(history_steps, tech_usage_history)

    return lib


def plot_fitness(steps, goal_data):
    plt.figure(figsize=(12, 8))
    # Plot a subset of interesting goals to avoid clutter
    # Or plot all with thin lines
    for name, dists in goal_data.items():
        # Clean None values for plotting
        clean_steps = []
        clean_dists = []
        for s, d in zip(steps, dists):
            if d is not None:
                clean_steps.append(s)
                clean_dists.append(d)
        
        if clean_steps:
            plt.plot(clean_steps, clean_dists, label=name)
    
    plt.xlabel("Step")
    plt.ylabel("Hamming Distance (lower is better)")
    plt.title("Goal Approximation Progress")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.yscale('symlog') # Log scale helps see progress near zero
    plt.tight_layout()
    plt.savefig("fitness_progress.png")
    print("Saved fitness_progress.png")


def plot_usage(steps, usage_history):
    # usage_history is list of dicts: [{tech_A: 1, tech_B: 2}, ...]
    
    # 1. Identify all unique techs that ever appeared
    all_techs = set()
    for usage in usage_history:
        all_techs.update(usage.keys())
    
    # Sort them (primitives first, then by ID)
    def sort_key(name):
        if name in ["NAND", "CONST_0", "CONST_1"]:
            return (0, name)
        # assuming tech_ID
        try:
            return (1, int(name.split('_')[1]))
        except:
            return (2, name)
            
    sorted_techs = sorted(list(all_techs), key=sort_key)
    
    # 2. Build time series for each
    series = {t: [] for t in sorted_techs}
    for usage in usage_history:
        for t in sorted_techs:
            series[t].append(usage.get(t, 0))
            
    # 3. Plot
    plt.figure(figsize=(14, 8))
    
    # Stacked area chart might be messy with many techs. 
    # A heatmap or just lines for the top N might be better.
    # Let's try lines but only for techs that have significant usage (> 0 at some point)
    # If there are too many, we might filter.
    
    # Filter: only plot techs that ever reached a usage count > 0 (already done by set collection)
    # Maybe only top 20 most used?
    
    # Calculate max usage for each tech
    max_usages = [(t, max(series[t])) for t in sorted_techs]
    max_usages.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top 20 most influential + primitives
    top_n = 20
    top_techs = [x[0] for x in max_usages[:top_n]]
    
    for t in top_techs:
        plt.plot(steps, series[t], label=t, linewidth=1.5)
        
    plt.xlabel("Step")
    plt.ylabel("Usage Count (in current best solutions)")
    plt.title(f"Technology Usage Over Time (Top {top_n} shown)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tech_usage.png")
    print("Saved tech_usage.png")


if __name__ == "__main__":
    print("Running evolution...")
    lib = run_evolution(steps=500, verbose_every=100, seed=42)
    print("\nFinal best for each goal:")
    for goal in lib.goals:
        if not goal.best_components:
            print(f"Goal {goal.name}: no solution")
        else:
            best = min(goal.best_components, key=lambda c: goal.target.distance_to(c.function))
            d = goal.target.distance_to(best.function)
            print(f"Goal {goal.name}: best={best.name}, dist={d}, cost={best.cost}")
