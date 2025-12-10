import random
import itertools
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Union
import copy
import matplotlib.pyplot as plt
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Reuse core definitions from the original file for compatibility
from tech_evolution import BoolFunction, Component, Goal, TechnologyLibrary, make_nand_primitive, make_constants, make_basic_goals

# ============================================================
# 1. Genome / Circuit Representation
# ============================================================

@dataclass
class Node:
    id: int
    type: str  # 'input', 'output', 'component'
    component_ref: Optional[Component] = None  # If type is 'component'
    # For a component node, we need to handle its internal inputs/outputs
    # But standard NEAT nodes are usually simple summing units.
    # Here, a "Node" in the graph is actually a *port* (pin).
    
    # Let's refine:
    # The graph consists of:
    # 1. Global Input Nodes (Sources)
    # 2. Component Instances (Black boxes with Input Ports and Output Ports)
    # 3. Global Output Nodes (Sinks)
    #
    # Wires connect: Source -> Destination
    # Source can be: Global Input OR Component Output Port
    # Destination can be: Global Output OR Component Input Port
    pass

@dataclass
class ConnectionGene:
    src_type: str # 'global_in' or 'comp_out'
    src_id: int   # if global_in: index. if comp_out: (comp_instance_id, output_pin_index)
    
    dst_type: str # 'global_out' or 'comp_in'
    dst_id: int   # if global_out: index. if comp_in: (comp_instance_id, input_pin_index)
    
    enabled: bool = True
    innovation_num: int = 0

@dataclass
class ComponentInstance:
    id: int
    component: Component

@dataclass
class Genome:
    id: int
    # Inputs/Outputs defining the interface of this circuit being evolved
    n_global_inputs: int
    n_global_outputs: int  # Dynamic? Or fixed per goal? 
    # For open-ended evolution, we might let the genome define how many outputs it exposes,
    # or we try to fit it to specific goals. 
    # Let's start with flexible outputs: The genome exposes N outputs.
    
    instances: List[ComponentInstance] = field(default_factory=list)
    connections: List[ConnectionGene] = field(default_factory=list)
    
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    species_id: int = 0

    def clone(self):
        new_g = Genome(self.id, self.n_global_inputs, self.n_global_outputs)
        new_g.instances = [copy.deepcopy(i) for i in self.instances]
        new_g.connections = [copy.deepcopy(c) for c in self.connections]
        new_g.fitness = self.fitness
        return new_g

@dataclass
class Species:
    id: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    age: int = 0
    stagnation: int = 0
    max_fitness: float = 0.0

# ============================================================
# 2. Genome Evaluator (Phenotype)
# ============================================================

def evaluate_genome(genome: Genome) -> BoolFunction:
    """
    Simulates the circuit defined by the genome to produce a BoolFunction.
    Handles potential cycles by aborting or returning 0.
    """
    # 1. Topological sort or simulation
    # Since we allow arbitrary wiring, we must check for cycles.
    # We can model this as a DAG of values.
    
    # Map: (type, id) -> value
    # value is a truth table column (tuple of bits for all input patterns)
    
    n_inputs = genome.n_global_inputs
    input_patterns = list(itertools.product([0, 1], repeat=n_inputs))
    n_patterns = len(input_patterns)
    
    # Debug safety
    if n_patterns == 0 or len(input_patterns) != 2**n_inputs:
         print(f"DEBUG EVAL: n_inputs={n_inputs}, n_patterns={n_patterns}")
    
    # Store calculated values for each pin to avoid re-calc
    # Key: ('global_in', idx) or ('comp_out', instance_id, pin_idx)
    # Value: tuple of bits (length n_patterns)
    memo: Dict[Tuple, Tuple[int, ...]] = {}

    # Pre-fill global inputs
    for i in range(n_inputs):
        # column i of inputs
        col = tuple(p[i] for p in input_patterns)
        memo[('global_in', i)] = col

    # Helper to resolve a specific signal source
    # Returns None if cycle detected (or depth limit exceeded)
    recursion_stack = set()

    def resolve_signal(key) -> Optional[Tuple[int, ...]]:
        if key in memo:
            return memo[key]
        if key in recursion_stack:
            return None # Cycle!
        
        recursion_stack.add(key)
        
        # Who drives this key?
        # We need a reverse map of connections to find who drives what.
        # But wait, 'key' here is a SOURCE (an output pin or global input).
        # Global inputs are already in memo.
        # So 'key' must be ('comp_out', inst_id, out_idx).
        
        # To calculate a component output, we need to calculate ALL its inputs first.
        c_type, inst_id, out_idx = key
        assert c_type == 'comp_out'
        
        instance = next(inst for inst in genome.instances if inst.id == inst_id)
        comp = instance.component
        
        # Gather inputs for this component instance
        # We need to find which wires connect TO ('comp_in', inst_id, in_idx)
        # This search is slow if not indexed. For prototype, iterate connections.
        
        input_values = []
        for in_pin in range(comp.function.n_inputs):
            # Find wire driving this pin
            driver_key = None
            for conn in genome.connections:
                if not conn.enabled: continue
                # dst is ('comp_in', inst_id, in_pin)
                # stored in gene as src_type, src_id, dst_type, dst_id
                # dst_id for comp_in is (inst_id, in_pin) (conceptually)
                
                # Let's fix ConnectionGene representation to be clearer:
                # dst_id is just a tuple? data classes with tuples can be messy in some parsers.
                # Let's assume we handle the gene parsing here.
                
                is_target = (conn.dst_type == 'comp_in' and conn.dst_id == (inst_id, in_pin))
                if is_target:
                    # Found driver
                    if conn.src_type == 'global_in':
                        driver_key = ('global_in', conn.src_id)
                    else:
                        driver_key = ('comp_out', conn.src_id[0], conn.src_id[1])
                    break
            
            if driver_key:
                val = resolve_signal(driver_key)
                if val is None: return None # Propagate error
                input_values.append(val)
            else:
                # Floating input - default to 0s
                input_values.append((0,) * n_patterns)

        # Now we have columns for all inputs. 
        # Evaluate component function for all rows.
        # Transpose input_values [pin][row] -> [row][pin]
        input_rows = zip(*input_values)
        
        # Calculate outputs
        # This is the heavy part: looking up truth table
        comp_output_cols = [[] for _ in range(comp.function.n_outputs)]
        
        for row_bits in input_rows:
            # bit list -> index
            idx = 0
            for b in row_bits:
                idx = (idx << 1) | b
            
            # Simple lookup, assuming n_inputs match
            # If not, let it crash or return garbage (we handle crashes by not selecting bad genomes via fitness?)
            # Actually, let's just use modulo if needed or pad?
            # Safer: if idx out of range, default to 0s (broken component)
            if idx < len(comp.function.truth_table):
                out_bits = comp.function.truth_table[idx]
            else:
                out_bits = (0,) * comp.function.n_outputs

            for o_i, bit in enumerate(out_bits):
                comp_output_cols[o_i].append(bit)
        
        # Store ALL output pins of this component in memo
        for o_i, col in enumerate(comp_output_cols):
            memo[('comp_out', inst_id, o_i)] = tuple(col)
        
        recursion_stack.remove(key)
        return memo[key]

    # Evaluate all global outputs
    global_output_values = []
    valid = True
    
    for i in range(genome.n_global_outputs):
        # Find driver for ('global_out', i)
        driver_key = None
        for conn in genome.connections:
            if not conn.enabled: continue
            if conn.dst_type == 'global_out' and conn.dst_id == i:
                if conn.src_type == 'global_in':
                    driver_key = ('global_in', conn.src_id)
                else:
                    driver_key = ('comp_out', conn.src_id[0], conn.src_id[1])
                break
        
        if driver_key:
            val = resolve_signal(driver_key)
            if val is None:
                valid = False
                break
            global_output_values.append(val)
        else:
            # Floating output
            global_output_values.append((0,) * n_patterns)

    if not valid:
        # Cycle or error
        return BoolFunction(n_inputs, genome.n_global_outputs, [(0,)*genome.n_global_outputs] * n_patterns)

    # Transpose back to truth table format
    final_tt = list(zip(*global_output_values))
    
    # SAFETY CHECK: If final_tt is empty, fill it with zeros
    # This happens if global_output_values is empty (no outputs) or rows are empty?
    if len(final_tt) != n_patterns:
        # This implies global_output_values has wrong length columns?
        # But we filled defaults.
        # Unless global_output_values is empty list (n_global_outputs=0)?
        pass
        
    return BoolFunction(n_inputs, genome.n_global_outputs, final_tt)


# ============================================================
# 3. Mutation Operators
# ============================================================

def mutate_add_connection(genome: Genome, prob: float = 0.5):
    #return
    if random.random() > prob: return

    # Pick random source and destination that are not already connected
    # Sources: Global Inputs, Component Outputs
    # Dests: Component Inputs, Global Outputs
    
    sources = []
    for i in range(genome.n_global_inputs):
        sources.append(('global_in', i))
    for inst in genome.instances:
        for o_i in range(inst.component.function.n_outputs):
            sources.append(('comp_out', (inst.id, o_i)))
            
    dests = []
    for i in range(genome.n_global_outputs):
        dests.append(('global_out', i))
    for inst in genome.instances:
        for i_i in range(inst.component.function.n_inputs):
            dests.append(('comp_in', (inst.id, i_i)))
            
    if not sources or not dests: return

    # Try N times to find unconnected pair
    for _ in range(5):
        src = random.choice(sources)
        dst = random.choice(dests)
        
        # Check if exists or if destination already has a driver
        exists = False
        for conn in genome.connections:
            if not conn.enabled:
                continue
            
            # Parse conn structure mapping
            # src is (type, id), dst is (type, id)
            # conn.src_id for comp_out is tuple, etc.
            
            c_src = (conn.src_type, conn.src_id)
            c_dst = (conn.dst_type, conn.dst_id)
            
            # Check if exact same connection already exists
            if c_src == src and c_dst == dst:
                exists = True
                break
            
            # Check if destination already has a driver (prevent multiple drivers)
            if c_dst == dst:
                exists = True
                break
        
        # Also simple cycle check: don't connect a component to itself?
        # resolve_signal handles cycles, but we can avoid obvious ones.
        if src[0] == 'comp_out' and dst[0] == 'comp_in':
            if src[1][0] == dst[1][0]: # Same instance ID
                exists = True
        
        if not exists:
            # Add
            new_conn = ConnectionGene(
                src_type=src[0], src_id=src[1],
                dst_type=dst[0], dst_id=dst[1],
                enabled=True
            )
            genome.connections.append(new_conn)
            break

def mutate_add_component(genome: Genome, lib: TechnologyLibrary, prob: float = 0.1):
    if random.random() > prob: return
    
    if not genome.connections: return
    
    # Get a valid enabled connection to split
    valid_conns = [c for c in genome.connections if c.enabled]
    if not valid_conns: return
    conn = random.choice(valid_conns)
    
    conn.enabled = False
    
    # Select component
    comp_template = copy.deepcopy(select_component_wrapper(lib))
    new_inst_id = max([i.id for i in genome.instances] + [0]) + 1
    new_inst = ComponentInstance(new_inst_id, comp_template)
    genome.instances.append(new_inst)

    # Wire only one random input and one random output of the new component
    # Choose random input pin
    random_in_pin = random.randint(0, comp_template.function.n_inputs - 1) if comp_template.function.n_inputs > 0 else 0
    
    # Connect source -> new_comp.in[random_in_pin]
    c_in = ConnectionGene(
        src_type=conn.src_type, 
        src_id=conn.src_id,
        dst_type='comp_in', 
        dst_id=(new_inst_id, random_in_pin),
        enabled=True
    )
    genome.connections.append(c_in)
    
    # Choose random output pin
    random_out_pin = random.randint(0, comp_template.function.n_outputs - 1) if comp_template.function.n_outputs > 0 else 0
    
    # Connect new_comp.out[random_out_pin] -> original destination
    c_out = ConnectionGene(
        src_type='comp_out', 
        src_id=(new_inst_id, random_out_pin),
        dst_type=conn.dst_type, 
        dst_id=conn.dst_id,
        enabled=True
    )
    genome.connections.append(c_out)

def select_component_wrapper(lib: TechnologyLibrary) -> Component:
    # Wrapper around the original random selection
    from tech_evolution import select_component
    return select_component(lib)

def mutate_remove_connection(genome: Genome, prob: float = 0.1):
    if random.random() > prob: return
    if not genome.connections: return
    # Just disable
    c = random.choice(genome.connections)
    c.enabled = False


# ============================================================
# 4. Speciation Logic
# ============================================================

def calculate_genome_distance(g1: Genome, g2: Genome, c_conn: float = 1.0, c_comp: float = 1.0) -> float:
    """
    NEAT compatibility distance.
    Combines connection topology difference and component type difference.
    
    d = c_conn * (disjoint_connections / N_conn) + c_comp * (disjoint_components / N_comp)
    """
    # 1. Connection topology distance
    def get_conn_keys(g):
        keys = set()
        for c in g.connections:
            # key = (src_type, src_id, dst_type, dst_id)
            keys.add((c.src_type, c.src_id, c.dst_type, c.dst_id))
        return keys

    k1 = get_conn_keys(g1)
    k2 = get_conn_keys(g2)
    
    disjoint_conn = k1.symmetric_difference(k2)
    N_conn = max(len(k1), len(k2), 1)
    dist_conn = len(disjoint_conn) / N_conn
    
    # 2. Component type distance
    # Compare the multiset of component names (types)
    def get_comp_types(g):
        # Use a sorted list (could use Counter but list is simpler for now)
        return sorted([inst.component.name for inst in g.instances])
    
    types1 = get_comp_types(g1)
    types2 = get_comp_types(g2)
    
    # Calculate symmetric difference in component types
    # Convert to multisets (Counter-like)
    from collections import Counter
    counter1 = Counter(types1)
    counter2 = Counter(types2)
    
    # Disjoint elements: sum of absolute differences
    all_types = set(counter1.keys()).union(set(counter2.keys()))
    diff_count = sum(abs(counter1.get(t, 0) - counter2.get(t, 0)) for t in all_types)
    
    N_comp = max(len(types1), len(types2), 1)
    dist_comp = diff_count / N_comp
    
    # Combined distance
    return c_conn * dist_conn + c_comp * dist_comp


def speciation(population: List[Genome], species_list: List[Species], threshold: float = 0.5, c_conn: float = 1.0, c_comp: float = 1.0) -> List[Species]:
    # Clear members
    for s in species_list:
        s.members = []
        s.max_fitness = 0.0 # Reset max fitness for this gen tracking
        
    # Place genomes
    for g in population:
        placed = False
        for s in species_list:
            d = calculate_genome_distance(g, s.representative, c_conn=c_conn, c_comp=c_comp)
            
            if d < threshold:
                s.members.append(g)
                g.species_id = s.id
                placed = True
                break
        
        if not placed:
            # New species
            new_id = len(species_list) + 1
            new_s = Species(id=new_id, representative=g, members=[g])
            g.species_id = new_id
            species_list.append(new_s)
            
    # Remove empty species
    species_list = [s for s in species_list if s.members]
    
    # Pick new representatives
    for s in species_list:
        if s.members:
            # Usually pick random or best? Standard NEAT: random from previous gen, or best from this gen?
            # Or just keep old rep if it's still close?
            # Simple: pick random member as new rep for next gen
            s.representative = random.choice(s.members)
            
    return species_list


def calculate_adjusted_fitness(species_list: List[Species]):
    # Adjusted fitness = fitness / size_of_species
    for s in species_list:
        size = len(s.members)
        if size == 0: continue
        for g in s.members:
            g.adjusted_fitness = g.fitness / size


# ============================================================
# 5. Parallel Evaluation Helper
# ============================================================

def evaluate_genome_parallel(genome_goal_tuple):
    """
    Top-level function for parallel evaluation.
    Takes (genome, goal) tuple and returns (genome_id, fitness, unconnected_count, phenotype, solved).
    """
    genome, goal = genome_goal_tuple
    
    phenotype = evaluate_genome(genome)
    dist = goal.target.distance_to(phenotype)
    
    max_dist = (2**goal.target.n_inputs) * goal.target.n_outputs
    fitness = max_dist - dist
    
    # Count unconnected pins
    unconnected_count = 0
    # Check global outputs
    for i in range(genome.n_global_outputs):
        connected = False
        for c in genome.connections:
            if c.enabled and c.dst_type == 'global_out' and c.dst_id == i:
                connected = True
                break
        if not connected: unconnected_count += 1
    
    # Check component inputs
    for inst in genome.instances:
        for pin in range(inst.component.function.n_inputs):
            connected = False
            for c in genome.connections:
                if c.enabled and c.dst_type == 'comp_in' and c.dst_id == (inst.id, pin):
                    connected = True
                    break
            if not connected: unconnected_count += 1
    
    fitness = max(0.0, fitness)
    solved = (fitness == max_dist)# and unconnected_count == 0)  # Perfect match AND valid
    
    return (genome.id, fitness, unconnected_count, phenotype, solved)


def evaluate_batch_parallel(batch_data):
    """
    Evaluate a batch of genomes in parallel to reduce pickling overhead.
    batch_data: (genome_list, goal)
    Returns: list of (genome_id, fitness, unconnected_count, phenotype, solved) tuples
    """
    genome_list, goal = batch_data
    results = []
    for genome in genome_list:
        result = evaluate_genome_parallel((genome, goal))
        results.append(result)
    return results


# ============================================================
# 6. Genome Visualization
# ============================================================

# Global dictionary to store genomes for discovered technologies
# Maps component name -> Genome that created it
_discovered_genomes: Dict[str, Genome] = {}

# Global list to track technology discoveries over time
# Each entry: (generation, goal_name, distance_from_target, cost, is_solved)
_technology_timeline: List[Tuple[int, str, int, int, bool]] = []


def visualize_technology_timeline(filename: str = "technology_development_timeline.png", title: str = "Technology Development Over Time"):
    """
    Create a comprehensive visualization showing when all technologies were discovered/improved.
    """
    if not _technology_timeline:
        print("No technology discoveries to visualize.")
        return
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Group discoveries by goal
    goal_discoveries = {}
    for gen, goal_name, dist, cost, is_solved in _technology_timeline:
        if goal_name not in goal_discoveries:
            goal_discoveries[goal_name] = []
        goal_discoveries[goal_name].append((gen, dist, cost, is_solved))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # === Subplot 1: Timeline scatter plot ===
    goal_names = sorted(goal_discoveries.keys())
    colors = plt.cm.tab20(range(len(goal_names)))
    goal_colors = {name: colors[i] for i, name in enumerate(goal_names)}
    
    # Plot discoveries for each goal
    for goal_idx, goal_name in enumerate(goal_names):
        discoveries = goal_discoveries[goal_name]
        
        # Separate solved from approximations
        solved_gens = [gen for gen, dist, cost, solved in discoveries if solved]
        approx_gens = [gen for gen, dist, cost, solved in discoveries if not solved]
        
        # Plot approximations as small circles
        if approx_gens:
            ax1.scatter(approx_gens, [goal_idx] * len(approx_gens), 
                       c=[goal_colors[goal_name]], alpha=0.4, s=50, marker='o')
        
        # Plot solutions as stars
        if solved_gens:
            ax1.scatter(solved_gens, [goal_idx] * len(solved_gens), 
                       c=[goal_colors[goal_name]], alpha=1.0, s=200, marker='*', 
                       edgecolors='black', linewidths=1.5)
    
    ax1.set_yticks(range(len(goal_names)))
    ax1.set_yticklabels(goal_names, fontsize=9)
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Technology Goal', fontsize=12, fontweight='bold')
    ax1.set_title('Technology Discovery Timeline', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Legend
    approx_patch = mpatches.Patch(color='gray', alpha=0.4, label='Approximation')
    solved_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                              markersize=15, markeredgecolor='black', markeredgewidth=1.5, 
                              label='Exact Solution')
    ax1.legend(handles=[approx_patch, solved_marker], loc='upper left', fontsize=10)
    
    # === Subplot 2: Cumulative solutions over time ===
    # Get all generations where solutions were found
    solution_events = [(gen, goal_name) for gen, goal_name, dist, cost, solved in _technology_timeline if solved]
    solution_events.sort()
    
    if solution_events:
        generations = []
        cumulative_count = []
        count = 0
        
        for gen, goal_name in solution_events:
            generations.append(gen)
            count += 1
            cumulative_count.append(count)
        
        ax2.plot(generations, cumulative_count, linewidth=3, color='#2E86AB', marker='o', 
                markersize=8, markerfacecolor='#A23B72', markeredgecolor='black', markeredgewidth=1)
        ax2.fill_between(generations, cumulative_count, alpha=0.2, color='#2E86AB')
        
        # Annotate final count
        if cumulative_count:
            ax2.text(generations[-1], cumulative_count[-1], f'  {cumulative_count[-1]} solved', 
                    fontsize=11, fontweight='bold', va='center')
    
    ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Solutions', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Technology Solutions', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved technology timeline visualization: {filename}")
    print(f"Total discoveries logged: {len(_technology_timeline)}")
    print(f"Total goals with discoveries: {len(goal_discoveries)}")
    print(f"Total exact solutions: {sum(1 for _, _, _, _, solved in _technology_timeline if solved)}")


def check_and_add_improvement(genome: Genome, phenotype: BoolFunction, goal: Goal, 
                               lib: TechnologyLibrary, generation: int) -> Optional[Component]:
    """
    Check if this genome improves upon existing best approximations for the goal.
    Returns the new component if it was an improvement, None otherwise.
    
    A circuit improves if:
    1. It better matches the truth table (fewer errors), OR
    2. It has identical function but lower cost
    """
    dist = phenotype.distance_to(goal.target)
    cost = sum(inst.component.cost for inst in genome.instances) + 1
    
    # Check against existing best approximations
    improved = False
    replaced_component = None
    
    if not goal.best_components:
        # No existing approximations, this is the first
        improved = True
    else:
        for existing_comp in goal.best_components:
            existing_dist = existing_comp.function.distance_to(goal.target)
            
            # Better match to truth table
            if dist < existing_dist:
                improved = True
                replaced_component = existing_comp
                break
            # Same function but lower cost
            elif dist == existing_dist and phenotype.is_equal(existing_comp.function) and cost < existing_comp.cost:
                improved = True
                replaced_component = existing_comp
                break
    
    if improved:
        # Create new component
        if dist == 0:
            # Exact match - use solved naming
            comp_name = lib.new_name(f"{goal.name}_solved")
            print(f"  SOLVED {goal.name} at Gen {generation}! (dist={dist}, cost={cost})")
        else:
            # Approximation - use tech naming
            comp_name = lib.new_name(f"{goal.name}_approx")
            print(f"  IMPROVED {goal.name} at Gen {generation}! (dist={dist}, cost={cost})")
        
        new_comp = Component(
            name=comp_name,
            function=phenotype,
            cost=cost,
            is_primitive=False
        )
        
        # Add to library
        from tech_evolution import evaluate_and_maybe_add
        evaluate_and_maybe_add(lib, new_comp)
        
        # Store the genome
        _discovered_genomes[new_comp.name] = copy.deepcopy(genome)
        
        # Log to timeline
        is_solved = (dist == 0)
        _technology_timeline.append((generation, goal.name, dist, cost, is_solved))
        
        # Update goal's best components
        if replaced_component:
            # Remove the old component from goal's best list
            goal.best_components = [c for c in goal.best_components if c.name != replaced_component.name]
            
            # Also remove from the global library to prevent it from being used in new circuits
            lib.technologies = [c for c in lib.technologies if c.name != replaced_component.name]
            
            # Remove from discovered genomes as well
            if replaced_component.name in _discovered_genomes:
                del _discovered_genomes[replaced_component.name]
            
            print(f"    Replaced {replaced_component.name} (dist={existing_dist}, cost={replaced_component.cost})")
        
        # Add new component
        goal.best_components.append(new_comp)
        
        return new_comp
    
    return None


def visualize_genome_detailed(genome: Genome, filename: str, title: str = "Genome", 
                                discovered_genomes: Optional[Dict[str, Genome]] = None,
                                visited: Optional[set] = None, prefix: str = ""):
    """
    Detailed visualization that recursively expands discovered technologies to show their internal structure.
    """
    if discovered_genomes is None:
        discovered_genomes = _discovered_genomes
    if visited is None:
        visited = set()
    
    G = nx.DiGraph()
    node_positions = {}
    y_level = 0
    
    # Add global inputs at the top
    for i in range(genome.n_global_inputs):
        node_id = f"{prefix}IN_{i}"
        G.add_node(node_id, node_type='input', label=f"In{i}", level=0)
        node_positions[node_id] = (i * 2, y_level)
    
    y_level = -1
    
    # Track which component instances have been expanded (so we can skip their original connections)
    expanded_instances = set()
    
    # Process component instances
    for inst in genome.instances:
        comp_name = inst.component.name
        
        # Check if this is a discovered technology that we should expand
        if discovered_genomes and comp_name in discovered_genomes and comp_name not in visited:
            expanded_instances.add(inst.id)
            # Recursively expand this component
            visited.add(comp_name)
            sub_genome = discovered_genomes[comp_name]
            
            # Create a subgraph for this component
            sub_prefix = f"{prefix}C{inst.id}_"
            sub_G, sub_positions, sub_height = visualize_genome_detailed(
                sub_genome, "", "", discovered_genomes, visited.copy(), sub_prefix
            )
            
            # Add subgraph nodes to main graph
            for node, data in sub_G.nodes(data=True):
                G.add_node(node, **data)
                # Offset positions to place this subgraph
                if node in sub_positions:
                    x, y = sub_positions[node]
                    node_positions[node] = (x + inst.id * 10, y + y_level)
            
            # Add subgraph edges
            for u, v, data in sub_G.edges(data=True):
                G.add_edge(u, v, **data)
            
            # Connect subgraph inputs to parent inputs
            # Map component instance input pins to sub-genome global inputs
            for in_pin in range(inst.component.function.n_inputs):
                # Find what drives this input in the parent genome
                for conn in genome.connections:
                    if not conn.enabled:
                        continue
                    if conn.dst_type == 'comp_in' and conn.dst_id == (inst.id, in_pin):
                        # This input is driven by conn.src
                        if conn.src_type == 'global_in':
                            src_node = f"{prefix}IN_{conn.src_id}"
                        else:
                            src_inst_id, src_pin = conn.src_id
                            src_node = f"{prefix}C{src_inst_id}_out{src_pin}"
                        
                        # Connect to subgraph's global input (mapped from component input pin)
                        # The sub-genome's global input index should match the component's input pin index
                        dst_node = f"{sub_prefix}IN_{in_pin}"
                        if src_node in G.nodes() and dst_node in G.nodes():
                            G.add_edge(src_node, dst_node, edge_type='connection', color='blue', width=3)
                        break
            
            # Connect subgraph outputs to parent outputs
            # Map component instance output pins to sub-genome global outputs
            for out_pin in range(inst.component.function.n_outputs):
                src_node = f"{sub_prefix}OUT_{out_pin}"
                # Find what this output drives in the parent genome
                for conn in genome.connections:
                    if not conn.enabled:
                        continue
                    if conn.src_type == 'comp_out' and conn.src_id == (inst.id, out_pin):
                        if conn.dst_type == 'global_out':
                            dst_node = f"{prefix}OUT_{conn.dst_id}"
                        else:
                            dst_inst_id, dst_pin = conn.dst_id
                            dst_node = f"{prefix}C{dst_inst_id}_in{dst_pin}"
                        
                        if src_node in G.nodes() and dst_node in G.nodes():
                            G.add_edge(src_node, dst_node, edge_type='connection', color='red', width=3)
                        break
            
            y_level -= sub_height - 1
        else:
            # Regular component - show as before
            for pin in range(inst.component.function.n_inputs):
                node_id = f"{prefix}C{inst.id}_in{pin}"
                G.add_node(node_id, node_type='comp_input', label=f"{comp_name}\nIn{pin}", comp_id=inst.id, level=1)
                node_positions[node_id] = (inst.id * 2 + pin * 0.3, y_level)
            
            for pin in range(inst.component.function.n_outputs):
                node_id = f"{prefix}C{inst.id}_out{pin}"
                G.add_node(node_id, node_type='comp_output', label=f"{comp_name}\nOut{pin}", comp_id=inst.id, level=1)
                node_positions[node_id] = (inst.id * 2 + pin * 0.3, y_level - 0.5)
            
            # Internal edges
            for in_pin in range(inst.component.function.n_inputs):
                for out_pin in range(inst.component.function.n_outputs):
                    G.add_edge(f"{prefix}C{inst.id}_in{in_pin}", f"{prefix}C{inst.id}_out{out_pin}", 
                              edge_type='internal', style='dashed')
    
    # Add global outputs at the bottom
    y_level -= 1
    for i in range(genome.n_global_outputs):
        node_id = f"{prefix}OUT_{i}"
        G.add_node(node_id, node_type='output', label=f"Out{i}", level=2)
        node_positions[node_id] = (i * 2, y_level)
    
    # Add connections (but skip connections involving expanded components, as they're already handled)
    for conn in genome.connections:
        if not conn.enabled:
            continue
        
        # Skip if this connection involves an expanded component instance
        if conn.src_type == 'comp_out':
            src_inst_id, _ = conn.src_id
            if src_inst_id in expanded_instances:
                continue  # Skip - already handled by sub-network expansion
        
        if conn.dst_type == 'comp_in':
            dst_inst_id, _ = conn.dst_id
            if dst_inst_id in expanded_instances:
                continue  # Skip - already handled by sub-network expansion
        
        if conn.src_type == 'global_in':
            src_node = f"{prefix}IN_{conn.src_id}"
        else:
            inst_id, pin = conn.src_id
            src_node = f"{prefix}C{inst_id}_out{pin}"
        
        if conn.dst_type == 'global_out':
            dst_node = f"{prefix}OUT_{conn.dst_id}"
        else:
            inst_id, pin = conn.dst_id
            dst_node = f"{prefix}C{inst_id}_in{pin}"
        
        if src_node in G.nodes() and dst_node in G.nodes():
            G.add_edge(src_node, dst_node, edge_type='connection')
    
    if filename:
        # Use hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            try:
                # Fallback to hierarchical positioning
                pos = {}
                for node, (x, y) in node_positions.items():
                    pos[node] = (x, -y)  # Flip y for top-to-bottom
            except:
                pos = nx.spring_layout(G, k=2, iterations=100)
        
        # Draw
        plt.figure(figsize=(20, 14))
        
        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'unknown')
            if node_type == 'input':
                node_colors.append('#90EE90')  # Light green
            elif node_type == 'output':
                node_colors.append('#FFB6C1')  # Light pink
            elif node_type == 'comp_input':
                node_colors.append('#87CEEB')  # Sky blue
            elif node_type == 'comp_output':
                node_colors.append('#FFD700')  # Gold
            else:
                node_colors.append('#D3D3D3')  # Light gray
        
        # Draw nodes with borders
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, 
                               edgecolors='black', linewidths=2, alpha=0.9)
        
        # Draw edges
        internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'internal']
        
        # Separate colored edges (for sub-graph connections)
        # Get all connection edges with their data
        all_connection_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'connection']
        blue_edges = [(u, v) for u, v, d in all_connection_edges if d.get('color') == 'blue']
        red_edges = [(u, v) for u, v, d in all_connection_edges if d.get('color') == 'red']
        black_edges = [(u, v) for u, v, d in all_connection_edges if d.get('color') != 'blue' and d.get('color') != 'red']
        
        nx.draw_networkx_edges(G, pos, edgelist=internal_edges, edge_color='gray', 
                              style='dashed', arrows=True, arrowsize=15, width=1.5, alpha=0.4)
        if blue_edges:
            nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color='blue', 
                                  arrows=True, arrowsize=20, arrowstyle='->', width=3, alpha=0.9)
        if red_edges:
            nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', 
                                  arrows=True, arrowsize=20, arrowstyle='->', width=3, alpha=0.9)
        if black_edges:
            nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color='black', 
                                  arrows=True, arrowsize=20, arrowstyle='->', width=2.5, alpha=0.8)
        
        # Labels
        labels = {n: G.nodes[n].get('label', n.split('_')[-1]) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        plt.title(title, fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved detailed genome visualization: {filename}")
    
    return G, node_positions, abs(y_level) + 1


def visualize_genome(genome: Genome, filename: str, title: str = "Genome", detailed: bool = False, 
                     discovered_genomes: Optional[Dict[str, Genome]] = None):
    """
    Visualize a genome as a directed graph showing components and connections.
    If detailed=True, recursively expands discovered technologies to show their internal structure.
    """
    if detailed:
        visualize_genome_detailed(genome, filename, title, discovered_genomes)
        return
    
    G = nx.DiGraph()
    
    # Add nodes
    # Global inputs
    for i in range(genome.n_global_inputs):
        G.add_node(f"IN_{i}", node_type='input', label=f"In{i}")
    
    # Component instances
    for inst in genome.instances:
        comp_name = inst.component.name
        # Add input pins as sub-nodes
        for pin in range(inst.component.function.n_inputs):
            node_id = f"C{inst.id}_in{pin}"
            G.add_node(node_id, node_type='comp_input', label=f"{comp_name}\nIn{pin}", comp_id=inst.id)
        
        # Add output pins
        for pin in range(inst.component.function.n_outputs):
            node_id = f"C{inst.id}_out{pin}"
            G.add_node(node_id, node_type='comp_output', label=f"{comp_name}\nOut{pin}", comp_id=inst.id)
        
        # Add internal edges showing the component connects its inputs to outputs
        # This visually groups the pins as belonging to the same component
        for in_pin in range(inst.component.function.n_inputs):
            for out_pin in range(inst.component.function.n_outputs):
                G.add_edge(f"C{inst.id}_in{in_pin}", f"C{inst.id}_out{out_pin}", 
                          edge_type='internal', style='dashed')
    
    # Global outputs
    for i in range(genome.n_global_outputs):
        G.add_node(f"OUT_{i}", node_type='output', label=f"Out{i}")
    
    # Add edges (connections)
    for conn in genome.connections:
        if not conn.enabled:
            continue
            
        # Source
        if conn.src_type == 'global_in':
            src_node = f"IN_{conn.src_id}"
        else: # 'comp_out'
            inst_id, pin = conn.src_id
            src_node = f"C{inst_id}_out{pin}"
        
        # Destination
        if conn.dst_type == 'global_out':
            dst_node = f"OUT_{conn.dst_id}"
        else: # 'comp_in'
            inst_id, pin = conn.dst_id
            dst_node = f"C{inst_id}_in{pin}"
        
        G.add_edge(src_node, dst_node, edge_type='connection')
    
    # Layout - use Kamada-Kawai for better visualization
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        try:
            pos = nx.spring_layout(G, k=1.5, iterations=50)
        except:
            pos = nx.random_layout(G)
    
    # Draw
    plt.figure(figsize=(12, 8))
    
    # Color nodes by type
    node_colors = []
    for node in G.nodes():
        if node.startswith('IN_'):
            node_colors.append('lightblue')
        elif node.startswith('OUT_'):
            node_colors.append('lightcoral')
        elif '_in' in node:
            node_colors.append('lightyellow')
        else:
            node_colors.append('lightgreen')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    
    # Draw edges with different styles
    # Internal (component) edges: dashed
    internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'internal']
    connection_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'connection']
    
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, edge_color='gray', 
                          style='dashed', arrows=True, arrowsize=15, width=2, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=connection_edges, edge_color='black', 
                          arrows=True, arrowsize=20, arrowstyle='->', width=2)
    
    # Labels
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Saved genome visualization: {filename}")


import argparse

# ============================================================
# 6. Main Evolutionary Loop (Hybrid)
# ============================================================

def run_hybrid_evolution(
    steps: int = 1000,
    pop_size: int = 100,
    verbose_every: int = 10,
    seed: Optional[int] = None,
    target_goals: Optional[List[str]] = None,
    generations_per_goal: int = 50,
    speciation_threshold: float = 0.6,
    distance_weight_connections: float = 1.0,
    distance_weight_components: float = 1.0,
    num_workers: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "assembly_search",
    init_with_library: bool = False,
    prob_add_connection: float = 0.8,
    prob_add_component: float = 0.2,
    prob_remove_connection: float = 0.1,
    use_detailed_viz: bool = False
):
    if seed is not None:
        random.seed(seed)
    
    # Initialize wandb if requested
    if use_wandb:
        import wandb
        # Generate a descriptive run name
        if target_goals and len(target_goals) <= 3:
            goal_names = "-".join(target_goals)
        elif target_goals:
            goal_names = f"{len(target_goals)}_goals"
        else:
            goal_names = "all_goals"
        
        init_suffix = "libinit" if init_with_library else "minimal"
        run_name = f"{goal_names}_{init_suffix}_gen{generations_per_goal}_pop{pop_size}_spec{speciation_threshold}_seed{seed if seed else 'random'}"
        
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "generations_per_goal": generations_per_goal,
                "pop_size": pop_size,
                "speciation_threshold": speciation_threshold,
                "distance_weight_connections": distance_weight_connections,
                "distance_weight_components": distance_weight_components,
                "num_workers": num_workers,
                "seed": seed,
                "target_goals": target_goals if target_goals else "all",
                "init_with_library": init_with_library,
                "prob_add_connection": prob_add_connection,
                "prob_add_component": prob_add_component,
                "prob_remove_connection": prob_remove_connection
            }
        )

    lib = TechnologyLibrary()
    make_nand_primitive(lib)
    make_constants(lib)
    make_basic_goals(lib)
    
    # Filter goals if targets specified
    goals_to_process = []
    if target_goals:
        # Normalize target names to uppercase for easier matching
        targets_upper = [t.upper() for t in target_goals]
        for g in lib.goals:
            if g.name.upper() in targets_upper:
                goals_to_process.append(g)
        
        if not goals_to_process:
            print(f"Warning: No goals matched {target_goals}. Available: {[g.name for g in lib.goals]}")
            return
    else:
        goals_to_process = lib.goals

    print(f"Starting Hybrid NEAT Evolution for {len(goals_to_process)} goals...")
    
    for goal in goals_to_process:
        print(f"\nTargeting Goal: {goal.name} (In: {goal.target.n_inputs}, Out: {goal.target.n_outputs})")
        
        # Create population for this goal
        population = []
        for i in range(pop_size):
            g = Genome(id=i, n_global_inputs=goal.target.n_inputs, n_global_outputs=goal.target.n_outputs)
            
            if init_with_library:
                # Add 2-12 random components from the library
                num_components = random.randint(2, 12)
                for comp_idx in range(num_components):
                    comp_template = copy.deepcopy(select_component_wrapper(lib))
                    new_inst = ComponentInstance(comp_idx, comp_template)
                    g.instances.append(new_inst)
                
                # Wire them up randomly with multiple connections
                # Factor in number of input/output pins for better connectivity
                total_input_pins = goal.target.n_inputs  # Global inputs
                total_output_pins = goal.target.n_outputs  # Global outputs
                
                for inst in g.instances:
                    total_input_pins += inst.component.function.n_inputs
                    total_output_pins += inst.component.function.n_outputs
                
                # Calculate total possible connection points
                # Sources: global inputs + component outputs
                # Destinations: global outputs + component inputs
                total_sources = goal.target.n_inputs + sum(inst.component.function.n_outputs for inst in g.instances)
                total_destinations = goal.target.n_outputs + sum(inst.component.function.n_inputs for inst in g.instances)
                
                # Use a fraction of total possible connections, with minimum based on components
                min_connections = max(num_components * 2, total_sources // 2, total_destinations // 2)
                max_connections = min(num_components * 5, int(total_sources * total_destinations * 0.3))
                max_connections = max(max_connections, min_connections)  # Ensure max >= min
                
                num_connections = random.randint(min_connections, max_connections) #was 2--5
                for _ in range(num_connections):
                    mutate_add_connection(g, prob=1.0)
            else:
                # Minimal initialization: just a few random connections
                # Add a few random connections (e.g. 1-5)
                for _ in range(random.randint(1, 5)):
                    mutate_add_connection(g, prob=1.0)
                
                # Maybe add a component or two?
                if random.random() < 0.5:
                    mutate_add_component(g, lib, prob=1.0)
                
            population.append(g)
            
        solved = False
        species_list = []  # Initialize species tracking
        max_dist = (2**goal.target.n_inputs) * goal.target.n_outputs  # Calculate once for this goal
        
        for generation in range(generations_per_goal): # Max generations per goal attempt
            # 1. Evaluate (in parallel if num_workers > 1)
            best_fitness = -1.0
            best_genome = None
            best_unconnected = 0
            
            if num_workers > 1:
                # Parallel evaluation with batching to reduce pickling overhead
                # Batching reduces the number of pickling operations and process communication overhead
                # Each batch contains multiple genomes, reducing the per-genome overhead
                batch_size = max(1, pop_size // (num_workers * 2))  # 2 batches per worker for better load balancing
                batches = []
                for i in range(0, len(population), batch_size):
                    batch = population[i:i + batch_size]
                    batches.append((batch, goal))
                
                results_map = {}
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(evaluate_batch_parallel, batch_data): batch_data[0] 
                              for batch_data in batches}
                    for future in as_completed(futures):
                        batch_results = future.result()
                        for genome_id, fitness, unconnected_count, phenotype, is_solved in batch_results:
                            results_map[genome_id] = (fitness, unconnected_count, phenotype, is_solved)
                
                # Apply results to genomes and check for improvements
                for g in population:
                    fitness, unconnected_count, phenotype, is_solved = results_map[g.id]
                    g.fitness = fitness
                    
                    # Check if this genome improves the goal (exact match or better approximation)
                    new_comp = check_and_add_improvement(g, phenotype, goal, lib, generation)
                    
                    if new_comp and is_solved:
                        solved = True
                        cost = new_comp.cost
                        
                        # Visualize the solution - always save both normal and detailed versions
                        viz_filename_normal = f"genome_{goal.name}_solved.png"
                        viz_filename_detailed = f"genome_{goal.name}_solved_detailed.png"
                        visualize_genome(g, viz_filename_normal, title=f"Solution for {goal.name} (Gen {generation})", 
                                        detailed=False, discovered_genomes=_discovered_genomes)
                        visualize_genome(g, viz_filename_detailed, title=f"Solution for {goal.name} - Detailed (Gen {generation})", 
                                        detailed=True, discovered_genomes=_discovered_genomes)
                        
                        # Log to wandb
                        if use_wandb:
                            import wandb
                            wandb.log({
                                f"{goal.name}/solved": True,
                                f"{goal.name}/solved_generation": generation,
                                f"{goal.name}/solved_cost": cost,
                                f"{goal.name}/solved_num_components": len(g.instances),
                                f"{goal.name}/solved_num_connections": sum(1 for c in g.connections if c.enabled),
                                f"{goal.name}/genome_viz": wandb.Image(viz_filename_normal),
                                f"{goal.name}/genome_viz_detailed": wandb.Image(viz_filename_detailed)
                            })
                        break
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_genome = g
                        best_unconnected = unconnected_count
                        
            else:
                # Serial evaluation (original code)
                for g in population:
                    phenotype = evaluate_genome(g)
                    dist = goal.target.distance_to(phenotype)
                    
                    fitness = max_dist - dist
                    g.fitness = fitness
                    
                    # Count unconnected inputs for this genome
                    unconnected_count = 0
                    # Check global outputs (must be driven)
                    for i in range(g.n_global_outputs):
                        connected = False
                        for c in g.connections:
                            if c.enabled and c.dst_type == 'global_out' and c.dst_id == i:
                                connected = True
                                break
                        if not connected: unconnected_count += 1
                    
                    # Check component inputs (must be driven)
                    for inst in g.instances:
                        for pin in range(inst.component.function.n_inputs):
                            connected = False
                            for c in g.connections:
                                if c.enabled and c.dst_type == 'comp_in' and c.dst_id == (inst.id, pin):
                                    connected = True
                                    break
                            if not connected: unconnected_count += 1

                    # Penalize unconnected inputs heavily
                    # If we have unconnected pins, fitness is reduced.
                    # Max penalty should ensure it's worse than a bad valid circuit?
                    # Or just subtract from the max score.
                    # Each unconnected pin is effectively a "wrong bit" for at least some cases.
                    # Let's subtract 2 points per unconnected pin to prioritize connectivity.
                    #!fitness -= (unconnected_count * 2.0)
                    
                    # Ensure fitness doesn't go below 0 (optional)
                    fitness = max(0.0, fitness)
                    g.fitness = fitness

                    # Check if this genome improves the goal (exact match or better approximation)
                    is_solved = (fitness == max_dist)
                    new_comp = check_and_add_improvement(g, phenotype, goal, lib, generation)
                    
                    if new_comp and is_solved:
                        solved = True
                        cost = new_comp.cost
                        
                        # Visualize the solution - always save both normal and detailed versions
                        viz_filename_normal = f"genome_{goal.name}_solved.png"
                        viz_filename_detailed = f"genome_{goal.name}_solved_detailed.png"
                        visualize_genome(g, viz_filename_normal, title=f"Solution for {goal.name} (Gen {generation})",
                                        detailed=False, discovered_genomes=_discovered_genomes)
                        visualize_genome(g, viz_filename_detailed, title=f"Solution for {goal.name} - Detailed (Gen {generation})",
                                        detailed=True, discovered_genomes=_discovered_genomes)
                        
                        # Log to wandb
                        if use_wandb:
                            import wandb
                            wandb.log({
                                f"{goal.name}/solved": True,
                                f"{goal.name}/solved_generation": generation,
                                f"{goal.name}/solved_cost": cost,
                                f"{goal.name}/solved_num_components": len(g.instances),
                                f"{goal.name}/solved_num_connections": sum(1 for c in g.connections if c.enabled),
                                f"{goal.name}/genome_viz": wandb.Image(viz_filename_normal),
                                f"{goal.name}/genome_viz_detailed": wandb.Image(viz_filename_detailed)
                            })
                        
                        break
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_genome = g
                        best_unconnected = unconnected_count

            if solved: break
            
            # Speciation
            species_list = speciation(population, species_list, threshold=speciation_threshold, c_conn=distance_weight_connections, c_comp=distance_weight_components)
            calculate_adjusted_fitness(species_list)
            
            if generation % 10 == 0:
                # Show best approximation info
                best_approx_str = ""
                if goal.best_components:
                    best_comp = min(goal.best_components, key=lambda c: c.function.distance_to(goal.target))
                    best_dist = best_comp.function.distance_to(goal.target)
                    best_approx_str = f", BestApprox: {best_dist}/{max_dist}"
                
                print(f"  Gen {generation}: Best Fitness {best_fitness}/{max_dist} (Unconnected: {best_unconnected}, Species: {len(species_list)}{best_approx_str})")
                
                # Log to wandb
                if use_wandb:
                    import wandb
                    log_dict = {
                        f"{goal.name}/generation": generation,
                        f"{goal.name}/best_fitness": best_fitness,
                        f"{goal.name}/max_fitness": max_dist,
                        f"{goal.name}/best_fitness_pct": (best_fitness / max_dist) * 100 if max_dist > 0 else 0,
                        f"{goal.name}/avg_fitness": sum(g.fitness for g in population) / len(population),
                        f"{goal.name}/num_species": len(species_list),
                        f"{goal.name}/best_unconnected": best_unconnected,
                        f"{goal.name}/best_num_components": len(best_genome.instances) if best_genome else 0,
                        f"{goal.name}/best_num_connections": sum(1 for c in best_genome.connections if c.enabled) if best_genome else 0,
                        "total_technologies": len(lib.technologies)
                    }
                    
                    # Add best approximation metrics
                    if goal.best_components:
                        best_comp = min(goal.best_components, key=lambda c: c.function.distance_to(goal.target))
                        best_dist = best_comp.function.distance_to(goal.target)
                        log_dict[f"{goal.name}/best_approx_dist"] = best_dist
                        log_dict[f"{goal.name}/best_approx_cost"] = best_comp.cost
                        log_dict[f"{goal.name}/num_approximations"] = len(goal.best_components)
                    
                    wandb.log(log_dict)

            # 2. Selection & Reproduction with Speciation
            # Sort by adjusted_fitness instead of raw fitness
            population.sort(key=lambda x: x.adjusted_fitness, reverse=True)
            
            # Reproduce per species
            next_pop = []
            
            for species in species_list:
                if not species.members:
                    continue
                    
                # Sort members by adjusted fitness
                species.members.sort(key=lambda x: x.adjusted_fitness, reverse=True)
                
                # How many offspring for this species?
                # Proportional to sum of adjusted fitnesses
                total_adjusted = sum(g.adjusted_fitness for g in population if g.adjusted_fitness > 0)
                if total_adjusted > 0:
                    species_adjusted = sum(g.adjusted_fitness for g in species.members)
                    offspring_count = int((species_adjusted / total_adjusted) * pop_size)
                else:
                    offspring_count = len(species.members)  # Fallback
                
                offspring_count = max(1, offspring_count)  # At least 1
                
                # Elitism: keep best from this species
                if offspring_count > 0 and species.members:
                    next_pop.append(species.members[0].clone())
                    offspring_count -= 1
                
                # Breed
                for _ in range(offspring_count):
                    if len(species.members) > 0:
                        parent = random.choice(species.members[:max(1, len(species.members)//2)])  # Select from top half
                        child = parent.clone()
                        child.id = len(next_pop) + generation * 10000  # Unique ID
                        
                        # Mutate
                        if random.random() < prob_add_connection: mutate_add_connection(child, prob=1.0)
                        if random.random() < prob_add_component: mutate_add_component(child, lib, prob=1.0)
                        if random.random() < prob_remove_connection: mutate_remove_connection(child, prob=1.0)
                        
                        next_pop.append(child)
            
            # If next_pop is too small (can happen if species get very small), fill with random
            while len(next_pop) < pop_size:
                if population:
                    parent = random.choice(population)
                    child = parent.clone()
                    child.id = len(next_pop) + generation * 10000
                    if random.random() < prob_add_connection: mutate_add_connection(child, prob=1.0)
                    if random.random() < prob_add_component: mutate_add_component(child, lib, prob=1.0)
                    next_pop.append(child)
                else:
                    break
            
            # Trim if too large
            next_pop = next_pop[:pop_size]
            
            population = next_pop
        
        # After all generations for this goal, check if solved
        if not solved:
            print(f"  Did not solve {goal.name} after {generations_per_goal} generations.")
            print(f"  Best fitness: {best_fitness}/{max_dist}")
            
            # Log to wandb
            if use_wandb:
                import wandb
                wandb.log({
                    f"{goal.name}/solved": False,
                    f"{goal.name}/final_best_fitness": best_fitness,
                    f"{goal.name}/final_max_fitness": max_dist,
                    f"{goal.name}/final_best_fitness_pct": (best_fitness / max_dist) * 100 if max_dist > 0 else 0
                })

    print("\nEvolution finished.")
    print(f"Total technologies in library: {len(lib.technologies)}")
    
    # Generate technology timeline visualization
    timeline_filename = "technology_development_timeline.png"
    visualize_technology_timeline(timeline_filename, "Assembly Search: Technology Development Timeline")
    
    # Finish wandb run
    if use_wandb:
        import wandb
        # Log the timeline visualization
        if _technology_timeline:
            wandb.log({"technology_timeline": wandb.Image(timeline_filename)})
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid NEAT evolution for technology discovery.")
    parser.add_argument("--targets", nargs='+', help="Specific goals to target (e.g. NOT AND_2 FULL_ADDER). If omitted, targets all.")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations to evolve towards each target.")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--speciation-threshold", type=float, default=0.7, help="Compatibility distance threshold for speciation (higher = fewer species).")
    parser.add_argument("--distance-weight-connections", type=float, default=0.3, help="Weight for connection topology in distance calculation.")
    parser.add_argument("--distance-weight-components", type=float, default=1.0, help="Weight for component types in distance calculation.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers for genome evaluation (default: 1 = serial, use -1 for CPU count, capped at 4).")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="assembly_search", help="Weights & Biases project name.")
    parser.add_argument("--init-with-library", action="store_true", help="Initialize genomes with 2-12 random library components wired together (default: minimal initialization).")
    parser.add_argument("--prob-add-connection", type=float, default=0.8, help="Probability of adding a connection during mutation (default: 0.8).")
    parser.add_argument("--prob-add-component", type=float, default=0.2, help="Probability of adding a component during mutation (default: 0.2).")
    parser.add_argument("--prob-remove-connection", type=float, default=0.1, help="Probability of removing a connection during mutation (default: 0.1).")
    parser.add_argument("--detailed-viz", action="store_true", help="Use detailed visualization that recursively expands discovered technologies to show their internal networks.")
    
    args = parser.parse_args()
    
    # Handle num_workers = -1 (auto-detect CPU count, capped at 4)
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = min(4, multiprocessing.cpu_count())
    
    run_hybrid_evolution(
        pop_size=args.pop_size,
        seed=args.seed,
        target_goals=args.targets,
        generations_per_goal=args.generations,
        speciation_threshold=args.speciation_threshold,
        distance_weight_connections=args.distance_weight_connections,
        distance_weight_components=args.distance_weight_components,
        num_workers=num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        init_with_library=args.init_with_library,
        prob_add_connection=args.prob_add_connection,
        prob_add_component=args.prob_add_component,
        prob_remove_connection=args.prob_remove_connection,
        use_detailed_viz=args.detailed_viz
    )
