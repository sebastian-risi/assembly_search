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
        
        # Check if exists
        exists = False
        for conn in genome.connections:
            # Parse conn structure mapping
            # src is (type, id), dst is (type, id)
            # conn.src_id for comp_out is tuple, etc.
            
            c_src = (conn.src_type, conn.src_id)
            c_dst = (conn.dst_type, conn.dst_id)
            
            if c_src == src and c_dst == dst:
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
    
    # NEAT style: Split an existing connection
    if not genome.connections: return
    
    conn = random.choice(genome.connections)
    if not conn.enabled: return
    
    #print name of conn
    conn.enabled = False
    
    # Select component from library to insert
    # Standard: Primitives or discovered techs
    # Use selection probabilities
    comp_template = copy.deepcopy(select_component_wrapper(lib))
    
    new_inst_id = max([i.id for i in genome.instances] + [0]) + 1
    new_inst = ComponentInstance(new_inst_id, comp_template)
    genome.instances.append(new_inst)
    #print name of selected tech    print(comp_template.name)
    #print(f"Selected tech: {comp_template.name}")

    # Wire: Src -> NewComp -> Dst
    # Connect original Src to first input of NewComp
    # Connect first output of NewComp to original Dst
    # (Simplified assumption: component has at least 1 in/1 out. Most do.)
    
    # 1. Src -> NewIn[0]
    if comp_template.function.n_inputs > 0:
        c1 = ConnectionGene(
            src_type=conn.src_type, src_id=conn.src_id,
            dst_type='comp_in', dst_id=(new_inst_id, 0),
            enabled=True
        )
        genome.connections.append(c1)
        
    # 2. NewOut[0] -> Dst
    if comp_template.function.n_outputs > 0:
        c2 = ConnectionGene(
            src_type='comp_out', src_id=(new_inst_id, 0),
            dst_type=conn.dst_type, dst_id=conn.dst_id,
            enabled=True
        )
        genome.connections.append(c2)

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
    solved = (fitness == max_dist and unconnected_count == 0)  # Perfect match AND valid
    
    return (genome.id, fitness, unconnected_count, phenotype, solved)


# ============================================================
# 6. Genome Visualization
# ============================================================

def visualize_genome(genome: Genome, filename: str, title: str = "Genome"):
    """
    Visualize a genome as a directed graph showing components and connections.
    Saves to filename.
    """
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
    
    # Layout
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
    num_workers: int = 1
):
    if seed is not None:
        random.seed(seed)

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
            
            # Initialize with random connections to ensure connectivity
            # Add a few random connections (e.g. 1-3)
            for _ in range(random.randint(1, 5)):
                mutate_add_connection(g)
            
            # Maybe add a component or two?
            if random.random() < 0.5:
                mutate_add_component(g, lib)
                
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
                # Parallel evaluation
                tasks = [(g, goal) for g in population]
                results_map = {}
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(evaluate_genome_parallel, task): task[0].id for task in tasks}
                    for future in as_completed(futures):
                        genome_id, fitness, unconnected_count, phenotype, is_solved = future.result()
                        results_map[genome_id] = (fitness, unconnected_count, phenotype, is_solved)
                
                # Apply results to genomes
                for g in population:
                    fitness, unconnected_count, phenotype, is_solved = results_map[g.id]
                    g.fitness = fitness
                    
                    if is_solved:
                        solved = True
                        # Encapsulate immediately
                        cost = sum(inst.component.cost for inst in g.instances) + 1
                        new_comp = Component(
                            name=lib.new_name(f"{goal.name}_solved"),
                            function=phenotype,
                            cost=cost
                        )
                        from tech_evolution import evaluate_and_maybe_add
                        evaluate_and_maybe_add(lib, new_comp)
                        print(f"  SOLVED {goal.name} at Gen {generation}! Added {new_comp.name}")
                        
                        # Visualize the solution
                        viz_filename = f"genome_{goal.name}_solved.png"
                        visualize_genome(g, viz_filename, title=f"Solution for {goal.name} (Gen {generation})")
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

                    if fitness == max_dist:# and unconnected_count == 0: # Perfect match AND valid
                        solved = True
                        # Encapsulate immediately
                        cost = sum(inst.component.cost for inst in g.instances) + 1 # Simple cost
                        new_comp = Component(
                            name=lib.new_name(f"{goal.name}_solved"),
                            function=phenotype,
                            cost=cost
                        )
                        # Add to library if it's new/better
                        from tech_evolution import evaluate_and_maybe_add
                        evaluate_and_maybe_add(lib, new_comp)
                        print(f"  SOLVED {goal.name} at Gen {generation}! Added {new_comp.name}")
                        
                        # Visualize the solution
                        viz_filename = f"genome_{goal.name}_solved.png"
                        visualize_genome(g, viz_filename, title=f"Solution for {goal.name} (Gen {generation})")
                        
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
                print(f"  Gen {generation}: Best Fitness {best_fitness}/{max_dist} (Unconnected: {best_unconnected}, Species: {len(species_list)})")

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
                        if random.random() < 0.8: mutate_add_connection(child)
                        if random.random() < 0.3: mutate_add_component(child, lib)
                        if random.random() < 0.1: mutate_remove_connection(child)
                        
                        next_pop.append(child)
            
            # If next_pop is too small (can happen if species get very small), fill with random
            while len(next_pop) < pop_size:
                if population:
                    parent = random.choice(population)
                    child = parent.clone()
                    child.id = len(next_pop) + generation * 10000
                    if random.random() < 0.8: mutate_add_connection(child)
                    if random.random() < 0.3: mutate_add_component(child, lib)
                    next_pop.append(child)
                else:
                    break
            
            # Trim if too large
            next_pop = next_pop[:pop_size]
            
            population = next_pop

    print("\nEvolution finished.")
    print(f"Total technologies in library: {len(lib.technologies)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid NEAT evolution for technology discovery.")
    parser.add_argument("--targets", nargs='+', help="Specific goals to target (e.g. NOT AND_2 FULL_ADDER). If omitted, targets all.")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations to evolve towards each target.")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--speciation-threshold", type=float, default=0.7, help="Compatibility distance threshold for speciation (higher = fewer species).")
    parser.add_argument("--distance-weight-connections", type=float, default=0.3, help="Weight for connection topology in distance calculation.")
    parser.add_argument("--distance-weight-components", type=float, default=1.0, help="Weight for component types in distance calculation.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers for genome evaluation (default: 1 = serial, use -1 for CPU count).")
    
    args = parser.parse_args()
    
    # Handle num_workers = -1 (auto-detect CPU count)
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()
    
    run_hybrid_evolution(
        pop_size=args.pop_size,
        seed=args.seed,
        target_goals=args.targets,
        generations_per_goal=args.generations,
        speciation_threshold=args.speciation_threshold,
        distance_weight_connections=args.distance_weight_connections,
        distance_weight_components=args.distance_weight_components,
        num_workers=num_workers
    )

