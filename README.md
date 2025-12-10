# Assembly Search: Evolutionary Technology Discovery

This repository implements **Assembly Search**, an evolutionary system (`tech_evolution_neat.py`) that combines the "combinatorial technology evolution" ideas of Arthur & Polak with speciated topology evolution.

Instead of randomly wiring components together, Assembly Search evolves the wiring diagram itself as a genome, allowing for incremental optimization of complex circuits through compositional building blocks.

---

## Core Concept

The system evolves a population of **Genomes**, where each genome represents a logic circuit.

### 1. Genome Representation
A genome consists of:
*   **Nodes**:
    *   **Global Inputs**: The external inputs to the circuit (e.g., A, B, CarryIn).
    *   **Global Outputs**: The external outputs (e.g., Sum, CarryOut).
    *   **Component Instances**: Encapsulated sub-circuits (e.g., NAND, or previously discovered tech like XOR).
*   **Connections (Genes)**:
    *   Directed wires connecting a *Source* (Global Input or Component Output Pin) to a *Destination* (Global Output or Component Input Pin).
    *   Each connection specifies exact pin addresses: `(source_type, source_id)` → `(dest_type, dest_id)`.
    *   For multi-pin components, `id` is a tuple `(component_instance_id, pin_index)`.
    *   Example: Wire output pin 1 of component 3 to input pin 0 of component 5: `('comp_out', (3,1))` → `('comp_in', (5,0))`.
    *   Connections can be enabled/disabled, allowing evolution to "turn off" pathways.

### 2. Evolutionary Operators
Assembly Search uses mutation operators to complexify the circuit:
*   **Add Connection**: Adds a wire between two previously unconnected points.
*   **Add Component (Node)**: Inserts a new component into an existing wire by:
    1. Selecting an existing connection (e.g., `InputA → OutputB`)
    2. Disabling that connection
    3. Creating a new component instance (e.g., a NAND gate or discovered FULL_ADDER)
    4. Wiring the original source to the **first input pin** of the new component
    5. Wiring the **first output pin** of the new component to the original destination
    6. **Other pins** of multi-pin components remain **unconnected** initially, allowing future mutations to wire them incrementally
*   **Remove Connection**: Disables a wire.

### 3. Speciation
The system uses speciation to maintain diversity:
*   Genomes are grouped into species based on structural similarity.
*   Fitness sharing prevents dominant simple solutions from outcompeting innovative complex ones.
*   This protects the exploration of multi-step compositional solutions.

### 4. Hierarchical Library & Module Composition
Just like the original model, Assembly Search maintains a **Technology Library**.
*   When the evolutionary process solves a specific Goal (e.g., "Full Adder"), the successful circuit is **encapsulated**.
*   This new technology becomes a reusable building block with its own input/output pins.
*   **Module Composition**: Encapsulated modules expose their original input/output pins. A `FULL_ADDER` (3 inputs, 2 outputs) can be wired like any primitive:
    *   Wire global input A to FULL_ADDER input pin 0 (bit A)
    *   Wire FULL_ADDER output pin 1 (carry) to another component's input
*   Future generations can add this new component (e.g., `FULL_ADDER_solved`) as a single node, enabling the evolution of even more complex systems (e.g., 4-bit Adders) by wiring multiple FULL_ADDER instances together, without needing to re-evolve the adder logic from scratch.

---

## Evolution Loop

Assembly Search iterates through a set of defined **Goals** (Logic Gates, Adders, Comparators). For each goal:

1.  **Initialize**: Create a population of genomes with random initial connections.
2.  **Evaluate**:
    *   Simulate the circuit for all input patterns (truth table).
    *   Calculate **Fitness** based on similarity to the Goal (Hamming distance).
    *   Report unconnected pins for diagnostics.
3.  **Speciate**:
    *   Group genomes into species based on structural similarity.
    *   Apply fitness sharing within species.
4.  **Reproduce**:
    *   Select the fittest genomes from each species.
    *   Apply mutations (add wires, add components, remove connections).
5.  **Success**:
    *   If a genome matches the goal perfectly, it is encapsulated and saved to the Library.
    *   The system moves to the next goal, now with an expanded toolkit.

---

## Usage

Run the hybrid simulation:

```bash
python tech_evolution_neat.py
```

### Command Line Options

```bash
python tech_evolution_neat.py --targets NOT AND_2 XOR_2 --generations 100 --pop-size 150
```

Available options:
- `--targets`: Space-separated list of specific goals to evolve (e.g., `NOT AND_2 FULL_ADDER`). If omitted, evolves all goals sequentially.
- `--generations`: Number of generations to spend on each goal (default: 1000).
- `--pop-size`: Population size (default: 100).
- `--seed`: Random seed for reproducibility (default: 42).
- `--speciation-threshold`: Compatibility distance threshold. Higher values = fewer, larger species (default: 0.7).
- `--distance-weight-connections`: Weight for connection topology in distance calculation (default: 0.3).
- `--distance-weight-components`: Weight for component types in distance calculation (default: 1.0).
- `--num-workers`: Number of parallel workers for genome evaluation (default: 1 = serial, use -1 for auto-detect CPU count).
- `--prob-add-connection`: Probability of adding a connection during mutation (default: 0.8).
- `--prob-add-component`: Probability of adding a component during mutation (default: 0.3).
- `--prob-remove-connection`: Probability of removing a connection during mutation (default: 0.1).

### Examples

Evolve only AND_2 with high speciation (many small species):
```bash
python tech_evolution_neat.py --targets AND_2 --speciation-threshold 0.3
```

Prioritize connection structure over component types:
```bash
python tech_evolution_neat.py --distance-weight-connections 2.0 --distance-weight-components 0.5
```

Use all CPU cores for parallel evaluation (8x+ speedup on 8-core machines):
```bash
python tech_evolution_neat.py --num-workers -1 --pop-size 200
```

Tune mutation rates for more aggressive exploration:
```bash
python tech_evolution_neat.py --prob-add-connection 0.9 --prob-add-component 0.5 --prob-remove-connection 0.2
```

---

## SLURM Batch Execution

For large-scale experiments on SLURM clusters, use the provided batch script:

```bash
sbatch batch.sh
```

The script runs multiple configurations in parallel:
- **All circuits** (60+ goals): Comprehensive exploration of all gates, bitwise ops, adders, and comparators
- **8-bit adder path**: Focused evolution from basic gates → FULL_ADDER → ADDER_8BIT
- Each configuration tests multiple generation budgets (2000 vs 5000)

The batch script automatically:
- Allocates CPU resources for parallel evaluation
- Creates unique output directories per configuration
- Manages job dependencies and queuing

**Monitor jobs:**
```bash
squeue -u $USER                    # Check job status
tail -f outputs/assembly-search-*.out  # View live progress
scancel <job_id>                   # Cancel specific job
scancel -u $USER                   # Cancel all jobs
```

Results are saved to `outputs/assembly_search/` with directories named by configuration.

---

### Debugging
If you see fitness scores like `63/64` (almost perfect), it often indicates a **floating input** issue. A circuit with one unconnected input (defaulting to 0) can be correct for all cases except the one where that input *must* be 1. The system reports unconnected pin counts to help diagnose such issues.

---

## Key Features
*   **Compositional Nodes**: Unlike simple neurons, nodes are functional building blocks (logic gates, adders, comparators) with truth tables.
*   **Multi-Pin Components**: Components have specific input/output pins, enabling precise wiring of complex circuits.
*   **Speciated Search**: Maintains diversity through species-based fitness sharing.
*   **Parallel Evaluation**: Supports multi-core processing for fast population evaluation.
*   **Feed-Forward Circuits**: Enforces acyclic graphs for stable boolean logic.

## References
*   Arthur, W. B., & Polak, W. (2006). "The Evolution of Technology within a Simple Computer Model".
*   Assembly theory and compositional complexity.

