# Hybrid Neuroevolution for Technology Discovery

This repository includes a hybrid evolutionary system (`tech_evolution_neat.py`) that combines the "combinatorial technology evolution" ideas of Arthur & Polak with **NeuroEvolution of Augmenting Topologies (NEAT)**.

Instead of randomly wiring components together, this system evolves the wiring diagram itself as a genome, allowing for incremental optimization of complex circuits.

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
    *   Directed wires connecting a *Source* (Global Input or Component Output) to a *Destination* (Global Output or Component Input).
    *   Connections can be enabled/disabled, allowing evolution to "turn off" pathways.

### 2. Evolutionary Operators
The system uses NEAT-like mutations to complexify the circuit:
*   **Add Connection**: Adds a wire between two previously unconnected points.
*   **Add Component (Node)**: Inserts a new component (e.g., a NAND gate) into an existing wire, effectively splitting it.
*   **Remove Connection**: Disables a wire.

### 3. Hierarchical Library
Just like the original model, this system maintains a **Technology Library**.
*   When the evolutionary process solves a specific Goal (e.g., "Full Adder"), the successful circuit is **encapsulated**.
*   This new technology becomes a reusable building block.
*   Future generations can add this new component (e.g., `FULL_ADDER_solved`) as a single node, enabling the evolution of even more complex systems (e.g., 4-bit Adders) without needing to re-evolve the adder logic from scratch.

---

## Evolution Loop

The system iterates through a set of defined **Goals** (Logic Gates, Adders, Comparators). For each goal:

1.  **Initialize**: Create a population of empty or sparse genomes.
2.  **Evaluate**:
    *   Simulate the circuit for all input patterns (truth table).
    *   Calculate **Fitness** based on similarity to the Goal (Hamming distance).
    *   *(Planned)* Penalize unconnected inputs to ensure valid circuits.
3.  **Reproduce**:
    *   Select the fittest genomes.
    *   Apply mutations (add wires, add components).
4.  **Success**:
    *   If a genome matches the goal perfectly, it is saved to the Library.
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

### debugging
If you see fitness scores like `63/64` (almost perfect), it often indicates a **floating input** issue. A circuit with one unconnected input (defaulting to 0) can be correct for all cases except the one where that input *must* be 1. The code includes debug prints to track unconnected pins.

---

## Comparison with Standard NEAT
*   **Nodes are Complex**: Standard NEAT nodes are simple neurons (sum & sigmoid). Here, nodes are logic gates or complex circuits with their own truth tables.
*   **Input/Output Pins**: Components have specific named input/output pins, not just a single "activation" value.
*   **Strictly Feed-Forward**: The current implementation enforces acyclic graphs for boolean logic stability.

## References
*   Stanley, K. O., & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies" (NEAT).
*   Arthur, W. B., & Polak, W. (2006). "The Evolution of Technology within a Simple Computer Model".

