# Technology Evolution Simulator (Arthur & Polak–style)

This repository contains a toy implementation of the evolutionary technology model described by W. Brian Arthur and Wolfgang Polak in “The Evolution of Technology within a Simple Computer Model”.

Technologies are modeled as logic circuits. Starting from a primitive gate (NAND) and constants (0, 1), the system:

- randomly combines existing components into new circuits,
- tests them against a set of goals (desired logical functions),
- encapsulates useful circuits as new technologies,
- and reuses them as building blocks for more complex circuits.

Over time, the system can discover non-trivial circuits like XOR, adders, and comparators, mimicking the “technology builds on technology” dynamic described in the paper.

---

## Repository structure

- `tech_evolution.py`

Contains:
- Boolean function representation
- Components / technologies
- Goals (needs)
- Technology library
- Random circuit combination (N-way wiring)
- Evolutionary search loop

---

## High-level design

### 1. Boolean functions

Represented by the `BoolFunction` class:

- `n_inputs`  
- `n_outputs`  
- `truth_table` (a list of output tuples for every input combination)

Methods:
- `distance_to(other)`
- `is_equal(other)`
- `from_lambda(...)`

A `BoolFunction` is the functional “phenotype” of a circuit.

---

### 2. Components (technologies)

Encapsulated logic blocks represented by the `Component` class:

- `name`
- `function` (a BoolFunction)
- `cost`
- `is_primitive`

Components can be primitives (like NAND) or discovered technologies created during evolution.

---

### 3. Goals (needs)

Each goal defines a desired logical function. The simulation includes a wide range of goals:

- **Basic**: NOT, IMPLY
- **N-way Gates (2-8 inputs)**: XOR, OR, AND
- **Bitwise Operations (2-7 bits)**: XOR, OR, AND
- **Arithmetic**: Full Adder, K-bit Adders (1-8 bits)
- **Comparators**: Equality, Less Than (1-8 bits)

Every goal tracks its current best approximations (by Hamming distance and cost).

---

### 4. Technology library

`TechnologyLibrary` stores:

- primitives
- constants
- discovered technologies
- goals

As evolution progresses, this library grows and becomes the building-block space for new circuits.

---

### 5. Random circuit combination

In each step, the system selects between **2 and 12 components** randomly from the library (primitives, constants, or existing technologies).

It generates a random **acyclic wiring**:
- The first component always accepts global inputs.
- Subsequent components can accept global inputs OR outputs from previous components in the chain.
- The system randomly selects a subset of *all* available outputs to expose as the new technology's interface.

This produces a new `BoolFunction` for the composite circuit. Large circuits (lots of global inputs) are skipped to keep truth tables manageable.

---

### 6. Evaluating new circuits

A new composite circuit is compared against each goal:

- If it is a better approximation (lower distance to target), or
- If it is equal functionally but cheaper

then it is encapsulated and added as a new technology.

This acts as the algorithm’s selection mechanism.

---

### 7. Selection probabilities

Approximate probabilities:

- 50% primitives
- 1.5% constants
- 48.5% discovered technologies

This biases the search toward reusing existing useful components.

---

### 8. Evolution loop

`run_evolution()`:

1. select $N$ components (random $N \in [2, 12]$)
2. randomly wire them together
3. build composite truth table
4. compute cost
5. evaluate against goals
6. possibly add to technologies

Progress is printed periodically, and final best technologies per goal are displayed.

---

## Running

Requirements:
- Python 3.8+
- no external dependencies

Run:

```bash
python tech_evolution.py
```

Example output:

```yaml
Step 500: techs=58, improved=False
  Goal NOT: best dist=1 #best=1
  Goal IMPLY: best dist=1 #best=1
  Goal XOR_2: best dist=1 #best=1
  ...
  Goal FULL_ADDER: best dist=6 #best=1
```

Numbers vary with randomness.

## Configuration

Edit the bottom of `tech_evolution.py`:

```python
lib = run_evolution(
    steps=500000,
    verbose_every=500,
    seed=42,
)
```

Adjust:

- number of steps
- verbosity
- input limit (currently ~18)
- goals

## Differences from the original model

Simplified elements:

- uses truth tables instead of BDDs,
- caps input size,
- doesn’t track avalanches or criticality,
- minimal cost heuristic.

Enhancements:
- Supports combining 2-12 components at once (closer to the original paper's "vector" combinations).
- Expanded goal set including N-way gates and multi-bit arithmetic.

## Reference

W. Brian Arthur & Wolfgang Polak  
“The Evolution of Technology within a Simple Computer Model”
