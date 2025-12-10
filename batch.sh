#!/bin/bash
#SBATCH --job-name=assembly-search
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu          # Use CPU partition
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4        # Allocate 4 CPUs per task for parallel evaluation
#SBATCH --ntasks-per-node=1      # 1 process per task
#SBATCH --array=0-3              # 8 runs: 2 target sets × 2 generation counts × 2 init modes
#SBATCH --output=outputs/%x-%A_%a.out
#SBATCH --error=outputs/%x-%A_%a.err

# No GPU modules needed for CPU-only execution

# -----------------------
# Wandb authentication
# -----------------------
# Set your wandb API key here (required for non-interactive SLURM jobs)
# Get your key from: https://wandb.ai/authorize
export WANDB_API_KEY="71c7aa7f441dd287514bd9fb25066ef1034d8f80"

# Alternatively, if you don't want to use wandb, remove --use-wandb flag below

# -----------------------
# Hyperparameter grids
# -----------------------
TARGET_LIST=(
    # Run 1: All circuits (comprehensive test)
    "NOT IMPLY AND_2 OR_2 XOR_2 AND_3 OR_3 XOR_3 AND_4 OR_4 XOR_4 AND_5 OR_5 XOR_5 AND_6 OR_6 XOR_6 AND_7 OR_7 XOR_7 AND_8 OR_8 XOR_8 BITWISE_AND_2 BITWISE_OR_2 BITWISE_XOR_2 BITWISE_AND_3 BITWISE_OR_3 BITWISE_XOR_3 BITWISE_AND_4 BITWISE_OR_4 BITWISE_XOR_4 BITWISE_AND_5 BITWISE_OR_5 BITWISE_XOR_5 BITWISE_AND_6 BITWISE_OR_6 BITWISE_XOR_6 BITWISE_AND_7 BITWISE_OR_7 BITWISE_XOR_7 FULL_ADDER ADDER_1BIT ADDER_2BIT ADDER_3BIT ADDER_4BIT ADDER_5BIT ADDER_6BIT ADDER_7BIT ADDER_8BIT EQUAL_1BIT LESS_1BIT EQUAL_2BIT LESS_2BIT EQUAL_3BIT LESS_3BIT EQUAL_4BIT LESS_4BIT EQUAL_5BIT LESS_5BIT EQUAL_6BIT LESS_6BIT EQUAL_7BIT LESS_7BIT EQUAL_8BIT LESS_8BIT"
    
    # Run 2: Minimal path to 8-bit adder (building blocks only)
    #"NOT AND_2 OR_2 XOR_2 FULL_ADDER ADDER_1BIT ADDER_2BIT ADDER_3BIT ADDER_4BIT ADDER_5BIT ADDER_6BIT ADDER_7BIT ADDER_8BIT"
)
POP_SIZE_LIST=(250)
SPECIATION_THRESHOLD_LIST=(0.7)
GENERATIONS_LIST=(2000 500)
INIT_WITH_LIBRARY_LIST=(0 1)  # 0 = minimal init, 1 = library init

NUM_TARGETS=${#TARGET_LIST[@]}
NUM_POP=${#POP_SIZE_LIST[@]}
NUM_SPEC=${#SPECIATION_THRESHOLD_LIST[@]}
NUM_GENS=${#GENERATIONS_LIST[@]}
NUM_INIT=${#INIT_WITH_LIBRARY_LIST[@]}

TOTAL_COMBOS=$(( NUM_TARGETS * NUM_POP * NUM_SPEC * NUM_GENS * NUM_INIT ))

# Safety check so --array matches total combos
if [ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBOS ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) >= TOTAL_COMBOS ($TOTAL_COMBOS)"
    exit 1
fi

# -----------------------
# Map SLURM_ARRAY_TASK_ID -> (target_idx, pop_idx, spec_idx, gen_idx, init_idx)
# Fastest varying: init_mode, then generations, then speciation_threshold, then pop_size, then targets
# -----------------------
TASK_ID=$SLURM_ARRAY_TASK_ID

INIT_INDEX=$(( TASK_ID % NUM_INIT ))
TMP=$(( TASK_ID / NUM_INIT ))

GEN_INDEX=$(( TMP % NUM_GENS ))
TMP=$(( TMP / NUM_GENS ))

SPEC_INDEX=$(( TMP % NUM_SPEC ))
TMP=$(( TMP / NUM_SPEC ))

POP_INDEX=$(( TMP % NUM_POP ))
TARGET_INDEX=$(( TMP / NUM_POP ))

TARGETS=${TARGET_LIST[$TARGET_INDEX]}
POP_SIZE=${POP_SIZE_LIST[$POP_INDEX]}
SPECIATION_THRESHOLD=${SPECIATION_THRESHOLD_LIST[$SPEC_INDEX]}
GENERATIONS=${GENERATIONS_LIST[$GEN_INDEX]}
INIT_WITH_LIBRARY=${INIT_WITH_LIBRARY_LIST[$INIT_INDEX]}

# Generate unique seed per job
SEED=$(( 42 + SLURM_ARRAY_TASK_ID ))

# -----------------------
# Per-run output dir
# -----------------------
# Determine run type for cleaner directory names
if [ $TARGET_INDEX -eq 0 ]; then
    RUN_NAME="all_circuits"
else
    RUN_NAME="adder8bit_path"
fi

# Add init mode to directory name
if [ $INIT_WITH_LIBRARY -eq 1 ]; then
    INIT_SUFFIX="libinit"
else
    INIT_SUFFIX="minimal"
fi

SAVE_DIR="outputs/assembly_search/${RUN_NAME}_${INIT_SUFFIX}_gen${GENERATIONS}_pop${POP_SIZE}_spec${SPECIATION_THRESHOLD}_seed${SEED}"
mkdir -p "$SAVE_DIR"

echo "============================================"
echo "TOTAL_COMBOS           = $TOTAL_COMBOS"
echo "SLURM_ARRAY_TASK_ID    = $SLURM_ARRAY_TASK_ID"
echo "RUN_NAME               = $RUN_NAME"
echo "TARGET_INDEX           = $TARGET_INDEX"
echo "GEN_INDEX              = $GEN_INDEX"
echo "INIT_INDEX             = $INIT_INDEX"
echo "GENERATIONS            = $GENERATIONS"
echo "POP_SIZE               = $POP_SIZE"
echo "SPECIATION_THRESHOLD   = $SPECIATION_THRESHOLD"
echo "INIT_WITH_LIBRARY      = $INIT_WITH_LIBRARY ($INIT_SUFFIX)"
echo "seed                   = $SEED"
echo "cpus_per_task          = $SLURM_CPUS_PER_TASK"
echo "save_dir               = $SAVE_DIR"
echo "Num targets            = $(echo $TARGETS | wc -w)"
echo "============================================"

# -----------------------
# Run experiment
# -----------------------
# Store script directory for absolute path
SCRIPT_DIR="/home/sebastianrisi_sakana_ai/code/assembly_search"

cd "$SAVE_DIR" || exit 1

# Build command with optional --init-with-library flag
CMD="python3 ${SCRIPT_DIR}/tech_evolution_neat.py \
  --targets $TARGETS \
  --generations $GENERATIONS \
  --pop-size $POP_SIZE \
  --seed $SEED \
  --speciation-threshold $SPECIATION_THRESHOLD \
  --num-workers -1 \
  --use-wandb"

if [ $INIT_WITH_LIBRARY -eq 1 ]; then
    CMD="$CMD --init-with-library"
fi

# Run the command
eval "$CMD" > run.log 2>&1

echo "Job completed. Results saved to $SAVE_DIR"

