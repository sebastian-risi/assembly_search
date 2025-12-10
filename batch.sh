#!/bin/bash
#SBATCH --job-name=assembly-search
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu          # Use CPU partition
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16       # Allocate 16 CPUs per task for parallel evaluation
#SBATCH --ntasks-per-node=1      # 1 process per task
#SBATCH --array=0-23             # 4 targets * 3 pop_size * 2 speciation = 24 combinations
#SBATCH --output=outputs/%x-%A_%a.out
#SBATCH --error=outputs/%x-%A_%a.err

# No GPU modules needed for CPU-only execution

# -----------------------
# Hyperparameter grids
# -----------------------
TARGET_LIST=("NOT AND_2 IMPLY" "XOR_2 OR_2 AND_2" "XOR_3 OR_3 AND_3" "ADDER_1BIT ADDER_2BIT")
POP_SIZE_LIST=(100 200 300)
SPECIATION_THRESHOLD_LIST=(0.5 0.7)

NUM_TARGETS=${#TARGET_LIST[@]}
NUM_POP=${#POP_SIZE_LIST[@]}
NUM_SPEC=${#SPECIATION_THRESHOLD_LIST[@]}

TOTAL_COMBOS=$(( NUM_TARGETS * NUM_POP * NUM_SPEC ))

# Safety check so --array matches total combos
if [ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBOS ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) >= TOTAL_COMBOS ($TOTAL_COMBOS)"
    exit 1
fi

# -----------------------
# Map SLURM_ARRAY_TASK_ID -> (target_idx, pop_idx, spec_idx)
# Fastest varying: speciation_threshold, then pop_size, then targets
# -----------------------
TASK_ID=$SLURM_ARRAY_TASK_ID

SPEC_INDEX=$(( TASK_ID % NUM_SPEC ))
TMP=$(( TASK_ID / NUM_SPEC ))

POP_INDEX=$(( TMP % NUM_POP ))
TARGET_INDEX=$(( TMP / NUM_POP ))

TARGETS=${TARGET_LIST[$TARGET_INDEX]}
POP_SIZE=${POP_SIZE_LIST[$POP_INDEX]}
SPECIATION_THRESHOLD=${SPECIATION_THRESHOLD_LIST[$SPEC_INDEX]}

# Generate unique seed per job
SEED=$(( 42 + SLURM_ARRAY_TASK_ID ))

# -----------------------
# Per-run output dir
# -----------------------
# Create a safe directory name from targets (replace spaces with underscores)
TARGETS_SAFE=$(echo "$TARGETS" | tr ' ' '_')
SAVE_DIR="outputs/assembly_search/targets_${TARGETS_SAFE}_pop${POP_SIZE}_spec${SPECIATION_THRESHOLD}_seed${SEED}"
mkdir -p "$SAVE_DIR"

echo "============================================"
echo "TOTAL_COMBOS           = $TOTAL_COMBOS"
echo "SLURM_ARRAY_TASK_ID    = $SLURM_ARRAY_TASK_ID"
echo "TARGET_INDEX           = $TARGET_INDEX -> targets=$TARGETS"
echo "POP_INDEX              = $POP_INDEX -> pop_size=$POP_SIZE"
echo "SPEC_INDEX             = $SPEC_INDEX -> speciation_threshold=$SPECIATION_THRESHOLD"
echo "seed                   = $SEED"
echo "cpus_per_task          = $SLURM_CPUS_PER_TASK"
echo "save_dir               = $SAVE_DIR"
echo "============================================"

# -----------------------
# Run experiment
# -----------------------
cd "$SAVE_DIR" || exit 1

python3 ../../tech_evolution_neat.py \
  --targets $TARGETS \
  --generations 500 \
  --pop-size "$POP_SIZE" \
  --seed "$SEED" \
  --speciation-threshold "$SPECIATION_THRESHOLD" \
  --num-workers -1 \
  > run.log 2>&1

echo "Job completed. Results saved to $SAVE_DIR"

