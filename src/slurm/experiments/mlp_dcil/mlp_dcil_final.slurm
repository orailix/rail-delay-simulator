#!/bin/bash
#SBATCH --array=0-9
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='final'
DATA_PATH='data/dataset'
ITINERARIES_PATH='data/itineraries'
EVAL_CONFIG_PATH='data/eval_configs/cfg.pkl'
DROPOUT=0.0
ACTIVATION='ReLU'

SIM_BATCH_SIZE=100
BUFFER_CAPACITY=30000
NEW_SAMPLES_PER_EPOCH=10000
NB_EPOCHS=620

LR=0.00005
BATCH_SIZE=16

WEIGHT_DECAY=0.001
CHECK_VAL_EVERY_N_EPOCH=1
NUM_WORKERS=10
MIN_EPOCHS=0
PATIENCE=0

HIDDEN_DIMS=(256 512 1024 512 256)

ALPHA=0.5
BETA=1.0
TRAJ_LEN=5

SEED=$SLURM_ARRAY_TASK_ID

# ---- launch ----
srun python3 -u -m src.algorithms.dcil.mlp_dcil \
    "$EXPERIMENT_NAME" \
    "$DATA_PATH" \
    "$ITINERARIES_PATH" \
    "$EVAL_CONFIG_PATH" \
    "$DROPOUT" \
    "$ACTIVATION" \
    "$TRAJ_LEN" \
    "$SIM_BATCH_SIZE" \
    "$BUFFER_CAPACITY" \
    "$NEW_SAMPLES_PER_EPOCH" \
    "$NB_EPOCHS" \
    "$ALPHA" \
    "$BETA" \
    "$BATCH_SIZE" \
    "$LR" \
    "$WEIGHT_DECAY" \
    "$CHECK_VAL_EVERY_N_EPOCH" \
    "$NUM_WORKERS" \
    "$MIN_EPOCHS" \
    "$PATIENCE" \
    --hidden-dims "${HIDDEN_DIMS[@]}" \
    --use-local-features \
    --eval-test \
    --seed "$SEED"