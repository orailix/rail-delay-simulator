#!/bin/bash
#SBATCH --array=0-31
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='tuning_phase_3'
DATA_PATH='data/dataset'
ITINERARIES_PATH='data/itineraries'
EVAL_CONFIG_PATH='data/eval_configs/cfg.pkl'
DROPOUT=0.0
ACTIVATION='ReLU'

SIM_BATCH_SIZE=100
BUFFER_CAPACITY=30000
NEW_SAMPLES_PER_EPOCH=10000
NB_EPOCHS=1500

LR=0.00005
BATCH_SIZE=16

WEIGHT_DECAY=0.001
CHECK_VAL_EVERY_N_EPOCH=10
NUM_WORKERS=10
MIN_EPOCHS=500
PATIENCE=50
VAL_RATIO=0.003

HIDDEN_DIMS=(256 512 1024 512 256)

ALPHA_ARR=(0.5 0.8 0.5 0.8 0.5 0.8 0.5 0.8)
BETA_ARR=(1.0 1.0 2.0 2.0 3.0 3.0 4.0 4.0)
TRAJ_LEN_ARR=(5 10 15 20)                

IDX=${SLURM_ARRAY_TASK_ID}

NUM_PAIR=${#ALPHA_ARR[@]}
NUM_TRAJ=${#TRAJ_LEN_ARR[@]}

PAIR_IDX=$((IDX / NUM_TRAJ))
TRAJ_IDX=$((IDX % NUM_TRAJ))

ALPHA=${ALPHA_ARR[$PAIR_IDX]}
BETA=${BETA_ARR[$PAIR_IDX]}
TRAJ_LEN=${TRAJ_LEN_ARR[$TRAJ_IDX]}

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
    --val-ratio "$VAL_RATIO" \
    --use-local-features