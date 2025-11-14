#!/bin/bash
#SBATCH --array=0-23
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='tuning_phase_1'
DATA_PATH='data/dataset'
ITI_PATH='data/itineraries'
EVAL_CONFIG_PATH='data/eval_configs/cfg.pkl'

HIDDEN_DIMS_ARR=(
    "64 128 256 128 64"
    "128 256 128"
    "128 256 512 256 128"
    "128 512 128"
    "128 256 512 1024 512 256 128"
    "256 512 1024 512 256"
    "256 512 1024 2048 1024 512 256"
    "512 1024 2048 1024 512"
)
LR_ARR=(0.001 0.0005 0.0001)

DROPOUT=0.0
ACTIVATION='ReLU'
BATCH_SIZE=32
WEIGHT_DECAY=0.001
NB_EPOCHS=160
CHECK_VAL_EVERY_N_EPOCH=4
NUM_WORKERS=10
MIN_EPOCHS=20
PATIENCE=20
VAL_RATIO=0.003

IDX=$SLURM_ARRAY_TASK_ID

HID_IDX=$((IDX / 3))
LR_IDX=$((IDX % 3))

HIDDEN_DIMS=${HIDDEN_DIMS_ARR[$HID_IDX]}
LR=${LR_ARR[$LR_IDX]}

srun python3 -u -m src.algorithms.bc.mlp_bc \
    "$EXPERIMENT_NAME" \
    "$DATA_PATH" \
    "$ITI_PATH" \
    "$EVAL_CONFIG_PATH" \
    "$DROPOUT" \
    "$ACTIVATION" \
    "$BATCH_SIZE" \
    "$WEIGHT_DECAY" \
    "$LR" \
    "$NB_EPOCHS" \
    "$CHECK_VAL_EVERY_N_EPOCH" \
    "$NUM_WORKERS" \
    "$MIN_EPOCHS" \
    "$PATIENCE" \
    --hidden-dims $HIDDEN_DIMS \
    --val-ratio $VAL_RATIO \
    --use-local-features