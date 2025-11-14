#!/bin/bash
#SBATCH --array=0-9
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='final'
DATA_PATH='data/dataset'
EVAL_CONFIG_PATH='data/eval_configs/cfg.pkl'

HIDDEN_DIMS=(512 1024 2048 1024 512)

DROPOUT=0.0
ACTIVATION='ReLU'
WEIGHT_DECAY=0.001
NB_EPOCHS=7
CHECK_VAL_EVERY_N_EPOCH=1
NUM_WORKERS=10
MIN_EPOCHS=0
PATIENCE=0
LR=0.0001
BATCH_SIZE=32
SEED=$SLURM_ARRAY_TASK_ID

srun python3 -u -m src.algorithms.regression.mlp_reg \
    "$EXPERIMENT_NAME" \
    "$DATA_PATH" \
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
    --hidden-dims "${HIDDEN_DIMS[@]}" \
    --use-local-features \
    --eval-test \
    --seed "$SEED"
