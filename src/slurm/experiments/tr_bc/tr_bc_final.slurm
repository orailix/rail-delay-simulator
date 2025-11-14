#!/bin/bash
#SBATCH --array=0-9
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='final'
DATA_PATH='data/dataset'
ITI_PATH='data/itineraries'
EVAL_CONFIG='data/eval_configs/cfg.pkl'

D_MODEL=512
DIM_FF=2048
NUM_LAYERS=4
NHEAD=8
ACTIVATION='relu'
WEIGHT_DECAY=0.01
NB_EPOCHS=56
CHECK_VAL_EVERY_N_EPOCH=1
NUM_WORKERS=8
MIN_EPOCHS=0
PATIENCE=0
BATCH_SIZE=64
LR=0.0001
DROPOUT=0.2
SEED=$SLURM_ARRAY_TASK_ID

srun python3 -u -m src.algorithms.bc.transformer_bc \
    "$EXPERIMENT_NAME" \
    "$DATA_PATH" \
    "$ITI_PATH" \
    "$EVAL_CONFIG" \
    "$D_MODEL" \
    "$NHEAD" \
    "$DIM_FF" \
    "$DROPOUT" \
    "$ACTIVATION" \
    "$NUM_LAYERS" \
    "$BATCH_SIZE" \
    "$WEIGHT_DECAY" \
    "$LR" \
    "$NB_EPOCHS" \
    "$CHECK_VAL_EVERY_N_EPOCH" \
    "$NUM_WORKERS" \
    "$MIN_EPOCHS" \
    "$PATIENCE" \
    --eval-test \
    --seed "$SEED"
