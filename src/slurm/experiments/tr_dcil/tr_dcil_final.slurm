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
NUM_LAYERS=6
NHEAD=8
ACTIVATION='relu'

BATCH_SIZE=128
LR=0.00005
DROPOUT=0.2

SIM_BATCH_SIZE=100
BUFFER_CAP=60000
NSPE=20000

WEIGHT_DECAY=0.01
NB_EPOCHS=420
CHECK_VAL_EVERY_N_EPOCH=999
NUM_WORKERS=8
MIN_EPOCHS=999
PATIENCE=999

ALPHA=0.8
BETA=2.0
TRAJ_LEN=10

SEED=$SLURM_ARRAY_TASK_ID

srun python3 -u -m src.algorithms.dcil.transformer_dcil \
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
    "$TRAJ_LEN" \
    "$SIM_BATCH_SIZE" \
    "$BUFFER_CAP" \
    "$NSPE" \
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
    --eval-test \
    --seed "$SEED"