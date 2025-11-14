#!/bin/bash
#SBATCH --array=0-53
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='tuning_phase_2'
DATA_PATH='data/dataset'
ITINERARIES_PATH='data/itineraries'
EVAL_CONFIG_PATH='data/eval_configs/cfg.pkl'

N_WORKERS=10
TRAIN_RATIO=0.1
VAL_RATIO=0.01

MAX_DEPTH=13
MIN_CHILD_WEIGHT=5.0
SUBSAMPLE=1.0
COLSAMPLE=0.8
GAMMA=1.0

LR_EST_ARR=(
    "0.03 2000"
    "0.04 1600"
    "0.06 1000"
    "0.07  800"
    "0.09  400"
    "0.1  200"
)
REG_ALPHA_ARR=(0 0.3 1)
REG_LAMBDA_ARR=(0 1 5)

IDX=$SLURM_ARRAY_TASK_ID

LAMBDA_IDX=$(( IDX % 3 ))
TMP=$(( IDX / 3 ))
ALPHA_IDX=$(( TMP % 3 ))
PAIR_IDX=$(( TMP / 3 ))

read LEARNING_RATE N_ESTIMATORS <<< "${LR_EST_ARR[$PAIR_IDX]}"
REG_ALPHA=${REG_ALPHA_ARR[$ALPHA_IDX]}
REG_LAMBDA=${REG_LAMBDA_ARR[$LAMBDA_IDX]}

srun python3 -u -m src.algorithms.bc.xgboost_bc \
    "$EXPERIMENT_NAME" \
    "$DATA_PATH" \
    "$ITINERARIES_PATH" \
    "$EVAL_CONFIG_PATH" \
    "$N_WORKERS" \
    "$N_ESTIMATORS" \
    "$MAX_DEPTH" \
    "$LEARNING_RATE" \
    "$SUBSAMPLE" \
    "$COLSAMPLE" \
    "$MIN_CHILD_WEIGHT" \
    "$GAMMA" \
    "$REG_ALPHA" \
    "$REG_LAMBDA" \
    --train-ratio $TRAIN_RATIO \
    --val-ratio $VAL_RATIO