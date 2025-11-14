#!/bin/bash
#SBATCH --array=0-44        
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='tuning_phase_1'
DATA_PATH='data/dataset'
ITINERARIES_PATH='data/itineraries'
EVAL_CONFIG_PATH='data/eval_configs/cfg.pkl'

N_ESTIMATORS=400
LEARNING_RATE=0.1
N_WORKERS=10
TRAIN_RATIO=0.1
VAL_RATIO=0.01

REG_ALPHA=0      
REG_LAMBDA=1

DEPTH_MCW_ARR=(
    "4 1"  "6 1"  "6 5"
    "9 5"  "13 5"
)
SUB_COL_ARR=(
    "0.6 0.5" "0.8 0.8" "1.0 0.8"
)
GAMMA_ARR=(0 1 5)

IDX=$SLURM_ARRAY_TASK_ID

GAMMA_IDX=$(( IDX % 3 ))
TMP=$(( IDX / 3 ))
SUBCOL_IDX=$(( TMP % 3 ))
DEPTH_IDX=$(( TMP / 3 ))

read MAX_DEPTH MIN_CHILD_WEIGHT <<< "${DEPTH_MCW_ARR[$DEPTH_IDX]}"
read SUBSAMPLE  COLSAMPLE       <<< "${SUB_COL_ARR[$SUBCOL_IDX]}"
GAMMA=${GAMMA_ARR[$GAMMA_IDX]}

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