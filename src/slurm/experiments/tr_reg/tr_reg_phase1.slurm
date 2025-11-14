#!/bin/bash
#SBATCH --array=0-23
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='tuning_phase_1'
DATA_PATH='data/dataset'
EVAL_CONFIG='data/eval_configs/cfg.pkl'

# Define search grid
d_models=(128 256 512 1024)
nlayers=(4 6)
lrs=(0.0001 0.00005 0.00001)

IDX=$SLURM_ARRAY_TASK_ID

configs_per_dmodel=$(( ${#nlayers[@]} * ${#lrs[@]} ))
dm_idx=$(( IDX / configs_per_dmodel ))
rest=$(( IDX % configs_per_dmodel ))
nl_idx=$(( rest / ${#lrs[@]} ))
lr_idx=$(( rest % ${#lrs[@]} ))

D_MODEL=${d_models[$dm_idx]}
NUM_LAYERS=${nlayers[$nl_idx]}
LR=${lrs[$lr_idx]}
DIM_FF=$(( 4 * D_MODEL ))

NHEAD=8
DROPOUT=0.2
ACTIVATION='relu'
BATCH_SIZE=64
WEIGHT_DECAY=0.01
NB_EPOCHS=80
CHECK_VAL_EVERY_N_EPOCH=1
NUM_WORKERS=8
MIN_EPOCHS=20
PATIENCE=20
VAL_RATIO=0.003

srun python3 -u -m src.algorithms.regression.transformer_reg \
    "$EXPERIMENT_NAME" \
    "$DATA_PATH" \
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
    --val-ratio $VAL_RATIO