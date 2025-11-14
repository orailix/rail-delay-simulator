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

SIM_BATCH_SIZE=100
BUFFER_CAP=60000
NSPE=20000
TRAJ_LEN=10
ALPHA=0.5
BETA=2.0

BATCH_SIZE=128
WEIGHT_DECAY=0.01

NB_EPOCHS=600
CHECK_VAL_EVERY_N_EPOCH=10
NUM_WORKERS=8
MIN_EPOCHS=100
PATIENCE=20
VAL_RATIO=0.003

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
    --val-ratio $VAL_RATIO