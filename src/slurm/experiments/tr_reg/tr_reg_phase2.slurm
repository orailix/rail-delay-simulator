#!/bin/bash
#SBATCH --array=0-26
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='tuning_phase_2'
DATA_PATH='data/dataset'
EVAL_CONFIG='data/eval_configs/cfg.pkl'

D_MODEL=512
DIM_FF=2048
NUM_LAYERS=4
NHEAD=8
ACTIVATION='relu'
WEIGHT_DECAY=0.01
NB_EPOCHS=80
CHECK_VAL_EVERY_N_EPOCH=1
NUM_WORKERS=8
MIN_EPOCHS=20
PATIENCE=20
VAL_RATIO=0.003

bss=(64 128 256)
lrs=(0.0001 0.00005 0.00002)
drops=(0.05 0.10 0.20)

IDX=$SLURM_ARRAY_TASK_ID
num_lr=${#lrs[@]}
num_bs=${#bss[@]}
num_dp=${#drops[@]}

bs_idx=$(( (IDX / num_dp) % num_bs ))   # 0–2
lr_idx=$(( (IDX / (num_bs * num_dp)) )) # 0–2
dp_idx=$(( IDX % num_dp ))              # 0–2

BATCH_SIZE=${bss[$bs_idx]}
LR=${lrs[$lr_idx]}
DROPOUT=${drops[$dp_idx]}

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