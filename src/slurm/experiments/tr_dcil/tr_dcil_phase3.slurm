#!/bin/bash
#SBATCH --array=0-31
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

EXPERIMENT_NAME='tuning_phase_3'
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
NB_EPOCHS=600
CHECK_VAL_EVERY_N_EPOCH=10
NUM_WORKERS=8
MIN_EPOCHS=100
PATIENCE=20
VAL_RATIO=0.003

alphas=(0.5 0.8 0.5 0.8 0.5 0.8 0.5 0.8)
betas=(1.0 1.0 2.0 2.0 3.0 3.0 4.0 4.0)  
traj_lens=(5 10 15 20)      

IDX=$SLURM_ARRAY_TASK_ID
num_ab=${#alphas[@]}
num_traj=${#traj_lens[@]}

PAIR_IDX=$((IDX / num_traj))   # 0‑7
TRAJ_IDX=$((IDX % num_traj))   # 0‑3

ALPHA=${alphas[$PAIR_IDX]}
BETA=${betas[$PAIR_IDX]}
TRAJ_LEN=${traj_lens[$TRAJ_IDX]}

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
    --val-ratio "$VAL_RATIO"