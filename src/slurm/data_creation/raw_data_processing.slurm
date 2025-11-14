#!/bin/bash
#SBATCH --array=0-35
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

BASE_YEAR=2022
BASE_MONTH=1

IDX=$SLURM_ARRAY_TASK_ID
MONTH_INDEX=$(( BASE_MONTH + IDX ))
YEAR=$(( BASE_YEAR + (MONTH_INDEX-1)/12 ))
MONTH=$(( (MONTH_INDEX-1)%12 + 1 ))

MM=$(printf "%02d" "$MONTH")

FOLDER_IN='data/raw'
FOLDER_OUT='data/processed_data'
DELTAT=30
NB_PAST=10
NB_FUTURE=30
IDLE_BEG=5
IDLE_END=5
SAMPLE_RATIO=0.1

echo "Task $IDX → processing $YEAR-$MM"

# Launch
srun python3 -u -m src.data.raw_data_processing \
    "$FOLDER_IN" \
    "$FOLDER_OUT" \
    "$YEAR" \
    "$MONTH" \
    "$DELTAT" \
    "$NB_PAST" \
    "$NB_FUTURE" \
    "$IDLE_BEG" \
    "$IDLE_END" \
    "$SAMPLE_RATIO"
