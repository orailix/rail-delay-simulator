#!/bin/bash
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

FOLDER_IN='data/raw'
FOLDER_OUT='data/itineraries'
STATIONS_EMB='data/embeddings/stations_emb_8.pkl'
LINES_EMB='data/embeddings/lines_emb_8.pkl'

FIRST_YEAR=2021
FIRST_MONTH=12
LAST_YEAR=2025
LAST_MONTH=1

DELTAT=30
NB_PAST_ST=5
NB_FUTURE_ST=5
IDLE_BEG=5
IDLE_END=5

srun python3 -u -m src.data.create_itineraries \
    "$FOLDER_IN" \
    "$FOLDER_OUT" \
    "$STATIONS_EMB" \
    "$LINES_EMB" \
    $FIRST_YEAR $FIRST_MONTH $LAST_YEAR $LAST_MONTH \
    $DELTAT $NB_PAST_ST $NB_FUTURE_ST $IDLE_BEG $IDLE_END
