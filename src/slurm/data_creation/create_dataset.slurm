#!/bin/bash
#YOUR SBATCH ARGS
# ...
#YOUR SBATCH ARGS

source ~/.bashrc
conda activate rail-delay-pred

FOLDER_IN='data/processed_data'
FOLDER_OUT='data/dataset'
STATIONS_EMB_PATH='data/embeddings/stations_emb_8.pkl'
LINES_EMB_PATH='data/embeddings/lines_emb_8.pkl'
NB_PAST_REG=5
NB_FUT_REG=15
NB_PAST_SIM=5
NB_FUT_SIM=5
CAT_ENC='one_hot'
TIME_FEAT_ENC='cyclical'
TEST_RATIO=0.008

TRAIN_MONTHS=(
  202201 202202 202203 202204 202205 202206 202207 202208 202209
  202210 202211 202212 202301 202302 202303 202304 202305 202306
  202307 202308 202309
)

VAL_MONTHS=(
  202310 202311 202312
)

TEST_MONTHS=(
  202401 202402 202403 202404 202405 202406 202407 202408 202409
  202410 202411 202412
)

srun python3 -u -m src.data.create_dataset \
    "$FOLDER_IN" \
    "$FOLDER_OUT" \
    "$STATIONS_EMB_PATH" \
    "$LINES_EMB_PATH" \
    "$NB_PAST_REG" \
    "$NB_FUT_REG" \
    "$NB_PAST_SIM" \
    "$NB_FUT_SIM" \
    "$CAT_ENC" \
    "$TIME_FEAT_ENC" \
    "$TEST_RATIO" \
    --train "${TRAIN_MONTHS[@]}" \
    --val "${VAL_MONTHS[@]}" \
    --test "${TEST_MONTHS[@]}"