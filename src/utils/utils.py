import torch
import datetime
import random
import pickle
import os
import json
import hashlib
import psutil
import calendar

import pandas as pd
import numpy as np

def save_results(run_path: str, results: dict, eval_cfg: dict) -> None:
    """
    Save evaluation results to a CSV file in the run directory.

    Args:
        run_path (str): Path to the run output directory.
        results (dict): Dictionary containing metrics ('mae_horizon', 'mae_delay', 'mae', 'mse', 'rmse').
        eval_cfg (dict): Evaluation config with 'horizon_obs_bins' and 'delay_delta_bins'.

    Returns:
        None
    """

    cols = ['epoch'] + [f"hor{i}" for i in range(len(eval_cfg['horizon_obs_bins'])-1)] + [f"del{i}" for i in range(len(eval_cfg['delay_delta_bins'])-1)] + ["mae"] + ['mse'] + ['rmse']
    row = [0] + results['mae_horizon'] + results['mae_delay'] + [results['mae'], results['mse'], results['rmse']]
    df = pd.DataFrame([row], columns=cols)
    df.to_csv(os.path.join(run_path, 'metrics_log.csv'))

def setup_run_folder(args: object, model_config: dict, algo: str) -> tuple:  
    """
    Create a new run folder with timestamp and config hash, and save configs.

    Args:
        args (object): Parsed command-line arguments (Namespace-like).
        model_config (dict): Model configuration to save.
        algo (str): Algorithm name for organizing runs.

    Returns:
        tuple: 
            - run_path (str): Path to the run directory.  
            - checkpoints_path (str): Path to the checkpoints subdirectory.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cfg_dict = {k: v for k, v in vars(args).items() if k != "seed"}
    cfg_blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode()
    cfg_hash = hashlib.sha1(cfg_blob).hexdigest()[:8]

    run_name = f"{timestamp}__{cfg_hash}__seed={args.seed}"
    run_path = os.path.join("runs",algo,args.experiment_name,run_name)
    checkpoints_path = os.path.join(run_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    
    save_pickle(model_config, os.path.join(run_path, "model_config.pkl"))
    save_pickle({k: v for k, v in vars(args).items()}, os.path.join(run_path, "args.pkl"))

    return run_path, checkpoints_path

def load_pickle(file_path: str) -> object:
    """
    Load a Python object from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        object: The deserialized Python object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj: object, file_path: str) -> None:
    """
    Save a Python object to a pickle file.

    Args:
        obj (object): The Python object to serialize.
        file_path (str): Path where the pickle file will be saved.

    Returns:
        None
    """

    dir_name = os.path.dirname(file_path)
    if dir_name:  # Only create directories if there's a valid directory path
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def get_subdir_path(file_name: str, base_path: str) -> str:
    """
    Build a subdirectory path using a hash prefix of the file name.

    Args:
        file_name (str): Name of the file.
        base_path (str): Base directory path.

    Returns:
        str: Full path combining base path, hash subdirectory, and file name.
    """

    hash_prefix = hashlib.md5(file_name.encode()).hexdigest()[:2]  # First 2 chars of hash
    return os.path.join(base_path, hash_prefix, file_name)

def get_dates(data_path: str, splits: list) -> list:
    """
    Collect all dates corresponding to the months in the dataset splits.

    Args:
        data_path (str): Path to the dataset directory containing `config.pkl`.
        splits (list): List of split names (e.g., ["train", "val", "test"]).

    Returns:
        list: Sorted list of date strings in format 'DDMONYYYY' (uppercase).
    """

    dataset_config = load_pickle(os.path.join(data_path, 'config.pkl'))
    months = []
    for split in splits:
        months += dataset_config[f'{split}_months']

    dates = set()
    for m in months:
        year, month = int(m[:4]), int(m[4:])
        first_day = datetime.date(year, month, 1)
        days_in_month = calendar.monthrange(year, month)[1]

        start = first_day - datetime.timedelta(days=1)
        end = first_day + datetime.timedelta(days=days_in_month)

        total_days = (end - start).days + 1
        for i in range(total_days):
            d = start + datetime.timedelta(days=i)
            dates.add(d.strftime('%d%b%Y').upper())

    return sorted(
        dates,
        key=lambda s: datetime.datetime.strptime(s, '%d%b%Y')
    )