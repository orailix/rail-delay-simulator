import os
import math
import torch
import datetime
import gc
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from collections import deque
from torch.nn.utils.rnn import pad_sequence

from src.utils.utils import load_pickle, save_pickle, get_subdir_path, get_dates, setup_run_folder, save_results
from src.utils.logger import CustomCSVLogger, MetricsLoggingCallback
from src.models.transformer import Transformer
from src.models.mlp import MLP
from src.utils.weightssaver import WeightSaver
from src.utils.metrics import _fill_missing_predictions
from src.environment.simulation import Simulator, load_itineraries_from_dates

class InitialStateDataset(Dataset):
    """
    Dataset wrapper for tensor loading of initial states used in the simulation.

    Args:
        base_path (str): Root directory of the dataset.
        config (dict): Config dict containing split sizes (e.g. "train_size").
        scheme (dict): Data scheme.
        ratio (float): Fraction of the split to keep (0–1).
        split (str): Dataset split ("train", "val", "test").
    """
    def __init__(self, base_path: str, config: dict, scheme: dict, ratio: float, split: str) -> None:
        self.kept_cols = scheme['cols_to_keep']
        self.ratio = ratio
        total_data = config[f"{split}_size"]
        self.x_path = os.path.join(base_path, split, 'x')
        self.y_path = os.path.join(base_path, split, 'y_actions')
        self.md_path = os.path.join(base_path, split, 'md')
        self.mapper = torch.linspace(0, total_data - 1, int(total_data*ratio), dtype=int)
        self.len = self.mapper.shape[0]

    def __len__(self) -> int:
        """
        Dataset size.

        Returns:
            int: Number of samples.
        """
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        """
        Load one sample from disk.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (x, md) where
                x (torch.Tensor): Input features with selected columns.
                md (torch.Tensor): Metadata for the sample.
        """
        idx = self.mapper[idx]
        x = torch.load(get_subdir_path(f'x_{idx}.pt',self.x_path), weights_only = False)[:, self.kept_cols]
        md = torch.load(get_subdir_path(f'md_{idx}.pt',self.md_path), weights_only = False)
        return x, md

def get_distrib(output: dict, itineraries_dict: dict, initial_state_metadata: list, start_state_time: int, predictive_horizon: int, 
            nb_pred_max: int, nb_future_st: int, horizon_obs_bins: list) -> tuple:
    """
    Collect predictive distributions and ground-truth arrival times for calibration analysis.

    This extracts, for each (DATDEP, TRAIN_NO) snapshot, the model's per-step predictive
    distributions (e.g., from an ensemble of stochastic trajectories) and aligns them with
    the corresponding observed times, filtered to valid horizon bins. The resulting pairs
    can be used to build calibration plots in the style of Kuleshov et al. (2018): compare
    predicted coverage to empirical coverage to assess over/under-confidence.

    Args:
        output (dict): Model outputs containing 'predictions', 'start_pos', and 'mapper' keyed by (DATDEP, TRAIN_NO).
        itineraries_dict (dict): Dictionary with per-date tensors ('data'), train index map ('train_no'),
            and 'max_pos' for valid sequence ends.
        initial_state_metadata (list): List of (DATDEP, TRAIN_NO) identifying evaluation snapshots.
        start_state_time (int): Reference time (seconds) for converting to relative horizons.
        predictive_horizon (int): Maximum allowed prediction horizon in seconds.
        nb_pred_max (int): Upper bound on the number of prediction steps to consider.
        nb_future_st (int): Number of trailing placeholder/future stations to exclude from evaluation.
        horizon_obs_bins (list): Monotonic bin edges (seconds) used to select valid observation horizons.

    Returns:
        tuple: (distribs, true) where
            distribs (list): List of per-step predictive distributions (one array-like per evaluated step),
                expressed as times relative to start_state_time.
            true (list): List of observed arrival times (seconds) relative to start_state_time,
                aligned with distribs for calibration.
    """
    distribs = []
    true = []

    for datdep, train_no in initial_state_metadata:
        iti_idx = itineraries_dict[datdep]['train_no'][train_no]
        itinerary = itineraries_dict[datdep]['data'][iti_idx]
        output_idx = output['mapper'][(datdep, train_no)]
        start_pos = output['start_pos'][output_idx]
        pred = output['predictions'][output_idx]

        horizon_obs = itinerary[:, 4] - start_state_time
        valid_horizon_mask = (horizon_obs_bins[0] <= horizon_obs) & (horizon_obs < horizon_obs_bins[-1])
        valid_horizon_mask[-nb_future_st:] = False
        if not valid_horizon_mask.any():
            continue
        max_end_pos = valid_horizon_mask.nonzero()[-1, 0].item()
        iti_max_pos = itineraries_dict[datdep]['max_pos'][iti_idx]
        max_end_pos = min(max_end_pos, start_pos + nb_pred_max, iti_max_pos)

        pred_slice = pred[:, start_pos + 1:max_end_pos + 1].cpu()
        pred_slice = _fill_missing_predictions(pred_slice, itinerary, start_pos, max_end_pos, start_state_time, predictive_horizon) - start_state_time

        observed_times = itinerary[start_pos + 1:max_end_pos + 1, 4] - start_state_time

        distribs.extend([pred_slice[:,i] for i in range(pred_slice.shape[1])])
        true.extend(observed_times.tolist())
    return distribs, true


def calibration_plot(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, itineraries: dict, action_constraint: bool, 
            nb_samples: int, predictive_horizon: int, nb_pred_max: int, device: torch.device, column_mapping: dict, 
            cat_cols_metadata: dict, stations_emb: dict, lines_emb: dict, dataset_config: dict, net_type: str, 
            horizon_obs_bins: list, save_path: str, local_features: bool = False) -> None:
    """
    Generate a calibration curve using PIT-based coverage from simulation outputs.

    For each initial state, runs an ensemble of stochastic trajectories, collects per-step
    predictive distributions and corresponding ground-truth times, computes PIT values
    (fraction of samples ≤ observed), bins them, and plots the cumulative observed coverage
    against the nominal confidence levels (diagonal is perfect calibration).

    Args:
        dataloader (torch.utils.data.DataLoader): Batches of initial states and metadata.
        model (torch.nn.Module): Trained model used by the simulator to produce predictions.
        itineraries (dict): Dictionary holding per-date itineraries, indices, and metadata.
        action_constraint (bool): Indicates whether to use action constraint.
        nb_samples (int): Number of stochastic trajectories per initial state.
        predictive_horizon (int): Maximum prediction horizon in seconds.
        nb_pred_max (int): Maximum number of prediction steps to consider per sequence.
        device (torch.device): Torch device on which to run simulations.
        column_mapping (dict): Feature/target column mapping required by the simulator.
        cat_cols_metadata (dict): Category metadata used to interpret one-hot features.
        stations_emb (dict): Station embeddings mapping.
        lines_emb (dict): Line embeddings mapping.
        dataset_config (dict): Dataset configuration (e.g., deltat, window sizes, embedding size).
        net_type (str): Model family identifier used by the simulator.
        horizon_obs_bins (list): Bin edges for valid observation horizons.
        save_path (str): File path to save the calibration figure.
        local_features (bool): Whether to include local neighbor features in simulation. Default is False.

    Returns:
        None
    """

    model.eval()

    cfg = dataset_config
    deltat = cfg['deltat']
    nb_past_station = cfg['nb_past_station_sim']
    nb_future_station = cfg['nb_future_station_sim']
    embedding_size = cfg['embedding_size']
    idle_time_end = cfg['idle_end']

    n_bins = 10
    hist_counts = np.zeros(n_bins, int)

    for initial_states, metadatas in tqdm(dataloader):

        states_time = [metadata[0, 0] for metadata in metadatas]
        initial_states_metadata = [metadata[:, 1:] for metadata in metadatas]
    
        sim = Simulator(
            model, deltat, column_mapping, cat_cols_metadata, stations_emb, lines_emb, device, 
            nb_past_station, nb_future_station, embedding_size, idle_time_end, net_type, local_features
        )
    
        s, it = sim.predict_delay(
            initial_states,
            initial_states_metadata,
            states_time,
            predictive_horizon,
            nb_samples,
            'sampling',
            action_constraint,
            itineraries=itineraries,
        )
    
        for sim_idx in range(len(metadatas)):
            distribs, true_vals = get_distrib(
                s.output[sim_idx], it,
                initial_states_metadata[sim_idx],
                states_time[sim_idx],
                predictive_horizon, nb_pred_max,
                nb_future_station, horizon_obs_bins
            )
        
            for d, t in zip(distribs, true_vals):
                samples = np.asarray(d)
                pit = (samples <= t).mean()
                bucket = min(int(pit * n_bins), n_bins - 1)
                hist_counts[bucket] += 1

    y = np.cumsum(np.insert(hist_counts, 0, 0)) / hist_counts.sum()
    x = np.arange(0,1.1,0.1)
    
    plt.plot(x,y, marker = '.', label='Ours', markersize=10)
    plt.plot(x,x, marker = '.', color='grey', label='Perfect calibration', markersize=10)
    plt.xlabel('Expected Confidence Level')  
    plt.ylabel('Observed Confidence Level') 
    
    plt.grid(True)
    plt.tight_layout()

    plt.legend()

    plt.savefig(save_path)
        
def main() -> None:
    """
    Entry point for generating a calibration plot for a trained model.

    Parses CLI arguments, loads configs, embeddings, itineraries, and checkpoint,
    builds the evaluation dataset and dataloader, runs simulation-based calibration,
    and saves the resulting figure to disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str, help="Path of the run.")
    parser.add_argument("cp", type=str, help="Checkpoint number.")
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("itineraries_path", type=str, help="Path to the itineraries.")
    parser.add_argument("eval_config_path", type=str, help="Path to the evaluation config.")
    parser.add_argument("save_path", type=str, help="Path to save the plot.")
    parser.add_argument("ratio", type=float, help="Ratio of kept data.")
    
    args = parser.parse_args()
    print(args)

    scheme = load_pickle(os.path.join(args.data_path, 'sc_sim_non.pkl'))
    cat = load_pickle(os.path.join(args.data_path, 'cat.pkl'))
    stations_emb = load_pickle(os.path.join(args.data_path, 'stations_emb.pkl'))
    lines_emb = load_pickle(os.path.join(args.data_path, 'lines_emb.pkl'))
    dataset_config = load_pickle(os.path.join(args.data_path, 'config.pkl'))

    eval_months = ['test']
    dates = get_dates(args.data_path, eval_months)
    itineraries = load_itineraries_from_dates(dates, args.itineraries_path, show_prog = True)

    eval_config = load_pickle(args.eval_config_path)

    model_config = load_pickle(os.path.join(args.run_path, 'model_config.pkl'))

    model = Transformer(**model_config)

    state = torch.load(os.path.join(args.run_path, 'checkpoints', f'model_epoch_{args.cp}.pt'), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    dataset = InitialStateDataset(args.data_path, dataset_config, scheme, args.ratio, 'test')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=lambda x: ([el[0] for el in x], [el[1] for el in x]))

    calibration_plot(dataloader, model, itineraries, False, 100, eval_config['pred_horizon'], 15, device, scheme['x'], cat, stations_emb, lines_emb, dataset_config, 'transformer', eval_config['horizon_obs_bins'], args.save_path)

if __name__ == "__main__":
    main()