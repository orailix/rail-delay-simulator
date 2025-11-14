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

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from collections import deque
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

from src.utils.utils import load_pickle, save_pickle, get_subdir_path, get_dates, setup_run_folder, save_results
from src.utils.logger import CustomCSVLogger, MetricsLoggingCallback
from src.models.transformer import Transformer
from src.utils.weightssaver import WeightSaver
from src.utils.metrics import simulate_and_compute_mae
from src.environment.simulation import Simulator, load_itineraries_from_dates

class DCIL(pl.LightningModule):
    """ 
    Drift Corrected Imitation Learning (DCIL) implementation with a Transformer policy.

    Args:
        policy_config (dict): Configuration dictionary passed to the MLP policy,
            e.g. number of layers, hidden size, activation function.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay coefficient for the optimizer.
        dataset_config (dict): Dataset configuration parameters for data loading.
        sim_config (dict): Simulation configuration (trajectories length, simulation batch size, ...).
        alpha (float): alpha parameter for DCIL weight.
        beta (float): beta parameter for DCIL weight.
        itineraries (dict): Dictionary of train itineraries used in simulation.
        sc (dict): Data scheme.
        cat (dict): Categorical feature definitions.
        stations_emb (dict): Station embeddings.
        lines_emb (dict): Line embeddings.
        eval_config (dict): Configuration for evaluation.
    """
    def __init__(self, policy_config: dict, lr: float, weight_decay: float, dataset_config: dict, sim_config: dict, alpha: float, beta: float, itineraries: dict, sc: dict, 
                cat: dict, stations_emb: dict, lines_emb: dict, eval_config: dict) -> None:
        super(DCIL, self).__init__()

        self.policy = Transformer(**policy_config)

        self.dataset_config = dataset_config
        self.sim_config = sim_config
        self.alpha = alpha
        self.beta = beta
        self.itineraries = itineraries
        self.horizon_obs_bins = eval_config['horizon_obs_bins']
        self.delay_delta_bins = eval_config['delay_delta_bins']
        self.nb_traj_eval = eval_config['nb_traj']
        self.pred_horizon = eval_config['pred_horizon']
        self.sc = sc
        self.cat = cat
        self.stations_emb = stations_emb
        self.lines_emb = lines_emb

        self.lr = lr
        self.weight_decay = weight_decay

        self.differences = None

    def configure_optimizers(self) -> list:
        """
        Configure AdamW optimizer for the policy.

        Returns:
            list[torch.optim.Optimizer]: List with one AdamW optimizer.
        """
        optimizer = optim.AdamW(
            self.policy.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return [optimizer]

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor, distances: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss with distance-based weights.

        Args:
            outputs (torch.Tensor): Logits of shape (batch, seq_len, num_classes).
            labels (torch.Tensor): One-hot targets of shape (batch, seq_len, num_classes).
            distances (torch.Tensor): Distance values used to weight examples of shape (batch, seq_len).
            padding_mask (torch.Tensor): Boolean mask of shape (batch, seq_len),
                True for padded positions.

        Returns:
            torch.Tensor: Scalar weighted loss.
        """
        mask = padding_mask.view(-1)
        outputs = outputs.view(-1, outputs.size(-1))[~mask]
        labels = labels.view(-1, labels.size(-1))[~mask]
        distances = torch.abs(distances.view(-1)[~mask]) # maybe do the abs in the sim ?
        weights = 1/(1 + self.alpha*torch.pow(distances, self.beta))
    
        criterion = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = criterion(outputs, labels)
        
        weighted_loss = per_token_loss * weights
        loss = weighted_loss.sum() / weights.sum() # Compute weighted average loss (weights effectively normalized by their batch sum)

        return loss
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        One training step.

        Args:
            batch (tuple): (inputs, targets, distances, padding_mask).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the batch.
        """
        x, y, dist, padding_mask = batch
        outputs = self.policy(x, padding_mask=padding_mask)
        loss = self.compute_loss(outputs, y, dist, padding_mask)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Update replay buffer at the end of each training epoch by adding new samples.

        Returns:
            None
        """
        self.trainer.datamodule.update_replay_buffer(nb_samples=self.sim_config['new_samples_per_epoch'])

    def on_validation_epoch_start(self) -> None:
        """
        Initialize validation metrics at epoch start.
        """
        self.mae_delay_val = []
        self.mae_horizon_val = []
        self.counter_delay_val = []
        self.counter_horizon_val = []
        self.sse_val = 0

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Perform one validation step by simulating trajectories and computing metrics.

        Args:
            batch (tuple): (initial_states, metadatas) for the trajectories.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        initial_states, metadatas = batch
        mae_delay, mae_horizon, counter_delay, counter_horizon, sse = simulate_and_compute_mae(initial_states, metadatas, self.policy, self.itineraries, False, 
                    self.nb_traj_eval, self.pred_horizon, 'median', self.dataset_config['nb_future_station_reg'], self.device, self.sc, self.cat, self.stations_emb, self.lines_emb, 
                    self.dataset_config, self.delay_delta_bins, self.horizon_obs_bins, 'transformer')

        self.mae_delay_val += mae_delay
        self.mae_horizon_val += mae_horizon
        self.counter_delay_val += counter_delay
        self.counter_horizon_val += counter_horizon
        self.sse_val += sse

    def on_validation_epoch_end(self) -> None:
        """
        Aggregate and log validation metrics at the end of an epoch.

        Computes weighted MAE per delay/horizon bin, overall MAE, and MSE/RMSE
        from accumulated values across validation steps.

        Returns:
            None
        """
        sum_del_counter = torch.stack(self.counter_delay_val).sum(dim=0)
        sum_hor_counter = torch.stack(self.counter_horizon_val).sum(dim=0)
        
        # Convert lists to tensors once
        mae_delay_tensor = torch.stack(self.mae_delay_val)
        counter_delay_tensor = torch.stack(self.counter_delay_val)
        
        mae_horizon_tensor = torch.stack(self.mae_horizon_val)
        counter_horizon_tensor = torch.stack(self.counter_horizon_val)
        
        mae_delay = (mae_delay_tensor * counter_delay_tensor).sum(dim=0) / sum_del_counter.clamp(min=1)
        mae_horizon = (mae_horizon_tensor * counter_horizon_tensor).sum(dim=0) / sum_hor_counter.clamp(min=1)

        for i in range(len(mae_horizon)):
            self.log(f"hor{i}",mae_horizon[i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        for i in range(len(mae_delay)):
            self.log(f"del{i}",mae_delay[i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        mae = torch.dot(mae_horizon, sum_hor_counter) / sum_hor_counter.sum()
        
        self.log(f"mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        mse = self.sse_val/ sum_hor_counter.sum()

        self.log(f"mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log(f"rmse", np.sqrt(mse), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

    def on_test_epoch_start(self) -> None:
        """
        Initialize test metrics at epoch start.
        """
        self.mae_delay_test = []
        self.mae_horizon_test = []
        self.counter_delay_test = []
        self.counter_horizon_test = []
        self.sse_test = 0
        
    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Perform one test step by simulating trajectories and computing metrics.

        Args:
            batch (tuple): (initial_states, metadatas) for the trajectories.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        initial_states, metadatas = batch
        mae_delay, mae_horizon, counter_delay, counter_horizon, sse = simulate_and_compute_mae(initial_states, metadatas, self.policy, self.itineraries, False, 
                    self.nb_traj_eval, self.pred_horizon, 'median', self.dataset_config['nb_future_station_reg'], self.device, self.sc, self.cat, self.stations_emb, 
                    self.lines_emb, self.dataset_config, self.delay_delta_bins, self.horizon_obs_bins, 'transformer')

        self.mae_delay_test += mae_delay
        self.mae_horizon_test += mae_horizon
        self.counter_delay_test += counter_delay
        self.counter_horizon_test += counter_horizon
        self.sse_test += sse

    def on_test_epoch_end(self) -> None:
        """
        Aggregate test metrics at the end of an epoch.

        Computes MAE per delay/horizon bin, overall MAE, MSE, and RMSE from
        accumulated values and stores them in `self.test_results`.

        Returns:
            None
        """
        sum_del_counter = torch.stack(self.counter_delay_test).sum(dim=0)
        sum_hor_counter = torch.stack(self.counter_horizon_test).sum(dim=0)
        
        mae_delay_tensor = torch.stack(self.mae_delay_test)
        counter_delay_tensor = torch.stack(self.counter_delay_test)
        
        mae_horizon_tensor = torch.stack(self.mae_horizon_test)
        counter_horizon_tensor = torch.stack(self.counter_horizon_test)
        
        mae_delay = (mae_delay_tensor * counter_delay_tensor).sum(dim=0) / sum_del_counter.clamp(min=1)
        mae_horizon = (mae_horizon_tensor * counter_horizon_tensor).sum(dim=0) / sum_hor_counter.clamp(min=1)
        mae = torch.dot(mae_horizon, sum_hor_counter) / sum_hor_counter.sum()

        mse = self.sse_test/ sum_hor_counter.sum()

        self.test_results = {
            "mae_horizon": mae_horizon.tolist(),
            "mae_delay": mae_delay.tolist(),
            "mae": mae.item(),
            "mse":mse.item(),
            "rmse":np.sqrt(mse).item(),
        }

def collate_fn(batch: tuple) -> tuple:
    """
    Pad variable-length sequences and create a padding mask.

    Args:
        batch (list): List of (x, y, dist) tensor tuples.

    Returns:
        tuple:
            x_padded (torch.Tensor): Padded inputs of shape (batch, max_len, ...).
            y_padded (torch.Tensor): Padded targets of shape (batch, max_len, ...).
            y_padded (torch.Tensor): Padded distances of shape (batch, max_len, ...).
            padding_mask (torch.Tensor): Bool mask (batch, max_len), True for padding.
    """
    x = [pair[0] for pair in batch] 
    y = [pair[1] for pair in batch]
    dist = [pair[2] for pair in batch]
    max_len = max(sequence.shape[0] for sequence in x)

    # Create padded tensors and padding mask
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    dist_padded = pad_sequence(dist, batch_first=True, padding_value=0)
    
    lengths = torch.tensor([seq.shape[0] for seq in x])
    padding_mask = torch.arange(x_padded.shape[1]) >= lengths.unsqueeze(1)
    
    return x_padded, y_padded, dist_padded, padding_mask

class ReplayBuffer(Dataset):
    """
    Simple replay buffer backed by a deque.
    """
    def __init__(self, capacity: int) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of elements to store.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        """
        Number of elements in the buffer.

        Returns:
            int: Current buffer size.
        """
        return len(self.buffer)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieve one element by index.

        Args:
            idx (int): Index in the buffer.

        Returns:
            torch.Tensor: Tensor stored at the given index.
        """
        return self.buffer[idx]

    def extend(self, data: list) -> None:
        """
        Extends the buffer we the given data, automatically discarding the oldest data if buffer has reached full capacity.

        Args:
          data (list): list of Tensors to add to the replay buffer 

        Returns:
            None
        """
        self.buffer.extend(data)

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
            tuple: (x, y, md) where
                x (torch.Tensor): Input features with selected columns.
                y (torch.Tensor): Target actions.
                md (torch.Tensor): Metadata for the sample.
        """
        idx = self.mapper[idx]
        x = torch.load(get_subdir_path(f'x_{idx}.pt',self.x_path), weights_only = False)[:, self.kept_cols]
        md = torch.load(get_subdir_path(f'md_{idx}.pt',self.md_path), weights_only = False)
        return x, md

class DataModule(pl.LightningDataModule):   
    """
    PyTorch Lightning DataModule for loading train/val/test datasets.

    Args:
        data_path (str): Root path to the dataset.
        dataset_config (dict): Dataset configuration (e.g., split sizes).
        sim_config (dict): Simulation configuration (trajectories length, simulation batch size, ...).
        scheme (dict): Data scheme.
        cat (dict): Categorical feature definitions.
        stations_emb (dict): Station embeddings.
        lines_emb (dict): Line embeddings.
        batch_size (int): Batch size.
        buffer_capacity (int): Capacity of the replay buffer.
        num_workers (int, optional): DataLoader workers. Default is 0.
        pin_memory (bool, optional): Whether to pin memory in DataLoader. Default is True.
        eval_test (bool, optional): If True, use test split for evaluation. Default is False.
        val_ratio (float, optional): Fraction of validation data to keep. Default is 1.0.
    """
    def __init__(self, data_path: str, dataset_config: dict, sim_config: dict, scheme: dict, cat: dict, stations_emb: dict, lines_emb: dict, batch_size: int, 
                buffer_capacity: int, num_workers=0, pin_memory=True, eval_test=False, val_ratio=1.0) -> None:
        super().__init__()
        self.base_path = data_path
        self.dataset_config = dataset_config
        self.scheme = scheme
        self.cat = cat
        self.stations_emb = stations_emb
        self.lines_emb = lines_emb

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sim_config = sim_config
        self.buffer_capacity = buffer_capacity
        self.eval_test = eval_test
        self.val_ratio = val_ratio

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def setup(self, stage: str) -> None:
        """
        Set up datasets and dataloaders for the given stage. If stage is "fit", perform the initial filling of the replay buffer.

        Args:
            stage (str): Current stage ("fit", "validate", "test", "predict").

        Returns:
            None
        """
        self.init_states_train_ds = InitialStateDataset(self.base_path, self.dataset_config, self.scheme, 1.0, 'train')
        if self.eval_test:
            self.init_states_val_ds = InitialStateDataset(self.base_path, self.dataset_config, self.scheme, 1.0, 'val')
            self.init_states_train_ds = ConcatDataset([self.init_states_train_ds, self.init_states_val_ds])
            self.init_states_test_ds  = InitialStateDataset(self.base_path, self.dataset_config, self.scheme, 1.0, 'test')
        else:
            self.init_states_val_ds = InitialStateDataset(self.base_path, self.dataset_config, self.scheme, self.val_ratio, 'val')
            
        self.train_initial_state_dataloader = self.get_initial_state_dataloader()
        
        print('Initial replay buffer filling.')
        if stage == "fit":
            self.update_replay_buffer(nb_samples=self.buffer_capacity) # initial fill of the replay buffer

    def get_initial_state_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for initial states.

        Returns:
            torch.utils.data.DataLoader: Initial states data loader.
        """
        return DataLoader(
            self.init_states_train_ds,
            batch_size=self.sim_config['sim_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda x: x # stacking with padding is done within the simulator
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for training set.

        Returns:
            torch.utils.data.DataLoader: Training data loader.
        """
        return DataLoader(
            self.replay_buffer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for validation set.

        Returns:
            torch.utils.data.DataLoader or list: Validation data loader,
            or an empty list if `eval_test` is True.
        """
        if self.eval_test:
            return []
        return DataLoader(
            self.init_states_val_ds,
            batch_size=2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: ([el[0] for el in x], [el[1] for el in x])
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader or list:
        """
        Create DataLoader for test set.

        Returns:
            torch.utils.data.DataLoader or list: Test data loader
            or an empty list if `eval_test` is False
        """
        if self.eval_test:
            return DataLoader(
                self.init_states_test_ds,
                batch_size=2,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=lambda x: ([el[0] for el in x], [el[1] for el in x])
            )
        else:
            return []            

    def update_replay_buffer(self, nb_samples: int) -> None:
        """
        Collect simulated samples and append them to the replay buffer.

        Args:
            nb_samples (int): Target number of samples to add.

        Returns:
            None
        """
        self.trainer.model.eval()

        sim = Simulator(self.trainer.model.policy, 
                        self.dataset_config['deltat'], 
                        self.scheme['x'],
                        self.cat,
                        self.stations_emb, 
                        self.lines_emb,
                        self.sim_config['device'], 
                        self.dataset_config['nb_past_station_sim'], 
                        self.dataset_config['nb_future_station_sim'], 
                        self.dataset_config['embedding_size'], 
                        self.dataset_config['idle_end'],
                       'transformer')
        
        nb_collected_samples = 0
    
        for batch in self.train_initial_state_dataloader:
            # maybe cut the batch here when possible to avoid additional computing ?
            initial_states = [el[0] for el in batch]
            metadatas = [el[1] for el in batch]
            states_time = [metadatas[i][0,0] for i in range(self.sim_config['sim_batch_size'])]
            initial_states_metadata = [metadatas[i][:,1:] for i in range(self.sim_config['sim_batch_size'])]
        
            with torch.no_grad():
                samples = sim.get_samples_dcil(initial_states, initial_states_metadata, states_time, self.sim_config['traj_len_train'], 1,'sampling', False, 
                itineraries = self.sim_config['itineraries'])
            nb_to_keep = min(nb_samples - nb_collected_samples, len(samples))
            self.replay_buffer.extend(samples[:nb_to_keep])
            nb_collected_samples += len(samples[:nb_to_keep])
            
            sim.reset()
            del batch, initial_states, metadatas, states_time, initial_states_metadata, samples
            gc.collect()
            torch.cuda.empty_cache()

            if nb_collected_samples >= nb_samples:
                break
        self.trainer.model.train()
        
def main() -> None:
    """
    Entry point for training and evaluation.

    Parses CLI arguments, sets up data, model, logger, callbacks, and
    launches training (and testing if `--eval-test` is set).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="Name of the experiment.")
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("itineraries_path", type=str, help="Path to the itineraries.")
    parser.add_argument("eval_config_path", type=str, help="Path to the evaluation config.")
    parser.add_argument("d_model", type=int, help="Dimensionality of the transformer embeddings (model hidden size).")
    parser.add_argument("nhead", type=int, help="Number of attention heads in each multi-head self-attention layer.")
    parser.add_argument("dim_feedforward", type=int, help="Hidden layer size of the position-wise feedforward sublayer.")
    parser.add_argument("dropout", type=float, help="Dropout probability for all dropout layers (0.0–1.0).")
    parser.add_argument("activation", type=str, choices=["relu","gelu","tanh"], help="Activation function to use in the feedforward layers.")
    parser.add_argument("num_layers", type=int, help="Number of stacked TransformerEncoder layers.")
    parser.add_argument("traj_len", type=int, help="Number of steps per trajectory.")
    parser.add_argument("sim_batch_size", type=int, help="Number of simulations to do in parallel.")
    parser.add_argument("buffer_capacity", type=int, help="Maximum capacity of the replay buffer.")
    parser.add_argument("new_samples_per_epoch", type=int, help="Number of new samples to generate per epoch.")
    parser.add_argument("nb_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("alpha", type=float, help="Alpha.")
    parser.add_argument("beta", type=float, help="Beta.")
    parser.add_argument("batch_size", type=int, help="Training batch size.")
    parser.add_argument("lr", type=float, help="LR for Adam")
    parser.add_argument("weight_decay", type=float, help="AdamW L2")
    parser.add_argument("check_val_every_n_epoch", type=int, help="How often (in epochs) to run validation.")
    parser.add_argument("num_workers", type=int, help="Number of subprocesses to use for data loading.")
    parser.add_argument("min_epochs", type=int, help="Minimum number of epochs before early stopping can trigger.")
    parser.add_argument("patience", type=int, help="Number of validation checks with no improvement before early stopping triggers.")
    parser.add_argument("--val-ratio", type=float, default=1.0, help="Ratio of validation data kept to compute metrics during training.")
    parser.add_argument("--eval-test", action="store_true", help="If true, train on train+val and evaluate on test, if false, train on train and evaluate on val.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed, workers=True)

    scheme = load_pickle(os.path.join(args.data_path, 'sc_sim_non.pkl'))
    cat = load_pickle(os.path.join(args.data_path, 'cat.pkl'))
    stations_emb = load_pickle(os.path.join(args.data_path, 'stations_emb.pkl'))
    lines_emb = load_pickle(os.path.join(args.data_path, 'lines_emb.pkl'))
    dataset_config = load_pickle(os.path.join(args.data_path, 'config.pkl'))

    model_config = {
        "input_dim":len(scheme['x'].items()),
        "d_model":args.d_model,
        "nhead":args.nhead,
        "dim_feedforward":args.dim_feedforward,
        "dropout":args.dropout,
        "activation":args.activation,
        "num_layers":args.num_layers,
        "num_classes":len(scheme['y'].items())
    }

    eval_months = ['train','val','test']
    dates = get_dates(args.data_path, eval_months)
    itineraries = load_itineraries_from_dates(dates, args.itineraries_path, show_prog = True)
    
    sim_config = {
        'traj_len_train':args.traj_len,
        'sim_batch_size':args.sim_batch_size,
        'new_samples_per_epoch':args.new_samples_per_epoch,
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'itineraries':itineraries
    }

    eval_config = load_pickle(args.eval_config_path)

    run_path, checkpoints_path = setup_run_folder(args, model_config, 'tr_dcil')
    
    data_module = DataModule(args.data_path, dataset_config, sim_config, scheme, cat, stations_emb, lines_emb,args.batch_size, buffer_capacity = args.buffer_capacity, 
    num_workers=args.num_workers, eval_test=args.eval_test, val_ratio=args.val_ratio)
    
    model = DCIL(model_config, args.lr, args.weight_decay, dataset_config, sim_config, args.alpha, args.beta, itineraries, scheme['x'], cat, stations_emb, lines_emb, eval_config)
    
    if not args.eval_test:
        logger = CustomCSVLogger(save_dir=run_path, 
                                 train_metrics=["train_loss"],
                                 val_metrics=[f"hor{i}" for i in range(len(eval_config['horizon_obs_bins'])-1)] + 
                                             [f"del{i}" for i in range(len(eval_config['delay_delta_bins'])-1)] + 
                                             ["mae","mse","rmse"])

        callbacks = [
            WeightSaver(save_path=checkpoints_path, 
                        target="policy", 
                        keep_module_prefix=False,
                        top_k=5,
                        monitor="mae",
                        mode="min"),
            MetricsLoggingCallback(logger)
        ]
        early_stop = EarlyStopping(
            monitor="mae",
            mode="min",
            patience=args.patience
        )
        callbacks.append(early_stop)
    else:
        callbacks = []
        logger = None
    
    trainer = Trainer(
        devices=1,
        strategy="auto",
        accelerator="gpu",
        min_epochs=args.min_epochs,
        max_epochs=args.nb_epochs,
        logger=logger,
        precision='bf16-mixed',
        reload_dataloaders_every_n_epochs=1, # because the replay buffer length changes initially
        check_val_every_n_epoch = args.check_val_every_n_epoch,
    	callbacks = callbacks,
        enable_checkpointing=False,
        num_sanity_val_steps = 0
    )
    
    trainer.fit(model, data_module)

    if args.eval_test:
        trainer.test(model, datamodule=data_module)
        print(model.test_results)
        save_results(run_path, model.test_results,eval_config)

if __name__ == "__main__":
    main()