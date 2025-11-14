import os
import argparse
import pickle
import datetime
import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.logger import CustomCSVLogger, MetricsLoggingCallback
from src.utils.weightssaver import WeightSaver
from src.utils.utils import load_pickle, save_pickle, get_subdir_path, get_dates, setup_run_folder, save_results
from src.models.mlp import MLP
from src.utils.metrics import simulate_and_compute_mae
from src.environment.simulation import load_itineraries_from_dates

class BC(pl.LightningModule):
    """ 
    Behavioral Cloning (BC) implementation with an MLP policy.

    It should be noted that the data is stored as a Tensor of shape (seq_len, nb_feat, therefore elements of any batch are actually Tensors 
    of shape (seq_len, nb_feat), which are concatenated to create the final batch.

    Args:
        model_config (dict): Configuration dictionary passed to the MLP policy,
            e.g. number of layers, hidden size, activation function.
        weight_decay (float): Weight decay coefficient for the optimizer.
        lr (float): Learning rate for the optimizer.
        dataset_config (dict): Dataset configuration parameters for data loading.
        itineraries (dict): Dictionary of train itineraries used in simulation.
        sc (dict): Data scheme.
        cat (dict): Categorical feature definitions.
        stations_emb (dict): Station embeddings.
        lines_emb (dict): Line embeddings.
        eval_config (dict): Configuration for evaluation.
        use_local_features (bool): If True, include local features in the input.
    """
    def __init__(self, model_config: dict, weight_decay: float, lr: float, dataset_config: dict, itineraries: dict, sc: dict, cat: dict, stations_emb:dict, lines_emb: dict, 
                eval_config: dict, use_local_features: bool) -> None:
        super(BC, self).__init__()

        self.policy = MLP(**model_config)

        self.dataset_config = dataset_config
        self.itineraries = itineraries
        self.horizon_obs_bins = eval_config['horizon_obs_bins']
        self.delay_delta_bins = eval_config['delay_delta_bins']
        self.nb_traj_eval = eval_config['nb_traj']
        self.pred_horizon = eval_config['pred_horizon']
        self.sc = sc
        self.cat = cat
        self.stations_emb = stations_emb
        self.lines_emb = lines_emb

        self.weight_decay = weight_decay
        self.lr = lr

        self.use_local_features = use_local_features

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

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the Cross-entropy loss.

        Args:
            outputs (torch.Tensor): Logits of shape (batch_size, num_classes).
            labels (torch.Tensor): Targets of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Scalar loss.
        """
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1]).to(self.device))
        loss = criterion(outputs, labels)
        return loss

    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        One training step.

        Args:
            batch (tuple): (inputs, targets).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the batch.
        """
        x, y = batch
        outputs = self.policy(x)
        loss = self.compute_loss(outputs, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

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
        mae_delay, mae_horizon, counter_delay, counter_horizon, sse = simulate_and_compute_mae(initial_states, metadatas, self.policy, self.itineraries, True, 
                    self.nb_traj_eval, self.pred_horizon, 'median', self.dataset_config['nb_future_station_reg'], self.device, self.sc, self.cat, self.stations_emb, 
                    self.lines_emb, self.dataset_config, self.delay_delta_bins, self.horizon_obs_bins, 'mlp', self.use_local_features)

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
        mae_delay, mae_horizon, counter_delay, counter_horizon, sse = simulate_and_compute_mae(initial_states, metadatas, self.policy, self.itineraries, True, 
                    self.nb_traj_eval, self.pred_horizon, 'median', self.dataset_config['nb_future_station_reg'], self.device, self.sc, self.cat, self.stations_emb, 
                    self.lines_emb, self.dataset_config, self.delay_delta_bins, self.horizon_obs_bins, 'mlp', self.use_local_features)

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

        mse = self.sse_test / sum_hor_counter.sum()

        self.test_results = {
            "mae_horizon": mae_horizon.tolist(),
            "mae_delay": mae_delay.tolist(),
            "mae": mae.item(),
            "mse":mse.item(),
            "rmse":np.sqrt(mse).item(),
        }

class TensorDataset(Dataset):
    """
    Dataset wrapper for tensor loading.

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
        x = torch.load(get_subdir_path(f'x_{idx}.pt',self.x_path))[:, self.kept_cols]
        y = torch.load(get_subdir_path(f'y_actions_{idx}.pt',self.y_path))
        md = torch.load(get_subdir_path(f'md_{idx}.pt',self.md_path), weights_only = False)
        return x, y, md

def collate_fn(batch: tuple) -> tuple:
    """
    Collate a list of samples into a batch.

    Args:
        batch (list): List of (x, y) tuples.

    Returns:
        tuple: (xs, ys) where
            xs (torch.Tensor): Batched inputs.
            ys (torch.Tensor): Batched targets.
    """
    xs = torch.cat([item[0] for item in batch], dim=0)
    ys = torch.cat([item[1] for item in batch], dim=0)
    
    return xs, ys

class MyDataModule(pl.LightningDataModule):   
    """
    PyTorch Lightning DataModule for loading train/val/test datasets.

    Args:
        data_path (str): Root path to the dataset.
        config (dict): Dataset configuration (e.g., split sizes).
        scheme (dict): Data scheme.
        batch_size (int): Batch size.
        num_workers (int, optional): DataLoader workers. Default is 0.
        pin_memory (bool, optional): Whether to pin memory in DataLoader. Default is True.
        eval_test (bool, optional): If True, use test split for evaluation. Default is False.
        val_ratio (float, optional): Fraction of validation data to keep. Default is 1.0.
    """
    def __init__(self, data_path: str, config: dict, scheme: dict, batch_size: int, num_workers: int = 0, pin_memory: bool = True, eval_test: bool = False, 
                val_ratio: float = 1.0) -> None:
        super().__init__()
        self.base_path = data_path
        self.config = config
        self.scheme = scheme
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.eval_test = eval_test

        self.train_ds = TensorDataset(self.base_path, self.config, self.scheme, 1.0, 'train')
        if self.eval_test:
            self.val_ds = TensorDataset(self.base_path, self.config, self.scheme, 1.0, 'val')
            self.train_ds = ConcatDataset([self.train_ds, self.val_ds])
            self.test_ds  = TensorDataset(self.base_path, self.config, self.scheme, 1.0, 'test')
        else:
            self.val_ds = TensorDataset(self.base_path, self.config, self.scheme, val_ratio, 'val')

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for training set.

        Returns:
            torch.utils.data.DataLoader: Training data loader.
        """
        return DataLoader(
            self.train_ds,
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
            self.val_ds,
            batch_size=2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: ([el[0] for el in x], [el[2] for el in x])
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
                self.test_ds,
                batch_size=2,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=lambda x: ([el[0] for el in x], [el[2] for el in x])
            )
        else:
            return []

def main() -> None:
    """
    Entry point for training and evaluation.

    Parses CLI arguments, sets up data, model, logger, callbacks, and
    launches training (and testing if `--eval-test` is set).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="A unique name/ID for this training run (used for logging & checkpoints).")
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("itineraries_path", type=str, help="Path to the itineraries.")
    parser.add_argument("eval_config_path", type=str, help="Path to the evaluation config.")
    parser.add_argument("dropout", type=float, help="Dropout probability for all dropout layers (0.0–1.0).")
    parser.add_argument("activation", type=str, help="Activation function to use in the feedforward layers.")
    parser.add_argument("batch_size", type=int, help="Number of samples per training batch.")
    parser.add_argument("weight_decay", type=float, help="Weight decay for AdamW")
    parser.add_argument("lr", type=float, help="Peak LR for OneCycle")
    parser.add_argument("nb_epochs", type=int, help="Total number of epochs to train the model.")
    parser.add_argument("check_val_every_n_epoch", type=int, help="How often (in epochs) to run validation.")
    parser.add_argument("num_workers", type=int, help="Number of subprocesses to use for data loading.")
    parser.add_argument("min_epochs", type=int, help="Minimum number of epochs before early stopping can trigger.")
    parser.add_argument("patience", type=int, help="Number of validation checks with no improvement before early stopping triggers.")
    parser.add_argument("--hidden-dims", nargs="+", type=int, required=True)
    parser.add_argument("--use-local-features", action="store_true", help="If true, uses local features.")
    parser.add_argument("--val-ratio", type=float, default=1.0, help="Ratio of validation data kept to compute metrics during training.")
    parser.add_argument("--eval-test", action="store_true", help="If true, train on train+val and evaluate on test, if false, train on train and evaluate on val.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed, workers=True)

    scheme_name = 'sc_sim_loc.pkl' if args.use_local_features else 'sc_sim_non.pkl'
    scheme = load_pickle(os.path.join(args.data_path, scheme_name))
    cat = load_pickle(os.path.join(args.data_path, 'cat.pkl'))
    stations_emb = load_pickle(os.path.join(args.data_path, 'stations_emb.pkl'))
    lines_emb = load_pickle(os.path.join(args.data_path, 'lines_emb.pkl'))
    dataset_config = load_pickle(os.path.join(args.data_path, 'config.pkl'))

    model_config = {
        "input_dim":len(scheme['x'].items()),
        "hidden_dims":args.hidden_dims,
        "dropout":args.dropout,
        "activation":args.activation,
        "output_dim":len(scheme['y'].items())
    }

    eval_config = load_pickle(args.eval_config_path)

    eval_months = ['test'] if args.eval_test else ['val']
    dates = get_dates(args.data_path, eval_months)
    itineraries = load_itineraries_from_dates(dates, args.itineraries_path, show_prog = True)

    run_path, checkpoints_path = setup_run_folder(args, model_config, 'mlp_bc')
    
    data_module = MyDataModule(args.data_path, dataset_config, scheme, batch_size=args.batch_size, num_workers=args.num_workers, eval_test=args.eval_test, 
                val_ratio=args.val_ratio)

    model = BC(model_config, args.weight_decay, args.lr, dataset_config, itineraries, scheme['x'], cat, stations_emb, lines_emb, eval_config, args.use_local_features)
    
    if not args.eval_test:
        logger = CustomCSVLogger(save_dir=run_path, 
                                 train_metrics=["train_loss"],
                                 val_metrics=[f"hor{i}" for i in range(len(eval_config['horizon_obs_bins'])-1)] + 
                                             [f"del{i}" for i in range(len(eval_config['delay_delta_bins'])-1)] + 
                                             ["mae", "mse", "rmse"])

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
        devices=torch.cuda.device_count(),
        strategy="auto",
        accelerator="auto",
        min_epochs=args.min_epochs,
        max_epochs=args.nb_epochs,
        logger=logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=32,
        callbacks=callbacks,
        enable_checkpointing=False
    )
    
    trainer.fit(model, data_module)

    if args.eval_test:
        trainer.test(model, datamodule=data_module)
        print(model.test_results)
        save_results(run_path, model.test_results,eval_config)

if __name__ == "__main__":
    main()