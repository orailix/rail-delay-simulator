import os
import re
import torch
import argparse

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.logger import CustomCSVLogger, MetricsLoggingCallback
from src.utils.utils import load_pickle, save_pickle, get_subdir_path, setup_run_folder, save_results
from src.utils.weightssaver import WeightSaver
from src.models.transformer import Transformer
from src.utils.metrics import compute_mae_regression_tr

class Regression(pl.LightningModule):
    """ 
    Regression implementation with a Transformer policy.

    Args:
        model_config (dict): Configuration dictionary passed to the MLP policy,
            e.g. number of layers, hidden size, activation function.
        weight_decay (float): Weight decay coefficient for the optimizer.
        lr (float): Learning rate for the optimizer.
        scheme (dict): Data scheme.
        eval_config (dict): Configuration for evaluation.
    """
    def __init__(self, model_config: dict, weight_decay: float, lr: float, scheme: dict, eval_config: dict) -> None:
        super(Regression, self).__init__()
        
        self.model = Transformer(**model_config)

        self.sc = scheme
        self.horizon_obs_bins = eval_config['horizon_obs_bins']
        self.delay_delta_bins = eval_config['delay_delta_bins']

        self.weight_decay = weight_decay
        self.lr = lr

    def configure_optimizers(self) -> list:
        """
        Configure AdamW optimizer for the policy.

        Returns:
            list[torch.optim.Optimizer]: List with one AdamW optimizer.
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return [optimizer]

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked mean squared error loss.

        Args:
            outputs (torch.Tensor): Model predictions of shape (batch_size, seq_len, nb_future_stations).
            labels (torch.Tensor): Ground-truth targets of shape (batch_size, seq_len, nb_future_stations).
            loss_mask (torch.Tensor): Boolean mask, True for positions to ignore of shape(batch_size, seq_len, nb_future_stations).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        criterion = nn.MSELoss().to(self.device)
        valid_outputs = outputs[~loss_mask]
        valid_labels = labels[~loss_mask]
        return criterion(valid_outputs, valid_labels)
        
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        One training step.

        Args:
            batch (tuple): (inputs, targets, loss_mak).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the batch.
        """
        x, y, padding_mask, loss_mask = batch
        outputs = self.model(x, padding_mask=padding_mask)
        loss = self.compute_loss(outputs, y, loss_mask)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """
        Initialize containers for validation data and predictions at the start of each epoch.

        Returns:
            None
        """
        self.x_val = []
        self.preds_val = []
        self.targets_val = []
        self.masks_val = []

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Run one validation step: forward pass and store inputs, predictions, targets, and masks.

        Args:
            batch (tuple): Input batch as (x, y, loss_mask).
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        x, y, padding_mask, loss_mask = batch
        outputs = self.model(x, padding_mask=padding_mask)

        self.x_val.append(x.detach().cpu())
        self.preds_val.append(outputs.detach().cpu())
        self.targets_val.append(y.detach().cpu())
        self.masks_val.append(loss_mask.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        """
        Compute validation metrics at the end of an epoch and log results.

        Returns:
            None
        """
        mae_delay, mae_horizon, counter_delay, counter_horizon, mse = compute_mae_regression_tr(self.preds_val, self.targets_val, self.x_val, self.masks_val, self.sc, self.delay_delta_bins, self.horizon_obs_bins)

        for i in range(len(mae_horizon)):
            self.log(f"hor{i}",mae_horizon[i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        for i in range(len(mae_delay)):
            self.log(f"del{i}",mae_delay[i], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        mae = np.dot(mae_horizon, counter_horizon) / counter_horizon.sum()
        
        self.log(f"mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log(f"mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log(f"rmse", np.sqrt(mse), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

    def on_test_epoch_start(self) -> None:
        """
        Initialize containers for test data and predictions at the start of each epoch.

        Returns:
            None
        """
        self.x_test = []
        self.preds_test = []
        self.targets_test = []
        self.masks_test = [] 

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Run one test step: forward pass and store inputs, predictions, targets, and masks.

        Args:
            batch (tuple): Input batch as (x, y, loss_mask).
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        x, y, padding_mask, loss_mask = batch
        outputs = self.model(x, padding_mask=padding_mask)

        self.x_test.append(x.detach().cpu())
        self.preds_test.append(outputs.detach().cpu())
        self.targets_test.append(y.detach().cpu())
        self.masks_test.append(loss_mask.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """
        Compute test metrics at the end of an epoch and store the results in self.test_results.

        Returns:
            None
        """
        mae_delay, mae_horizon, counter_delay, counter_horizon, mse = compute_mae_regression_tr(self.preds_test, self.targets_test, self.x_test, self.masks_test, self.sc, self.delay_delta_bins, self.horizon_obs_bins)  
        mae = np.dot(mae_horizon, counter_horizon) / counter_horizon.sum()

        self.test_results = {
            "mae_horizon": mae_horizon.tolist(),
            "mae_delay": mae_delay.tolist(),
            "mae": mae.item(),
            "mse":mse.item(),
            "rmse":np.sqrt(mse).item(),
        }

def transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply sign-invariant square-root and normalization.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor.
    """
    x = torch.sign(x) * torch.sqrt(torch.abs(x))  # Sign-invariant square-root
    x = x / 6  # Apply normalization by the precomputed std
    return x

def inverse_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Invert the transform by undoing normalization and square-root scaling.

    Args:
        x (torch.Tensor): Transformed tensor.

    Returns:
        torch.Tensor: Original-scale tensor.
    """
    return torch.sign(x)*((x*6)**2)

class TensorDataset(Dataset):
    """
    Dataset wrapper for tensor loading.

    Args:
        base_path (str): Root directory of the dataset.
        config (dict): Config dict containing split sizes (e.g. "train_size").
        scheme (dict): Data scheme.
        ratio (float): Fraction of the split to keep (0–1).
        split (str): Dataset split ("train", "val", "test").
        apply_transform (bool, optional): Whether to apply transform to targets. Default is True.
    """
    def __init__(self, base_path: str, config: dict, scheme: dict, ratio: float, split: str, apply_transform: bool = True) -> None:
        self.apply_transform = apply_transform
        self.kept_cols = scheme['cols_to_keep']
        self.ratio = ratio
        total_data = config[f"{split}_size"]
        self.x_path = os.path.join(base_path, split, 'x')
        self.y_path = os.path.join(base_path, split, 'y_delays')
        self.mapper = torch.linspace(0, total_data - 1, int(total_data*ratio), dtype=int)
        self.len = self.mapper.shape[0]
        self.future_stations_emb_0_indices = [scheme['x'][s] for s in scheme['x'].keys() if re.match(r"^FUTURE_STATIONS_.*_embedding_0$", s)]

    def __len__(self) -> int:
        """
        Dataset size.

        Returns:
            int: Number of samples.
        """

    def __getitem__(self, idx: int) -> tuple:
        """
        Load one sample from disk, applies transform on target (relative delay instead of absolute) if requested.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (x, y, ph_mask)
                x (torch.Tensor): Input features.
                y (torch.Tensor): Target delays (transformed if enabled).
                ph_mask (torch.Tensor): Mask for padded future stations.
        """
        idx = self.mapper[idx]
        x = torch.load(get_subdir_path(f'x_{idx}.pt',self.x_path))[:, self.kept_cols]
        y = torch.load(get_subdir_path(f'y_delays_{idx}.pt',self.y_path))
        if self.apply_transform:
            y = transform(inverse_transform(y) - inverse_transform(x[:,4]).unsqueeze(-1).repeat(1, 15))
        ph_mask = x[:, self.future_stations_emb_0_indices] == 0
        return x, y, ph_mask

def collate_fn(batch: list) -> tuple:
    """
    Pad variable-length sequences and create a padding mask.

    Args:
        batch (list): List of tuples (x, y, ph_mask).

    Returns:
        tuple: (x_padded, y_padded, padding_mask, loss_mask)
            x_padded (torch.Tensor): Padded inputs, shape (batch, max_len, ...).
            y_padded (torch.Tensor): Padded targets, shape (batch, max_len, ...).
            padding_mask (torch.Tensor): Bool mask of shape (batch, max_len), True for padding.
            loss_mask (torch.Tensor): Bool mask of shape (batch, max_len, ...), True where loss should be ignored.
    """
    x = [pair[0] for pair in batch] 
    y = [pair[1] for pair in batch]
    ph_mask = [pair[2] for pair in batch]
    max_len = max(sequence.shape[0] for sequence in x)
    
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    ph_mask_padded = pad_sequence(ph_mask, batch_first=True, padding_value=0)
    
    lengths = torch.tensor([seq.shape[0] for seq in x])
    padding_mask = torch.arange(x_padded.shape[1]) >= lengths.unsqueeze(1)
    
    loss_mask = padding_mask.unsqueeze(-1).expand_as(y_padded) | ph_mask_padded

    return x_padded, y_padded, padding_mask, loss_mask

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

        self.train_ds = TensorDataset(self.base_path, self.config, self.scheme, 1.0, 'train', apply_transform = True)
        if self.eval_test:
            self.val_ds = TensorDataset(self.base_path, self.config, self.scheme, 1.0, 'val', apply_transform = True)
            self.train_ds = ConcatDataset([self.train_ds, self.val_ds])
            self.test_ds  = TensorDataset(self.base_path, self.config, self.scheme, 1.0, 'test', apply_transform = False)
        else:
            self.val_ds = TensorDataset(self.base_path, self.config, self.scheme, val_ratio, 'val', apply_transform = False)

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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
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
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
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
    parser.add_argument("eval_config_path", type=str, help="Path to the evaluation config.")
    parser.add_argument("d_model", type=int, help="Dimensionality of the transformer embeddings (model hidden size).")
    parser.add_argument("nhead", type=int, help="Number of attention heads in each multi-head self-attention layer.")
    parser.add_argument("dim_feedforward", type=int, help="Hidden layer size of the position-wise feedforward sublayer.")
    parser.add_argument("dropout", type=float, help="Dropout probability for all dropout layers (0.0–1.0).")
    parser.add_argument("activation", type=str, choices=["relu","gelu","tanh"], help="Activation function to use in the feedforward layers.")
    parser.add_argument("num_layers", type=int, help="Number of stacked TransformerEncoder layers.")
    parser.add_argument("batch_size", type=int, help="Number of samples per training batch.")
    parser.add_argument("weight_decay", type=float, help="Weight decay for AdamW")
    parser.add_argument("lr", type=float, help="AdamW LR")
    parser.add_argument("nb_epochs", type=int, help="Total number of epochs to train the model without early stopping.")
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

    scheme = load_pickle(os.path.join(args.data_path, 'sc_reg_non.pkl'))
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

    run_path, checkpoints_path = setup_run_folder(args, model_config, 'tr_reg')

    eval_config = load_pickle(args.eval_config_path)
    
    data_module = MyDataModule(args.data_path, dataset_config, scheme, batch_size=args.batch_size, num_workers=args.num_workers, eval_test=args.eval_test, val_ratio=args.val_ratio)

    model = Regression(model_config, args.weight_decay, args.lr, scheme, eval_config)
    
    if not args.eval_test:
        logger = CustomCSVLogger(save_dir=run_path, 
                                 train_metrics=["train_loss"],
                                 val_metrics=[f"hor{i}" for i in range(len(eval_config['horizon_obs_bins'])-1)] + 
                                             [f"del{i}" for i in range(len(eval_config['delay_delta_bins'])-1)] + 
                                             ["mae","mse","rmse"])

        callbacks = [
            WeightSaver(save_path=checkpoints_path, 
                        target="model", 
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
        precision='16',
        callbacks=callbacks,
        enable_checkpointing=False,
        num_sanity_val_steps=0
    )
    
    trainer.fit(model, data_module)

    if args.eval_test:
        trainer.test(model, datamodule=data_module)
        print(model.test_results)
        save_results(run_path, model.test_results,eval_config)

if __name__ == "__main__":
    main()