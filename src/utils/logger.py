import csv
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger, CSVLogger
from pathlib import Path

class CustomCSVLogger(Logger):
    """
    Logger that writes training and validation metrics to a CSV file.

    HOW TO USE:
    logger = CustomCSVLogger(save_dir="logs", metrics=["train_loss", "train_accuracy","val_loss", "val_accuracy"])
    metrics_callback = MetricsLoggingCallback(logger)
    trainer = Trainer(logger=logger, max_epochs=5, callbacks=[metrics_callback], enable_checkpointing=False)

    Args:
        save_dir (str): Directory where the log file will be saved.
        train_metrics (list): Names of training metrics to log.
        val_metrics (list): Names of validation metrics to log.
    """
    def __init__(self, save_dir: str, train_metrics: list, val_metrics: list) -> None:        
        super().__init__()
        self._save_dir = save_dir
        
        os.makedirs(self._save_dir, exist_ok=True)
        self.log_file = os.path.join(self._save_dir, "metrics_log.csv")
        
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.metric_headers = ["epoch"] + train_metrics + val_metrics
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.metric_headers)
    
    @property
    def name(self) -> str:
        """
        Logger name identifier.

        Returns:
            str: Logger name.
        """
        return ""

    @property
    def version(self) -> str:
        """
        Logger version identifier.

        Returns:
            str: Logger version.
        """
        return ""

    @property
    def save_dir(self) -> str:
        """
        Directory where logs are saved.

        Returns:
            str: Path to save directory.
        """
        return self._save_dir

    def log_hyperparams(self, params: dict) -> None:
        """
        Log hyperparameters (not implemented).

        Args:
            params (dict): Hyperparameters to log.

        Returns:
            None
        """
        pass
    
    def log_metrics(self, metrics: dict, step: int = None) -> None:
        """
        Log metrics (not implemented).

        Args:
            metrics (dict): Dictionary of metrics.
            step (int, optional): Training step. Default is None.

        Returns:
            None
        """
        pass
    
    def log_epoch_metrics(self, trainer, pl_module) -> None:
        """
        Log metrics at the end of each epoch.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: LightningModule being trained.

        Returns:
            None
        """
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        row = [epoch] + [metrics.get(m, "").item() if m in metrics else "" for m in self.train_metrics]

        if (epoch + 1) % trainer.check_val_every_n_epoch == 0:
            row += [metrics.get(m, "").item() if m in metrics else "" for m in self.val_metrics]
        else:
            row += ["" for m in self.val_metrics]

        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
    
    @property
    def experiment(self) -> None:
        """
        Experiment handle (not used).

        Returns:
            None
        """
        return None

    def save(self) -> None:
        """
        Save logger state (not implemented).

        Returns:
            None
        """
        pass

    def finalize(self, status: str) -> None:
        """
        Finalize logger (not implemented).

        Args:
            status (str): Final training status.

        Returns:
            None
        """
        pass

class MetricsLoggingCallback(pl.Callback):
    """
    Callback to log metrics at the end of each training epoch.
    
    Args:
        logger (CustomCSVLogger): Logger instance.
    """
    def __init__(self, logger: CustomCSVLogger) -> None:
        super().__init__()
        self.logger = logger
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Trigger logging at the end of each epoch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: LightningModule being trained.

        Returns:
            None
        """
        self.logger.log_epoch_metrics(trainer, pl_module)