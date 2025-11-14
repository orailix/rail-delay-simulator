import os
import torch

import pytorch_lightning as pl

class WeightSaver(pl.Callback):
    """
    Save top-k model weights during training based on a monitored metric.

    Args:
        save_path (str): Directory where weights are saved.
        target (str, optional): Attribute path to the module to save (e.g., "policy").
        keep_module_prefix (bool, optional): If False, strip "module." prefixes from keys.
        top_k (int, optional): Number of best models to keep. Defaults to 5.
        monitor (str, optional): Metric name to monitor. Defaults to "loss".
        mode (str, optional): "min" for lower is better, "max" for higher is better.
    """
    def __init__(self, save_path: str, target: str = None, keep_module_prefix: bool = False,
                 top_k: int = 5, monitor: str = "loss", mode: str = "min") -> None:
        super().__init__()
        self.save_path = save_path
        self.target = target
        self.keep_prefix = keep_module_prefix
        self.top_k = top_k
        self.monitor = monitor
        self.mode = mode
        self.is_better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        self.saved = []
        os.makedirs(save_path, exist_ok=True)

    def _current_score(self, trainer: pl.Trainer) -> float | None:
        """
        Get the current monitored metric as a float.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer.

        Returns:
            float | None: Metric value if available, else None.
        """
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return None
        return float(score.detach().cpu())

    def _save_state(self, trainer: pl.Trainer, module: torch.nn.Module) -> str:
        """
        Save the target module's state_dict for the current epoch.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer.
            module (torch.nn.Module): Module to serialize.

        Returns:
            str: Path to the saved checkpoint file.
        """
        fname = f"model_epoch_{trainer.current_epoch}.pt"
        fpath = os.path.join(self.save_path, fname)
        torch.save(module.state_dict(), fpath)
        return fpath

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Save weights if this is a validation epoch and the monitored metric is enough to be in the top-k epochs.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer.
            pl_module (pl.LightningModule): LightningModule being trained.

        Returns:
            None
        """
        if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch != 0: # dont trigger when not on a validation epoch
            return
        
        if not trainer.is_global_zero:          # multi-GPU safety
            return

        score = self._current_score(trainer)
        if score is None:
            return                              # metric not available yet

        module = pl_module
        if self.target:
            for attr in self.target.split("."):
                module = getattr(module, attr)

        if len(self.saved) < self.top_k:
            path = self._save_state(trainer, module)
            self.saved.append((score, path))
            return

        worst_idx = max(range(len(self.saved)),
                        key=lambda i: self.saved[i][0]) if self.mode == "min" else \
                     min(range(len(self.saved)),
                        key=lambda i: self.saved[i][0])
        worst_score, worst_path = self.saved[worst_idx]

        if self.is_better(score, worst_score):
            path = self._save_state(trainer, module)
            os.remove(worst_path)
            self.saved[worst_idx] = (score, path)