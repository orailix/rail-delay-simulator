import os
import re
import argparse
import random
import torch

import numpy as np
import pandas as pd

from tqdm.contrib.concurrent import thread_map
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.utils.utils import load_pickle, get_subdir_path, save_results, setup_run_folder, save_pickle
from src.utils.metrics import compute_mae_regression_mlp

def transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply sign-invariant square-root and normalization.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor.
    """
    x = torch.sign(x) * torch.sqrt(torch.abs(x))
    return x / 6

def inverse_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Invert the transform by undoing normalization and square-root scaling.

    Args:
        x (torch.Tensor): Transformed tensor.

    Returns:
        torch.Tensor: Original-scale tensor.
    """
    return torch.sign(x) * ((x * 6) ** 2)

def load_sample(idx: int, paths: dict, kept_cols: list, fut_idx: list, apply_transform: bool) -> tuple:
    """
    Load one sample from disk.

    Args:
        idx (int): Sample index.
        paths (dict): Dict with keys 'x' and 'y' pointing to directories.
        kept_cols (list): Indices of feature columns to keep from x.
        fut_idx (list): Indices of future station embedding columns.
        apply_transform (bool): Whether to apply transform to the targets.

    Returns:
        tuple: (x, y, mask)
            x (numpy.ndarray): Input features.
            y (numpy.ndarray): Target delays (possibly transformed).
            mask (numpy.ndarray): Mask indicating padded future stations.
    """
    x = torch.load(get_subdir_path(f"x_{idx}.pt", paths['x']))[:, kept_cols]
    y = torch.load(get_subdir_path(f"y_delays_{idx}.pt", paths['y']))
    if apply_transform:
        y = transform(inverse_transform(y) - inverse_transform(x[:, 4]).unsqueeze(-1).repeat(1, 15))
    mask = x[:, fut_idx] == 0
    return x.numpy(), y.numpy(), mask.numpy()

def build_dataset(split: str, base_path: str, cfg: dict, scheme: dict, ratio: float, n_workers: int, apply_transform: bool) -> tuple:
    """
    Build dataset arrays for a given split.

    Args:
        split (str): Dataset split ("train", "val", "test").
        base_path (str): Root path to the dataset.
        cfg (dict): Config with split sizes (e.g. "train_size").
        scheme (dict): Data scheme.
        ratio (float): Fraction of data to load (0–1).
        n_workers (int): Number of worker threads for loading.
        apply_transform (bool): Whether to apply transform to targets.

    Returns:
        tuple: (X, Y, mask)
            X (numpy.ndarray): Concatenated input features.
            Y (numpy.ndarray): Concatenated targets.
            mask (numpy.ndarray): Concatenated masks for future stations.
    """
    total = cfg[f"{split}_size"]
    idxs = np.linspace(0, total - 1, int(total * ratio), dtype=int)
    kept_cols = scheme['cols_to_keep']
    fut_idx = [scheme['x'][k] for k in scheme['x'] if re.match(r"^FUTURE_STATIONS_.*_embedding_0$", k)]
    paths = {
        'x': os.path.join(base_path, split, 'x'),
        'y': os.path.join(base_path, split, 'y_delays')
    }
    results = thread_map(
        lambda i: load_sample(i, paths, kept_cols, fut_idx, apply_transform),
        idxs,
        max_workers=n_workers,
        desc=f"Loading {split}",
        chunksize=128
    )
    xs, ys, masks = zip(*results)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(masks, axis=0)
def evaluate(model, X: np.ndarray, y: np.ndarray, masks: np.ndarray, scheme: dict, eval_cfg: dict) -> dict:
    """
    Evaluate a regression model and compute error metrics.

    Args:
        model: Trained regression model with a `predict` method.
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Ground-truth targets.
        masks (numpy.ndarray): Padding or station masks.
        scheme (dict): Feature scheme used for evaluation.
        eval_cfg (dict): Evaluation config (bins and horizon settings).

    Returns:
        dict: Evaluation results with keys:
            - "mae_horizon" (list[float])
            - "mae_delay" (list[float])
            - "mae" (float)
            - "mse" (float)
            - "rmse" (float)
    """

    preds = model.predict(X)
    mae_d, mae_h, c_d, c_h, mse = compute_mae_regression_mlp(
        [torch.tensor(preds)], [torch.tensor(y)], [torch.tensor(X)], [torch.tensor(masks)],
        scheme, eval_cfg['delay_delta_bins'], eval_cfg['horizon_obs_bins']
    )
    mae = np.dot(mae_h, c_h) / c_h.sum()

    return {
        "mae_horizon": mae_h.tolist(),
        "mae_delay": mae_d.tolist(),
        "mae": mae.item(),
        "mse":mse.item(),
        "rmse":np.sqrt(mse).item(),
    }

def main() -> None:
    """
    Entry point for training and evaluating an XGBoost model.

    Parses CLI arguments, loads data and configurations, builds the dataset,
    trains the XGBoost classifier, and evaluates it on validation or test data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="A unique name/ID for this training run (used for logging & checkpoints).")
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("eval_config_path", type=str, help="Path to the evaluation config.")
    parser.add_argument('n_workers', type=int, help='Number of parallel worker processes')
    parser.add_argument('n_estimators', type=int, help='Number of boosting rounds')
    parser.add_argument('max_depth', type=int, help='Maximum tree depth for base learners')
    parser.add_argument('learning_rate', type=float, help='Step size shrinkage used in update to prevent overfitting')
    parser.add_argument('subsample', type=float, help='Subsample ratio of the training instances')
    parser.add_argument('colsample_bytree', type=float, help='Subsample ratio of columns when constructing each tree')
    parser.add_argument('min_child_weight', type=float, help='Minimum sum of instance weights needed in a child.')
    parser.add_argument('gamma', type=float, help='Minimum loss reduction required to make a split.')
    parser.add_argument('reg_alpha', type=float, help='L1 regularisation term on leaf weights.')
    parser.add_argument('reg_lambda', type=float, help='L2 regularisation term on leaf weights.')
    parser.add_argument("--train-ratio", type=float, default=1.0, help="Ratio of train data kept.")
    parser.add_argument("--val-ratio", type=float, default=1.0, help="Ratio of validation data kept.")
    parser.add_argument("--eval-test", action="store_true", help="If true, train on train+val and evaluate on test, if false, train on train and evaluate on val.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    scheme = load_pickle(os.path.join(args.data_path, 'sc_reg_loc.pkl'))
    cfg = load_pickle(os.path.join(args.data_path, 'config.pkl'))
    eval_cfg = load_pickle(args.eval_config_path)

    run_path, checkpoints_path = setup_run_folder(args, None, 'xgboost_reg')

    X_train, y_train, _ = build_dataset('train', args.data_path, cfg, scheme, args.train_ratio, args.n_workers, True)
    if args.eval_test:
        X_val, y_val, _ = build_dataset('val', args.data_path, cfg, scheme, args.val_ratio, args.n_workers, True)
        X_train = np.concatenate([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])
        X_test, y_test, masks_test = build_dataset('test', args.data_path, cfg, scheme, 1.0, args.n_workers, False)
    else:
        X_val, y_val, masks_val = build_dataset('val', args.data_path, cfg, scheme, args.val_ratio, args.n_workers, False)

    model = MultiOutputRegressor(
        XGBRegressor(
            objective='reg:squarederror',
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_weight=args.min_child_weight,
            gamma=args.gamma,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            n_jobs=args.n_workers,
            device='cuda:0',
            verbosity=2,
            seed=args.seed
        )
    )
    model.fit(X_train, y_train)

    save_pickle(model, os.path.join(checkpoints_path, 'model.pkl'))

    if args.eval_test:
        results = evaluate(model, X_test, y_test, masks_test, scheme, eval_cfg)
        print(results)
        save_results(run_path, results, eval_cfg)
    else:
        results = evaluate(model, X_val, y_val, masks_val, scheme, eval_cfg)
        print(results)
        save_results(run_path, results, eval_cfg)

if __name__ == "__main__":
    main()