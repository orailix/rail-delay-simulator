import os
import re
import argparse
import random
import torch

import numpy as np

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils.metrics import simulate_and_compute_mae
from torch.utils.data import Dataset, DataLoader

from src.environment.simulation import load_itineraries_from_dates
from src.utils.utils import load_pickle, save_pickle, get_subdir_path, get_dates, setup_run_folder, save_results

def load_sample(idx: int, paths: dict, kept_cols: list) -> tuple:
    """
    Load one sample from disk.

    Args:
        idx (int): Sample index.
        paths (dict): Dict with keys 'x' and 'y' pointing to directories.
        kept_cols (list): Indices of feature columns to keep from x.

    Returns:
        tuple:
            x (numpy.ndarray): Input features.
            y (numpy.ndarray): Target actions as one-hot vectors.
    """
    x = torch.load(get_subdir_path(f"x_{idx}.pt", paths['x']))[:, kept_cols]
    y = torch.load(get_subdir_path(f"y_actions_{idx}.pt", paths['y']))

    return x.numpy(), y.numpy().astype(int)

def build_dataset(split: str, base_path: str, cfg: dict, scheme: dict, ratio: float, n_workers: int) -> tuple:
    """
    Build dataset arrays for a given split.

    Args:
        split (str): Dataset split ("train", "val", "test").
        base_path (str): Root path to the dataset.
        cfg (dict): Config with split sizes (e.g. "train_size").
        scheme (dict): Data scheme.
        ratio (float): Fraction of data to load (0–1).
        n_workers (int): Number of worker threads for loading.

    Returns:
        tuple:
            X (numpy.ndarray): Stacked input features.
            Y (numpy.ndarray): Target labels (class indices).
    """
    total = cfg[f"{split}_size"]
    idxs = np.linspace(0, total-1, int(total*ratio), dtype=int)
    kept_cols = scheme['cols_to_keep']
    paths = {'x': os.path.join(base_path, split, 'x'),
             'y': os.path.join(base_path, split, 'y_actions')}
    results = thread_map(
        lambda i: load_sample(i, paths, kept_cols),
        idxs, max_workers=n_workers, desc=f"Loading {split}", chunksize=128
    )
    Xs, ys = zip(*results)
    X = np.vstack(Xs)
    Y = np.vstack(ys).argmax(axis=1)
    
    return X, Y

class TensorDataset(Dataset):
    """
    Dataset wrapper for tensor loading, used for simulation in the evaluation.

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
    def __len__(self) -> None:
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

def evaluate(base_path: str, dataset_config: dict, scheme: dict, ratio: float, split: str, policy: XGBClassifier, itineraries: dict, cat: dict, stations_emb: dict, 
            lines_emb: dict, eval_config: dict) -> dict:
    """
    Evaluate a policy on a dataset split and compute error metrics.

    Args:
        base_path (str): Root path to the dataset.
        dataset_config (dict): Dataset configuration (e.g. split sizes).
        scheme (dict): Data scheme.
        ratio (float): Fraction of the split to load (0–1).
        split (str): Dataset split ("train", "val", "test").
        policy (XGBClassifier): Model or policy to evaluate.
        itineraries (dict): Trajectory itineraries.
        cat (dict): Categorical feature definitions.
        stations_emb (dict): Station embeddings.
        lines_emb (dict): Line embeddings.
        eval_config (dict): Configuration for evaluation.

    Returns:
        dict: Evaluation results with keys "mae_horizon", "mae_delay", "mae", "mse", "rmse".
    """
    ds = TensorDataset(base_path, dataset_config, scheme, ratio, split)
    dataloader = DataLoader(
            ds,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda x: ([el[0] for el in x], [el[2] for el in x])
        )
    
    mae_delay_list, mae_horizon_list, counter_delay_list, counter_horizon_list = [],[],[],[]
    ssse = 0

    for batch in tqdm(dataloader):
        initial_states, metadatas = batch
        mae_delay, mae_horizon, counter_delay, counter_horizon, sse = simulate_and_compute_mae(initial_states, metadatas, policy, itineraries, True, eval_config['nb_traj'], 
                    eval_config['pred_horizon'], 'median', dataset_config['nb_future_station_reg'], 'cuda:0', scheme['x'], cat, stations_emb, lines_emb, dataset_config, 
                    eval_config['delay_delta_bins'], eval_config['horizon_obs_bins'], 'xgboost')

        mae_delay_list += mae_delay
        mae_horizon_list += mae_horizon
        counter_delay_list += counter_delay
        counter_horizon_list += counter_horizon
        ssse += sse

    sum_del_counter = torch.stack(counter_delay_list).sum(dim=0)
    sum_hor_counter = torch.stack(counter_horizon_list).sum(dim=0)
    
    mae_delay_tensor = torch.stack(mae_delay_list)
    counter_delay_tensor = torch.stack(counter_delay_list)
    
    mae_horizon_tensor = torch.stack(mae_horizon_list)
    counter_horizon_tensor = torch.stack(counter_horizon_list)
    
    mae_delay = (mae_delay_tensor * counter_delay_tensor).sum(dim=0) / sum_del_counter.clamp(min=1)
    mae_horizon = (mae_horizon_tensor * counter_horizon_tensor).sum(dim=0) / sum_hor_counter.clamp(min=1)
    mae = torch.dot(mae_horizon, sum_hor_counter) / sum_hor_counter.sum()

    mse = ssse / sum_hor_counter.sum()
    
    return {
        "mae_horizon": mae_horizon.tolist(),
        "mae_delay": mae_delay.tolist(),
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
    parser.add_argument("itineraries_path", type=str, help="Path to the itineraries.")
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

    scheme = load_pickle(os.path.join(args.data_path, 'sc_sim_loc.pkl'))
    cat = load_pickle(os.path.join(args.data_path, 'cat.pkl'))
    stations_emb = load_pickle(os.path.join(args.data_path, 'stations_emb.pkl'))
    lines_emb = load_pickle(os.path.join(args.data_path, 'lines_emb.pkl'))
    cfg = load_pickle(os.path.join(args.data_path, 'config.pkl'))
    eval_cfg = load_pickle(args.eval_config_path)

    run_path, checkpoints_path = setup_run_folder(args, None, 'xgboost_bc')

    eval_months = ['test'] if args.eval_test else ['val']
    dates = get_dates(args.data_path, eval_months)
    itineraries = load_itineraries_from_dates(dates, args.itineraries_path, show_prog = True)

    X_train, y_train = build_dataset('train', args.data_path, cfg, scheme, args.train_ratio, args.n_workers)
    if args.eval_test:
        X_val, y_val = build_dataset('val', args.data_path, cfg, scheme, args.val_ratio, args.n_workers)
        X_train = np.concatenate([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(scheme['y'].items()),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        device='cuda:0',
        verbosity=2,
        seed=args.seed
    )
    model.fit(X_train, y_train)

    save_pickle(model, os.path.join(checkpoints_path, 'model.pkl'))

    if args.eval_test:
        results = evaluate(args.data_path, cfg, scheme, 1.0, 'test', model, itineraries, cat, stations_emb, lines_emb, eval_cfg)
        print(results)
        save_results(run_path, results, eval_cfg)
    else:
        results = evaluate(args.data_path, cfg, scheme, args.val_ratio, 'val', model, itineraries, cat, stations_emb, lines_emb, eval_cfg)
        print(results)
        save_results(run_path, results, eval_cfg)

if __name__ == "__main__":
    main()