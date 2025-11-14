import torch
import os
import re
import argparse
import pathlib

import pandas as pd
import numpy as np

from tqdm import tqdm
from itertools import repeat
from scipy.spatial.distance import cdist

from src.utils.utils import load_pickle, save_pickle, get_subdir_path
from src.data.raw_data_processing import get_optimal_column_order

def check_parameters(source_config: dict, nb_past: int, nb_future: int, num_encoding_func: callable, cat_enc: str, 
            stations_enc: dict | None, lines_enc: dict | None, time_feat_enc: str | None) -> None:
    """
    Validate parameters.

    Args:
        source_config (dict): Configuration with keys 'nb_past_stations' and 'nb_future_stations'
            defining the maximum supported context sizes.
        nb_past (int): Requested number of past stations; must be > 0 and ≤ source_config['nb_past_stations'].
        nb_future (int): Requested number of future stations; must be > 0 and ≤ source_config['nb_future_stations'].
        num_encoding_func (callable): Function used to encode/transform numerical columns.
        cat_enc (str): Categorical encoding strategy. Supported: 'one_hot'.
        stations_enc (dict | None): Optional mapping for station identifiers; None to skip.
        lines_enc (dict | None): Optional mapping for line identifiers; None to skip.
        time_feat_enc (str | None): Time feature encoding strategy. Supported: 'cyclical' or None.

    Returns:
        None

    Raises:
        AssertionError: If any argument fails validation (type, range, or supported options).
    """
    max_past = source_config['nb_past_stations']
    max_future = source_config['nb_future_stations']
    
    # Validate numerical parameters
    assert isinstance(nb_past, int) and max_past >= nb_past > 0, f"nb_past must be a positive integer and equal or bellow {max_past}"
    assert isinstance(nb_future, int) and max_future >= nb_future > 0, f"nb_future must be a positive integer and equal or bellow {max_future}"
    
    # Validate encoding function
    assert callable(num_encoding_func), "num_encoding_func must be callable"
    
    # Validate categorical encoding
    assert cat_enc in ['one_hot'], "cat_enc must be 'one_hot'"
    
    # Validate optional dictionaries
    assert isinstance(stations_enc, (dict, type(None))), "stations_enc must be a dict or None"
    assert isinstance(lines_enc, (dict, type(None))), "lines_enc must be a dict or None"
    
    # Validate time feature encoding
    assert time_feat_enc in [None, 'cyclical'], "time_feat_enc must be None or 'cyclical'"

def drop_out_of_scope_cols(df: pd.DataFrame, source_config: dict, nb_past: int, nb_future: int) -> pd.DataFrame:
    """
    Drop past/future context columns that exceed the requested scope.

    Args:
        df (pandas.DataFrame): Input DataFrame with past/future context columns.
        source_config (dict): Configuration with keys 'nb_past_stations' and 'nb_future_stations'
            defining the maximum available context sizes.
        nb_past (int): Number of past stations to keep.
        nb_future (int): Number of future stations to keep.

    Returns:
        pandas.DataFrame: DataFrame with only the requested number of past/future context columns.
    """
    past_col_prefixes = ['PAST_STATIONS', 'PAST_PLANNED_TIME_NUM', 'PAST_DELAYS', 'PAST_TYPES', 'PAST_LINES']
    future_col_prefixes = ['FUTURE_STATIONS', 'FUTURE_PLANNED_TIME_NUM', 'FUTURE_DELAYS', 'FUTURE_TYPES', 'FUTURE_LINES']

    cols_to_drop = []
    for prefix in past_col_prefixes:
        for i in range(nb_past + 1, source_config['nb_past_stations'] + 1):
            cols_to_drop.append(f'{prefix}_{i}')
            
    for prefix in future_col_prefixes:
        for i in range(nb_future + 1, source_config['nb_future_stations'] + 1):
            cols_to_drop.append(f'{prefix}_{i}')

    df = df.drop(columns = cols_to_drop)

    return df

def apply_num_encoding_func(df: pd.DataFrame, func: callable, nb_past: int, nb_future: int) -> pd.DataFrame:
    """
    Apply a numerical encoding function to past and future time/delay columns.

    Args:
        df (pandas.DataFrame): Input DataFrame with numerical past/future columns.
        func (callable): Function applied elementwise to each selected column.
        nb_past (int): Number of past stations to process.
        nb_future (int): Number of future stations to process.

    Returns:
        pandas.DataFrame: DataFrame with transformed numerical columns.
    """
    past_num_prefixes = ['PAST_PLANNED_TIME_NUM', 'PAST_DELAYS']
    future_num_prefixes = ['FUTURE_PLANNED_TIME_NUM', 'FUTURE_DELAYS']

    num_cols = []
    for prefix in past_num_prefixes:
        for i in range(1, nb_past + 1):
            num_cols.append(f'{prefix}_{i}')

    for prefix in future_num_prefixes:
        for i in range(1, nb_future + 1):
            num_cols.append(f'{prefix}_{i}')

    for col in num_cols:
        df[col] = func(df[col])

    return df

def apply_cat_encoding(df: pd.DataFrame, cat_enc: str, nb_past: int, nb_future: int) -> tuple:
    """
    Apply categorical encoding to relation, action, time, and type features.

    Args:
        df (pandas.DataFrame): Input DataFrame with categorical columns to encode.
        cat_enc (str): Categorical encoding strategy. Supported: 'one_hot'.
        nb_past (int): Number of past stations (used to locate type-related columns).
        nb_future (int): Number of future stations (used to locate type-related columns).

    Returns:
        tuple: (df_encoded, category_dict) where
            df_encoded (pandas.DataFrame): DataFrame with one-hot encoded categorical columns.
            category_dict (dict): Dictionary of categories used for encoding.
    """
    if cat_enc == 'one_hot':
        category_dict = {
            'RELATION_TYPE': ['ICE','INT','IC','EURST','THAL','L','P','EXTRA','TGV','CHARTER', 'ICT','IZY'],
            'action': ['same', 'next1', 'next2'],
            'day_of_week': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
            'TYPES': ['D', 'P', 'A']
        }
        
        # List to keep track of columns to be one-hot encoded
        cols_to_encode = []
    
        # Prepare the columns for encoding based on the dictionary
        for key, categories in category_dict.items():
            if key in df.columns:
                # If the key is a direct column in the DataFrame, process it normally
                df[key] = pd.Categorical(df[key], categories=categories)
                cols_to_encode.append(key)
            elif any(key in col for col in df.columns):
                # If the key is a substring of column names in the DataFrame, process all such columns
                matching_columns = [col for col in df.columns if key in col]
                for col in matching_columns:
                    df[col] = pd.Categorical(df[col], categories=categories)
                    cols_to_encode.append(col)
            else:
                print(f'Column or pattern {key} not found in the DataFrame.')
    
        # Apply one-hot encoding to all relevant columns at once
        df_encoded = pd.get_dummies(df, columns=cols_to_encode, dummy_na=False, dtype=int)

    return df_encoded, category_dict

def apply_stations_encoding(df: pd.DataFrame, stations_enc: dict | None) -> pd.DataFrame:
    """
    Apply station encoding or embeddings and drop raw station columns.

    Args:
        df (pandas.DataFrame): Input DataFrame with station-related columns.
        stations_enc (dict | None): Mapping of station identifiers to embeddings,
            or None to skip encoding.

    Returns:
        pandas.DataFrame: DataFrame with station columns replaced by embeddings (if provided).
    """
    stations_cols = [c for c in df.columns if 'STATIONS' in c]
    
    if stations_enc is None:
        pass
    elif isinstance(stations_enc, dict):
        df = inject_embeddings(df, stations_enc, stations_cols)

    df = df.drop(columns = stations_cols)
    
    return df

def apply_lines_encoding(df: pd.DataFrame, lines_enc: dict | None) -> pd.DataFrame:
    """
    Apply line encoding or embeddings and drop raw line columns.

    Args:
        df (pandas.DataFrame): Input DataFrame with line-related columns.
        lines_enc (dict | None): Mapping of line identifiers to embeddings,
            or None to skip encoding.

    Returns:
        pandas.DataFrame: DataFrame with line columns replaced by embeddings (if provided).
    """
    lines_cols = [c for c in df.columns if 'LINES' in c]
    
    if lines_enc is None:
        pass
    elif isinstance(lines_enc, dict):
        df = inject_embeddings(df, lines_enc, lines_cols)

    df = df.drop(columns = lines_cols)
    
    return df

def inject_embeddings(df: pd.DataFrame, embeddings_dict: dict, cols: list) -> pd.DataFrame:
    """
    Replace categorical identifiers with their embedding vectors.

    Args:
        df (pandas.DataFrame): Input DataFrame containing categorical identifier columns.
        embeddings_dict (dict): Mapping from identifier to embedding vector (all vectors must have same length).
        cols (list): List of column names in df to replace with embeddings.

    Returns:
        pandas.DataFrame: DataFrame with embedding columns appended for each identifier column.
    """
    nb_dims = len(next(iter(embeddings_dict.values()))) # get the len of the embeddings
    new_columns = []
    for col in cols:
        unique_names = df[col].unique()
        unique_embeddings = np.array([embeddings_dict.get(x)#, np.zeros(nb_dims))
                                      for x in unique_names], 
                                     dtype='float32')
    
        name_to_index = {name: idx for idx, name in enumerate(unique_names)}
        indices = np.array([name_to_index[name] for name in df[col]])
        
        embeddings = unique_embeddings[indices]
        new_columns_names = [f'{col}_embedding_{i}' for i in range(nb_dims)]
        new_columns.append(pd.DataFrame(embeddings, columns=new_columns_names, index=df.index))

    # Concatenate all the new columns into the original DataFrame
    df = pd.concat([df] + new_columns, axis=1)

    return df

def apply_time_feat_encoding(df: pd.DataFrame, time_feat_enc: str | None) -> pd.DataFrame:
    """
    Encode temporal features with optional cyclical encoding.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'hour', 'day_of_year', and 'DATETIME' columns.
        time_feat_enc (str | None): Encoding strategy. Supported: 'cyclical' or None.

    Returns:
        pandas.DataFrame: DataFrame with time features encoded and raw columns dropped.
    """
    if time_feat_enc is None:
        pass
    elif time_feat_enc == 'cyclical':  
        frequencies = [1, 2, 4]
        for freq in frequencies:
            df[f'hour_sin_{freq}'] = np.sin(freq * 2 * np.pi * df['hour'] / 24)
            df[f'hour_cos_{freq}'] = np.cos(freq * 2 * np.pi * df['hour'] / 24)
        # Cyclical encoding of the time of year (365-day cycle)
        for freq in frequencies:
            df[f'year_sin_{freq}'] = np.sin(freq * 2 * np.pi * df['day_of_year'] / 365)
            df[f'year_cos_{freq}'] = np.cos(freq * 2 * np.pi * df['day_of_year'] / 365)
        
    df = df.drop(columns=['hour', 'day_of_year', 'DATETIME'])

    return df

def apply_local_features(df: pd.DataFrame, radiuses: list = [0.1, 0.3, 0.6, 1.0, 2.0], normalizers: list = [2.0, 7.0, 22.0, 64.0, 262.0]) -> pd.DataFrame:
    """
    Compute local neighbor-based features for each state time group (meaning all trains 
    present on the network at a given STATE_TIME).
    For various radiuses, count for each train the number of neighbours in the embedding
    space and compute their mean delay. 

    Args:
        df (pandas.DataFrame): Input DataFrame containing station embeddings and delays.
        radiuses (list): Distance thresholds for neighborhood definitions.
        normalizers (list): Normalization factors for neighbor counts corresponding to radiuses.

    Returns:
        pandas.DataFrame: DataFrame with added local feature columns:
            - count_rX: Normalized neighbor counts within radius X.
            - mean_delay_rX: Mean delay of neighbors within radius X.
    """
    df = df.reset_index(drop=True)

    n_r = len(radiuses)

    past = df[[f'PAST_STATIONS_1_embedding_{i}' for i in range(8)]].to_numpy()
    fut = df[[f'FUTURE_STATIONS_1_embedding_{i}' for i in range(8)]].to_numpy()
    coords = (past + fut) / 2.0
    delays = df['PAST_DELAYS_1'].to_numpy()

    counts = np.zeros((len(df), n_r), dtype=int)
    mean_delay = np.zeros((len(df), n_r), dtype=float)

    for idx in tqdm(df.groupby('STATE_TIME').indices.values(), desc='local_features', unit='group'):
        d = cdist(coords[idx], coords[idx], metric='euclidean')
        np.fill_diagonal(d, np.inf)

        for k, r in enumerate(radiuses):
            mask = d <= r
            cnt = mask.sum(1)
            sm = mask.dot(delays[idx])
            mdelay = np.divide(sm, cnt, out=np.zeros_like(sm, dtype=float), where=cnt > 0)

            counts[idx, k] = cnt
            mean_delay[idx, k] = mdelay

    for k, r in enumerate(radiuses):
        df[f'count_r{r}'] = counts[:, k] / normalizers[k]
        df[f'mean_delay_r{r}'] = mean_delay[:, k]

        print(k, (counts[:, k] / normalizers[k]).mean(), mean_delay[:, k].mean())

    return df

def sample_by_state_time(df: pd.DataFrame, ratio_kept: float, random_state: int = 42) -> pd.DataFrame:
    """
    Subsample the dataset by randomly selecting STATE_TIME groups.

    Args:
        df (pandas.DataFrame): Input DataFrame with a 'STATE_TIME' column.
        ratio_kept (float): Fraction of unique STATE_TIME groups to keep, in [0, 1].
        random_state (int): Seed for reproducibility. Default is 42.

    Returns:
        pandas.DataFrame: Subsampled DataFrame containing only the selected STATE_TIME groups.
    """
    if not (0.0 < ratio_kept <= 1.0):
        raise ValueError("ratio_kept must be in the interval (0, 1].")

    state_times = df["STATE_TIME"].unique()
    n_keep = max(1, int(len(state_times) * ratio_kept))

    rng = np.random.default_rng(random_state) # for reproducibility
    keep_mask = rng.choice(state_times, size=n_keep, replace=False)

    return df[df["STATE_TIME"].isin(keep_mask)]

def create_dataset(df: pd.DataFrame, source_config: dict, nb_past: int, nb_future: int, num_encoding_func: callable, ratio_kept: float, 
            cat_enc: str = 'one_hot', stations_enc: dict | None = None, lines_enc: dict | None = None, 
            time_feat_enc: str | None = None) -> tuple:
    """
    Build an ML-ready dataset from processed punctuality data.

    The pipeline validates parameters, optionally subsamples by STATE_TIME,
    trims context columns to the requested window, applies numerical and
    categorical encodings, injects station/line embeddings, encodes time
    features, and computes local neighbor features if required.

    Args:
        df (pandas.DataFrame): Input DataFrame after monthly preprocessing.
        source_config (dict): Source configuration of processed data with 'nb_past_stations' and 'nb_future_stations'.
        nb_past (int): Number of past stations to keep.
        nb_future (int): Number of future stations to keep.
        num_encoding_func (callable): Function applied to numerical context columns.
        ratio_kept (float): Fraction of STATE_TIME groups to retain (≤ 1.0 keeps all).
        cat_enc (str): Categorical encoding strategy. Default is 'one_hot'.
        stations_enc (dict | None): Optional station embedding mapping; None to skip.
        lines_enc (dict | None): Optional line embedding mapping; None to skip.
        time_feat_enc (str | None): Time feature encoding strategy ('cyclical' or None).

    Returns:
        tuple: (df_out, category_dict) where
            df_out (pandas.DataFrame): Final ML-ready feature table.
            category_dict (dict): Categories used for categorical encoding.
    """

    check_parameters(source_config, nb_past, nb_future, num_encoding_func, cat_enc, stations_enc, lines_enc, time_feat_enc)

    if ratio_kept < 1.0:
        df = sample_by_state_time(df, ratio_kept)
    
    df = drop_out_of_scope_cols(df, source_config, nb_past, nb_future)
    df = apply_num_encoding_func(df, num_encoding_func, nb_past, nb_future)
    df, cat = apply_cat_encoding(df, cat_enc, nb_past, nb_future)
    df = apply_stations_encoding(df, stations_enc)
    df = apply_lines_encoding(df, lines_enc)
    df = apply_time_feat_encoding(df, time_feat_enc)
    df = apply_local_features(df)
    
    return df, cat

def sign_invariant_sqrt_std_normalization(x: pd.Series) -> np.ndarray:
    """
    Apply sign-invariant square-root transformation and standard deviation normalization.

    Args:
        x (pandas.Series): Input numerical series.

    Returns:
        numpy.ndarray: Transformed values.
    """
    x = x.values
    x = np.sign(x) * np.sqrt(np.abs(x))  # Sign-invariant square-root
    x = x / 6  # Apply normalization by the precomputed std
    return x

def save_data_point(i: int, x: torch.Tensor, y_d: torch.Tensor, y_a: torch.Tensor, md: dict, base_path: str) -> None:
    """
    Save a single data point's tensors and metadata to disk.
    One data point equals one STATE_TIME with 1 or more trains.

    Args:
        i (int): Index of the data point.
        x (torch.Tensor): Input feature tensor.
        y_d (torch.Tensor): Target delay tensor.
        y_a (torch.Tensor): Target action tensor.
        md (dict): Metadata dictionary for the data point.
        base_path (str): Root directory where files will be saved.

    Returns:
        None
    """
    data = [x, y_d, y_a, md]
    prefixes = ['x', 'y_delays', 'y_actions', 'md']

    for d, prefix in zip(data, prefixes):
        file_name = f'{prefix}_{i}.pt'
        folder_path = os.path.join(base_path, prefix)
        file_path = get_subdir_path(file_name, folder_path, makes=True)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(d, file_path)

def ensure_leaf_dirs(root: Path) -> None:
    """
    Ensure subdirectories for data storage exist under the given root.

    Args:
        root (pathlib.Path): Base directory where leaf subdirectories will be created.

    Returns:
        None
    """
    for leaf in ("x", "y_delays", "y_actions", "md"):
        (root / leaf).mkdir(parents=True, exist_ok=True)

def column_groups(df: pd.DataFrame) -> tuple:
    """
    Split columns into feature, delay, action, and metadata groups and build an index schema.

    Args:
        df (pandas.DataFrame): Input DataFrame with feature, delay, action, and metadata columns.

    Returns:
        tuple: (feature_cols, delay_cols, action_cols, md_cols, schema) where
            feature_cols (list): Names of feature columns.
            delay_cols (list): Names of future delay target columns.
            action_cols (list): Names of action target columns.
            md_cols (list): Names of metadata columns.
            schema (dict): Mapping of group names to {column: index} dictionaries.
    """
    md_cols      = ["STATE_TIME", "DATDEP", "TRAIN_NO"]
    action_cols  = ["action_same", "action_next1", "action_next2"]
    delay_cols   = [c for c in df.columns if "FUTURE_DELAYS" in c]
    feature_cols = [c for c in df.columns
                    if c not in md_cols + action_cols + delay_cols]

    schema = {
        "x"        : {c: i for i, c in enumerate(feature_cols)},
        "y_delays" : {c: i for i, c in enumerate(delay_cols)},
        "y_actions": {c: i for i, c in enumerate(action_cols)},
        "md"       : {c: i for i, c in enumerate(md_cols)},
    }
    return feature_cols, delay_cols, action_cols, md_cols, schema

def save_month(parquet_path: Path, subset_root: Path, start_idx: int, cfg: dict, st_emb: dict, ln_emb: dict, nb_past: int, 
            nb_future: int, ratio_kept: float, cat_enc: str, time_feat_enc: str | None, 
            precision: torch.dtype = torch.float32) -> tuple:
    """
    Process a monthly parquet file into ML-ready tensors and save grouped data points.

    Args:
        parquet_path (pathlib.Path): Path to the parquet file containing processed monthly data.
        subset_root (pathlib.Path): Root folder where output tensors will be stored.
        start_idx (int): Starting index for naming saved data points.
        cfg (dict): Source configuration dictionary.
        st_emb (dict): Station embeddings dictionary.
        ln_emb (dict): Line embeddings dictionary.
        nb_past (int): Number of past stations to keep.
        nb_future (int): Number of future stations to keep.
        ratio_kept (float): Fraction of STATE_TIME groups to retain.
        cat_enc (str): Categorical encoding strategy.
        time_feat_enc (str | None): Time feature encoding strategy.
        precision (torch.dtype): Torch tensor dtype for saved data. Default is torch.float32.

    Returns:
        tuple: (global_idx, category_dict, column_scheme) where
            global_idx (int): Next available index after processing the month.
            category_dict (dict): Categories used for categorical encoding.
            column_scheme (dict): Schema mapping column groups to indices.
    """
    df  = pd.read_parquet(parquet_path)
    out, cat = create_dataset(
        df, cfg, nb_past, nb_future,
        sign_invariant_sqrt_std_normalization, ratio_kept,
        cat_enc, st_emb, ln_emb, time_feat_enc
    )
    out = out[get_optimal_column_order(out, nb_past, nb_future, 8)]

    fcols, dcols, acols, mcols, column_scheme = column_groups(out)

    global_idx = start_idx
    for _, grp in tqdm(out.groupby("STATE_TIME"), desc=f"→ {parquet_path.name}", unit="group"):
        x  = torch.as_tensor(grp[fcols].values, dtype=precision)
        yd = torch.as_tensor(grp[dcols].values, dtype=precision)
        ya = torch.as_tensor(grp[acols].values, dtype=precision)
        md = grp[mcols].values

        save_data_point(global_idx, x, yd, ya, md, subset_root)
        global_idx += 1

    return global_idx, cat, column_scheme

def keep(col: str, max_past: int, max_future: int) -> bool:
    """
    Check whether a column is within the allowed past/future context window.

    Args:
        col (str): Column name to check.
        max_past (int): Maximum number of past stations to keep.
        max_future (int): Maximum number of future stations to keep.

    Returns:
        bool: True if the column should be kept, False otherwise.
    """
    _first_number = re.compile(r'\d+')
    if col.startswith('PAST_'):
        return int(_first_number.search(col).group()) <= max_past
    if col.startswith('FUTURE_'):
        return int(_first_number.search(col).group()) <= max_future
    return True

def is_local(col: str) -> bool:
    """
    Check whether a column corresponds to a local feature.

    Args:
        col (str): Column name to check.

    Returns:
        bool: True if the column is a local feature (count_r* or mean_delay_r*), False otherwise.
    """
    return col.startswith(('count_r', 'mean_delay_r'))

def _touch(d: dict, lst: list, col: str, old_idx: int) -> None:
    """
    Update schema dictionary and index list with a new column.

    Args:
        d (dict): Schema dictionary mapping column names to new indices.
        lst (list): List of old indices to preserve column ordering.
        col (str): Column name being added.
        old_idx (int): Original column index.

    Returns:
        None
    """
    d[col] = len(d)
    lst.append(old_idx)

def get_schemes(scheme: dict, nb_past_reg: int, nb_past_sim: int, nb_future_reg: int, nb_future_sim: int) -> tuple:
    """
    Build feature index schemes for regression and simulation tasks.
    Each pair X_keep - X gives: X_keep the indices of columns to keep 
    when loading and X the new column scheme after selection. 

    Args:
        scheme (dict): Mapping of column names to original indices.
        nb_past_reg (int): Max number of past stations for regression.
        nb_past_sim (int): Max number of past stations for simulation.
        nb_future_reg (int): Max number of future stations for regression.
        nb_future_sim (int): Max number of future stations for simulation.

    Returns:
        tuple: (reg_loc, reg_loc_keep, reg_noloc, reg_noloc_keep,
                sim_loc, sim_loc_keep, sim_noloc, sim_noloc_keep)
            where each scheme is a dict mapping kept column names to new indices,
            and each *_keep is a list of old indices for column selection.
    """
    reg_loc, reg_loc_keep = {}, []
    reg_noloc, reg_noloc_keep = {}, []
    sim_loc, sim_loc_keep = {}, []
    sim_noloc, sim_noloc_keep = {}, []

    for col, old_idx in scheme.items():
        if keep(col, nb_past_reg, nb_future_reg):
            _touch(reg_loc, reg_loc_keep, col, old_idx)
            if not is_local(col):
                _touch(reg_noloc, reg_noloc_keep, col, old_idx)

        if keep(col, nb_past_sim, nb_future_sim):
            _touch(sim_loc, sim_loc_keep, col, old_idx)
            if not is_local(col):
                _touch(sim_noloc, sim_noloc_keep, col, old_idx)

    return (reg_loc, reg_loc_keep, reg_noloc, reg_noloc_keep, sim_loc,   sim_loc_keep, sim_noloc, sim_noloc_keep)

def main() -> None:
    """
    Entry point to build ML-ready datasets from processed monthly parquet files.

    Parses CLI arguments, loads embeddings and config, ensures split directories,
    iterates over train/val/test months to generate and save per-state tensors,
    compiles dataset metadata and feature schemas for regression and simulation,
    and writes all artifacts to the output folder.
    """
    p = argparse.ArgumentParser()
    p.add_argument("folder_in_path", help="Raw data folder")
    p.add_argument("folder_out_path", help="Destination folder")
    p.add_argument("stations_emb_path",help="Pickle with station embeddings")
    p.add_argument("lines_emb_path", help="Pickle with line embeddings")
    p.add_argument("nb_past_stations_reg", type=int, help="# past stations for regression")
    p.add_argument("nb_future_stations_reg",type=int, help="# future stations for regression")
    p.add_argument("nb_past_stations_sim", type=int, help="# past stations for simulation")
    p.add_argument("nb_future_stations_sim",type=int, help="# future stations for simulation")
    p.add_argument("cat_enc", help="Cat-encoding")
    p.add_argument("time_feat_enc", help="Time-encoding")
    p.add_argument("test_ratio_kept", type=float, help="Ratio of test data kept in the final test set")
    p.add_argument("--train", nargs="+", required=True, help="Train months")
    p.add_argument("--val", nargs="+", required=True, help="Val months")
    p.add_argument("--test", nargs="+", required=True, help="Test months")

    args = p.parse_args()

    in_root = pathlib.Path(args.folder_in_path)
    out_root = pathlib.Path(args.folder_out_path)
    st_emb = load_pickle(args.stations_emb_path)
    ln_emb = load_pickle(args.lines_emb_path)
    cfg = load_pickle(in_root / "config.pkl")

    split_dirs = {s: out_root / s for s in ("train", "val", "test")}
    for p in split_dirs.values():
        ensure_leaf_dirs(p)

    splits = {"train": args.train, "val": args.val, "test": args.test}
    global_idx = dict.fromkeys(splits, 0)

    for split_name, months in splits.items():
        root = split_dirs[split_name]
        for month in months:
            parquet_file = in_root / f"processed_data_{month}.brotli.parquet"
            
            global_idx[split_name], cat, column_scheme = save_month(
                parquet_file,
                root,
                global_idx[split_name],
                cfg = cfg,
                st_emb = st_emb,
                ln_emb = ln_emb,
                nb_past = max(args.nb_past_stations_reg, args.nb_past_stations_sim),
                nb_future = max(args.nb_future_stations_reg, args.nb_future_stations_sim),
                ratio_kept = args.test_ratio_kept if split_name == 'test' else 1.0,
                cat_enc = args.cat_enc,
                time_feat_enc = args.time_feat_enc,
            )
    dataset_config = {
        'deltat':cfg['deltat'],
        'nb_past_station_reg':args.nb_past_stations_reg,
        'nb_future_station_reg':args.nb_future_stations_reg,
        'nb_past_station_sim':args.nb_past_stations_sim,
        'nb_future_station_sim':args.nb_future_stations_sim,
        'idle_beg':cfg['idle_beg'],
        'idle_end':cfg['idle_end'],
        'embedding_size':len(st_emb[list(st_emb)[0]]),
        'train_months':args.train,
        'val_months':args.val,
        'test_months':args.test,
        'train_size':global_idx['train'],
        'val_size':global_idx['val'],
        'test_size':global_idx['test']
    }

    reg_loc_scheme, reg_loc_keep, reg_non_scheme, reg_non_keep, sim_loc_scheme, sim_loc_keep, sim_non_scheme, sim_non_keep = get_schemes(column_scheme['x'], args.nb_past_stations_reg, args.nb_past_stations_sim, args.nb_future_stations_reg, args.nb_future_stations_sim)

    sc_reg_loc = {
        'cols_to_keep':reg_loc_keep,
        'x':reg_loc_scheme,
        'y':column_scheme['y_delays'],
        'md':column_scheme['md'],
    }
    
    sc_reg_non = {
        'cols_to_keep':reg_non_keep,
        'x':reg_non_scheme,
        'y':column_scheme['y_delays'],
        'md':column_scheme['md'],
    }

    sc_sim_loc = {
        'cols_to_keep':sim_loc_keep,
        'x':sim_loc_scheme,
        'y':column_scheme['y_actions'],
        'md':column_scheme['md'],
    }
    
    sc_sim_non = {
        'cols_to_keep':sim_non_keep,
        'x':sim_non_scheme,
        'y':column_scheme['y_actions'],
        'md':column_scheme['md'],
    }

    save_pickle(cat, out_root / 'cat.pkl')
    save_pickle(dataset_config, out_root / 'config.pkl')
    save_pickle(sc_reg_loc, out_root / 'sc_reg_loc.pkl')
    save_pickle(sc_reg_non, out_root / 'sc_reg_non.pkl')
    save_pickle(sc_sim_loc, out_root / 'sc_sim_loc.pkl')
    save_pickle(sc_sim_non, out_root / 'sc_sim_non.pkl')
    save_pickle(st_emb, out_root / 'stations_emb.pkl')
    save_pickle(ln_emb, out_root / 'lines_emb.pkl')

if __name__ == "__main__":
    main()