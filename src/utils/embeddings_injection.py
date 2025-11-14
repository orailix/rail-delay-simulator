import pickle
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

def inject_embedding(df: pd.DataFrame, embeddings_dict: dict, col_prefix: str, nb_cols: int, nb_dims: int) -> pd.DataFrame:
    """
    Replace categorical columns with embedding vectors.

    Args:
        df (pd.DataFrame): Input dataframe.
        embeddings_dict (dict): Mapping from categorical string name to embedding vector.
        col_prefix (str): Prefix of categorical columns to replace.
        nb_cols (int): Number of prefixed columns to inject on.
        nb_dims (int): Number of embedding dimension to inject on.

    Returns:
        pd.DataFrame: Dataframe with embedding columns injected and original categorical columns dropped.
    """
    for i in tqdm(range(nb_cols), desc = f'Injecting embeddings in columns with prefix: {col_prefix}'):
        unique_names = df[f"{col_prefix}_{1 + i}"].unique()
        unique_embeddings = np.array([embeddings_dict.get(x, np.zeros(nb_dims))[:nb_dims] 
                                      for x in unique_names], 
                                     dtype='float32')
    
        name_to_index = {name: idx for idx, name in enumerate(unique_names)}
        indices = np.array([name_to_index[name] for name in df[f"{col_prefix}_{1 + i}"]])
        
        embeddings = unique_embeddings[indices]
        new_columns_names = [f'{col_prefix}_{1 + i}_embedding_{j}' for j in range(nb_dims)]
        df[new_columns_names] = embeddings
    df.drop(columns = [f'{col_prefix}_{1 + i}' for i in range(nb_cols)], inplace = True)
    return df

def inject_all_embeddings(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Inject embeddings for all categorical column groups defined in config.

    Args:
        df (pd.DataFrame): Input dataframe.
        config (dict): Mapping {col_prefix: (embeddings_path, nb_cols, nb_dims)}.

    Returns:
        pd.DataFrame: Dataframe with all specified embeddings injected.
    """

    for col_prefix, values in config.items():
        embeddigns_path, nb_cols, nb_dims = values
        with open(embeddigns_path, 'rb') as file:
            embeddings_dict = pickle.load(file)
        df = inject_embedding(df, embeddings_dict, col_prefix, nb_cols, nb_dims)
    return df