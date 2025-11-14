import torch
import torch.nn as nn
import time
import bisect
import os
import tempfile
import random
import pickle
import copy

import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torch.nn.utils.rnn as rnn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.cuda import amp
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from matplotlib.ticker import FuncFormatter

from src.utils.utils import load_pickle, save_pickle

MAX_ITI_LEN = 140 # maximum length of an itinerary (computed on the paper data span)

def load_itineraries_from_dates(dates: list, folder_path: str, show_prog: bool = False) -> dict:
    """
    Load saved itineraries tensors for given dates.

    Args:
        dates (list): List of date identifiers (e.g., 'DDMMMYYYY') to load.
        folder_path (str): Directory containing itinerary files.
        show_prog (bool): If True, display a progress bar while loading. Default is False.

    Returns:
        dict: Mapping from date string to loaded itinerary tensors.
    """
    itineraries = {}
    iterator = tqdm(dates, desc="Loading itineraries") if show_prog else dates

    for date in iterator:
        file_path = os.path.join(folder_path, f'itineraries_{date}.pt')
        itineraries[date] = torch.load(file_path)
        
    return itineraries

def get_possible_dates(state_time: int) -> list:
    """
    Get possible departure dates around a given state time.

    Note: Could be optimized further by taking into account the bound of the simulation, reducing useless loading

    Args:
        state_time (int): Time in seconds since 2012-01-01.

    Returns:
        list: Three formatted dates (previous, current, next) in '%d%b%Y' format, uppercase.
    """

    base_timestamp = pd.Timestamp("2012-01-01")
    
    current_datetime = base_timestamp + pd.to_timedelta(state_time, unit='s')
    
    previous_datetime = current_datetime - pd.Timedelta(days=1)
    next_datetime = current_datetime + pd.Timedelta(days=1)
    
    formatted_current = current_datetime.strftime('%d%b%Y').upper()
    formatted_previous = previous_datetime.strftime('%d%b%Y').upper()
    formatted_next = next_datetime.strftime('%d%b%Y').upper()
    
    return [formatted_previous, formatted_current, formatted_next]

def get_group_feature_mapping(column_mapping: dict, nb_past_stations: int, nb_future_stations: int, embeddings_size: int, device: torch.device, local_features: bool) -> dict:
    """
    Build grouped feature indices for simulation optimisation.

    Groups indices of features into tensors or lists, organized by feature type. 
    This mapping is later used to slice states and update them in a vectorized fashion.

    Args:
        column_mapping (dict): Mapping from column names to feature indices.
        nb_past_stations (int): Number of past stations to include in context.
        nb_future_stations (int): Number of future stations to include in context.
        embeddings_size (int): Dimensionality of station/line embeddings.
        device (torch.device): Torch device to place tensors on.
        local_features (bool): Whether to include local neighbor count and delay features.

    Returns:
        dict: Feature mapping where keys are feature groups (e.g., 'PLANNED_TIME_NUM',
        'PAST_DELAYS', 'RELATION_TYPE', 'STATIONS_embedding') and values are index tensors
        or dictionaries (for temporal features).
    """
    group_feature_mapping = {}

    past_col_names = [f"PAST_PLANNED_TIME_NUM_{nb_past_stations - i}" for i in range(nb_past_stations)]
    future_col_names = [f"FUTURE_PLANNED_TIME_NUM_{i+1}" for i in range(nb_future_stations)]
    group_feature_mapping['PLANNED_TIME_NUM'] = torch.tensor([column_mapping[col_name] for col_name in past_col_names + future_col_names]).to(device)
    
    col_names = [f"PAST_DELAYS_{nb_past_stations - i}" for i in range(nb_past_stations)]
    group_feature_mapping['PAST_DELAYS'] = torch.tensor([column_mapping[col_name] for col_name in col_names]).to(device)

    temp_cols_mapping = {
        f"{cycle}_{trig}_{freq}": column_mapping[f"{cycle}_{trig}_{freq}"]
        for cycle in ['hour', 'year']
        for freq in [1, 2, 4]
        for trig in ['sin', 'cos']
    }
    group_feature_mapping['temporal_features'] = temp_cols_mapping

    rel_type_cols = [col for col in column_mapping.keys() if col.startswith('RELATION_TYPE_')]
    group_feature_mapping['RELATION_TYPE'] = torch.tensor([column_mapping[col_name] for col_name in rel_type_cols]).to(device)

    day_of_week_cols = [col for col in column_mapping.keys() if col.startswith('day_of_week_')]
    group_feature_mapping['day_of_week'] = torch.tensor([column_mapping[col_name] for col_name in day_of_week_cols]).to(device)

    for typ in ['P', 'D', 'A']:
        past_col_names = [f"PAST_TYPES_{nb_past_stations - i}_{typ}" for i in range(nb_past_stations)]
        future_col_names = [f"FUTURE_TYPES_{i+1}_{typ}" for i in range(nb_future_stations)]
        group_feature_mapping[f'TYPES_{typ}'] = torch.tensor([column_mapping[col_name] for col_name in past_col_names + future_col_names]).to(device)

    for entity in ['STATIONS', 'LINES']:
        group_feature_mapping[f"{entity}_embedding"] = []
        for i in range(nb_past_stations):
            station_key = f"PAST_{entity}_{nb_past_stations - i}_embedding"
            group_feature_mapping[f"{entity}_embedding"].extend([column_mapping[f"{station_key}_{emb_idx}"] for emb_idx in range(embeddings_size)])
        for i in range(nb_future_stations):
            station_key = f"FUTURE_{entity}_{i+1}_embedding"
            group_feature_mapping[f"{entity}_embedding"].extend([column_mapping[f"{station_key}_{emb_idx}"] for emb_idx in range(embeddings_size)])
        group_feature_mapping[f"{entity}_embedding"] = torch.tensor(group_feature_mapping[f"{entity}_embedding"]).to(device)

    if local_features:
        radiuses = [0.1,0.3,0.6,1.0,2.0]
        group_feature_mapping[f"count_r"] = [column_mapping[f"count_r{rad}"] for rad in radiuses]
        group_feature_mapping[f"mean_delay_r"] = [column_mapping[f"mean_delay_r{rad}"] for rad in radiuses]

    return group_feature_mapping

class StatesManager:
    """
    Manage batched simulation states, metadata, and buffers for prediction, DCIL or GAIL modes.

    Initializes per-simulation timestamps, padded state tensors, metadata (positions and
    max positions), and raw itinerary-aligned data. In 'predict' mode, allocates a
    prediction buffer and output structures; in 'dcil' mode, computes expected positions 
    (according to ground truth obeserved data) over a fixed number of steps.

    For optimization purposes, all simulations and trajectories operations are computed in
    parallel by using a single global padded tensor/array for each data structure. As a
    result, these data structures are often of shape (nb_sim*nb_samples, max_seq_len, ...) or
    (nb_sim*nb_samples, ...), meaning that the first index indicates which simulation and
    trajectory the data is about.

    Args:
        states: List of per-simulation initial state tensors.
        states_metadata: List of per-simulation metadata arrays (DATDEP, TRAIN_NO).
        state_times: List of per-simulation reference times (seconds since 2012-01-01).
        itineraries: Dictionary holding itinerary tensors and lookup mappers.
        schedules: Schedule of trains to add at each step, used to initialize prediction outputs (predict mode).
        nb_samples (int): Number of stochastic trajectories per simulation.
        nb_past (int): Number of past stations included in each state.
        nb_future (int): Number of future stations included in each state.
        emb_size (int): Number of dimension of station/line embeddings to use.
        cat_cols_md (dict): Categorical columns metadata (e.g., one-hot layouts).
        group_features_mapper (dict): Mapping from feature groups to column indices/tensors.
        stations_emb (dict): Station embeddings mapping (trimmed to emb_size internally).
        lines_emb (dict): Line embeddings mapping (trimmed to emb_size internally).
        device: Torch device for tensor allocation.
        deltat (int): Time-step size in seconds.
        nb_steps (int): Number of rollout steps.
        mode (str): Operating mode, either 'predict' or 'dcil'.

    Attributes:
        nb_sim (int): Number of simulations in the batch (can be different initial states at different initial states times).
        nb_samples (int): Number of stochastic trajectories per simulation.
        nb_past (int): Past context length.
        nb_future (int): Future context length.
        emb_size (int): Embedding dimensionality in use.
        cat_cols_md (dict): Stored categorical columns metadata.
        group_features_mapper (dict): Stored feature-group index mapping.
        stations_emb (dict): Station embeddings truncated to emb_size.
        lines_emb (dict): Line embeddings truncated to emb_size.
        device: Torch device used for tensors.
        deltat (int): Time-step size in seconds.
        mode (str): Current mode ('predict' or 'dcil').
        states_time (torch.Tensor): Tensorized per-simulation reference times of shape (nb_sim*nb_samples,).
        states (torch.Tensor): Batched, padded state tensor of shape (nb_sim*nb_samples, max_seq_len, nb_feats).
        padding_mask (torch.Tensor): Boolean mask marking padded positions in states of shape (nb_sim*nb_samples, max_seq_len).
        md (np.ndarray): Per-simulation metadata tensor of shape (nb_sim*nb_samples, max_seq_len, 3).
        positions (torch.Tensor): Current positions of each train within itineraries of shape (nb_sim*nb_samples, max_seq_len).
        max_positions (torch.Tensor): Maximum valid positions per itinerary of shape (nb_sim*nb_samples, max_seq_len).
        raw_data (torch.Tensor): Raw inputs aligned with itineraries and state times of shape (nb_sim*nb_samples, MAX_ITI_LEN, nb_raw_data_feat).
        pred_buffer: Prediction buffer used to store running trains predictions of shape (nb_sim*nb_samples, max_seq_len, MAX_ITI_LEN) (predict mode).
        output (dict): Dictionary with keys ('end_pos' and 'predictions') storing end positions and predictions for trains no longer running, completed at the end of simulation (predict mode).
        expected_pos (torch.Tensor): Expected positions over nb_steps according to ground truth data, of shape (nb_sim*nb_samples, max_seq_len, nb_steps) (dcil mode).
    """
    def __init__(self,
             states: list,
             states_metadata: list,
             state_times: list,
             itineraries: dict,
             schedules: dict,
             nb_samples: int,
             nb_past: int,
             nb_future: int,
             emb_size: int,
             cat_cols_md: dict,
             group_features_mapper: dict,
             stations_emb: dict,
             lines_emb: dict,
             device: torch.device,
             deltat: int,
             nb_steps: int,
             mode: str) -> None:

        self.nb_sim = len(states)
        self.nb_samples = nb_samples
        self.nb_past = nb_past
        self.nb_future = nb_future
        self.emb_size = emb_size
        self.cat_cols_md = cat_cols_md
        self.group_features_mapper = group_features_mapper
        self.stations_emb = {k:v[:emb_size] for k,v in stations_emb.items()}
        self.lines_emb = {k:v[:emb_size] for k,v in lines_emb.items()}
        self.device = device
        self.deltat = deltat
        self.mode = mode
        
        self.states_time = self.init_states_time(state_times)
        self.states, self.padding_mask = self.init_states(states)
        self.md, self.positions, self.max_positions = self.init_metadatas(states_metadata, itineraries)
        self.raw_data = self.init_raw_data(itineraries, 'states_iti_mapper', self.positions, self.states_time)
        if self.mode == 'predict':
            self.pred_buffer = torch.full((*self.padding_mask.shape, MAX_ITI_LEN), -1, dtype=torch.float64, device=device)
            self.output = self.init_predictions(schedules)
        elif self.mode == 'dcil':
            self.expected_pos = self.init_expected_pos(itineraries, nb_steps)
    
    def init_states_time(self, states_times: list) -> torch.Tensor:
        """
        Initialize simulation start times for all trajectories.

        Each provided state time is repeated `nb_samples` times so that every
        stochastic trajectory of a simulation shares the same initial timestamp.

        Args:
            states_times (list): List of per-simulation reference times (seconds since 2012-01-01).

        Returns:
            torch.Tensor: Flattened tensor of start times with shape (nb_sim * nb_samples,).
        """
        return torch.tensor(
            [state for state in states_times for _ in range(self.nb_samples)],
            dtype=torch.int32,
            device=self.device
        )

    def init_states(self, states: list) -> tuple:
        """
        Initialize padded state tensors and their padding masks.

        Each simulation state is repeated `nb_samples` times, padded to the maximum
        sequence length in the batch, and stacked into a single tensor. A boolean mask
        is also built to mark the padded (empty) positions.

        Args:
            states (list): List of per-simulation state tensors, each of variable length
                with shape (seq_len, nb_feats).

        Returns:
            tuple: (padded_states, padding_mask) where
                padded_states (torch.Tensor): Batched state tensor of shape
                    (nb_sim * nb_samples, max_seq_len, nb_feats).
                padding_mask (torch.Tensor): Boolean mask of shape
                    (nb_sim * nb_samples, max_seq_len), where 1 marks padding positions.
         """
        repeated_states = [state for state in states for _ in range(self.nb_samples)]
        padded_states = pad_sequence(repeated_states, batch_first=True, padding_value=0)

        max_n, m = padded_states.size(1), padded_states.size(2)
        padding_mask = torch.zeros((len(repeated_states), max_n), dtype=torch.bool)
        
        for i, state in enumerate(repeated_states):
            n = state.size(0)
            padding_mask[i, n:] = 1
        
        return padded_states.to(self.device), padding_mask.to(self.device)

    def init_expected_pos(self, itineraries: dict, nb_steps: int) -> torch.Tensor:
        """
        Compute ground-truth expected positions of trains over future steps.

        Uses itineraries data: 
        compares observed arrival times with future step times to derive 
        expected position trajectories. Padding positions are masked out.

        Args:
            itineraries (dict): Relevant itineraries with keys like 'data' and 'max_pos' and 'states_iti_mapper'.
            nb_steps (int): Number of rollout steps.

        Returns:
            torch.Tensor: Expected positions of shape
                (nb_sim * nb_samples, max_seq_len, nb_steps), with empty positions masked.
        """
        mapper = itineraries['states_iti_mapper']
        n, m = self.states.shape[:2]
        obs = itineraries['data'][mapper][:,:,:,4].unsqueeze(-1).repeat(1,1,1,nb_steps)
        max_pos = itineraries['max_pos'][mapper].clone().view(n,m,1).repeat(1,1,nb_steps)
        steps_states_times = self.states_time.view(n, 1, 1, 1).repeat(1, m, MAX_ITI_LEN, nb_steps) + torch.arange(self.deltat,self.deltat*(nb_steps+1), self.deltat).to(self.device)

        return (max_pos - (obs > steps_states_times).sum(axis = 2) + self.nb_future) * ~self.padding_mask.view(n,m,1).repeat(1,1,nb_steps) # I apologies

    def init_metadatas(self, metadatas: list, itineraries: dict) -> tuple:
        """
        Initialize metadata, current positions, and maximum positions for each train.

        Uses itineraries data and observed arrival times to compute each train's
        current position relative to its itinerary. Metadata arrays (DATDEP, TRAIN_NO)
        are expanded across samples, and positions are added as the third entry.
        Padding positions are masked out.

        Args:
            metadatas (list): Per-simulation metadata arrays (DATDEP, TRAIN_NO).
            itineraries (dict): Relevant itineraries with keys like 'data',
                'max_pos', and 'states_iti_mapper'.

        Returns:
            tuple: (md, pos, max_pos) where
                md (np.ndarray): Metadata of shape (nb_sim * nb_samples, max_seq_len, 3).
                pos (torch.Tensor): Current positions of shape (nb_sim * nb_samples, max_seq_len).
                max_pos (torch.Tensor): Maximum valid positions per itinerary
                    of shape (nb_sim * nb_samples, max_seq_len).
        """
        mapper = itineraries['states_iti_mapper']
        pos = torch.zeros(self.states.shape[0], self.states.shape[1], dtype = torch.int32).to(self.device)
        md = np.full((self.states.shape[0], self.states.shape[1], 3), None, dtype = 'object')
        max_pos = itineraries['max_pos'][mapper].clone()

        obs = itineraries['data'][mapper][:,:,:,4]
        pos = (max_pos - (obs > self.states_time.view(obs.shape[0], 1, 1).expand(*obs.shape[:3])).sum(axis = -1) + self.nb_future) * ~self.padding_mask
        
        for i in range(self.nb_sim):
            md[i*self.nb_samples:(i+1)*self.nb_samples,:metadatas[i].shape[0], :2] = metadatas[i]
        md[:,:,2] = pos.detach().cpu().numpy()
            
        return md, pos, max_pos

    def init_raw_data(self, itineraries: dict, mapper_key: str, pos: torch.Tensor, states_time: torch.Tensor) -> torch.Tensor:
        """
        Initialize raw input features aligned with itineraries and states.

        Builds per-train raw feature tensors by combining theoretical times, station/line types,
        and embeddings. Sequences are padded to `MAX_ITI_LEN` and shifted so the current position
        aligns consistently across samples. Used to update state in a vectorized fashion.

        Used both for main states (states_iti_mapper, shape (nb_sim*nb_samples, max_seq_len)) and 
        for scheduled trains (tta_iti_mapper, shape (nb_trains, 1)) raw data creation.

        Args:
            itineraries (dict): Itineraries dict with tensors (e.g., 'data') and mappers.
            mapper_key (str): Key for selecting the itinerary mapper (e.g., 'states_iti_mapper' or 'tta_iti_mapper').
            pos (torch.Tensor): Current train positions of shape (nb_sim * nb_samples, max_seq_len) or (1, n_trains) for TTA.
            states_time (torch.Tensor): Reference times of shape (nb_sim * nb_samples,) or (n_trains,) for TTA.

        Returns:
            torch.Tensor: Raw features of shape (B, T, MAX_ITI_LEN, nb_raw_features) where
                nb_raw_features = 1 (theoretical time) + 3 (types) + 2 * emb_size, and:
                - if mapper_key == 'states_iti_mapper': B = nb_sim * nb_samples, T = max_seq_len;
                - if mapper_key == 'tta_iti_mapper': B = number of scheduled trains, T = 1.
        """
        mapper = itineraries[mapper_key]
        max_nb_trains = mapper.shape[1]
        raw_data = torch.zeros((mapper.shape[0], max_nb_trains, MAX_ITI_LEN, 1 + 3 + 2 * self.emb_size), 
                               dtype=torch.float32, device=self.device)
        raw_data[:, :, :, 0] = self.get_theoretical_times(itineraries, mapper_key, states_time)
        raw_data[:, :, :, 1:4] = self.get_types(itineraries, max_nb_trains, mapper_key)
        raw_data[:, :, :, 4:4+self.emb_size] = self.get_embeddings(itineraries, self.stations_emb, 'stations', mapper_key)
        raw_data[:, :, :, 4+self.emb_size:4+2*self.emb_size] = self.get_embeddings(itineraries, self.lines_emb, 'lines', mapper_key)

        S, T, L, F = raw_data.shape
        raw_data = raw_data.reshape(S * T, L, F)
        pos_reshaped = pos.reshape(S * T)
        
        for v in torch.unique(pos_reshaped):
            offset = v - self.nb_past + 1
            if offset > 0:
                mask = pos_reshaped == v
                raw_data[mask, :-offset, :] = raw_data[mask, offset:, :]
        
        raw_data = raw_data.reshape(S, T, L, F)

        return raw_data

    def init_predictions(self, schedules: list) -> list:
        """
        Initialize prediction buffers and train mappings for each simulation.

        For each simulation, builds:
        - a mapper linking (DATDEP, TRAIN_NO) for running and scheduled trains to indices in predictions tensor,
        - start positions for active and incoming trains,
        - end position buffers (initially zeros),
        - prediction tensors filled with -1.

        Args:
            schedules (list[dict]): Per-simulation schedules. Each dict maps
                state times to lists of ((datdep, train_no), _) tuples.

        Returns:
            list[dict]: One dict per simulation with keys:
                - 'mapper': dict mapping train identifiers to indices in predictions tensor.
                - 'start_pos': torch.Tensor of shape (nb_trains,).
                - 'end_pos': torch.Tensor of shape (nb_trains, nb_samples).
                - 'predictions': torch.Tensor of shape
                (nb_trains, nb_samples, MAX_ITI_LEN), initialized to -1.
        """
        output = []
        nb_trains = (~self.padding_mask).sum(axis = 1)
        for i in range(self.nb_sim):
            output.append({})
            trains_to_add = []
            for state_time in schedules[i].keys():
                for name, _ in schedules[i][state_time]:
                    trains_to_add.append(name)
            array_size = nb_trains[i*self.nb_samples]
            output[i]['mapper'] = {
                **{tuple(self.md[i * self.nb_samples, idx, :2]): idx for idx in range(array_size)},
                **{name: idx for idx, name in enumerate(trains_to_add, start=array_size)}
            }
            output[i]['start_pos'] = torch.cat([self.positions[i*self.nb_samples, :array_size], torch.full((len(trains_to_add),), 4).to(self.device)])
            output[i]['end_pos'] = torch.zeros((array_size + len(trains_to_add), self.nb_samples), dtype=int, device = self.device)
            output[i]['predictions'] = torch.full((array_size + len(trains_to_add), self.nb_samples, MAX_ITI_LEN), -1, dtype=self.pred_buffer.dtype, device=self.device) # can possibly be directly on cpu ???

        return output

    def get_theoretical_times(self, itineraries: dict, mapper_key: str, states_time: torch.Tensor) -> torch.Tensor:
        """
        Compute relative planned times for all trains.

        Subtracts the reference state time from planned arrival/departure
        times in the itineraries, yielding time deltas aligned with the
        current simulation step.

        Args:
            itineraries (dict): Relevant itineraries containing 'data' tensor
                with planned times at index [:,:,:,3].
            mapper_key (str): Key for selecting the correct itinerary mapper
                (e.g., 'states_iti_mapper' or 'tta_iti_mapper').
            states_time (torch.Tensor): Reference times of shape
                (nb_sim*nb_samples,) for states, or (n_trains,) for TTA.

        Returns:
            torch.Tensor: Relative planned times of shape
                (B, T, MAX_ITI_LEN), dtype float32, where:
                - if mapper_key == 'states_iti_mapper': B = nb_sim*nb_samples, T = max_seq_len
                - if mapper_key == 'tta_iti_mapper':     B = number of scheduled trains, T = 1
        """
        mapper = itineraries[mapper_key]
        theo = itineraries['data'][mapper][:,:,:,3].clone()
        st = states_time.view(theo.shape[0], 1, 1).expand(*theo.shape[:3]).to(dtype = torch.int32)*(theo != 0) # converted to int32 because theo looses precision when auto-casted to float32 with - operator
        return (theo - st).to(dtype=torch.float32)

    def get_types(self, itineraries: dict, max_nb_trains: int, mapper_key: str) -> torch.Tensor:
        """
        Retrieve one-hot encoded stop types (D, A, P).

        Args:
            itineraries (dict): Relevant itineraries containing 'data' tensor,
                where column 1 stores stop types as integer codes.
            max_nb_trains (int): Maximum number of trains per simulation batch
                (padded dimension length).
            mapper_key (str): Key for selecting the correct itinerary mapper
                (e.g., 'states_iti_mapper' or 'tta_iti_mapper').

        Returns:
            torch.Tensor: One-hot encoded stop types of shape
                (B, T, MAX_ITI_LEN, 3), where:
                - if mapper_key == 'states_iti_mapper': B = nb_sim*nb_samples, T = max_seq_len
                - if mapper_key == 'tta_iti_mapper':     B = number of scheduled trains, T = 1
        """
        mapper = itineraries[mapper_key]
        one_hot_flat = torch.nn.functional.one_hot(itineraries['data'][mapper][:,:,:,1].clone().view(-1).long(), num_classes=3).float()
        result_tensor = one_hot_flat.view(mapper.shape[0], max_nb_trains, MAX_ITI_LEN, 3)
    
        return result_tensor

    def get_embeddings(self, itineraries: dict, emb_dict: dict, emb_type: str, mapper_key: str) -> torch.Tensor:
        """
        Retrieve station or line embeddings for each stop in the itineraries.

        Args:
            itineraries (dict): Relevant itineraries containing 'data' tensor,
                where columns index stations (col=0) and lines (col=2).
            emb_dict (dict): Mapping from entity ID to its embedding vector,
                already trimmed to the configured embedding size.
            emb_type (str): Type of embeddings to extract, either 'stations' or 'lines'.
            mapper_key (str): Key for selecting the correct itinerary mapper
                (e.g., 'states_iti_mapper' or 'tta_iti_mapper').

        Returns:
            torch.Tensor: Embedding tensor of shape
                (B, T, MAX_ITI_LEN, emb_size), where:
                - if mapper_key == 'states_iti_mapper': B = nb_sim*nb_samples, T = max_seq_len
                - if mapper_key == 'tta_iti_mapper':     B = number of scheduled trains, T = 1
        """
        mapper = itineraries[mapper_key]

        if emb_type == 'stations':
            emb_index = 0  # 'stations' corresponds to index 0
        elif emb_type == 'lines':
            emb_index = 2  # 'lines' corresponds to index 2
        index_to_vector = torch.from_numpy(np.array(list(emb_dict.values()), dtype=np.float32)).to(self.device)
        result = index_to_vector[itineraries['data'][mapper][:,:,:,emb_index].clone().long()]
        
        return result
        
    def create_data(self, raw_data: torch.Tensor, states_time: torch.Tensor, rel_types: torch.Tensor) -> torch.Tensor:
        """
        Create initial feature vectors for scheduled trains using raw data.

        Combines temporal encodings, day-of-week, planned times, relation type,
        station/line types, and embeddings into a single flat feature tensor.

        This is only used to create the initial states of trains not yet on the 
        network (from 'tta_iti_mapper').

        Args:
            raw_data (torch.Tensor): Raw per-train data of shape
                (nb_trains, MAX_ITI_LEN, nb_raw_data_feat).
            states_time (torch.Tensor): Reference times for each scheduled train
                (nb_trains,).
            rel_types (torch.Tensor): Relation type IDs for each train, used
                to one-hot encode relation categories.

        Returns:
            torch.Tensor: Flattened feature tensor of shape
                (nb_trains, nb_features), where nb_features matches the size of
                self.states.shape[2].
        """
        lookup_range = self.nb_past + self.nb_future
        data = torch.zeros(raw_data.shape[0],self.states.shape[2]).to(self.device)
        data[:,list(self.group_features_mapper['temporal_features'].values())] = torch.stack([self.temporal_features_values[state_time] for state_time in states_time], dim=0)
        data[:,self.group_features_mapper['day_of_week']] = torch.stack([self.day_of_week_values[state_time] for state_time in states_time], dim=0)
        data[:,self.group_features_mapper['PLANNED_TIME_NUM']] = self.apply_sign_invariant_sqrt_and_std(raw_data[:, :lookup_range, 0])
        data[:,self.group_features_mapper['RELATION_TYPE']] = F.one_hot(rel_types, num_classes=len(self.cat_cols_md['RELATION_TYPE'])).float().to(self.device)
        data[:,self.group_features_mapper['TYPES_D']] = raw_data[:, :lookup_range, 1]
        data[:,self.group_features_mapper['TYPES_P']] = raw_data[:, :lookup_range, 2]
        data[:,self.group_features_mapper['TYPES_A']] = raw_data[:, :lookup_range, 3]
        data[:,self.group_features_mapper['STATIONS_embedding']] = raw_data[:, :lookup_range, 4:4+self.emb_size].reshape(data.shape[0], -1)
        data[:,self.group_features_mapper['LINES_embedding']] = raw_data[:, :lookup_range, 4+self.emb_size:4+2*self.emb_size].reshape(data.shape[0], -1)

        return data        

    def update_positions(self, actions: torch.Tensor) -> None:
        """
        Update current positions by applying actions and, in predict mode, shift 
        prediction buffers and updates it with the new prediction(s).

        Args:
            actions (torch.Tensor): Action tensor of shape (nb_sim * nb_samples, max_seq_len) with values in {0, 1, 2}.

        Returns:
            None
        """
        self.positions += actions
        
        if self.mode == 'predict':
            reshaped_actions = actions.reshape(-1)
    
            reshaped_pred_buffer = self.pred_buffer.reshape(-1, MAX_ITI_LEN)
            reshaped_states_times = self.states_time.double().repeat_interleave(self.pred_buffer.shape[1])
            
            next1_mask = reshaped_actions == 1
            reshaped_pred_buffer[next1_mask, :-1] = reshaped_pred_buffer[next1_mask, 1:]
            reshaped_pred_buffer[next1_mask, -1] = reshaped_states_times[next1_mask]
    
            next2_mask = reshaped_actions == 2
            reshaped_pred_buffer[next2_mask, :-2] = reshaped_pred_buffer[next2_mask, 2:]
            reshaped_pred_buffer[next2_mask, -2:] = reshaped_states_times[next2_mask].unsqueeze(1).repeat(1, 2)
    
            self.pred_buffer = reshaped_pred_buffer.reshape(self.pred_buffer.shape)

    def update(self, actions: torch.Tensor) -> None:
        """
        Update states and raw buffers after applying actions.

        Shifts raw_data and delay features according to actions, refreshes temporal/day-of-week encodings,
        and keeps planned times, stop types, and embeddings consistent using the new version of raw_data.

        Args:
            actions (torch.Tensor): Action codes for each train (0=same, 1=next1, 2=next2).

        Returns:
            None
        """
        lookup_range = self.nb_past + self.nb_future
        s0, s1 = self.raw_data.shape[:2]
        
        self.states[:, :, list(self.group_features_mapper['temporal_features'].values())] = torch.stack([self.temporal_features_values[t.item()] for t in self.states_time], dim=0).unsqueeze(1).expand(-1, self.states.shape[1], -1)
        self.states[:, :, self.group_features_mapper['day_of_week']] = torch.stack([self.day_of_week_values[t.item()] for t in self.states_time], dim=0).unsqueeze(1).expand(-1, self.states.shape[1], -1)
        
        actions = actions.reshape(-1)
        next1_mask = actions == 1
        next2_mask = actions == 2
        #self.positions = self.positions.reshape(-1)
        reshaped_states_times = self.states_time.double().repeat_interleave(self.states.shape[1])
        
        self.raw_data[:, :, :, 0] -= self.deltat # update theoretical times
        self.raw_data = self.raw_data.reshape(s0*s1, MAX_ITI_LEN, self.raw_data.shape[3])
        self.raw_data[next1_mask, :-1, :] = self.raw_data[next1_mask, 1:, :]
        self.raw_data[next2_mask, :-2, :] = self.raw_data[next2_mask, 2:, :]
        self.states = self.states.reshape(s0*s1, -1)
        # not very optimized to update delays like this, (2/3 of update time), maybe use raw_data to store delays 
        delays_indexes = self.group_features_mapper['PAST_DELAYS'] # this will only hold because the indexes are in order
        self.states[next1_mask, delays_indexes[0]:delays_indexes[-1]] = self.states[next1_mask, delays_indexes[1]:delays_indexes[-1]+1]
        self.states[next1_mask, delays_indexes[-1]] = self.apply_sign_invariant_sqrt_and_std(-self.raw_data[next1_mask, self.nb_past-1, 0])
        self.states[next2_mask, delays_indexes[0]:delays_indexes[-2]] = self.states[next2_mask, delays_indexes[2]:delays_indexes[-1]+1]
        self.states[next2_mask, delays_indexes[-2]:delays_indexes[-1]+1] = self.apply_sign_invariant_sqrt_and_std(-self.raw_data[next2_mask, self.nb_past-2:self.nb_past, 0])

        self.states = self.states.reshape(s0, s1, -1)
        self.raw_data = self.raw_data.reshape(s0, s1, MAX_ITI_LEN, -1)

        # this will replace even values that dont change, but this is faster because GPU
        self.states[:, :, self.group_features_mapper['PLANNED_TIME_NUM']] = self.apply_sign_invariant_sqrt_and_std(self.raw_data[:,:,:self.nb_past+self.nb_future, 0])
        self.states[:, :, self.group_features_mapper['TYPES_D']] = self.raw_data[:,:,:lookup_range, 1]
        self.states[:, :, self.group_features_mapper['TYPES_P']] = self.raw_data[:,:,:lookup_range, 2]
        self.states[:, :, self.group_features_mapper['TYPES_A']] = self.raw_data[:,:,:lookup_range, 3]
        self.states[:, :, self.group_features_mapper['STATIONS_embedding']] = self.raw_data[:,:,:lookup_range, 4:4+self.emb_size].reshape(s0, s1, -1)
        self.states[:, :, self.group_features_mapper['LINES_embedding']] = self.raw_data[:,:,:lookup_range, 4+self.emb_size:4+2*self.emb_size].reshape(s0, s1, -1)
        
    def precompute_temporal_features(self, init_states_time: list, nb_steps: int, deltat: int) -> None:
        """
        Precompute cyclical temporal encodings (hour, day-of-year) for all possible future steps.

        Stores results in `self.temporal_features_values`, indexed by absolute state time.
        Note: relies on the fixed ordering of temporal features (hour first, then year).

        Args:
            init_states_time (list[int]): Initial reference times (seconds since 2012-01-01).
            nb_steps (int): Number of rollout steps to cover.
            deltat (int): Step size in seconds.

        Returns:
            None
        """
        times = set()
        for st in init_states_time:
            for i in range(nb_steps+1):
                times.add(st + i*deltat)
        times = list(times)
                
        datetimes = pd.Timestamp("2012-01-01") + pd.to_timedelta(times, unit='s')
        hours = (datetimes.hour + datetimes.minute / 60.0 + datetimes.second / 3600.0).to_numpy()
        days_of_year = (datetimes.dayofyear).to_numpy()
    
        hours_rep = torch.tensor(hours).unsqueeze(1).repeat(1, 6)  # Repeat each hour value 6 times
        days_rep = torch.tensor(days_of_year).unsqueeze(1).repeat(1, 6)  # Repeat each day_of_year value 6 times
        
        # Concatenate hours and days along the last dimension to form a tensor of shape [n, 12]
        out = torch.cat([hours_rep, days_rep], dim=1).to(self.device).to(torch.float32)
        frequencies = [1, 2, 4]
        for i in range(len(frequencies)):
            x = frequencies[i] * 2 * np.pi
            out[:,[2*i, 2*i+1]] *= x/24
            out[:,[2*i+6, 2*i+1+6]] *= x/365
    
        out[:, 0::2] = torch.sin(out[:, 0::2])
        out[:, 1::2] = torch.cos(out[:, 1::2])
        
        self.temporal_features_values = {times[i]: out[i] for i in range(len(times))}

    def precompute_day_of_week(self, init_states_time: list, nb_steps: int, deltat: int) -> None:
        """
        Precompute one-hot encodings of the day of week for all possible future steps.

        Stores results in `self.day_of_week_values`, indexed by absolute state time.

        Args:
            init_states_time (list[int]): Initial reference times (seconds since 2012-01-01).
            nb_steps (int): Number of rollout steps to cover.
            deltat (int): Step size in seconds.

        Returns:
            None
        """
        times = set()
        for st in init_states_time:
            for i in range(nb_steps+1):
                times.add(st + i*deltat)
        times = list(times)
        
        datetime = pd.Timestamp("2012-01-01") + pd.to_timedelta(times, unit='s')
        day_of_week = datetime.day_name()
        day_of_week_categories = np.array(self.cat_cols_md['day_of_week'])
        
        day_of_week_one_hot = np.array([day_of_week_categories == d for d in day_of_week])
        day_of_week_one_hot_tensor = torch.tensor(day_of_week_one_hot, dtype=torch.float32).to(self.device)
    
        self.day_of_week_values = {times[i]: day_of_week_one_hot_tensor[i] for i in range(len(times))}

    def apply_sign_invariant_sqrt_and_std(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sign-invariant square root transform and normalize by a fixed std.

        Args:
            x (torch.Tensor): Input values.

        Returns:
            torch.Tensor: Transformed and normalized values.
        """
        return (torch.sign(x) * torch.sqrt(torch.abs(x))) / 6

    def manage_new_and_old_trains(self, train_add_schedule: list, train_remove_schedules: list, sim, return_new_ids=False):
        """
        Manage removal of finished trains and addition of newly scheduled trains.

        Args:
            train_add_schedule (list): Per-simulation schedules for adding trains.
            train_remove_schedules (list): Per-sample schedules for removing trains.
            sim (Simulator): Simulator object.
            return_new_ids (bool): If True, return indices of inserted slots per simulation.

        Returns:
            None or list: If return_new_ids is True, returns a list of lists of indices of new trains.
        """
        max_seq_len = self.compute_new_max_seq_len(train_add_schedule, train_remove_schedules)
        if max_seq_len > self.states.shape[1]:  # if the max sequence increases, we extend the relevant tensors
            self.grow_tensors_to(max_seq_len, sim)
        self.remove_old_trains(train_remove_schedules)

        if return_new_ids:
            return self.add_new_trains(train_add_schedule, sim, return_new_ids=True)
        else:
            self.add_new_trains(train_add_schedule, sim, return_new_ids=False)

    def compute_new_max_seq_len(self, train_add_schedule: dict, train_remove_schedules: dict) -> int:
        """
        Compute the new maximum sequence length after applying scheduled additions and removals
        at each simulation's current state time.

        Args:
            train_add_schedule (list): Per-simulation schedules for adding trains.
            train_remove_schedules (list): Per-sample schedules for removing trains.

        Returns:
            int: The maximum sequence length needed across all simulations/samples after applying additions and
                removals at the current time
        """

        max_seq_len = 0
        for sim_id in range(self.nb_sim):
            st = int(self.states_time[sim_id*self.nb_samples])
            if st in train_add_schedule[sim_id]: # if we dont add trains then the size cant increase
                nb_new_train = len(train_add_schedule[sim_id][st])
                for sample_id in range(self.nb_samples):
                    idx = sim_id*self.nb_samples + sample_id
                    if st in train_remove_schedules[idx]:
                        nb_train_rem = len(train_remove_schedules[idx][st])
                    else:
                        nb_train_rem = 0

                    sample_seq_len = torch.sum(~self.padding_mask[idx]) + nb_new_train - nb_train_rem
                    max_seq_len = max(max_seq_len, sample_seq_len)
        return max_seq_len

    def grow_tensors_to(self, max_seq_len: int, sim) -> None:
        """
        Expand all state-related tensors and arrays to match a new maximum sequence length.

        Args:
            max_seq_len (int): Target maximum sequence length to grow tensors to.
            sim (Simulator): Simulator object that may also hold tensors (e.g., previous_prob) requiring resizing.

        Returns:
            None
        """
        diff = max_seq_len - self.states.shape[1]

        padding_states = torch.zeros((self.states.shape[0], diff, self.states.shape[2]), 
                                        dtype=self.states.dtype, device=self.states.device)
        self.states = torch.cat((self.states, padding_states), dim=1)

        padding_mask = torch.full((self.padding_mask.shape[0], diff), True, device=self.padding_mask.device)
        self.padding_mask = torch.cat((self.padding_mask, padding_mask), dim=1)

        padding_raw_data = torch.zeros((self.raw_data.shape[0], diff, self.raw_data.shape[2], self.raw_data.shape[3]), 
                                        dtype=self.raw_data.dtype, device=self.raw_data.device)
        self.raw_data = torch.cat((self.raw_data, padding_raw_data), dim=1)

        if self.mode == 'predict':
            padding_pred_buffer = torch.full((self.pred_buffer.shape[0], diff, self.pred_buffer.shape[2]), -1,
                                            dtype=self.pred_buffer.dtype, device=self.pred_buffer.device)
            self.pred_buffer = torch.cat((self.pred_buffer, padding_pred_buffer), dim=1)

        padding_pos = torch.zeros((self.positions.shape[0], diff), 
                                        dtype=self.positions.dtype, device=self.positions.device)
        self.positions = torch.cat((self.positions, padding_pos), dim=1)

        padding_maxpos = torch.zeros((self.max_positions.shape[0], diff), 
                                        dtype=self.max_positions.dtype, device=self.max_positions.device)
        self.max_positions = torch.cat((self.max_positions, padding_maxpos), dim=1)

        if self.mode == 'dcil':
            padding_expected_pos = torch.zeros((self.expected_pos.shape[0], diff, self.expected_pos.shape[2]), 
                                        dtype=self.expected_pos.dtype, device=self.expected_pos.device)
            self.expected_pos = torch.cat((self.expected_pos, padding_expected_pos), dim=1)

        padding_md = np.full((self.md.shape[0], diff, 3), None, dtype='object')
        self.md = np.concatenate((self.md, padding_md), axis=1)

        if hasattr(sim, "previous_prob"):
            pad = max_seq_len - sim.previous_prob.shape[1]
            if pad > 0:
                sim.previous_prob = torch.cat(
                    [sim.previous_prob,
                        torch.zeros(sim.previous_prob.shape[0], pad, 3, device=sim.device)],
                    dim=1
                )

    def remove_old_trains(self, train_remove_schedules: list) -> None:
        """
        Remove trains from the current state according to the given removal schedules.
        Also registers predictions for removed trains if the simulator is in 'predict' mode.

        Args:
            train_remove_schedules (list): Per-sample schedules for removing trains.

        Returns:
            None
        """
        for sim_id in range(self.nb_sim):
            st = int(self.states_time[sim_id*self.nb_samples])
            for sample_id in range(self.nb_samples):
                idx = sim_id*self.nb_samples + sample_id
                if st in train_remove_schedules[idx]:
                    for i in train_remove_schedules[idx][st]:
                        self.padding_mask[idx, i] = True
                        if self.mode == 'predict':
                            self.register_prediction(sim_id, sample_id, i)

    def add_new_trains(self, train_add_schedule: list, sim, return_new_ids: bool = False) -> list | None:
        """
        Add newly scheduled trains to the current simulation states and update all relevant tensors.
        Optionally return the indices of the inserted trains.

        Args:
            train_add_schedule (list): Per-simulation schedules for adding trains.
            sim (Simulator): Simulator object that may also hold tensors (e.g., previous_prob) requiring updates
                when new trains are added.
            return_new_ids (bool, optional): If True, returns indices of the newly inserted trains per simulation.
                Defaults to False.

        Returns:
            list | None: A list of lists with the indices of newly added trains if return_new_ids is True.
            Otherwise, returns None.
        """
        if return_new_ids:
            new_ids = []

        for sim_id in range(self.nb_sim):
            st = int(self.states_time[sim_id*self.nb_samples])
            if st in train_add_schedule[sim_id]:
                new_trains = train_add_schedule[sim_id][st]
                new_states = torch.stack([train['data'] for train in new_trains])
                new_raw_data = torch.stack([train['raw_data'] for train in new_trains])
                new_max_positions = torch.tensor([train['max_position'] for train in new_trains], dtype=self.max_positions.dtype).to(self.device)
                if self.mode == 'dcil':
                    new_expected_pos = torch.stack([train['expected_pos'] for train in new_trains])
                new_md = np.array([[train['datdep'],train['train_no'],self.nb_past-1] for train in new_trains], dtype='object')
                for sample_id in range(self.nb_samples):
                    idx = sim_id*self.nb_samples + sample_id
                    empty_indices = torch.nonzero(self.padding_mask[idx]).squeeze(1)[:len(new_trains)]
                    if return_new_ids : # because gail has self.nb_samples == 1, this works
                        new_ids.append(empty_indices.cpu().tolist())
                    self.padding_mask[idx, empty_indices] = False
                    self.positions[idx, empty_indices] = self.nb_past - 1
                    self.states[idx, empty_indices] = new_states.clone()
                    self.raw_data[idx, empty_indices] = new_raw_data.clone()
                    if self.mode == 'predict':
                        self.pred_buffer[idx, empty_indices] = -1 # reset pred buffer
                    if self.mode == 'dcil':
                        self.expected_pos[idx, empty_indices] = new_expected_pos.clone()
                    self.max_positions[idx, empty_indices] = new_max_positions.clone()
                    self.md[idx, empty_indices.cpu()] = new_md.copy()
                    if hasattr(sim, "previous_prob"):
                        sim.previous_prob[idx, empty_indices.cpu()] = 0
            else:
                if return_new_ids :
                    new_ids.append([])

        if return_new_ids :
            return new_ids

    def get_expected_action(self, step_id: int) -> tuple:
        """
        Compute one-hot actions that minimizes DCIL a* equation (cf paper) by retrieving the 
        action that minimizes the distance in the itinerary to expected pos at next step, also
        returns the difference for sample weight (cf paper).

        Args:
            step_id (int): Index of the rollout step (0-based) to compare against `expected_pos`.

        Returns:
            tuple: (actions, difference) where
                actions (torch.Tensor): One-hot action codes of shape (nb_sim*nb_samples, max_seq_len, 3).
                difference (torch.Tensor): Position deltas of shape (nb_sim*nb_samples, max_seq_len).
        """
        difference = (self.expected_pos[:,:,step_id] - self.positions)*~self.padding_mask*(self.max_positions != self.positions)
        actions = torch.clamp(difference, 0, 2)
        actions = F.one_hot(actions.long(), num_classes=3).float()
        
        return actions.cpu(), difference.cpu()

    def register_prediction(self, sim_id: int, sample_id: int, train_idx: int) -> None:
        """
        Register final prediction for a train that is being removed (predict mode).

        Args:
            sim_id (int): Simulation index in the batch.
            sample_id (int): Sample index within the simulation.
            train_idx (int): Train index within the simulation sequence.

        Returns:
            None
        """
        idx = sim_id * self.nb_samples + sample_id
        datdep, train_no, start_pos = self.md[idx, train_idx]
        end_pos = self.positions[idx, train_idx]
        mapper_idx = self.output[sim_id]['mapper'][(datdep,train_no)]
        self.output[sim_id]['end_pos'][mapper_idx,sample_id] = end_pos
        if start_pos != end_pos:
            self.output[sim_id]['predictions'][mapper_idx,sample_id] = self.pred_buffer[idx, train_idx]
    
    def register_remaining_predictions(self) -> None:
        """
        Finalize predictions for all trains still running at the end of simulation,
        parallelized only across trains of the same trajectory (predict mode).

        Args:
            None

        Returns:
            None
        """
        for sim_id in range(self.nb_sim):
            for sample_id in range(self.nb_samples):
                idx = sim_id*self.nb_samples + sample_id
                to_write_mask = ~self.padding_mask[idx]
                md = self.md[idx, to_write_mask.cpu().numpy(), :2]
                dest_idxs = torch.tensor([self.output[sim_id]['mapper'][tuple(md[i])] for i in range(len(md))], device = self.device)
                if not dest_idxs.any(): # if nothing to register, then continue
                    continue
                self.output[sim_id]['end_pos'][dest_idxs, sample_id] = self.positions[idx, to_write_mask]
                self.output[sim_id]['predictions'][dest_idxs, sample_id] = self.pred_buffer[idx, to_write_mask]
            roll_values = self.output[sim_id]['end_pos'] + 1
            for roll_val in roll_values.unique():
                if roll_val > 0:
                    mask = (roll_values == roll_val)
                    self.output[sim_id]['predictions'][mask, ] = self.output[sim_id]['predictions'][mask].roll(shifts=roll_val.item(), dims=-1)

    def update_local_features(self) -> None:
        """
        Update local neighbor count and delay features for each train.

        Computes pairwise distances between trains, counts neighbors within fixed
        radiuses, and averages their delays. Results are normalized and stored in
        the state tensor.

        Args:
            None

        Returns:
            None
        """
        radiuses = torch.tensor([0.1,0.3,0.6,1.0,2.0], device = self.device)
        normalizers = torch.tensor([2.0, 7.0, 22.0, 64.0, 262.0], device = self.device)

        past_1_idxs = self.group_features_mapper['STATIONS_embedding'][self.emb_size*(self.nb_past-1):self.emb_size*(self.nb_past)]
        future_1_idxs = self.group_features_mapper['STATIONS_embedding'][self.emb_size*(self.nb_past):self.emb_size*(self.nb_past+1)]

        coords = (self.states[:,:,past_1_idxs] + self.states[:,:,future_1_idxs]) / 2.0
        delays = self.states[:,:,self.group_features_mapper['PAST_DELAYS'][-1]]
        
        B, N = coords.shape[:2]
        R  = radiuses.shape[0]
        
        dist = torch.cdist(coords, coords, compute_mode='use_mm_for_euclid_dist')
        
        valid_pair = (~self.padding_mask)[:, :, None] & (~self.padding_mask)[:, None, :]
        valid_pair &= ~torch.eye(N, dtype=torch.bool, device=dist.device)
        
        mask = (dist.unsqueeze(1) <= radiuses.view(1, R, 1, 1))
        mask &= valid_pair.unsqueeze(1)
        
        sum_dels  = torch.einsum('brij,bj->bri', mask.float(), delays)
        counts    = mask.sum(dim=3).transpose(1,2)
        mean_delay = sum_dels.transpose(1,2) / counts.clamp(min=1)

        self.states[:,:,self.group_features_mapper['count_r']] = counts / normalizers
        self.states[:,:,self.group_features_mapper['mean_delay_r']] = mean_delay

class Simulator:
    """
    Run policy-driven rollouts over batched train states and itineraries for prediction, DCIL, or GAIL.

    Initializes feature-group mappings, moves the policy to device, and stores configuration
    for context lengths, embeddings, categorical metadata, and optional local neighbor features.

    Args:
        model: Policy model used to sample actions (e.g., torch.nn.Module or XGBoost wrapper).
        deltat (int): Time-step size in seconds.
        column_mapping (dict): Mapping from feature names to column indices.
        cat_cols_md (dict): Categorical columns metadata (e.g., one-hot categories).
        stations_emb (dict): Station embeddings mapping.
        lines_emb (dict): Line embeddings mapping.
        device (torch.device): Torch device for tensors and model.
        nb_past_stations (int): Number of past stations to include in context.
        nb_future_stations (int): Number of future stations to include in context.
        embedding_size (int): Dimensionality to use from station/line embeddings.
        idle_time_end (int): Maximum idle time threshold in seconds (used for constraints/logic elsewhere).
        net_type (str): Model type identifier (e.g., 'xgboost', 'mlp', 'transformer'); non-xgboost is put in eval mode and moved to device.
        local_features (bool): If True, compute local neighbor count and delay features during updates.

    Attributes:
        policy: Stored policy model (moved to device if net_type != 'xgboost' and set to eval()).
        deltat (int): Time-step size in seconds.
        group_features_mapper (dict): Grouped feature indices built from `column_mapping`.
        device (torch.device): Torch device used for tensors and model.
        nb_past (int): Past context length.
        nb_future (int): Future context length.
        emb_size (int): Embedding dimensionality in use.
        cat_cols_md (dict): Stored categorical columns metadata.
        idle_time_end (int): Maximum idle time threshold in seconds.
        stations_emb (dict): Station embeddings.
        lines_emb (dict): Line embeddings.
        net_type (str): Model type identifier.
        local_features (bool): Whether local neighbor features are enabled.
    """
    def __init__(self,
             model,
             deltat: int,
             column_mapping: dict,
             cat_cols_md: dict,
             stations_emb: dict,
             lines_emb: dict,
             device: torch.device,
             nb_past_stations: int,
             nb_future_stations: int,
             embedding_size: int,
             idle_time_end: int,
             net_type: str,
             local_features: bool = False) -> None:

        self.policy = model
        self.deltat = deltat
        self.group_features_mapper = get_group_feature_mapping(column_mapping,nb_past_stations,nb_future_stations,embedding_size, device, local_features)
        self.device = device
        if net_type != 'xgboost':
            self.policy.eval()
            self.policy = self.policy.to(device)
        self.nb_past = nb_past_stations
        self.nb_future = nb_future_stations
        self.emb_size = embedding_size
        self.cat_cols_md = cat_cols_md
        self.idle_time_end = idle_time_end
        self.stations_emb = stations_emb
        self.lines_emb = lines_emb
        self.net_type = net_type
        self.local_features = local_features

    def init_simulation(self, states: list, states_metadata: list, states_time: list,
                     nb_steps: int, nb_samples: int, itineraries: dict, mode: str) -> tuple:
        """
        Prepare itineraries, build StatesManager, and construct add/remove schedules.

        Loads or filters itineraries, creates schedules, extracts relevant subsets,
        instantiates StatesManager, precomputes temporal features, and builds the
        add/remove schedules used by the main loop.

        Args:
            states (list): Per-simulation state tensors, each (seq_len, nb_feats).
            states_metadata (list): Per-simulation metadata arrays.
            states_time (list): Per-simulation reference times (seconds since 2012-01-01).
            nb_steps (int): Number of rollout steps.
            nb_samples (int): Number of stochastic trajectories per simulation.
            itineraries (dict): Preloaded itineraries or None to load from dates.
            mode (str): One of {'predict','dcil','gail'}.

        Returns:
            tuple: (states_manager, itineraries, train_add_schedule, train_remove_schedules, schedules, appearance_times, trains_to_add) where
                states_manager (StatesManager): Batched state manager (uses max_seq_len to padded across simulations).
                itineraries (dict): Relevant itineraries for the batch.
                train_add_schedule (list): Per-simulation schedules for adding trains.
                train_remove_schedules (list): Per-sample schedules for removing trains.
                schedules (list): Raw schedules used to initialize predictions.
                appearance_times (dict): Mapping of train to first appearance time.
                trains_to_add (list): Trains detected for future insertion.
        """
        train_remove_schedules = []

        dates = {date for state_time in states_time for date in get_possible_dates(state_time)}
        if itineraries == None: # if we haven't pre-loaded the itineraries
            itineraries = load_itineraries_from_dates(dates)

        trains_to_add, appearance_times, schedules = self.get_trains_to_add(states_time, itineraries, dates, nb_steps)

        relevant_itineraries = self.extract_relevant_itineraries(states_metadata, trains_to_add, itineraries, nb_samples)
        
        states_manager = StatesManager(states, states_metadata, states_time, relevant_itineraries, schedules, nb_samples, self.nb_past, self.nb_future, self.emb_size, self.cat_cols_md,
                                       self.group_features_mapper, self.stations_emb, self.lines_emb, self.device, self.deltat, nb_steps, mode)
        states_manager.precompute_temporal_features(states_time, nb_steps, self.deltat)
        states_manager.precompute_day_of_week(states_time, nb_steps, self.deltat)
        dcil_flag = (mode == 'dcil')
        train_add_schedule = self.construct_new_trains(states_time, relevant_itineraries, states_manager, nb_steps, appearance_times, schedules, dcil=dcil_flag)
        for i in range(len(states_time)):
            initial_train_remove_schedule = self.get_initial_train_remove_schedule(relevant_itineraries, states_metadata,states_manager, i, nb_samples)
            for _ in range(nb_samples):
                train_remove_schedules.append(copy.deepcopy(initial_train_remove_schedule))

        return states_manager, itineraries, train_add_schedule, train_remove_schedules, schedules, appearance_times, trains_to_add

    def compute_nb_steps(self, predictive_horizon: int) -> int:
        """
        Compute the number of rollout steps from a horizon in minutes.

        Adds 10% extra steps to minimize translation completion.

        Args:
            predictive_horizon (int): Horizon in minutes.

        Returns:
            int: Number of steps (ceiled to int) using deltat.
        """
        nb_steps = predictive_horizon*60/self.deltat
        nb_steps += nb_steps/10 # Simulate 10% more to minimize translation completion
        nb_steps = int(nb_steps)
        return nb_steps


    def predict_delay(self, states: list, states_metadata: list, states_time: list, predictive_horizon: int,
                    nb_samples: int, sampling_method: str, action_constraint: bool, itineraries: dict = None) -> tuple:
        """
        Run a rollout in predict mode to generate delay predictions.

        Args:
            states (list): Per-simulation state tensors, each (seq_len, nb_feats).
            states_metadata (list): Per-simulation metadata arrays.
            states_time (list): Per-simulation reference times (seconds since 2012-01-01).
            predictive_horizon (int): Prediction horizon in minutes.
            nb_samples (int): Number of stochastic trajectories per simulation.
            sampling_method (str): Sampling strategy for the policy.
            action_constraint (bool): Whether to constrain actions.
            itineraries (dict, optional): Preloaded itineraries, or None to load from dates.

        Returns:
            tuple: (states_manager, itineraries) after running the simulation.
        """
        nb_steps = self.compute_nb_steps(predictive_horizon)
        states_manager, itineraries, train_add_schedule, train_remove_schedules, schedules, _, _ = \
            self.init_simulation(states, states_metadata, states_time, nb_steps, nb_samples, itineraries, 'predict')

        for _ in range(nb_steps):
            valid_actions_mask = self.get_valid_actions(states_manager)
            actions, prob = self.get_actions(states_manager, sampling_method, action_constraint, valid_actions_mask)
            states_manager.states_time += self.deltat
            states_manager.update_positions(actions)
            train_remove_schedules = self.update_train_remove_schedules(actions, states_manager, train_remove_schedules)
            states_manager.update(actions)
            states_manager.manage_new_and_old_trains(train_add_schedule, train_remove_schedules, self)
            if self.local_features:
                states_manager.update_local_features()

        states_manager.register_remaining_predictions()

        return states_manager, itineraries

    def get_samples_dcil(self, states: list, states_metadata: list, states_time: list, nb_steps: int,
                     nb_samples: int, sampling_method: str, action_constraint: bool,
                     itineraries: dict = None) -> list:
        """
        Run a rollout in DCIL mode and collect state/action/distance samples.

        Args:
            states (list): Per-simulation state tensors, each (seq_len, nb_feats).
            states_metadata (list): Per-simulation metadata arrays.
            states_time (list): Per-simulation reference times (seconds since 2012-01-01).
            nb_steps (int): Number of rollout steps.
            nb_samples (int): Number of stochastic trajectories per simulation.
            sampling_method (str): Sampling strategy for the policy.
            action_constraint (bool): Whether to constrain actions.
            itineraries (dict, optional): Preloaded itineraries, or None to load from dates.

        Returns:
            list: Collected samples, each tuple (state, expected_action, difference) with shapes:
                state (torch.Tensor): Features of shape (seq_len, nb_feats) for one train.
                expected_action (torch.Tensor): One-hot action codes of shape (seq_len, 3).
                difference (torch.Tensor): Position deltas of shape (seq_len,).
        """
        states_manager, itineraries, train_add_schedule, train_remove_schedules, schedules, _, _ = \
            self.init_simulation(states, states_metadata, states_time, nb_steps, nb_samples, itineraries, 'dcil')

        samples = []
        
        for step_id in range(nb_steps):
            valid_actions_mask = self.get_valid_actions(states_manager)
            actions, _ = self.get_actions(states_manager, sampling_method, action_constraint, valid_actions_mask)
            expected_actions, difference = states_manager.get_expected_action(step_id)
            states = states_manager.states.clone().cpu()
            mask = ~states_manager.padding_mask.clone().cpu()
            for i in range(states_manager.states.shape[0]):
                samples.append((states[i][mask[i]], expected_actions[i][mask[i]], difference[i][mask[i]]))
            states_manager.states_time += self.deltat
            states_manager.update_positions(actions)
            train_remove_schedules = self.update_train_remove_schedules(actions, states_manager, train_remove_schedules)
            states_manager.update(actions)
            states_manager.manage_new_and_old_trains(train_add_schedule, train_remove_schedules, self)
            if self.local_features:
                states_manager.update_local_features()

        return samples

    def get_samples_gail(self, states: list, states_metadata: list, states_time: list, nb_steps: int,
                     nb_samples: int, sampling_method: str, action_constraint: bool,
                     itineraries: dict = None) -> tuple:
        """
        Run a rollout in GAIL mode and collect states, actions, and probabilities.

        This method records all step-by-step data needed for adversarial training:
        the raw states, the chosen actions, their probabilities, and consistent IDs
        to track trains over time. Final states and IDs are also needed for PPO GAE 
        advantages computation.

        Args:
            states (list): Per-simulation state tensors, each (seq_len, nb_feats).
            states_metadata (list): Per-simulation metadata arrays.
            states_time (list): Per-simulation reference times (seconds since 2012-01-01).
            nb_steps (int): Number of rollout steps.
            nb_samples (int): Number of stochastic trajectories per simulation.
            sampling_method (str): Sampling strategy for the policy.
            action_constraint (bool): Whether to constrain actions.
            itineraries (dict, optional): Preloaded itineraries, or None to load from dates.

        Returns:
            tuple: (states_list, one_hot_actions_list, prob_list, ids_list, final_states, final_ids, valid_actions_mask_list) where
                states_list (list): Step-wise states, each (seq_len, nb_feats).
                one_hot_actions_list (list): Step-wise one-hot actions, each (seq_len, 3).
                prob_list (list): Step-wise action probabilities, each (seq_len, 3).
                ids_list (list): Step-wise train IDs, each (seq_len,).
                final_states (list): Final states per simulation, each (seq_len, nb_feats).
                final_ids (list): Final train IDs per simulation, each (seq_len,).
                valid_actions_mask_list (list): Step-wise valid action masks, each (seq_len, 3).
        """

        states_manager, itineraries, train_add_schedule, train_remove_schedules, schedules, _, _ = \
            self.init_simulation(states, states_metadata, states_time, nb_steps, nb_samples, itineraries, 'gail')
                
        nb_sim = states_manager.states.shape[0]

        nonpad = (~states_manager.padding_mask).to(torch.long)
        states_ids = (nonpad.cumsum(dim=-1) - 1) * nonpad
        next_ids = states_ids.max(dim=-1)[0] + 1

        states_list = [None]*nb_sim*nb_steps
        one_hot_actions_list = [None]*nb_sim*nb_steps
        prob_list = [None]*nb_sim*nb_steps
        ids_list = [None]*nb_sim*nb_steps
        valid_actions_mask_list = [None]*nb_sim*nb_steps
        
        for step_id in range(nb_steps):
            valid_actions_mask = self.get_valid_actions(states_manager)
            actions, prob = self.get_actions(states_manager, sampling_method, action_constraint, valid_actions_mask)
            states = states_manager.states.clone().cpu()
            one_hot_actions = F.one_hot(actions, num_classes=3).clone().cpu()
            prob = prob.clone().cpu()
            mask = ~states_manager.padding_mask.clone().cpu()
            for i in range(nb_sim):
                slot = i * nb_steps + step_id
                states_list[slot] = states[i][mask[i]]
                one_hot_actions_list[slot] = one_hot_actions[i][mask[i]]
                prob_list[slot] = prob[i][mask[i]]
                ids_list[slot] = states_ids[i][mask[i]].clone()
                valid_actions_mask_list[slot] = valid_actions_mask[i][mask[i]].cpu()
        
            states_manager.states_time += self.deltat
            states_manager.update_positions(actions)
            train_remove_schedules = self.update_train_remove_schedules(actions, states_manager, train_remove_schedules)
            states_manager.update(actions)
            new_ids = states_manager.manage_new_and_old_trains(train_add_schedule, train_remove_schedules, self, return_new_ids =True)
            if self.local_features:
                states_manager.update_local_features()
            new_cols = states_manager.states.shape[1] - states_ids.shape[1]

            if new_cols > 0: # need to augment the size of states_ids
                states_ids = F.pad(states_ids, pad=(0, new_cols), mode='constant', value=0)
            for i in range(nb_sim):
                for v in new_ids[i]:
                    states_ids[i,v] = next_ids[i]
                    next_ids[i] += 1

        states = states_manager.states.clone().cpu()
        mask = ~states_manager.padding_mask.clone().cpu()
        final_states = [states[i][mask[i]] for i in range(nb_sim)]
        final_ids = [states_ids[i][mask[i]].clone() for i in range(nb_sim)]

        return states_list, one_hot_actions_list, prob_list, ids_list, final_states, final_ids, valid_actions_mask_list

    def model_forward(self, states_manager: StatesManager) -> torch.Tensor:
        """
        Apply the policy model to current states and return action logits.

        Args:
            states_manager (StatesManager): Manager holding batched states of shape
                (nb_sim*nb_samples, max_seq_len, nb_feats) and padding_mask.

        Returns:
            torch.Tensor: Action logits of shape (nb_sim*nb_samples, max_seq_len, 3).
        """
        if self.net_type == 'transformer':
            logits = self.policy(states_manager.states, padding_mask=states_manager.padding_mask)
        elif self.net_type == 'mlp':
            B, L, F = states_manager.states.shape
            logits = self.policy(states_manager.states.reshape(B*L, F)).reshape(B, L, 3) 
        elif self.net_type == 'xgboost':
            B, L, F = states_manager.states.shape
            logits_np = self.policy.predict(states_manager.states.reshape(B*L, F).cpu().numpy(), output_margin=True)
            logits = torch.from_numpy(logits_np).to(states_manager.states.device).reshape(B, L, 3) 
        else:
            raise ValueError(f"Unknown net_type: {self.net_type!r}. Expected 'transformer' or 'mlp'.")

        return logits
    
    def get_actions(self, states_manager: StatesManager, sampling_method: str, action_constraint: bool,
                valid_action_mask: torch.Tensor) -> tuple:
        """
        Sample or select actions (depending on the samplig method) from 
        the policy given current states. Applies action constraint if required.

        Args:
            states_manager (StatesManager): Manager holding batched states of shape
                (nb_sim*nb_samples, max_seq_len, nb_feats) and padding_mask.
            sampling_method (str): Either "sampling" (multinomial) or "greedy" (argmax).
            action_constraint (bool): Whether to enforce monotonic constraints on action probabilities.
            valid_action_mask (torch.Tensor): Mask of valid actions of shape
                (nb_sim*nb_samples, max_seq_len, 3).

        Returns:
            tuple: (actions, probabilities) where
                actions (torch.Tensor): Chosen actions of shape (nb_sim*nb_samples, max_seq_len).
                probabilities (torch.Tensor): Action probabilities of shape (nb_sim*nb_samples, max_seq_len, 3).
        """
        logits = self.model_forward(states_manager)

        logits = logits.masked_fill(~valid_action_mask, float("-inf"))

        fully_padded = states_manager.padding_mask.all(dim=1)  # (batch_size,) detects samples with no train to avoid nans
        logits[fully_padded] = torch.zeros_like(logits[fully_padded])
        
        probabilities = torch.softmax(logits, dim=-1)  # shape [batch_size, seq_len, num_classes]
        batch_size, seq_len, num_classes = probabilities.shape        
        probabilities = probabilities.detach()
        probabilities_2d = probabilities.reshape(-1, num_classes)  # shape [batch_size * seq_len, num_classes]
        
        if action_constraint:
            if hasattr(self, "previous_prob"):
                previous = self.previous_prob.clone()
            else:
                previous = torch.zeros(batch_size, seq_len, num_classes, device=self.device)
            previous = previous.reshape(-1, num_classes).to(self.device)
            uses_previous = (probabilities_2d[:,1] <= previous[:,1]) & (previous.sum(dim=1) > 0) # if the prob of next is lower than last step and this is not padding
            probabilities_2d[uses_previous] = previous[uses_previous]
            self.previous_prob = probabilities_2d.reshape(batch_size,seq_len,num_classes)
    
        if sampling_method == "sampling":
            actions_2d = torch.multinomial(probabilities_2d, num_samples=1)  # shape [batch_size * seq_len, 1]
        elif sampling_method == "greedy":
            actions_2d = torch.argmax(probabilities_2d, dim=-1, keepdim=True)  # shape [batch_size * seq_len, 1]
        else:
            raise ValueError(f"Invalid sampling_method '{sampling_method}'. Choose 'sampling' or 'greedy'.")

        actions = actions_2d.view(batch_size, seq_len)  # shape [batch_size, seq_len]
        actions *= ~states_manager.padding_mask

        if action_constraint:
            self.previous_prob *= (actions.unsqueeze(-1).expand(-1, -1, 3) == 0) # resets prob to 0 for moving trains
        
        actions *= ~states_manager.padding_mask # set actions of padding trains as 0
        
        return actions, probabilities


    def get_valid_actions(self, states_manager: StatesManager) -> torch.Tensor:
        """
        Build a mask of valid actions for each train.

        Args:
            states_manager (StatesManager): Manager holding positions and max_positions
                of shape (nb_sim*nb_samples, max_seq_len).

        Returns:
            torch.Tensor: Boolean mask of shape (nb_sim*nb_samples, max_seq_len, 3),
                where True marks a valid action.
        """
        max_possible = states_manager.max_positions - states_manager.positions
        valid_actions_mask = torch.arange(3, device=states_manager.device) <= max_possible.unsqueeze(-1)

        return valid_actions_mask

    def get_iti_mapper(self, itineraries_index: dict, metadatas: list, max_seq_len: int,
                   nb_samples: int) -> torch.Tensor:
        """
        Build a padded mapper from (DATDEP, TRAIN_NO) metadata to itinerary row indices.

        Note: this is also used to create the train to add (tta) mapper. In that case,
        nb_samples = 1, len(metadata) = 1 and max_seq_len = nb_of_trains_to_add

        Args:
            itineraries_index (dict): Mapping from (datdep, train_no) to itinerary row index.
            metadatas (list): Per-simulation metadata arrays (DATDEP, TRAIN_NO).
            max_seq_len (int): Padded sequence length for the output mapper.
            nb_samples (int): Number of stochastic trajectories per simulation.

        Returns:
            torch.Tensor: Mapper tensor of shape (len(metadatas) * nb_samples, max_seq_len),
                filled with itinerary indices and -1 for padding.
        """
        ret = torch.full((len(metadatas)*nb_samples, max_seq_len), -1,device=self.device)
        for i in range(len(metadatas)):
            for j in range(len(metadatas[i])):
                ret[i*nb_samples:(i+1)*nb_samples,j] = itineraries_index[tuple(metadatas[i][j])]
        return ret

    def extract_relevant_itineraries(self, states_metadata: list, trains_to_add: np.ndarray,
                                 itineraries: dict, nb_samples: int) -> dict:
        """
        Gather only the itineraries needed for the batch and build index mappers.  
        Combines initial states and scheduled trains into a compact set of relevant
        itineraries with consistent indexing for later state and schedule updates.

        Args:
            states_metadata (list): Per-simulation metadata arrays (DATDEP, TRAIN_NO); lengths define max_seq_len per sim.
            trains_to_add (np.ndarray): Array of shape (n_trains, 2) with [datdep, train_no].
            itineraries (dict): Source itineraries by date with keys 'data', 'max_pos', 'rel_types', and 'train_no'.
            nb_samples (int): Number of stochastic trajectories per simulation.

        Returns:
            dict: Relevant itineraries and mappers with keys:
                'data' (torch.Tensor): Concatenated itinerary tensors of shape (n_relevant+1, MAX_ITI_LEN, feat), last row is padding.
                'max_pos' (torch.Tensor): Max positions of shape (n_relevant+1,).
                'rel_types' (torch.Tensor): Relation type ids of shape (n_relevant+1,).
                'states_iti_mapper' (torch.Tensor): Mapper of shape (nb_sim*nb_samples, max_seq_len) to rows in 'data' (−1 for padding).
                'tta_iti_mapper' (torch.Tensor): Mapper of shape (n_trains_to_add, 1) to rows in 'data'.
        """
        if len(trains_to_add) > 0:
            relevant_trains = np.concatenate((*states_metadata, trains_to_add))
        else: # Handles when there are no trains to add
            relevant_trains = np.concatenate(states_metadata)
    
        dates = np.unique(relevant_trains[:,0])

        indexes = []
        index_mapping = {}
        flat_index = 0
        for i, date in enumerate(dates):
            train_nos = np.unique(relevant_trains[relevant_trains[:,0] == date, 1])
            indexes.append([itineraries[date]['train_no'][train_no] for train_no in train_nos])
            for train_no in train_nos:
                index_mapping[(date, train_no)] = flat_index
                flat_index += 1
    
        index_mapping[(None, None)] = len(index_mapping)

        states_iti_mapper = self.get_iti_mapper(index_mapping, states_metadata, max([len(v) for v in states_metadata]), nb_samples)
        tta_iti_mapper = self.get_iti_mapper(index_mapping, [trains_to_add], len(trains_to_add), 1).transpose(0, 1)
        
        rel_itineraries = {
            'data': torch.cat([
                torch.cat([itineraries[dates[i]]['data'][indexes[i]].to(self.device) for i in range(len(dates))]),
                torch.zeros(1, *itineraries[dates[0]]['data'][0].shape,dtype=torch.int32).to(self.device)
            ]),
            'max_pos': torch.cat([
                torch.cat([itineraries[dates[i]]['max_pos'][indexes[i]].to(self.device) for i in range(len(dates))]),
                torch.tensor([0]).to(self.device)
            ]),
            'rel_types': torch.cat([
                torch.cat([itineraries[dates[i]]['rel_types'][indexes[i]].to(self.device) for i in range(len(dates))]),
                torch.tensor([0]).to(self.device)
            ]),
            'states_iti_mapper':states_iti_mapper,
            'tta_iti_mapper':tta_iti_mapper
        }

        return rel_itineraries            

    def get_trains_to_add(self, states_time: list, itineraries: dict, dates: list, nb_steps: int) -> tuple:
        """
        Build future add-schedules by scanning itineraries within each simulation window.

        Args:
            states_time (list): Per-simulation start times (seconds since 2012-01-01).
            itineraries (dict): Per-date itineraries with 'data' and 'train_no' lookups.
            dates (list): Candidate date keys to scan in `itineraries`.
            nb_steps (int): Number of rollout steps to define each simulation window (uses deltat).

        Returns:
            tuple: (trains_to_add, appearance_times, schedules) where
                trains_to_add (np.ndarray): Array of shape (n_trains, 2) with [datdep, train_no].
                appearance_times (np.ndarray): First real appearance times of shape (n_trains,).
                schedules (list): One dict per simulation mapping time (int) -> list of ((datdep, train_no), train_idx).
        """
        schedules = [defaultdict(list) for _ in range(len(states_time))]
        possible_bounds = [[st, st + self.deltat * nb_steps] for st in states_time]
        trains_to_add = []
        appearance_times = []
        added_trains = {}  # Map from (datdep, train_no) to index in trains_to_add
    
        for datdep in dates:
            real_appearance_time = itineraries[datdep]['data'][:, 0, 4]
            th_appearance_time = itineraries[datdep]['data'][:, 0, 3]
            train_no_list = list(itineraries[datdep]['train_no'].keys())
    
            for i, bounds in enumerate(possible_bounds): # possibly here we can optimize by precomputing which bound goes in which date(s) ...
                to_add_mask = (real_appearance_time > bounds[0]) & (th_appearance_time < bounds[1])
    
                for idx in torch.where(to_add_mask)[0]:
                    train_no = train_no_list[idx]
                    train_pair = (datdep, train_no)
    
                    if train_pair not in added_trains:
                        index_in_trains_to_add = len(trains_to_add)
                        trains_to_add.append([datdep, train_no])
                        appearance_times.append(real_appearance_time[idx].item())
                        added_trains[train_pair] = index_in_trains_to_add
                    else:
                        index_in_trains_to_add = added_trains[train_pair]
    
                    schedules[i][real_appearance_time[idx].item()].append((train_pair, index_in_trains_to_add))
        return np.array(trains_to_add, dtype=object), np.array(appearance_times), schedules

    def construct_new_trains(self, states_time: list, itineraries: dict, states_manager: StatesManager,
                         nb_steps: int, appearance_times: np.ndarray, schedules: list,
                         dcil: bool = False) -> list:
        """
        Build per-simulation add-schedules and precompute initial features (states) for future trains.

        Args:
            states_time (list): Per-simulation start times (seconds since 2012-01-01).
            itineraries (dict): Relevant itineraries and mappers, including 'data', 'max_pos', 'rel_types', 'tta_iti_mapper'.
            states_manager (StatesManager): Manager used to create initial feature vectors from raw data.
            nb_steps (int): Number of rollout steps (used to compute expected positions when dcil is True).
            appearance_times (np.ndarray): Planned appearance times for scheduled trains, shape (n_trains,).
            schedules (list): One dict per simulation mapping time (int) -> list of ((datdep, train_no), index_in_trains_to_add).
            dcil (bool): If True, include 'expected_pos' per train over nb_steps.

        Returns:
            list: One dict per simulation mapping time (int) -> list of train items, where each item contains:
                'data' (torch.Tensor): Flat feature vector of shape (nb_feats,).
                'raw_data' (torch.Tensor): Raw per-train buffer of shape (MAX_ITI_LEN, nb_raw_data_feat).
                'max_position' (int): Maximum valid position on the itinerary.
                'train_no' (int): Train identifier.
                'datdep' (str): Date key used in itineraries.
                'expected_pos' (torch.Tensor): Only when dcil is True, shape (nb_steps,).
        """
        train_add_schedules = [defaultdict(list) for _ in range(len(states_time))]
        if len(appearance_times) == 0:
            return train_add_schedules
        raw_data = states_manager.init_raw_data(itineraries, 'tta_iti_mapper', torch.full((1, len(appearance_times)),4), torch.tensor(appearance_times).to(self.device)).squeeze(1)
        rel_types = itineraries['rel_types'][itineraries['tta_iti_mapper']].squeeze(1)
        data = states_manager.create_data(raw_data, appearance_times, rel_types)

        if dcil:
            obs = itineraries['data'][itineraries['tta_iti_mapper']][:,0,:,4].unsqueeze(-1).repeat(1,1,nb_steps)
            max_pos = itineraries['max_pos'][itineraries['tta_iti_mapper']][:,0].unsqueeze(-1).repeat(1,nb_steps)

        for i in range(len(schedules)):
            if dcil:
                steps_states_times = (torch.tensor(states_time[i]).view(1,-1).repeat(MAX_ITI_LEN, nb_steps) + torch.arange(self.deltat,self.deltat*(nb_steps+1), self.deltat)).to(self.device)
            for state_time in schedules[i]:
                for md, index in schedules[i][state_time]:
                    item = {
                        'data': data[index],
                        'raw_data': raw_data[index],
                        'max_position': itineraries['max_pos'][itineraries['tta_iti_mapper'][index]].item(),
                        'train_no': md[1],
                        'datdep': md[0],
                    }
                    if dcil:
                        expected_pos = max_pos[index] - (obs[index] > steps_states_times).sum(axis = 0) + self.nb_future
                        item['expected_pos'] = expected_pos

                    train_add_schedules[i][state_time].append(item)
            
        return train_add_schedules

    def get_initial_train_remove_schedule(self, itineraries: dict, metadatas: list,
                                      states_manager: StatesManager, idx: int, nb_samples: int) -> dict:
        """
        Build the initial per-time removal schedule for trains already at their final position.

        Args:
            itineraries (dict): Itinerary data and mappers, including 'data', 'max_pos', and 'states_iti_mapper'.
            metadatas (list): Per-simulation metadata arrays (DATDEP, TRAIN_NO); length defines max_seq_len per sim.
            states_manager: Object holding positions, max_positions, and padding_mask of shape (nb_sim*nb_samples, max_seq_len).
            idx (int): Simulation index (0-based) whose initial schedule is constructed.
            nb_samples (int): Number of trajectories per simulation (the schedule is derived from idx*nb_samples).

        Returns:
            dict: Mapping from removal_time (int) to list of train indices (list) in [0, max_seq_len),
                for trains at their max position at initialization.
        """
        adjusted_idx = idx * nb_samples
        to_remove_mask = (states_manager.positions[adjusted_idx] == states_manager.max_positions[adjusted_idx]) & ~states_manager.padding_mask[adjusted_idx]
        train_remove_schedule = defaultdict(list)
    
        for i in torch.where(to_remove_mask)[0]:
            datdep, train_no = metadatas[idx][i]
            index = itineraries['states_iti_mapper'][adjusted_idx, i]
            removal_time = int(itineraries['data'][index, itineraries['max_pos'][index] + 1, 4])
            train_remove_schedule[removal_time].append(i)
    
        return train_remove_schedule

    def update_train_remove_schedules(self, actions: torch.Tensor, states_manager: StatesManager,
                                  train_remove_schedules: list) -> list:
        """
        Update per-sample removal schedules for trains that reached max position and moved this step.

        Args:
            actions (torch.Tensor): Action codes of shape (nb_sim*nb_samples, max_seq_len).
            states_manager (StatesManager): Manager holding positions, max_positions, states_time, and nb_sim/nb_samples.
            train_remove_schedules (list): Per-sample dicts mapping removal_time (int) to list of train indices.

        Returns:
            list: Updated `train_remove_schedules` with new indices appended at time
                states_time + idle_time_end*60 for each affected train.
        """
        stopped_trains = (states_manager.positions == states_manager.max_positions)*(actions != 0)
        relevant_indices = [torch.nonzero(stopped_trains[i]).squeeze(1).tolist() for i in range(states_manager.nb_sim * states_manager.nb_samples)]
        for sim_id in range(states_manager.nb_sim):
            removal_time = int(states_manager.states_time[sim_id*states_manager.nb_samples] + self.idle_time_end*60)
            for sample_id in range(states_manager.nb_samples):
                idx = sim_id*states_manager.nb_samples + sample_id
                for i in relevant_indices[idx]:
                    if removal_time not in train_remove_schedules[idx]:
                        train_remove_schedules[idx][removal_time] = []
                    train_remove_schedules[idx][removal_time].append(i)

        return train_remove_schedules

    def reset(self) -> None:
        """
        Reset simulator state by clearing cached probabilities if present.

        Returns:
            None
        """
        if hasattr(self, 'previous_prob'):
            del self.previous_prob
