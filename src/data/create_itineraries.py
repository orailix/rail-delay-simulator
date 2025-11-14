import os
import torch
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm

from src.data.raw_data_processing import load_and_filter_col, set_stations_types, get_numerical_times, delete_train_with_missing_value, encode_itinerary
from src.utils.utils import load_pickle, save_pickle

def save_itineraries_as_dict(input_folder_path: str, stations_emb_path: str, lines_emb_path: str, out_folder_path: str, first_year: int, first_month: int, 
            last_year: int, last_month: int, deltat: int, idle_time_beggining: int, idle_time_end: int, nb_past_st: int, nb_future_st: int) -> None:
    """
    Build fixed-length itinerary tensors per departure date and save them to disk.

    Each output file contains a dictionary with:
        - data (torch.Tensor): Batched itineraries of shape (num_trains, MAX_ITI_LEN, 5),
          where columns correspond to [station_idx, type_idx, line_idx, planned_time, observed_time].
        - train_no (dict): Mapping of train numbers to their row index in the batch.
        - rel_types (torch.Tensor): Relation type indices per train.
        - max_pos (torch.Tensor): Index of the last real station before placeholder padding.

    This format allows very optimized creation of data structures used in simulation by allowing fully vectorized operations.

    Args:
        input_folder_path (str): Path to the folder containing raw monthly punctuality CSV files.
        stations_emb_path (str): Path to the pickle file with station embeddings (including placeholders).
        lines_emb_path (str): Path to the pickle file with line embeddings (including placeholder).
        out_folder_path (str): Output folder where per-date .pt files will be written.
        first_year (int): Starting year of the data to process, inclusive.
        first_month (int): Starting month of the data to process, inclusive.
        last_year (int): Ending year of the data to process, inclusive.
        last_month (int): Ending month of the data to process, inclusive.
        deltat (int): Time-step size in seconds used to discretize times.
        idle_time_beggining (int): Minutes to prepend before the first theoretical/observed time.
        idle_time_end (int): Minutes to append after the last theoretical/observed time.
        nb_past_st (int): Number of placeholder past stations to prepend.
        nb_future_st (int): Number of placeholder future stations to append.

    Returns:
        None
    """

    relevant_cols = ['PTCAR_LG_NM_NL', 'type', 'LINE_NO' ,'PLANNED_TIME_NUM', 'REAL_TIME_NUM']
    os.makedirs(out_folder_path, exist_ok=True)

    file_names = []

    years = np.arange(first_year, last_year+1, 1)

    for year in years:
        start_month = first_month if year == first_year else 1
        end_month   = last_month  if year == last_year  else 12
        for month in range(start_month, end_month + 1):
            file_names.append(f"Data_raw_punctuality_{year}{month:02d}.csv")

    MAX_ITI_LEN = 130 + nb_past_st + nb_future_st # maximum length of an itinerary (computed on the paper data span)

    st_emb = load_pickle(stations_emb_path)
    string_to_index_station = {s: i for i, s in enumerate(st_emb.keys())}
    pl_beg_id = string_to_index_station['placeholder_begin_station']
    pl_beg_end = string_to_index_station['placeholder_end_station']
    
    l_emb = load_pickle(lines_emb_path)
    string_to_index_line = {s: i for i, s in enumerate(l_emb.keys())}
    pl_line = string_to_index_line['placeholder_line']
    
    types_mapper = {v:i for i, v in enumerate(['D', 'P', 'A'])}
    rel_type_mapper = {v:i for i, v in enumerate(['ICE','INT','IC','EURST','THAL','L','P','EXTRA','TGV','CHARTER', 'ICT','IZY'])}
    
    for file_name in file_names:
        print(file_name)
        path = os.path.join(input_folder_path, file_name)
        df = load_and_filter_col(path)
        df = set_stations_types(df)
        df = encode_itinerary(df)
        df = delete_train_with_missing_value(df)
        df = get_numerical_times(df, deltat)
        groups = df.groupby(['DATDEP','TRAIN_NO'])
        itineraries = {}

        for datdep, g1 in tqdm(df.groupby(['DATDEP'])):
            train_nos = {}
            rel_types = []
            max_pos = []
            tensor = []
            
            for i, (train_no, g2) in enumerate(g1.groupby(['TRAIN_NO'])):
                data = g2[relevant_cols].values
                observed_times = data[:, 4]

                if not np.all(np.diff(observed_times) >= 0):
                    prev = observed_times[:-1] 
                    cur = observed_times[1:]
                    delta  = prev - cur
                    
                    glitch = (delta > 0) & (delta <= 180)
                    observed_times[1:][glitch] = prev[glitch]
                    
                    if not np.all(np.diff(observed_times) >= 0):
                        continue

                    data[:, 4] = observed_times

                begin_th = data[0, 3] - 60 * idle_time_beggining
                begin_obs = min(data[0, 3] - 60 * idle_time_beggining, data[0, 4])
                end_th = data[-1, 3] + 60 * idle_time_end
                end_obs = data[-1, 4] + 60 * idle_time_end
                
                if begin_obs < begin_th - 10 * (60 * idle_time_beggining):
                    continue
        
                begin_rows = np.array([
                    [pl_beg_id, 0, pl_line, begin_th, begin_obs]
                ] * nb_past_st, dtype=np.int32)
        
                end_rows = np.array([
                    [pl_beg_end, 2, pl_line, end_th, end_obs]
                ] * nb_future_st, dtype=np.int32)
                
                # Convert station names and line names to indices
                replace_station = np.vectorize(lambda x: string_to_index_station.get(x, pl_beg_end))
                replace_line = np.vectorize(lambda x: string_to_index_line.get(x, pl_line))
                replace_type = np.vectorize(lambda x: types_mapper.get(x, 2))  # Default to 2 for unknown types
        
                data[:, 0] = replace_station(data[:, 0])  # Convert station names
                data[:, 1] = replace_type(data[:, 1])  # Convert relation type
                data[:, 2] = replace_line(data[:, 2])  # Convert line names
        
                array = np.vstack([begin_rows, data, end_rows]).astype(int)
                current_tensor = torch.zeros((MAX_ITI_LEN, 5), dtype=torch.int32)
                current_tensor[:len(array), :] = torch.tensor(array)
                tensor.append(current_tensor)
        
                # Assign index for this train number
                train_nos[train_no[0]] = len(train_nos)
                
                # Assign relation type and max position
                rel_types.append(rel_type_mapper[g2['RELATION_TYPE'].iloc[0]])
                max_pos.append(len(array) - nb_future_st - 1)
        
            # Convert lists to tensors after processing
            tensor = torch.stack(tensor)  # Stack the tensors into a single tensor, respecting the MAX_ITI_LEN constraint
            rel_types = torch.tensor(rel_types, dtype=torch.int32)
            max_pos = torch.tensor(max_pos, dtype=torch.int32)

            torch.save({
                'data': tensor,
                'train_no':train_nos,
                'rel_types':rel_types,
                'max_pos':max_pos, 
            }, os.path.join(out_folder_path, f'itineraries_{datdep[0]}.pt'))

def main() -> None:
    """
    Entry point to construct and save itinerary tensors.

    Parses CLI arguments, loads embeddings, processes monthly raw CSVs
    over the specified date range, encodes itineraries into fixed-length
    tensors, and saves one .pt file per departure date to the output folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_in_path", type=str, help="Path of the folder of the raw data.")
    parser.add_argument("folder_out_path", type=str, help="Path of the folder the itineraries will be stored in.")
    parser.add_argument("stations_emb_path", type=str, help="Stations embeddings path.")
    parser.add_argument("lines_emb_path", type=str, help="Lines embeddings path.")
    parser.add_argument("first_year", type=int, help="Year of the first file.")
    parser.add_argument("first_month", type=int, help="Month of the first file.")
    parser.add_argument("last_year", type=int, help="Year of the last file.")
    parser.add_argument("last_month", type=int, help="Month of the last file.")
    parser.add_argument("deltat", type=int, help="Number of seconds of each time step.")
    parser.add_argument("nb_past_stations", type=int, help="Number of past stations kept as features.")
    parser.add_argument("nb_future_stations", type=int, help="Number of future stations kept as features.")
    parser.add_argument("idle_time_beggining", type=int, help="Number of minutes the train will be on the network before it starts.")
    parser.add_argument("idle_time_end", type=int, help="Number of minutes the train will be on the network before it disapears after reaching the last station.")
    
    args = parser.parse_args()
    print(args)

    save_itineraries_as_dict(args.folder_in_path, args.stations_emb_path, args.lines_emb_path, args.folder_out_path, args.first_year, args.first_month, args.last_year, args.last_month, args.deltat, args.idle_time_beggining, args.idle_time_end, args.nb_past_stations, args.nb_future_stations)

if __name__ == "__main__":
    main()