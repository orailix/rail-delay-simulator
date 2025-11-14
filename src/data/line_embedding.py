import pickle
import os
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.raw_data_processing import load_and_filter_col,set_stations_types
from src.utils.utils import load_pickle, save_pickle

def create_line_station_dict(data_folder_path: str, first_year: int, first_month: int, last_year: int, last_month: int) -> dict:
    """
    Build a mapping from line number to the set of station names across the given month range.

    Args:
        data_folder_path (str): Folder containing monthly CSV files.
        first_year (int): Year of the first file to include.
        first_month (int): Month (1–12) of the first file to include.
        last_year (int): Year of the last file to include.
        last_month (int): Month (1–12) of the last file to include.

    Returns:
        dict: Map {line_no: {station_name, ...}} aggregated over all files.
    """
    link_dict = {}
    
    years = np.arange(first_year, last_year+1, 1)

    for year in years:
        start_month = first_month if year == first_year else 1
        end_month   = last_month  if year == last_year  else 12
        for month in range(start_month, end_month + 1):
            mm = f"{month:02d}"
            path = os.path.join(data_folder_path, f'Data_raw_punctuality_{year}{mm}.csv')
            print(f"Extracting links in {path}") 
            df = load_and_filter_col(path)
            df = set_stations_types(df)
            df = df.fillna('NaN')

            for line in tqdm(df['LINE_NO'].unique()):
                if line not in link_dict:
                    link_dict[line] = set()
                for station in df[df['LINE_NO'] == line]['PTCAR_LG_NM_NL'].unique():
                    link_dict[line].add(station)
    return link_dict

def create_line_embeddings(link_dict: dict, st_emb_path: str) -> dict:
    """
    Create line-level embeddings by averaging embeddings of stations belonging to each line.

    Args:
        link_dict (dict): Mapping of line numbers to sets of station names.
        st_emb_path (str): Path to the pickle file containing station embeddings.

    Returns:
        dict: Mapping of line numbers to mean embedding vectors. 
              Includes a 'placeholder_line' with a zero vector of the same dimension.
    """
    embeddings = load_pickle(st_emb_path)
    dim = len(embeddings[list(embeddings)[0]])

    embeddings_line = {}
    for line, stations in link_dict.items():
        station_embeddings = [embeddings[station] for station in stations]
        embeddings_line[line] = np.mean(station_embeddings, axis=0)

    embeddings_line['placeholder_line'] = np.zeros(dim)

    return embeddings_line

def save_line_embeddings(line_embeddings: dict, save_folder_path: str) -> None:
    """
    Save line embeddings to a pickle file in the given folder.

    Args:
        line_embeddings (dict): Dictionary of line embeddings.
        save_folder_path (str): Path to the folder where embeddings will be saved.

    Returns:
        None
    """

    dim = len(line_embeddings[list(line_embeddings)[0]])
    path = os.path.join(save_folder_path, f'lines_emb_{dim}.pkl')

    print(f'Saving embeddings at {path}')
    save_pickle(line_embeddings, path)
    

def main():
    """ 
    Entry point for generating line embeddings.

    Parses command-line arguments, builds line-to-station mappings,
    creates line embeddings from station embeddings, and saves them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder_path", type=str, help="Raw data folder path.")
    parser.add_argument("st_emb_path", type=str, help="Station embeddings path.")
    parser.add_argument("save_folder_path", type=str, help="Path of the folder to save the data.")
    parser.add_argument("first_year", type=int, help="Year of the first file.")
    parser.add_argument("first_month", type=int, help="Month of the first file.")
    parser.add_argument("last_year", type=int, help="Year of the last file.")
    parser.add_argument("last_month", type=int, help="Month of the last file.")

    args = parser.parse_args()

    link_dict = create_line_station_dict(args.data_folder_path, args.first_year, args.first_month, args.last_year, args.last_month)
    line_embeddings = create_line_embeddings(link_dict, args.st_emb_path)
    save_line_embeddings(line_embeddings, args.save_folder_path)

if __name__ == "__main__":
    main()