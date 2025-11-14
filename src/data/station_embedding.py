import glob
import os
import math
import argparse

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from collections import defaultdict
from tqdm import tqdm

from src.utils.utils import load_pickle, save_pickle

def create_adj_dictionnary(data_folder_path: str, first_year: int, first_month: int, last_year: int, last_month: int) -> dict:
    """
    Create an adjacency dictionary of stations from punctuality CSV files.

    Args:
        data_folder_path (str): Path to the folder containing punctuality CSV files.
        first_year (int): Starting year of the data to process, inclusive.
        first_month (int): Starting month of the data to process, inclusive.
        last_year (int): Ending year of the data to process, inclusive.
        last_month (int): Ending month of the data to process, inclusive.

    Returns:
        dict: Adjacency dictionary where keys are station names and values are sets of neighboring stations.
    """
    adj_dict = {}

    years = np.arange(first_year, last_year+1, 1)

    for year in years:
        start_month = first_month if year == first_year else 1
        end_month   = last_month  if year == last_year  else 12
        for month in range(start_month, end_month + 1):
            mm = f"{month:02d}"
            path = os.path.join(data_folder_path, f'Data_raw_punctuality_{year}{mm}.csv')
            print(f"Extracting links in {path}") 
            df = pd.read_csv(path, usecols = ["TRAIN_NO", "DATDEP", "PTCAR_LG_NM_NL"])

            for train_no, group in tqdm(df.groupby(['TRAIN_NO', 'DATDEP'])):
                itinerary = group['PTCAR_LG_NM_NL'].values
                for i in range(len(itinerary)):
                    station = itinerary[i]
                    if station not in adj_dict:
                        adj_dict[station] = set()
                    if i > 0:
                        adj_dict[itinerary[i-1]].add(station)
                    if i < len(itinerary) - 1:
                        adj_dict[station].add(itinerary[i+1])
    return adj_dict

def create_embeddings(adj_dict: dict, dim: int) -> tuple:
    """
    Create node embeddings from an adjacency dictionary using Laplacian eigenmaps.

    Args:
        adj_dict (dict): Adjacency dictionary where keys are nodes and values are sets of neighboring nodes.
        dim (int): Dimension of the embedding space.

    Returns:
        tuple: (embeddings, G) where
            embeddings (numpy.ndarray): Node embeddings of shape (number of nodes, dim).
            G (networkx.Graph): Graph constructed from the adjacency dictionary.
    """

    nodes = list(adj_dict.keys())
    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=int)
    for i, node in enumerate(nodes):
        for neighbor in adj_dict[node]:
            j = nodes.index(neighbor)
            adj_matrix[i][j] = 1
    np.fill_diagonal(adj_matrix, 0)
    G = nx.from_numpy_array(adj_matrix)
    L = nx.normalized_laplacian_matrix(nx.to_undirected(G)).toarray()
    eigenvalues, eigenvectors = np.linalg.eig(L)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    zero_mask = np.abs(eigenvalues) < 1e-12
    nonzero_idx = np.where(~zero_mask)[0]
    embeddings = eigenvectors[:, nonzero_idx[:dim]]

    embeddings = embeddings / np.maximum(1e-8, np.linalg.norm(embeddings, axis=1, keepdims=True)) # Handles norm = 0 for isolated stations
    return embeddings, G

def get_positions(adj_dict: dict) -> dict:
    """
    Get geographic positions for stations from an adjacency dictionary.

    Args:
        adj_dict (dict): Adjacency dictionary where keys are station names and values are sets of neighboring stations.

    Returns:
        dict: Dictionary mapping station indices to (latitude, longitude) coordinates.
    """
    geospace_df = pd.read_csv('data/Stations/stations_geospace.csv')
    positions = {}
    for i, station in enumerate(adj_dict.keys()):
        match = geospace_df[geospace_df['PTCAR_LG_NM_NL'] == station]
        if match.empty:
            continue
        geo_point = match['Geo Point'].values[0]
        if pd.isna(geo_point):
            continue
        try:
            lon, lat = geo_point.split(', ')
            positions[i] = (float(lat), float(lon))
        except ValueError:
            continue
    return positions


def plot_embeddings(G: nx.Graph, embeddings: np.ndarray, positions: dict, nb_dim: int, size: int) -> None:
    """
    Plot node embeddings on a graph using geographic positions.

    Args:
        G (networkx.Graph): Graph constructed from the adjacency dictionary.
        embeddings (numpy.ndarray): Node embeddings of shape (number of nodes, embedding dimension).
        positions (dict): Dictionary mapping node indices to (latitude, longitude) coordinates.
        nb_dim (int): Number of embedding dimensions to plot.
        size (int): Figure size for the plot.

    Returns:
        None
    """

    embeddings = embeddings[:-2, :]
    nrows = int(nb_dim / 2)
    fig, axs = plt.subplots(nrows, 2, figsize=(size, size))
    axs = axs.flatten()

    vmin = np.min(embeddings[:, :nb_dim])
    vmax = np.max(embeddings[:, :nb_dim])
    sm = ScalarMappable(cmap='plasma', norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    nodes_with_pos = list(positions.keys())
    subgraph = G.subgraph(nodes_with_pos)

    for idx in range(nb_dim):
        node_colors = embeddings[nodes_with_pos, idx]
        nx.draw(
            subgraph,
            pos={n: positions[n] for n in nodes_with_pos},
            ax=axs[idx],
            with_labels=False,
            node_size=10,
            node_color=node_colors,
            edge_color='gray',
            vmin=vmin,
            vmax=vmax,
            cmap='plasma'
        )

        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        axs[idx].set_title(f'Dim {idx}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Embedding Value', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_station_embedding_distance(G: nx.Graph, adj_dict: dict, embeddings: np.ndarray, positions: dict, selected_node) -> None:
    """
    Plot embedding-based distances of stations from a selected node on the graph.

    Args:
        G (networkx.Graph): Graph constructed from the adjacency dictionary.
        adj_dict (dict): Adjacency dictionary where keys are station names and values are sets of neighboring stations.
        embeddings (numpy.ndarray): Node embeddings of shape (number of nodes, embedding dimension).
        positions (dict): Dictionary mapping node indices to (latitude, longitude) coordinates.
        selected_node (int or str): Node index, station name, or 'random' to select a random node.

    Returns:
        None
    """
    embeddings = embeddings[:-2, :]

    if selected_node == 'random':
        selected_node_index = np.random.randint(embeddings.shape[0])
    elif isinstance(selected_node, str):
        try:
            selected_node_index = list(adj_dict.keys()).index(selected_node)
        except ValueError:
            raise ValueError(f"Station {selected_node} not found in adjacency dictionary.")
    else:
        selected_node_index = selected_node

    distances = np.linalg.norm(embeddings - embeddings[selected_node_index], axis=1)
    normalized_distances = 1 - ((distances - np.min(distances)) / (np.max(distances) - np.min(distances)))

    cmap = get_cmap('YlOrRd')
    norm = Normalize(vmin=0, vmax=1)

    nodes_with_pos = list(positions.keys())
    subgraph = G.subgraph(nodes_with_pos)

    node_colors = [cmap(norm(value)) for value in normalized_distances[nodes_with_pos]]

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw(
        subgraph,
        pos={n: positions[n] for n in nodes_with_pos},
        ax=ax,
        with_labels=False,
        node_size=10,
        node_color=node_colors,
        edge_color='gray'
    )

    if selected_node_index in nodes_with_pos:
        nx.draw_networkx_nodes(
            subgraph,
            pos={selected_node_index: positions[selected_node_index]},
            ax=ax,
            nodelist=[selected_node_index],
            node_color='green',
            node_size=20
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Normalized proximity to Selected Node in the embedding space', rotation=270, labelpad=15)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Selected Node',
                              markerfacecolor='g', markersize=10)]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


def save_embeddings(save_folder_path: str, adj_dict: dict, embeddings: np.ndarray) -> None:
    """
    Save station embeddings to a pickle file.

    Args:
        save_folder_path (str): Path to the folder where embeddings will be saved.
        adj_dict (dict): Adjacency dictionary where keys are station names and values are sets of neighboring stations.
        embeddings (numpy.ndarray): Node embeddings of shape (number of nodes, embedding dimension).

    Returns:
        None
    """
    embeddings_dict = {}

    nodes = list(adj_dict.keys())
    for idx, station in enumerate(nodes):
        embeddings_dict[station] = embeddings[idx]

    embeddings_dict['placeholder_begin_station'] = np.zeros(embeddings.shape[1])
    embeddings_dict['placeholder_end_station'] = np.zeros(embeddings.shape[1])
    
    file_path = os.path.join(save_folder_path, f'stations_emb_{embeddings.shape[1]}.pkl')

    print(f'Saving embeddings at {file_path}')
    save_pickle(embeddings_dict, file_path)

def save_graph(save_folder_path: str, G: nx.Graph) -> None:
    """
    Save a graph object to a pickle file.

    Args:
        save_folder_path (str): Path to the folder where the graph will be saved.
        G (networkx.Graph): Graph constructed from the adjacency dictionary.

    Returns:
        None
    """
    file_path = os.path.join(save_folder_path, f'station_graph.pkl')

    print(f'Saving graph at {file_path}')
    save_pickle(G, file_path)

def save_adjacency_dict(save_folder_path: str, adj_dict: dict) -> None:
    """
    Save an adjacency dictionary to a pickle file.

    Args:
        save_folder_path (str): Path to the folder where the adjacency dictionary will be saved.
        adj_dict (dict): Adjacency dictionary where keys are station names and values are sets of neighboring stations.

    Returns:
        None
    """
    file_path = os.path.join(save_folder_path, f'station_adjacency_dict.pkl')

    print(f'Saving adjacency dict at {file_path}')
    save_pickle(adj_dict, file_path)

def create_station_embeddings(data_folder_path: str, save_folder_path: str, first_year: int, first_month: int, last_year: int, last_month: int, embedding_size: int) -> None:
    """
    Create and save station embeddings, adjacency dictionary, and graph.

    Args:
        data_folder_path (str): Path to the folder containing punctuality CSV files.
        save_folder_path (str): Path to the folder where outputs will be saved.
        first_year (int): Starting year of the data to process, inclusive.
        first_month (int): Starting month of the data to process, inclusive.
        last_year (int): Ending year of the data to process, inclusive.
        last_month (int): Ending month of the data to process, inclusive.
        embedding_size (int): Dimension of the embedding space.

    Returns:
        None
    """
    os.makedirs(save_folder_path, exist_ok=True)
    adj_dict = create_adj_dictionnary(data_folder_path, first_year, first_month, last_year, last_month)
    embeddings, G = create_embeddings(adj_dict, embedding_size)
    save_embeddings(save_folder_path, adj_dict, embeddings)
    save_graph(save_folder_path, G)
    save_adjacency_dict(save_folder_path, adj_dict)

def main() -> None:
    """
    Entry point for creating and saving station embeddings.

    Parses CLI arguments, builds adjacency dictionary and graph,
    generates embeddings, and saves all outputs to the specified folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder_path", type=str, help="Raw data folder path.")
    parser.add_argument("save_folder_path", type=str, help="Path of the folder to save the data.")
    parser.add_argument("first_year", type=int, help="Year of the first file.")
    parser.add_argument("first_month", type=int, help="Month of the first file.")
    parser.add_argument("last_year", type=int, help="Year of the last file.")
    parser.add_argument("last_month", type=int, help="Month of the last file.")
    parser.add_argument("embedding_size", type=int, help="Number of dimensions of the embeddings.")

    args = parser.parse_args()
    
    create_station_embeddings(args.data_folder_path, args.save_folder_path, args.first_year, args.first_month, args.last_year, args.last_month, args.embedding_size)

if __name__ == "__main__":
    main()