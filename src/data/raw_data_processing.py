import argparse
import os
import pickle
import calendar
import gc
import torch
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

from src.utils.utils import save_pickle

def load_and_filter_col(path: str) -> pd.DataFrame:
    """
    Load a raw punctuality CSV file and keep only relevant columns.

    Args:
        path (str): Path to the raw CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing only the selected relevant columns.
    """
    df = pd.read_csv(path)
    relevant_columns = ['RELATION','DATDEP','PLANNED_DATE_ARR','PLANNED_DATE_DEP','REAL_DATE_ARR','REAL_DATE_DEP','TRAIN_NO','PLANNED_TIME_ARR','REAL_TIME_ARR','PLANNED_TIME_DEP','REAL_TIME_DEP','PTCAR_LG_NM_NL', 'PTCAR_NO', 'LINE_NO_DEP', 'LINE_NO_ARR']
    df = df[relevant_columns]
    return df

def concat_last_day_of_previous_month(df: pd.DataFrame, folder_path: str, year: int, month: int) -> pd.DataFrame:
    """
    Concatenate the last day of the previous month's data to the current month's DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame of the current month's punctuality data.
        folder_path (str): Path to the folder containing raw monthly CSV files.
        year (int): Year of the current month.
        month (int): Month of the current DataFrame.

    Returns:
        pandas.DataFrame: Concatenated DataFrame including the last day of the previous month.
    """
    prev_month = 12 if month == 1 else month - 1
    prev_year  = year - 1 if month == 1 else year
    csv_name = f"Data_raw_punctuality_{prev_year}{prev_month:02d}.csv"
    prev_path = os.path.join(folder_path, csv_name)
    df_prev = load_and_filter_col(prev_path)
    last_day_num = calendar.monthrange(prev_year, prev_month)[1]
    MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    month_abbr   = MONTH_ABBR[prev_month - 1]
    last_day_str = f"{last_day_num:02d}{month_abbr}{prev_year}"
    df_prev_last_day = df_prev.loc[df_prev["DATDEP"] == last_day_str]

    return pd.concat([df_prev_last_day, df], ignore_index=True)

def set_stations_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split raw punctuality data into departure, arrival, and passing events with unified columns.

    Args:
        df (pandas.DataFrame): Raw punctuality DataFrame containing planned/real times and line numbers
            for arrivals and departures.

    Returns:
        pandas.DataFrame: Processed DataFrame where each row represents either a departure ('D'),
        arrival ('A'), or passing ('P') event, with standardized columns:
        ['type', 'PLANNED_TIME', 'REAL_TIME', 'PLANNED_DATE', 'REAL_DATE', 'LINE_NO', ...].
    """
    df['type'] = np.where(pd.isna(df['PLANNED_TIME_DEP']), 'A',
                np.where(pd.isna(df['PLANNED_TIME_ARR']), 'D',
                np.where(df['PLANNED_TIME_DEP'] == df['PLANNED_TIME_ARR'], 'P', 'M')))  # 'M' for mixed (both A and D)

    df_departure = df[df['type'].isin(['D', 'M'])].copy()
    df_departure['type'] = 'D'
    df_departure['PLANNED_TIME'] = df_departure['PLANNED_TIME_DEP']
    df_departure['REAL_TIME'] = df_departure['REAL_TIME_DEP']
    df_departure['PLANNED_DATE'] = df_departure['PLANNED_DATE_DEP']
    df_departure['REAL_DATE'] = df_departure['REAL_DATE_DEP']
    df_departure['LINE_NO'] = df_departure['LINE_NO_DEP']

    df_arrival = df[df['type'].isin(['A', 'M'])].copy()
    df_arrival['type'] = 'A'
    df_arrival['PLANNED_TIME'] = df_arrival['PLANNED_TIME_ARR']
    df_arrival['REAL_TIME'] = df_arrival['REAL_TIME_ARR']
    df_arrival['PLANNED_DATE'] = df_arrival['PLANNED_DATE_ARR']
    df_arrival['REAL_DATE'] = df_arrival['REAL_DATE_ARR']
    df_arrival['LINE_NO'] = df_arrival['LINE_NO_ARR']

    df_passing = df[df['type'].isin(['P'])].copy()
    df_passing['PLANNED_TIME'] = df_passing['PLANNED_TIME_ARR']
    df_passing['REAL_TIME'] = df_passing['REAL_TIME_ARR']
    df_passing['PLANNED_DATE'] = df_passing['PLANNED_DATE_ARR']
    df_passing['REAL_DATE'] = df_passing['REAL_DATE_ARR']
    df_passing['LINE_NO'] = df_passing['LINE_NO_ARR']

    new_df = pd.concat([df_departure, df_arrival, df_passing], ignore_index=True)

    new_df['LINE_NO'] = new_df['LINE_NO'].fillna('NaN')

    new_df.drop(columns=['PLANNED_TIME_DEP', 'PLANNED_TIME_ARR','REAL_TIME_ARR','REAL_TIME_DEP','PLANNED_DATE_ARR','PLANNED_DATE_DEP','REAL_DATE_ARR','REAL_DATE_DEP', 'LINE_NO_DEP','LINE_NO_ARR'], inplace=True)

    return new_df

def encode_itinerary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract relation type from the relation string and encode it as a new column.

    Args:
        df (pandas.DataFrame): DataFrame containing a 'RELATION' column with relation descriptions.

    Returns:
        pandas.DataFrame: DataFrame with a new 'RELATION_TYPE' column containing the first token
        of the relation string, and without the original 'RELATION' column.
    """

    df['RELATION_TYPE'] = df['RELATION'].astype(str).apply(lambda x: x.split()[0])
    df.drop(columns = ['RELATION'], inplace = True)
    return df

def delete_train_with_missing_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove train itineraries containing missing times or invalid relation types.

    Args:
        df (pandas.DataFrame): DataFrame of train events with columns including
            'TRAIN_NO', 'PLANNED_TIME', 'REAL_TIME', and 'RELATION_TYPE'.

    Returns:
        pandas.DataFrame: Filtered DataFrame containing only trains where all events
        have non-missing planned and real times, and relation types do not contain 'nan'.
    """
    grouped = df.groupby('TRAIN_NO')

    filtered_groups = []

    for _, group in tqdm(grouped, desc='Filtering groups'):
        if not group[['PLANNED_TIME', 'REAL_TIME']].isnull().any().any() and 'nan' not in group['RELATION_TYPE'].values:
            filtered_groups.append(group)

    filtered_df = pd.concat(filtered_groups)

    return filtered_df        

def get_numerical_times(df: pd.DataFrame, deltat: int) -> pd.DataFrame:
    """
    Convert planned and real datetimes into numerical timestamps and compute delays.

    Args:
        df (pandas.DataFrame): DataFrame containing planned and real dates/times for train events.
        deltat (int): Time step in seconds used to discretize timestamps.

    Returns:
        pandas.DataFrame: DataFrame with numerical time columns:
            - 'PLANNED_TIME_NUM': Discretized planned time in seconds since 2012-01-01.
            - 'REAL_TIME_NUM': Discretized real time in seconds since 2012-01-01.
            - 'DELAY': Difference between real and planned times in seconds.
        Original time/date string columns are dropped, and rows are sorted by
        'TRAIN_NO' and 'PLANNED_TIME_NUM'.
    """
    df['PLANNED_DATETIME'] = pd.to_datetime(df['PLANNED_DATE'] + ' ' + df['PLANNED_TIME'], format='%d%b%Y %H:%M:%S')
    df['REAL_DATETIME'] = pd.to_datetime(df['REAL_DATE'] + ' ' + df['REAL_TIME'], format='%d%b%Y %H:%M:%S')

    df['PLANNED_TIME_NUM'] = (df['PLANNED_DATETIME'] - pd.Timestamp("2012-01-01")) // pd.Timedelta('1s')
    df['REAL_TIME_NUM'] = (df['REAL_DATETIME'] - pd.Timestamp("2012-01-01")) // pd.Timedelta('1s')

    df.drop(columns = ['PLANNED_TIME','REAL_TIME','PLANNED_DATE','REAL_DATE'] ,inplace=True)
    df['REAL_TIME_NUM'] = df['REAL_TIME_NUM'].apply(lambda x: round(x / deltat) * deltat)
    df['PLANNED_TIME_NUM'] = df['PLANNED_TIME_NUM'].apply(lambda x: round(x / deltat) * deltat)
    df['DELAY'] = df['REAL_TIME_NUM'] - df['PLANNED_TIME_NUM']

    df.sort_values(['TRAIN_NO','PLANNED_TIME_NUM'], inplace=True)
    
    return df

def set_past_and_future_stations_group(group: pd.DataFrame, nb_past_stations: int, nb_future_stations: int, idle_time_beggining: int, idle_time_end: int) -> pd.DataFrame:
    """
    Augment a train's ordered event group with windowed past/future context and a leading placeholder row.

    The function:
      - Inserts a first placeholder row ('placeholder_begin_station', 'placeholder_line') shifted
        by `idle_time_beggining` minutes before the group's first planned time (with safeguards for
        non-monotonic REAL_TIME_NUM).
      - Pads the itinerary with (nb_past_stations - 1) past placeholders and nb_future_stations future
        placeholders to build fixed-size sliding windows.
      - Adds context columns per row:
        PAST_STATIONS, FUTURE_STATIONS, PAST_TYPES, FUTURE_TYPES,
        PAST_LINES, FUTURE_LINES, PAST_PLANNED_TIME_NUM, FUTURE_PLANNED_TIME_NUM,
        PAST_DELAYS, FUTURE_DELAYS.

    Args:
        group (pandas.DataFrame): Single-train, time-ordered rows with at least
            'PTCAR_LG_NM_NL', 'type', 'LINE_NO', 'PLANNED_TIME_NUM',
            'REAL_TIME_NUM', 'PLANNED_DATETIME', 'REAL_DATETIME', 'DELAY'.
        nb_past_stations (int): Number of past stations to include in the window (including current row).
        nb_future_stations (int): Number of future stations to include in the window.
        idle_time_beggining (int): Minutes to shift the inserted leading placeholder before the first stop.
        idle_time_end (int): Minutes used to extend the last planned time when padding futures.

    Returns:
        pandas.DataFrame: The input group augmented with the leading placeholder and context columns,
        one row per stop after window construction.
    """

    first_row = group.iloc[0].to_dict()

    first_row['PLANNED_DATETIME'] = first_row['PLANNED_DATETIME'] - pd.Timedelta(minutes=idle_time_beggining)
    first_row['REAL_DATETIME'] = first_row['REAL_DATETIME']
    first_row['DELAY'] = 0
    first_row['PTCAR_LG_NM_NL'] = 'placeholder_begin_station'
    first_row['type'] = 'D'
    first_row['PLANNED_TIME_NUM'] = first_row['PLANNED_TIME_NUM'] - (60 * idle_time_beggining)
    first_row['REAL_TIME_NUM'] = min(first_row['REAL_TIME_NUM'], first_row['PLANNED_TIME_NUM']) # if the train started before the idle time beginning, we start the snapshot here (mimics real conditions)
    if first_row['REAL_TIME_NUM'] < first_row['PLANNED_TIME_NUM'] - 10*(60 * idle_time_beggining):
        first_row['REAL_TIME_NUM'] = first_row['PLANNED_TIME_NUM'] + 5555 # if the train started more than 10* before the idle time, we discard it (will be deleted because non monotonic increasing)

        

    first_row['LINE_NO'] = 'placeholder_line'
    
    first_row_df = pd.DataFrame([first_row])
    group = pd.concat([first_row_df, group], ignore_index=True)

    stations = group['PTCAR_LG_NM_NL'].tolist()
    types = group['type'].tolist()
    lines = group['LINE_NO'].tolist()
    
    padded_stations = np.array((['placeholder_begin_station'] * (nb_past_stations - 1) + stations + ['placeholder_end_station'] * nb_future_stations))
    padded_types = (['D'] * (nb_past_stations - 1) + types + ['A'] * nb_future_stations)
    padded_lines = (['placeholder_line'] * (nb_past_stations - 1) + lines + ['placeholder_line'] * nb_future_stations)
    group['PAST_STATIONS'] = [padded_stations[i-nb_past_stations+1:i+1] for i in range(nb_past_stations-1, len(padded_stations) - nb_future_stations)]
    group['FUTURE_STATIONS'] = [padded_stations[i+1:i+1+nb_future_stations] for i in range(nb_past_stations-1, len(padded_stations) - nb_future_stations)]
    group['PAST_TYPES'] = [padded_types[i-nb_past_stations+1:i+1] for i in range(nb_past_stations-1, len(padded_types) - nb_future_stations)]
    group['FUTURE_TYPES'] = [padded_types[i+1:i+1+nb_future_stations] for i in range(nb_past_stations-1, len(padded_types) - nb_future_stations)]
    group['PAST_LINES'] = [padded_lines[i-nb_past_stations+1:i+1] for i in range(nb_past_stations-1, len(padded_lines) - nb_future_stations)]
    group['FUTURE_LINES'] = [padded_lines[i+1:i+1+nb_future_stations] for i in range(nb_past_stations-1, len(padded_lines) - nb_future_stations)]

    planned_time_num = group['PLANNED_TIME_NUM'].tolist()
    padded_planned_time_num = ([planned_time_num[0]] * (nb_past_stations - 1) + planned_time_num + [planned_time_num[-1] + (60*idle_time_end)] * nb_future_stations)
    group['PAST_PLANNED_TIME_NUM'] = [padded_planned_time_num[i-nb_past_stations+1:i+1] for i in range(nb_past_stations-1, len(padded_stations) - nb_future_stations)]
    group['FUTURE_PLANNED_TIME_NUM'] = [padded_planned_time_num[i+1:i+1+nb_future_stations] for i in range(nb_past_stations-1, len(padded_stations) - nb_future_stations)]

    delays = group['DELAY'].tolist()
    padded_delays = ([delays[0]] * (nb_past_stations - 1) + delays + [delays[-1]] * nb_future_stations)
    group['PAST_DELAYS'] = [padded_delays[i-nb_past_stations+1:i+1] for i in range(nb_past_stations-1, len(padded_stations) - nb_future_stations)]
    group['FUTURE_DELAYS'] = [padded_delays[i+1:i+1+nb_future_stations] for i in range(nb_past_stations-1, len(padded_stations) - nb_future_stations)]
    
    return group 

def set_past_and_future_stations(df: pd.DataFrame, nb_past_stations: int, nb_future_stations: int, idle_time_beggining: int, idle_time_end: int) -> pd.DataFrame:
    """
    Apply past/future station windowing to each train group.

    Args:
        df (pandas.DataFrame): Input DataFrame of train events.
        nb_past_stations (int): Number of past stations to include.
        nb_future_stations (int): Number of future stations to include.
        idle_time_beggining (int): Minutes to prepend before the first stop.
        idle_time_end (int): Minutes to append after the last stop.

    Returns:
        pandas.DataFrame: DataFrame with added past/future context columns.
    """
    grouped = df.groupby(['TRAIN_NO', 'DATDEP'])

    new_rows = []
    for _, group in tqdm(grouped, desc="Processing Groups"):
        new_rows.append(set_past_and_future_stations_group(group, nb_past_stations, nb_future_stations, idle_time_beggining, idle_time_end))
    
    return pd.concat(new_rows, ignore_index=True)

def convert_list_to_columns(df: pd.DataFrame, nb_past_stations: int, nb_future_stations: int) -> pd.DataFrame:
    """
    Expand past and future context lists into separate DataFrame columns.

    Args:
        df (pandas.DataFrame): Input DataFrame with list-based context columns.
        nb_past_stations (int): Number of past stations included in each row.
        nb_future_stations (int): Number of future stations included in each row.

    Returns:
        pandas.DataFrame: DataFrame with context lists expanded into individual columns.
    """

    for col in ['PAST_STATIONS','PAST_PLANNED_TIME_NUM','PAST_DELAYS','PAST_TYPES', 'PAST_LINES']:
        col_names = [f"{col}_{nb_past_stations - i}" for i in range(nb_past_stations)]
        col_df = pd.DataFrame(df[col].to_list(), columns=col_names)  # Convert list to DataFrame
        df = pd.concat([df.drop(columns=[col]), col_df], axis=1)  # Drop the original list column and join the new columns

    for col in ['FUTURE_STATIONS','FUTURE_TYPES','FUTURE_DELAYS', 'FUTURE_PLANNED_TIME_NUM', 'FUTURE_LINES']:
        col_names = [f"{col}_{i+1}" for i in range(nb_future_stations)]
        col_df = pd.DataFrame(df[col].to_list(), columns=col_names)
        df = pd.concat([df.drop(columns=[col]), col_df], axis=1)

    df.drop(columns=['PLANNED_DATETIME','REAL_DATETIME','PLANNED_TIME_NUM','type','DELAY','PTCAR_LG_NM_NL','PTCAR_NO', 'LINE_NO'], inplace=True)	
    
    return df

def expand_and_add_actions_group(group: pd.DataFrame, deltat: int, idle_time_end: int) -> tuple:
    """
    Expand a single train group to a fixed time grid and derive step actions.

    The group is checked for monotonic REAL_TIME_NUM; minor glitches (≤180s) are corrected by
    forward-filling the previous value. If still non-monotonic, the group is discarded.
    The time axis spans from the group's min REAL_TIME_NUM to max REAL_TIME_NUM + idle_time_end*60,
    sampled every `deltat` seconds. Rows are backfilled via merge-asof. An 'action' column is
    added indicating whether the state stays at the same station ('same') or moves to the next
    station ('next1', 'next2', ...). Only 'same', 'next1', and 'next2' are accepted.

    Args:
        group (pandas.DataFrame): One (TRAIN_NO, DATDEP) group's rows, already augmented with past/future
            context columns (e.g., 'PAST_STATIONS_1', 'PAST_TYPES_1') and REAL_TIME_NUM.
        deltat (int): Time-step size in seconds for the expansion grid.
        idle_time_end (int): Minutes to extend beyond the last REAL_TIME_NUM when building the grid.

    Returns:
        tuple: (expanded_df, status) where
            expanded_df (pandas.DataFrame): Expanded time-indexed DataFrame including 'STATE_TIME' and 'action'.
            status (int): 0 if OK; 1 if discarded due to non-monotonic times after correction;
                          2 if invalid actions detected (beyond 'same', 'next1', 'next2').
    """

    if not group['REAL_TIME_NUM'].is_monotonic_increasing:
        # if data is incorrect up to 3 minutes, we try to correct it
        prev = group['REAL_TIME_NUM'].shift(1)
        diff = prev - group['REAL_TIME_NUM']
        mask = (diff > 0) & (diff <= 180)
        group.loc[mask, 'REAL_TIME_NUM'] = prev[mask] # put real time from previous station
        if not group['REAL_TIME_NUM'].is_monotonic_increasing: # if it is still incorrect, we discard it
            return pd.DataFrame(columns=[*group.columns.drop('REAL_TIME_NUM'), 'STATE_TIME', 'action']), 1

    min_time = group['REAL_TIME_NUM'].min()
    max_time = group['REAL_TIME_NUM'].max() + (60 * idle_time_end)
    
    time_points = np.arange(min_time, max_time, deltat)
    expanded_group = pd.DataFrame({'STATE_TIME': time_points})
    
    expanded_group['STATE_TIME'] = expanded_group['STATE_TIME'].astype(group['REAL_TIME_NUM'].dtype)
    
    expanded_group = pd.merge_asof(expanded_group, group, left_on='STATE_TIME', right_on='REAL_TIME_NUM', 
                                   direction='backward').drop(columns=['REAL_TIME_NUM'])
    
    stations = (group['PAST_STATIONS_1'].astype(str) + ' - ' + group['PAST_TYPES_1']).tolist()
    index_map = {value: idx for idx, value in enumerate(stations)}
    
    current_stations = expanded_group['PAST_STATIONS_1'].astype(str) + ' - ' + expanded_group['PAST_TYPES_1']
    next_stations = current_stations.shift(-1)
    
    current_indices = current_stations.map(index_map)
    next_indices = next_stations.map(index_map)
    
    station_diff = next_indices.fillna(current_indices) - current_indices
    
    expanded_group['action'] = np.where(station_diff != 0, 'next' + station_diff.astype(int).astype(str), 'same')
    
    if expanded_group['action'].isin(['same', 'next1','next2']).all():
        return expanded_group, 0
    else:
        return pd.DataFrame(columns=expanded_group.columns), 2
    

def expand_and_add_actions(df: pd.DataFrame, deltat: int, idle_time_end: int) -> pd.DataFrame:
    """
    Expand all train groups to fixed time grids (1 row every deltat seconds) and add action labels.

    Args:
        df (pandas.DataFrame): Input DataFrame containing train events grouped by TRAIN_NO and DATDEP.
        deltat (int): Time-step size in seconds for the expansion grid.
        idle_time_end (int): Minutes to extend beyond the last REAL_TIME_NUM when building the grid.

    Returns:
        pandas.DataFrame: Concatenated expanded groups with 'STATE_TIME' and 'action' columns.
    """
    grouped = df.groupby(['TRAIN_NO', 'DATDEP'])

    codes = {'0': 0, '1': 0, '2': 0}
    new_rows = []

    pbar = tqdm(grouped, desc='Processing Groups')
    for _, group in pbar:
        result, code = expand_and_add_actions_group(group, deltat, idle_time_end)
        codes[str(code)] += 1
        pbar.set_postfix(codes)
        if not result.empty:
            new_rows.append(result)
    pbar.close()

    return pd.concat(new_rows, ignore_index=True)

def filter_and_sample(df: pd.DataFrame, current_year: int, current_month: int, sample_ratio: float) -> pd.DataFrame:
    """
    Filter states to the given month and randomly subsample them.

    Args:
        df (pandas.DataFrame): DataFrame with a 'STATE_TIME' column of integer timestamps.
        current_year (int): Year to filter.
        current_month (int): Month to filter.
        sample_ratio (float): Fraction of rows to sample from the filtered set.

    Returns:
        pandas.DataFrame: Subsampled DataFrame restricted to the given month.
    """
    possible_values = df.STATE_TIME.unique()
    reference_date = pd.Timestamp("2012-01-01")
    start_date = pd.Timestamp(f"{current_year}-{current_month}-01 00:00:00")
    last_day = calendar.monthrange(int(current_year), int(current_month))[1]
    end_date = pd.Timestamp(f"{current_year}-{current_month}-{last_day} 23:59:59")
    
    start_state_time = int((start_date - reference_date) / pd.Timedelta('1s'))
    end_state_time = int((end_date - reference_date) / pd.Timedelta('1s'))

    # filter STATE_TIME rows not in the current month
    filtered_values = [v for v in possible_values if start_state_time <= v <= end_state_time]
    num_samples = int(len(filtered_values) * sample_ratio)
    sampled_values = random.sample(filtered_values, num_samples)
    return df[df['STATE_TIME'].isin(sampled_values)]

def convert_num_to_delta(df: pd.DataFrame, nb_past_stations: int, nb_future_stations: int) -> pd.DataFrame:
    """
    Convert absolute planned times into deltas relative to STATE_TIME.

    Args:
        df (pandas.DataFrame): DataFrame with STATE_TIME and planned time columns.
        nb_past_stations (int): Number of past stations included.
        nb_future_stations (int): Number of future stations included.

    Returns:
        pandas.DataFrame: DataFrame with planned times replaced by deltas to STATE_TIME.
    """
    for i in range(nb_past_stations):
        df[f'PAST_PLANNED_TIME_NUM_{i+1}'] = df[f'PAST_PLANNED_TIME_NUM_{i+1}'] - df['STATE_TIME']
    for i in range(nb_future_stations):
        df[f'FUTURE_PLANNED_TIME_NUM_{i+1}'] = df[f'FUTURE_PLANNED_TIME_NUM_{i+1}'] - df['STATE_TIME']

    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add datetime-based features derived from STATE_TIME.

    Args:
        df (pandas.DataFrame): DataFrame with a 'STATE_TIME' column of integer seconds since 2012-01-01.

    Returns:
        pandas.DataFrame: DataFrame with added 'DATETIME', 'day_of_week', 'hour', and 'day_of_year' columns.
    """
    df['DATETIME'] = pd.Timestamp("2012-01-01") + pd.to_timedelta(df['STATE_TIME'], unit='s')

    df['day_of_week'] = df['DATETIME'].dt.day_name()
 
    # Cyclical encoding of the time of day (24-hour cycle)
    df['hour'] = df['DATETIME'].dt.hour + df['DATETIME'].dt.minute / 60.0 + df['DATETIME'].dt.second / 3600.0
    df['day_of_year'] = df['DATETIME'].dt.dayofyear

    return df

def cast_categorical_data(df: pd.DataFrame, category_dict: dict) -> pd.DataFrame:
    """
    Cast specified columns to categorical types and one-hot encode them.

    Args:
        df (pandas.DataFrame): Input DataFrame with categorical columns to encode.
        category_dict (dict): Mapping of column names (or substrings to match) to
            lists of allowed categories.

    Returns:
        pandas.DataFrame: DataFrame with specified columns cast to categorical and
        replaced by one-hot encoded dummy variables.
    """

    cols_to_encode = []

    for key, categories in tqdm(category_dict.items(), desc='Casting categorical data'):
        if key in df.columns:
            df[key] = pd.Categorical(df[key], categories=categories)
            cols_to_encode.append(key)
        elif any(key in col for col in df.columns):
            matching_columns = [col for col in df.columns if key in col]
            for col in matching_columns:
                df[col] = pd.Categorical(df[col], categories=categories)
                cols_to_encode.append(col)
        else:
            print(f'Column or pattern {key} not found in the DataFrame.')

    df_encoded = pd.get_dummies(df, columns=cols_to_encode, dummy_na=False, dtype=int)

    return df_encoded

def sign_invariant_sqrt_std_normalization(df: pd.DataFrame, num_columns_prefix: list, num_std_value: float) -> pd.DataFrame:
    """
    Apply sign-invariant square-root and standard deviation normalization.

    Args:
        df (pandas.DataFrame): Input DataFrame with numerical columns.
        num_columns_prefix (list): List of column name prefixes to select numerical columns.
        num_std_value (float): Normalization constant (e.g., global standard deviation).

    Returns:
        pandas.DataFrame: DataFrame with transformed numerical columns.
    """
    for prefix in num_columns_prefix:
        matching_columns = [col for col in df.columns if col.startswith(prefix)]
        for col in matching_columns:
            data = df[col].values
            sign_sqrt = np.sign(data) * np.sqrt(np.abs(data))
            df[col] = sign_sqrt / num_std_value

    return df

def process_month(folder_path: str, year: int, month: int, deltat: int, nb_past_stations: int, nb_future_stations: int, 
            idle_time_beggining: int, idle_time_end: int, sample_ratio: float) -> pd.DataFrame:
    """
    Process one month of punctuality data into a feature-rich DataFrame.

    Applies the full preprocessing pipeline: load and filter columns, merge last
    day of previous month, encode station events, clean missing values, compute
    numerical times and delays, add past/future context, expand to fixed time
    steps with actions, subsample, convert times to deltas, and add temporal
    features.

    Args:
        folder_path (str): Path to the folder containing monthly CSV files.
        year (int): Year of the month to process.
        month (int): Month to process.
        deltat (int): Time-step size in seconds.
        nb_past_stations (int): Number of past stations to include as features.
        nb_future_stations (int): Number of future stations to include as features.
        idle_time_beggining (int): Minutes to prepend before the first stop.
        idle_time_end (int): Minutes to append after the last stop.
        sample_ratio (float): Fraction of rows to sample.

    Returns:
        pandas.DataFrame: Fully processed DataFrame for the given month.
    """

    path = os.path.join(folder_path, 'Data_raw_punctuality_' + str(year) + str(month).zfill(2) + '.csv')
    
    df = load_and_filter_col(path)
    df = concat_last_day_of_previous_month(df, folder_path, year, month)
    df = set_stations_types(df)
    df = encode_itinerary(df)
    df = delete_train_with_missing_value(df)
    df = get_numerical_times(df, deltat)
    df = set_past_and_future_stations(df, nb_past_stations, nb_future_stations, idle_time_beggining, idle_time_end)
    df = convert_list_to_columns(df, nb_past_stations, nb_future_stations)
    df = expand_and_add_actions(df, deltat, idle_time_end)
    df = filter_and_sample(df, year, month, sample_ratio)
    df = convert_num_to_delta(df, nb_past_stations, nb_future_stations)
    df = create_time_features(df)

    return df

def get_optimal_column_order(original_cols: list, nb_past_stations: int, nb_future_stations: int, embedding_size: int) -> list:
    """
    Build the optimal feature column order for downstream simulation.

    The order groups delay features, one-hot type features (D/A/P), and station/line
    embedding features (past then future, by embedding index), then appends any
    remaining original columns to preserve all inputs. This layout matches the
    expected input structure of the simulation pipeline.

    Args:
        original_cols (list): Existing column names to include after the core feature blocks.
        nb_past_stations (int): Number of past stations represented in features.
        nb_future_stations (int): Number of future stations represented in features.
        embedding_size (int): Dimension of station/line embeddings.

    Returns:
        list: Ordered list of column names optimized for the simulation step.
    """
    final_cols = []
    for col in [f"PAST_DELAYS_{nb_past_stations - i}" for i in range(nb_past_stations)]:
        final_cols.append(col)

    for typ in ['D','A','P']:
        for col in [f"PAST_TYPES_{nb_past_stations - i}_{typ}" for i in range(nb_past_stations)]:
            final_cols.append(col)
        
        for col in  [f"FUTURE_TYPES_{i+1}_{typ}" for i in range(nb_future_stations)]:
            final_cols.append(col)
    
    for j in range(embedding_size):
        for col in [f"PAST_STATIONS_{nb_past_stations - i}_embedding_{j}" for i in range(nb_past_stations)]:
            final_cols.append(col)
        for col in [f"FUTURE_STATIONS_{i+1}_embedding_{j}" for i in range(nb_future_stations)]:
            final_cols.append(col)
        
    
    for j in range(embedding_size):
        for col in [f"PAST_LINES_{nb_past_stations - i}_embedding_{j}" for i in range(nb_past_stations)]:
            final_cols.append(col)
        for col in [f"FUTURE_LINES_{i+1}_embedding_{j}" for i in range(nb_future_stations)]:
            final_cols.append(col)

    for col in original_cols:
        if col not in final_cols:
            final_cols.append(col)
            
    return final_cols

def main() -> None:
    """
    Entry point to preprocess one month's punctuality data.

    Parses CLI arguments, applies the preprocessing pipeline for the given month,
    saves the processed data as a parquet file using brotli compression, and stores 
    the preprocessing configuration for later use.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_in_path", type=str, help="Path of the folder of the raw data.")
    parser.add_argument("folder_out_path", type=str, help="Path of the folder the processed data will be stored in.")
    parser.add_argument("year", type=int, help="Year of the processed file.")
    parser.add_argument("month", type=int, help="Month of the processed file.")
    parser.add_argument("deltat", type=int, help="Number of seconds of each time step.")
    parser.add_argument("nb_past_stations", type=int, help="Number of past stations kept as features.")
    parser.add_argument("nb_future_stations", type=int, help="Number of future stations kept as features.")
    parser.add_argument("idle_time_beggining", type=int, help="Number of minutes the train will be on the network before it starts.")
    parser.add_argument("idle_time_end", type=int, help="Number of minutes the train will be on the network before it disapears after reaching the last station.")
    parser.add_argument("sample_ratio", type=float, help="Ratio of kept data points.")
    
    args = parser.parse_args()
    print(args)

    os.makedirs(args.folder_out_path, exist_ok=True)

    df = process_month(args.folder_in_path, args.year, args.month, args.deltat, args.nb_past_stations, args.nb_future_stations, args.idle_time_beggining, args.idle_time_end, args.sample_ratio)
    
    out_path = os.path.join(args.folder_out_path, 'processed_data_' + str(args.year) + str(args.month).zfill(2) + '.brotli.parquet')
    df.to_parquet(out_path,compression='brotli')

    config = {
        'deltat':args.deltat,
        'nb_past_stations':args.nb_past_stations,
        'nb_future_stations':args.nb_future_stations,
        'idle_beg':args.idle_time_beggining,
        'idle_end':args.idle_time_end,
    }

    save_pickle(config, os.path.join(args.folder_out_path, 'config.pkl'))

if __name__ == "__main__":   
    main()