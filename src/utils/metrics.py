import torch
import re

import numpy as np

from src.environment.simulation import Simulator

# These functions need so work on naming, especially the MAE/MSE part, also the typing stuff between numpy/torch is horrible, should be fixed too


def inverse_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Invert the transform by undoing normalization and square-root scaling.

    Args:
        x (torch.Tensor): Transformed tensor.

    Returns:
        torch.Tensor: Original-scale tensor.
    """
    return torch.sign(x) * (x * 6) ** 2


def _apply_inverse_round(tensor: torch.Tensor) -> torch.Tensor:
    """
    Inverse-transform then round to fix numerical drift (e.g., 29.999 → 30).

    Args:
        tensor (torch.Tensor): Input tensor in transformed space.

    Returns:
        torch.Tensor: Rounded tensor in original scale.
    """

    return torch.round(inverse_transform(tensor))


def _update_mae_bins(
    mae_arr: np.ndarray | torch.Tensor,
    counter_arr: np.ndarray | torch.Tensor,
    abs_errors: torch.Tensor,
    values: torch.Tensor,
    bins: np.ndarray,
    extra_mask: torch.Tensor | None = None
) -> None:
    """
    Update MAE accumulators and counts for the given bins.

    Args:
        mae_arr (np.ndarray | torch.Tensor): Sum of absolute errors per bin (modified in-place).
        counter_arr (np.ndarray | torch.Tensor): Count per bin (modified in-place).
        abs_errors (torch.Tensor): Absolute errors per element.
        values (torch.Tensor): Values to bin (e.g., delay deltas or horizons).
        bins (np.ndarray): 1D bin edges (len = #bins + 1).
        extra_mask (torch.Tensor | None): Optional boolean mask to AND with bin mask.
    """
    if extra_mask is None:
        extra_mask = torch.ones_like(values, dtype=torch.bool)
    for i in range(len(bins) - 1):
        bin_mask = (bins[i] <= values) & (values < bins[i + 1]) & extra_mask
        mae_arr[i] += torch.sum(abs_errors[bin_mask]).item()
        counter_arr[i] += torch.sum(bin_mask).item()


def _finalize_mae(
    mae_arr: np.ndarray | torch.Tensor,
    counter_arr: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Compute per-bin MAE by dividing accumulated errors by counts,
    safely handling zero counts.

    Args:
        mae_arr (np.ndarray | torch.Tensor): Sum of absolute errors per bin.
        counter_arr (np.ndarray | torch.Tensor): Count of elements per bin.

    Returns:
        np.ndarray | torch.Tensor: Mean absolute error per bin.
    """

    return mae_arr / np.where(counter_arr == 0, 1, counter_arr)

def compute_mae_regression_tr(predictions: list, targets: list, x: list, masks: list, sc: dict, delay_delta_bins: np.ndarray, 
            horizon_obs_bins: np.ndarray) -> tuple:
    """
    Compute MAE (by delay-delta and horizon bins) and MSE for Transformer-shaped outputs.

    Args:
        predictions (list): Predicted tensors per sample, shape (B, T, F).
        targets (list): Target tensors per sample, shape (B, T, F).
        x (list): Input tensors per sample, shape (B, T, D) for context features.
        masks (list): Boolean masks per sample, shape (B, T, F); True for padding/ignore.
        sc (dict): Feature schema (e.g., indices in sc['x']).
        delay_delta_bins (np.ndarray): Bin edges for delay delta.
        horizon_obs_bins (np.ndarray): Bin edges for observed horizon.

    Returns:
        tuple: (mae_delay, mae_horizon, counter_delay, counter_horizon, mse_global)
            - mae_delay (np.ndarray): MAE per delay-delta bin.
            - mae_horizon (np.ndarray): MAE per horizon bin.
            - counter_delay (np.ndarray): Counts per delay-delta bin.
            - counter_horizon (np.ndarray): Counts per horizon bin.
            - mse_global (float): Global mean squared error over valid horizons.
    """
    nb_future_stations = predictions[0].shape[-1]
    mae_delay = np.zeros(len(delay_delta_bins) - 1)
    mae_horizon = np.zeros(len(horizon_obs_bins) - 1)
    counter_delay = np.zeros_like(mae_delay)
    counter_horizon = np.zeros_like(mae_horizon)
    sse = 0.0
    n_total = 0

    pattern = r"^FUTURE_PLANNED_TIME_NUM_*"

    for i in range(len(x)):
        mask = ~masks[i]
        past_delay_1 = _apply_inverse_round(x[i][:, :, sc['x']['PAST_DELAYS_1']])
        observed = _apply_inverse_round(targets[i])
        futur_planned_idx = [sc['x'][s] for s in sc['x'] if re.match(pattern, s)]
        futur_planned = _apply_inverse_round(x[i][:, :, futur_planned_idx])

        pred = inverse_transform(predictions[i]) + np.repeat(past_delay_1[:, :, None], nb_future_stations, axis=2)
        translation = past_delay_1 - ((futur_planned[:, :, 0] + past_delay_1) < 0) * (futur_planned[:, :, 0] + past_delay_1)
        translation = np.repeat(translation[:, :, None], nb_future_stations, axis=2)

        err = (pred - observed)[mask]
        mae_reg = np.abs(err)

        horizon_obs = futur_planned[mask] + observed[mask]
        valid_horizon_mask = (horizon_obs_bins[0] <= horizon_obs) & (horizon_obs < horizon_obs_bins[-1])
        delay_delta = (observed - translation)[mask]

        _update_mae_bins(mae_delay, counter_delay, mae_reg, delay_delta, delay_delta_bins, valid_horizon_mask)
        _update_mae_bins(mae_horizon, counter_horizon, mae_reg, horizon_obs, horizon_obs_bins)

        se = np.square(err, dtype=np.float64)[valid_horizon_mask]
        sse += se.sum()
        n_total += se.shape[0]

    mae_delay = _finalize_mae(mae_delay, counter_delay)
    mae_horizon = _finalize_mae(mae_horizon, counter_horizon)
    mse_global = sse / n_total if n_total else np.nan
    return mae_delay, mae_horizon, counter_delay, counter_horizon, mse_global

def compute_mae_regression_mlp(predictions: list, targets: list, x: list, masks: list, sc: dict, delay_delta_bins: np.ndarray, horizon_obs_bins: np.ndarray) -> tuple:
    """
    Compute MAE by delay-delta and horizon bins, plus global MSE, for MLP-shaped outputs.

    Args:
        predictions (list): Per-sample predicted tensors, shape (B, F).
        targets (list): Per-sample target tensors, shape (B, F).
        x (list): Per-sample input tensors for context, shape (B, D).
        masks (list): Per-sample boolean masks, shape (B, F); True = ignore.
        sc (dict): Feature schema (e.g., indices in sc['x']).
        delay_delta_bins (np.ndarray): Bin edges for delay delta.
        horizon_obs_bins (np.ndarray): Bin edges for observed horizon.

    Returns:
        tuple: (mae_delay, mae_horizon, counter_delay, counter_horizon, mse_global)
            - mae_delay (np.ndarray): MAE per delay-delta bin.
            - mae_horizon (np.ndarray): MAE per horizon bin.
            - counter_delay (np.ndarray): Counts per delay-delta bin.
            - counter_horizon (np.ndarray): Counts per horizon bin.
            - mse_global (float): Global mean squared error over valid horizons.
    """
    nb_future_stations = predictions[0].shape[-1]
    mae_delay = np.zeros(len(delay_delta_bins) - 1)
    mae_horizon = np.zeros(len(horizon_obs_bins) - 1)
    counter_delay = np.zeros_like(mae_delay)
    counter_horizon = np.zeros_like(mae_horizon)
    sse = 0.0
    n_total = 0
    
    pattern = r"^FUTURE_PLANNED_TIME_NUM_*"

    for i in range(len(x)):
        mask = ~masks[i]
        past_delay_1 = _apply_inverse_round(x[i][:, sc['x']['PAST_DELAYS_1']])
        observed = _apply_inverse_round(targets[i])
        futur_planned_idx = [sc['x'][s] for s in sc['x'] if re.match(pattern, s)]
        futur_planned = _apply_inverse_round(x[i][:, futur_planned_idx])

        pred = inverse_transform(predictions[i]) + np.repeat(past_delay_1[:, None], nb_future_stations, axis=1)
        translation = past_delay_1 - ((futur_planned[:, 0] + past_delay_1) < 0) * (futur_planned[:, 0] + past_delay_1)
        translation = np.repeat(translation[:, None], nb_future_stations, axis=1)

        err = (pred - observed)[mask]
        mae_reg = np.abs(err)

        horizon_obs = futur_planned[mask] + observed[mask]
        valid_horizon_mask = (horizon_obs_bins[0] <= horizon_obs) & (horizon_obs < horizon_obs_bins[-1])
        delay_delta = (observed - translation)[mask]

        _update_mae_bins(mae_delay, counter_delay, mae_reg, delay_delta, delay_delta_bins, valid_horizon_mask)
        _update_mae_bins(mae_horizon, counter_horizon, mae_reg, horizon_obs, horizon_obs_bins)

        se = np.square(err, dtype=np.float64)[valid_horizon_mask]
        sse += se.sum()
        n_total += se.shape[0]

    mae_delay = _finalize_mae(mae_delay, counter_delay)
    mae_horizon = _finalize_mae(mae_horizon, counter_horizon)

    mse_global = sse / n_total if n_total else np.nan
    return mae_delay, mae_horizon, counter_delay, counter_horizon, mse_global

def _fill_missing_predictions(pred_slice: torch.Tensor, itinerary: torch.Tensor, start_pos: int, max_end_pos: int, start_state_time: float, 
            predictive_horizon: int) -> torch.Tensor:
    """
    Translate and fill simulator gaps (-1) in predicted arrival times using last known delay shifted toward present if needed.

    Args:
        pred_slice (torch.Tensor): Predicted times, shape (num_samples, L), with -1 where missing.
        itinerary (torch.Tensor): Itinerary matrix with planned/observed times; uses cols 3 (planned) and 4 (observed).
        start_pos (int): Index of the starting station in the itinerary.
        max_end_pos (int): Last station index to consider when filling.
        start_state_time (float): Start time (seconds) of the current state.
        predictive_horizon (int): Horizon (minutes) used to clamp minimal predicted times.

    Returns:
        torch.Tensor: The filled prediction slice of shape (num_samples, L).
    """

    no_pred_mask_all = pred_slice == -1
    for j, no_pred_mask in enumerate(no_pred_mask_all):
        if no_pred_mask.any():
            last_pred_idx = start_pos + no_pred_mask.int().argmax()
            if last_pred_idx == start_pos:
                last_delay = itinerary[last_pred_idx, 4] - itinerary[last_pred_idx, 3]
            else:
                last_delay = float(pred_slice[j, last_pred_idx - start_pos - 1] - itinerary[last_pred_idx, 3])
            boundary = start_state_time + predictive_horizon * 60
            if itinerary[last_pred_idx + 1, 3] + last_delay < boundary:
                last_delay += boundary - (itinerary[last_pred_idx + 1, 3] + last_delay)
            pred_slice[j, no_pred_mask] = itinerary[last_pred_idx + 1:max_end_pos + 1, 3].double() + last_delay
    return pred_slice

def _reduce_predictions(pred_slice: torch.Tensor, reduce_fnc: str) -> torch.Tensor:
    """
    Aggregate multiple simulated prediction trajectories into a single estimate.

    Args:
        pred_slice (torch.Tensor): Predicted times, shape (num_samples, L).
        reduce_fnc (str): Reduction function, either 'median' or 'mean'.

    Returns:
        torch.Tensor: Reduced predictions, shape (L,).
    """
    if reduce_fnc == 'median':
        return torch.median(pred_slice, dim=0).values
    elif reduce_fnc == 'mean':
        # Treat -1 as missing by replacing with NaN, then compute nanmean
        pred_slice = pred_slice.clone().float()
        pred_slice[pred_slice == -1] = float('nan')
        return torch.nanmean(pred_slice, dim=0)
    else:
        raise ValueError(f"Unsupported reduce function: {reduce_fnc}")


def compute_mae_simulation(
    output: dict,
    itineraries_dict: dict,
    initial_state_metadata: list,
    start_state_time: float,
    predictive_horizon: int,
    reduce_fnc: str,
    delay_delta_bins: np.ndarray,
    horizon_obs_bins: np.ndarray,
    nb_pred_max: int,
    nb_future_st: int
) -> tuple:
    """
    Compute simulation-based MAE (per delay-delta and horizon bins) and total SSE
    for a rollout, given simulator outputs and itinerary metadata.

    Args:
        output (dict): Simulator outputs containing predictions info (e.g., 'predictions', 'mapper', 'start_pos').
        itineraries_dict (dict): Itineraries store indexed by date; contains routes and indices.
        initial_state_metadata (list): Metadata rows (e.g., (date_dep, train_no)) for the initial state.
        start_state_time (float): Start time of the state (in seconds).
        predictive_horizon (int): Prediction horizon in minutes.
        reduce_fnc (str): Reduction function for multiple simulations ('median' or 'mean').
        delay_delta_bins (np.ndarray): Bin edges for delay deltas.
        horizon_obs_bins (np.ndarray): Bin edges for observed horizons.
        nb_pred_max (int): Maximum number of future predictions per rollout.
        nb_future_st (int): Number of future stations to ignore at the tail.

    Returns:
        tuple: (mae_delay, mae_horizon, counter_delay, counter_horizon, sse)
            - mae_delay (torch.Tensor): MAE per delay-delta bin.
            - mae_horizon (torch.Tensor): MAE per horizon bin.
            - counter_delay (torch.Tensor): Counts per delay-delta bin.
            - counter_horizon (torch.Tensor): Counts per horizon bin.
            - sse (float): Sum of squared errors over valid horizons.
    """

    mae_delay = torch.zeros((len(delay_delta_bins)-1,))
    mae_horizon = torch.zeros((len(horizon_obs_bins)-1,))
    counter_delay = torch.zeros((len(delay_delta_bins)-1,))
    counter_horizon = torch.zeros((len(horizon_obs_bins)-1,))
    sse = 0.0

    end_time = start_state_time + 60 * predictive_horizon

    for datdep, train_no in initial_state_metadata:
        iti_idx = itineraries_dict[datdep]['train_no'][train_no]
        itinerary = itineraries_dict[datdep]['data'][iti_idx]
        output_idx = output['mapper'][(datdep, train_no)]
        start_pos = output['start_pos'][output_idx]
        pred = output['predictions'][output_idx]

        horizon_obs = itinerary[:, 4] - start_state_time
        valid_horizon_mask = (horizon_obs_bins[0] <= horizon_obs) & (horizon_obs < horizon_obs_bins[-1])
        valid_horizon_mask[-nb_future_st:] = False
        if not valid_horizon_mask.any():
            continue
        max_end_pos = valid_horizon_mask.nonzero()[-1, 0].item()
        iti_max_pos = itineraries_dict[datdep]['max_pos'][iti_idx]
        max_end_pos = min(max_end_pos, start_pos + nb_pred_max, iti_max_pos)

        pred_slice = pred[:, start_pos + 1:max_end_pos + 1].cpu()
        pred_slice = _fill_missing_predictions(pred_slice, itinerary, start_pos, max_end_pos, start_state_time, predictive_horizon)
        median_simulated_times = _reduce_predictions(pred_slice, reduce_fnc)

        past_delay_1 = itinerary[start_pos, 4] - itinerary[start_pos, 3]
        translation_next_station = itinerary[start_pos + 1, 3] + past_delay_1
        delay = past_delay_1
        if translation_next_station < start_state_time:
            delay += start_state_time - translation_next_station

        translation = itinerary[start_pos + 1:max_end_pos + 1, 3] + delay
        observed_times = itinerary[start_pos + 1:max_end_pos + 1, 4]
        err = median_simulated_times - observed_times
        mae_sim = np.abs(err)

        horizon_obs = itinerary[start_pos + 1:max_end_pos + 1, 4] - start_state_time
        valid_horizon_mask = (horizon_obs_bins[0] <= horizon_obs) & (horizon_obs < horizon_obs_bins[-1])
        delay_delta = (itinerary[start_pos + 1:max_end_pos + 1, 4] - itinerary[start_pos + 1:max_end_pos + 1, 3]) - delay

        _update_mae_bins(mae_delay, counter_delay, mae_sim, delay_delta, delay_delta_bins, valid_horizon_mask)
        _update_mae_bins(mae_horizon, counter_horizon, mae_sim, horizon_obs, horizon_obs_bins)

        sse += np.square(err[valid_horizon_mask], dtype=np.float64).sum()
        
    mae_delay = _finalize_mae(mae_delay, counter_delay)
    mae_horizon = _finalize_mae(mae_horizon, counter_horizon)

    return mae_delay, mae_horizon, counter_delay, counter_horizon, sse

def simulate_and_compute_mae(initial_states: list, metadatas: list, model, itineraries: dict, action_constraint: bool, nb_samples: int, predictive_horizon: int, reduce_fnc: str, nb_pred_max: int, device: str, column_mapping: dict, cat_cols_metadata: dict, stations_emb: dict, lines_emb: dict, dataset_config: dict, delay_delta_bins: np.ndarray, horizon_obs_bins: np.ndarray, net_type: str, local_features: bool=False) -> tuple:
    """
    Run the simulator on initial states and compute MAE and squared error metrics per sample.

    Args:
        initial_states (list): Batch of initial state tensors.
        metadatas (list): Batch of metadata tensors per state.
        model: Policy/model used by the simulator.
        itineraries (dict): Itineraries store.
        action_constraint (bool): Whether to constrain actions.
        nb_samples (int): Number of simulated trajectories per state.
        predictive_horizon (int): Horizon in minutes.
        reduce_fnc (str): 'median' or 'mean' reduction across samples.
        nb_pred_max (int): Max number of future stations on which to make a prediction per rollout.
        device (str): Torch device string.
        column_mapping (dict): Feature mapping for the simulator.
        cat_cols_metadata (dict): Categorical feature metadata.
        stations_emb (dict): Station embeddings.
        lines_emb (dict): Line embeddings.
        dataset_config (dict): Simulator/dataset configuration.
        delay_delta_bins (np.ndarray): Bin edges for delay delta.
        horizon_obs_bins (np.ndarray): Bin edges for observed horizon.
        net_type (str): Network type ('xgboost', 'mlp', 'transformer').
        local_features (bool, optional): Use local features. Default is False.

    Returns:
        tuple:
            mae_delay_list (list): MAE per delay-delta bin, one array per sample.
            mae_horizon_list (list): MAE per horizon bin, one array per sample.
            counter_delay_list (list): Counts per delay-delta bin, one array per sample.
            counter_horizon_list (list): Counts per horizon bin, one array per sample.
            ssse (float): Sum of squared errors across all samples.
    """


    if net_type != 'xgboost': 
        model.eval()

    cfg = dataset_config
    deltat = cfg['deltat']
    nb_past_station = cfg['nb_past_station_sim']
    nb_future_station = cfg['nb_future_station_sim']
    embedding_size = cfg['embedding_size']
    idle_time_end = cfg['idle_end']

    mae_delay_list = []
    mae_horizon_list = []
    counter_delay_list = []
    counter_horizon_list = []
    ssse = 0

    states_time = [metadata[0, 0] for metadata in metadatas]
    initial_states_metadata = [metadata[:, 1:] for metadata in metadatas]

    sim = Simulator(
        model, deltat, column_mapping, cat_cols_metadata, stations_emb, lines_emb, device, 
        nb_past_station, nb_future_station, embedding_size, idle_time_end, net_type, local_features
    )

    s, it = sim.predict_delay(
        initial_states,
        initial_states_metadata,
        states_time,
        predictive_horizon,
        nb_samples,
        'sampling',
        action_constraint,
        itineraries=itineraries,
    )

    for sim_idx in range(len(metadatas)):
        mae_delay, mae_horizon, counter_delay, counter_horizon, sse = compute_mae_simulation(
            s.output[sim_idx],
            it,
            initial_states_metadata[sim_idx],
            states_time[sim_idx],
            predictive_horizon,
            reduce_fnc,
            delay_delta_bins,
            horizon_obs_bins,
            nb_pred_max,
            nb_future_station,
        )
        mae_delay_list.append(mae_delay)
        mae_horizon_list.append(mae_horizon)
        counter_delay_list.append(counter_delay)
        counter_horizon_list.append(counter_horizon)
        ssse += sse

    return mae_delay_list, mae_horizon_list, counter_delay_list, counter_horizon_list, ssse