import argparse
import numpy as np
import os
from src.utils.utils import save_pickle

def main() -> None:
    """ 
    Entry point to build and save an evaluation config.

    Parses CLI arguments, constructs evaluation configuration with
    prediction horizon, trajectory count, and bin definitions, then
    saves it to the specified path.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str, help="Path to save the config.")
    parser.add_argument("pred_horizon", type=int, help="Number of minutes to simulate.")
    parser.add_argument("nb_traj", type=int, help="Number of trajectories to simulate in parallel.")
    parser.add_argument("--horizon-obs-bins", nargs="+", type=int, required=True, help="Horizon bins edges in minute.")
    parser.add_argument("--delay-delta-bins", nargs="+", type=int, required=True, help="Delay delta bins edges in minute. Will add -inf at the beggining and +inf at the end.")
    
    args = parser.parse_args()
    print(args)

    horizon_obs_bins = np.array(args.horizon_obs_bins) * 60
    delay_delta_bins = np.array([-np.inf] + args.delay_delta_bins + [np.inf]) * 60

    eval_config = {
        'pred_horizon':args.pred_horizon,
        'nb_traj':args.nb_traj,
        'horizon_obs_bins':horizon_obs_bins,
        'delay_delta_bins':delay_delta_bins
    }

    save_pickle(eval_config, args.save_path)

if __name__ == "__main__":
    main()