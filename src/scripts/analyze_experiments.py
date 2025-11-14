import os
import math
import argparse
import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from src.utils.utils import load_pickle

def make_hashable(x: object) -> object:
    """
    Convert lists and dicts into hashable structures.

    Args:
        x (object): Input object (may be a list, dict, or any other type).

    Returns:
        object: Hashable equivalent of the input (lists → tuples, dicts → sorted tuples),
        or the object itself if already hashable.
    """
    if isinstance(x, list):
        return tuple(make_hashable(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
    return x

def analyze_experiments(base_path: str, metrics_to_show: list, target_metric: str, param_groups: list, nb_to_show: int = 5, highlight_best: int | None = None, 
            best_color: str = "black", save_path: str | None = None) -> pd.DataFrame:
    """
    Aggregate and plot experiment runs, optionally saving figures and returning a summary table.

    Args:
        base_path (str): Directory containing run subfolders, each with 'args.pkl' and 'metrics_log.csv'.
        metrics_to_show (list): List of metric column names to plot over epochs.
        target_metric (str): Metric used to rank runs (smaller is better).
        param_groups (list): List of parameter-name groups (lists/tuples) to define color groupings in plots.
        nb_to_show (int): Number of top runs to display in legends or summaries.
        highlight_best (int | None): If set, highlight the N-th best run (1-indexed) by target_metric.
        best_color (str): Matplotlib color for the highlighted runs.
        save_path (str | None): If provided, save figures to a multi-page PDF at this path.

    Returns:
        pandas.DataFrame: Summary table where each row corresponds to a run's best epoch by target_metric,
        including selected params and metric snapshots at that epoch.
    """
    runs = []
    for run_name in os.listdir(base_path):
        run_dir = os.path.join(base_path, run_name)
        if not os.path.isdir(run_dir):
            continue
        args_path = os.path.join(run_dir, "args.pkl")
        metrics_path = os.path.join(run_dir, "metrics_log.csv")
        if not (os.path.exists(args_path) and os.path.exists(metrics_path)):
            continue
        args = load_pickle(args_path)
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        runs.append({"name": run_name, "args": args, "df": df})

    if not runs:
        print(f"No valid runs found in {base_path}")
        return pd.DataFrame()

    ranked_runs = []
    for r in runs:
        if target_metric in r["df"].columns:
            v = r["df"][target_metric].dropna().min()
            if not np.isnan(v):
                ranked_runs.append((v, r["name"]))
    ranked_runs.sort(key=lambda x: x[0])  # smallest value is best

    highlight_run_name = None
    if isinstance(highlight_best, int) and highlight_best > 0 and highlight_best <= len(ranked_runs):
        highlight_run_name = ranked_runs[highlight_best - 1][1]

    pdf = PdfPages(save_path) if save_path else None
    
    for group in param_groups:
        combos = []
        for run in runs:
            vals = tuple(make_hashable(run["args"].get(p)) for p in group)
            combos.append(vals)
        unique_combos = sorted(set(combos))
        cmap = plt.get_cmap("tab10")
        color_map = {combo: cmap(i % 10) for i, combo in enumerate(unique_combos)}

        n_metrics = len(metrics_to_show)
        n_cols = 3
        n_rows = math.ceil(n_metrics / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_show):
            ax = axes[idx]
            seen = set()
            for run, combo in zip(runs, combos):
                df = run["df"]
                if metric not in df.columns:
                    continue
                epoch_col = df.columns[0]
                x_all = df[epoch_col]
                y_all = df[metric]
                mask = ~y_all.isna()
                if not mask.any():
                    continue
                x = x_all[mask]
                y = y_all[mask]

                if highlight_run_name and run["name"] == highlight_run_name:
                    color, lw, zo = best_color, 2.5, 10     # front
                else:
                    color, lw, zo = color_map[combo], 1.0, 1 # back

                label = str(combo) if combo not in seen else None
                if label:
                    seen.add(combo)
                ax.plot(x, y, label=label, color=color, linewidth=lw, zorder=zo)
            ax.set_xlabel(epoch_col)
            ax.set_ylabel(metric)
            title = ",".join(group) if isinstance(group, (list, tuple)) else str(group)
            ax.set_title(f"{metric} ({title})")
            if seen:
                ax.legend(title=title)
        for j in range(n_metrics, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        
        if pdf:                   
            pdf.savefig(fig)
            plt.close(fig)         
        else:
            plt.show()
            
    if pdf:
        pdf.close()

    summaries = []
    flat_params = [p for grp in param_groups for p in grp]
    for run in runs:
        df = run["df"]
        if target_metric not in df.columns:
            continue
        series = df[target_metric].dropna()
        if series.empty:
            continue
        best_idx = series.idxmin()
        epoch = int(df.iloc[best_idx, 0])
        value = series.min()
        params = {p: run["args"].get(p) for p in flat_params}
        metric_vals = {m: (df.at[best_idx, m] if m in df.columns else np.nan) for m in metrics_to_show}
        row = {**params, "best_epoch": epoch, "best_value": value, **metric_vals}
        summaries.append(row)

    if not summaries:
        print(f"No runs with metric {target_metric} found.")
        return pd.DataFrame()

    df_summary = pd.DataFrame(summaries).sort_values("best_value").head(nb_to_show).reset_index(drop=True)

    print(df_summary.to_markdown(index=False))

    best_run = min(
        runs,
        key=lambda r: (
            r["df"][target_metric].dropna().min() if target_metric in r["df"].columns else float("inf")
        )
    )
    print("\nBest config (full args.pkl):")
    best_args = best_run["args"]
    pprint.pprint(vars(best_args) if hasattr(best_args, "__dict__") else best_args)

    print(f"\nBest run directory: {os.path.join(base_path, best_run['name'])}")

def _csv(value: str) -> list:
    """
    Parse a comma-separated string into a list of stripped values.

    Args:
        value (str): Comma-separated string.

    Returns:
        list: List of non-empty, trimmed string values.
    """
    return [v.strip() for v in value.split(',') if v.strip()]

def _param_groups(value: str) -> list:
    """
    Parse a semicolon-separated string of parameter groups.

    Args:
        value (str): String where groups are separated by ';' and items within a group by ','.

    Returns:
        list: List of parameter groups, each as a tuple of strings.
    """
    if not value:
        return []
    groups = []
    for grp in value.split(';'):
        items = _csv(grp)
        if items:
            groups.append(tuple(items))
    return groups

def main() -> None:
    """
    Entry point for analyzing ML experiment runs.

    Parses CLI arguments, loads run arguments and metrics, ranks runs by the
    target metric, and generates plots (saved as a multi-page PDF).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path",        type=str,  help="Folder containing all run sub-directories.")
    parser.add_argument("metrics_to_show", type=_csv, help="Comma-separated list of metrics to plot.")
    parser.add_argument("target_metric",   type=str,  help="Metric used to rank runs (minimize).")
    parser.add_argument("save_path",       type=str,  help="Path of the single output file (e.g. plots.pdf).")
    parser.add_argument("--param_groups",  type=_param_groups,
                        default=[], help='Semicolon-separated groups of comma-separated parameter names, '
                                         'e.g. "lr;lr,momentum".')

    args = parser.parse_args()

    analyze_experiments(
        base_path=args.exp_path,
        metrics_to_show=args.metrics_to_show,
        target_metric=args.target_metric,
        param_groups=args.param_groups,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()