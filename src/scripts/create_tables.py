import os

import pandas as pd
import numpy as np

from collections import defaultdict

def create_table(runs: list, metrics_to_show: list | None = None, digits: int = 4, pm_symbol: str = "±") -> pd.DataFrame:
    """
    Aggregate metrics from experiment runs into a summary table.

    Args:
        runs (list): List of tuples (model, method, base_path) pointing to experiment directories.
        metrics_to_show (list | None): Metrics to include in the table. If None, detect numeric metrics automatically.
        digits (int): Number of decimal places for formatted results. Default is 4.
        pm_symbol (str): Symbol used to display "mean ± std". Default is "±".

    Returns:
        pandas.DataFrame: Table indexed by (model, method), with metrics formatted as "mean ± std".
    """
    bucket = defaultdict(lambda: defaultdict(list))

    for model, method, base_path in runs:
        if not os.path.isdir(base_path):
            raise FileNotFoundError(base_path)
        for run_dir in os.listdir(base_path):
            csv_path = os.path.join(base_path, run_dir, "metrics_log.csv")
            if not os.path.isfile(csv_path):
                continue
            try:
                row = pd.read_csv(csv_path).iloc[-1]
            except Exception:
                continue

            if metrics_to_show is None:
                metrics_to_show = [m for m, v in row.items()
                                   if np.issubdtype(type(v), np.number)]

            for metric in metrics_to_show:
                if metric in row and np.issubdtype(type(row[metric]), np.number):
                    bucket[(model, method)][metric].append(float(row[metric]))

    if not bucket:
        print("No numeric metrics found.")
        return pd.DataFrame()

    fmt = f"{{:.{digits}f}} {pm_symbol} {{:.{digits}f}}"
    records = []

    for (model, method), metrics_dict in bucket.items():
        record = {}
        for metric in metrics_to_show:
            vals = metrics_dict.get(metric, [])
            if vals:
                mu, sd = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                record[metric] = fmt.format(mu, sd)
            else:
                record[metric] = "—"
        records.append(((model, method), record))

    df = (pd.DataFrame.from_dict(dict(records), orient="index")
                    .rename_axis(index=["model", "method"]))

    return df

def main() -> None:
    """
    Entry point for summarizing experiment results into tables.

    Builds tables of aggregated metrics from predefined experiment runs,
    prints them in plain text and LaTeX formats for reporting.
    """
    print('Table 1:')

    runs_list = [
        ('Transformer','Reg',f'runs/tr_reg/final/'),
        ('Transformer','BC',f'runs/tr_bc/final/'),
        ('Transformer','DCIL',f'runs/tr_dcil/final/'),
        ('MLP','Reg',f'runs/mlp_reg/final/'),
        ('MLP','BC',f'runs/mlp_bc/final/'),
        ('MLP','DCIL',f'runs/mlp_dcil/final/'),
        ('XGBOOST','Reg',f'runs/xgboost_reg/final/'),
        ('XGBOOST','BC',f'runs/xgboost_bc/final/'),
    ]
    metrics = ['mae','rmse','mse']
    
    tbl = create_table(runs_list, metrics, digits = 2)
    print('\n',tbl)

    latex = tbl.to_latex(
        multirow=True,
        caption="Test-set errors (mean ± std)",
        label="tab:test_errors"
    )
    print('\n',latex)

    print('\n\n\nTable 2:')

    metrics = [f'hor{i}' for i in range(6)]

    tbl = create_table(runs_list, metrics, digits = 2)
    print('\n',tbl)

    latex = tbl.to_latex(
        multirow=True,
        caption="Test-set errors (mean ± std)",
        label="tab:test_errors"
    )
    print('\n',latex)

if __name__ == "__main__":
    main()