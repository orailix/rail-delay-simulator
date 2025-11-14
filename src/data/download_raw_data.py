import argparse
import os
import requests

import numpy as np

def main() -> None:
    """ 
    Entry point for downloading punctuality CSV files.

    Parses CLI arguments, iterates over the specified time range,
    and downloads monthly CSV files from the Infrabel public data portal.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("save_folder_path", type=str, help="Path of the folder to save the data.")
    parser.add_argument("first_year", type=int, help="Year of the first file.")
    parser.add_argument("first_month", type=int, help="Month of the first file.")
    parser.add_argument("last_year", type=int, help="Year of the last file.")
    parser.add_argument("last_month", type=int, help="Month of the last file.")
    
    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_folder_path, exist_ok=True)
    
    years = np.arange(args.first_year, args.last_year+1, 1)

    for year in years:
        start_month = args.first_month if year == args.first_year else 1
        end_month   = args.last_month  if year == args.last_year  else 12
        for month in range(start_month, end_month + 1):
            mm = f"{month:02d}"
            url = f'https://fr.ftp.opendatasoft.com/infrabel/PunctualityHistory/Data_raw_punctuality_{year}{mm}.csv' 
            save_path = os.path.join(args.save_folder_path, f'Data_raw_punctuality_{year}{mm}.csv')

            print(f"Downloading {url} → {save_path}")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  ✗ failed: {e}")
                continue

            with open(save_path, "wb") as f:
                f.write(resp.content)
            print("  ✓ saved")

if __name__ == "__main__":
    main()