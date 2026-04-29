
from typing import Union
import os
import pandas as pd
from pathlib import Path

def load_disk_data_example(data_dir_path: Union[str, Path]) -> pd.DataFrame:
    """ Extracts a data from a specific disk from given Backblaze Drive Stat data.

    Args:
        data_dir_path (Union[str, Path]): path to the data directory.

    Returns:
        pd.DataFrame: All datapoints for disk model 'ST4000DM000' with serial number 'S301KWJY'
    """
    data_dir = Path(data_dir_path)
    dfs = []
    for file_path in data_dir.iterdir():
        if file_path.is_file():
            df = pd.read_csv(file_path, parse_dates=['date'],
                             usecols=['date', 'serial_number', 'model', 'failure',
                                      'smart_4_raw', 'smart_7_raw', 'smart_9_raw',
                                      'smart_190_raw', 'smart_193_raw', 'smart_194_raw',
                                      'smart_240_raw', 'smart_241_raw', 'smart_242_raw'
                                      ]
                             )
            df = df[df['model'] == 'ST4000DM000']
            df = df[df['serial_number'] == 'S301KWJY']
            dfs.append(df)
        else:
            pass
    model_df = pd.concat(dfs)
    model_df = model_df.set_index('date').sort_index()
    return model_df
