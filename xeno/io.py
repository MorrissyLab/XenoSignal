import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path


def read_ipsae_results(data_path):
    all_dfs = []
    data_path = Path(data_path)  # Convert string to Path object
    for sample_dir in tqdm(data_path.iterdir(), desc="Processing samples"):
        if not sample_dir.is_dir():
            continue

        for i in range(5):
            file_path = sample_dir / f"fold_{sample_dir.name}_model_{i}_10_10.txt"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, sep=r'\s+')
                    # Precompute metadata
                    df['job_name'] = sample_dir.name
                    df['model_id'] = str(i)
                    df = df.rename(columns={'Model': 'model_path'})
                    all_dfs.append(df)
                except:
                    print(file_path)
    df_all = pd.concat(all_dfs, ignore_index=True)
    cols = ['job_name', 'model_id'] + [col for col in df_all.columns if col not in ['job_name', 'model_id']]
    df_all = df_all[cols]
    return df_all