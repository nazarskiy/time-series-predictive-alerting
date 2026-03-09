import os
import pandas as pd
from typing import Tuple

def load_smd_machine(base_path: str, machine_name: str) -> pd.DataFrame:
    """
    Loads the metrics and labels for a specific machine and combines them
    """

    # loading from the test folder since in this dataset there are no incident = 1 in train, splitting into train/test later on
    metrics_path = os.path.join(base_path, 'test', f'{machine_name}.txt')
    labels_path = os.path.join(base_path, 'test_label', f'{machine_name}.txt')
    
    metrics_df = pd.read_csv(metrics_path, header=None)
    metrics_df.columns = [f'metric_{i}' for i in range(metrics_df.shape[1])]
    
    labels_df = pd.read_csv(labels_path, header=None, names=['is_incident'])
    
    df = pd.concat([metrics_df, labels_df], axis=1)
    
    return df

def sliding_window_transform(df, W: int = 30, H: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms raw time-series metrics into a sliding-window supervised learning dataset
    """
    
    # The W=30, H=10 decision is explained in the documentation
    target = df['is_incident'].rolling(window=H, min_periods=1).max().shift(-H)
    target.name = 'target'
    
    X_list = []
    metrics_df = df.drop(columns=['is_incident'])
    
    for i in range(W, 0, -1):
        shifted_metrics = metrics_df.shift(i)
        shifted_metrics.columns = [f"{col}_t-{i}" for col in metrics_df.columns]
        X_list.append(shifted_metrics)
        
    X = pd.concat(X_list, axis=1)
    
    dataset = pd.concat([X, target], axis=1)
    dataset = dataset.dropna()
    
    y_final = dataset['target']
    X_final = dataset.drop(columns=['target'])
    
    print(f"{X_final.shape}")
    print(f"{y_final.shape}")
    print(f"Incidents to predict: {int(y_final.sum())}")
    
    return X_final, y_final

def split_time_series(X, y, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Splits the data chronologically into train, validation, and test sets
    """
    total_samples = X.shape[0]
    
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def add_window_features(X, W: int = 30, num_metrics: int = 38) -> pd.DataFrame:
    """
    Calculates summary statistics for each metric's window and adds them as new features
    """
    X_new = X.copy()
    
    for m in range(num_metrics):
        metric_cols = [f"metric_{m}_t-{i}" for i in range(W, 0, -1)]
        
        window_data = X[metric_cols]
        
        X_new[f"metric_{m}_mean"] = window_data.mean(axis=1)
        X_new[f"metric_{m}_std"] = window_data.std(axis=1)
        X_new[f"metric_{m}_min"] = window_data.min(axis=1)
        X_new[f"metric_{m}_max"] = window_data.max(axis=1)
        X_new[f"metric_{m}_diff"] = X[f"metric_{m}_t-1"] - X[f"metric_{m}_t-{W}"]
        
    return X_new
