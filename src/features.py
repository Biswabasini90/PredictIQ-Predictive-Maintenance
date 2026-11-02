"""
PredictIQ - Feature Engineering Module
Contains physics-informed transformations and preprocessing functions.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS_RAW = [
    'air_temp',
    'process_temp',
    'rot_speed',
    'torque',
    'tool_wear',
    'Type_L',
    'Type_M'
]

FEATURE_COLUMNS_ENGINEERED = [
    'air_temp',
    'process_temp',
    'rot_speed',
    'torque',
    'tool_wear',
    'Type_L',
    'Type_M',
    'temp_diff',
    'power',
    'wear_torque'
]

COLUMN_RENAME_MAP = {
    'Air temperature [K]': 'air_temp',
    'Process temperature [K]': 'process_temp',
    'Rotational speed [rpm]': 'rot_speed',
    'Torque [Nm]': 'torque',
    'Tool wear [min]': 'tool_wear',
    'Machine failure': 'failure'
}

TARGET_LEAKAGE_COLS = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to clean alphanumeric names (compatible with XGBoost),
    drops identifiers and target leakage columns, and one-hot encodes 'Type'.
    """
    df = df.copy()
    # Rename columns to remove square brackets
    df = df.rename(columns=COLUMN_RENAME_MAP)
    
    # Drop identifiers and target leakage
    cols_to_drop = [c for c in TARGET_LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # One-hot encode Type (L, M, H - H is dropped as reference category)
    if 'Type' in df.columns:
        df = pd.get_dummies(df, columns=['Type'], drop_first=True, dtype=int)
        
    return df


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds physics-informed domain features based on machine operation equations:
    1. temp_diff: Difference between process temperature and ambient air temperature (Heat dissipation).
    2. power: Rotational mechanical power in Watts = Torque * Rotational Speed * 2*pi / 60.
    3. wear_torque: Overstrain metric = Tool wear * Torque.
    """
    df = df.copy()
    
    # 1. Temperature difference (Heat Dissipation Failure physics)
    df['temp_diff'] = df['process_temp'] - df['air_temp']
    
    # 2. Mechanical Power (W) (Power Failure physics: P = tau * omega)
    # Omega = (rot_speed * 2 * pi) / 60
    df['power'] = df['torque'] * df['rot_speed'] * (2 * np.pi / 60.0)
    
    # 3. Tool Overstrain (Overstrain Failure physics)
    df['wear_torque'] = df['tool_wear'] * df['torque']
    
    return df


def prepare_single_sample(air_temp: float,
                          process_temp: float,
                          rot_speed: float,
                          torque: float,
                          tool_wear: float,
                          machine_type: str = 'L') -> pd.DataFrame:
    """
    Prepares a single machine reading dictionary/DataFrame for live inference.
    """
    # Type one-hot encoding: L -> Type_L=1, Type_M=0; M -> Type_L=0, Type_M=1; H -> Type_L=0, Type_M=0
    type_l = 1 if machine_type.upper() == 'L' else 0
    type_m = 1 if machine_type.upper() == 'M' else 0
    
    data = {
        'air_temp': [float(air_temp)],
        'process_temp': [float(process_temp)],
        'rot_speed': [float(rot_speed)],
        'torque': [float(torque)],
        'tool_wear': [float(tool_wear)],
        'Type_L': [type_l],
        'Type_M': [type_m]
    }
    
    df = pd.DataFrame(data)
    df = engineer_physics_features(df)
    
    # Ensure correct column ordering
    return df[FEATURE_COLUMNS_ENGINEERED]
