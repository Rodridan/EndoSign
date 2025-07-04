"""
Data cleaning and preprocessing routines for EndoSign.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning: drop duplicates, handle missing values.
    Args:
        df (pd.DataFrame): Raw data.
    Returns:
        pd.DataFrame: Cleaned data.
    """
    df = df.copy()
    df.drop_duplicates(subset="Patient_ID", inplace=True)
    df.fillna(method='ffill', inplace=True)  # Forward-fill as default; adjust as needed
    return df

def normalize_biomarkers(df: pd.DataFrame, biomarker_cols=None) -> pd.DataFrame:
    """
    Standardize biomarker columns (mean=0, std=1).
    Args:
        df (pd.DataFrame): Data with biomarker columns.
        biomarker_cols (list): List of biomarker column names.
    Returns:
        pd.DataFrame: DataFrame with normalized biomarker values.
    """
    df = df.copy()
    if biomarker_cols is None:
        biomarker_cols = [col for col in df.columns if col.startswith("Cytokine_")]
    scaler = StandardScaler()
    df[biomarker_cols] = scaler.fit_transform(df[biomarker_cols])
    return df

def encode_rasrm_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode rASRM stage as ordinal values for modeling.
    """
    stage_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, "": 0}
    df = df.copy()
    if 'rASRM_stage' in df.columns:
        df['rASRM_stage_num'] = df['rASRM_stage'].map(stage_map).astype('Int64')
    return df

def fill_missing_biomarkers(df: pd.DataFrame, biomarker_cols=None, method='median') -> pd.DataFrame:
    """
    Fill missing biomarker values using the chosen method.
    """
    df = df.copy()
    if biomarker_cols is None:
        biomarker_cols = [col for col in df.columns if col.startswith("Cytokine_")]
    for col in biomarker_cols:
        if method == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif method == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
    return df
