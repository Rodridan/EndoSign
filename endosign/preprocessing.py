"""
Data cleaning and preprocessing routines for EndoSign.
"""
import os
import numpy as np
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
    df = df.ffill()  # Forward-fill as default; adjust as needed
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
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
    return df

def remove_biomarker_outliers(
    df: pd.DataFrame,
    biomarker_cols=None,
    outliers_path="data/interim/biomarker_outliers.csv",
    iqr_multiplier=1.5
) -> pd.DataFrame:
    """
    Remove outliers from biomarker columns using the IQR rule.
    Outlier rows are saved separately for review.
    
    Args:
        df (pd.DataFrame): Input data.
        biomarker_cols (list): Biomarker columns to check. If None, auto-detect.
        outliers_path (str): Where to save outlier rows.
        iqr_multiplier (float): IQR multiplier (1.5=standard).
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    df = df.copy()
    if biomarker_cols is None:
        biomarker_cols = [col for col in df.columns if col.startswith("Cytokine_")]
    
    # Identify outliers in any biomarker column
    outlier_mask = pd.Series(False, index=df.index)
    for col in biomarker_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outlier_mask = outlier_mask | (df[col] < lower) | (df[col] > upper)
    
    # Save outliers
    outliers = df[outlier_mask]
    if not outliers.empty:
        os.makedirs(os.path.dirname(outliers_path), exist_ok=True)
        outliers.to_csv(outliers_path, index=False)
    
    # Return cleaned DataFrame
    cleaned_df = df[~outlier_mask].reset_index(drop=True)
    return cleaned_df

def log_transform_biomarkers(df: pd.DataFrame, biomarker_cols=None) -> pd.DataFrame:
    """
    Log-transform biomarker columns using log1p (log(x+1)).
    Args:
        df (pd.DataFrame): DataFrame with biomarker columns.
        biomarker_cols (list): Columns to transform.
    Returns:
        pd.DataFrame: DataFrame with log-transformed biomarkers.
    """
    df = df.copy()
    if biomarker_cols is None:
        biomarker_cols = [col for col in df.columns if col.startswith("Cytokine_")]
    df[biomarker_cols] = np.log1p(df[biomarker_cols])
    return df


