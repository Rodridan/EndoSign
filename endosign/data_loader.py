"""
Data loading and validation functions for EndoSign.
"""

import pandas as pd
import os

REQUIRED_COLUMNS = [
    "Patient_ID", "Age", "BMI", "Leiomyoma", "Group"
    # Extend as needed: "rASRM_stage", Enzian columns, at least one biomarker
]

def load_patient_data(filename: str) -> pd.DataFrame:
    """
    Load patient and biomarker data from CSV or Excel.
    Args:
        filename (str): Path to data file.
    Returns:
        pd.DataFrame: Loaded data.
    Raises:
        FileNotFoundError, ValueError
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(filename)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filename)
    else:
        raise ValueError("Supported formats: .csv, .xlsx")
    return df

def validate_patient_data(df: pd.DataFrame, required_cols=None) -> bool:
    """
    Validate that the DataFrame contains all required columns.
    Args:
        df (pd.DataFrame): Data to validate.
        required_cols (list): List of required column names.
    Returns:
        bool: True if valid, False otherwise.
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return False
    return True

def guess_biomarker_columns(df: pd.DataFrame) -> list:
    """
    Return all columns that match typical biomarker naming conventions.
    """
    return [col for col in df.columns if col.startswith("Cytokine_") or col.startswith("IL") or col.startswith("TNF")]

# Optionally: utility to add 'Group' column if missing, e.g. assign all as 'Endometriosis' or 'Control'
def assign_group(df: pd.DataFrame, control_patients=None) -> pd.DataFrame:
    """
    Assign a Group column based on Patient_ID or other rules.
    """
    if 'Group' in df.columns:
        return df
    if control_patients is not None:
        df['Group'] = df['Patient_ID'].isin(control_patients).map({True: 'Control', False: 'Endometriosis'})
    else:
        df['Group'] = 'Endometriosis'
    return df

