"""
Analysis and exploration of plasma biomarker landscapes.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

def analyze_biomarker_distributions(df: pd.DataFrame, biomarker_cols: list, group_col: str = "Group"):
    """
    Computes summary stats (mean, std, median, IQR) for each biomarker by group.
    Returns: DataFrame with results.
    """
    summary = {}
    for biomarker in biomarker_cols:
        group_stats = df.groupby(group_col)[biomarker].agg(['mean', 'std', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]).rename(
            columns={'<lambda_0>': 'q25', '<lambda_1>': 'q75'}
        )
        summary[biomarker] = group_stats
    return summary

def compare_biomarkers_statistical(
    df: pd.DataFrame, biomarker_cols: list, group_col: str = "Group",
    group_a: str = "Endometriosis", group_b: str = "Control", method: str = "mannwhitney"
):
    """
    For each biomarker, compare two groups using t-test or Mann-Whitney U test.
    Returns: DataFrame of p-values and effect sizes.
    """
    results = []
    for biomarker in biomarker_cols:
        x = df[df[group_col] == group_a][biomarker].dropna()
        y = df[df[group_col] == group_b][biomarker].dropna()
        if method == "ttest":
            stat, pval = ttest_ind(x, y, equal_var=False)
        elif method == "mannwhitney":
            stat, pval = mannwhitneyu(x, y, alternative="two-sided")
        else:
            raise ValueError("Unsupported method")
        # Effect size (Cohen's d)
        eff_size = (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2)
        results.append({
            "biomarker": biomarker,
            "p_value": pval,
            "effect_size": eff_size
        })
    return pd.DataFrame(results).sort_values("p_value")

def find_candidate_biomarkers(
    stats_df: pd.DataFrame, p_thresh: float = 0.05, effect_thresh: float = 0.5
):
    """
    Return biomarkers that are statistically significant and have substantial effect size.
    """
    candidates = stats_df[(stats_df["p_value"] < p_thresh) & (stats_df["effect_size"].abs() > effect_thresh)]
    return candidates["biomarker"].tolist()

