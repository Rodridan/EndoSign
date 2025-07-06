import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from statsmodels.stats.multitest import multipletests

def compare_biomarkers(
    df: pd.DataFrame,
    biomarker_cols: list,
    group_col: str = "Group",
    group1: str = "Endometriosis",
    group2: str = "Control",
    correction: str = "fdr_bh"
) -> pd.DataFrame:
    """
    Compare each biomarker between two groups (default: Endometriosis vs Control) using Mann-Whitney U test.
    Returns a DataFrame with group medians, U-statistic, p-value, and adjusted p-value.
    """
    results = []
    for biomarker in biomarker_cols:
        group1_vals = df[df[group_col] == group1][biomarker].dropna()
        group2_vals = df[df[group_col] == group2][biomarker].dropna()
        if len(group1_vals) > 0 and len(group2_vals) > 0:
            stat, p = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided')
            results.append({
                "Biomarker": biomarker,
                f"{group1}_median": np.median(group1_vals),
                f"{group2}_median": np.median(group2_vals),
                "U_stat": stat,
                "p_value": p
            })
    stats_df = pd.DataFrame(results)
    if not stats_df.empty and correction:
        _, p_adj, _, _ = multipletests(stats_df["p_value"], method=correction)
        stats_df["p_adj"] = p_adj
    return stats_df.sort_values("p_value")

def biomarker_kruskal(
    df: pd.DataFrame,
    biomarker_cols: list,
    stage_col: str = "rASRM_stage",
    stage_order: list = ["I", "II", "III", "IV"],
    group_col: str = "Group",
    case_value: str = "Endometriosis"
) -> pd.DataFrame:
    """
    Kruskal-Wallis test for trend across rASRM stages for each biomarker (cases only).
    Returns H and p-value for each biomarker.
    """
    results = []
    df_case = df[df[group_col] == case_value]
    for biomarker in biomarker_cols:
        data = []
        for stage in stage_order:
            vals = df_case[df_case[stage_col] == stage][biomarker].dropna()
            if len(vals) > 0:
                data.append(vals)
        if len(data) == len(stage_order):
            stat, p = kruskal(*data)
            results.append({
                "Biomarker": biomarker,
                "H_stat": stat,
                "p_value": p
            })
    return pd.DataFrame(results).sort_values("p_value")

def biomarker_corr_with_feature(
    df: pd.DataFrame,
    biomarker_cols: list,
    feature_col: str = "Pain_Score",
    group_col: str = "Group",
    group_value: str = "Endometriosis"
) -> pd.DataFrame:
    """
    Spearman correlation of each biomarker with a given clinical feature (e.g. Pain_Score), in selected group.
    """
    df_sel = df[df[group_col] == group_value]
    results = []
    for biomarker in biomarker_cols:
        vals = df_sel[biomarker].dropna()
        feature = df_sel.loc[vals.index, feature_col].dropna()
        if len(vals) == len(feature) and len(vals) > 0:
            rho, p = spearmanr(vals, feature)
            results.append({
                "Biomarker": biomarker,
                "rho": rho,
                "p_value": p
            })
    results_df = pd.DataFrame(results).sort_values("p_value")
    # Optional: FDR correction
    if not results_df.empty:
        _, p_adj, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
        results_df["p_adj"] = p_adj
    return results_df

def biomarker_stats(
    df: pd.DataFrame,
    biomarker_cols: list,
    group_col: str = "Group"
) -> pd.DataFrame:
    """
    Returns group-wise medians and IQRs for each biomarker.
    """
    stats = []
    for biomarker in biomarker_cols:
        grouped = df.groupby(group_col)[biomarker].agg(['median', 'mean', 'std', 'min', 'max', 'count'])
        grouped['biomarker'] = biomarker
        stats.append(grouped)
    return pd.concat(stats, axis=0)

