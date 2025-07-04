"""
Visualization tools for EndoSign analysis results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_group_counts(df: pd.DataFrame, group_col: str = "Group", ax=None):
    """
    Plot the count of patients in each group.
    """
    if ax is None:
        ax = plt.gca()
    sns.countplot(x=group_col, data=df, ax=ax, palette="muted")
    ax.set_title(f"{group_col} Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Count")

def plot_biomarker_boxplots(df: pd.DataFrame, biomarker_cols: list, group_col: str = "Group", n_cols: int = 4):
    """
    Multi-panel boxplots for biomarker values by group.
    """
    n_biomarkers = len(biomarker_cols)
    n_rows = int(np.ceil(n_biomarkers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    for idx, biomarker in enumerate(biomarker_cols):
        row, col = divmod(idx, n_cols)
        sns.boxplot(
            x=group_col,
            y=biomarker,
            data=df,
            ax=axes[row][col],
            palette="Set2"
        )
        axes[row][col].set_title(biomarker)
        axes[row][col].set_xlabel("")
        axes[row][col].set_ylabel("Value")
    # Hide any empty subplots
    for i in range(idx+1, n_rows*n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_umap(
    umap_embedding, labels, title="UMAP Projection", ax=None
):
    """
    Scatter plot of UMAP embedding colored by labels.
    Args:
        umap_embedding: np.ndarray shape (n_samples, 2)
        labels: list or Series of group labels
    """
    import seaborn as sns
    import pandas as pd
    if ax is None:
        ax = plt.gca()
    plot_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
    plot_df["Label"] = labels
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="Label", palette="Set1", data=plot_df, ax=ax, alpha=0.8
    )
    ax.set_title(title)
    ax.legend(title="Group")

def plot_feature_importance(importance, feature_names, top_n=20):
    """
    Barplot of top feature importances.
    Args:
        importance: array-like of importances
        feature_names: list of names
    """
    importances = pd.Series(importance, index=feature_names).sort_values(ascending=False)[:top_n]
    plt.figure(figsize=(8, max(4, top_n * 0.4)))
    sns.barplot(x=importances.values, y=importances.index, palette="Blues_r")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def plot_biomarker_heatmap(
    df: pd.DataFrame,
    biomarker_cols: list,
    group_col: str = "Group",
    by_group_mean: bool = True,
    cmap: str = "vlag"
):
    """
    Plot a heatmap of biomarker values.
    - If by_group_mean: shows group-wise average expression.
    - Else: shows all patients (may be large).
    """
    if by_group_mean:
        data = df.groupby(group_col)[biomarker_cols].mean().T
        plt.figure(figsize=(max(8, len(df[group_col].unique())*2), max(6, len(biomarker_cols)*0.5)))
        sns.heatmap(data, annot=True, cmap=cmap, cbar_kws={"label": "Mean Expression"})
        plt.title("Group-wise Mean Biomarker Heatmap")
        plt.xlabel("Group")
        plt.ylabel("Biomarker")
    else:
        data = df[biomarker_cols].T
        plt.figure(figsize=(max(10, df.shape[0] * 0.15), max(6, len(biomarker_cols) * 0.5)))
        sns.heatmap(data, cmap=cmap, cbar_kws={"label": "Expression"})
        plt.title("Biomarker Heatmap (All Patients)")
        plt.xlabel("Patient")
        plt.ylabel("Biomarker")
    plt.tight_layout()
    plt.show()

def plot_biomarker_correlation(
    df: pd.DataFrame,
    biomarker_cols: list,
    method: str = "spearman",
    annot: bool = False,
    cmap: str = "coolwarm"
):
    """
    Plot a correlation matrix heatmap for biomarkers.
    Args:
        method: 'spearman' (recommended for non-normal) or 'pearson'
        annot: Show correlation values on the heatmap
    """
    corr = df[biomarker_cols].corr(method=method)
    plt.figure(figsize=(max(8, len(biomarker_cols) * 0.5), max(6, len(biomarker_cols) * 0.5)))
    sns.heatmap(corr, annot=annot, cmap=cmap, center=0, cbar_kws={"label": f"{method.title()} Correlation"})
    plt.title(f"Biomarker Correlation Matrix ({method.title()})")
    plt.xlabel("Biomarker")
    plt.ylabel("Biomarker")
    plt.tight_layout()
    plt.show()
