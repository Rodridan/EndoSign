"""
Visualization tools for EndoSign analysis results.
"""
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

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
    
def plot_venn(df):
    """
    Plots a Venn diagram showing the intersection between Endometriosis, Leiomyoma, and Controls.
    Expects columns: 'Group' (Control/Endometriosis), 'Leiomyoma' (Yes/No)
    """
    # Build sets of patient IDs for each group
    set_endometriosis = set(df[df['Group'] == 'Endometriosis']['Patient_ID'])
    set_leiomyoma = set(df[df['Leiomyoma'] == 'Yes']['Patient_ID'])
    set_controls = set(df[df['Group'] == 'Control']['Patient_ID'])
    
    plt.figure(figsize=(7, 7))
    venn = venn3(
        [set_endometriosis, set_leiomyoma, set_controls],
        set_labels=('Endometriosis', 'Leiomyoma', 'Controls')
    )
    plt.title("Patient Group Overlap: Endometriosis, Leiomyoma, Controls")
    plt.show()
    
def stacked_percentage_barplot(data, categories, group_col, ax, palette, title):
    # Compute counts
    count_df = (
        data.groupby([group_col])[categories]
        .apply(lambda x: (x == np.arange(x.shape[1])).sum(axis=0) if x.shape[1] > 1 else x.value_counts())
        .unstack(fill_value=0)
    )
    # Normalize to percentage
    count_df_pct = count_df.div(count_df.sum(axis=1), axis=0) * 100
    count_df_pct.plot(kind='bar', stacked=True, ax=ax, color=palette, edgecolor='k')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.legend(title=categories, bbox_to_anchor=(1.05, 1), loc='upper left')
    for label in ax.get_xticklabels():
        label.set_rotation(0)

def multiplot_endo_class(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios':[1,2]})
    group_col = "Group"
    enzian_cols = [col for col in df.columns if col.startswith("Enzian_")]
    stage_order = ['I', 'II', 'III', 'IV']
    enzian_03 = [c for c in enzian_cols if c not in ["Enzian_Fa", "Enzian_Fu", "Enzian_Fi", "Enzian_Fother"]]
    enzian_bin = ["Enzian_Fa", "Enzian_Fu", "Enzian_Fi", "Enzian_Fother"]

    # rASRM stacked barplot
    r_df = df[df[group_col] == "Endometriosis"]
    stage_palette = sns.color_palette("Blues", len(stage_order))
    stage_counts = (
        r_df['rASRM_stage'].value_counts(normalize=True).reindex(stage_order, fill_value=0)
    ) * 100
    axes[0].bar(stage_order, stage_counts, color=stage_palette, edgecolor='k')
    axes[0].set_ylabel("Percentage (%)", fontsize=16)
    axes[0].set_title("rASRM Stage Distribution", fontsize=18)
    for i, val in enumerate(stage_counts):
        axes[0].text(i, val + 1, f"{val:.1f}%", ha='center', color=stage_palette[i], fontweight='bold')
    axes[0].set_ylim(0, max(stage_counts.max() * 1.15, 35))
    axes[0].set_xlabel("rASRM Stage")

    # Enzian stacked barplot
    enzian_xticks = enzian_03 + enzian_bin
    enzian_labels = [c.replace("Enzian_", "") for c in enzian_xticks]
    value_range = range(4)  # 0, 1, 2, 3
    data_perc = {}
    for enz in enzian_xticks:
        if enz in enzian_03:
            v = r_df[enz].value_counts(normalize=True).reindex(value_range, fill_value=0)
            data_perc[enz] = v.values * 100
        else:  # binary
            v = r_df[enz].value_counts(normalize=True).reindex([0,1], fill_value=0)
            padded = np.zeros(4)
            padded[0:2] = v.values * 100
            data_perc[enz] = padded
    enzian_df = pd.DataFrame(data_perc, index=[0,1,2,3])  # Always 4 rows

    blues_palette = sns.color_palette("Blues", 4)
    bar_bottom = np.zeros(len(enzian_xticks))
    for value_idx in range(4):
        heights = enzian_df.iloc[value_idx].values
        axes[1].bar(
            range(len(enzian_xticks)), heights, bottom=bar_bottom,
            color=blues_palette[value_idx],
            edgecolor='k', label=f"Value {value_idx}"
        )
        bar_bottom += heights
    axes[1].set_ylabel("Percentage (%)", fontsize=16)
    axes[1].set_title("#Enzian Category Values", fontsize=18)
    axes[1].set_xticks(range(len(enzian_xticks)))
    axes[1].set_xticklabels(enzian_labels, rotation=30)
    axes[1].legend([f"Value {v}" for v in range(4)], title="Category Value",
                   bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.show()

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

def plot_dimensionality_reduction(
    df, biomarker_cols, 
    method='umap', 
    color_by=['Group'], 
    n_neighbors=15, min_dist=0.1, perplexity=30, random_state=42,
    figsize=(5,5)
):
    X = df[biomarker_cols].values
    method = method.lower()
    if method == 'umap':
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        embedding = reducer.fit_transform(X)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        embedding = reducer.fit_transform(X)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(X)
    else:
        raise ValueError("Method must be 'umap', 'tsne', or 'pca'.")

    plot_df = df.copy()
    plot_df['DR1'] = embedding[:, 0]
    plot_df['DR2'] = embedding[:, 1]

    n_plots = len(color_by)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0]*n_plots, figsize[1]))
    if n_plots == 1:
        axes = [axes]

    for ax, group_col in zip(axes, color_by):
        if group_col == 'Group':
            palette = "muted"
        elif group_col == 'rASRM_stage':
            palette = sns.color_palette("Blues", n_colors=4)
            # Also enforce correct order for hue
            plot_df[group_col] = pd.Categorical(plot_df[group_col], categories=['I','II','III','IV'], ordered=True)
        else:
            palette = "Blues"

        sns.scatterplot(
            data=plot_df, x="DR1", y="DR2", hue=group_col, palette=palette,
            alpha=0.8, s=60, edgecolor='k', ax=ax
        )
        ax.set_title(f"{method.upper()}\ncolored by {group_col}", fontsize=18)
        ax.set_xlabel(f"{method.upper()}1", fontsize=16)
        ax.set_ylabel(f"{method.upper()}2", fontsize=16)
        ax.set_aspect('equal', adjustable='box')
        leg = ax.legend(
            title=group_col,
            bbox_to_anchor=(0.5, -0.18),
            loc='upper center',
            borderaxespad=0,
            frameon=False,
            ncol=2
        )


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
