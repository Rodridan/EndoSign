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
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib_venn import venn3

def plot_group_counts(df, group_col="Group"):
    """
    Multiplot: Left=barplot of patient group counts (1/3 width); Right=Venn diagram (2/3 width).
    Venn diagram custom-labeled and shows patient counts per subgroup.
    """
    # Define colors
    color_endo = '#1f77b4'
    color_control = '#ff7f0e'
    color_leiomyoma = '#2ca02c'
    group_palette = {"Endometriosis": color_endo, "Control": color_control}

    # Prepare Venn sets
    set_endo = set(df[df['Group'] == 'Endometriosis']['Patient_ID'])
    set_control = set(df[df['Group'] == 'Control']['Patient_ID'])
    set_myoma = set(df[df['Leiomyoma'] == 'Yes']['Patient_ID'])

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={'width_ratios': [1, 2]}
    )

    # -- Left: Barplot
    sns.countplot(
        x=group_col, data=df, ax=axes[0],
        palette=group_palette
    )
    axes[0].set_title(f"{group_col} Distribution", fontsize=14)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)

    # -- Calculate counts for each region (binary: Endo, Myoma, Control)
    venn_counts = {
        '100': len(set_endo - set_myoma - set_control),                                # Endo only
        '010': len(set_myoma - set_endo - set_control),                                # Myoma only
        '001': len(set_control - set_endo - set_myoma),                                # Control only
        '110': len(set_endo & set_myoma - set_control),                                # Endo + Myoma
        '011': len(set_myoma & set_control - set_endo),                                # Control + Myoma
        '101': len(set_endo & set_control - set_myoma),                                # Endo + Control (should be 0 in well-defined datasets)
        '111': len(set_endo & set_myoma & set_control),                                # All three (should be 0)
    }

    # -- Right: Venn diagram
    venn = venn3(
        [set_endo, set_myoma, set_control],
        set_labels=( 'Endometriosis', 'Myoma', 'Control'),
        set_colors=(color_endo, color_leiomyoma, color_control),
        alpha=0.5,
        ax=axes[1]
    )

    # Black border for circles
    for patch in venn.patches:
        if patch:
            patch.set_edgecolor('black')
            patch.set_linewidth(2.2)

    # Custom intersection labels, only showing non-empty regions
    custom_labels = {
        '100': "Endo",
        '010': "Myoma",
        '001': "CTRL",
        '110': "Endo\n+\nMyoma",
        '011': "CTRL\n+\nMyoma",
        '101': "Endo\n+\nCTRL",
        '111': "All three",
    }
    for region, label in custom_labels.items():
        n = venn_counts[region]
        subset = venn.get_label_by_id(region)
        if subset and n > 0:
            subset.set_text(f"{label}\n(n={n})")
        elif subset:
            subset.set_text("")  # Hide if region exists but count is 0

    plt.tight_layout()
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
    
def multiplot_biomarkers(
        df, biomarker_cols, 
    group_by='rASRM_stage',
    group_order=None,
    palette=None,
    subset_condition=None,
    n_cols=4,
    jitter=0.2,
    alpha=0.6,
    marker_size=5,
    show_swarm=False,
    swarm_kwargs=None,
):
    """
    Multiplot boxplots of each biomarker, grouped by the chosen categorical variable,
    with overlaid individual data points.
    
    Args:
        df: DataFrame
        biomarker_cols: list of biomarker columns
        group_by: column to group by (default: 'rASRM_stage')
        group_order: order for grouping variable (list)
        palette: seaborn color palette
        subset_condition: function or boolean mask to filter df
        n_cols: number of columns in multiplot
        jitter: stripplot jitter
        alpha: transparency of points
        marker_size: size of points
        show_swarm: if True, use swarmplot instead of stripplot
        swarm_kwargs: dict, extra kwargs for swarmplot (e.g., {"linewidth":0.5, "edgecolor":"gray"})
    """
    data = df
    if subset_condition is not None:
        data = data[subset_condition(data)] if callable(subset_condition) else data[subset_condition]

    n_biomarkers = len(biomarker_cols)
    n_rows = int(np.ceil(n_biomarkers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for i, biomarker in enumerate(biomarker_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        sns.boxplot(
            x=group_by, y=biomarker, hue=group_by,
            data=data, order=group_order, palette=palette, ax=ax,
            fliersize=0, legend=False
        )
        if ax.get_legend():
            ax.get_legend().remove()
        if show_swarm:
            _swarm_kwargs = dict(size=marker_size, alpha=alpha)
            if swarm_kwargs:
                _swarm_kwargs.update(swarm_kwargs)
            sns.swarmplot(
                x=group_by, y=biomarker, data=data,
                order=group_order, ax=ax, palette=palette, **_swarm_kwargs
            )
        else:
            sns.stripplot(
                x=group_by, y=biomarker, data=data,
                order=group_order, ax=ax, color='k',
                jitter=jitter, alpha=alpha, size=marker_size, dodge=True
            )
        ax.set_title(biomarker)
        ax.set_xlabel(group_by)
        ax.set_ylabel("Value")
    # Hide empty plots
    for j in range(n_biomarkers, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].axis('off')
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
    myoma_col: str = "Leiomyoma",
    rasrm_col: str = "rASRM_stage",
    cmap: str = "RdYlBu_r"
):
    """
    Multiplot: 
      Left: Heatmap of mean biomarker values for each clinical subgroup (Group + Leiomyoma)
      Right: Heatmap of mean biomarker values for each rASRM stage
    """
    # Build subgroup label
    subgroups = df[group_col] + " + " + df[myoma_col]
    df_sub = df.copy()
    df_sub["Subgroup"] = subgroups

    # Left: Mean per Subgroup
    subgroup_means = df_sub.groupby("Subgroup")[biomarker_cols].mean().T

    # Right: Mean per rASRM stage (Endometriosis patients only)
    rasrm_means = (
        df[df[group_col] == "Endometriosis"]
        .groupby(rasrm_col)[biomarker_cols].mean().T
    )
    # Order columns for clarity
    subgroup_order = sorted(subgroup_means.columns, key=lambda x: (x.startswith("Control"), x))
    rasrm_order = ["I", "II", "III", "IV"]
    rasrm_means = rasrm_means[[c for c in rasrm_order if c in rasrm_means.columns]]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(biomarker_cols) * 0.5)))

    # Left Heatmap: Subgroups
    sns.heatmap(
        subgroup_means[subgroup_order],
        annot=True, fmt=".2f", cmap=cmap, ax=axes[0],
        cbar_kws={"label": "Mean Expression"}
    )
    axes[0].set_title("Biomarker Means by Subgroup")
    axes[0].set_xlabel("Subgroup")
    axes[0].set_ylabel("Biomarker")

    # Right Heatmap: rASRM
    sns.heatmap(
        rasrm_means,
        annot=True, fmt=".2f", cmap=cmap, ax=axes[1],
        cbar_kws={"label": "Mean Expression"}
    )
    axes[1].set_title("Biomarker Means by rASRM Stage")
    axes[1].set_xlabel("rASRM Stage")
    axes[1].set_ylabel("")

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
    
def multiplot_embedding(df, embedding_2d, embedding_3d, color_by='Group', cluster_col='Cluster2D'):
    """
    Multiplot of 2D and 3D embeddings of patients, colored by clinical group and optionally cluster.
    """
    fig = plt.figure(figsize=(16, 7))
    
    # 2D plot
    ax1 = fig.add_subplot(1, 2, 1)
    scatter = sns.scatterplot(
        x=embedding_2d.iloc[:, 0], y=embedding_2d.iloc[:, 1],
        hue=df[color_by], style=df[cluster_col] if cluster_col else None,
        palette='muted', s=70, edgecolor='k', alpha=0.8, ax=ax1
    )
    ax1.set_title(f"2D UMAP ({color_by})", fontsize=15)
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend(title=color_by, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)
    
    # 3D plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    unique_groups = df[color_by].unique()
    palette = sns.color_palette('muted', n_colors=len(unique_groups))
    color_dict = {g: palette[i] for i, g in enumerate(unique_groups)}
    for g in unique_groups:
        mask = df[color_by] == g
        ax2.scatter(
            embedding_3d.loc[mask, "DR1"], 
            embedding_3d.loc[mask, "DR2"], 
            embedding_3d.loc[mask, "DR3"],
            label=str(g), s=45, alpha=0.85, color=color_dict[g], edgecolor='k'
        )
    ax2.set_title(f"3D UMAP ({color_by})", fontsize=15)
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")
    ax2.set_zlabel("UMAP3")
    ax2.legend(title=color_by, loc='upper left', bbox_to_anchor=(1.08, 1), frameon=False)
    plt.tight_layout()
    plt.show()
