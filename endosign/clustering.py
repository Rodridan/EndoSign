import pandas as pd
from typing import List, Tuple, Optional

def reduce_dimensions(
    X, method='umap', n_components=2, umap_params=None, random_state=42
):
    if method == 'umap':
        import umap
        params = dict(n_neighbors=10, min_dist=0.2, random_state=random_state)
        if umap_params:
            params.update(umap_params)
        embedding = umap.UMAP(n_components=n_components, **params).fit_transform(X)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        embedding = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        embedding = TSNE(n_components=n_components, random_state=random_state).fit_transform(X)
    else:
        raise ValueError("method must be 'umap', 'pca', or 'tsne'")
    import pandas as pd
    return pd.DataFrame(embedding, columns=[f"DR{i+1}" for i in range(n_components)])

def cluster_patients(
    X, method="kmeans", n_clusters=4, random_state=42, dbscan_params=None
):
    if method == "kmeans":
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(X)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        params = dict(eps=0.5, min_samples=5)
        if dbscan_params:
            params.update(dbscan_params)
        model = DBSCAN(**params)
        labels = model.fit_predict(X)
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
    else:
        raise ValueError("Unsupported clustering method")
    import pandas as pd
    return pd.Series(labels, index=X.index, name="Cluster")