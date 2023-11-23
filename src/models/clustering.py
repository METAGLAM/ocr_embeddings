from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from tqdm import tqdm

from src.models import evaluate
from src.tools.startup import logger

_kmeans = 'KMeans'
_agglomerative = "Agglomerative"


def create_model(
        algorithm: str, n_clusters: int, random_state: Optional[int] = 90,
        **kwargs) -> callable:
    """
    Create a clustering model based on the specified algorithm.

    Args:
        algorithm (str): the clustering algorithm to use. Currently supports
            'KMeans' for KMeans and 'Agglomerative' for Agglomerative
            Clustering. The default value is 'KMeans'.
        n_clusters (int): the number of clusters to form.
        random_state (Optional[int]): deed for reproducibility. Default is 90.

    Returns:
     (callable): An instance of the specified clustering model.

    """
    if algorithm == _kmeans:
        model = KMeans(
            n_clusters, random_state=random_state, n_init='auto', **kwargs)
    elif algorithm == _agglomerative:
        model = AgglomerativeClustering(n_clusters, **kwargs)
    else:
        raise ValueError("Invalid clustering algorithm!")

    return model


def perform_clustering(
        data: np.ndarray,
        algorithm: Optional[str] = _kmeans,
        max_clusters: Optional[int] = 10,
        use_silhouette: Optional[bool] = True,
        manual_k: Optional[int] = None,
        random_state: Optional[int] = 90,
        cluster_params: Optional[dict] = None) \
        -> Tuple[dict, np.ndarray, dict, int]:
    """
    This function trains a cluster algorithm with from 2 to 'max_clusters' of
    clusters. For each model it computes the Silhouette Score, Davis-Bouldin
    Score and Calinski and Harabasz Score.
    Finally, it trains the model with best score and generates the prediction
    for the input data.

    Args:
        data (np.ndarray): the input data array for clustering.
        algorithm (Optional[str]): the clustering algorithm to use.
            Default is 'KMeans'.
        max_clusters (Optional[int]): the maximum number of clusters
            to consider. Default is 10.
        use_silhouette (Optional[bool]): whether to use the silhouette method
            for determining the optimal number of clusters. Default is True.
        manual_k (Optional[int]): A manually specified number of clusters,
            used if not using the silhouette method. Default is None.
        random_state (Optional[int]): Seed for reproducibility. Default is 90.
        cluster_params (Optional[dict]): Additional keyword arguments for
            cluster model creation. Default is None.

    Returns:
        (Tuple[dict, np.ndarray, dict, int]) Tuple containing the following:
            clusters (dict): Dictionary mapping cluster labels to the indices
                of data points in each cluster.
            labels (np.ndarray): Array of cluster labels assigned to each
                data point.
            scores (dict): Dictionary containing evaluation scores for
                different cluster numbers if using the silhouette method,
                otherwise an empty dictionary.
            optimal_k (int): The optimal number of clusters determined by the
                silhouette method or the manually specified value.
    """
    if use_silhouette and manual_k:
        raise ValueError(
            "Cannot use both silhouette method and manual number of clusters.")

    scores = {}
    max_clusters = min(max_clusters, len(data))

    if use_silhouette:
        with tqdm(
                range(2, max_clusters + 1),
                unit="it",
                desc=f'Fitting clustering model') as pbar:
            for k in pbar:
                # Adding k value to progress bar.
                progress_bar = {
                    'k': k
                }
                pbar.set_postfix(progress_bar)

                if cluster_params is None:
                    cluster_params = {}
                # Create new cluster model.
                model = create_model(
                    algorithm,
                    n_clusters=k,
                    random_state=random_state,
                    **cluster_params)
                # Generate predictions.
                labels = model.fit_predict(data)
                # Compute metrics.
                silhouette, davies_bouldin, calinski_harabasz = \
                    evaluate.evaluate_clustering(data, labels)

                scores[k] = {
                    "Silhouette Score": silhouette,
                    "Davis-Bouldin Score": davies_bouldin,
                    "Calinski and Harabasz Score": calinski_harabasz,
                }

        optimal_k = max(scores, key=lambda k: scores[k]["Silhouette Score"])
        logger.info(f'Optimal k is {optimal_k}')
        logger.info(f'Scores of optimal k:\n{scores[optimal_k]}')

    elif manual_k is not None:
        optimal_k = manual_k
    else:
        raise ValueError("Please choose either silhouette method or "
                         "manual number of clusters.")

    # Create optimal cluster model and fit it with the data.
    model = create_model(
        algorithm, n_clusters=optimal_k, random_state=random_state)
    labels = model.fit_predict(data)

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return clusters, labels, scores, optimal_k
