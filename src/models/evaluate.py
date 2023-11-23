from typing import Tuple

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, \
    calinski_harabasz_score


def evaluate_clustering(
        data: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Evaluate the quality of clustering results using various metrics. The
    metrics are the following:
    - Silhouette Score: A measure of how similar an object is to its own
        cluster (cohesion) compared to other clusters (separation).
    - Davies-Bouldin Score: Represents the average similarity between each
        cluster and its most similar cluster. Lower values indicate
        better clustering.
    - Calinski and Harabasz Score: Measures the ratio of between-cluster
        variance to within-cluster variance. Higher values indicate
        better clustering.

    Args:
        data (np.ndarray): the input data array used for clustering.
        labels (np.ndarray): array of cluster labels assigned to each
            data point.

    Returns:
        (Tuple[float, float, float]) Tuple of three float values representing
            the evaluation scores.
    """
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)

    return silhouette, davies_bouldin, calinski_harabasz
