from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.tools.startup import logger

plt.style.use('fivethirtyeight')


def create_pie_chart_with_grouped_threshold(
        input_df: pd.DataFrame, column_name: str, ax: plt.Axes, title: str,
        threshold: float = 0.01, grouped_label: str = 'Others',
        x_label: str = '', y_label: str = '', font_size: int = 14,
        start_angle: int = 90) -> None:
    """
    Create Pie chart from dataframe and grouped values by threshold.

    Args:
        input_df (pd.DataFrame): dataframe to plot.
        column_name (str): column to plot.
        ax (plt.Axes): Matplotlib axes.
        title (str): plot's title.
        threshold (optional, float): threshold to apply. Default it is 0.01.
        grouped_label (optional, str): label value to put when grouped
            values are greater than a threshold.
            Default value is 'Others'
        x_label (optional, str): label for x axis.
        y_label (optional, str): label for y axis.
        font_size (optional, str): font size of plot.
        start_angle (optional, int): start angle of Pie chart.

    """
    grouped_df = input_df \
        .groupby(column_name) \
        .size() \
        .to_frame() \
        .reset_index() \
        .rename(columns={0: 'sizes'})

    if not grouped_df.empty:
        grouped_df['perc'] = grouped_df.sizes / grouped_df.sizes.sum()

        cond_lt_threshold = grouped_df.perc < threshold
        grouped_df.loc[cond_lt_threshold, 'label_cleaned'] = grouped_label
        grouped_df['label_cleaned'] = grouped_df \
            .label_cleaned \
            .fillna(grouped_df[column_name])

        grouped_df \
            .groupby('label_cleaned') \
            .sizes \
            .sum() \
            .sort_values() \
            .plot(kind='pie', autopct='%1.1f%%', title=title, ax=ax,
                  legend=None, xlabel=x_label, ylabel=y_label,
                  fontsize=font_size, startangle=start_angle)
    else:
        logger.warning(f'Empty grouped dataframe for column {column_name}')
        ax.remove()


def plot_correlation_heat_map(
        input_df: pd.DataFrame, title: str, color_map='BrBG',
        fig_size: Tuple[int, int] = (25, 25),
        correlation_method: str = 'pearson') -> None:
    """
    Given a dataframe 'input_df', this function plots a correlation heat map
    between all available columns.

    Args:
        input_df (pd.DataFrame): a dataframe for which to plot the heat map
        title (str): heat map title
        color_map (str, optional): the colormap to use
        fig_size (Tuple[int, int], optional): plot's figure size
        correlation_method (str, optional): correlation method to apply

    """
    plt.figure(figsize=fig_size)

    mask = np.triu(np.ones_like(input_df.corr(), dtype=np.bool))
    heatmap = sns.heatmap(
        input_df.corr(method=correlation_method), mask=mask, vmin=-1,
        vmax=1, annot=True, cmap=color_map)
    heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)


def plot_correlation_heat_map_target(
        input_df: pd.DataFrame, target_column: str, title: str,
        color_map: str = 'BrBG', fig_size: Tuple[int, int] = (8, 12)) -> None:
    """
    Given a dataframe and a target column, this function computes correlation
    of all non 'target_column' variables against 'target_column' and plots them
    as a heat map.

    Args:
        input_df (pd.DataFrame): dataframe with all variables of interest
        target_column (str): name of the target column
        title (str): plot's title
        color_map (str, optional): the colormap to use
        fig_size (Tuple[int, int], optional): plot's figure size

    """
    corr_df = input_df.corr()[[target_column]] \
        .sort_values(by=target_column, ascending=False)

    plt.figure(figsize=fig_size)
    heatmap = sns.heatmap(corr_df, vmin=-1, vmax=1, annot=True, cmap=color_map)
    heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)

    return corr_df


def bar_plot_n_unique(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Given a dataframe and columns list, this function computes the number of
    unique values for columns. Finally, it generates a bar plot.

    Args:
        df (pd.DataFrame): dataframe to plot number of unique values.
        columns (List[str]): columns to compute number of unique values.

    """
    # Get max y value
    unique_values = []
    for col in columns:
        unique_values.append(df[col].nunique())
    y_max = np.max(unique_values)

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle('Unique count per column')
    for i, col in enumerate(columns, start=1):
        ax = fig.add_subplot(3, 3, i)
        y_count = [df[col].nunique()]
        x_label = [col]
        ax.bar(x_label, y_count)
        ax.set_ylim(0, y_max)
        ax.set_title(f'{col}: {y_count[0]} unique')


def generate_box_plot(
        df: pd.DataFrame, title: str, ax: plt.Axes, y: str = 'price') -> None:
    """
    Given a dataframe, title, Matplotlib axes instance and a column name, this
    function generates a box plot with column name as y-axis. In addition, the
    y-axis minimum value is set to zero.

    Args:
        df (pd.DataFrame): dataframe to generate box plot.
        title (str): plot's title.
        ax (plt.Axes): plot's axes.
        y (Optional, str): dataframe's column name to put in y axis. Default
            value is 'price'.
    """
    _ = sns.boxplot(data=df, y=y, ax=ax)
    ax.set_ylim(bottom=0)
    ax.set_title(title)


def generate_clustering_scores_plot(
        max_clusters: int, scores: dict,
        score_type: Optional[int] = "Silhouette Score") -> None:
    """
    Generate a plot to visualize clustering scores for different numbers
    of clusters.

    Args:
        max_clusters (int): The maximum number of clusters considered in
            the analysis.
        scores (dict): Dictionary containing evaluation scores for different
            cluster numbers.
        score_type (Optional[str]): The type of clustering score to visualize.
            Default is "Silhouette Score".
    """
    x = range(2, max_clusters + 1)
    silhouette_scores = [scores[k][score_type] for k in x]

    plt.plot(x, silhouette_scores, label=score_type)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.title(f"{score_type} method")
    plt.legend()
    plt.show()


def plot_clusters(clusters, data, fig_size: Tuple[int, int] = (8, 8)) -> None:
    """
    Visualize clustering results by plotting data points with cluster
    assignments in a 2D or 3D space.

    Args:
        clusters (dict): dictionary mapping cluster labels to the indices of
            data points in each cluster.
        data (np.ndarray): the input data array used for clustering.
        fig_size (Tuple[int, int]): The size of the figure. Default is (8, 8).
    """
    all_points = []
    all_labels = []

    for cluster_id, data_indexes in clusters.items():
        cluster_points = data[data_indexes]
        cluster_labels = [cluster_id] * len(cluster_points)
        all_points.extend(cluster_points)
        all_labels.extend(cluster_labels)

    all_points = np.array(all_points)
    all_labels = np.array(all_labels)

    unique_labels = np.unique(all_labels)
    label_to_color = {
        label: f"C{i}" for i, label in enumerate(unique_labels)}

    plt.figure(figsize=fig_size)
    # 2D plot.
    if data.shape[1] == 2:
        for label in unique_labels:
            cluster_points = all_points[all_labels == label]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c=label_to_color[label], label=f"Cluster {label}")

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("Clustering Results")
        plt.legend()
        plt.show()
    # 3D plot.
    else:
        ax = plt.axes(projection='3d')
        for label in unique_labels:
            cluster_points = all_points[all_labels == label]

            ax.scatter3D(
                cluster_points[:, 0], cluster_points[:, 1],
                cluster_points[:, 2], c=label_to_color[label],
                label=f"Cluster {label}")

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title('3 components data representation')
